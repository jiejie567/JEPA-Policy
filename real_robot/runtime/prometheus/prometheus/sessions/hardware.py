from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from hydra.utils import instantiate

from prometheus.data.types import DataStream, normalize_streams
from prometheus.sessions.session import (
    BaseSession,
    SessionMode,
    SessionState,
    SessionStateError,
    SessionStatus,
    SessionTimeoutError,
)


@dataclass(frozen=True)
class _Component:
    kind: str
    name: str
    obj: Any


class HardwareSession(BaseSession):
    """Owns the hardware runtime composition for a workflow."""

    def __init__(
        self,
        *,
        run_id: str,
        runtime_dir: str | Path,
        mode: str | SessionMode = SessionMode.OWNED,
        robot: Mapping[str, Any] | None = None,
        sensors: Mapping[str, Any] | None = None,
        streams: Mapping[str, Any] | None = None,
        name: str = "hardware",
    ):
        super().__init__(name=name, mode=SessionMode(mode))
        if self.mode != SessionMode.OWNED:
            raise ValueError("HardwareSession currently supports only mode='owned'")
        if not run_id:
            raise ValueError("run_id must be non-empty")
        if robot is None:
            raise ValueError("robot config is required; use {} for no robot")
        if sensors is None:
            raise ValueError("hardware.sensors is required; use {} for no sensors")
        if streams is None:
            raise ValueError("hardware.streams is required; use {} for no streams")
        self.run_id = str(run_id)
        self.runtime_dir = Path(runtime_dir).expanduser()
        self.robot = _mapping(robot, "robot")
        self.sensors = _mapping(sensors, "hardware.sensors")
        self._streams = normalize_streams(_mapping(streams, "hardware.streams"))
        self._robot_driver: Any | None = None
        self._robot_owner: Any | None = None
        self._robot_client: Any | None = None
        self._sensor_nodes: dict[str, Any] = {}
        self._components: list[_Component] = []
        self._cleanup_errors: list[str] = []

    @property
    def streams(self) -> Mapping[str, DataStream]:
        return dict(self._streams)

    @property
    def robot_driver(self) -> Any:
        if self._robot_driver is None:
            raise SessionStateError("hardware session has no robot driver")
        return self._robot_driver

    @property
    def robot_owner(self) -> Any:
        if self._robot_owner is None:
            raise SessionStateError("hardware session has no robot owner")
        return self._robot_owner

    @property
    def robot_client(self) -> Any:
        if self._robot_client is None:
            raise SessionStateError("hardware session has no robot client")
        return self._robot_client

    @property
    def sensor_nodes(self) -> Mapping[str, Any]:
        return dict(self._sensor_nodes)

    def start(self) -> None:
        status = self.status()
        if status.state == SessionState.READY:
            return
        if status.state in {SessionState.STOPPING, SessionState.STOPPED, SessionState.CLOSED}:
            raise SessionStateError(f"cannot start hardware session from state {status.state.value}")

        self._set_status(
            state=SessionState.STARTING,
            message="starting hardware session",
            details=self._status_details(),
        )
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._build_robot_runtime()
            self._sensor_nodes = self._build_components("sensor", self.sensors)
            for component in self._components:
                self._start_component(component)
            self._set_status(
                state=SessionState.STARTING,
                message="hardware session started",
                details=self._status_details(),
            )
        except Exception as exc:
            self.stop()
            self._mark_failed(str(exc), details=self._status_details())
            raise

    def wait_ready(self, timeout_s: float) -> SessionStatus:
        status = self.status()
        if status.state == SessionState.NEW:
            raise SessionStateError(f"hardware session {self.name!r} has not been started")
        if status.state == SessionState.FAILED:
            raise SessionStateError(status.message or "hardware session failed")
        if status.ready:
            return status

        deadline = time.monotonic() + float(timeout_s)
        try:
            for component in self._components:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SessionTimeoutError(
                        f"timed out waiting for hardware session {self.name!r}"
                    )
                self._wait_component_ready(component, remaining)
            self._set_status(
                state=SessionState.READY,
                message="hardware session ready",
                details=self._status_details(),
            )
            return self.status()
        except Exception as exc:
            self._mark_failed(str(exc), details=self._status_details())
            raise

    def stop(self) -> None:
        status = self.status()
        if status.state in {SessionState.NEW, SessionState.STOPPED, SessionState.CLOSED}:
            return
        if status.state == SessionState.STOPPING:
            return
        self._set_status(
            state=SessionState.STOPPING,
            message="stopping hardware session",
            details=self._status_details(),
        )
        self._cleanup_errors = []
        if self._robot_client is not None:
            self._safe_cleanup("robot_client.release_control", self._release_robot_client)
        for component in reversed(self._components):
            self._safe_cleanup(
                f"{component.kind}.{component.name}.stop",
                lambda item=component: self._stop_component(item),
            )
        for component in reversed(self._components):
            self._safe_cleanup(
                f"{component.kind}.{component.name}.close",
                lambda item=component: self._close_component(item),
            )
        self._set_status(
            state=SessionState.STOPPED,
            message=(
                "hardware session stopped"
                if not self._cleanup_errors
                else "hardware session stopped with cleanup errors"
            ),
            details=self._status_details(),
        )

    def close(self) -> None:
        status = self.status()
        if status.state == SessionState.CLOSED:
            return
        if status.state not in {SessionState.NEW, SessionState.STOPPED}:
            self.stop()
        self._set_status(
            state=SessionState.CLOSED,
            message="hardware session closed",
            details=self._status_details(),
        )

    def _build_robot_runtime(self) -> None:
        driver_cfg = _mapping(self.robot.get("driver"), "robot.driver")
        if driver_cfg:
            self._robot_driver = instantiate(driver_cfg, _convert_="all")

        owner_cfg = _mapping(self.robot.get("owner"), "robot.owner")
        if owner_cfg:
            extra = {}
            if self._robot_driver is not None:
                extra["driver"] = self._robot_driver
            self._robot_owner = instantiate(owner_cfg, **extra, _convert_="all")
            self._components.append(_Component("robot_owner", "robot_owner", self._robot_owner))
        elif self._robot_driver is not None:
            self._components.append(_Component("robot_driver", "robot_driver", self._robot_driver))

        client_cfg = _mapping(self.robot.get("client"), "robot.client")
        if client_cfg:
            self._robot_client = instantiate(client_cfg, _convert_="all")
            self._components.append(_Component("robot_client", "robot_client", self._robot_client))

    def _build_components(self, kind: str, raw: Mapping[str, Any]) -> dict[str, Any]:
        components: dict[str, Any] = {}
        for name, spec in raw.items():
            cfg = _mapping(spec, f"{kind}s.{name}")
            obj = instantiate(cfg, _convert_="all")
            key = str(name)
            components[key] = obj
            self._components.append(_Component(kind, key, obj))
        return components

    def _start_component(self, component: _Component) -> None:
        start = getattr(component.obj, "start", None)
        if callable(start):
            start()
            return
        if component.kind == "robot_driver":
            component.obj.connect()

    def _wait_component_ready(self, component: _Component, timeout_s: float) -> None:
        wait_ready = getattr(component.obj, "wait_ready", None)
        if callable(wait_ready):
            wait_ready(timeout_s=float(timeout_s))
            return
        wait_for_owner = getattr(component.obj, "wait_for_owner", None)
        if callable(wait_for_owner):
            wait_for_owner(timeout_s=float(timeout_s))

    def _release_robot_client(self) -> None:
        release = getattr(self._robot_client, "release_control", None)
        if callable(release):
            release(timeout=1.0, wait=True)

    def _stop_component(self, component: _Component) -> None:
        stop = getattr(component.obj, "stop", None)
        if callable(stop):
            stop()

    def _close_component(self, component: _Component) -> None:
        close = getattr(component.obj, "close", None)
        if callable(close):
            close()

    def _safe_cleanup(self, label: str, fn: Any) -> None:
        try:
            fn()
        except Exception as exc:
            self._cleanup_errors.append(f"{label}: {exc}")

    def _status_details(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "runtime_dir": str(self.runtime_dir),
            "robot_id": self.robot.get("id"),
            "robot_driver": _object_status(self._robot_driver),
            "robot_owner": _object_status(self._robot_owner),
            "robot_client": _object_status(self._robot_client),
            "sensors": {
                name: _object_status(node)
                for name, node in self._sensor_nodes.items()
            },
            "sensor_names": sorted(self.sensors),
            "cleanup_errors": list(self._cleanup_errors),
            "streams": {name: stream.as_dict() for name, stream in self._streams.items()},
        }


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _object_status(obj: Any | None) -> dict[str, Any] | None:
    if obj is None:
        return None
    status = getattr(obj, "status", None)
    if callable(status):
        payload = status()
        if not isinstance(payload, Mapping):
            raise ValueError(f"{type(obj).__name__}.status() must return a mapping")
        return dict(payload)
    return {"type": type(obj).__name__}
