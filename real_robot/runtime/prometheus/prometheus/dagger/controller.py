from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from prometheus.dagger.phase import DAggerPhase, semantic_disposition


@dataclass(frozen=True)
class DAggerConfig:
    cycle_wait_s: float = 30.0
    reset_home_timeout_s: float = 30.0
    event_name: str = "event_label"


@dataclass(frozen=True)
class ApplyResult:
    accepted: bool
    ignored: bool = False
    deferred: bool = False
    from_phase: DAggerPhase | None = None
    to_phase: DAggerPhase | None = None


class DAggerController:
    def __init__(
        self,
        *,
        robot_client: Any,
        data_session: Any,
        config: DAggerConfig | Mapping[str, Any] | None = None,
        run_id: str = "",
        get_robot_stamp_ns: Callable[[], int | None] | None = None,
        episode_index_start: int = 1,
        task_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._robot = robot_client
        self._data = data_session
        self._config = _coerce_config(config)
        self.run_id = str(run_id)
        self._get_robot_stamp_ns = get_robot_stamp_ns or (lambda: time.time_ns())
        self._next_episode_index = int(episode_index_start)
        self._task_metadata = dict(task_metadata or {})

        self._phase = DAggerPhase.IDLE
        self._event_level = 0
        self._pending_pause = False
        self._cycle_deadline: float | None = None
        self._hold_action: Any | None = None
        self._ignored_events = 0

    @property
    def phase(self) -> DAggerPhase:
        return self._phase

    @property
    def event_level(self) -> int:
        return self._event_level

    @property
    def pending_pause(self) -> bool:
        return self._pending_pause

    @property
    def hold_action(self) -> Any | None:
        return self._hold_action

    @property
    def ignored_events(self) -> int:
        return self._ignored_events

    @property
    def can_infer(self) -> bool:
        return self.phase == DAggerPhase.AUTONOMOUS and not self.pending_pause

    @property
    def can_send_scheduler(self) -> bool:
        return self.can_infer

    @property
    def should_hold(self) -> bool:
        return self.phase == DAggerPhase.PAUSED

    def apply_semantic(self, semantic: str) -> ApplyResult:
        from_phase = self.phase
        disposition = semantic_disposition(from_phase, semantic)
        if disposition == "ignore":
            self._ignored_events += 1
            return ApplyResult(
                accepted=False,
                ignored=True,
                from_phase=from_phase,
                to_phase=from_phase,
            )

        if disposition == "defer":
            self._pending_pause = True
            return ApplyResult(
                accepted=True,
                deferred=True,
                from_phase=from_phase,
                to_phase=from_phase,
            )

        if semantic == "teaching":
            self._enter_correcting()
        elif semantic == "cycle":
            self._begin_cycle()
        else:
            raise ValueError(f"unsupported deferred semantic: {semantic!r}")

        return ApplyResult(
            accepted=True,
            from_phase=from_phase,
            to_phase=self.phase,
        )

    apply_event = apply_semantic

    def bootstrap_autonomous(self) -> None:
        if self.phase != DAggerPhase.IDLE:
            raise RuntimeError(f"bootstrap_autonomous requires IDLE, got {self.phase.value}")
        self._pending_pause = False
        self._hold_action = None
        self._event_level = 0
        self._start_autonomous_episode()
        self._phase = DAggerPhase.AUTONOMOUS

    def enter_paused(self, hold_action: Any) -> None:
        if not self._pending_pause:
            raise RuntimeError("enter_paused requires pending_pause")
        if self.phase != DAggerPhase.AUTONOMOUS:
            raise RuntimeError(f"enter_paused requires AUTONOMOUS, got {self.phase.value}")

        self._data.recorder.pause()
        self._hold_action = hold_action
        self._pending_pause = False
        self._phase = DAggerPhase.PAUSED

    def tick_cycle_wait(self) -> bool:
        if self.phase != DAggerPhase.CYCLE_WAIT:
            return False
        if self._cycle_deadline is None:
            raise RuntimeError("cycle wait deadline is not initialized")
        if time.monotonic() < self._cycle_deadline:
            return False
        self._finish_cycle()
        return True

    def status(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "event_level": self.event_level,
            "pending_pause": self.pending_pause,
            "ignored_events": self.ignored_events,
            "next_episode_index": self._next_episode_index,
            "cycle_wait_active": self.phase == DAggerPhase.CYCLE_WAIT,
        }

    def _begin_cycle(self) -> None:
        if self._data.recorder.recording:
            self._record_event_level(0)
            self._data.stop_episode(accept=True)

        self._reset_home()
        self._pending_pause = False
        self._hold_action = None
        self._event_level = 0
        self._cycle_deadline = time.monotonic() + float(self._config.cycle_wait_s)
        self._phase = DAggerPhase.CYCLE_WAIT

    def _finish_cycle(self) -> None:
        self._start_autonomous_episode()
        self._cycle_deadline = None
        self._phase = DAggerPhase.AUTONOMOUS

    def _start_autonomous_episode(self) -> None:
        episode_id = f"{self._next_episode_index:04d}"
        self._next_episode_index += 1
        metadata = {
            "workflow": "rollout_dagger",
            "run_id": self.run_id,
            **self._task_metadata,
        }
        self._data.start_episode(episode_id, metadata=metadata)
        self._set_position_mode()
        self._data.recorder.resume()
        self._record_event_level(0)

    def _enter_correcting(self) -> None:
        self._pending_pause = False
        self._hold_action = None
        self._set_teach_mode()
        self._data.recorder.resume()
        self._record_event_level(1)
        self._phase = DAggerPhase.CORRECTING

    def _record_event_level(self, level: int) -> None:
        stamp_ns = self._get_robot_stamp_ns()
        if stamp_ns is None:
            stamp_ns = time.time_ns()
        self._data.record_event(
            self._config.event_name,
            int(level),
            stamp_ns=int(stamp_ns),
        )
        self._event_level = int(level)

    def _reset_home(self) -> None:
        reset_home = getattr(self._robot, "reset_home", None)
        if not callable(reset_home):
            raise RuntimeError("robot client does not support reset_home()")
        result = reset_home(timeout=float(self._config.reset_home_timeout_s), wait=True)
        if isinstance(result, dict) and not bool(result.get("accepted", False)):
            raise RuntimeError(f"robot reset_home rejected: {result}")

    def _set_teach_mode(self) -> None:
        set_teach_mode = getattr(self._robot, "set_teach_mode", None)
        if not callable(set_teach_mode):
            raise RuntimeError("robot client does not support set_teach_mode()")
        result = set_teach_mode(timeout=5.0, wait=True)
        if isinstance(result, dict) and not bool(result.get("accepted", False)):
            raise RuntimeError(f"robot set_teach_mode rejected: {result}")

    def _set_position_mode(self) -> None:
        set_position_mode = getattr(self._robot, "set_position_mode", None)
        if not callable(set_position_mode):
            raise RuntimeError("robot client does not support set_position_mode()")
        result = set_position_mode(timeout=5.0, wait=True)
        if isinstance(result, dict) and not bool(result.get("accepted", False)):
            raise RuntimeError(f"robot set_position_mode rejected: {result}")


def _coerce_config(config: DAggerConfig | Mapping[str, Any] | None) -> DAggerConfig:
    if config is None:
        return DAggerConfig()
    if isinstance(config, DAggerConfig):
        return config
    return DAggerConfig(
        cycle_wait_s=float(config.get("cycle_wait_s", 30.0)),
        reset_home_timeout_s=float(config.get("reset_home_timeout_s", 30.0)),
        event_name=str(config.get("event_name", "event_label")),
    )
