from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from prometheus.policy.scheduler import ActionChunk
from prometheus.runtime import ActionScheduler, ActionTrace, RobotClientActionTarget, replace_pending
from prometheus.transport.zmq_env import ZmqEnvServer
from prometheus.utils.workflow import (
    build_data_session,
    build_hardware_session,
    config_dict,
    instantiate_config,
    prepare_run,
    section,
    write_json,
)


@dataclass(frozen=True, slots=True)
class TerminalRequest:
    accept: bool
    reason: str


class DataSessionObsProvider:
    """Policy-facing obs reader backed by DataSession's async numpy buffer."""

    def __init__(self, data_session: Any):
        self.data_session = data_session

    def available(self) -> tuple[str, ...]:
        keys = {"robot/qpos", "robot/qvel", "robot/effort"}
        try:
            for name in self.data_session.stream_names():
                keys.add(str(name))
                camera = _camera_key(name)
                tactile = _tactile_key(name)
                if camera:
                    keys.add(camera)
                if tactile:
                    keys.add(tactile)
        except Exception:
            pass
        return tuple(sorted(keys))

    def latest(self, keys: Sequence[str], *, timeout_s: float | None) -> dict[str, np.ndarray]:
        return self._wait(keys, timeout_s=timeout_s, baselines=None)

    def next(
        self,
        keys: Sequence[str],
        *,
        timeout_s: float | None,
        baselines: dict[str, int] | None = None,
    ) -> dict[str, np.ndarray]:
        if baselines is None:
            baselines = self.stamps(keys)
        return self._wait(keys, timeout_s=timeout_s, baselines=baselines)

    def aligned_window(
        self,
        keys: Sequence[str],
        *,
        anchor: str,
        count: int,
        slop_ms: float,
    ) -> tuple[list[dict[str, Any]], list[dict[str, int]], list[int]]:
        selected = tuple(str(key) for key in keys)
        stream_names = tuple(self._stream_name(key) for key in selected)
        frames = self.data_session.window_numpy(
            anchor=self._stream_name(str(anchor)),
            names=stream_names,
            count=int(count),
            stride=1,
            slop_ms=float(slop_ms),
            wait_latest=True,
            timeout_ms=20.0,
        )
        history: list[dict[str, Any]] = []
        stamps: list[dict[str, int]] = []
        frame_stamps: list[int] = []
        for frame in frames:
            values: dict[str, Any] = {}
            stamp_values: dict[str, int] = {}
            for key, stream_name in zip(selected, stream_names):
                sample = frame.samples[stream_name]
                values[key] = self._value_for_key(key, sample)
                stamp_values[key] = int(sample.stamp_ns)
            history.append(values)
            stamps.append(stamp_values)
            frame_stamps.append(int(frame.stamp_ns))
        return history, stamps, frame_stamps

    def stamps(self, keys: Sequence[str]) -> dict[str, int]:
        stamps: dict[str, int] = {}
        for key in keys:
            stream_name = self._stream_name(str(key))
            sample = self.data_session.latest_numpy([stream_name])[stream_name]
            stamps[str(key)] = int(sample.stamp_ns)
        return stamps

    def _wait(
        self,
        keys: Sequence[str],
        *,
        timeout_s: float | None,
        baselines: dict[str, int] | None,
    ) -> dict[str, np.ndarray]:
        selected = tuple(str(key) for key in keys)
        if not selected:
            raise ValueError("obs_keys is empty")
        deadline = None if timeout_s is None else time.monotonic() + max(0.0, float(timeout_s))
        last_error: Exception | None = None
        while True:
            try:
                samples = {
                    key: self.data_session.latest_numpy([self._stream_name(key)])[self._stream_name(key)]
                    for key in selected
                }
                if baselines is None or all(int(samples[key].stamp_ns) != int(baselines.get(key, 0)) for key in selected):
                    return {key: self._value_for_key(key, samples[key]) for key in selected}
            except Exception as exc:
                last_error = exc
            if deadline is not None and time.monotonic() >= deadline:
                if baselines is None:
                    raise TimeoutError(f"obs keys not ready: {selected}: {last_error}")
                raise TimeoutError(f"obs keys did not receive next sample: {selected}: {last_error}")
            time.sleep(0.005)

    def _stream_name(self, key: str) -> str:
        if key in {"robot/qpos", "robot/qvel", "robot/effort"}:
            return "robot_state"
        if key.startswith("stream/"):
            return key.removeprefix("stream/")
        if key.startswith("camera/") and key.endswith("/rgb"):
            return f"{key.split('/')[1]}_color"
        if key.startswith("tactile/") and key.endswith("/flow"):
            return f"{key.split('/')[1]}_tactile_flow"
        return key

    def _value_for_key(self, key: str, sample: Any) -> np.ndarray:
        data = sample.data
        if key == "robot/qpos":
            return np.asarray(data["position"], dtype=np.float32).copy()
        if key == "robot/qvel":
            return np.asarray(data["velocity"], dtype=np.float32).copy()
        if key == "robot/effort":
            return np.asarray(data["effort"], dtype=np.float32).copy()
        if isinstance(data, dict):
            return {
                str(item_key): np.asarray(item_value).copy()
                for item_key, item_value in data.items()
            }
        return np.asarray(data).copy()


class RunEnvSession:
    def __init__(
        self,
        *,
        robot_client: Any,
        obs_provider: DataSessionObsProvider,
        default_obs_keys: Sequence[str],
        action_hz: float,
        max_chunk_horizon: int,
        policy_low_watermark: int,
        reset_timeout_s: float,
        step_timeout_s: float | None,
        action_rewriter: Callable[..., np.ndarray] = replace_pending,
        dry_run: bool = False,
        action_log_interval: int = 0,
        send_timeout_s: float = 3.0,
        send_wait: bool = False,
        data_session: Any | None = None,
        recording_enabled: bool = False,
        episode_id: str = "0000",
        metadata: dict[str, Any] | None = None,
    ):
        self.robot_client = robot_client
        self.obs_provider = obs_provider
        self.default_obs_keys = tuple(str(key) for key in default_obs_keys)
        self.reset_timeout_s = float(reset_timeout_s)
        self.step_timeout_s = None if step_timeout_s is None else float(step_timeout_s)
        self.data_session = data_session
        self.recording_enabled = bool(recording_enabled)
        self.episode_id = str(episode_id)
        self.metadata = dict(metadata or {})
        self.target = RobotClientActionTarget(robot_client, timeout_s=send_timeout_s, wait=send_wait)
        self.scheduler = ActionScheduler(
            action_target=self.target,
            action_hz=float(action_hz),
            max_chunk_horizon=int(max_chunk_horizon),
            policy_low_watermark=int(policy_low_watermark),
            rewriter=action_rewriter,
            dry_run=bool(dry_run),
            action_log_interval=int(action_log_interval),
        )
        self._started = False
        self._recording_started = False
        self._last_obs_stamps: dict[str, int] = {}

    @property
    def closed(self) -> bool:
        return not self._started

    def reset(self, *, obs_keys: Sequence[str], options: Any | None = None) -> dict[str, Any]:
        if self._started:
            raise RuntimeError("run_env session is already reset")
        keys = select_obs_keys(obs_keys, default_obs_keys=self.default_obs_keys)
        validate_reset_options(
            options,
            action_space=self.scheduler.action_space,
            action_dim=self.scheduler.action_dim,
            action_hz=self.scheduler.action_hz,
            max_chunk_horizon=self.scheduler.max_chunk_horizon,
            policy_low_watermark=self.scheduler.policy_low_watermark,
        )
        if self.recording_enabled:
            if self.data_session is None:
                raise RuntimeError("recording requires data_session")
            self.data_session.start_episode(self.episode_id, metadata=self.metadata | {"workflow": "run_env"})
            self._recording_started = True
            self.scheduler.trace_hook = self._record_action_trace
        obs = self.obs_provider.latest(keys, timeout_s=self.reset_timeout_s)
        self._last_obs_stamps = self.obs_provider.stamps(keys)
        option_data = dict(options or {})
        history_count = int(option_data.get("history_count", 1))
        info: dict[str, Any] = {"available_obs": self.obs_provider.available()}
        if history_count > 1:
            history, history_stamps, frame_stamps = self.obs_provider.aligned_window(
                keys,
                anchor=str(option_data.get("history_anchor", keys[0])),
                count=history_count,
                slop_ms=float(option_data.get("history_slop_ms", 60.0)),
            )
            info["obs_history"] = history
            info["obs_history_timestamps"] = history_stamps
            info["obs_history_frame_timestamps"] = frame_stamps
        self.scheduler.start()
        self._started = True
        return {"obs": obs, "info": info}

    def step(
        self,
        actions: Any,
        *,
        obs_keys: Sequence[str],
        history_count: int = 1,
        history_anchor: str | None = None,
        history_slop_ms: float = 60.0,
    ) -> dict[str, Any]:
        if not self._started:
            raise RuntimeError("run_env session has not been reset")
        step_start = time.monotonic()
        keys = select_obs_keys(obs_keys, default_obs_keys=self.default_obs_keys)
        action_array = np.asarray(actions, dtype=np.float32)
        if self.recording_enabled and self.data_session is not None:
            self.data_session.record_policy_action_chunk(
                ActionChunk(action_array, action_space=self.scheduler.action_space, hz=self.scheduler.action_hz),
                metadata={"source": "run_env"},
            )
        scheduler_start = time.monotonic()
        info: dict[str, Any] = dict(self.scheduler.submit_and_wait(action_array, timeout_s=self.step_timeout_s))
        info["scheduler_wait_ms"] = (time.monotonic() - scheduler_start) * 1000.0
        obs_start = time.monotonic()
        obs = self.obs_provider.next(keys, timeout_s=self.step_timeout_s, baselines=self._last_obs_stamps)
        self._last_obs_stamps = self.obs_provider.stamps(keys)
        info["obs_wait_ms"] = (time.monotonic() - obs_start) * 1000.0
        if int(history_count) > 1:
            history, history_stamps, frame_stamps = self.obs_provider.aligned_window(
                keys,
                anchor=history_anchor or keys[0],
                count=int(history_count),
                slop_ms=float(history_slop_ms),
            )
            info["obs_history"] = history
            info["obs_history_timestamps"] = history_stamps
            info["obs_history_frame_timestamps"] = frame_stamps
        info["server_step_ms"] = (time.monotonic() - step_start) * 1000.0
        return {"obs": obs, "reward": 0.0, "terminated": False, "truncated": False, "info": info}

    def close(self, *, accept: bool = True) -> None:
        error: BaseException | None = None
        if self._started:
            try:
                self.scheduler.stop()
            except BaseException as exc:
                accept = False
                error = exc
        if self._recording_started and self.data_session is not None:
            try:
                self.data_session.stop_episode(accept=accept and error is None)
            except BaseException as exc:
                if error is None:
                    error = exc
        self.scheduler.trace_hook = None
        self._recording_started = False
        self._started = False
        if error is not None:
            raise error

    def _record_action_trace(self, trace: ActionTrace) -> None:
        if self.data_session is None:
            return
        self.data_session.record_action(
            trace.action,
            action_space=trace.action_space,
            hz=self.scheduler.action_hz,
            stamp_ns=time.time_ns(),
            metadata={
                "source": "run_env",
                "sent_action_count": trace.sent_action_count,
                "queue_len_after_send": trace.queue_len_after_send,
                "scheduler_stamp_ns": trace.scheduler_stamp_ns,
                "send_start_ns": trace.send_start_ns,
                "send_done_ns": trace.send_done_ns,
            },
        )


def serve_forever(server: ZmqEnvServer, session_factory: Callable[[], RunEnvSession]) -> None:
    session: RunEnvSession | None = None
    try:
        while True:
            request = server.recv()
            request_id = str(request.get("request_id", ""))
            request_type = str(request.get("type", ""))
            try:
                if request_type == "reset":
                    if session is not None:
                        raise RuntimeError("client is already attached; call close before reset")
                    session = session_factory()
                    try:
                        payload = session.reset(obs_keys=tuple(request.get("obs_keys", ())), options=request.get("options"))
                    except Exception:
                        session.close(accept=False)
                        session = None
                        raise
                    server.reply(request_id, **payload)
                elif request_type == "step":
                    if session is None:
                        raise RuntimeError("no attached client; call reset first")
                    server.reply(
                        request_id,
                        **session.step(
                            request.get("actions"),
                            obs_keys=tuple(request.get("obs_keys", ())),
                            history_count=int(request.get("history_count", 1)),
                            history_anchor=(
                                None
                                if request.get("history_anchor") is None
                                else str(request.get("history_anchor"))
                            ),
                            history_slop_ms=float(request.get("history_slop_ms", 60.0)),
                        ),
                    )
                elif request_type == "close":
                    if session is not None:
                        session.close()
                        session = None
                    server.reply(request_id, info={})
                else:
                    raise ValueError(f"unknown run_env request type {request_type!r}")
            except Exception as exc:
                server.reject(request_id, str(exc))
    finally:
        if session is not None:
            session.close(accept=False)


def run_from_config(cfg: Any) -> int:
    data = config_dict(cfg, "run_env")
    run_cfg = section(data, "run")
    robot_cfg = section(data, "robot")
    hardware_cfg = section(data, "hardware")
    data_cfg = section(data, "data")
    env_cfg = section(data, "env")
    task_cfg = section(data, "task")
    run = prepare_run(run_cfg)
    run.runtime_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PROMETHEUS_CAMERA_TIMING_TRACE"] = str(run.runtime_dir / "camera_publish_timing.jsonl")

    hardware = build_hardware_session(run=run, hardware_cfg=hardware_cfg, robot_cfg=robot_cfg)
    data_session = build_data_session(data_cfg=data_cfg, hardware=hardware)
    action_rewriter = instantiate_config(section(env_cfg, "action_rewriter"), "env.action_rewriter")
    status = "running"
    stop_reason = "not_started"
    try:
        hardware.start()
        data_session.start()
        hardware.wait_ready(float(run_cfg["wait_ready_timeout_s"]))
        data_session.wait_ready(float(run_cfg["wait_ready_timeout_s"]))
        obs_provider = DataSessionObsProvider(data_session)
        obs_provider.latest(tuple(env_cfg["ready_obs_keys"]), timeout_s=float(env_cfg["reset_timeout_s"]))

        recording_cfg = section(data_cfg, "recording")
        episode_counter = {"next": 0}

        def make_session() -> RunEnvSession:
            episode_id = _next_episode_id(recording_cfg, episode_counter)
            return RunEnvSession(
                robot_client=hardware.robot_client,
                obs_provider=obs_provider,
                default_obs_keys=tuple(env_cfg["default_obs_keys"]),
                action_hz=float(env_cfg["action_hz"]),
                max_chunk_horizon=int(env_cfg["max_chunk_horizon"]),
                policy_low_watermark=int(env_cfg["policy_low_watermark"]),
                reset_timeout_s=float(env_cfg["reset_timeout_s"]),
                step_timeout_s=none_or_float(env_cfg.get("step_timeout_s")),
                action_rewriter=action_rewriter,
                dry_run=bool(env_cfg.get("dry_run", False)),
                action_log_interval=int(env_cfg.get("action_log_interval", 0)),
                send_timeout_s=float(env_cfg.get("send_timeout_s", 3.0)),
                send_wait=bool(env_cfg.get("send_wait", False)),
                data_session=data_session,
                recording_enabled=bool(recording_cfg.get("enabled", False)),
                episode_id=episode_id,
                metadata={"run_id": run.run_id, "task": task_cfg},
            )

        print(f"[run_env] serving {env_cfg['addr']}", flush=True)
        with ZmqEnvServer(addr=str(env_cfg["addr"])) as server:
            serve_forever(server, make_session)
        status = "completed"
        stop_reason = "server_closed"
        return 0
    except KeyboardInterrupt:
        status = "interrupted"
        stop_reason = "keyboard_interrupt"
        return 130
    except Exception:
        status = "error"
        stop_reason = "exception"
        raise
    finally:
        try:
            data_session.stop_ingress()
        except Exception:
            pass
        data_session.close()
        hardware.close()
        write_json(
            run.manifest_path,
            {
                "run_id": run.run_id,
                "status": status,
                "stop_reason": stop_reason,
                "time_ns": time.time_ns(),
                "output_dir": str(run.output_dir),
                "runtime_dir": str(run.runtime_dir),
                "task": task_cfg,
                "hardware": hardware.status().as_dict(),
                "data": data_session.status().as_dict(),
            },
        )


def main() -> None:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../configs", config_name="run_env")
    def _main(cfg: DictConfig) -> None:
        raise SystemExit(run_from_config(cfg))

    _main()


def select_obs_keys(obs_keys: Sequence[str], *, default_obs_keys: Sequence[str]) -> tuple[str, ...]:
    keys = tuple(str(key) for key in obs_keys) or tuple(str(key) for key in default_obs_keys)
    if not keys:
        raise ValueError("obs_keys is empty and env.default_obs_keys is empty")
    return keys


def validate_reset_options(
    options: Any | None,
    *,
    action_space: str,
    action_dim: int,
    action_hz: float,
    max_chunk_horizon: int,
    policy_low_watermark: int,
) -> None:
    data = dict(options or {})
    expect = dict(data.pop("expect", {}))
    data.pop("history_count", None)
    data.pop("history_anchor", None)
    data.pop("history_slop_ms", None)
    if data:
        raise ValueError(f"unsupported reset options: {sorted(data)}")
    supported = {"action_space", "action_dim", "action_hz", "policy_low_watermark", "action_chunk_horizon"}
    unknown = set(expect).difference(supported)
    if unknown:
        raise ValueError(f"unsupported reset expect keys: {sorted(unknown)}")

    if "action_space" in expect and str(expect["action_space"]) != str(action_space):
        raise ValueError(f"action_space mismatch: expected {expect['action_space']!r}, server has {action_space!r}")
    if "action_dim" in expect and int(expect["action_dim"]) != int(action_dim):
        raise ValueError(f"action_dim mismatch: expected {int(expect['action_dim'])}, server has {int(action_dim)}")
    if "action_hz" in expect and not float_equal(float(expect["action_hz"]), float(action_hz)):
        raise ValueError(f"action_hz mismatch: expected {float(expect['action_hz'])}, server has {float(action_hz)}")
    if "policy_low_watermark" in expect and int(expect["policy_low_watermark"]) != int(policy_low_watermark):
        raise ValueError(
            "policy_low_watermark mismatch: "
            f"expected {int(expect['policy_low_watermark'])}, server has {int(policy_low_watermark)}"
        )
    if "action_chunk_horizon" in expect:
        horizon = int(expect["action_chunk_horizon"])
        if not 1 <= horizon <= int(max_chunk_horizon):
            raise ValueError(f"action_chunk_horizon must be in [1,{int(max_chunk_horizon)}], got {horizon}")
        if horizon <= int(policy_low_watermark):
            raise ValueError(
                "action_chunk_horizon must be greater than policy_low_watermark: "
                f"{horizon} <= {int(policy_low_watermark)}"
            )


def float_equal(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= 1e-6


def none_or_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _camera_key(name: str) -> str:
    if name.endswith("_color"):
        return f"camera/{name.removesuffix('_color')}/rgb"
    return ""


def _tactile_key(name: str) -> str:
    if name.endswith("_tactile_flow"):
        return f"tactile/{name.removesuffix('_tactile_flow')}/flow"
    return ""


def _next_episode_id(recording_cfg: dict[str, Any], counter: dict[str, int]) -> str:
    base = str(recording_cfg.get("episode_id", "0000"))
    if not bool(recording_cfg.get("enabled", False)):
        return base
    output_dir = Path(str(recording_cfg["output_dir"])).expanduser()
    start = int(base) if base.isdigit() else int(counter["next"])
    index = max(start, int(counter["next"]))
    while True:
        episode_id = f"{index:04d}" if base.isdigit() else f"{base}_{index:04d}"
        if not (output_dir / episode_id).exists():
            counter["next"] = index + 1
            return episode_id
        index += 1


if __name__ == "__main__":
    main()
