from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from prometheus.policy.scheduler import ActionChunk, ActionScheduler
from prometheus.utils.robot import quat_wxyz_to_rpy
from prometheus.utils.workflow import (
    build_hardware_session,
    config_dict,
    instantiate_config,
    prepare_run,
    section,
    write_json,
)

ROBOT_SIDES = ("left", "right")
CONTROL_MODES = {
    "joint": "abs_qpos",
    "eef": "abs_eef",
}


def resolve_robot_state_path(input_path: str | Path) -> Path:
    path = Path(input_path).expanduser()
    if path.is_file():
        if path.suffix != ".pkl":
            raise ValueError(f"robot state input must be a .pkl file, got {path}")
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"replay input does not exist: {path}")
    candidate = path / "robot" / "robot_state_dict.pkl"
    if not candidate.is_file():
        raise FileNotFoundError(f"robot state pickle not found: {candidate}")
    return candidate


def load_robot_state(path: str | Path) -> dict[str, Any]:
    state_path = resolve_robot_state_path(path)
    with state_path.open("rb") as file:
        robot_state = pickle.load(file)
    if not isinstance(robot_state, Mapping):
        raise ValueError(f"robot state must be a mapping, got {type(robot_state).__name__}")
    for side in ROBOT_SIDES:
        if side not in robot_state or not isinstance(robot_state[side], Mapping):
            raise ValueError(f"robot state missing side {side!r}")
        side_state = robot_state[side]
        for key in ("joint", "eef", "timestamps"):
            if key not in side_state:
                raise ValueError(f"robot state {side!r} missing key {key!r}")
    left_ts = robot_state["left"]["timestamps"]
    right_ts = robot_state["right"]["timestamps"]
    if list(left_ts) != list(right_ts):
        raise ValueError("left/right robot timestamps must match")
    return dict(robot_state)


def downsample_to_hz(timestamps_ms: Any, target_hz: float) -> np.ndarray:
    if float(target_hz) <= 0:
        raise ValueError("target_hz must be positive")
    timestamp = np.asarray(timestamps_ms, dtype=np.float64).reshape(-1)
    if timestamp.size == 0:
        raise ValueError("timestamps must be non-empty")
    step = 1000.0 / float(target_hz)
    start_time = float(timestamp[0])
    end_time = float(timestamp[-1])
    num_out = int(np.floor((end_time - start_time) / step)) + 1
    if num_out <= 0:
        raise ValueError("downsample produced no target timestamps")
    target_times = start_time + np.arange(num_out, dtype=np.float64) * step

    idx = np.searchsorted(timestamp, target_times, side="left")
    idx_r = np.clip(idx, 0, timestamp.size - 1)
    idx_l = np.clip(idx - 1, 0, timestamp.size - 1)
    d_r = np.abs(timestamp[idx_r] - target_times)
    d_l = np.abs(timestamp[idx_l] - target_times)
    return np.where(d_l <= d_r, idx_l, idx_r).astype(np.int64)


def filter_valid_indices(robot_state: Mapping[str, Any], indices: np.ndarray, *, mode: str) -> tuple[np.ndarray, int]:
    control_mode = _normalize_control_mode(mode)
    kept: list[int] = []
    skipped = 0
    for index in np.asarray(indices, dtype=np.int64).reshape(-1):
        index = int(index)
        if _frame_is_valid(robot_state, index, mode=control_mode):
            kept.append(index)
        else:
            skipped += 1
    if not kept:
        raise ValueError(f"no valid replay frames remain for control_mode={control_mode!r}")
    return np.asarray(kept, dtype=np.int64), skipped


def build_replay_actions(robot_state: Mapping[str, Any], indices: np.ndarray, *, mode: str) -> np.ndarray:
    control_mode = _normalize_control_mode(mode)
    actions = []
    for index in np.asarray(indices, dtype=np.int64).reshape(-1):
        index = int(index)
        if not _frame_is_valid(robot_state, index, mode=control_mode):
            raise ValueError(f"invalid replay frame at index {index} for control_mode={control_mode!r}")
        parts = []
        for side in ROBOT_SIDES:
            side_state = robot_state[side]
            joint = np.asarray(side_state["joint"][index], dtype=np.float32).reshape(7)
            if control_mode == "joint":
                parts.append(joint)
                continue
            eef = np.asarray(side_state["eef"][index], dtype=np.float32).reshape(7)
            parts.append(
                np.concatenate(
                    [
                        eef[:3],
                        quat_wxyz_to_rpy(eef[3:7]),
                        np.asarray([joint[6]], dtype=np.float32),
                    ]
                ).astype(np.float32)
            )
        actions.append(np.concatenate(parts).astype(np.float32))
    return np.stack(actions, axis=0).astype(np.float32)


def load_replay_chunk(
    input_path: str | Path,
    *,
    mode: str,
    hz: float,
) -> tuple[ActionChunk, dict[str, Any]]:
    robot_state = load_robot_state(input_path)
    indices = downsample_to_hz(robot_state["left"]["timestamps"], hz)
    indices, skipped = filter_valid_indices(robot_state, indices, mode=mode)
    actions = build_replay_actions(robot_state, indices, mode=mode)
    action_space = CONTROL_MODES[_normalize_control_mode(mode)]
    timestamps = [int(robot_state["left"]["timestamps"][index]) for index in indices]
    metadata = {
        "source": str(resolve_robot_state_path(input_path)),
        "control_mode": _normalize_control_mode(mode),
        "source_frames": len(robot_state["left"]["timestamps"]),
        "replay_frames": int(actions.shape[0]),
        "skipped_frames": int(skipped),
        "start_ms": timestamps[0] if timestamps else None,
        "end_ms": timestamps[-1] if timestamps else None,
    }
    return ActionChunk(actions, action_space=action_space, hz=float(hz), metadata=metadata), metadata


def run_from_config(cfg: Any) -> int:
    data = config_dict(cfg, "replay")
    run_cfg = section(data, "run")
    robot_cfg = section(data, "robot")
    task_cfg = section(data, "task")
    hardware_cfg = section(data, "hardware")
    scheduler_cfg = section(data, "scheduler")
    workflow_cfg = section(data, "workflow")
    run = prepare_run(run_cfg)

    control_mode = _normalize_control_mode(workflow_cfg["control_mode"])
    action_hz = float(workflow_cfg["action_hz"])
    if action_hz <= 0:
        raise ValueError("workflow.action_hz must be positive")
    expected_action_space = CONTROL_MODES[control_mode]
    configured_action_space = _robot_action_space(robot_cfg)
    if configured_action_space != expected_action_space:
        raise ValueError(
            f"robot action_space={configured_action_space!r} does not match "
            f"workflow.control_mode={control_mode!r} (expected {expected_action_space!r})"
        )

    input_path = str(workflow_cfg["input"]).strip()
    if not input_path:
        raise ValueError("workflow.input must be non-empty")

    chunk, replay_metadata = load_replay_chunk(
        input_path,
        mode=control_mode,
        hz=action_hz,
    )

    hardware = build_hardware_session(run=run, hardware_cfg=hardware_cfg, robot_cfg=robot_cfg)
    robot_client = None
    scheduler = None
    wait_ready_timeout_s = float(run_cfg["wait_ready_timeout_s"])
    reset_timeout_s = float(workflow_cfg.get("reset_home_timeout_s", 30.0))
    actions_sent = 0
    status = "running"
    stop_reason = "not_started"
    return_code = 0

    try:
        hardware.start()
        hardware.wait_ready(wait_ready_timeout_s)
        robot_client = hardware.robot_client
        scheduler = instantiate_config(scheduler_cfg, "scheduler", robot_client=robot_client)
        scheduler.reset()
        actions_sent = scheduler.execute(chunk)
        _reset_robot_home(hardware, timeout_s=reset_timeout_s)
        status = "completed"
        stop_reason = "replay_completed"
        return 0
    except KeyboardInterrupt:
        status = "interrupted"
        stop_reason = "keyboard_interrupt"
        try:
            _reset_robot_home(hardware, timeout_s=reset_timeout_s)
        except Exception as exc:
            print(f"[replay] reset_home after interrupt failed: {exc}", flush=True)
        return_code = 130
        return return_code
    except Exception:
        status = "error"
        stop_reason = "exception"
        return_code = 1
        raise
    finally:
        try:
            hardware.stop()
        finally:
            hardware.close()
        write_json(
            run.manifest_path,
            {
                "run_id": run.run_id,
                "status": status,
                "stop_reason": stop_reason,
                "actions_sent": actions_sent,
                "time_ns": time.time_ns(),
                "output_dir": str(run.output_dir),
                "runtime_dir": str(run.runtime_dir),
                "task": task_cfg,
                "workflow": {
                    "input": input_path,
                    "control_mode": control_mode,
                    "action_hz": action_hz,
                    "action_space": expected_action_space,
                },
                "replay": replay_metadata,
                "hardware": hardware.status().as_dict(),
                "robot_client": None if robot_client is None else robot_client.status(),
                "scheduler": {
                    "dry_run": None if scheduler is None else scheduler.dry_run,
                    "history": 0 if scheduler is None else len(scheduler.history),
                    "filters": [] if scheduler is None else [item.name for item in scheduler.filters],
                },
            },
        )
        if return_code:
            return return_code


def main() -> None:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../configs", config_name="replay")
    def _main(cfg: DictConfig) -> None:
        raise SystemExit(run_from_config(cfg))

    _main()

def _normalize_control_mode(mode: str) -> str:
    value = str(mode).strip().lower()
    if value not in CONTROL_MODES:
        raise ValueError(f"workflow.control_mode must be one of {sorted(CONTROL_MODES)}, got {mode!r}")
    return value

def _robot_action_space(robot_cfg: Mapping[str, Any]) -> str:
    inner = robot_cfg.get("robot")
    if isinstance(inner, Mapping) and "action_space" in inner:
        return str(inner["action_space"])
    if "action_space" in robot_cfg:
        return str(robot_cfg["action_space"])
    raise ValueError("robot config must define action_space")

def _frame_is_valid(robot_state: Mapping[str, Any], index: int, *, mode: str) -> bool:
    for side in ROBOT_SIDES:
        side_state = robot_state[side]
        joint = side_state["joint"][index]
        if joint is None:
            return False
        joint_arr = np.asarray(joint, dtype=np.float32).reshape(-1)
        if joint_arr.shape != (7,) or not np.all(np.isfinite(joint_arr)):
            return False
        if mode != "eef":
            continue
        eef = side_state["eef"][index]
        if eef is None:
            return False
        eef_arr = np.asarray(eef, dtype=np.float32).reshape(-1)
        if eef_arr.shape != (7,) or not np.all(np.isfinite(eef_arr)):
            return False
    return True

def _reset_robot_home(hardware: Any, *, timeout_s: float) -> None:
    client = getattr(hardware, "robot_client", None)
    reset_home = getattr(client, "reset_home", None)
    if not callable(reset_home):
        print("[replay] reset_home skipped: robot client has no reset_home()", flush=True)
        return
    print("[replay] resetting robot home.", flush=True)
    result = reset_home(timeout=float(timeout_s), wait=True)
    if isinstance(result, Mapping) and not bool(result.get("accepted", False)):
        raise RuntimeError(f"robot reset_home rejected: {result}")


if __name__ == "__main__":
    main()
