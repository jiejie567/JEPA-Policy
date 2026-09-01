#!/usr/bin/env python3
"""Load one exported policy and validate its robot contract without hardware."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


CAMERA_STREAMS = {
    "base_image": "base_0_color",
    "left_wrist_image": "left_wrist_0_color",
    "right_wrist_image": "right_wrist_0_color",
}
JOINT_NAMES = [
    *(f"left_joint_{index}" for index in range(1, 7)),
    "left_gripper",
    *(f"right_joint_{index}" for index in range(1, 7)),
    "right_gripper",
]


class SyntheticAlignedData:
    """Small DataSession stand-in that verifies the adapter's alignment request."""

    def __init__(self, states: np.ndarray) -> None:
        self.states = np.asarray(states, dtype=np.float32)
        self.request: dict[str, Any] | None = None

    def window_numpy(
        self,
        *,
        anchor: str,
        names: tuple[str, ...],
        count: int,
        stride: int,
        slop_ms: float,
        wait_latest: bool,
        timeout_ms: float,
        min_anchor_stamp_ns: int | None = None,
    ) -> list[Any]:
        self.request = {
            "anchor": anchor,
            "names": list(names),
            "count": int(count),
            "stride": int(stride),
            "slop_ms": float(slop_ms),
            "wait_latest": bool(wait_latest),
            "timeout_ms": float(timeout_ms),
            "min_anchor_stamp_ns": min_anchor_stamp_ns,
        }
        expected_names = [*CAMERA_STREAMS.values(), "robot_state"]
        if anchor != "base_0_color":
            raise AssertionError(f"unexpected alignment anchor: {anchor}")
        if list(names) != expected_names:
            raise AssertionError(f"unexpected stream order: {list(names)}")
        if (count, stride) != (2, 3):
            raise AssertionError(
                f"expected two observations at a 30Hz stride of 3, got {(count, stride)}"
            )
        if float(slop_ms) != 60.0:
            raise AssertionError(f"expected 60ms alignment slop, got {slop_ms}")

        frames = []
        for frame_index in range(2):
            # Distinct RGB patterns make accidental camera reordering observable.
            images = {
                "base_0_color": _camera_image(frame_index, (180, 35, 20)),
                "left_wrist_0_color": _camera_image(frame_index, (25, 175, 45)),
                "right_wrist_0_color": _camera_image(frame_index, (30, 50, 185)),
            }
            samples = {
                name: SimpleNamespace(data=image) for name, image in images.items()
            }
            samples["robot_state"] = SimpleNamespace(
                data={
                    "name": list(JOINT_NAMES),
                    "position": self.states[frame_index].copy(),
                    "velocity": np.zeros(14, dtype=np.float32),
                }
            )
            frames.append(
                SimpleNamespace(
                    stamp_ns=1_000_000_000 + frame_index * 100_000_000,
                    samples=samples,
                )
            )
        return frames


class RejectingRobotClient:
    """Proves that ActionScheduler(dry_run=True) never calls send_action."""

    action_space = "abs_qpos"

    def send_action(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("dry-run attempted to call the robot client")

    def spin_once(self, _timeout_s: float = 0.0) -> None:
        return None


def _camera_image(frame_index: int, rgb: tuple[int, int, int]) -> np.ndarray:
    image = np.empty((240, 320, 3), dtype=np.uint8)
    image[...] = np.asarray(rgb, dtype=np.uint8)
    image[:, :, frame_index] = np.clip(
        image[:, :, frame_index].astype(np.int16) + 10, 0, 255
    ).astype(np.uint8)
    return image


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("jepa", "mip", "diffusion_policy"), required=True)
    parser.add_argument("--task", default="cabinet")
    parser.add_argument("--step", default="100000")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--denoise-steps", type=int, default=100)
    parser.add_argument("--latency-compensation-steps", type=int, default=3)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--jepa-root", type=Path, required=True)
    parser.add_argument("--diffusion-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def _load_stats(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    stats = json.loads(path.read_text(encoding="utf-8"))
    minimum = np.asarray(stats["state"]["min"], dtype=np.float32)
    maximum = np.asarray(stats["state"]["max"], dtype=np.float32)
    if minimum.shape != (14,) or maximum.shape != (14,):
        raise ValueError("task state stats must both contain 14 values")
    midpoint = (minimum + maximum) * 0.5
    states = np.stack([midpoint, midpoint], axis=0)
    return stats, states


def _build_policy(args: argparse.Namespace) -> Any:
    from prometheus.policy.diffusion_policy import DiffusionPolicy
    from prometheus.policy.jepa_policy import JEPAPolicy, MIPPolicy

    common_inputs = {
        "anchor": "base_0_color",
        "slop_ms": 60.0,
        "window_size": 2,
        "stride": 3,
        "wait_latest": False,
        "wait_latest_timeout_ms": 5.0,
        "camera_streams": CAMERA_STREAMS,
        "state_stream": "robot_state",
    }
    if args.method in {"jepa", "mip"}:
        config_name = (
            "exps/arx_r5_mip_future4_ratio010"
            if args.method == "jepa"
            else "exps/arx_r5_mip_baseline"
        )
        checkpoint = (
            args.asset_root
            / "checkpoints"
            / args.method
            / args.task
            / f"model_step{args.step}.pt"
        )
        inference = {
            "project_root": str(args.jepa_root),
            "config_name": config_name,
            "task": f"{args.task}_arx_r5_image",
            "checkpoint": str(checkpoint),
            "stats_path": str(args.asset_root / "stats" / args.task / "stats.json"),
            "device": args.device,
            "num_steps": 2,
        }
        policy_class = JEPAPolicy if args.method == "jepa" else MIPPolicy
        return policy_class(inputs=common_inputs, inference=inference, action_hz=10.0)

    checkpoint = (
        args.asset_root
        / "checkpoints"
        / "diffusion_policy"
        / args.task
        / f"step={args.step}.ckpt"
    )
    return DiffusionPolicy(
        inputs=common_inputs,
        inference={
            "project_root": str(args.diffusion_root),
            "checkpoint": str(checkpoint),
            "device": args.device if ":" in args.device else f"{args.device}:0",
            "num_inference_steps": args.denoise_steps,
            "latency_compensation_steps": args.latency_compensation_steps,
            "use_ema": True,
        },
        action_hz=10.0,
    )


def _x5_driver_input_limits() -> tuple[np.ndarray, np.ndarray]:
    import arx5_interface

    config = arx5_interface.RobotConfigFactory.get_instance().get_config("X5")
    arm_min = np.asarray(config.joint_pos_min, dtype=np.float32)
    arm_max = np.asarray(config.joint_pos_max, dtype=np.float32)
    lower = np.concatenate([arm_min, [0.0], arm_min, [0.0]]).astype(np.float32)
    # The deployed driver subtracts 0.005 m before its physical clip.
    upper = np.concatenate([arm_max, [0.087], arm_max, [0.087]]).astype(np.float32)
    return lower, upper


def main() -> int:
    args = _parse_args()
    if not 1 <= args.denoise_steps <= 100:
        raise ValueError("--denoise-steps must be in [1,100]")
    if not 0 <= args.latency_compensation_steps <= 8:
        raise ValueError("--latency-compensation-steps must be in [0,8]")
    for path in (args.asset_root, args.source_root, args.jepa_root, args.diffusion_root):
        if not path.is_dir():
            raise FileNotFoundError(path)
    sys.path.insert(0, str(args.source_root))
    if args.method == "diffusion_policy":
        sys.path.insert(0, str(args.diffusion_root))

    stats_path = args.asset_root / "stats" / args.task / "stats.json"
    stats, states = _load_stats(stats_path)
    data = SyntheticAlignedData(states)
    started = time.perf_counter()
    policy = _build_policy(args)
    try:
        chunk = policy.infer(data)
        elapsed_s = time.perf_counter() - started
        if chunk.action_space != "abs_qpos":
            raise AssertionError(f"unexpected action space: {chunk.action_space}")
        expected_horizon = (
            min(8, 9 - args.latency_compensation_steps)
            if args.method == "diffusion_policy"
            else 8
        )
        if chunk.actions.shape != (expected_horizon, 14):
            raise AssertionError(
                f"expected {(expected_horizon, 14)} latency-aligned action slice, "
                f"got {chunk.actions.shape}"
            )
        if chunk.hz != 10.0:
            raise AssertionError(f"expected 10Hz action chunk, got {chunk.hz}")
        if not np.all(np.isfinite(chunk.actions)):
            raise AssertionError("model produced NaN or inf")

        from prometheus.policy.scheduler import ActionScheduler

        scheduler = ActionScheduler(RejectingRobotClient(), dry_run=True)
        scheduled = scheduler.execute(chunk, steps=1)
        if scheduled != 1 or len(scheduler.history) != 1:
            raise AssertionError("dry-run scheduler did not validate exactly one action")

        current = np.asarray(chunk.metadata["current_action"], dtype=np.float32)
        deltas = np.diff(np.vstack([current, chunk.actions]), axis=0)
        lower, upper = _x5_driver_input_limits()
        driver_input_limit_ok = bool(
            np.all(chunk.actions >= lower[None, :])
            and np.all(chunk.actions <= upper[None, :])
        )
        effective_actions = chunk.actions.copy()
        effective_actions[:, [6, 13]] = np.clip(
            effective_actions[:, [6, 13]] - 0.005, 0.0, 0.080
        )
        action_min = np.asarray(stats["action"]["min"], dtype=np.float32)
        action_max = np.asarray(stats["action"]["max"], dtype=np.float32)
        training_range_ok = bool(
            np.all(chunk.actions >= action_min[None, :])
            and np.all(chunk.actions <= action_max[None, :])
        )
        result = {
            "status": "passed",
            "mode": "offline_no_hardware",
            "method": args.method,
            "task": args.task,
            "step": args.step,
            "model": chunk.metadata.get("model"),
            "robot": {
                "driver": "Arx5Driver",
                "sdk_model": "X5",
                "joint_names": JOINT_NAMES,
            },
            "input": {
                "camera_order": list(CAMERA_STREAMS),
                "stream_order": [*CAMERA_STREAMS.values(), "robot_state"],
                "state_shape": [2, 14],
                "state_order": "left_arm_6,left_gripper,right_arm_6,right_gripper",
                "alignment_request": data.request,
                "observation_hz": 10.0,
            },
            "output": {
                "action_space": chunk.action_space,
                "shape": list(chunk.actions.shape),
                "hz": chunk.hz,
                "slice": (
                    f"[{1 + args.latency_compensation_steps}:10]"
                    if args.method == "diffusion_policy"
                    else "[1:9]"
                ),
                "latency_compensation_steps": (
                    args.latency_compensation_steps
                    if args.method == "diffusion_policy"
                    else 0
                ),
                "first_action": chunk.actions[0].tolist(),
                "last_action": chunk.actions[-1].tolist(),
                "max_abs_delta_from_previous": float(np.max(np.abs(deltas))),
                "left_gripper_range": [
                    float(np.min(chunk.actions[:, 6])),
                    float(np.max(chunk.actions[:, 6])),
                ],
                "right_gripper_range": [
                    float(np.min(chunk.actions[:, 13])),
                    float(np.max(chunk.actions[:, 13])),
                ],
                "within_x5_driver_input_limits": driver_input_limit_ok,
                "effective_left_gripper_range": [
                    float(np.min(effective_actions[:, 6])),
                    float(np.max(effective_actions[:, 6])),
                ],
                "effective_right_gripper_range": [
                    float(np.min(effective_actions[:, 13])),
                    float(np.max(effective_actions[:, 13])),
                ],
                "within_task_training_range": training_range_ok,
            },
            "scheduler": {
                "dry_run": True,
                "validated_actions": scheduled,
                "robot_send_action_calls": 0,
            },
            "runtime": policy.runtime.status(),
            "elapsed_s": elapsed_s,
        }
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        return 0
    finally:
        policy.close()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
