#!/usr/bin/env python3
"""Validate the pinned RoboTwin source, assets, caches, and policy adapters."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

from mip.datasets.robot_dataset import make_dataset


REPO = Path(__file__).resolve().parents[1]
ROBOTWIN_ROOT = Path(
    os.environ.get("ROBOTWIN_SOURCE_ROOT", REPO / "third_party" / "robotwin")
).resolve()
DATASET_ROOT = Path(
    os.environ.get("ROBOTWIN_DATASET_ROOT", REPO / "datasets/robotwin/cache")
).resolve()
PINNED_COMMIT = "c3ddfa8b97d5519efa828b075999bd0006778e5e"
TASKS = {
    "stack_bowls_three_robotwin_image": "stack_bowls_three",
    "handover_block_robotwin_image": "handover_block",
    "put_object_cabinet_robotwin_image": "put_object_cabinet",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=sorted(TASKS),
        default=sorted(TASKS),
    )
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--env-smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def check_source_and_assets():
    commit = subprocess.check_output(
        ["git", "-C", str(ROBOTWIN_ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError(f"RoboTwin commit {commit}, expected {PINNED_COMMIT}")

    required = [
        ROBOTWIN_ROOT
        / "assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf",
        ROBOTWIN_ROOT
        / "assets/embodiments/aloha-agilex/srdf/arx5_description_isaac.srdf",
        ROBOTWIN_ROOT / "assets/objects/002_bowl",
        ROBOTWIN_ROOT / "assets/objects/036_cabinet",
        ROBOTWIN_ROOT / "assets/objects/047_mouse",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing RoboTwin assets: {missing}")
    return commit


def compose_config(task_name, variant):
    dataset_name = TASKS[task_name]
    with initialize_config_dir(
        version_base=None,
        config_dir=str((REPO / "examples/configs").resolve()),
    ):
        return compose(
            config_name=f"exps/robotwin_mip_{variant}",
            overrides=[
                f"task={task_name}",
                f"task.robotwin_root={ROBOTWIN_ROOT}",
                f"task.dataset_path={DATASET_ROOT / dataset_name}",
            ],
        )


def check_dataset(task_name, variant):
    config = compose_config(task_name, variant)
    dataset = make_dataset(config.task)
    marker_path = Path(config.task.dataset_path) / ".jepa_robotwin_cache.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker["total_episodes"] != 50 or marker["partial"]:
        raise ValueError(f"Invalid cache marker: {marker_path}")
    if marker["state_dim"] != 14 or marker["action_dim"] != 14:
        raise ValueError(f"Invalid RoboTwin dimensions: {marker_path}")

    first = dataset[0]
    last = dataset[len(dataset) - 1]
    for sample in (first, last):
        if sample["action"].shape != (10, 14):
            raise ValueError(f"Unexpected action shape: {sample['action'].shape}")
        for key in (
            "head_camera_image",
            "left_wrist_image",
            "right_wrist_image",
        ):
            if sample["obs"][key].shape != (2, 3, 128, 128):
                raise ValueError(f"Unexpected {key}: {sample['obs'][key].shape}")
    has_future = "future_obs" in first
    if has_future != (variant == "future4_ratio010"):
        raise ValueError(f"Future sample mismatch for {task_name}/{variant}")
    return marker["total_frames"]


def check_gpu(require_gpu):
    import torch

    if not torch.__version__.startswith("2.6.0"):
        raise RuntimeError(f"Unexpected torch build: {torch.__version__}")
    available = torch.cuda.is_available()
    if require_gpu and not available:
        raise RuntimeError("PPU/CUDA is unavailable")
    if available:
        tensor = torch.ones(4, device="cuda:0")
        if tensor.sum().item() != 4:
            raise RuntimeError("GPU arithmetic check failed")
    return available, torch.cuda.device_count()


def smoke_env(task_name, seed):
    from mip.envs.robot_env import make_vec_env

    config = compose_config(task_name, "baseline")
    config.task.num_envs = 1
    env = make_vec_env(config.task, seed=seed)
    try:
        observation, info = env.reset(seed=[seed])
        expected = (1, 2, 3, 128, 128)
        for key in (
            "head_camera_image",
            "left_wrist_image",
            "right_wrist_image",
        ):
            if observation[key].shape != expected:
                raise ValueError(
                    f"Unexpected env {key}: {observation[key].shape}"
                )
        state = observation["state"][:, -1, :]
        action = np.repeat(state[:, None, :], config.task.act_steps, axis=1)
        result = env.step(action.astype(np.float32))
        if len(result) != 5:
            raise ValueError("RoboTwin env did not return Gymnasium step output")
        return int(np.asarray(info["actual_seed"]).reshape(-1)[0])
    finally:
        env.close()


def main():
    args = parse_args()
    commit = check_source_and_assets()
    gpu_available, gpu_count = check_gpu(args.require_gpu)
    summaries = []
    for task_name in args.tasks:
        baseline_frames = check_dataset(task_name, "baseline")
        future_frames = check_dataset(task_name, "future4_ratio010")
        if baseline_frames != future_frames:
            raise AssertionError((baseline_frames, future_frames))
        actual_seed = (
            smoke_env(task_name, args.seed) if args.env_smoke else None
        )
        summaries.append(
            {
                "task": task_name,
                "frames": baseline_frames,
                "env_actual_seed": actual_seed,
            }
        )
    print(
        json.dumps(
            {
                "status": "ROBOTWIN_SETUP_OK",
                "commit": commit,
                "gpu_available": gpu_available,
                "gpu_count": gpu_count,
                "tasks": summaries,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
