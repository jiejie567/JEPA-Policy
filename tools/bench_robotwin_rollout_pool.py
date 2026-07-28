#!/usr/bin/env python3
"""Smoke-test the persistent RoboTwin environment-only rollout pool."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

from mip.envs.persistent_image_rollout import PersistentImageRolloutPool


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="handover_block_robotwin_image",
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--cycles", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers < 1 or args.cycles < 1:
        raise ValueError("workers and cycles must be positive")
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path("examples/configs").resolve()),
    ):
        config = compose(
            config_name="exps/robotwin_mip_baseline",
            overrides=[
                f"task={args.task}",
                "eval.parallel_rollout=true",
                f"eval.parallel_rollout_workers={args.workers}",
                "eval.worker_timeout_seconds=3600",
            ],
        )

    pool = PersistentImageRolloutPool(config, lazy=True)
    try:
        started = time.perf_counter()
        reset_results = pool.reset(
            [int(config.eval.rollout_seed) + i for i in range(args.workers)]
        )
        reset_seconds = time.perf_counter() - started
        expected_image_shape = (1, 2, 3, 128, 128)
        actions = []
        for observation, _ in reset_results:
            if observation["head_camera_image"].shape != expected_image_shape:
                raise RuntimeError(
                    f"Unexpected image shape: "
                    f"{observation['head_camera_image'].shape}"
                )
            state = observation["state"][:, -1, :]
            actions.append(
                np.repeat(
                    state[:, None, :],
                    int(config.task.act_steps),
                    axis=1,
                ).astype(np.float32)
            )

        started = time.perf_counter()
        for _ in range(args.cycles):
            step_results = pool.step(actions)
            if len(step_results) != args.workers:
                raise RuntimeError("Rollout pool returned the wrong step count")
        step_seconds = time.perf_counter() - started
        print(
            "ROBOTWIN_ROLLOUT_POOL_OK "
            f"workers={args.workers} cycles={args.cycles} "
            f"startup_and_reset_seconds={reset_seconds:.3f} "
            f"step_seconds={step_seconds:.3f} "
            f"seconds_per_cycle={step_seconds / args.cycles:.6f}"
        )
    finally:
        pool.close()


if __name__ == "__main__":
    main()
