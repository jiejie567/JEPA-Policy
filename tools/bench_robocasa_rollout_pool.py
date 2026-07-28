#!/usr/bin/env python3
"""Smoke-test RoboCasa's persistent env-only rollout pool."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

from examples.train_robomimic import _extract_success_info
from mip.envs.persistent_image_rollout import PersistentImageRolloutPool


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="steam_in_microwave_robocasa_image",
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
            config_name="exps/robocasa_mip_baseline",
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
        startup_and_reset_seconds = time.perf_counter() - started
        if len(reset_results) != args.workers:
            raise RuntimeError("Rollout pool returned the wrong reset count")
        expected_image_shape = (
            1,
            int(config.task.obs_steps),
            3,
            int(config.task.eval_image_size),
            int(config.task.eval_image_size),
        )
        for observation, _ in reset_results:
            actual_shape = tuple(
                observation["agentview_left_image"].shape
            )
            if actual_shape != expected_image_shape:
                raise RuntimeError(
                    "Unexpected rollout image shape: "
                    f"expected={expected_image_shape} actual={actual_shape}"
                )

        actions = [
            np.zeros(
                (
                    1,
                    int(config.task.act_steps),
                    int(config.task.act_dim),
                ),
                dtype=np.float32,
            )
            for _ in range(args.workers)
        ]
        started = time.perf_counter()
        for _ in range(args.cycles):
            step_results = pool.step(actions)
            if len(step_results) != args.workers:
                raise RuntimeError("Rollout pool returned the wrong step count")
            for worker_index, result in enumerate(step_results):
                if len(result) != 5:
                    raise RuntimeError(
                        "Rollout worker returned an invalid step tuple: "
                        f"worker={worker_index} length={len(result)}"
                    )
                info = result[4]
                success = _extract_success_info(info, 1)
                if success.shape != (1,) or success.dtype != np.bool_:
                    raise RuntimeError(
                        "RoboCasa success reduction returned an invalid result: "
                        f"worker={worker_index} shape={success.shape} "
                        f"dtype={success.dtype}"
                    )
        step_seconds = time.perf_counter() - started
        print(
            "ROBOCASA_ROLLOUT_POOL_OK "
            f"workers={args.workers} cycles={args.cycles} "
            f"startup_and_reset_seconds={startup_and_reset_seconds:.3f} "
            f"step_seconds={step_seconds:.3f} "
            f"seconds_per_cycle={step_seconds / args.cycles:.6f}"
        )
    finally:
        pool.close()


if __name__ == "__main__":
    main()
