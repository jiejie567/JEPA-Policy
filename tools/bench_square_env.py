#!/usr/bin/env python3
import argparse
import json
import os
import time

import numpy as np
from omegaconf import OmegaConf

from jepa_policy.envs.robomimic.robomimic_env import make_vec_env


def reset_envs(envs):
    result = envs.reset()
    if isinstance(result, tuple):
        return result[0]
    return result


def benchmark(task_cfg, num_envs: int, warmup_steps: int, benchmark_steps: int):
    cfg = OmegaConf.create(OmegaConf.to_container(task_cfg, resolve=True))
    cfg.num_envs = num_envs
    cfg.act_steps = 1
    cfg.save_video = False

    t0 = time.perf_counter()
    envs = make_vec_env(cfg, seed=42)
    create_s = time.perf_counter() - t0

    try:
        t0 = time.perf_counter()
        reset_envs(envs)
        reset_s = time.perf_counter() - t0

        # cfg.act_dim is the transformed policy action dimension.
        # The simulator needs raw actions from its own action space.
        single_action_space = envs.single_action_space
        if not hasattr(single_action_space, "shape"):
            raise TypeError(
                f"Expected a Box-like single_action_space, got {single_action_space!r}"
            )

        actions = np.zeros(
            (num_envs, *single_action_space.shape),
            dtype=single_action_space.dtype,
        )

        print(
            "policy_act_dim =", cfg.act_dim,
            "| raw_env_action_shape =", single_action_space.shape,
            "| vector_action_shape =", actions.shape,
        )

        for _ in range(warmup_steps):
            envs.step(actions)

        t0 = time.perf_counter()
        for _ in range(benchmark_steps):
            envs.step(actions)
        step_s = time.perf_counter() - t0

        result = {
            "num_envs": num_envs,
            "create_s": round(create_s, 4),
            "reset_s": round(reset_s, 4),
            "benchmark_steps": benchmark_steps,
            "total_step_s": round(step_s, 4),
            "per_vector_step_ms": round(step_s / benchmark_steps * 1000, 2),
            "per_individual_env_step_ms": round(
                step_s / benchmark_steps / num_envs * 1000, 2
            ),
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result
    finally:
        envs.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--benchmark-steps", type=int, default=100)
    parser.add_argument("--env-counts", type=int, nargs="+", default=[1, 4])
    args = parser.parse_args()

    print("=== Runtime ===")
    print("MUJOCO_GL =", os.environ.get("MUJOCO_GL"))
    print("PYOPENGL_PLATFORM =", os.environ.get("PYOPENGL_PLATFORM"))
    print("MUJOCO_EGL_DEVICE_ID =", os.environ.get("MUJOCO_EGL_DEVICE_ID"))
    print()

    full_cfg = OmegaConf.load(args.config)
    task_cfg = full_cfg.task

    print("=== Resolved task config ===")
    print("env_name =", task_cfg.env_name)
    print("max_episode_steps =", task_cfg.max_episode_steps)
    print("act_dim =", task_cfg.act_dim)
    print("original_num_envs =", task_cfg.num_envs)
    print()

    all_results = []
    for num_envs in args.env_counts:
        print(f"=== Benchmark: num_envs={num_envs} ===")
        all_results.append(
            benchmark(
                task_cfg=task_cfg,
                num_envs=num_envs,
                warmup_steps=args.warmup_steps,
                benchmark_steps=args.benchmark_steps,
            )
        )

    if len(all_results) == 2:
        one_env = all_results[0]["per_vector_step_ms"]
        four_env = all_results[1]["per_vector_step_ms"]
        print()
        print("=== Comparison ===")
        print(
            f"vector-step ratio ({all_results[1]['num_envs']} env / "
            f"{all_results[0]['num_envs']} env) = {four_env / one_env:.2f}x"
        )


if __name__ == "__main__":
    main()
