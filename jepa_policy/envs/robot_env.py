"""Environment factory for robot-manipulation benchmarks."""

from __future__ import annotations

from jepa_policy.libero_utils import is_libero_task


def make_vec_env(task_config, seed=None):
    if is_libero_task(task_config):
        from jepa_policy.envs.libero.libero_env import make_vec_env as make_libero_vec_env

        return make_libero_vec_env(task_config, seed=seed)
    from jepa_policy.envs.robomimic.robomimic_env import (
        make_vec_env as make_robomimic_vec_env,
    )

    return make_robomimic_vec_env(task_config, seed=seed)
