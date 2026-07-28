"""Environment factory for robot-manipulation benchmarks."""

from __future__ import annotations

from mip.libero_utils import is_libero_task
from mip.robocasa_utils import is_robocasa_task
from mip.robotwin_utils import is_robotwin_task


def make_vec_env(task_config, seed=None):
    if is_libero_task(task_config):
        from mip.envs.libero.libero_env import make_vec_env as make_libero_vec_env

        return make_libero_vec_env(task_config, seed=seed)
    if is_robocasa_task(task_config):
        from mip.envs.robocasa.robocasa_env import make_vec_env as make_robocasa_vec_env

        return make_robocasa_vec_env(task_config, seed=seed)
    if is_robotwin_task(task_config):
        from mip.envs.robotwin.robotwin_env import make_vec_env as make_robotwin_vec_env

        return make_robotwin_vec_env(task_config, seed=seed)
    from mip.envs.robomimic.robomimic_env import (
        make_vec_env as make_robomimic_vec_env,
    )

    return make_robomimic_vec_env(task_config, seed=seed)
