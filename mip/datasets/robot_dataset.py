"""Dataset factory for robot-manipulation benchmarks."""

from __future__ import annotations

from mip.libero_utils import is_libero_task
from mip.robocasa_utils import is_robocasa_task
from mip.robotwin_utils import is_robotwin_task


def make_dataset(task_config, mode="train"):
    if is_libero_task(task_config):
        from mip.datasets.libero_dataset import make_dataset as make_libero_dataset

        return make_libero_dataset(task_config, mode=mode)
    if is_robocasa_task(task_config):
        from mip.datasets.robocasa_dataset import make_dataset as make_robocasa_dataset

        return make_robocasa_dataset(task_config, mode=mode)
    if is_robotwin_task(task_config):
        from mip.datasets.robotwin_dataset import make_dataset as make_robotwin_dataset

        return make_robotwin_dataset(task_config, mode=mode)
    from mip.datasets.robomimic_dataset import make_dataset as make_robomimic_dataset

    return make_robomimic_dataset(task_config, mode=mode)
