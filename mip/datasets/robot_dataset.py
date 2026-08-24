"""Dataset factory for robot-manipulation benchmarks."""

from __future__ import annotations

from mip.libero_utils import is_libero_task


def make_dataset(task_config, mode="train"):
    if is_libero_task(task_config):
        from mip.datasets.libero_dataset import make_dataset as make_libero_dataset

        return make_libero_dataset(task_config, mode=mode)
    from mip.datasets.robomimic_dataset import make_dataset as make_robomimic_dataset

    return make_robomimic_dataset(task_config, mode=mode)
