"""Dataset factory for robot-manipulation benchmarks."""

from __future__ import annotations

from mip.datasets.libero_dataset import make_dataset as make_libero_dataset
from mip.datasets.robomimic_dataset import make_dataset as make_robomimic_dataset
from mip.libero_utils import is_libero_task


def make_dataset(task_config, mode="train"):
    if is_libero_task(task_config):
        return make_libero_dataset(task_config, mode=mode)
    return make_robomimic_dataset(task_config, mode=mode)
