"""Dataset factory for robot-manipulation benchmarks."""

from __future__ import annotations

from jepa_policy.libero_utils import is_libero_task


def make_dataset(task_config, mode="train"):
    if is_libero_task(task_config):
        from jepa_policy.datasets.libero_dataset import make_dataset as make_libero_dataset

        return make_libero_dataset(task_config, mode=mode)
    from jepa_policy.datasets.robomimic_dataset import make_dataset as make_robomimic_dataset

    return make_robomimic_dataset(task_config, mode=mode)
