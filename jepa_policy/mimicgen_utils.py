"""Helpers for identifying MimicGen tasks and their isolated runtime."""

from __future__ import annotations


MIMICGEN_TASK_NAMES = {
    "coffee_preparation",
    "three_piece_assembly",
    "mimicgen_kitchen",
    "hammer_cleanup",
}


def is_mimicgen_task(task_config) -> bool:
    env_type = getattr(task_config, "env_type", None)
    env_name = getattr(task_config, "env_name", None)
    return env_type == "mimicgen" or env_name in MIMICGEN_TASK_NAMES
