"""LIBERO dataset adapters built on top of the robomimic-format loaders."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download

from jepa_policy.datasets.robomimic_dataset import RobomimicDataset, RobomimicImageDataset
from jepa_policy.libero_utils import get_libero_dataset_filename, resolve_libero_asset_dir


def _resolve_dataset_path(task_config) -> str:
    dataset_filename = getattr(task_config, "dataset_filename", None)
    if dataset_filename is None:
        dataset_filename = get_libero_dataset_filename(task_config)

    dataset_path = getattr(task_config, "dataset_path", None)
    if dataset_path:
        return os.path.expanduser(dataset_path)

    try:
        dataset_dir = resolve_libero_asset_dir(task_config, "datasets")
    except ImportError:
        dataset_dir = None

    if dataset_dir is not None:
        candidate = Path(dataset_dir) / dataset_filename
        if candidate.exists():
            return str(candidate)

    dataset_repo = getattr(task_config, "dataset_repo", None)
    if dataset_repo:
        return hf_hub_download(
            repo_id=dataset_repo,
            filename=dataset_filename,
            repo_type="dataset",
        )

    raise ValueError(
        "LIBERO task requires a valid local dataset_path, a dataset in the local "
        "LIBERO datasets directory, or dataset_repo/dataset_filename."
    )


def make_dataset(task_config, mode="train"):
    dataset_path = _resolve_dataset_path(task_config)

    if task_config.obs_type == "state":
        return RobomimicDataset(
            dataset_path,
            horizon=task_config.horizon,
            obs_keys=task_config.obs_keys,
            pad_before=task_config.obs_steps - 1,
            pad_after=task_config.act_steps - 1,
            abs_action=task_config.abs_action,
            mode=mode,
            val_dataset_percentage=task_config.val_dataset_percentage,
        )

    if task_config.obs_type == "image":
        return RobomimicImageDataset(
            dataset_path,
            horizon=task_config.horizon,
            shape_meta=task_config.shape_meta,
            n_obs_steps=task_config.obs_steps,
            pad_before=task_config.obs_steps - 1,
            pad_after=task_config.act_steps - 1,
            abs_action=task_config.abs_action,
            val_dataset_percentage=task_config.val_dataset_percentage,
            mode=mode,
            future_state_enabled=getattr(task_config, "future_state_enabled", False),
            future_state_steps=getattr(task_config, "future_state_steps", 1),
            future_state_steps_list=getattr(task_config, "future_state_steps_list", []),
        )

    raise ValueError(f"Invalid LIBERO observation type: {task_config.obs_type}")
