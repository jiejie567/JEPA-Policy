"""Standalone Diffusion Policy adapter for validated ARX R5 array caches."""

import json
import os
from pathlib import Path

import numpy as np
import torch

from diffusion_policy.common.normalize_util import (
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer


ARRAYS = {
    "base_image": "base_image.npy",
    "left_wrist_image": "left_wrist_image.npy",
    "right_wrist_image": "right_wrist_image.npy",
    "state": "state.npy",
    "action": "action.npy",
}


class ARXR5ImageDataset(BaseImageDataset):
    """Read raw RGB/joint sequences without importing the JEPA/MIP package."""

    def __init__(
        self,
        dataset_path,
        task_name,
        horizon=10,
        n_obs_steps=2,
        val_ratio=0.02,
        mode="train",
    ):
        self.root = Path(dataset_path).expanduser().resolve()
        self.dataset_path = str(self.root)
        self.task_name = str(task_name)
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.val_ratio = float(val_ratio)
        self.mode = str(mode)

        marker_path = self.root / ".jepa_arx_r5_cache.json"
        stats_path = self.root / "stats.json"
        if not marker_path.is_file() or not stats_path.is_file():
            raise FileNotFoundError(f"ARX R5 cache is incomplete: {self.root}")
        self.marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if self.marker.get("cache_format_version") != 1:
            raise ValueError("unsupported ARX R5 cache format")
        if self.marker.get("partial", True):
            raise ValueError(f"partial ARX R5 cache cannot be trained: {self.root}")
        if self.marker.get("task") != self.task_name:
            raise ValueError(
                f"cache task {self.marker.get('task')!r} != {self.task_name!r}"
            )
        if not np.isclose(float(self.marker["frequency_hz"]), 10.0):
            raise ValueError("ARX R5 cache must use a 10 Hz timestamp grid")

        self.total_frames = int(self.marker["total_frames"])
        self.total_episodes = int(self.marker["total_episodes"])
        self.image_size = int(self.marker["image_size"])
        self.array_paths = {key: self.root / filename for key, filename in ARRAYS.items()}
        for key, path in self.array_paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"missing ARX R5 {key}: {path}")

        self.global_episode_ends = np.asarray(
            np.load(self.root / "episode_ends.npy", mmap_mode="r"), dtype=np.int64
        ).copy()
        if (
            self.global_episode_ends.shape != (self.total_episodes,)
            or int(self.global_episode_ends[-1]) != self.total_frames
            or np.any(np.diff(self.global_episode_ends) <= 0)
        ):
            raise ValueError("invalid ARX R5 episode boundaries")

        val_count = int(self.total_episodes * self.val_ratio)
        train_count = self.total_episodes - val_count
        if self.mode == "train":
            selected = np.arange(train_count if val_count else self.total_episodes)
        elif self.mode == "val" and val_count:
            selected = np.arange(train_count, self.total_episodes)
        else:
            raise ValueError("validation requires val_ratio > 0")
        self.selected_episodes = selected.astype(np.int64)
        global_starts = np.concatenate(
            [np.zeros(1, dtype=np.int64), self.global_episode_ends[:-1]]
        )
        lengths = (
            self.global_episode_ends[self.selected_episodes]
            - global_starts[self.selected_episodes]
        )
        self.local_episode_ends = np.cumsum(lengths, dtype=np.int64)
        self.obs_offsets = np.arange(self.n_obs_steps, dtype=np.int64)
        self.action_offsets = np.arange(self.horizon, dtype=np.int64)
        self._arrays = {}
        self._arrays_pid = None
        self.stats = json.loads(stats_path.read_text(encoding="utf-8"))

    def __len__(self):
        return int(self.local_episode_ends[-1])

    def _get_array(self, key):
        pid = os.getpid()
        if self._arrays_pid != pid:
            self._arrays = {}
            self._arrays_pid = pid
        if key not in self._arrays:
            array = np.load(self.array_paths[key], mmap_mode="r")
            expected_tail = (
                (3, self.image_size, self.image_size)
                if key.endswith("_image")
                else (14,)
            )
            expected_dtype = np.dtype(np.uint8 if key.endswith("_image") else np.float32)
            if array.shape != (self.total_frames, *expected_tail) or array.dtype != expected_dtype:
                raise ValueError(f"invalid {key} array: {array.shape} {array.dtype}")
            self._arrays[key] = array
        return self._arrays[key]

    def _global_index_and_bounds(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        position = int(np.searchsorted(self.local_episode_ends, index, side="right"))
        local_start = 0 if position == 0 else int(self.local_episode_ends[position - 1])
        episode = int(self.selected_episodes[position])
        global_start = 0 if episode == 0 else int(self.global_episode_ends[episode - 1])
        global_end = int(self.global_episode_ends[episode])
        return global_start + index - local_start, global_start, global_end

    @staticmethod
    def _indices(index, start, end, offsets):
        return np.clip(index + offsets, start, end - 1)

    def __getitem__(self, index):
        global_index, start, end = self._global_index_and_bounds(index)
        obs_indices = self._indices(global_index, start, end, self.obs_offsets)
        action_indices = self._indices(global_index, start, end, self.action_offsets)
        obs = {}
        for key in ("base_image", "left_wrist_image", "right_wrist_image", "state"):
            value = np.asarray(self._get_array(key)[obs_indices]).copy()
            tensor = torch.from_numpy(value)
            if key.endswith("_image"):
                tensor = tensor.to(dtype=torch.float32).div_(255.0)
            obs[key] = tensor
        action = np.asarray(
            self._get_array("action")[action_indices], dtype=np.float32
        ).copy()
        return {"obs": obs, "action": torch.from_numpy(action)}

    def get_validation_dataset(self):
        return ARXR5ImageDataset(
            self.dataset_path,
            self.task_name,
            self.horizon,
            self.n_obs_steps,
            self.val_ratio,
            mode="val",
        )

    def get_normalizer(self, **kwargs):
        del kwargs
        normalizer = LinearNormalizer()
        for key in ("base_image", "left_wrist_image", "right_wrist_image"):
            normalizer[key] = get_image_range_normalizer()
        for key in ("state", "action"):
            stat = {
                "min": np.asarray(self.stats[key]["min"], dtype=np.float32),
                "max": np.asarray(self.stats[key]["max"], dtype=np.float32),
            }
            normalizer[key] = get_range_normalizer_from_stat(stat)
        return normalizer

    def get_all_actions(self):
        action = np.load(self.array_paths["action"], mmap_mode="r")
        return torch.from_numpy(np.asarray(action).copy())
