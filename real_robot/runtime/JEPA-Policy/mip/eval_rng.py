"""Deterministic, training-independent random state for rollout evaluation."""

from __future__ import annotations

import random
from contextlib import contextmanager
from typing import Iterator, Sequence

import numpy as np
import torch


def get_rollout_seed(config) -> int:
    """Return the evaluation-only seed, with a safe default for old configs."""
    seed = int(getattr(config.eval, "rollout_seed", 12345))
    if seed < 0:
        raise ValueError("eval.rollout_seed must be non-negative")
    return seed


def get_episode_seeds(base_seed: int, episode_ids: Sequence[int]) -> list[int]:
    """Assign a stable environment seed to each logical episode."""
    base_seed = int(base_seed)
    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")
    seeds = []
    for episode_id in episode_ids:
        episode_id = int(episode_id)
        if episode_id < 0:
            raise ValueError("episode IDs must be non-negative")
        seeds.append(base_seed + episode_id)
    return seeds


@contextmanager
def isolated_torch_rng(seed: int, device: str | torch.device) -> Iterator[None]:
    """Seed rollout RNGs temporarily and restore all training RNGs afterward."""
    seed = int(seed)
    if seed < 0:
        raise ValueError("rollout RNG seed must be non-negative")

    torch_device = torch.device(device)
    cuda_devices: list[int] = []
    if torch_device.type == "cuda":
        cuda_devices = [
            torch_device.index
            if torch_device.index is not None
            else torch.cuda.current_device()
        ]

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if cuda_devices:
                torch.cuda.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
