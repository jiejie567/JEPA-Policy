import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mip.datasets.robocasa_dataset import (
    ARRAY_CACHE_MARKER,
    SOURCE_TO_POLICY_KEY,
    RoboCasaDataset,
    _ArrayCachedRoboCasaDataset,
)


def _write_tiny_cache(root):
    (root / "meta").mkdir(parents=True)
    (root / "arrays").mkdir()
    info = {
        "total_frames": 5,
        "total_episodes": 2,
        "fps": 20,
        "features": {
            "observation.state": {"shape": [2]},
            "action": {"shape": [2]},
        },
    }
    stats = {
        "observation.state": {
            "min": [0.0, 10.0],
            "max": [4.0, 14.0],
        },
        "action": {"min": [0.0, 20.0], "max": [4.0, 24.0]},
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    (root / "meta" / "stats.json").write_text(json.dumps(stats))

    arrays = {}
    for source_key, policy_key in SOURCE_TO_POLICY_KEY.items():
        if source_key.startswith("observation.images."):
            value = np.broadcast_to(
                np.arange(5, dtype=np.uint8)[:, None, None, None],
                (5, 3, 2, 2),
            ).copy()
            filename = f"{policy_key}.npy"
        else:
            value = np.stack(
                [np.arange(5), np.arange(10, 15)], axis=-1
            ).astype(np.float32)
            filename = "state.npy"
        np.save(root / "arrays" / filename, value)
        arrays[source_key] = f"arrays/{filename}"
    action = np.stack(
        [np.arange(5), np.arange(20, 25)], axis=-1
    ).astype(np.float32)
    np.save(root / "arrays" / "action.npy", action)
    arrays["action"] = "arrays/action.npy"
    np.save(
        root / "arrays" / "episode_ends.npy",
        np.asarray([3, 5], dtype=np.int64),
    )
    marker = {
        "cache_format_version": 2,
        "partial": False,
        "total_frames": 5,
        "total_episodes": 2,
        "frame_shape_chw": [3, 2, 2],
        "arrays": arrays,
        "episode_ends": "arrays/episode_ends.npy",
    }
    (root / ARRAY_CACHE_MARKER).write_text(json.dumps(marker))
    return info, marker


def test_array_cache_clamps_queries_at_episode_boundaries(tmp_path):
    info, marker = _write_tiny_cache(tmp_path)
    dataset = _ArrayCachedRoboCasaDataset(
        tmp_path,
        marker,
        info,
        selected_episodes=None,
        frame_offsets=[0, 1, 5],
        action_horizon=4,
    )

    last_first_episode = dataset[2]
    assert last_first_episode["action"][:, 0].tolist() == [2.0] * 4
    assert last_first_episode["observation.state"][:, 0].tolist() == [
        2.0,
        2.0,
        2.0,
    ]

    first_second_episode = dataset[3]
    assert first_second_episode["action"][:, 0].tolist() == [
        3.0,
        4.0,
        4.0,
        4.0,
    ]
    assert first_second_episode["observation.state"][:, 0].tolist() == [
        3.0,
        4.0,
        4.0,
    ]
    for source_key in SOURCE_TO_POLICY_KEY:
        if source_key.startswith("observation.images."):
            assert first_second_episode[source_key].dtype == torch.uint8


def test_array_cache_maps_selected_episode_local_indices(tmp_path):
    info, marker = _write_tiny_cache(tmp_path)
    dataset = _ArrayCachedRoboCasaDataset(
        tmp_path,
        marker,
        info,
        selected_episodes=[1],
        frame_offsets=[0, 1],
        action_horizon=2,
    )

    assert len(dataset) == 2
    assert dataset[0]["observation.state"][:, 0].tolist() == [3.0, 4.0]
    assert dataset[1]["observation.state"][:, 0].tolist() == [4.0, 4.0]


def test_partial_array_cache_is_rejected(tmp_path):
    info, marker = _write_tiny_cache(tmp_path)
    marker["partial"] = True
    with pytest.raises(ValueError, match="Partial RoboCasa array cache"):
        _ArrayCachedRoboCasaDataset(
            tmp_path,
            marker,
            info,
            selected_episodes=None,
            frame_offsets=[0, 1],
            action_horizon=2,
        )


def test_public_dataset_keeps_cached_images_uint8(tmp_path):
    _write_tiny_cache(tmp_path)
    task = SimpleNamespace(
        dataset_path=str(tmp_path),
        env_name="tiny",
        obs_steps=2,
        horizon=4,
        future_state_enabled=True,
        future_state_steps_list=[4],
        future_state_steps=4,
        val_dataset_percentage=0.0,
    )
    dataset = RoboCasaDataset(task)
    sample = dataset[2]

    assert sample["obs"]["agentview_left_image"].dtype == torch.uint8
    assert sample["obs"]["agentview_left_image"][:, 0, 0, 0].tolist() == [
        2,
        2,
    ]
    assert (
        sample["future_obs"]["agentview_left_image"][0, 0, 0].item() == 2
    )
    assert sample["obs"]["state"].dtype == torch.float32
    assert sample["action"].dtype == torch.float32

    task.future_state_enabled = False
    baseline = RoboCasaDataset(task)
    baseline_sample = baseline[2]
    assert "future_obs" not in baseline_sample
    assert (
        baseline_sample["obs"]["agentview_left_image"].dtype
        == torch.uint8
    )
