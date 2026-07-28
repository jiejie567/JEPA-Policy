import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mip.datasets.robotwin_dataset import RoboTwinDataset


def _write_tiny_cache(root):
    total_frames = 5
    image_shape = (total_frames, 3, 2, 2)
    for filename in ("head_camera.npy", "left_camera.npy", "right_camera.npy"):
        image = np.broadcast_to(
            np.arange(total_frames, dtype=np.uint8)[:, None, None, None],
            image_shape,
        ).copy()
        np.save(root / filename, image)

    state = np.stack(
        [np.arange(total_frames), np.arange(10, 10 + total_frames)], axis=-1
    ).astype(np.float32)
    action = np.stack(
        [np.arange(total_frames), np.arange(20, 20 + total_frames)], axis=-1
    ).astype(np.float32)
    np.save(root / "state.npy", state)
    np.save(root / "action.npy", action)
    np.save(root / "episode_ends.npy", np.asarray([3, 5], dtype=np.int64))
    (root / "stats.json").write_text(
        json.dumps(
            {
                "state": {"min": [0.0, 10.0], "max": [4.0, 14.0]},
                "action": {"min": [0.0, 20.0], "max": [4.0, 24.0]},
            }
        )
    )
    marker = {
        "cache_format_version": 1,
        "partial": False,
        "total_episodes": 2,
        "total_frames": total_frames,
        "state_dim": 2,
        "action_dim": 2,
        "image_size": 2,
    }
    (root / ".jepa_robotwin_cache.json").write_text(json.dumps(marker))
    return marker


def _task(root, future=True):
    return SimpleNamespace(
        dataset_path=str(root),
        dataset_num_episodes=2,
        obs_steps=2,
        horizon=4,
        future_state_enabled=future,
        future_state_steps_list=[4],
        future_state_steps=4,
        eval_image_size=2,
        obs_dim=2,
        act_dim=2,
    )


def test_robotwin_cache_clamps_current_future_and_action_per_episode(tmp_path):
    _write_tiny_cache(tmp_path)
    dataset = RoboTwinDataset(_task(tmp_path))

    last_first_episode = dataset[2]
    assert last_first_episode["obs"]["state"][:, 0].tolist() == [0.0, 0.0]
    assert last_first_episode["action"][:, 0].tolist() == [0.0] * 4
    assert (
        last_first_episode["future_obs"]["head_camera_image"][0, 0, 0].item()
        == 2
    )

    first_second_episode = dataset[3]
    assert first_second_episode["obs"]["head_camera_image"][:, 0, 0, 0].tolist() == [
        3,
        4,
    ]
    assert first_second_episode["action"][:, 0].tolist() == [0.5, 1.0, 1.0, 1.0]
    assert (
        first_second_episode["future_obs"]["head_camera_image"][0, 0, 0].item()
        == 4
    )


def test_robotwin_cache_keeps_images_uint8_and_baseline_has_no_future(tmp_path):
    _write_tiny_cache(tmp_path)

    future_sample = RoboTwinDataset(_task(tmp_path, future=True))[0]
    assert future_sample["obs"]["head_camera_image"].dtype == torch.uint8
    assert future_sample["future_obs"]["head_camera_image"].dtype == torch.uint8
    assert future_sample["obs"]["state"].dtype == torch.float32
    assert future_sample["action"].dtype == torch.float32

    baseline_sample = RoboTwinDataset(_task(tmp_path, future=False))[0]
    assert "future_obs" not in baseline_sample


def test_robotwin_partial_cache_is_rejected(tmp_path):
    marker = _write_tiny_cache(tmp_path)
    marker["partial"] = True
    (tmp_path / ".jepa_robotwin_cache.json").write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="Partial RoboTwin cache"):
        RoboTwinDataset(_task(tmp_path))
