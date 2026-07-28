#!/usr/bin/env python3
"""Validate the isolated RoboCasa runtime, selected datasets, and environments."""

from __future__ import annotations

import argparse
import json
import os
import sys
from importlib import metadata
from pathlib import Path

import gymnasium as gym
import robocasa  # noqa: F401 - registers gym environments

from robocasa import macros
from robocasa.utils.dataset_registry_utils import get_ds_meta


ROOT = Path("/mnt/data_nas/ykj_jepa_policy")
REPO = ROOT / "code/JEPA-Policy"
DATA_ROOT = REPO / "datasets/robocasa/v1.0/target/composite"
TASKS = {
    "SteamInMicrowave": ("20250814", 2100),
    "StoreLeftoversInBowl": ("20250813", 2550),
    "LoadDishwasher": ("20250811", 1800),
}
EXPECTED_VERSIONS = {
    "robosuite": "1.5.2",
    "numpy": "2.2.5",
    "numba": "0.61.2",
    "scipy": "1.15.3",
    "mujoco": "3.3.1",
    "lerobot": "0.3.3",
}
EXPECTED_PYTHON = ROOT / "venvs/robocasa/bin/python"
EXPECTED_CAMERAS = {
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
    "observation.images.robot0_eye_in_hand",
}


def validate_isolation() -> None:
    assert Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(), sys.executable
    assert os.environ.get("PYTHONNOUSERSITE") == "1"
    assert macros.DATASET_BASE_PATH == str(REPO / "datasets/robocasa")
    module_paths = {
        "robocasa": Path(robocasa.__file__).resolve(),
    }
    for name, path in module_paths.items():
        assert REPO in path.parents, (name, path)
    print(f"ISOLATION_OK python={sys.executable} dataset_root={macros.DATASET_BASE_PATH}")


def validate_versions() -> None:
    actual = {name: metadata.version(name) for name in EXPECTED_VERSIONS}
    assert actual == EXPECTED_VERSIONS, (actual, EXPECTED_VERSIONS)
    assert robocasa.__version__ == "1.0.1", robocasa.__version__
    print(
        "VERSIONS_OK",
        "robocasa=1.0.1",
        " ".join(f"{k}={v}" for k, v in actual.items()),
    )


def validate_datasets() -> None:
    for task, (date, horizon) in TASKS.items():
        expected = DATA_ROOT / task / date / "lerobot"
        meta = get_ds_meta(task=task, split="target", source="human")
        assert meta is not None
        assert Path(meta["path"]).resolve() == expected.resolve(), meta
        assert int(meta["horizon"]) == horizon, meta
        info_path = expected / "meta/info.json"
        assert info_path.is_file(), info_path
        info = json.loads(info_path.read_text())
        episodes = info.get("total_episodes", info.get("total_episodes_num"))
        # The release promises 500 target demonstrations per task. Updated
        # archives can include a small number of additional instruction
        # variants, so reject incomplete data rather than requiring equality.
        assert isinstance(episodes, int) and episodes >= 500, (task, episodes)
        features = info["features"]
        cameras = {name for name in features if name.startswith("observation.images.")}
        assert cameras == EXPECTED_CAMERAS, (task, cameras)
        assert info["total_videos"] == episodes * len(EXPECTED_CAMERAS), (
            task,
            info["total_videos"],
        )
        for directory in ("data", "videos", "extras"):
            assert (expected / directory).is_dir(), expected / directory
        print(
            f"DATASET_OK task={task} episodes={episodes} "
            f"frames={info['total_frames']} videos={info['total_videos']} "
            f"cameras={len(cameras)} horizon={horizon}"
        )


def validate_envs() -> None:
    for task in TASKS:
        env = gym.make(f"robocasa/{task}", split="target", seed=12345)
        try:
            observation, info = env.reset()
            assert observation is not None
            assert isinstance(info, dict)
            action = env.action_space.sample()
            step = env.step(action)
            assert len(step) == 5
        finally:
            env.close()
        print(f"ENV_OK task={task}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-reset", action="store_true")
    args = parser.parse_args()
    validate_isolation()
    validate_versions()
    validate_datasets()
    if args.env_reset:
        validate_envs()


if __name__ == "__main__":
    main()
