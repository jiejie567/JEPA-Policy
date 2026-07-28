"""RoboTwin Gym adapter for the common JEPA-Policy rollout interface."""

from __future__ import annotations

import gc
import importlib
import os
from contextlib import contextmanager
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

from mip.env_utils import MultiStepWrapper


IMAGE_KEYS = {
    "head_camera": "head_camera_image",
    "left_camera": "left_wrist_image",
    "right_camera": "right_wrist_image",
}


def _read_yaml(path: Path):
    with path.open("r", encoding="utf-8") as source:
        return yaml.safe_load(source)


@contextmanager
def _robotwin_working_directory(root: Path):
    """Contain upstream's setup-time relative asset paths."""
    previous = Path.cwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(previous)


def _build_task_args(task_config):
    root = Path(task_config.robotwin_root).expanduser().resolve()
    args = _read_yaml(
        root / "task_config" / f"{task_config.robotwin_task_config}.yml"
    )
    args["task_name"] = task_config.robotwin_task_name
    args["task_config"] = task_config.robotwin_task_config
    args["render_freq"] = 0
    args["save_data"] = False
    args["collect_data"] = False
    args["eval_mode"] = True
    args["eval_video_log"] = False
    args["eval_video_save_dir"] = None
    # A policy qpos rollout uses MPLib only for TOPP interpolation. CuRobo is
    # reserved for expert trajectory generation / official seed filtering.
    args["need_plan"] = False

    embodiment_config = _read_yaml(root / "task_config/_embodiment_config.yml")
    embodiment = args["embodiment"]
    if len(embodiment) == 1:
        relative = embodiment_config[embodiment[0]]["file_path"]
        robot_file = (root / relative).resolve()
        args["left_robot_file"] = str(robot_file)
        args["right_robot_file"] = str(robot_file)
        args["dual_arm_embodied"] = True
    elif len(embodiment) == 3:
        left_relative = embodiment_config[embodiment[0]]["file_path"]
        right_relative = embodiment_config[embodiment[1]]["file_path"]
        args["left_robot_file"] = str((root / left_relative).resolve())
        args["right_robot_file"] = str((root / right_relative).resolve())
        args["embodiment_dis"] = embodiment[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError(f"Invalid RoboTwin embodiment: {embodiment!r}")

    args["left_embodiment_config"] = _read_yaml(
        Path(args["left_robot_file"]) / "config.yml"
    )
    args["right_embodiment_config"] = _read_yaml(
        Path(args["right_robot_file"]) / "config.yml"
    )
    return args


class RoboTwinPolicyEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, task_config, seed=None):
        self.task_config = task_config
        self.image_size = int(task_config.eval_image_size)
        self.default_seed = None if seed is None else int(seed)
        self.root = Path(task_config.robotwin_root).expanduser().resolve()
        self.args = _build_task_args(task_config)

        module = importlib.import_module(
            f"envs.{task_config.robotwin_task_name}"
        )
        env_class = getattr(module, task_config.robotwin_task_name)
        self.task = env_class()
        self.last_obs = None
        self.actual_seed = None
        self.episode_index = 0

        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(task_config.act_dim),),
            dtype=np.float32,
        )
        obs_spaces = {
            key: spaces.Box(
                low=0.0,
                high=1.0,
                shape=(3, self.image_size, self.image_size),
                dtype=np.float32,
            )
            for key in IMAGE_KEYS.values()
        }
        obs_spaces["state"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(task_config.obs_dim),),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(obs_spaces)

    def _convert_obs(self, raw_obs):
        obs = {}
        for source_key, policy_key in IMAGE_KEYS.items():
            image = raw_obs["observation"][source_key]["rgb"]
            image = cv2.resize(
                image,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_AREA,
            )
            obs[policy_key] = (
                np.moveaxis(image, -1, 0).astype(np.float32) / 255.0
            )
        joint = raw_obs["joint_action"]
        obs["state"] = np.concatenate(
            [
                np.asarray(joint["left_arm"], dtype=np.float32),
                np.atleast_1d(
                    np.asarray(joint["left_gripper"], dtype=np.float32)
                ),
                np.asarray(joint["right_arm"], dtype=np.float32),
                np.atleast_1d(
                    np.asarray(joint["right_gripper"], dtype=np.float32)
                ),
            ]
        )
        if obs["state"].shape != (int(self.task_config.obs_dim),):
            raise ValueError(f"Unexpected RoboTwin state: {obs['state'].shape}")
        return obs

    def reset(self, seed=None, options=None):
        del options
        requested_seed = (
            self.default_seed if seed is None else int(seed)
        )
        if requested_seed is None:
            requested_seed = 0

        # Match official robustness to unstable scene samples while keeping the
        # selected seed deterministic across baseline and Future runs.
        from envs.utils.create_actor import UnStableError

        last_error = None
        for candidate_seed in range(requested_seed, requested_seed + 100):
            try:
                with _robotwin_working_directory(self.root):
                    self.task.setup_demo(
                        now_ep_num=self.episode_index,
                        seed=candidate_seed,
                        is_test=True,
                        **self.args,
                    )
                self.actual_seed = candidate_seed
                break
            except UnStableError as exc:
                last_error = exc
                try:
                    self.task.close_env(clear_cache=True)
                except Exception:
                    pass
        else:
            raise RuntimeError(
                f"No stable RoboTwin scene in 100 seeds from {requested_seed}"
            ) from last_error

        self.episode_index += 1
        self.last_obs = self._convert_obs(self.task.get_obs())
        return self.last_obs, {
            "requested_seed": requested_seed,
            "actual_seed": self.actual_seed,
        }

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (int(self.task_config.act_dim),):
            raise ValueError(
                f"Expected RoboTwin qpos action "
                f"({self.task_config.act_dim},), got {action.shape}"
            )
        self.task.take_action(action, action_type="qpos")
        self.last_obs = self._convert_obs(self.task.get_obs())
        success = bool(self.task.eval_success or self.task.check_success())
        limit_reached = self.task.take_action_cnt >= self.task.step_lim
        return (
            self.last_obs,
            float(success),
            success,
            bool(limit_reached and not success),
            {"success": success, "actual_seed": self.actual_seed},
        )

    def render(self):
        if self.last_obs is None:
            return None
        image = self.last_obs["head_camera_image"]
        return np.moveaxis(np.clip(image * 255.0, 0, 255).astype(np.uint8), 0, -1)

    def close(self):
        if getattr(self, "task", None) is not None:
            try:
                self.task.close_env(clear_cache=True)
            finally:
                self.task = None
                gc.collect()


def make_vec_env(task_config, seed=None):
    def thunk(index):
        def create():
            env_seed = None if seed is None else int(seed) + index
            env = RoboTwinPolicyEnv(task_config, seed=env_seed)
            return MultiStepWrapper(
                env,
                n_obs_steps=int(task_config.obs_steps),
                n_action_steps=int(task_config.act_steps),
                max_episode_steps=int(task_config.max_episode_steps),
            )

        return create

    return gym.vector.SyncVectorEnv(
        [thunk(index) for index in range(int(task_config.num_envs))]
    )
