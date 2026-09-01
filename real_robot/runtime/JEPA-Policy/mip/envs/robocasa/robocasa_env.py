"""RoboCasa Gym adapter for the common JEPA-Policy rollout interface."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from mip.env_utils import MultiStepWrapper
from mip.envs.egl_device import install_robosuite_egl_device_override


IMAGE_KEYS = {
    "video.robot0_agentview_left": "agentview_left_image",
    "video.robot0_agentview_right": "agentview_right_image",
    "video.robot0_eye_in_hand": "robot0_eye_in_hand_image",
}
STATE_KEYS = (
    "state.base_position",
    "state.base_rotation",
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.gripper_qpos",
)
ACTION_SLICES = (
    # Keep this order aligned with RoboCasa's PandaOmron modality.json and
    # the raw LeRobot action vectors used for training.
    ("action.base_motion", 4),
    ("action.control_mode", 1),
    ("action.end_effector_position", 3),
    ("action.end_effector_rotation", 3),
    ("action.gripper_close", 1),
)


class RoboCasaPolicyEnv(gym.Env):
    def __init__(self, task_config, seed=None):
        import robocasa  # noqa: F401 - registers environments

        # CUDA_VISIBLE_DEVICES contains a physical training slot before
        # PyTorch narrows it to local cuda:0. Mesa exposes only local EGL
        # device 0, so install the explicit namespace translation before
        # robosuite creates its first offscreen context.
        install_robosuite_egl_device_override()
        self.image_size = int(
            getattr(task_config, "eval_image_size", 128)
        )
        if self.image_size < 1:
            raise ValueError("task.eval_image_size must be positive")
        self.env = gym.make(
            f"robocasa/{task_config.env_name}",
            split=getattr(task_config, "robocasa_split", "target"),
            seed=seed,
            camera_widths=self.image_size,
            camera_heights=self.image_size,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )
        obs_spaces = {
            policy_key: spaces.Box(
                low=0.0,
                high=1.0,
                shape=(3, self.image_size, self.image_size),
                dtype=np.float32,
            )
            for policy_key in IMAGE_KEYS.values()
        }
        obs_spaces["state"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32
        )
        self.observation_space = spaces.Dict(obs_spaces)

    @staticmethod
    def _convert_obs(raw_obs):
        obs = {
            policy_key: np.moveaxis(raw_obs[source_key], -1, 0).astype(np.float32)
            / 255.0
            for source_key, policy_key in IMAGE_KEYS.items()
        }
        obs["state"] = np.concatenate(
            [np.asarray(raw_obs[key], dtype=np.float32) for key in STATE_KEYS]
        )
        return obs

    @staticmethod
    def _convert_action(action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (12,):
            raise ValueError(f"Expected flat RoboCasa action (12,), got {action.shape}")
        output = {}
        start = 0
        for key, width in ACTION_SLICES:
            output[key] = action[start : start + width]
            start += width
        return output

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._convert_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(
            self._convert_action(action)
        )
        return self._convert_obs(obs), reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


def make_vec_env(task_config, seed=None):
    def thunk(index):
        def create():
            env_seed = None if seed is None else int(seed) + index
            env = RoboCasaPolicyEnv(task_config, seed=env_seed)
            return MultiStepWrapper(
                env,
                n_obs_steps=int(task_config.obs_steps),
                n_action_steps=int(task_config.act_steps),
                max_episode_steps=int(task_config.max_episode_steps),
            )

        return create

    # RoboCasa render contexts are process-local and expensive; the persistent
    # rollout pool provides process parallelism when requested.
    return gym.vector.SyncVectorEnv(
        [thunk(index) for index in range(int(task_config.num_envs))]
    )
