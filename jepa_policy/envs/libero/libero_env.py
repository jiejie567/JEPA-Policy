"""Environment creation for LIBERO tasks."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from jepa_policy.config import TaskConfig
from jepa_policy.envs.egl_device import (
    configured_egl_device_id,
    install_robosuite_egl_device_override,
)
from jepa_policy.env_utils import MultiStepWrapper
from jepa_policy.libero_utils import (
    get_libero_import_paths,
    get_libero_task_spec,
    resolve_libero_asset_dir,
)

_LIBERO_OBS_KEY_MAP = {
    "agentview_rgb": "agentview_image",
    "eye_in_hand_rgb": "robot0_eye_in_hand_image",
    "joint_states": "robot0_joint_pos",
    "gripper_states": "robot0_gripper_qpos",
    "ee_pos": "robot0_eef_pos",
    "ee_ori": "robot0_eef_quat",
    "ee_states": "robot0_proprio-state",
}


def _make_lowdim_box(example: np.ndarray) -> spaces.Box:
    low = np.full_like(example, fill_value=-1, dtype=np.float32)
    high = np.full_like(example, fill_value=1, dtype=np.float32)
    return spaces.Box(low=low, high=high, shape=example.shape, dtype=np.float32)


class LiberoLowdimWrapper(gym.Env):
    def __init__(self, env, obs_keys: list[str], init_states=None):
        self.env = env
        self.obs_keys = obs_keys
        self.init_states = init_states
        self._seed = None

        action_low, action_high = self.env.env.action_spec
        self.action_space = spaces.Box(
            low=action_low.astype(np.float32),
            high=action_high.astype(np.float32),
            dtype=np.float32,
        )

        example_obs = self._convert_obs(self.env.reset())
        self.observation_space = _make_lowdim_box(example_obs)

    def _convert_obs(self, raw_obs):
        parts = []
        for key in self.obs_keys:
            env_key = _LIBERO_OBS_KEY_MAP.get(key, key)
            parts.append(raw_obs[env_key])
        return np.concatenate(parts, axis=0).astype(np.float32)

    def seed(self, seed=None):
        self._seed = seed
        if seed is not None:
            self.env.seed(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        raw_obs = self.env.reset()
        if self._seed is not None and self.init_states is not None and len(self.init_states) > 0:
            idx = int(self._seed % len(self.init_states))
            raw_obs = self.env.set_init_state(self.init_states[idx])
            self._seed = None

        return self._convert_obs(raw_obs), {}

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        return self._convert_obs(raw_obs), reward, done, info

    def close(self):
        self.env.close()


class LiberoImageWrapper(gym.Env):
    def __init__(self, env, shape_meta: dict, init_states=None):
        self.env = env
        self.shape_meta = shape_meta
        self.init_states = init_states
        self._seed = None

        action_low, action_high = self.env.env.action_spec
        self.action_space = spaces.Box(
            low=action_low.astype(np.float32),
            high=action_high.astype(np.float32),
            dtype=np.float32,
        )

        observation_space = spaces.Dict()
        for key, value in shape_meta["obs"].items():
            if value.get("type", "low_dim") == "rgb":
                observation_space[key] = spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=tuple(value["shape"]),
                    dtype=np.float32,
                )
            else:
                observation_space[key] = spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=tuple(value["shape"]),
                    dtype=np.float32,
                )
        self.observation_space = observation_space

    def _convert_obs(self, raw_obs):
        obs = {}
        for key in self.shape_meta["obs"]:
            env_key = _LIBERO_OBS_KEY_MAP.get(key, key)
            value = raw_obs[env_key]
            if self.shape_meta["obs"][key].get("type", "low_dim") == "rgb":
                value = np.moveaxis(value, -1, 0).astype(np.float32) / 255.0
            else:
                value = value.astype(np.float32)
            obs[key] = value
        return obs

    def seed(self, seed=None):
        self._seed = seed
        if seed is not None:
            self.env.seed(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        raw_obs = self.env.reset()
        if self._seed is not None and self.init_states is not None and len(self.init_states) > 0:
            idx = int(self._seed % len(self.init_states))
            raw_obs = self.env.set_init_state(self.init_states[idx])
            self._seed = None

        return self._convert_obs(raw_obs), {}

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        return self._convert_obs(raw_obs), reward, done, info

    def close(self):
        self.env.close()


def _load_init_states(task_config: TaskConfig, spec) -> np.ndarray:
    import torch

    init_states_dir = resolve_libero_asset_dir(task_config, "init_states")
    init_states_path = init_states_dir / spec.benchmark_name / spec.init_states_file
    init_states = torch.load(init_states_path, map_location="cpu", weights_only=False)
    if isinstance(init_states, np.ndarray):
        return init_states
    return init_states.numpy()


def _build_env(task_config: TaskConfig):
    install_robosuite_egl_device_override()
    libero_root = getattr(task_config, "libero_root", None)
    if libero_root:
        os.environ.setdefault(
            "LIBERO_CONFIG_PATH", str(Path(libero_root).expanduser() / ".libero_config")
        )
    try:
        try:
            from libero.libero.envs import OffScreenRenderEnv
        except ImportError:
            from libero.envs import OffScreenRenderEnv
    except ImportError as exc:
        for import_path in get_libero_import_paths(task_config):
            if not import_path.exists():
                continue
            sys.path.insert(0, str(import_path))
            try:
                try:
                    from libero.libero.envs import OffScreenRenderEnv
                except ImportError:
                    from libero.envs import OffScreenRenderEnv
                break
            except ImportError:
                continue
        else:
            raise ImportError(
                "LIBERO support requires either an importable LIBERO checkout or "
                "task.libero_root pointing at a local LIBERO repository root."
            ) from exc

    spec = get_libero_task_spec(task_config)
    bddl_dir = resolve_libero_asset_dir(task_config, "bddl_files")
    bddl_path = Path(bddl_dir) / spec.benchmark_name / spec.bddl_file
    camera_size = getattr(task_config, "libero_camera_size", 128)
    egl_device_id = configured_egl_device_id()
    render_kwargs = (
        {} if egl_device_id is None else {"render_gpu_device_id": egl_device_id}
    )
    return OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=camera_size,
        camera_widths=camera_size,
        **render_kwargs,
    )


def make_env(task_config: TaskConfig, idx, render=False, seed=None):
    spec = get_libero_task_spec(task_config)
    init_states = _load_init_states(task_config, spec)

    def thunk():
        env = _build_env(task_config)
        if task_config.obs_type == "state":
            wrapped = LiberoLowdimWrapper(env=env, obs_keys=task_config.obs_keys, init_states=init_states)
        else:
            wrapped = LiberoImageWrapper(env=env, shape_meta=task_config.shape_meta, init_states=init_states)

        env_seed = None if seed is None else seed + idx
        wrapped = MultiStepWrapper(
            wrapped,
            n_obs_steps=task_config.obs_steps,
            n_action_steps=task_config.act_steps,
            max_episode_steps=task_config.max_episode_steps,
        )
        if env_seed is not None:
            wrapped.seed(env_seed)
        return wrapped

    return thunk


def make_vec_env(task_config: TaskConfig, seed=None):
    env_cls = gym.vector.SyncVectorEnv if (task_config.num_envs == 1 or task_config.obs_type == "image") else gym.vector.AsyncVectorEnv
    return env_cls([make_env(task_config, idx, False, seed=seed) for idx in range(task_config.num_envs)])
