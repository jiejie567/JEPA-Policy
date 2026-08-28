import os
import sys

import gym
import numpy as np
import torch
from gym import spaces


OBS_KEY_MAP = {
    "agentview_rgb": "agentview_image",
    "eye_in_hand_rgb": "robot0_eye_in_hand_image",
    "joint_states": "robot0_joint_pos",
    "gripper_states": "robot0_gripper_qpos",
}


class LiberoImageWrapper(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(
        self,
        libero_root,
        bddl_file,
        init_states_file,
        shape_meta,
        camera_size=128,
        render_obs_key="agentview_rgb",
    ):
        super().__init__()
        libero_root = os.path.abspath(os.path.expanduser(libero_root))
        os.environ.setdefault(
            "LIBERO_CONFIG_PATH", os.path.join(libero_root, ".libero_config")
        )
        try:
            from libero.libero.envs import OffScreenRenderEnv
        except ImportError:
            # The checkout uses a namespace package at <root>/libero and the
            # actual package at <root>/libero/libero.
            if libero_root not in sys.path:
                sys.path.insert(0, libero_root)
            from libero.libero.envs import OffScreenRenderEnv

        self.env = OffScreenRenderEnv(
            bddl_file_name=os.path.abspath(os.path.expanduser(bddl_file)),
            camera_heights=camera_size,
            camera_widths=camera_size,
        )
        init_states = torch.load(
            os.path.abspath(os.path.expanduser(init_states_file)),
            map_location="cpu",
            weights_only=False,
        )
        self.init_states = (
            init_states if isinstance(init_states, np.ndarray) else init_states.numpy()
        )
        self.shape_meta = shape_meta
        self.render_obs_key = render_obs_key
        self._seed = 0
        self._last_raw_obs = None

        action_low, action_high = self.env.env.action_spec
        self.action_space = spaces.Box(
            low=action_low.astype(np.float32),
            high=action_high.astype(np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                key: spaces.Box(
                    low=0.0 if attr.get("type", "low_dim") == "rgb" else -np.inf,
                    high=1.0 if attr.get("type", "low_dim") == "rgb" else np.inf,
                    shape=tuple(attr["shape"]),
                    dtype=np.float32,
                )
                for key, attr in shape_meta["obs"].items()
            }
        )

    def _convert_obs(self, raw_obs):
        result = {}
        for key, attr in self.shape_meta["obs"].items():
            value = raw_obs[OBS_KEY_MAP.get(key, key)]
            if attr.get("type", "low_dim") == "rgb":
                value = np.moveaxis(value, -1, 0).astype(np.float32) / 255.0
            else:
                value = value.astype(np.float32)
            result[key] = value
        return result

    def seed(self, seed=None):
        self._seed = 0 if seed is None else int(seed)
        if hasattr(self.env, "seed"):
            self.env.seed(self._seed)
        return [self._seed]

    def reset(self):
        self.env.reset()
        idx = self._seed % len(self.init_states)
        self._last_raw_obs = self.env.set_init_state(self.init_states[idx])
        return self._convert_obs(self._last_raw_obs)

    def step(self, action):
        self._last_raw_obs, reward, done, info = self.env.step(action)
        return self._convert_obs(self._last_raw_obs), reward, done, info

    def render(self, mode="rgb_array"):
        if self._last_raw_obs is None:
            raise RuntimeError("reset must be called before render")
        return self._last_raw_obs[OBS_KEY_MAP[self.render_obs_key]]

    def close(self):
        self.env.close()
