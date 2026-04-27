"""Robomimic state dataset.

Author: Chaoyi Pan
Date: 2025-10-03
"""

import concurrent.futures
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import zarr
from huggingface_hub import hf_hub_download
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm

from mip.dataset_utils import (
    ImageNormalizer,
    MinMaxNormalizer,
    ReplayBuffer,
    RotationTransformer,
    SequenceSampler,
    dict_apply,
)
from mip.datasets.base import BaseDataset
from mip.datasets.imagecodecs import register_codecs

register_codecs()

_ROBOMIMIC_DATASET_REPO = "amandlek/robomimic"


def _load_robomimic_image_task_config(task: str, source: str):
    config_path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "configs"
        / "task"
        / f"{task}_{source}_image.yaml"
    )
    if not config_path.exists():
        raise FileNotFoundError(f"Task config not found: {config_path}")
    return OmegaConf.load(config_path)


def _get_camera_names_from_task_config(task: str, source: str) -> list[str]:
    cfg = _load_robomimic_image_task_config(task, source)
    camera_names = []
    for key, value in cfg.shape_meta.obs.items():
        if value.get("type") == "rgb":
            camera_names.append(key[:-6] if key.endswith("_image") else key)
    if not camera_names:
        raise ValueError(f"No RGB observations found for task={task}, source={source}")
    return camera_names


def _get_camera_size_from_task_config(task: str, source: str) -> tuple[int, int]:
    cfg = _load_robomimic_image_task_config(task, source)
    for _key, value in cfg.shape_meta.obs.items():
        if value.get("type") == "rgb":
            _channels, height, width = list(value.shape)
            return int(height), int(width)
    raise ValueError(f"No RGB observations found for task={task}, source={source}")


def download_robomimic_dataset(
    task: str,
    source: str,
    dataset_type: str = "low_dim",
) -> str:
    if dataset_type == "low_dim":
        filename = f"v1.5/{task}/{source}/low_dim_v15.hdf5"
    elif dataset_type == "demo":
        filename = f"v1.5/{task}/{source}/demo_v15.hdf5"
    else:
        raise ValueError(
            f"Unsupported dataset_type={dataset_type}. Expected 'low_dim' or 'demo'."
        )

    return hf_hub_download(
        repo_id=_ROBOMIMIC_DATASET_REPO,
        filename=filename,
        repo_type="dataset",
    )


def process_demo_to_image_dataset(
    demo_path: str,
    output_path: str,
    task: str,
    source: str = "ph",
    camera_names: list[str] | None = None,
    camera_height: int | None = None,
    camera_width: int | None = None,
    n_demos: int = -1,
) -> str:
    if camera_names is None:
        camera_names = _get_camera_names_from_task_config(task, source)
    if camera_height is None or camera_width is None:
        height, width = _get_camera_size_from_task_config(task, source)
        camera_height = height if camera_height is None else camera_height
        camera_width = width if camera_width is None else camera_width

    output_path = os.path.abspath(os.path.expanduser(output_path))
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_demo_path = os.path.join(tmp_dir, "demo.hdf5")
        output_name = os.path.basename(output_path)
        tmp_output_path = os.path.join(tmp_dir, output_name)
        shutil.copy2(demo_path, tmp_demo_path)

        cmd = [
            sys.executable,
            "-m",
            "robomimic.scripts.dataset_states_to_obs",
            f"--dataset={tmp_demo_path}",
            f"--output_name={output_name}",
            "--done_mode=2",
            f"--camera_height={camera_height}",
            f"--camera_width={camera_width}",
        ]
        if n_demos != -1:
            cmd.append(f"--n={n_demos}")
        if camera_names:
            cmd.append("--camera_names")
            cmd.extend(camera_names)

        env = os.environ.copy()
        env.setdefault("MUJOCO_GL", "osmesa")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Failed to process demo dataset to image dataset.\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )

        shutil.copy2(tmp_output_path, output_path)
    return output_path


def download_and_process_image_dataset(
    task: str,
    source: str,
    output_path: str,
    camera_names: list[str] | None = None,
    camera_height: int | None = None,
    camera_width: int | None = None,
    n_demos: int = -1,
) -> str:
    demo_path = download_robomimic_dataset(
        task=task,
        source=source,
        dataset_type="demo",
    )
    return process_demo_to_image_dataset(
        demo_path=demo_path,
        output_path=output_path,
        task=task,
        source=source,
        camera_names=camera_names,
        camera_height=camera_height,
        camera_width=camera_width,
        n_demos=n_demos,
    )


def make_dataset(task_config, mode="train"):
    # Check if we should download from HuggingFace
    if hasattr(task_config, "dataset_repo") and hasattr(
        task_config, "dataset_filename"
    ):
        # Auto-download from HuggingFace
        logger.info(
            f"Downloading dataset from {task_config.dataset_repo}/{task_config.dataset_filename}"
        )
        dataset_path = hf_hub_download(
            repo_id=task_config.dataset_repo,
            filename=task_config.dataset_filename,
            repo_type="dataset",
        )
        logger.info(f"Downloaded dataset to: {dataset_path}")
    elif hasattr(task_config, "dataset_path"):
        # Use explicit path if provided
        dataset_path = os.path.expanduser(task_config.dataset_path)
    else:
        raise ValueError(
            "Either dataset_repo/dataset_filename or dataset_path must be provided"
        )

    if task_config.env_name in ["can", "lift", "square", "tool_hang", "transport"]:
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
        elif task_config.obs_type == "image":
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
        else:
            raise ValueError(f"Invalid observation type: {task_config.obs_type}")
    else:
        raise ValueError(f"Environment {task_config.env_name} not supported")


class RobomimicDataset(BaseDataset):
    def __init__(
        self,
        dataset_dir,
        horizon=1,
        pad_before=0,
        pad_after=0,
        obs_keys=("object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"),
        abs_action=False,
        rotation_rep="rotation_6d",
        val_dataset_percentage=0.0,
        mode="train",
        use_key_state_for_val: bool = False,
    ):
        super().__init__()
        self.rotation_transformer = RotationTransformer(
            from_rep="axis_angle", to_rep=rotation_rep
        )
        self.val_dataset_percentage = val_dataset_percentage
        self.mode = mode

        self.replay_buffer = ReplayBuffer.create_empty_numpy()
        with h5py.File(dataset_dir) as file:
            demos = file["data"]
            total_demos = len(demos)

            # Calculate split indices
            if val_dataset_percentage > 0.0:
                val_count = int(total_demos * val_dataset_percentage)
                train_count = total_demos - val_count

                # Use deterministic split based on indices
                if mode == "train":
                    demo_indices = list(range(train_count))
                elif mode == "val":
                    demo_indices = list(range(train_count, total_demos))
                else:
                    raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'val'")
            else:
                # Use all data for training when no validation split
                demo_indices = list(range(total_demos))

            if use_key_state_for_val:
                import robomimic.utils.env_utils as EnvUtils
                import robomimic.utils.file_utils as FileUtils
                import robomimic.utils.obs_utils as ObsUtils

                # Initialize observation utilities with dummy spec
                dummy_spec = {
                    "obs": {
                        "low_dim": ["robot0_eef_pos"],
                        "rgb": [],
                    },
                }
                ObsUtils.initialize_obs_utils_with_obs_specs(
                    obs_modality_specs=dummy_spec
                )

                # Create environment from dataset metadata
                env_meta = FileUtils.get_env_metadata_from_dataset(
                    dataset_path=dataset_dir
                )
                env = EnvUtils.create_env_from_metadata(
                    env_meta=env_meta, render=False, render_offscreen=False
                )

                # Check if this is a robosuite environment
                is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

            for i in tqdm(demo_indices, desc=f"Loading {mode} hdf5 to ReplayBuffer"):
                demo = demos[f"demo_{i}"]

                if use_key_state_for_val:
                    states = demo["states"][:]
                    # Prepare initial state for environment reset
                    initial_state = {"states": states[0]}
                    if is_robosuite_env:
                        initial_state["model"] = demo.attrs["model_file"]
                        initial_state["ep_meta"] = demo.attrs.get("ep_meta", None)

                    # Reset environment to initial state
                    env.reset_to(initial_state)

                    # Evaluate key states in the trajectory
                    for _j, state in enumerate(states):
                        env.reset_to({"states": state})

                        # Get distance between frame and stand (example evaluation metric)
                        frame_site_name = "frame_tip_site"
                        stand_site_name = "stand_mount_site"

                        frame_site_pos = env.sim.data.site_xpos[
                            env.obj_site_id[frame_site_name]
                        ]
                        stand_site_pos = env.sim.data.site_xpos[
                            env.obj_site_id[stand_site_name]
                        ]
                        distance = np.linalg.norm(frame_site_pos - stand_site_pos)
                        logger.debug(distance)
                    exit()

                episode = _data_to_obs(
                    raw_obs=demo["obs"],
                    raw_actions=demo["actions"][:].astype(np.float32),
                    obs_keys=obs_keys,
                    abs_action=abs_action,
                    rotation_transformer=self.rotation_transformer,
                )
                self.replay_buffer.add_episode(episode)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
        )

        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.abs_action = abs_action
        self.normalizer = self.get_normalizer()

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1, 2, 10)

        d_rot = action.shape[-1] - 4
        pos = action[..., :3]
        rot = action[..., 3 : 3 + d_rot]
        gripper = action[..., [-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([pos, rot, gripper], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction

    def get_normalizer(self):
        if self.abs_action:
            state_normalizer = MinMaxNormalizer(
                self.replay_buffer["obs"][:]
            )  # (N, obs_dim)
            action_normalizer = MinMaxNormalizer(
                self.replay_buffer["action"][:]
            )  # (N, action_dim)
        else:
            state_normalizer = MinMaxNormalizer(
                self.replay_buffer["obs"][:]
            )  # (N, obs_dim)
            action_normalizer = MinMaxNormalizer(
                self.replay_buffer["action"][:]
            )  # (N, action_dim)
        return {"obs": {"state": state_normalizer}, "action": action_normalizer}

    def sample_to_data(self, sample):
        state = sample["obs"].astype(np.float32)
        state = self.normalizer["obs"]["state"].normalize(state)

        action = sample["action"].astype(np.float32)
        action = self.normalizer["action"].normalize(action)
        data = {
            "obs": {"state": state},
            "action": action,
        }
        return data

    def __str__(self) -> str:
        return f"Keys: {self.replay_buffer.keys()} Steps: {self.replay_buffer.n_steps} Episodes: {self.replay_buffer.n_episodes}"

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self.sample_to_data(sample)
        torch_data = dict_apply(data, torch.tensor)
        return torch_data

def _data_to_obs(raw_obs, raw_actions, obs_keys, abs_action, rotation_transformer):
    obs = np.concatenate([raw_obs[key] for key in obs_keys], axis=-1).astype(np.float32)

    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            # dual arm
            raw_actions = raw_actions.reshape(-1, 2, 7)
            is_dual_arm = True

        pos = raw_actions[..., :3]
        rot = raw_actions[..., 3:6]
        gripper = raw_actions[..., 6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)

        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1, 20)

    data = {"obs": obs, "action": raw_actions}
    return data


class RobomimicImageDataset(BaseDataset):
    def __init__(
        self,
        dataset_dir,
        shape_meta: dict,
        n_obs_steps=None,
        horizon=1,
        pad_before=0,
        pad_after=0,
        abs_action=False,
        rotation_rep="rotation_6d",
        val_dataset_percentage=0.0,
        mode="train",
        future_state_enabled=False,
        future_state_steps=1,
        future_state_steps_list=None,
    ):
        super().__init__()
        self.rotation_transformer = RotationTransformer(
            from_rep="axis_angle", to_rep=rotation_rep
        )
        self.val_dataset_percentage = val_dataset_percentage
        self.mode = mode

        self.replay_buffer = _convert_robomimic_to_replay(
            store=zarr.storage.MemoryStore(),
            shape_meta=shape_meta,
            dataset_path=dataset_dir,
            abs_action=abs_action,
            rotation_transformer=self.rotation_transformer,
            val_dataset_percentage=val_dataset_percentage,
            mode=mode,
        )

        rgb_keys = []
        lowdim_keys = []
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            if type == "rgb":
                rgb_keys.append(key)
            elif type == "low_dim":
                lowdim_keys.append(key)

        future_state_steps_list = list(future_state_steps_list or [])
        future_steps = (
            future_state_steps_list if len(future_state_steps_list) > 0 else [future_state_steps]
        )

        key_first_k = {}
        if n_obs_steps is not None:
            # only take first k obs from images
            obs_key_keep_steps = n_obs_steps
            if future_state_enabled:
                obs_key_keep_steps = max(
                    obs_key_keep_steps,
                    n_obs_steps + max(future_steps),
                )
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = obs_key_keep_steps
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            key_first_k=key_first_k,
        )

        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.abs_action = abs_action
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        self.future_state_enabled = future_state_enabled
        self.future_state_steps = future_state_steps
        self.future_state_steps_list = future_state_steps_list
        self.future_steps = future_steps

        self.normalizer = self.get_normalizer()

    def get_normalizer(self):
        normalizer = defaultdict(dict)
        for key in self.lowdim_keys:
            normalizer["obs"][key] = MinMaxNormalizer(self.replay_buffer[key][:])
        for key in self.rgb_keys:
            normalizer["obs"][key] = ImageNormalizer()
        normalizer["action"] = MinMaxNormalizer(self.replay_buffer["action"][:])

        return normalizer

    def __str__(self) -> str:
        return f"Keys: {self.replay_buffer.keys()} Steps: {self.replay_buffer.n_steps} Episodes: {self.replay_buffer.n_episodes}"

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)

        T_slice = slice(self.n_obs_steps)

        obs_dict = {}
        future_obs_dict = None
        future_frame_indices = None
        if self.future_state_enabled:
            future_obs_dict = {}
            # The last observed frame is at n_obs_steps - 1. Predict one or more
            # future frames that are `future_steps` transitions ahead of it.
            future_frame_indices = [
                min(self.n_obs_steps - 1 + future_step, sample["action"].shape[0] - 1)
                for future_step in self.future_steps
            ]

        for key in self.rgb_keys:
            # current observation sequence
            obs_dict[key] = (
                np.moveaxis(sample[key][T_slice], -1, 1).astype(np.float32) / 255.0
            )
            obs_dict[key] = self.normalizer["obs"][key].normalize(obs_dict[key])

            if self.future_state_enabled:
                future_frames = [
                    np.moveaxis(sample[key][future_frame_idx], -1, 0).astype(np.float32)
                    / 255.0
                    for future_frame_idx in future_frame_indices
                ]
                if len(future_frames) == 1:
                    future_obs_dict[key] = self.normalizer["obs"][key].normalize(
                        future_frames[0]
                    )
                else:
                    stacked = np.stack(future_frames, axis=0)
                    future_obs_dict[key] = self.normalizer["obs"][key].normalize(
                        stacked
                    )

            del sample[key]

        for key in self.lowdim_keys:
            obs_dict[key] = sample[key][T_slice].astype(np.float32)
            obs_dict[key] = self.normalizer["obs"][key].normalize(obs_dict[key])
            if self.future_state_enabled:
                future_frames = [
                    sample[key][future_frame_idx].astype(np.float32)
                    for future_frame_idx in future_frame_indices
                ]
                if len(future_frames) == 1:
                    future_obs_dict[key] = self.normalizer["obs"][key].normalize(
                        future_frames[0]
                    )
                else:
                    stacked = np.stack(future_frames, axis=0)
                    future_obs_dict[key] = self.normalizer["obs"][key].normalize(
                        stacked
                    )
            del sample[key]

        action = sample["action"].astype(np.float32)
        action = self.normalizer["action"].normalize(action)

        torch_data = {
            "obs": dict_apply(obs_dict, torch.tensor),
            "action": torch.tensor(action),
        }
        if future_obs_dict is not None:
            torch_data["future_obs"] = dict_apply(future_obs_dict, torch.tensor)
        return torch_data


    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1, 2, 10)

        d_rot = action.shape[-1] - 4
        pos = action[..., :3]
        rot = action[..., 3 : 3 + d_rot]
        gripper = action[..., [-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([pos, rot, gripper], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction


def _convert_actions(raw_actions, abs_action, rotation_transformer):
    actions = raw_actions
    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            # dual arm
            raw_actions = raw_actions.reshape(-1, 2, 7)
            is_dual_arm = True

        pos = raw_actions[..., :3]
        rot = raw_actions[..., 3:6]
        gripper = raw_actions[..., 6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)

        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1, 20)
        actions = raw_actions
    return actions


def _convert_robomimic_to_replay(
    store,
    shape_meta,
    dataset_path,
    abs_action,
    rotation_transformer,
    n_workers=None,
    max_inflight_tasks=None,
    val_dataset_percentage=0.0,
    mode="train",
):
    """Convert Robomimic dataset to ReplayBuffer.

    A ReplayBuffer is a `zarr.Group` or Dict[str, dict] that contains the following keys:
    - data: zarr.Group or Dict[str, dict]
        Contains the data. All data should be stored as numpy arrays with the same length.
    - meta: zarr.Group or Dict[str, dict]
        Contains key "episode_ends", which is a numpy array of shape (n_episodes,) that contains the
        end index of each episode in the data.

    Args:
    - store: zarr.Store
        zarr.MemoryStore()
    - shape_meta: dict
        Shape metadata of the dataset. Should contain keys 'obs', 'action'.
        For example:
        shape_meta = {
            "action": {"shape": [10, ]},
            "obs": {
                "agentview_image": {"shape": [84, 84, 3], "type": "rgb"},
                "robot0_eef_pos":  {"shape": [3, ],       "type": "low_dim"},
            }}
    - dataset_path: str
        Path to the Robomimic dataset
    - abs_action: bool
        Whether to use position or velocity control
    - rotation_transformer: RotationTransformer
        Rotation transformer to convert rotation representation
    """
    """ Dataset structure of Can-PH, as an example:
    - data
        - demo_0
            - actions  (118, 7)
            - dones     (118, )
            - next_obs
                - agentview_image  (118, 84, 84, 3)
                - object            (118, 14)
                - robot0_eef_pos   (118, 3)
                - robot0_eef_quat
                - robot0_eef_vel_ang
                - robot0_eef_vel_lin
                - robot0_eye_in_hand_image
                - robot0_gripper_qpos
                - robot0_gripper_qvel
                - robot0_joint_pos
                - robot0_joint_pos_cos
                - robot0_joint_pos_sin
                - robot0_joint_vel
            - obs
                ...
            - rewards   (118, )
            - states    (118, 71)
        - demo_1
        ...(x200 demos)
    - mask
        - 20_percent
        - 20_percent_train
        - 20_percent_valid
        - 50_percent
        - 50_percent_train
        - 50_percent_valid
        - train (180,)
        - valid (20,)

    Suppose that the `shape_meta` is:
    shape_meta = {
    "action": {"shape": [10, ]},
    "obs": {
        "agentview_image": {
            "shape": [3, 84, 84], "type": "rgb", },
        "robot0_eye_in_hand_image": {
            "shape": [3, 84, 84], "type": "rgb", },
        "robot0_eef_pos": {
            "shape": [3, ], "type": "low_dim", },
        "robot0_eef_quat": {
            "shape": [4, ], "type": "low_dim", },
        "robot0_gripper_qpos": {
            "shape": [2, ], "type": "low_dim", }, }}
    """

    import multiprocessing

    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = []
    lowdim_keys = []
    # construct compressors and chunks
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        shape = attr["shape"]
        type = attr.get("type", "low_dim")
        if type == "rgb":
            rgb_keys.append(key)
        elif type == "low_dim":
            lowdim_keys.append(key)
    # rgb_keys = ['agentview_image', 'robot0_eye_in_hand_image']
    # lowdim_keys = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']

    # create zarr group
    root = zarr.group(store)
    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)

    with h5py.File(dataset_path) as file:
        # count total steps
        demos = file["data"]
        total_demos = len(demos)

        # Calculate split indices
        if val_dataset_percentage > 0.0:
            val_count = int(total_demos * val_dataset_percentage)
            train_count = total_demos - val_count

            # Use deterministic split based on indices
            if mode == "train":
                demo_indices = list(range(train_count))
            elif mode == "val":
                demo_indices = list(range(train_count, total_demos))
            else:
                raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'val'")
        else:
            # Use all data for training when no validation split
            demo_indices = list(range(total_demos))

        episode_ends = []
        prev_end = 0
        for i in demo_indices:
            demo = demos[f"demo_{i}"]
            episode_length = demo["actions"].shape[0]
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)
        n_steps = episode_ends[-1] if episode_ends else 0
        episode_starts = [0] + episode_ends[:-1]
        _ = meta_group.create_array(
            name="episode_ends",
            data=np.array(episode_ends, dtype=np.int64),
            compressor=None,
            overwrite=True,
        )

        # save lowdim data
        for key in tqdm(lowdim_keys + ["action"], desc=f"Loading {mode} lowdim data"):
            data_key = "obs/" + key
            if key == "action":
                data_key = "actions"
            this_data = []
            for i in demo_indices:
                demo = demos[f"demo_{i}"]
                this_data.append(demo[data_key][:].astype(np.float32))
            this_data = np.concatenate(this_data, axis=0) if this_data else np.array([])
            if key == "action":
                this_data = _convert_actions(
                    raw_actions=this_data,
                    abs_action=abs_action,
                    rotation_transformer=rotation_transformer,
                )
                assert this_data.shape == (n_steps,) + tuple(
                    shape_meta["action"]["shape"]
                )
            else:
                assert this_data.shape == (n_steps,) + tuple(
                    shape_meta["obs"][key]["shape"]
                )
            _ = data_group.create_array(
                name=key,
                data=this_data,
                chunks=this_data.shape,
                compressor=None,
                overwrite=True,
            )

        def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
            try:
                zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
                # make sure we can successfully decode
                _ = zarr_arr[zarr_idx]
                return True
            except Exception:
                return False

        with tqdm(
            total=n_steps * len(rgb_keys),
            desc=f"Loading {mode} image data",
            mininterval=1.0,
        ) as pbar:
            # one chunk per thread, therefore no synchronization needed
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=n_workers
            ) as executor:
                futures = set()
                for key in rgb_keys:
                    data_key = "obs/" + key
                    shape = tuple(shape_meta["obs"][key]["shape"])
                    c, h, w = shape
                    # Use None compressor for zarr v3 compatibility in tests
                    img_arr = data_group.require_dataset(
                        name=key,
                        shape=(n_steps, h, w, c),
                        chunks=(1, h, w, c),
                        compressor=None,
                        dtype=np.uint8,
                    )
                    for demo_list_idx, episode_idx in enumerate(demo_indices):
                        demo = demos[f"demo_{episode_idx}"]
                        hdf5_arr = demo["obs"][key]
                        for hdf5_idx in range(hdf5_arr.shape[0]):
                            if len(futures) >= max_inflight_tasks:
                                # limit number of inflight tasks
                                completed, futures = concurrent.futures.wait(
                                    futures,
                                    return_when=concurrent.futures.FIRST_COMPLETED,
                                )
                                for f in completed:
                                    if not f.result():
                                        raise RuntimeError("Failed to encode image!")
                                pbar.update(len(completed))

                            zarr_idx = episode_starts[demo_list_idx] + hdf5_idx
                            futures.add(
                                executor.submit(
                                    img_copy, img_arr, zarr_idx, hdf5_arr, hdf5_idx
                                )
                            )
                completed, futures = concurrent.futures.wait(futures)
                for f in completed:
                    if not f.result():
                        raise RuntimeError("Failed to encode image!")
                pbar.update(len(completed))

    replay_buffer = ReplayBuffer(root)
    return replay_buffer
