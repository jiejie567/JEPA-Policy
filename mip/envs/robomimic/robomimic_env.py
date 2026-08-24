"""This file contains the functions to create the environment."""

import collections
import io
import os
import sys

import gymnasium as gym
from loguru import logger

from mip.config import TaskConfig
from mip.envs.egl_device import override_robomimic_egl_probe
from mip.env_utils import MultiStepWrapper, VideoRecorder, VideoRecordingWrapper
from mip.mimicgen_utils import is_mimicgen_task


_ROBOMIMIC_TASKS = {"can", "lift", "square", "tool_hang", "transport"}
_CONTROLLER_INPUT_TYPES = {"absolute", "delta"}


def _is_supported_task(task_config: TaskConfig) -> bool:
    return task_config.env_name in _ROBOMIMIC_TASKS or is_mimicgen_task(task_config)


def _apply_controller_input_type_override(env_meta: dict, input_type: str | None) -> None:
    """Apply an explicitly requested Robosuite arm-controller input mode.

    This is intentionally opt-in. Existing tasks that do not set
    ``task.robosuite_controller_input_type_override`` retain the historical
    controller setup, so state-absolute experiments cannot change concurrent
    image or delta-action training behavior.
    """
    if input_type is None:
        return
    if input_type not in _CONTROLLER_INPUT_TYPES:
        raise ValueError(
            "robosuite_controller_input_type_override must be one of "
            f"{sorted(_CONTROLLER_INPUT_TYPES)}, got {input_type!r}"
        )

    controller_config = env_meta["env_kwargs"]["controller_configs"]
    body_parts = controller_config.get("body_parts")
    if body_parts is not None:
        updated_parts = []
        for part_name, part_config in body_parts.items():
            if not isinstance(part_config, dict):
                continue
            # Robosuite 1.5 controller metadata stores the arm configuration
            # under body_parts. ``control_delta`` is the legacy field retained
            # in converted Robomimic datasets; the runtime consumes input_type.
            if "control_delta" in part_config or "input_type" in part_config:
                part_config["input_type"] = input_type
                updated_parts.append(part_name)
        if not updated_parts:
            raise ValueError(
                "controller input override requested, but no Robosuite arm "
                "controller was found under controller_configs.body_parts"
            )
        return

    # Robosuite <=1.4 used a flat controller configuration. Set both spellings
    # so an explicit override remains valid across the supported versions.
    controller_config["input_type"] = input_type
    controller_config["control_delta"] = input_type == "delta"


def _assert_controller_input_type(env, expected: str | None) -> None:
    """Verify the realized Robosuite arm controllers after construction."""
    if expected is None:
        return

    robosuite_env = env
    robots = None
    for _ in range(8):
        robots = getattr(robosuite_env, "robots", None)
        if robots:
            break
        robosuite_env = getattr(robosuite_env, "env", None)
        if robosuite_env is None:
            break
    if not robots:
        raise RuntimeError(
            "controller input override requested, but the constructed environment "
            "does not expose Robosuite robots"
        )

    realized = []
    for robot_index, robot in enumerate(robots):
        composite = getattr(robot, "composite_controller", None)
        part_controllers = getattr(composite, "part_controllers", {})
        for part_name, controller in part_controllers.items():
            input_type = getattr(controller, "input_type", None)
            if input_type is not None:
                realized.append((robot_index, part_name, input_type))

    if not realized:
        raise RuntimeError(
            "controller input override requested, but no realized arm controller "
            "exposes input_type"
        )
    mismatched = [item for item in realized if item[2] != expected]
    if mismatched:
        raise RuntimeError(
            f"Robosuite controller input type mismatch: expected={expected!r}, "
            f"realized={realized!r}"
        )
    logger.info(
        "Verified Robosuite controller input override: expected={} realized={}",
        expected,
        realized,
    )


def make_env(task_config: TaskConfig, idx, render=False, seed=None):
    if _is_supported_task(task_config):
        return make_robomimic_env(task_config, idx, render, seed=seed)
    else:
        raise ValueError(f"Environment {task_config.env_name} not supported")


def make_vec_env(task_config: TaskConfig, seed=None):
    # Suppress output by redirecting stdout temporarily
    original_stdout = sys.stdout
    sys.stdout = io.StringIO()  # Redirect stdout to a string buffer
    # Use SyncVectorEnv for image-based tasks (rendering contexts can't be pickled)
    # or when num_envs=1 or save_video=True
    if (
        task_config.num_envs == 1
        or task_config.save_video
        or task_config.obs_type == "image"
    ):
        vnc_env_class = gym.vector.SyncVectorEnv
    else:
        vnc_env_class = gym.vector.AsyncVectorEnv
    if _is_supported_task(task_config):
        try:
            envs = vnc_env_class(
                [
                    make_robomimic_env(task_config, idx, False, seed=seed)
                    for idx in range(task_config.num_envs)
                ],
            )
        finally:
            sys.stdout = original_stdout  # Restore stdout
        return envs
    else:
        raise ValueError(f"Environment {task_config.env_name} not supported")


def make_robomimic_env(task_config: TaskConfig, idx, render=False, seed=None):
    from mip.envs.robomimic.robomimic_image_wrapper import (
        RobomimicImageWrapper,
    )
    from mip.envs.robomimic.robomimic_lowdim_wrapper import (
        RobomimicLowdimWrapper,
    )

    def thunk():
        if is_mimicgen_task(task_config):
            # Importing MimicGen registers its custom robosuite environments
            # before robomimic reconstructs the environment from HDF5 metadata.
            import mimicgen  # noqa: F401

        import robomimic.utils.env_utils as EnvUtils
        import robomimic.utils.file_utils as FileUtils
        import robomimic.utils.obs_utils as ObsUtils

        def create_robomimic_env(
            env_meta, obs_keys=None, shape_meta=None, enable_render=True
        ):
            if task_config.obs_type == "state":
                ObsUtils.initialize_obs_modality_mapping_from_dict(
                    {"low_dim": obs_keys}
                )
            else:  # image observation
                modality_mapping = collections.defaultdict(list)
                for key, attr in shape_meta["obs"].items():
                    modality_mapping[attr.get("type", "low_dim")].append(key)
                ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

            with override_robomimic_egl_probe():
                env = EnvUtils.create_env_from_metadata(
                    env_meta=env_meta,
                    render=False,
                    render_offscreen=enable_render
                    if task_config.obs_type == "image"
                    else False,
                    use_image_obs=enable_render
                    if task_config.obs_type == "image"
                    else False,
                )
            return env

        # Get dataset path (either from explicit path or HuggingFace download)
        explicit_dataset_path = getattr(task_config, "dataset_path", None)
        dataset_repo = getattr(task_config, "dataset_repo", None)
        dataset_filename = getattr(task_config, "dataset_filename", None)
        if explicit_dataset_path:
            dataset_path = os.path.expanduser(explicit_dataset_path)
        elif dataset_repo and dataset_filename:
            from huggingface_hub import hf_hub_download

            dataset_path = hf_hub_download(
                repo_id=dataset_repo,
                filename=dataset_filename,
                repo_type="dataset",
            )
        else:
            raise ValueError(
                "Either dataset_repo/dataset_filename or dataset_path must be provided"
            )

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        if task_config.obs_type == "image":
            # disable object state observation for image mode
            env_meta["env_kwargs"]["use_object_obs"] = False
        abs_action = task_config.abs_action
        controller_input_type_override = getattr(
            task_config, "robosuite_controller_input_type_override", None
        )
        if controller_input_type_override is not None:
            _apply_controller_input_type_override(
                env_meta, controller_input_type_override
            )
        elif abs_action:
            # Preserve the historical behavior for every existing task. The
            # explicit override above is used only by isolated state-absolute
            # runs that need Robosuite 1.5's nested input_type field.
            env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False

        if task_config.obs_type == "state":
            env = create_robomimic_env(env_meta=env_meta, obs_keys=task_config.obs_keys)
            env = RobomimicLowdimWrapper(
                env=env,
                obs_keys=task_config.obs_keys,
                init_state=None,
                render_hw=(256, 256),
                render_camera_name="agentview",
            )
        else:  # image observation
            env = create_robomimic_env(
                env_meta=env_meta, shape_meta=task_config.shape_meta
            )
            # Robosuite's hard reset causes excessive memory consumption.
            # Disabled to run more envs.
            env.env.hard_reset = False
            env = RobomimicImageWrapper(
                env=env,
                shape_meta=task_config.shape_meta,
                init_state=None,
                render_obs_key=task_config.render_obs_key,
            )

        _assert_controller_input_type(env, controller_input_type_override)

        video_recoder = VideoRecorder.create_h264(
            fps=10,
            codec="h264",
            input_pix_fmt="rgb24",
            crf=22,
            thread_type="FRAME",
            thread_count=1,
        )
        file_path = None if not render else "results/video.mp4"
        env = VideoRecordingWrapper(
            env, video_recoder, file_path=file_path, steps_per_render=2
        )
        env = MultiStepWrapper(
            env,
            n_obs_steps=task_config.obs_steps,
            n_action_steps=task_config.act_steps,
            max_episode_steps=task_config.max_episode_steps,
        )
        if seed is not None:
            env.seed(seed + idx)
            logger.info(f"Env seed: {seed + idx}")
        return env

    return thunk
