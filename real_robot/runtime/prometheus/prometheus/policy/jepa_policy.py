from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from prometheus.policy.scheduler import ActionChunk


DEFAULT_CAMERA_STREAMS = {
    "base_image": "base_0_color",
    "left_wrist_image": "left_wrist_0_color",
    "right_wrist_image": "right_wrist_0_color",
}


class JEPAPolicy:
    """Prometheus adapter for the ARX-R5 JEPA-Policy checkpoint format."""

    policy_name = "jepa_policy"

    def __init__(
        self,
        *,
        inputs: Mapping[str, Any] | None = None,
        inference: Mapping[str, Any] | None = None,
        action_hz: float = 10.0,
        runtime: Any | None = None,
    ) -> None:
        self.inputs = dict(inputs or {})
        self.inference = dict(inference or {})
        self.camera_streams = dict(
            self.inputs.get("camera_streams", DEFAULT_CAMERA_STREAMS)
        )
        if set(self.camera_streams) != set(DEFAULT_CAMERA_STREAMS):
            raise ValueError(
                "camera_streams must map base_image, left_wrist_image, and "
                "right_wrist_image"
            )
        self.state_stream = str(self.inputs.get("state_stream", "robot_state"))
        self.anchor = str(
            self.inputs.get("anchor", self.camera_streams["base_image"])
        )
        self.window_size = int(self.inputs.get("window_size", 2))
        self.stride = int(self.inputs.get("stride", 3))
        self.startup_camera_frames = int(
            self.inputs.get("startup_camera_frames", 1)
        )
        self.slop_ms = float(self.inputs.get("slop_ms", 60.0))
        self.wait_latest = bool(self.inputs.get("wait_latest", True))
        self.wait_latest_timeout_ms = float(
            self.inputs.get("wait_latest_timeout_ms", 5.0)
        )
        self.observation_barrier_timeout_ms = float(
            self.inputs.get("observation_barrier_timeout_ms", 300.0)
        )
        self._minimum_observation_stamp_ns: int | None = None
        self._last_observation_barrier_stamp_ns: int | None = None
        self.action_hz = float(action_hz)
        if (
            self.window_size <= 0
            or self.stride <= 0
            or self.startup_camera_frames <= 0
        ):
            raise ValueError(
                "window_size, stride, and startup_camera_frames must be positive"
            )
        if self.action_hz <= 0:
            raise ValueError("action_hz must be positive")
        if self.observation_barrier_timeout_ms <= 0:
            raise ValueError("observation_barrier_timeout_ms must be positive")
        self.runtime = runtime if runtime is not None else JEPAInferenceRuntime(
            **self.inference
        )

    @property
    def stream_names(self) -> tuple[str, ...]:
        return (*self.camera_streams.values(), self.state_stream)

    def infer(self, data: Any) -> ActionChunk:
        minimum_observation_stamp_ns = self._minimum_observation_stamp_ns
        frames = data.window_numpy(
            anchor=self.anchor,
            names=self.stream_names,
            count=self.window_size,
            stride=self.stride,
            slop_ms=self.slop_ms,
            wait_latest=self.wait_latest,
            timeout_ms=(
                self.observation_barrier_timeout_ms
                if minimum_observation_stamp_ns is not None
                else self.wait_latest_timeout_ms
            ),
            min_anchor_stamp_ns=minimum_observation_stamp_ns,
        )
        if (
            minimum_observation_stamp_ns is not None
            and int(frames[-1].stamp_ns) < minimum_observation_stamp_ns
        ):
            raise RuntimeError(
                "policy observation freshness barrier failed: "
                f"frame={int(frames[-1].stamp_ns)} "
                f"required={minimum_observation_stamp_ns}"
            )
        self._minimum_observation_stamp_ns = None
        self._last_observation_barrier_stamp_ns = minimum_observation_stamp_ns
        obs = self._observation_from_frames(frames)
        actions = np.asarray(self.runtime.predict(obs), dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 14:
            raise ValueError(
                f"JEPA runtime must return (horizon, 14), got {actions.shape}"
            )
        current_state = frames[-1].samples[self.state_stream].data
        current_action = np.asarray(
            current_state["position"], dtype=np.float32
        ).reshape(-1)
        if current_action.shape != (14,):
            raise ValueError(
                f"robot state position must have shape (14,), got {current_action.shape}"
            )
        current_velocity = np.asarray(
            current_state["velocity"], dtype=np.float32
        ).reshape(-1)
        if current_velocity.shape != (14,):
            raise ValueError(
                f"robot state velocity must have shape (14,), got {current_velocity.shape}"
            )
        if not np.all(np.isfinite(current_velocity)):
            raise ValueError("robot state velocity contains NaN or inf")
        if os.environ.get("PROMETHEUS_POLICY_DIAGNOSTICS", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            previous = np.vstack([current_action, actions[:-1]])
            diagnostics = {
                "model": self.policy_name,
                "camera_order": list(self.camera_streams),
                "stream_order": list(self.stream_names),
                "frame_stamp_ns": [int(frame.stamp_ns) for frame in frames],
                "state_order": "left_arm_6,left_gripper,right_arm_6,right_gripper",
                "current_action": current_action.tolist(),
                "current_velocity": current_velocity.tolist(),
                "output_shape": list(actions.shape),
                "output_actions": actions.tolist(),
                "max_abs_delta_from_previous": float(
                    np.max(np.abs(actions - previous))
                ),
                "action_space": "abs_qpos",
                "action_hz": self.action_hz,
            }
            print(
                "[PolicyDiagnostics] "
                + json.dumps(diagnostics, separators=(",", ":")),
                flush=True,
            )
        return ActionChunk(
            actions=actions,
            action_space="abs_qpos",
            hz=self.action_hz,
            metadata={
                "current_action": current_action,
                "current_velocity": current_velocity,
                "model": self.policy_name,
                "window_size": self.window_size,
                "stride": self.stride,
                "frame_stamp_ns": int(frames[-1].stamp_ns),
                "observation_barrier_stamp_ns": minimum_observation_stamp_ns,
            },
        )

    def require_observation_after(self, stamp_ns: int) -> None:
        stamp = int(stamp_ns)
        if stamp <= 0:
            raise ValueError("observation freshness stamp must be positive")
        self._minimum_observation_stamp_ns = stamp

    def reset_episode(self) -> None:
        """Drop temporal barriers owned by the preceding rollout episode."""
        self._minimum_observation_stamp_ns = None
        self._last_observation_barrier_stamp_ns = None

    def _observation_from_frames(
        self, frames: list[Any]
    ) -> dict[str, np.ndarray]:
        if len(frames) != self.window_size:
            raise ValueError(
                f"expected {self.window_size} aligned frames, got {len(frames)}"
            )
        obs: dict[str, np.ndarray] = {}
        for model_key, stream_name in self.camera_streams.items():
            images = []
            for frame in frames:
                image = np.asarray(frame.samples[stream_name].data)
                if image.ndim != 3 or image.shape[2] != 3:
                    raise ValueError(
                        f"{stream_name} must be HWC RGB, got {image.shape}"
                    )
                images.append(np.moveaxis(image, -1, 0))
            obs[model_key] = np.stack(images).astype(np.uint8, copy=False)
        states = [
            np.asarray(frame.samples[self.state_stream].data["position"], dtype=np.float32)
            for frame in frames
        ]
        obs["state"] = np.stack(states)
        if obs["state"].shape != (self.window_size, 14):
            raise ValueError(
                "robot state history must have shape "
                f"({self.window_size}, 14), got {obs['state'].shape}"
            )
        return obs

    def close(self) -> None:
        close = getattr(self.runtime, "close", None)
        if callable(close):
            close()

    def status(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "stream_names": list(self.stream_names),
            "anchor": self.anchor,
            "window_size": self.window_size,
            "stride": self.stride,
            "startup_camera_frames": self.startup_camera_frames,
            "observation_barrier_timeout_ms": self.observation_barrier_timeout_ms,
            "pending_observation_barrier_stamp_ns": self._minimum_observation_stamp_ns,
            "last_observation_barrier_stamp_ns": self._last_observation_barrier_stamp_ns,
            "action_hz": self.action_hz,
            "runtime": getattr(self.runtime, "status", lambda: {})(),
        }


class JEPAInferenceRuntime:
    """Loads a JEPA-Policy resolved config, checkpoint, and dataset statistics."""

    def __init__(
        self,
        *,
        project_root: str,
        run_dir: str | None = None,
        config_path: str | None = None,
        config_name: str | None = None,
        task: str | None = None,
        checkpoint: str = "models/model_latest.pt",
        stats_path: str | None = None,
        device: str = "cuda",
        num_steps: int = 2,
    ) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        if not self.project_root.is_dir():
            raise FileNotFoundError(f"JEPA-Policy project not found: {self.project_root}")
        self.run_dir = (
            None if run_dir is None else Path(run_dir).expanduser().resolve()
        )
        if config_path is None and self.run_dir is not None:
            resolved_config_path = self.run_dir / "resolved_config.yaml"
        elif config_path is not None:
            resolved_config_path = Path(config_path).expanduser().resolve()
        else:
            resolved_config_path = None
        if resolved_config_path is not None and not resolved_config_path.is_file():
            raise FileNotFoundError(
                f"resolved config not found: {resolved_config_path}"
            )
        if resolved_config_path is None and (config_name is None or task is None):
            raise ValueError(
                "provide run_dir/config_path, or both config_name and task"
            )
        checkpoint_path = Path(checkpoint).expanduser()
        if not checkpoint_path.is_absolute():
            if self.run_dir is None:
                raise ValueError(
                    "relative checkpoint requires inference.run_dir; use an "
                    "absolute checkpoint with config_path"
                )
            checkpoint_path = self.run_dir / checkpoint_path
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        root_str = str(self.project_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        import torch
        from omegaconf import OmegaConf
        from mip.agent import TrainingAgent

        self.torch = torch
        self.device = str(device)
        self.num_steps = int(num_steps)
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        self.config_path = resolved_config_path
        if resolved_config_path is None:
            self.config = _compose_jepa_config(
                OmegaConf,
                project_root=self.project_root,
                config_name=str(config_name),
                task=str(task),
            )
        else:
            self.config = OmegaConf.load(resolved_config_path)
        self.config.optimization.device = self.device
        self.config.task.obs_dim = self.config.network.emb_dim
        self.image_shapes: dict[str, tuple[int, int, int]] = {}
        for key in DEFAULT_CAMERA_STREAMS:
            shape = tuple(
                int(value) for value in self.config.task.shape_meta.obs[key].shape
            )
            if len(shape) != 3 or shape[0] != 3:
                raise ValueError(
                    f"JEPA/MIP RGB shape for {key} must be (3, H, W), got {shape}"
                )
            self.image_shapes[key] = shape
        self.agent = TrainingAgent(self.config)
        self.agent.load(str(checkpoint_path), load_optimizer=False)
        self.agent.eval()
        self.checkpoint_path = checkpoint_path
        self._last_state_clip_dimensions: tuple[int, ...] = ()

        if stats_path is None:
            stats_file = Path(self.config.task.dataset_path) / "stats.json"
        else:
            stats_file = Path(stats_path).expanduser()
        if not stats_file.is_file():
            raise FileNotFoundError(f"normalization stats not found: {stats_file}")
        stats = json.loads(stats_file.read_text(encoding="utf-8"))
        self.state_min, self.state_range = _stats_range(stats, "state")
        self.action_min, self.action_range = _stats_range(stats, "action")

    def predict(self, obs: Mapping[str, np.ndarray]) -> np.ndarray:
        torch = self.torch
        model_obs: dict[str, Any] = {}
        for key in DEFAULT_CAMERA_STREAMS:
            value = _resize_rgb_chw(
                np.asarray(obs[key]),
                target_shape=self.image_shapes[key],
                key=key,
            ).astype(np.float32) / 255.0
            model_obs[key] = torch.as_tensor(
                value * 2.0 - 1.0, device=self.device, dtype=torch.float32
            ).unsqueeze(0)
        state, clipped_dimensions = _normalize_state_with_clip(
            np.asarray(obs["state"], dtype=np.float32),
            self.state_min,
            self.state_range,
        )
        if clipped_dimensions != self._last_state_clip_dimensions:
            if clipped_dimensions:
                print(
                    "[JEPAInferenceRuntime] clipped normalized state to the "
                    "training range [-1, 1]; dimensions="
                    f"{list(clipped_dimensions)}",
                    flush=True,
                )
            self._last_state_clip_dimensions = clipped_dimensions
        model_obs["state"] = torch.as_tensor(
            state, device=self.device, dtype=torch.float32
        ).unsqueeze(0)
        act_0 = torch.zeros(
            (1, int(self.config.task.horizon), int(self.config.task.act_dim)),
            device=self.device,
        )
        output = self.agent.sample(
            act_0=act_0,
            obs=model_obs,
            num_steps=self.num_steps,
            use_ema=True,
            return_future=bool(
                getattr(self.config.optimization, "future_joint_mode", False)
            ),
        )
        if isinstance(output, tuple):
            output = output[0]
        action_norm = output.detach().cpu().numpy()[0]
        action = _unnormalize(action_norm, self.action_min, self.action_range)
        start = int(self.config.task.obs_steps) - 1
        end = start + int(self.config.task.act_steps)
        return np.asarray(action[start:end], dtype=np.float32)

    def status(self) -> dict[str, Any]:
        return {
            "project_root": str(self.project_root),
            "run_dir": "" if self.run_dir is None else str(self.run_dir),
            "config_path": "" if self.config_path is None else str(self.config_path),
            "checkpoint": str(self.checkpoint_path),
            "device": self.device,
            "num_steps": self.num_steps,
            "image_shapes": {
                key: list(shape) for key, shape in self.image_shapes.items()
            },
            "image_resize": "direct_cv2_inter_area_to_checkpoint_shape",
            "state_normalization_clip": [-1.0, 1.0],
        }


def _stats_range(stats: Mapping[str, Any], key: str) -> tuple[np.ndarray, np.ndarray]:
    minimum = np.asarray(stats[key]["min"], dtype=np.float32)
    maximum = np.asarray(stats[key]["max"], dtype=np.float32)
    value_range = maximum - minimum
    value_range[value_range == 0] = 1.0
    return minimum, value_range


def _compose_jepa_config(
    omega_conf: Any,
    *,
    project_root: Path,
    config_name: str,
    task: str,
) -> Any:
    """Compose JEPA Hydra config in a child process to isolate global Hydra state."""
    config_dir = project_root / "examples" / "configs"
    script = """
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import sys
with initialize_config_dir(version_base=None, config_dir=sys.argv[1]):
    cfg = compose(config_name=sys.argv[2], overrides=[f"task={sys.argv[3]}"])
print(OmegaConf.to_yaml(cfg, resolve=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(config_dir), config_name, task],
        check=True,
        capture_output=True,
        text=True,
    )
    return omega_conf.create(result.stdout)


def _normalize(value: np.ndarray, minimum: np.ndarray, value_range: np.ndarray) -> np.ndarray:
    return ((value - minimum) / value_range) * 2.0 - 1.0


def _normalize_state_with_clip(
    value: np.ndarray,
    minimum: np.ndarray,
    value_range: np.ndarray,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Keep live state inputs inside the support seen during training."""
    normalized = _normalize(value, minimum, value_range)
    outside = (normalized < -1.0 - 1e-6) | (normalized > 1.0 + 1e-6)
    if normalized.ndim == 1:
        dimensions = tuple(int(index) for index in np.flatnonzero(outside))
    else:
        dimensions = tuple(
            int(index)
            for index in np.flatnonzero(np.any(outside, axis=tuple(range(normalized.ndim - 1))))
        )
    return np.clip(normalized, -1.0, 1.0), dimensions


def _unnormalize(value: np.ndarray, minimum: np.ndarray, value_range: np.ndarray) -> np.ndarray:
    return ((value + 1.0) / 2.0) * value_range + minimum


def _resize_rgb_chw(
    value: np.ndarray,
    *,
    target_shape: tuple[int, int, int],
    key: str,
) -> np.ndarray:
    """Match live RGB observations to the checkpoint's training cache shape."""
    array = np.asarray(value)
    channels, target_height, target_width = target_shape
    if array.ndim != 4 or array.shape[1] != channels:
        raise ValueError(
            f"{key} must have shape (T, {channels}, H, W), got {array.shape}"
        )
    if array.shape[2:] == (target_height, target_width):
        return array

    import cv2

    resized = np.empty(
        (array.shape[0], channels, target_height, target_width),
        dtype=array.dtype,
    )
    for index, frame in enumerate(array):
        hwc = np.moveaxis(frame, 0, -1)
        output = cv2.resize(
            hwc,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )
        resized[index] = np.moveaxis(output, -1, 0)
    return resized


class MIPPolicy(JEPAPolicy):
    """Explicit deployment name for baseline and future-supervised MIP runs."""

    policy_name = "mip"
