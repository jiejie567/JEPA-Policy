from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from prometheus.policy.jepa_policy import JEPAPolicy, _resize_rgb_chw


class DiffusionPolicy(JEPAPolicy):
    """Prometheus adapter for aligned ARX-R5 Diffusion Policy checkpoints."""

    policy_name = "diffusion_policy"

    def __init__(
        self,
        *,
        inputs: Mapping[str, Any] | None = None,
        inference: Mapping[str, Any] | None = None,
        action_hz: float = 10.0,
        runtime: Any | None = None,
    ) -> None:
        inference_dict = dict(inference or {})
        if runtime is None:
            runtime = DiffusionInferenceRuntime(**inference_dict)
        super().__init__(
            inputs=inputs,
            inference=inference_dict,
            action_hz=action_hz,
            runtime=runtime,
        )

    def reset_episode(self) -> None:
        super().reset_episode()
        reset_runtime_episode = getattr(self.runtime, "reset_episode", None)
        if callable(reset_runtime_episode):
            reset_runtime_episode()

    def infer(self, data: Any):
        chunk = super().infer(data)
        status = getattr(self.runtime, "status", lambda: {})()
        chunk.metadata.update(
            {
                "latency_compensation_steps": int(
                    status.get("latency_compensation_steps", 0)
                ),
                "uncompensated_action_start": int(
                    status.get("uncompensated_action_start", 1)
                ),
                "compensated_action_start": int(
                    status.get("compensated_action_start", 1)
                ),
                "full_action_horizon": int(status.get("full_action_horizon", 0)),
                "executable_action_horizon": int(
                    status.get("executable_action_horizon", chunk.horizon)
                ),
            }
        )
        return chunk


class DiffusionInferenceRuntime:
    """Restores a serialized Diffusion Policy workspace and its EMA policy."""

    def __init__(
        self,
        *,
        project_root: str,
        checkpoint: str,
        device: str = "cuda:0",
        num_inference_steps: int = 100,
        latency_compensation_steps: int = 3,
        use_ema: bool = True,
        seed: int | None = 41,
    ) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        self.checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not self.project_root.is_dir():
            raise FileNotFoundError(
                f"Diffusion Policy project not found: {self.project_root}"
            )
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Diffusion Policy checkpoint not found: {self.checkpoint_path}"
            )
        if int(num_inference_steps) <= 0:
            raise ValueError("num_inference_steps must be positive")
        if int(latency_compensation_steps) < 0:
            raise ValueError("latency_compensation_steps must be non-negative")
        root_str = str(self.project_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        import dill
        import hydra
        import torch
        from omegaconf import OmegaConf

        OmegaConf.register_new_resolver("eval", eval, replace=True)
        payload = torch.load(
            self.checkpoint_path.open("rb"),
            map_location="cpu",
            pickle_module=dill,
        )
        cfg = payload["cfg"]
        workspace_class = hydra.utils.get_class(cfg._target_)
        workspace = workspace_class(cfg)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        if bool(use_ema) and bool(cfg.training.use_ema):
            policy = workspace.ema_model
        else:
            policy = workspace.model
        if policy is None:
            raise RuntimeError("selected Diffusion Policy model is unavailable")
        self.torch = torch
        self.cfg = cfg
        self.workspace = workspace
        self.policy = policy
        self.device = str(device)
        self.num_inference_steps = int(num_inference_steps)
        self.latency_compensation_steps = int(latency_compensation_steps)
        self.use_ema = bool(use_ema and cfg.training.use_ema)
        self.seed = None if seed is None else int(seed)
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be non-negative or None")
        self.inference_index = 0
        self.episode_index = -1
        self.episode_inference_index = 0
        self.episode_seed = self.seed
        self.image_shapes: dict[str, tuple[int, int, int]] = {}
        for key in ("base_image", "left_wrist_image", "right_wrist_image"):
            shape = tuple(int(value) for value in cfg.shape_meta.obs[key].shape)
            if len(shape) != 3 or shape[0] != 3:
                raise ValueError(
                    f"Diffusion Policy RGB shape for {key} must be (3, H, W), "
                    f"got {shape}"
                )
            self.image_shapes[key] = shape
        self.policy.num_inference_steps = self.num_inference_steps
        self.policy.eval().to(torch.device(self.device))
        self.full_action_horizon = int(self.policy.horizon)
        self.uncompensated_action_start = int(self.policy.n_obs_steps) - 1
        self.compensated_action_start = (
            self.uncompensated_action_start + self.latency_compensation_steps
        )
        self.executable_action_horizon = min(
            int(self.policy.n_action_steps),
            self.full_action_horizon - self.compensated_action_start,
        )
        if self.executable_action_horizon <= 0:
            raise ValueError(
                "latency compensation exhausts the checkpoint action horizon: "
                f"horizon={self.full_action_horizon} "
                f"start={self.compensated_action_start}"
            )
        state_stats = self.policy.normalizer["state"].get_input_stats()
        self.state_min = _normalizer_stat_numpy(state_stats, "min")
        self.state_max = _normalizer_stat_numpy(state_stats, "max")
        if self.state_min.shape != (14,) or self.state_max.shape != (14,):
            raise ValueError(
                "Diffusion Policy state normalizer must contain 14-dimensional "
                f"min/max statistics, got {self.state_min.shape} and "
                f"{self.state_max.shape}"
            )
        if not np.all(np.isfinite(self.state_min)) or not np.all(
            np.isfinite(self.state_max)
        ):
            raise ValueError("Diffusion Policy state normalizer contains NaN or inf")
        if np.any(self.state_max <= self.state_min):
            dimensions = np.flatnonzero(self.state_max <= self.state_min).tolist()
            raise ValueError(
                "Diffusion Policy state normalizer has non-positive ranges; "
                f"dimensions={dimensions}"
            )
        self._last_state_clip_dimensions: tuple[int, ...] = ()

    def predict(self, obs: Mapping[str, np.ndarray]) -> np.ndarray:
        torch = self.torch
        obs_dict: dict[str, Any] = {}
        for key in ("base_image", "left_wrist_image", "right_wrist_image"):
            value = _resize_rgb_chw(
                np.asarray(obs[key]),
                target_shape=self.image_shapes[key],
                key=key,
            ).astype(np.float32) / 255.0
            obs_dict[key] = torch.as_tensor(
                value, device=self.device, dtype=torch.float32
            ).unsqueeze(0)
        state, clipped_dimensions = _clip_state_to_training_support(
            np.asarray(obs["state"], dtype=np.float32),
            self.state_min,
            self.state_max,
        )
        if clipped_dimensions != self._last_state_clip_dimensions:
            if clipped_dimensions:
                print(
                    "[DiffusionInferenceRuntime] clipped raw state to the "
                    "checkpoint normalizer's training support; dimensions="
                    f"{list(clipped_dimensions)}",
                    flush=True,
                )
            self._last_state_clip_dimensions = clipped_dimensions
        obs_dict["state"] = torch.as_tensor(
            state,
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()
        device = torch.device(self.device)
        fork_devices: list[int] = []
        if device.type == "cuda":
            fork_devices = [
                torch.cuda.current_device() if device.index is None else device.index
            ]
        with torch.random.fork_rng(devices=fork_devices), torch.inference_mode():
            if self.episode_seed is not None:
                # Reuse one initial diffusion noise realization throughout an
                # episode. Changing observations still changes the prediction,
                # but chunk boundaries no longer inject a new random mode.
                torch.manual_seed(self.episode_seed)
            result = self.policy.predict_action(obs_dict)
        self.inference_index += 1
        self.episode_inference_index += 1
        if "action_pred" not in result:
            raise KeyError(
                "Diffusion Policy result is missing the full 'action_pred' required "
                "for latency compensation"
            )
        full_actions = np.asarray(
            result["action_pred"][0].detach().cpu().numpy(), dtype=np.float32
        )
        if full_actions.shape != (self.full_action_horizon, 14):
            raise ValueError(
                "Diffusion Policy full action prediction shape mismatch: "
                f"expected {(self.full_action_horizon, 14)}, got {full_actions.shape}"
            )
        return _latency_compensated_actions(
            full_actions,
            n_obs_steps=int(self.policy.n_obs_steps),
            n_action_steps=int(self.policy.n_action_steps),
            latency_compensation_steps=self.latency_compensation_steps,
        )

    def reset_episode(self) -> None:
        self.episode_index += 1
        self.episode_inference_index = 0
        self.episode_seed = (
            None if self.seed is None else self.seed + self.episode_index
        )

    def status(self) -> dict[str, Any]:
        return {
            "project_root": str(self.project_root),
            "checkpoint": str(self.checkpoint_path),
            "device": self.device,
            "num_inference_steps": self.num_inference_steps,
            "latency_compensation_steps": self.latency_compensation_steps,
            "latency_compensation_source": "measured_live_profile",
            "full_action_horizon": self.full_action_horizon,
            "uncompensated_action_start": self.uncompensated_action_start,
            "compensated_action_start": self.compensated_action_start,
            "executable_action_horizon": self.executable_action_horizon,
            "use_ema": self.use_ema,
            "seed": self.seed,
            "seed_strategy": "fixed_within_episode_increment_between_episodes",
            "episode_seed": self.episode_seed,
            "episode_index": self.episode_index,
            "episode_inference_index": self.episode_inference_index,
            "inference_index": self.inference_index,
            "workspace": str(self.cfg._target_),
            "image_shapes": {
                key: list(shape) for key, shape in self.image_shapes.items()
            },
            "image_resize": "direct_cv2_inter_area_to_checkpoint_shape",
            "state_normalization_clip": [-1.0, 1.0],
            "state_training_min": self.state_min.tolist(),
            "state_training_max": self.state_max.tolist(),
        }


def _normalizer_stat_numpy(stats: Mapping[str, Any], key: str) -> np.ndarray:
    if key not in stats:
        raise KeyError(f"Diffusion Policy state normalizer is missing {key!r}")
    value = stats[key]
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    return np.asarray(value, dtype=np.float32).copy()


def _latency_compensated_actions(
    full_actions: np.ndarray,
    *,
    n_obs_steps: int,
    n_action_steps: int,
    latency_compensation_steps: int,
) -> np.ndarray:
    """Drop action positions that expired while synchronous inference ran."""
    actions = np.asarray(full_actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"full_actions must be 2D, got {actions.shape}")
    start = int(n_obs_steps) - 1 + int(latency_compensation_steps)
    if start < 0 or start >= actions.shape[0]:
        raise ValueError(
            "latency compensation start is outside the full action horizon: "
            f"start={start} horizon={actions.shape[0]}"
        )
    stop = min(start + int(n_action_steps), actions.shape[0])
    if stop <= start:
        raise ValueError("latency compensation produced an empty action slice")
    return actions[start:stop].copy()


def _clip_state_to_training_support(
    value: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Clip raw DP state to the per-task bounds embedded in its checkpoint."""
    array = np.asarray(value, dtype=np.float32)
    minimum = np.asarray(minimum, dtype=np.float32)
    maximum = np.asarray(maximum, dtype=np.float32)
    if array.ndim == 0 or array.shape[-1] != minimum.shape[0]:
        raise ValueError(
            "Diffusion Policy state must end in the checkpoint state dimension, "
            f"got value={array.shape}, minimum={minimum.shape}"
        )
    if maximum.shape != minimum.shape:
        raise ValueError(
            "Diffusion Policy state min/max shape mismatch: "
            f"{minimum.shape} != {maximum.shape}"
        )
    value_range = maximum - minimum
    # This is equivalent to the JEPA/MIP normalized-domain tolerance of 1e-6,
    # while keeping the DP observation in raw qpos units for its own normalizer.
    tolerance = np.maximum(
        value_range * np.float32(5e-7),
        np.finfo(np.float32).eps
        * np.maximum(np.maximum(np.abs(minimum), np.abs(maximum)), 1.0),
    )
    outside = (array < minimum - tolerance) | (array > maximum + tolerance)
    if array.ndim == 1:
        dimensions = tuple(int(index) for index in np.flatnonzero(outside))
    else:
        dimensions = tuple(
            int(index)
            for index in np.flatnonzero(
                np.any(outside, axis=tuple(range(array.ndim - 1)))
            )
        )
    return np.clip(array, minimum, maximum), dimensions
