from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from prometheus.policy.async_rtc import (
    AsyncRTCConfig,
    RealtimePolicyOutput,
)
from prometheus.policy.scheduler import ActionChunk

DEFAULT_STREAM_NAMES = (
    "base_0_color",
    "left_wrist_0_color",
    "right_wrist_0_color",
    "robot_state",
    "left_eef",
    "right_eef",
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FLOW_MATCHING_ROOT = _PROJECT_ROOT / "third_party" / "flow_matching"


def _ensure_flow_matching_importable() -> Path:
    root = _FLOW_MATCHING_ROOT.resolve()
    if not root.is_dir():
        raise ImportError(f"flow_matching vendor not found: {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def resolve_run_dir(path: str | Path) -> Path:
    run_dir = Path(path).expanduser()
    if not run_dir.is_absolute():
        run_dir = (_PROJECT_ROOT / run_dir).resolve()
    else:
        run_dir = run_dir.resolve()
    config_path = run_dir / "resolved_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing resolved config: {config_path}")
    return run_dir


TACTILE_STREAM_SUFFIX = "_tactile_flow"


class FMPolicy:
    """In-process Flow Matching policy adapter for Prometheus rollout."""

    def __init__(
        self,
        *,
        inputs: Mapping[str, Any] | None = None,
        inference: Mapping[str, Any] | None = None,
        robot: Mapping[str, Any] | None = None,
        async_inference: Mapping[str, Any] | None = None,
        action_hz: float | None = None,
        runtime: Any | None = None,
    ):
        self.inputs = dict(inputs or {})
        self.inference = dict(inference or {})
        self.robot_cfg = dict(robot or {})
        self.async_config = AsyncRTCConfig.from_mapping(async_inference)
        self.anchor = str(self.inputs.get("anchor", "left_wrist_0_color"))
        self.slop_ms = float(self.inputs.get("slop_ms", 100.0))
        self.window_size = int(self.inputs.get("window_size", 8))
        self.stride = int(self.inputs.get("stride", 1))
        self.wait_latest = bool(self.inputs.get("wait_latest", True))
        self.wait_latest_timeout_ms = float(self.inputs.get("wait_latest_timeout_ms", 5.0))
        self.stream_names = _resolve_stream_names(self.inputs)
        self.n_action_steps = _optional_int(self.inference.get("n_action_steps"))
        self.action_hz = _optional_float(action_hz)
        if self.action_hz is None:
            self.action_hz = _optional_float(self.inference.get("action_hz"))
        if self.action_hz is None:
            self.action_hz = _optional_float(self.inputs.get("action_hz"))
        self.runtime = runtime if runtime is not None else self._build_runtime()
        self._validate_async_config()

    @property
    def use_tactile(self) -> bool:
        return bool(getattr(self.runtime, "use_tactile", False))

    @property
    def async_inference_enabled(self) -> bool:
        return self.async_config.enabled

    def infer(self, data: Any) -> ActionChunk:
        frames = data.window_numpy(
            anchor=self.anchor,
            names=self._active_stream_names(),
            count=self.window_size,
            stride=self.stride,
            slop_ms=self.slop_ms,
            wait_latest=self.wait_latest,
            timeout_ms=self.wait_latest_timeout_ms,
        )
        if hasattr(self.runtime, "infer_from_window") and not hasattr(self.runtime, "predict_rot6d_abs"):
            chunk = self.runtime.infer_from_window(
                frames,
                robot=self.robot_cfg,
                num_inference_steps=_optional_int(self.inference.get("num_inference_steps")),
                solver=_optional_str(self.inference.get("solver")),
            )
            return self._to_action_chunk(chunk)
        chunk = self._infer_from_numpy_window(
            frames,
            num_inference_steps=_optional_int(self.inference.get("num_inference_steps")),
            solver=_optional_str(self.inference.get("solver")),
        )
        return self._to_action_chunk(chunk)

    def infer_realtime(
        self,
        data: Any,
        *,
        previous_actions_abs: np.ndarray | None,
        inference_delay_steps: int,
        prefix_attention_horizon: int | None = None,
    ) -> RealtimePolicyOutput:
        if not self.async_config.enabled:
            raise RuntimeError("infer_realtime requires async_inference.enabled=true")
        runtime_method = getattr(self.runtime, "predict_rot6d_abs_realtime", None)
        if not callable(runtime_method):
            raise TypeError(
                "FM runtime does not implement predict_rot6d_abs_realtime"
            )
        frames = data.window_numpy(
            anchor=self.anchor,
            names=self._active_stream_names(),
            count=self.window_size,
            stride=self.stride,
            slop_ms=self.slop_ms,
            wait_latest=self.wait_latest,
            timeout_ms=self.wait_latest_timeout_ms,
        )
        from infer.postprocess import apply_action_process
        from infer.preprocess import parse_preprocess_config

        from prometheus.policy.numpy_preprocess import build_obs_from_numpy_frames

        preprocess_cfg = parse_preprocess_config(self.runtime.cfg, robot=self.robot_cfg)
        if len(preprocess_cfg.camera_views) != self.runtime.n_image_views:
            raise ValueError(
                "preprocess camera_views count does not match model n_image_views: "
                f"{len(preprocess_cfg.camera_views)} != {self.runtime.n_image_views}"
            )
        if bool(self.runtime.use_tactile) != bool(preprocess_cfg.use_tactile):
            raise ValueError(
                "runtime.use_tactile does not match resolved_config data.use_tactile: "
                f"{self.runtime.use_tactile} != {preprocess_cfg.use_tactile}"
            )
        obs, state_raw = build_obs_from_numpy_frames(
            frames,
            preprocess_cfg,
            self.runtime.normalizer,
            window_size=self.runtime.window_size,
        )
        pred_abs, pred_norm = runtime_method(
            obs,
            state_raw=state_raw,
            prev_actions_abs=previous_actions_abs,
            inference_delay=int(inference_delay_steps),
            prefix_attention_horizon=prefix_attention_horizon,
            rtc_config=self.async_config.rtc,
            num_inference_steps=_optional_int(
                self.inference.get("num_inference_steps")
            ),
            solver=_optional_str(self.inference.get("solver")),
        )
        deploy_actions = apply_action_process(
            pred_abs,
            self.runtime.deploy.action_process,
        )
        expected = (self.runtime.action_horizon, 14)
        if deploy_actions.shape != expected:
            raise ValueError(
                f"postprocess shape {deploy_actions.shape} != expected {expected}"
            )

        deploy_actions = _slice_actions(deploy_actions, self.n_action_steps)
        model_abs = _slice_actions(pred_abs, self.n_action_steps)
        model_norm = _slice_actions(pred_norm, self.n_action_steps)
        chunk = ActionChunk(
            deploy_actions,
            action_space=str(self.runtime.deploy.action_process),
            hz=self._action_hz(default=self.runtime.deploy.action_hz),
            metadata={
                "action_horizon": self.runtime.action_horizon,
                "window_size": self.runtime.window_size,
                "timestamp": obs["timestamp"],
                "n_action_steps": int(deploy_actions.shape[0]),
                "async_inference": True,
                "rtc_enabled": bool(self.async_config.rtc["enabled"]),
                "inference_delay_steps": int(inference_delay_steps),
                "prefix_attention_horizon": (
                    None
                    if prefix_attention_horizon is None
                    else int(prefix_attention_horizon)
                ),
            },
        )
        return RealtimePolicyOutput(
            chunk=chunk,
            model_actions_abs=model_abs,
            model_actions_norm=model_norm,
        )

    def _to_action_chunk(self, chunk: Any) -> ActionChunk:
        actions = _slice_actions(chunk.actions, self.n_action_steps)
        metadata = dict(chunk.metadata)
        metadata["n_action_steps"] = int(actions.shape[0])
        if self.n_action_steps is not None:
            metadata["n_action_steps_requested"] = int(self.n_action_steps)
        return ActionChunk(
            actions,
            action_space=str(chunk.action_space),
            hz=self._action_hz(default=chunk.hz),
            metadata=metadata,
        )

    def close(self) -> None:
        return None

    def status(self) -> dict[str, Any]:
        runtime = self.runtime
        device = getattr(runtime, "device", None)
        return {
            "type": type(self).__name__,
            "anchor": self.anchor,
            "slop_ms": self.slop_ms,
            "window_size": self.window_size,
            "stride": self.stride,
            "wait_latest": self.wait_latest,
            "wait_latest_timeout_ms": self.wait_latest_timeout_ms,
            "stream_names": list(self.stream_names),
            "active_stream_names": list(self._active_stream_names()),
            "use_tactile": self.use_tactile,
            "n_action_steps": self.n_action_steps,
            "run_dir": str(getattr(runtime, "run_dir", "")),
            "checkpoint": str(getattr(runtime, "checkpoint_path", "")),
            "device": str(device) if device is not None else "",
            "action_hz": self._action_hz(default=None),
            "async_inference": {
                "enabled": self.async_config.enabled,
                "initial_delay_steps": self.async_config.initial_delay_steps,
                "latency_history": self.async_config.latency_history,
                "join_timeout_s": self.async_config.join_timeout_s,
                "rtc": dict(self.async_config.rtc),
            },
        }

    def _build_runtime(self) -> Any:
        _ensure_flow_matching_importable()
        from infer import FMInferenceRuntime

        if "run_dir" not in self.inference:
            raise ValueError("inference.run_dir is required for FMPolicy")
        run_dir = resolve_run_dir(self.inference["run_dir"])
        checkpoint = self.inference.get("checkpoint")
        return FMInferenceRuntime(
            run_dir,
            checkpoint=checkpoint,
            device=_optional_str(self.inference.get("device")),
            warmup=bool(self.inference.get("warmup", True)),
        )

    def _validate_async_config(self) -> None:
        if not self.async_config.enabled:
            return
        action_horizon = int(getattr(self.runtime, "action_horizon", 0))
        if action_horizon <= 0:
            raise ValueError(
                "async inference requires runtime.action_horizon to be positive"
            )
        executable_horizon = (
            action_horizon
            if self.n_action_steps is None
            else int(self.n_action_steps)
        )
        if executable_horizon > action_horizon:
            raise ValueError(
                f"n_action_steps={executable_horizon} exceeds predicted action "
                f"horizon {action_horizon}"
            )
        if self.async_config.execution_horizon >= executable_horizon:
            raise ValueError(
                "async_inference.rtc.execution_horizon must be smaller than "
                f"the executable horizon ({self.async_config.execution_horizon} "
                f">= {executable_horizon})"
            )
        if bool(self.async_config.rtc["enabled"]):
            solver = _optional_str(self.inference.get("solver"))
            if solver is None:
                solver = str(getattr(self.runtime, "solver", ""))
            if str(solver).lower() != "euler":
                raise ValueError(
                    "async RTC inference currently requires solver='euler', "
                    f"got {solver!r}"
                )

    def _infer_from_numpy_window(
        self,
        frames: Any,
        *,
        num_inference_steps: int | None,
        solver: str | None,
    ) -> Any:
        from infer.postprocess import apply_action_process
        from infer.preprocess import parse_preprocess_config
        from infer.types import InferenceChunk

        from prometheus.policy.numpy_preprocess import build_obs_from_numpy_frames

        preprocess_cfg = parse_preprocess_config(self.runtime.cfg, robot=self.robot_cfg)
        if len(preprocess_cfg.camera_views) != self.runtime.n_image_views:
            raise ValueError(
                "preprocess camera_views count does not match model n_image_views: "
                f"{len(preprocess_cfg.camera_views)} != {self.runtime.n_image_views}"
            )
        if bool(self.runtime.use_tactile) != bool(preprocess_cfg.use_tactile):
            raise ValueError(
                "runtime.use_tactile does not match resolved_config data.use_tactile: "
                f"{self.runtime.use_tactile} != {preprocess_cfg.use_tactile}"
            )
        obs, state_raw = build_obs_from_numpy_frames(
            frames,
            preprocess_cfg,
            self.runtime.normalizer,
            window_size=self.runtime.window_size,
        )
        pred_rot6d = self.runtime.predict_rot6d_abs(
            obs,
            state_raw=state_raw,
            num_inference_steps=num_inference_steps,
            solver=solver,
        )
        actions = apply_action_process(pred_rot6d, self.runtime.deploy.action_process)
        expected = (self.runtime.action_horizon, 14)
        if actions.shape != expected:
            raise ValueError(f"postprocess shape {actions.shape} != expected {expected}")
        return InferenceChunk(
            actions=actions,
            action_space=self.runtime.deploy.action_process,
            hz=self._action_hz(default=self.runtime.deploy.action_hz),
            metadata={
                "action_horizon": self.runtime.action_horizon,
                "window_size": self.runtime.window_size,
                "timestamp": obs["timestamp"],
            },
        )

    def _action_hz(self, *, default: Any) -> float:
        hz = self.action_hz
        if hz is None:
            if default is None:
                default = getattr(getattr(self.runtime, "deploy", None), "action_hz", None)
            hz = default
        if hz is None:
            raise ValueError("policy action_hz is not configured")
        if float(hz) <= 0:
            raise ValueError("policy action_hz must be positive")
        return float(hz)


    def _active_stream_names(self) -> tuple[str, ...]:
        if self.use_tactile:
            return self.stream_names
        return tuple(
            name for name in self.stream_names if not str(name).endswith(TACTILE_STREAM_SUFFIX)
        )


def _resolve_stream_names(inputs: Mapping[str, Any]) -> tuple[str, ...]:
    names = list(inputs.get("stream_names", DEFAULT_STREAM_NAMES))
    tactile_names = inputs.get("tactile_stream_names")
    if tactile_names is not None:
        if isinstance(tactile_names, str):
            raise TypeError("tactile_stream_names must be a sequence, not a string")
        names.extend(str(item) for item in tactile_names)
    return _names(names)


def _names(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        raise TypeError("stream names must be a sequence, not a string")
    return tuple(str(item) for item in value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _slice_actions(actions: Any, n_action_steps: int | None) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if n_action_steps is None:
        return arr
    n = int(n_action_steps)
    if n < 1:
        raise ValueError(f"n_action_steps must be positive, got {n}")
    if n > arr.shape[0]:
        raise ValueError(
            f"n_action_steps={n} exceeds predicted action horizon {arr.shape[0]}"
        )
    return arr[:n].copy()
