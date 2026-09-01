from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class ActionChunk:
    """A finite command sequence in one robot action space."""

    actions: np.ndarray
    action_space: str
    hz: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if actions.ndim != 2:
            raise ValueError(f"actions must be 1D or 2D, got shape {actions.shape}")
        if actions.shape[0] <= 0 or actions.shape[1] <= 0:
            raise ValueError(f"actions must be non-empty, got shape {actions.shape}")
        if not np.all(np.isfinite(actions)):
            raise ValueError("actions contain NaN or inf")
        if not str(self.action_space):
            raise ValueError("action_space must be non-empty")
        if float(self.hz) <= 0:
            raise ValueError("hz must be positive")

        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "action_space", str(self.action_space))
        object.__setattr__(self, "hz", float(self.hz))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def horizon(self) -> int:
        return int(self.actions.shape[0])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[1])

    @property
    def dt(self) -> float:
        return 1.0 / self.hz

    def __len__(self) -> int:
        return self.horizon


@dataclass(frozen=True)
class ActionContext:
    chunk: ActionChunk
    index: int
    previous_action: np.ndarray | None

    @property
    def dt(self) -> float:
        return self.chunk.dt

    @property
    def action_space(self) -> str:
        return self.chunk.action_space


class ActionFilter:
    """One independently switchable action transformation or guard."""

    def __init__(self, name: str, *, enabled: bool = True):
        if not name:
            raise ValueError("filter name must be non-empty")
        self.name = str(name)
        self.enabled = bool(enabled)

    def reset(self) -> None:
        pass

    def apply(self, action: np.ndarray, context: ActionContext) -> np.ndarray:
        return action

    def commit(self, action: np.ndarray, context: ActionContext) -> None:
        """Commit the final command after every downstream guard accepts it."""

    def consume_stop_chunk_request(self) -> bool:
        """Return whether execution should replan after the committed action."""
        return False


class SafetyGuardError(ValueError):
    """A command was rejected before it reached the robot."""


class EmaFilter(ActionFilter):
    """Elementwise exponential smoothing over scheduled actions."""

    def __init__(self, alpha: float, *, name: str = "ema", enabled: bool = True):
        super().__init__(name, enabled=enabled)
        if not 0.0 <= float(alpha) < 1.0:
            raise ValueError("alpha must be in [0, 1)")
        self.alpha = float(alpha)
        self._state: np.ndarray | None = None

    def reset(self) -> None:
        self._state = None

    def apply(self, action: np.ndarray, context: ActionContext) -> np.ndarray:
        if self._state is None:
            self._state = action.copy()
            return action
        out = self.alpha * self._state + (1.0 - self.alpha) * action
        self._state = out.copy()
        return out.astype(np.float32)


class HoldDimensionsFilter(ActionFilter):
    """Hold task-inactive dimensions at the measured start of each chunk."""

    def __init__(
        self,
        dimensions: list[int] | tuple[int, ...],
        *,
        name: str = "hold_dimensions",
        enabled: bool = True,
    ):
        super().__init__(name, enabled=enabled)
        indexes = tuple(int(index) for index in dimensions)
        if not indexes or len(set(indexes)) != len(indexes) or any(index < 0 for index in indexes):
            raise ValueError("dimensions must contain unique non-negative indexes")
        self.dimensions = indexes

    def apply(self, action: np.ndarray, context: ActionContext) -> np.ndarray:
        if any(index >= action.size for index in self.dimensions):
            raise ValueError(
                f"{self.name} dimensions {self.dimensions} are not compatible "
                f"with action shape {action.shape}"
            )
        current = context.chunk.metadata.get("current_action")
        if current is None:
            raise ValueError(f"{self.name} requires current_action metadata")
        current = np.asarray(current, dtype=np.float32).reshape(-1)
        if current.shape != action.shape or not np.all(np.isfinite(current)):
            raise ValueError(
                f"{self.name} current_action must match action shape {action.shape}"
            )
        out = action.copy()
        selected = np.asarray(self.dimensions, dtype=np.int64)
        out[selected] = current[selected]
        return out.astype(np.float32)


class MaxDeltaFilter(ActionFilter):
    """Clamp elementwise action changes relative to observed or commanded qpos."""

    def __init__(
        self,
        max_delta: float | np.ndarray,
        *,
        require_previous: bool = False,
        report_clipping: bool = False,
        name: str = "max_delta",
        enabled: bool = True,
    ):
        super().__init__(name, enabled=enabled)
        value = np.asarray(max_delta, dtype=np.float32)
        if not np.all(np.isfinite(value)) or np.any(value <= 0):
            raise ValueError("max_delta must be finite and positive")
        self.max_delta = value
        self.require_previous = bool(require_previous)
        self.report_clipping = bool(report_clipping)

    def apply(self, action: np.ndarray, context: ActionContext) -> np.ndarray:
        previous = context.previous_action
        if previous is None:
            if self.require_previous:
                raise ValueError(
                    f"{self.name} requires current_action metadata for the first command"
                )
            return action
        requested_delta = action - previous
        try:
            applied_delta = np.clip(
                requested_delta,
                -self.max_delta,
                self.max_delta,
            )
            exceeded = np.abs(requested_delta) > np.nextafter(
                self.max_delta,
                np.float32(np.inf),
            )
        except ValueError as exc:
            raise ValueError(
                f"{self.name} max_delta is not compatible with action shape {action.shape}"
            ) from exc
        if self.report_clipping and np.any(exceeded):
            indexes = np.flatnonzero(exceeded).tolist()
            print(
                f"[{self.name}] clipped action index {context.index}; "
                f"dimensions={indexes}, "
                f"requested_delta={requested_delta.tolist()}, "
                f"applied_delta={applied_delta.tolist()}, "
                f"max_delta={np.broadcast_to(self.max_delta, action.shape).tolist()}",
                flush=True,
            )
        return (previous + applied_delta).astype(np.float32)


class GripperCloseLeadFilter(ActionFilter):
    """Advance only meaningful gripper-closing targets within a policy chunk.

    Opening targets retain their original timing. Downstream dynamics filters
    remain responsible for limiting the resulting command velocity and
    acceleration.
    """

    def __init__(
        self,
        lead_steps: int,
        gripper_indices: list[int] | tuple[int, ...],
        *,
        min_close_delta: float = 0.0,
        name: str = "gripper_close_lead",
        enabled: bool = True,
    ):
        super().__init__(name, enabled=enabled)
        if isinstance(lead_steps, bool) or int(lead_steps) != lead_steps:
            raise ValueError("lead_steps must be a non-negative integer")
        if int(lead_steps) < 0:
            raise ValueError("lead_steps must be a non-negative integer")
        indices = tuple(int(index) for index in gripper_indices)
        if not indices or len(set(indices)) != len(indices) or any(index < 0 for index in indices):
            raise ValueError("gripper_indices must contain unique non-negative indexes")
        if not np.isfinite(min_close_delta) or float(min_close_delta) < 0:
            raise ValueError("min_close_delta must be finite and non-negative")
        self.lead_steps = int(lead_steps)
        self.gripper_indices = indices
        self.min_close_delta = float(min_close_delta)

    def apply(self, action: np.ndarray, context: ActionContext) -> np.ndarray:
        if self.lead_steps == 0:
            return action
        if any(index >= action.size for index in self.gripper_indices):
            raise ValueError(
                f"{self.name} gripper indexes {self.gripper_indices} are not compatible "
                f"with action shape {action.shape}"
            )
        stop = min(context.chunk.horizon, context.index + self.lead_steps + 1)
        future = context.chunk.actions[context.index:stop]
        out = action.copy()
        for index in self.gripper_indices:
            closing_target = float(np.min(future[:, index]))
            if float(action[index]) - closing_target >= self.min_close_delta:
                out[index] = closing_target
        return out.astype(np.float32)


class GripperCloseHeightGateFilter(ActionFilter):
    """Keep an open gripper open until its filtered arm target is low enough.

    A permitted close is latched until the policy explicitly reopens the
    gripper. A blocked close asks the scheduler to stop the current chunk after
    sending the arm target with the gripper held open, so the policy can replan
    from a fresh image and measured state instead of executing the remainder of
    an unsafe open-loop grasp attempt.
    """

    def __init__(
        self,
        max_close_height_m: float | list[float] | tuple[float, ...],
        gripper_indices: list[int] | tuple[int, ...],
        arm_starts: list[int] | tuple[int, ...],
        *,
        arm_dof: int = 6,
        release_opening: float = 0.06,
        min_close_delta: float = 0.005,
        request_replan: bool = True,
        robot_model: str = "X5",
        urdf_path: str | None = None,
        height_fn: Callable[[np.ndarray], float] | None = None,
        name: str = "gripper_close_height_gate",
        enabled: bool = True,
    ):
        super().__init__(name, enabled=enabled)
        indices = tuple(int(index) for index in gripper_indices)
        starts = tuple(int(index) for index in arm_starts)
        if not indices or len(indices) != len(starts):
            raise ValueError("gripper_indices and arm_starts must have equal non-zero length")
        if len(set(indices)) != len(indices) or any(index < 0 for index in indices):
            raise ValueError("gripper_indices must contain unique non-negative indexes")
        if any(index < 0 for index in starts):
            raise ValueError("arm_starts must contain non-negative indexes")
        if isinstance(arm_dof, bool) or int(arm_dof) != arm_dof or int(arm_dof) <= 0:
            raise ValueError("arm_dof must be a positive integer")
        heights = np.asarray(max_close_height_m, dtype=np.float32)
        try:
            heights = np.broadcast_to(heights, (len(indices),)).copy()
        except ValueError as exc:
            raise ValueError("max_close_height_m must be scalar or match gripper_indices") from exc
        if not np.all(np.isfinite(heights)) or np.any(heights <= 0):
            raise ValueError("max_close_height_m must be finite and positive")
        if not np.isfinite(release_opening) or float(release_opening) <= 0:
            raise ValueError("release_opening must be finite and positive")
        if not np.isfinite(min_close_delta) or float(min_close_delta) < 0:
            raise ValueError("min_close_delta must be finite and non-negative")

        self.max_close_height_m = heights
        self.gripper_indices = indices
        self.arm_starts = starts
        self.arm_dof = int(arm_dof)
        self.release_opening = float(release_opening)
        self.min_close_delta = float(min_close_delta)
        self.request_replan = bool(request_replan)
        self.robot_model = str(robot_model)
        self.urdf_path = None if urdf_path is None else str(Path(urdf_path).expanduser().resolve())
        self._height_fn = height_fn or self._build_height_fn()
        self._closing_latched = np.zeros(len(indices), dtype=bool)
        self._blocked = np.zeros(len(indices), dtype=bool)
        self._stop_chunk_requested = False

    def _build_height_fn(self) -> Callable[[np.ndarray], float]:
        if self.urdf_path is None:
            raise ValueError("urdf_path is required when height_fn is not provided")
        if not Path(self.urdf_path).is_file():
            raise FileNotFoundError(f"X5 URDF not found: {self.urdf_path}")
        import arx5_interface as arx5

        config = arx5.RobotConfigFactory.get_instance().get_config(self.robot_model)
        if int(config.joint_dof) != self.arm_dof:
            raise ValueError(
                f"{self.robot_model} joint_dof={config.joint_dof} does not match arm_dof={self.arm_dof}"
            )
        solver = arx5.Arx5Solver(
            self.urdf_path,
            config.joint_dof,
            config.joint_pos_min,
            config.joint_pos_max,
            config.base_link_name,
            config.eef_link_name,
            config.gravity_vector,
        )

        def height(joints: np.ndarray) -> float:
            pose = np.asarray(
                solver.forward_kinematics(np.asarray(joints, dtype=np.float64)),
                dtype=np.float64,
            ).reshape(-1)
            if pose.size < 3 or not np.isfinite(pose[2]):
                raise ValueError(f"{self.name} forward kinematics returned invalid pose")
            return float(pose[2])

        return height

    def reset(self) -> None:
        self._closing_latched.fill(False)
        self._blocked.fill(False)
        self._stop_chunk_requested = False

    def apply(self, action: np.ndarray, context: ActionContext) -> np.ndarray:
        if context.action_space != "abs_qpos":
            raise ValueError(f"{self.name} requires abs_qpos actions")
        previous = context.previous_action
        if previous is None:
            raise ValueError(f"{self.name} requires a previous or observed action")
        if any(index >= action.size for index in self.gripper_indices) or any(
            start + self.arm_dof > action.size for start in self.arm_starts
        ):
            raise ValueError(f"{self.name} indexes are not compatible with action shape {action.shape}")

        self._stop_chunk_requested = False
        out = action.copy()
        for slot, (gripper_index, arm_start) in enumerate(
            zip(self.gripper_indices, self.arm_starts)
        ):
            target = float(out[gripper_index])
            current = float(previous[gripper_index])
            if self._closing_latched[slot]:
                if target >= self.release_opening and target > current:
                    self._closing_latched[slot] = False
                continue
            if current - target < self.min_close_delta:
                self._blocked[slot] = False
                continue

            height_m = float(self._height_fn(out[arm_start : arm_start + self.arm_dof]))
            if not np.isfinite(height_m):
                raise ValueError(f"{self.name} produced a non-finite EEF height")
            if height_m > float(self.max_close_height_m[slot]):
                out[gripper_index] = current
                self._stop_chunk_requested = self._stop_chunk_requested or self.request_replan
                if not self._blocked[slot]:
                    print(
                        f"[{self.name}] blocked gripper index {gripper_index}: "
                        f"eef_z={height_m:.6f}m > "
                        f"max={float(self.max_close_height_m[slot]):.6f}m; replanning",
                        flush=True,
                    )
                self._blocked[slot] = True
                continue

            self._closing_latched[slot] = True
            self._blocked[slot] = False
            print(
                f"[{self.name}] allowed gripper index {gripper_index}: "
                f"eef_z={height_m:.6f}m <= "
                f"max={float(self.max_close_height_m[slot]):.6f}m",
                flush=True,
            )
        return out.astype(np.float32)

    def consume_stop_chunk_request(self) -> bool:
        requested = self._stop_chunk_requested
        self._stop_chunk_requested = False
        return bool(requested)


class StatefulDynamicsFilter(ActionFilter):
    """Limit command velocity and acceleration using measured feedback.

    Within a chunk, the filter retains the previous filtered command and delta.
    At every new policy chunk it rebases to measured qpos and measured qvel.
    This prevents an unattained target from running farther ahead of hardware,
    while preserving physical velocity for a bounded deceleration/reversal.
    """

    def __init__(
        self,
        max_delta: float | np.ndarray,
        max_delta_change: float | np.ndarray,
        *,
        lower: float | np.ndarray | None = None,
        upper: float | np.ndarray | None = None,
        name: str = "stateful_dynamics",
        enabled: bool = True,
    ):
        super().__init__(name, enabled=enabled)
        self.max_delta = self._positive_finite(max_delta, "max_delta")
        self.max_delta_change = self._positive_finite(
            max_delta_change,
            "max_delta_change",
        )
        if (lower is None) != (upper is None):
            raise ValueError("lower and upper must either both be provided or both be omitted")
        self.lower = None if lower is None else np.asarray(lower, dtype=np.float32)
        self.upper = None if upper is None else np.asarray(upper, dtype=np.float32)
        if self.lower is not None and self.upper is not None:
            if not np.all(np.isfinite(self.lower)) or not np.all(np.isfinite(self.upper)):
                raise ValueError("bounds must be finite")
            if np.any(self.lower >= self.upper):
                raise ValueError("every lower bound must be less than its upper bound")
        self._state: np.ndarray | None = None
        self._previous_delta: np.ndarray | None = None

    @staticmethod
    def _positive_finite(value: float | np.ndarray, label: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if not np.all(np.isfinite(array)) or np.any(array <= 0):
            raise ValueError(f"{label} must be finite and positive")
        return array

    def reset(self) -> None:
        self._state = None
        self._previous_delta = None

    def apply(self, action: np.ndarray, context: ActionContext) -> np.ndarray:
        if context.index == 0:
            current = context.chunk.metadata.get("current_action")
            velocity = context.chunk.metadata.get("current_velocity")
            if current is None or velocity is None:
                raise ValueError(
                    f"{self.name} requires current_action and current_velocity metadata"
                )
            current = np.asarray(current, dtype=np.float32).reshape(-1)
            velocity = np.asarray(velocity, dtype=np.float32).reshape(-1)
            if current.shape != action.shape or velocity.shape != action.shape:
                raise ValueError(
                    f"{self.name} feedback shapes must match action shape {action.shape}"
                )
            if not np.all(np.isfinite(current)) or not np.all(np.isfinite(velocity)):
                raise ValueError(f"{self.name} feedback contains NaN or inf")
            self._state = current.copy()
            # Do not pre-clip measured velocity. If hardware is already moving
            # faster than the deployment rate limit, the only acceleration-
            # continuous command is a temporary braking target outside that
            # soft rate. The downstream hard guard remains authoritative.
            self._previous_delta = (velocity * context.dt).astype(np.float32)
        elif self._state is None or self._previous_delta is None:
            raise RuntimeError(f"{self.name} received a non-initial action before state")

        assert self._state is not None
        assert self._previous_delta is not None
        try:
            desired_delta = np.clip(
                action - self._state,
                -self.max_delta,
                self.max_delta,
            )
            delta = np.clip(
                desired_delta,
                self._previous_delta - self.max_delta_change,
                self._previous_delta + self.max_delta_change,
            )
            if self.lower is not None and self.upper is not None:
                lower = np.broadcast_to(self.lower, action.shape)
                upper = np.broadcast_to(self.upper, action.shape)
                if np.any(self._state < lower) or np.any(self._state > upper):
                    indexes = np.flatnonzero(
                        (self._state < lower) | (self._state > upper)
                    ).tolist()
                    raise SafetyGuardError(
                        f"{self.name} measured state is outside hard bounds; "
                        f"dimensions={indexes}, state={self._state.tolist()}"
                    )
                # Hard position bounds take precedence over the soft
                # acceleration envelope when braking near a joint limit.
                delta = np.clip(delta, lower - self._state, upper - self._state)
        except SafetyGuardError:
            raise
        except ValueError as exc:
            raise ValueError(
                f"{self.name} limits are not compatible with action shape {action.shape}"
            ) from exc

        out = (self._state + delta).astype(np.float32)
        return out

    def commit(self, action: np.ndarray, context: ActionContext) -> None:
        if self._state is None:
            raise RuntimeError(f"{self.name} cannot commit before apply")
        delta = np.asarray(action - self._state, dtype=np.float32)
        self._state = np.asarray(action, dtype=np.float32).copy()
        self._previous_delta = delta.copy()


class BoundsGuard(ActionFilter):
    """Reject bounds violations, optionally clipping only a small raw margin."""

    def __init__(
        self,
        lower: float | np.ndarray,
        upper: float | np.ndarray,
        *,
        clip_tolerance: float | np.ndarray = 0.0,
        name: str = "bounds_guard",
        enabled: bool = True,
    ):
        super().__init__(name, enabled=enabled)
        self.lower = np.asarray(lower, dtype=np.float32)
        self.upper = np.asarray(upper, dtype=np.float32)
        self.clip_tolerance = np.asarray(clip_tolerance, dtype=np.float32)
        if not np.all(np.isfinite(self.lower)) or not np.all(np.isfinite(self.upper)):
            raise ValueError("bounds must be finite")
        if np.any(self.lower >= self.upper):
            raise ValueError("every lower bound must be less than its upper bound")
        if not np.all(np.isfinite(self.clip_tolerance)) or np.any(self.clip_tolerance < 0):
            raise ValueError("clip_tolerance must be finite and non-negative")

    def apply(self, action: np.ndarray, context: ActionContext) -> np.ndarray:
        try:
            below = action < self.lower
            above = action > self.upper
        except ValueError as exc:
            raise ValueError(
                f"{self.name} bounds are not compatible with action shape {action.shape}"
            ) from exc
        if np.any(below) or np.any(above):
            indexes = np.flatnonzero(below | above).tolist()
            try:
                hard_below = action < self.lower - self.clip_tolerance
                hard_above = action > self.upper + self.clip_tolerance
            except ValueError as exc:
                raise ValueError(
                    f"{self.name} clip_tolerance is not compatible with action shape "
                    f"{action.shape}"
                ) from exc
            if not np.any(hard_below) and not np.any(hard_above):
                clipped = np.clip(action, self.lower, self.upper)
                print(
                    f"[{self.name}] clipped small raw bounds overshoot at action "
                    f"index {context.index}; dimensions={indexes}; "
                    f"clip_tolerance={np.broadcast_to(self.clip_tolerance, action.shape).tolist()}",
                    flush=True,
                )
                return np.asarray(clipped, dtype=np.float32)
            message = (
                f"{self.name} rejected action index {context.index}; "
                f"out-of-bounds dimensions={indexes}, action={action.tolist()}, "
                f"clip_tolerance={np.broadcast_to(self.clip_tolerance, action.shape).tolist()}"
            )
            print(f"[SafetyGuard] {message}", flush=True)
            raise SafetyGuardError(message)
        return action


class MaxDeltaGuard(ActionFilter):
    """Reject elementwise jumps from the observed or previous commanded qpos."""

    def __init__(
        self,
        max_delta: float | np.ndarray,
        *,
        name: str = "max_delta_guard",
        enabled: bool = True,
    ):
        super().__init__(name, enabled=enabled)
        self.max_delta = np.asarray(max_delta, dtype=np.float32)
        if not np.all(np.isfinite(self.max_delta)) or np.any(self.max_delta <= 0):
            raise ValueError("max_delta must be finite and positive")

    def apply(self, action: np.ndarray, context: ActionContext) -> np.ndarray:
        previous = context.previous_action
        if previous is None:
            raise ValueError(
                f"{self.name} requires current_action metadata for the first command"
            )
        try:
            delta = np.abs(action - previous)
            # Filters and actions use float32.  A command constructed exactly
            # at the configured limit can therefore land one representable
            # float above that limit after subtraction (for example,
            # 0.0330000035 versus 0.0329999998).  Treat that single ULP as the
            # same boundary value, while rejecting the very next float and
            # every materially larger jump.
            float32_boundary = np.nextafter(
                self.max_delta,
                np.float32(np.inf),
            )
            exceeded = delta > float32_boundary
        except ValueError as exc:
            raise ValueError(
                f"{self.name} max_delta is not compatible with action shape {action.shape}"
            ) from exc
        if np.any(exceeded):
            indexes = np.flatnonzero(exceeded).tolist()
            message = (
                f"{self.name} rejected action index {context.index}; "
                f"dimensions={indexes}, abs_delta={delta.tolist()}, "
                f"max_delta={np.broadcast_to(self.max_delta, action.shape).tolist()}"
            )
            print(f"[SafetyGuard] {message}", flush=True)
            raise SafetyGuardError(message)
        return action


class ActionScheduler:
    """Executes ActionChunk through a robot control client."""

    def __init__(
        self,
        robot_client: Any,
        *,
        filters: list[ActionFilter] | None = None,
        dry_run: bool = False,
        wait: bool = True,
        timeout_s: float = 3.0,
        history: int = 1024,
    ):
        if history <= 0:
            raise ValueError("history must be positive")
        self.robot_client = robot_client
        self.dry_run = bool(dry_run)
        self.wait = bool(wait)
        self.timeout_s = float(timeout_s)
        self.history: deque[np.ndarray] = deque(maxlen=int(history))
        self._filters: dict[str, ActionFilter] = {}
        self._previous_action: np.ndarray | None = None
        self._next_action_time: float | None = None
        for item in filters or []:
            self.add_filter(item)

    @property
    def filters(self) -> tuple[ActionFilter, ...]:
        return tuple(self._filters.values())

    def add_filter(self, item: ActionFilter) -> None:
        if item.name in self._filters:
            raise ValueError(f"duplicate action filter {item.name!r}")
        self._filters[item.name] = item

    def set_filter_enabled(self, name: str, enabled: bool) -> None:
        item = self._filters[name]
        enabled = bool(enabled)
        if item.enabled != enabled:
            item.reset()
        item.enabled = enabled

    def reset(self) -> None:
        self._previous_action = None
        self._next_action_time = None
        for item in self._filters.values():
            item.reset()

    def idle(self, duration_s: float) -> None:
        """Keep servicing the robot client while waiting for fresh sensors."""
        duration_s = float(duration_s)
        if duration_s < 0:
            raise ValueError("duration_s must be non-negative")
        if duration_s > 0:
            self._spin_sleep(duration_s)

    def execute(
        self,
        chunk: ActionChunk,
        *,
        steps: int | None = None,
        on_action: Callable[[np.ndarray, ActionContext], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> int:
        if chunk.action_space != str(self.robot_client.action_space):
            raise ValueError(
                f"chunk action_space={chunk.action_space!r} does not match "
                f"robot action_space={self.robot_client.action_space!r}"
            )
        observed_current = self._current_action_from_metadata(chunk)
        if observed_current is not None:
            # A new policy chunk is conditioned on a fresh robot observation.
            # The previous chunk's final target can be ahead of the hardware
            # while it is still tracking, so guard the new first action from
            # the observed qpos. Subsequent actions remain guarded from the
            # preceding command below.
            self._previous_action = observed_current
        n_steps = chunk.horizon if steps is None else min(int(steps), chunk.horizon)
        if n_steps <= 0:
            raise ValueError("steps must be positive")

        sent = 0
        now = time.monotonic()
        next_time = now if self._next_action_time is None else max(now, self._next_action_time)
        for index in range(n_steps):
            if should_stop is not None and should_stop():
                break
            delay = next_time - time.monotonic()
            if delay > 0:
                self._spin_sleep(delay)
            if should_stop is not None and should_stop():
                break

            action = chunk.actions[index].copy()
            context = ActionContext(chunk=chunk, index=index, previous_action=self._previous_action)
            action = self._apply_filters(action, context)
            self._send(action)
            for item in self._filters.values():
                if item.enabled:
                    item.commit(action, context)
            self._previous_action = action.copy()
            self.history.append(action.copy())
            sent += 1
            # The command has already reached the robot at this point. Record
            # it in scheduler accounting before an asynchronous recorder
            # callback can fail, so error manifests never under-report motion.
            if on_action is not None:
                on_action(action.copy(), context)

            stop_chunk = any(
                item.consume_stop_chunk_request()
                for item in self._filters.values()
                if item.enabled
            )

            next_time += chunk.dt
            # Preserve the next 10 Hz deadline across policy chunks. Without
            # this, the first action after a fast re-inference can be sent less
            # than one model period after the previous chunk's final action.
            self._next_action_time = next_time
            if stop_chunk:
                break
        return sent

    def _apply_filters(self, action: np.ndarray, context: ActionContext) -> np.ndarray:
        for item in self._filters.values():
            if not item.enabled:
                continue
            action = np.asarray(item.apply(action, context), dtype=np.float32).reshape(-1)
            if action.shape != (context.chunk.action_dim,):
                raise ValueError(
                    f"filter {item.name!r} returned shape {action.shape}, "
                    f"expected {(context.chunk.action_dim,)}"
                )
            if not np.all(np.isfinite(action)):
                raise ValueError(f"filter {item.name!r} returned NaN or inf")
        return action

    def _current_action_from_metadata(self, chunk: ActionChunk) -> np.ndarray | None:
        value = chunk.metadata.get("current_action")
        if value is None:
            return None
        current = np.asarray(value, dtype=np.float32).reshape(-1)
        if current.shape != (chunk.action_dim,):
            raise ValueError(
                f"metadata current_action shape {current.shape} does not match "
                f"chunk action dim {(chunk.action_dim,)}"
            )
        if not np.all(np.isfinite(current)):
            raise ValueError("metadata current_action contains NaN or inf")
        return current.copy()

    def _send(self, action: np.ndarray) -> dict[str, Any] | None:
        if self.dry_run:
            print(f"[ActionScheduler] dry_run action={action.tolist()}", flush=True)
            return None
        result = self.robot_client.send_action(action, timeout=self.timeout_s, wait=self.wait)
        if self.wait:
            if not isinstance(result, Mapping):
                raise RuntimeError("send_action must return a mapping when wait=True")
            accepted = bool(result.get("accepted", False))
            if not accepted:
                raise RuntimeError(str(result.get("message", "robot rejected action")))
        return result

    def _spin_sleep(self, duration_s: float) -> None:
        deadline = time.monotonic() + float(duration_s)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self.robot_client.spin_once(min(0.01, remaining))
