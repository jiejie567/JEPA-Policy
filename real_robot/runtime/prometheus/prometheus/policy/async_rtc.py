from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from prometheus.policy.scheduler import ActionChunk, ActionContext


@dataclass(frozen=True)
class AsyncRTCConfig:
    enabled: bool
    initial_delay_steps: int | None
    latency_history: int
    join_timeout_s: float
    rtc: dict[str, Any]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> AsyncRTCConfig:
        config = dict(raw or {})
        rtc = dict(config.get("rtc") or {})
        initial_delay = config.get("initial_delay_steps")
        result = cls(
            enabled=bool(config.get("enabled", False)),
            initial_delay_steps=(
                None if initial_delay is None else int(initial_delay)
            ),
            latency_history=int(config.get("latency_history", 100)),
            join_timeout_s=float(config.get("join_timeout_s", 3.0)),
            rtc={
                "enabled": bool(rtc.get("enabled", True)),
                "execution_horizon": int(rtc.get("execution_horizon", 10)),
                "prefix_attention_schedule": str(
                    rtc.get("prefix_attention_schedule", "exp")
                ).strip().lower(),
                "max_guidance_weight": float(
                    rtc.get("max_guidance_weight", 5.0)
                ),
            },
        )
        result.validate()
        return result

    @property
    def execution_horizon(self) -> int:
        return int(self.rtc["execution_horizon"])

    def validate(self) -> None:
        if self.initial_delay_steps is not None and self.initial_delay_steps < 0:
            raise ValueError(
                "async_inference.initial_delay_steps must be non-negative"
            )
        if self.latency_history <= 0:
            raise ValueError("async_inference.latency_history must be positive")
        if self.join_timeout_s <= 0:
            raise ValueError("async_inference.join_timeout_s must be positive")
        if int(self.rtc["execution_horizon"]) <= 0:
            raise ValueError(
                "async_inference.rtc.execution_horizon must be positive"
            )
        if str(self.rtc["prefix_attention_schedule"]) not in {
            "zeros",
            "ones",
            "linear",
            "exp",
        }:
            raise ValueError(
                "async_inference.rtc.prefix_attention_schedule must be one of "
                "zeros, ones, linear, exp"
            )
        if float(self.rtc["max_guidance_weight"]) <= 0:
            raise ValueError(
                "async_inference.rtc.max_guidance_weight must be positive"
            )


@dataclass(frozen=True)
class RealtimePolicyOutput:
    chunk: ActionChunk
    model_actions_abs: np.ndarray
    model_actions_norm: np.ndarray

    def __post_init__(self) -> None:
        absolute = np.asarray(self.model_actions_abs, dtype=np.float32)
        normalized = np.asarray(self.model_actions_norm, dtype=np.float32)
        if absolute.ndim != 2 or normalized.ndim != 2:
            raise ValueError("model actions must be 2D")
        if absolute.shape != normalized.shape:
            raise ValueError(
                f"absolute/normalized shape mismatch: {absolute.shape} != {normalized.shape}"
            )
        if absolute.shape[0] != self.chunk.horizon:
            raise ValueError(
                f"model action horizon {absolute.shape[0]} != chunk horizon "
                f"{self.chunk.horizon}"
            )
        object.__setattr__(self, "model_actions_abs", absolute.copy())
        object.__setattr__(self, "model_actions_norm", normalized.copy())


@dataclass(frozen=True)
class InferenceSnapshot:
    previous_actions_abs: np.ndarray
    previous_deploy_actions: np.ndarray
    action_before_inference: np.ndarray | None
    generation: int
    action_index: int
    remaining_steps: int
    start_time_s: float


@dataclass(frozen=True)
class QueueReplacementResult:
    actual_delay_steps: int
    diagnostics: dict[str, Any]


class InferenceLatencyTracker:
    def __init__(self, history: int):
        if int(history) <= 0:
            raise ValueError("history must be positive")
        self._values: deque[float] = deque(maxlen=int(history))

    def add(self, elapsed_s: float) -> None:
        elapsed_s = float(elapsed_s)
        if elapsed_s < 0:
            raise ValueError("elapsed_s must be non-negative")
        self._values.append(elapsed_s)

    def estimated_steps(self, hz: float) -> int:
        if not self._values:
            return 0
        return latency_to_steps(max(self._values), hz)

    def status(self) -> dict[str, Any]:
        return {
            "samples": len(self._values),
            "max_s": 0.0 if not self._values else max(self._values),
            "latest_s": 0.0 if not self._values else self._values[-1],
        }


class InferenceDelayTracker:
    """Official RTC delay queue Q, measured in consumed controller steps."""

    def __init__(self, history: int):
        if int(history) <= 0:
            raise ValueError("history must be positive")
        self._values: deque[int] = deque(maxlen=int(history))

    def add(self, steps: int) -> None:
        steps = int(steps)
        if steps < 0:
            raise ValueError("steps must be non-negative")
        self._values.append(steps)

    def estimated_steps(self) -> int:
        return 0 if not self._values else max(self._values)

    def status(self) -> dict[str, Any]:
        return {
            "samples": len(self._values),
            "max_steps": 0 if not self._values else max(self._values),
            "latest_steps": 0 if not self._values else self._values[-1],
        }


def latency_to_steps(elapsed_s: float, hz: float) -> int:
    """ceil(inference_time / sample_time), where sample_time = 1 / hz."""
    elapsed_s = float(elapsed_s)
    hz = float(hz)
    if elapsed_s < 0:
        raise ValueError("elapsed_s must be non-negative")
    if hz <= 0:
        raise ValueError("hz must be positive")
    return int(math.ceil(elapsed_s / (1.0 / hz)))


@dataclass
class _QueueState:
    output: RealtimePolicyOutput
    generation: int
    next_index: int
    on_action: Callable[[np.ndarray, ActionContext], None] | None

    @property
    def remaining(self) -> int:
        return self.output.chunk.horizon - self.next_index


class AsyncChunkExecutor:
    """One-owner background action sender with atomic RTC queue replacement."""

    def __init__(self, scheduler: Any, *, config: AsyncRTCConfig):
        self.scheduler = scheduler
        self.config = config
        self._cv = threading.Condition()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._state: _QueueState | None = None
        self._error: BaseException | None = None
        self._started = False
        self._sent_action_count = 0
        self._replacement_count = 0
        self._underflow_count = 0

    @property
    def sent_action_count(self) -> int:
        with self._cv:
            return self._sent_action_count

    def start(self) -> None:
        with self._cv:
            if self._thread is not None:
                raise RuntimeError("AsyncChunkExecutor is already started")
            self._stop_event.clear()
            self._error = None
            self._started = False
            self._thread = threading.Thread(
                target=self._loop,
                name="prometheus_async_action_executor",
                daemon=False,
            )
            self._thread.start()

    def submit_initial(
        self,
        output: RealtimePolicyOutput,
        *,
        generation: int,
        on_action: Callable[[np.ndarray, ActionContext], None] | None,
    ) -> None:
        state = self._make_state(
            output,
            generation=generation,
            next_index=0,
            on_action=on_action,
        )
        with self._cv:
            self._raise_if_error_locked()
            if self._state is not None or self._started:
                raise RuntimeError("initial async chunk has already been submitted")
            self._state = state
            self._started = True
            self._cv.notify_all()

    def wait_replan_boundary(self) -> None:
        with self._cv:
            self._cv.wait_for(
                lambda: (
                    self._error is not None
                    or self._stop_event.is_set()
                    or (
                        self._state is not None
                        and self._state.next_index
                        >= self.config.execution_horizon
                    )
                )
            )
            self._raise_if_error_locked()
            if self._stop_event.is_set():
                raise RuntimeError("async executor stopped before replan boundary")

    def begin_inference(self) -> InferenceSnapshot:
        with self._cv:
            self._raise_if_error_locked()
            state = self._require_state_locked()
            previous_actions_abs = state.output.model_actions_abs[
                state.next_index :
            ].copy()
            previous_deploy_actions = state.output.chunk.actions[
                state.next_index :
            ].copy()
            if previous_actions_abs.shape[0] <= 0:
                raise RuntimeError("no previous actions remain for asynchronous inference")
            return InferenceSnapshot(
                previous_actions_abs=previous_actions_abs,
                previous_deploy_actions=previous_deploy_actions,
                action_before_inference=(
                    None
                    if state.next_index <= 0
                    else state.output.chunk.actions[state.next_index - 1].copy()
                ),
                generation=state.generation,
                action_index=state.next_index,
                remaining_steps=int(previous_actions_abs.shape[0]),
                start_time_s=time.perf_counter(),
            )

    def replace(
        self,
        output: RealtimePolicyOutput,
        *,
        generation: int,
        snapshot: InferenceSnapshot,
        estimated_delay_steps: int,
        on_action: Callable[[np.ndarray, ActionContext], None] | None,
    ) -> QueueReplacementResult:
        with self._cv:
            self._raise_if_error_locked()
            previous = self._require_state_locked()
            if previous.generation != snapshot.generation:
                raise RuntimeError(
                    "asynchronous generation changed during inference: "
                    f"{previous.generation} != {snapshot.generation}"
                )
            if previous.next_index < snapshot.action_index:
                raise RuntimeError(
                    "asynchronous action index moved backwards during inference"
                )
            actual_delay_steps = previous.next_index - snapshot.action_index
            if actual_delay_steps >= output.chunk.horizon:
                raise RuntimeError(
                    f"inference consumed {actual_delay_steps} actions and exhausted "
                    f"the new chunk horizon {output.chunk.horizon}"
                )
            self._state = self._make_state(
                output,
                generation=generation,
                next_index=actual_delay_steps,
                on_action=on_action,
            )
            self._replacement_count += 1
            self._cv.notify_all()

        try:
            diagnostics = _replacement_diagnostics(
                old_generation=snapshot.generation,
                new_generation=int(generation),
                snapshot=snapshot,
                output=output,
                estimated_delay_steps=int(estimated_delay_steps),
                actual_delay_steps=int(actual_delay_steps),
            )
        except Exception as exc:
            diagnostics = {
                "time_ns": time.time_ns(),
                "old_generation": int(snapshot.generation),
                "new_generation": int(generation),
                "inference_start_action_index": int(snapshot.action_index),
                "estimated_delay_steps": int(estimated_delay_steps),
                "actual_delay_steps": int(actual_delay_steps),
                "diagnostic_error": repr(exc),
            }
        return QueueReplacementResult(
            actual_delay_steps=int(actual_delay_steps),
            diagnostics=diagnostics,
        )

    def stop(self) -> None:
        self._stop_event.set()
        with self._cv:
            self._cv.notify_all()
        thread = self._thread
        if thread is None:
            return
        thread.join(timeout=self.config.join_timeout_s)
        if thread.is_alive():
            raise TimeoutError(
                "AsyncChunkExecutor did not stop within "
                f"{self.config.join_timeout_s:.3f}s"
            )
        self._thread = None
        with self._cv:
            self._raise_if_error_locked()

    def status(self) -> dict[str, Any]:
        with self._cv:
            state = self._state
            thread = self._thread
            return {
                "enabled": True,
                "thread_alive": bool(thread is not None and thread.is_alive()),
                "generation": None if state is None else state.generation,
                "pending_steps": 0 if state is None else state.remaining,
                "sent_action_count": self._sent_action_count,
                "replacement_count": self._replacement_count,
                "underflow_count": self._underflow_count,
                "error": None if self._error is None else repr(self._error),
            }

    def _make_state(
        self,
        output: RealtimePolicyOutput,
        *,
        generation: int,
        next_index: int,
        on_action: Callable[[np.ndarray, ActionContext], None] | None,
    ) -> _QueueState:
        horizon = output.chunk.horizon
        execution_horizon = self.config.execution_horizon
        if execution_horizon >= horizon:
            raise ValueError(
                "async_inference.rtc.execution_horizon must be smaller than "
                f"the executable chunk horizon ({execution_horizon} >= {horizon})"
            )
        if next_index < 0 or next_index >= horizon:
            raise ValueError(
                f"next_index must be in [0, {horizon}), got {next_index}"
            )
        return _QueueState(
            output=output,
            generation=int(generation),
            next_index=int(next_index),
            on_action=on_action,
        )

    def _loop(self) -> None:
        next_time = time.perf_counter()
        try:
            while not self._stop_event.is_set():
                with self._cv:
                    self._cv.wait_for(
                        lambda: self._stop_event.is_set() or self._state is not None
                    )
                    if self._stop_event.is_set():
                        return

                delay = next_time - time.perf_counter()
                if delay > 0 and self._stop_event.wait(delay):
                    return
                with self._cv:
                    state = self._require_state_locked()
                    if state.remaining <= 0:
                        self._underflow_count += 1
                        raise RuntimeError(
                            "asynchronous action queue underflowed while inference "
                            "was still running"
                        )
                    chunk = state.output.chunk
                    action_index = state.next_index
                    action = chunk.actions[action_index].copy()
                    on_action = state.on_action
                    state.next_index += 1
                    self._cv.notify_all()

                single = ActionChunk(
                    actions=action.reshape(1, -1),
                    action_space=chunk.action_space,
                    hz=chunk.hz,
                    metadata=dict(chunk.metadata),
                )

                def callback(
                    sent_action: np.ndarray,
                    scheduler_context: ActionContext,
                ) -> None:
                    if on_action is None:
                        return
                    on_action(
                        sent_action,
                        ActionContext(
                            chunk=chunk,
                            index=action_index,
                            previous_action=scheduler_context.previous_action,
                        ),
                    )

                self.scheduler.execute(
                    single,
                    steps=1,
                    on_action=callback if on_action is not None else None,
                )
                with self._cv:
                    self._sent_action_count += 1
                    self._cv.notify_all()

                next_time += chunk.dt
                now = time.perf_counter()
                if next_time < now:
                    next_time = now + chunk.dt
        except BaseException as exc:
            with self._cv:
                self._error = exc
                self._cv.notify_all()
            self._stop_event.set()

    def _require_state_locked(self) -> _QueueState:
        if self._state is None:
            raise RuntimeError("no asynchronous action chunk is available")
        return self._state

    def _raise_if_error_locked(self) -> None:
        if self._error is not None:
            raise RuntimeError("AsyncChunkExecutor failed") from self._error


def _replacement_diagnostics(
    *,
    old_generation: int,
    new_generation: int,
    snapshot: InferenceSnapshot,
    output: RealtimePolicyOutput,
    estimated_delay_steps: int,
    actual_delay_steps: int,
) -> dict[str, Any]:
    overlap_steps = min(
        snapshot.previous_deploy_actions.shape[0],
        output.chunk.horizon,
    )
    frozen_prefix_steps = min(max(estimated_delay_steps, 0), overlap_steps)
    deploy_prefix_error = (
        output.chunk.actions[:overlap_steps]
        - snapshot.previous_deploy_actions[:overlap_steps]
    )
    model_abs_prefix_error = (
        output.model_actions_abs[:overlap_steps]
        - snapshot.previous_actions_abs[:overlap_steps]
    )

    if actual_delay_steps > 0:
        previous_action = _row_or_none(
            snapshot.previous_deploy_actions,
            actual_delay_steps - 1,
        )
    else:
        previous_action = (
            None
            if snapshot.action_before_inference is None
            else snapshot.action_before_inference.copy()
        )
    old_expected_action = _row_or_none(
        snapshot.previous_deploy_actions,
        actual_delay_steps,
    )
    new_selected_action = _row_or_none(
        output.chunk.actions,
        actual_delay_steps,
    )
    old_continuation_delta = _difference_or_none(
        old_expected_action,
        previous_action,
    )
    replacement_boundary_delta = _difference_or_none(
        new_selected_action,
        previous_action,
    )
    replacement_vs_old_expected = _difference_or_none(
        new_selected_action,
        old_expected_action,
    )

    return {
        "time_ns": time.time_ns(),
        "old_generation": int(old_generation),
        "new_generation": int(new_generation),
        "action_space": output.chunk.action_space,
        "action_hz": float(output.chunk.hz),
        "sample_time_s": float(output.chunk.dt),
        "inference_start_action_index": int(snapshot.action_index),
        "previous_remaining_steps": int(snapshot.remaining_steps),
        "new_chunk_horizon": int(output.chunk.horizon),
        "estimated_delay_steps": int(estimated_delay_steps),
        "actual_delay_steps": int(actual_delay_steps),
        "old_previous_action_index": int(
            snapshot.action_index + actual_delay_steps - 1
        ),
        "old_expected_action_index": int(
            snapshot.action_index + actual_delay_steps
        ),
        "selected_new_action_index": int(actual_delay_steps),
        "overlap_steps": int(overlap_steps),
        "rtc_frozen_prefix_steps": int(frozen_prefix_steps),
        "boundary": {
            "previous_action": _tolist_or_none(previous_action),
            "old_expected_action": _tolist_or_none(old_expected_action),
            "new_selected_action": _tolist_or_none(new_selected_action),
            "old_continuation_delta": _tolist_or_none(
                old_continuation_delta
            ),
            "replacement_boundary_delta": _tolist_or_none(
                replacement_boundary_delta
            ),
            "replacement_vs_old_expected": _tolist_or_none(
                replacement_vs_old_expected
            ),
            "old_continuation": _vector_stats(old_continuation_delta),
            "replacement_boundary": _vector_stats(
                replacement_boundary_delta
            ),
            "replacement_vs_old_expected_stats": _vector_stats(
                replacement_vs_old_expected
            ),
        },
        "deploy_prefix_error": {
            "full_overlap": _matrix_stats(deploy_prefix_error),
            "rtc_frozen_prefix": _matrix_stats(
                deploy_prefix_error[:frozen_prefix_steps]
            ),
        },
        "model_abs_prefix_error": {
            "full_overlap": _matrix_stats(model_abs_prefix_error),
            "rtc_frozen_prefix": _matrix_stats(
                model_abs_prefix_error[:frozen_prefix_steps]
            ),
        },
    }


def _row_or_none(values: np.ndarray, index: int) -> np.ndarray | None:
    if index < 0 or index >= values.shape[0]:
        return None
    return values[index].copy()


def _difference_or_none(
    left: np.ndarray | None,
    right: np.ndarray | None,
) -> np.ndarray | None:
    if left is None or right is None:
        return None
    return left - right


def _tolist_or_none(value: np.ndarray | None) -> list[float] | None:
    if value is None:
        return None
    return [float(item) for item in value.tolist()]


def _vector_stats(value: np.ndarray | None) -> dict[str, Any] | None:
    if value is None:
        return None
    absolute = np.abs(np.asarray(value, dtype=np.float64))
    return {
        "max_abs": float(np.max(absolute)),
        "l2": float(np.linalg.norm(value)),
        "max_abs_joint": int(np.argmax(absolute)),
    }


def _matrix_stats(value: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape[0] == 0:
        return {
            "steps": 0,
            "max_abs": None,
            "p99_abs": None,
            "mean_abs": None,
            "max_step_l2": None,
            "max_abs_per_joint": [],
        }
    absolute = np.abs(matrix)
    return {
        "steps": int(matrix.shape[0]),
        "max_abs": float(np.max(absolute)),
        "p99_abs": float(np.percentile(absolute, 99)),
        "mean_abs": float(np.mean(absolute)),
        "max_step_l2": float(np.max(np.linalg.norm(matrix, axis=1))),
        "max_abs_per_joint": [
            float(item) for item in np.max(absolute, axis=0).tolist()
        ],
    }
