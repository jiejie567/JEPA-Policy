from __future__ import annotations

import json
import os
import time
import threading
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from prometheus.policy.async_rtc import (
    AsyncChunkExecutor,
    InferenceDelayTracker,
    InferenceLatencyTracker,
    latency_to_steps,
)
from prometheus.policy.scheduler import SafetyGuardError
from prometheus.sessions.policy import PolicySession
from prometheus.utils.workflow import (
    build_data_session,
    build_hardware_session,
    config_dict,
    instantiate_config,
    prepare_run,
    section,
    write_json,
)


def run_from_config(cfg: Any) -> int:
    # Step 1: Resolve Hydra config into plain runtime sections.
    data = config_dict(cfg, "rollout_sync")
    run_cfg = section(data, "run")
    robot_cfg = section(data, "robot")
    task_cfg = section(data, "task")
    hardware_cfg = section(data, "hardware")
    data_cfg = section(data, "data")
    policy_cfg = section(data, "policy")
    policy_session_cfg = section(data, "policy_session")
    scheduler_cfg = section(data, "scheduler")
    workflow_cfg = section(data, "workflow")
    run = prepare_run(run_cfg)
    run.runtime_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PROMETHEUS_CAMERA_TIMING_TRACE"] = str(run.runtime_dir / "camera_publish_timing.jsonl")

    # Step 2: Build workflow-owned sessions and policy object.
    hardware = build_hardware_session(run=run, hardware_cfg=hardware_cfg, robot_cfg=robot_cfg)
    data_session = build_data_session(data_cfg=data_cfg, hardware=hardware)
    policy = instantiate_config(policy_cfg, "policy")
    policy_session = PolicySession(
        policy=policy,
        name=str(policy_session_cfg["name"]),
        mode=str(policy_session_cfg["mode"]),
    )
    robot_client = None
    scheduler = None
    async_status: dict[str, Any] = {"enabled": False}

    status = "running"
    stop_reason = "not_started"
    wait_ready_timeout_s = float(run_cfg["wait_ready_timeout_s"])
    steps = int(run_cfg["steps"])
    if steps <= 0:
        raise ValueError("run.steps must be positive")
    stop_on_done = bool(workflow_cfg["stop_on_done"])
    action_steps_per_chunk = workflow_cfg.get("action_steps_per_chunk")
    if action_steps_per_chunk is not None:
        action_steps_per_chunk = int(action_steps_per_chunk)
        if action_steps_per_chunk <= 0:
            raise ValueError("workflow.action_steps_per_chunk must be positive")
    post_chunk_observation_delay_s = float(
        workflow_cfg.get("post_chunk_observation_delay_s", 0.0)
    )
    if post_chunk_observation_delay_s < 0:
        raise ValueError("workflow.post_chunk_observation_delay_s must be non-negative")
    chunks = 0
    actions_sent = 0
    reset_home_succeeded = False
    guard_rejection_reset_attempted = False
    interrupt_reset_attempted = False
    error_reset_attempted = False
    startup_home_checked = False
    startup_home_reset_attempted = False
    startup_home_before: list[float] = []
    startup_home_after: list[float] = []
    startup_action_guard_checks: list[dict[str, Any]] = []
    observation_barrier_targets_ns: list[int] = []
    failure = ""
    return_code = 0
    try:
        # Step 3: Start hardware and policy resources, then wait for readiness.
        hardware.start()
        data_session.start()
        policy_session.start()
        hardware.wait_ready(wait_ready_timeout_s)
        # Bind as soon as the owner is ready. Input-alignment failures happen
        # before scheduler construction and still need an ordered reset.
        robot_client = hardware.robot_client
        data_session.wait_ready(wait_ready_timeout_s)
        policy_session.wait_ready(wait_ready_timeout_s)
        if bool(workflow_cfg.get("ensure_home_before_rollout", False)):
            startup_home_checked = True
            (
                startup_home_reset_attempted,
                startup_home_before,
                startup_home_after,
            ) = _ensure_startup_home(
                robot_client,
                data_session,
                state_stream=str(
                    workflow_cfg.get("startup_home_state_stream", "robot_state")
                ),
                target=workflow_cfg.get("startup_home_target"),
                tolerance=workflow_cfg.get("startup_home_tolerance"),
                timeout_s=float(workflow_cfg.get("reset_home_timeout_s", 30.0)),
                settle_s=float(workflow_cfg.get("startup_home_settle_s", 0.2)),
            )
            if startup_home_reset_attempted:
                reset_home_succeeded = True
        _wait_policy_inputs_ready(data_session, policy_session.policy, wait_ready_timeout_s)

        # Step 4: Construct the scheduler only after home/input checks pass.
        scheduler = instantiate_config(scheduler_cfg, "scheduler", robot_client=robot_client)
        scheduler.reset()
        if data_session.recording_enabled:
            recording_cfg = section(data_cfg, "recording")
            data_session.start_episode(
                str(recording_cfg["episode_id"]),
                metadata={"workflow": "rollout_sync", "run_id": run.run_id, "task": task_cfg},
            )
            # Starting three independent NVENC writers can take longer than
            # ROS/session readiness on the first run. Keep this as a separate
            # barrier: no action is sent until all required streams have
            # completed at least one write.
            _wait_recording_ready(data_session, max(wait_ready_timeout_s, 30.0))

        # Step 5: Keep the original synchronous path completely separate.
        if bool(getattr(policy_session.policy, "async_inference_enabled", False)):
            async_status = {"enabled": True}
            (
                chunks,
                actions_sent,
                stop_reason,
                async_status,
            ) = _run_async_rollout(
                steps=steps,
                stop_on_done=stop_on_done,
                policy=policy_session.policy,
                data_session=data_session,
                scheduler=scheduler,
                status_out=async_status,
                replacement_diagnostics_path=(
                    run.runtime_dir / "rtc_queue_replacements.jsonl"
                ),
            )
        else:
            while chunks < steps:
                robot_client.spin_once(0.0)
                chunk = policy_session.policy.infer(data_session)
                guard_check = _check_startup_action_semantics(
                    chunk,
                    workflow_cfg.get("startup_action_guard", {}),
                    chunk_index=chunks,
                )
                if guard_check is not None:
                    startup_action_guard_checks.append(guard_check)
                    if bool(guard_check["rejected"]):
                        raise SafetyGuardError(
                            "startup_action_guard rejected policy chunk "
                            f"{chunks}; inactive_dimensions="
                            f"{guard_check['inactive_dimensions']} "
                            f"max_displacement={guard_check['max_displacement']:.6f} "
                            f"limit={guard_check['limit']:.6f} "
                            "first_action_exceeded_dimensions="
                            f"{guard_check['first_action_exceeded_dimensions']}"
                        )
                chunk_recorded = data_session.record_policy_action_chunk(
                    chunk,
                    metadata={"chunk_index": chunks},
                )
                if data_session.recording_enabled and not chunk_recorded:
                    raise RuntimeError(
                        "failed to enqueue policy action chunk for recording"
                    )
                actions_sent += scheduler.execute(
                    chunk,
                    steps=action_steps_per_chunk,
                    on_action=_record_action(data_session, chunks) if data_session.recording_enabled else None,
                )
                chunks += 1
                if stop_on_done and bool(chunk.metadata.get("done", False)):
                    stop_reason = "policy_done"
                    break
                if chunks < steps:
                    require_observation_after = getattr(
                        policy_session.policy,
                        "require_observation_after",
                        None,
                    )
                    if callable(require_observation_after):
                        # Eight commands at 10 Hz are issued at t=0...0.7 s;
                        # the final command must own its complete 0.1 s control
                        # interval before the observation for the next chunk.
                        # DataSession then waits for an actually decoded camera
                        # frame at or after this wall-clock/recv-time boundary.
                        target_stamp_ns = time.time_ns() + int(
                            round(float(chunk.dt) * 1_000_000_000)
                        )
                        require_observation_after(target_stamp_ns)
                        observation_barrier_targets_ns.append(target_stamp_ns)
                if chunks < steps and post_chunk_observation_delay_s > 0:
                    scheduler.idle(post_chunk_observation_delay_s)
            else:
                stop_reason = "steps_completed"

        if bool(workflow_cfg.get("reset_home_on_success", False)):
            _reset_robot_home(
                robot_client,
                timeout_s=float(workflow_cfg.get("reset_home_timeout_s", 30.0)),
            )
            reset_home_succeeded = True
            print("[rollout_sync] reset_home_on_success completed", flush=True)

        status = "completed"
        return 0
    except KeyboardInterrupt:
        status = "interrupted"
        stop_reason = "keyboard_interrupt"
        return_code = 130
        if bool(workflow_cfg.get("reset_home_on_interrupt", False)):
            interrupt_reset_attempted = True
            try:
                print(
                    "[rollout_sync] Ctrl+C received; rollout stopped. "
                    "Starting reset_home before damping",
                    flush=True,
                )
                _reset_robot_home(
                    robot_client,
                    timeout_s=float(
                        workflow_cfg.get("reset_home_timeout_s", 30.0)
                    ),
                )
                reset_home_succeeded = True
                print(
                    "[rollout_sync] reset_home_on_interrupt completed; "
                    "entering damping cleanup",
                    flush=True,
                )
            except KeyboardInterrupt:
                failure = (
                    "reset_home_on_interrupt aborted by a second Ctrl+C; "
                    "entering damping cleanup"
                )
                print(f"[rollout_sync] {failure}", flush=True)
            except Exception as reset_exc:
                failure = (
                    "reset_home_on_interrupt failed: "
                    f"{type(reset_exc).__name__}: {reset_exc}; "
                    "entering damping cleanup"
                )
                print(f"[rollout_sync] {failure}", flush=True)
        return return_code
    except Exception as exc:
        status = "error"
        stop_reason = "exception"
        failure = f"{type(exc).__name__}: {exc}"
        reset_handled = False
        if isinstance(exc, SafetyGuardError) and bool(
            workflow_cfg.get("reset_home_on_guard_rejection", False)
        ):
            reset_handled = True
            guard_rejection_reset_attempted = True
            try:
                print(
                    "[rollout_sync] safety guard rejected a command; "
                    "starting reset_home before damping",
                    flush=True,
                )
                _reset_robot_home(
                    robot_client,
                    timeout_s=float(
                        workflow_cfg.get("reset_home_timeout_s", 30.0)
                    ),
                )
                reset_home_succeeded = True
                print(
                    "[rollout_sync] reset_home_on_guard_rejection completed",
                    flush=True,
                )
            except Exception as reset_exc:
                failure += (
                    "; guard rejection reset failed: "
                    f"{type(reset_exc).__name__}: {reset_exc}"
                )
                print(
                    "[rollout_sync] guard rejection reset failed; "
                    "hardware cleanup will continue: "
                    f"{type(reset_exc).__name__}: {reset_exc}",
                    flush=True,
                )
        if (
            not reset_handled
            and bool(workflow_cfg.get("reset_home_on_error", False))
            and robot_client is not None
        ):
            error_reset_attempted = True
            try:
                print(
                    "[rollout_sync] rollout error; starting reset_home before damping",
                    flush=True,
                )
                _reset_robot_home(
                    robot_client,
                    timeout_s=float(
                        workflow_cfg.get("reset_home_timeout_s", 30.0)
                    ),
                )
                reset_home_succeeded = True
                print(
                    "[rollout_sync] reset_home_on_error completed; "
                    "entering damping cleanup",
                    flush=True,
                )
            except KeyboardInterrupt:
                failure += (
                    "; reset_home_on_error aborted by Ctrl+C; "
                    "entering damping cleanup"
                )
                print(f"[rollout_sync] {failure}", flush=True)
            except Exception as reset_exc:
                failure += (
                    "; reset_home_on_error failed: "
                    f"{type(reset_exc).__name__}: {reset_exc}"
                )
                print(
                    "[rollout_sync] error reset failed; hardware cleanup will continue: "
                    f"{type(reset_exc).__name__}: {reset_exc}",
                    flush=True,
                )
        print(f"[rollout_sync] fatal: {failure}", flush=True)
        traceback.print_exc()
        return_code = 1
        raise
    finally:
        if bool(async_status.get("enabled", False)):
            actions_sent = max(
                actions_sent,
                int(async_status.get("sent_action_count", 0)),
            )
            generation = async_status.get("generation")
            if generation is not None:
                chunks = max(chunks, int(generation) + 1)
        if scheduler is not None:
            actions_sent = max(actions_sent, len(scheduler.history))
        # Step 6: Freeze data ingress, then let robot cleanup/reset and data finalization overlap.
        try:
            data_session.stop_ingress()
        except Exception:
            pass
        if data_session.recording_enabled:
            data_session.stop_episode(accept=status == "completed")
        try:
            policy_session.stop()
        finally:
            policy_session.close()
        _stop_data_and_hardware(data_session, hardware)
        write_json(
            run.manifest_path,
            {
                "run_id": run.run_id,
                "status": status,
                "stop_reason": stop_reason,
                "failure": failure,
                "chunks": chunks,
                # Scheduler.execute counts validated/processed actions even in
                # dry-run. Keep the two meanings explicit in the manifest.
                "actions_processed": actions_sent,
                "actions_sent": (
                    0
                    if scheduler is None or scheduler.dry_run
                    else actions_sent
                ),
                "time_ns": time.time_ns(),
                "output_dir": str(run.output_dir),
                "runtime_dir": str(run.runtime_dir),
                "task": task_cfg,
                "hardware": hardware.status().as_dict(),
                "data": data_session.status().as_dict(),
                "policy": policy_session.status().as_dict(),
                "robot_client": None if robot_client is None else robot_client.status(),
                "scheduler": {
                    "dry_run": None if scheduler is None else scheduler.dry_run,
                    "history": 0 if scheduler is None else len(scheduler.history),
                    "action_steps_per_chunk": action_steps_per_chunk,
                    "post_chunk_observation_delay_s": post_chunk_observation_delay_s,
                    "observation_barrier_count": len(
                        observation_barrier_targets_ns
                    ),
                    "observation_barrier_targets_ns": observation_barrier_targets_ns,
                    "filters": [] if scheduler is None else [item.name for item in scheduler.filters],
                },
                "reset_home_on_success": bool(
                    workflow_cfg.get("reset_home_on_success", False)
                ),
                "reset_home_on_interrupt": bool(
                    workflow_cfg.get("reset_home_on_interrupt", False)
                ),
                "reset_home_succeeded": reset_home_succeeded,
                "guard_rejection_reset_attempted": guard_rejection_reset_attempted,
                "interrupt_reset_attempted": interrupt_reset_attempted,
                "error_reset_attempted": error_reset_attempted,
                "startup_home_checked": startup_home_checked,
                "startup_home_reset_attempted": startup_home_reset_attempted,
                "startup_home_before": startup_home_before,
                "startup_home_after": startup_home_after,
                "startup_action_guard_checks": startup_action_guard_checks,
                "async_inference": async_status,
            },
        )
        if return_code:
            return return_code


def main() -> None:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../configs", config_name="rollout_sync")
    def _main(cfg: DictConfig) -> None:
        raise SystemExit(run_from_config(cfg))

    _main()


def _record_action(data_session: Any, chunk_index: int):
    def _callback(action, context) -> None:
        ok = data_session.record_action(
            action,
            action_space=context.action_space,
            hz=context.chunk.hz,
            metadata={"chunk_index": chunk_index, "action_index": context.index},
        )
        if not ok:
            raise RuntimeError("failed to enqueue executed action for recording")

    return _callback


def _check_startup_action_semantics(
    chunk: Any,
    guard_cfg: Any,
    *,
    chunk_index: int,
) -> dict[str, Any] | None:
    """Reject a startup plan that moves an arm inactive in demonstrations."""
    cfg = dict(guard_cfg or {})
    if not bool(cfg.get("enabled", False)):
        return None
    chunk_count = int(cfg.get("chunk_count", 1))
    if chunk_count <= 0:
        raise ValueError("startup_action_guard.chunk_count must be positive")
    if int(chunk_index) >= chunk_count:
        return None
    inactive_dimensions = [int(value) for value in cfg.get("inactive_dimensions", [])]
    first_action_max_delta = cfg.get("first_action_max_delta")
    if not inactive_dimensions and first_action_max_delta is None:
        raise ValueError(
            "startup_action_guard requires inactive_dimensions or "
            "first_action_max_delta"
        )
    if inactive_dimensions and (
        min(inactive_dimensions) < 0
        or max(inactive_dimensions) >= chunk.action_dim
    ):
        raise ValueError(
            "startup_action_guard.inactive_dimensions are outside action shape"
        )
    limit = float(cfg.get("max_displacement", 0.0))
    if limit < 0:
        raise ValueError("startup_action_guard.max_displacement must be non-negative")
    current = np.asarray(
        chunk.metadata.get("current_action"), dtype=np.float32
    ).reshape(-1)
    if current.shape != (chunk.action_dim,) or not np.all(np.isfinite(current)):
        raise ValueError(
            "startup_action_guard requires finite metadata current_action"
        )
    if inactive_dimensions:
        selected = np.asarray(inactive_dimensions, dtype=np.int64)
        displacement = np.abs(chunk.actions[:, selected] - current[selected])
        max_displacement = float(np.max(displacement))
    else:
        max_displacement = 0.0
    inactive_rejected = bool(max_displacement > limit + 1e-6)
    first_action_exceeded_dimensions: list[int] = []
    if first_action_max_delta is not None:
        first_limit = np.asarray(first_action_max_delta, dtype=np.float32)
        try:
            first_limit = np.broadcast_to(first_limit, (chunk.action_dim,))
        except ValueError as exc:
            raise ValueError(
                "startup_action_guard.first_action_max_delta must be scalar "
                f"or length {chunk.action_dim}"
            ) from exc
        if not np.all(np.isfinite(first_limit)) or np.any(first_limit <= 0):
            raise ValueError(
                "startup_action_guard.first_action_max_delta must be finite and positive"
            )
        first_action_dimensions = [
            int(value) for value in cfg.get("first_action_dimensions", [])
        ]
        if not first_action_dimensions:
            first_action_dimensions = list(range(chunk.action_dim))
        if (
            min(first_action_dimensions) < 0
            or max(first_action_dimensions) >= chunk.action_dim
        ):
            raise ValueError(
                "startup_action_guard.first_action_dimensions are outside action shape"
            )
        first_delta = np.abs(chunk.actions[0] - current)
        first_selected = np.asarray(first_action_dimensions, dtype=np.int64)
        exceeded_selected = first_delta[first_selected] > np.nextafter(
            first_limit[first_selected],
            np.float32(np.inf),
        )
        first_action_exceeded_dimensions = [
            int(index) for index in first_selected[exceeded_selected]
        ]
    rejected = bool(inactive_rejected or first_action_exceeded_dimensions)
    result = {
        "chunk_index": int(chunk_index),
        "inactive_dimensions": inactive_dimensions,
        "max_displacement": max_displacement,
        "limit": limit,
        "first_action_exceeded_dimensions": first_action_exceeded_dimensions,
        "rejected": rejected,
    }
    print(
        "[rollout_sync] startup action semantics: "
        + json.dumps(result, separators=(",", ":")),
        flush=True,
    )
    return result


def _wait_recording_ready(data_session: Any, timeout_s: float) -> None:
    stream_requirements = (
        ("base_0_color", "camera.base_0_rgb"),
        ("left_wrist_0_color", "camera.left_wrist_0_rgb"),
        ("right_wrist_0_color", "camera.right_wrist_0_rgb"),
        ("robot_state", "robot.robot_state"),
    )
    available_streams = set(getattr(data_session, "streams", {}))
    required_counts = tuple(
        count_name
        for stream_name, count_name in stream_requirements
        if stream_name in available_streams
    )
    if not required_counts:
        return
    deadline = time.monotonic() + float(timeout_s)
    last_counts: dict[str, int] = {}
    while time.monotonic() < deadline:
        recorder = getattr(data_session, "recorder", None)
        if recorder is not None and callable(getattr(recorder, "details", None)):
            # BaseSession.status() contains the details snapshot from the last
            # lifecycle transition. Recorder details are live and therefore
            # must be queried directly while waiting for its worker threads.
            details = dict(recorder.details())
        else:
            status = data_session.status()
            details = dict(status.details or {})
        failure = str(details.get("failure", "")).strip()
        if failure:
            raise RuntimeError(
                f"data recorder failed before action execution: {failure}"
            )
        last_counts = {
            str(name): int(value)
            for name, value in dict(details.get("record_counts", {})).items()
        }
        if all(last_counts.get(name, 0) >= 1 for name in required_counts):
            print(
                "[rollout_sync] recorder ready before action execution: "
                + ", ".join(
                    f"{name}={last_counts[name]}" for name in required_counts
                ),
                flush=True,
            )
            return
        time.sleep(0.02)
    raise TimeoutError(
        "timed out waiting for recorder readiness before action execution: "
        f"required={required_counts} counts={last_counts}"
    )


def _run_async_rollout(
    *,
    steps: int,
    stop_on_done: bool,
    policy: Any,
    data_session: Any,
    scheduler: Any,
    status_out: dict[str, Any] | None = None,
    replacement_diagnostics_path: str | os.PathLike[str] | None = None,
) -> tuple[int, int, str, dict[str, Any]]:
    config = getattr(policy, "async_config", None)
    infer_realtime = getattr(policy, "infer_realtime", None)
    if config is None or not bool(config.enabled) or not callable(infer_realtime):
        raise TypeError(
            "async rollout requires a policy with enabled async_config "
            "and infer_realtime()"
        )

    latency = InferenceLatencyTracker(config.latency_history)
    delay = InferenceDelayTracker(config.latency_history)
    replacement_diagnostics = _ReplacementDiagnosticLog(
        replacement_diagnostics_path
    )
    executor = AsyncChunkExecutor(scheduler, config=config)
    chunks = 0
    stop_reason = "steps_completed"
    executor.start()
    try:
        first_start = time.perf_counter()
        output = infer_realtime(
            data_session,
            previous_actions_abs=None,
            inference_delay_steps=0,
        )
        first_elapsed_s = time.perf_counter() - first_start
        latency.add(first_elapsed_s)
        measured_initial_delay_steps = latency_to_steps(
            first_elapsed_s,
            output.chunk.hz,
        )
        initial_delay_steps = max(
            measured_initial_delay_steps,
            (
                measured_initial_delay_steps
                if config.initial_delay_steps is None
                else int(config.initial_delay_steps)
            ),
        )
        delay.add(initial_delay_steps)
        if initial_delay_steps > config.execution_horizon:
            raise RuntimeError(
                f"initial inference delay {initial_delay_steps} exceeds RTC "
                f"execution_horizon {config.execution_horizon}; official RTC "
                "requires inference_delay <= execution_horizon"
            )
        if config.execution_horizon > output.chunk.horizon - initial_delay_steps:
            raise RuntimeError(
                f"RTC execution_horizon {config.execution_horizon} exceeds "
                f"chunk_horizon - inference_delay "
                f"({output.chunk.horizon} - {initial_delay_steps}); official RTC "
                "requires execution_horizon <= H - inference_delay"
            )
        _record_async_policy_chunk(
            data_session,
            output.chunk,
            chunk_index=chunks,
            estimated_delay_steps=initial_delay_steps,
            actual_delay_steps=0,
            inference_elapsed_s=first_elapsed_s,
            inference_start_action_index=0,
            prefix_attention_horizon=0,
        )
        executor.submit_initial(
            output,
            generation=chunks,
            on_action=(
                _record_action(data_session, chunks)
                if data_session.recording_enabled
                else None
            ),
        )
        chunks += 1
        if stop_on_done and bool(output.chunk.metadata.get("done", False)):
            stop_reason = "policy_done"

        while chunks < steps and stop_reason != "policy_done":
            executor.wait_replan_boundary()
            snapshot = executor.begin_inference()
            estimated_delay_steps = delay.estimated_steps()
            if estimated_delay_steps > snapshot.action_index:
                raise RuntimeError(
                    f"estimated inference delay {estimated_delay_steps} exceeds "
                    f"the current execution horizon {snapshot.action_index}"
                )
            if estimated_delay_steps > snapshot.remaining_steps:
                raise RuntimeError(
                    f"estimated inference delay {estimated_delay_steps} exceeds "
                    f"the available previous-action prefix {snapshot.remaining_steps}"
                )
            output = infer_realtime(
                data_session,
                previous_actions_abs=snapshot.previous_actions_abs,
                inference_delay_steps=estimated_delay_steps,
                prefix_attention_horizon=snapshot.remaining_steps,
            )
            elapsed_s = time.perf_counter() - snapshot.start_time_s
            latency.add(elapsed_s)
            replacement = executor.replace(
                output,
                generation=chunks,
                snapshot=snapshot,
                estimated_delay_steps=estimated_delay_steps,
                on_action=(
                    _record_action(data_session, chunks)
                    if data_session.recording_enabled
                    else None
                ),
            )
            actual_delay_steps = replacement.actual_delay_steps
            replacement_diagnostics.append(
                {
                    **replacement.diagnostics,
                    "inference_elapsed_s": float(elapsed_s),
                }
            )
            delay.add(actual_delay_steps)
            _record_async_policy_chunk(
                data_session,
                output.chunk,
                chunk_index=chunks,
                estimated_delay_steps=estimated_delay_steps,
                actual_delay_steps=actual_delay_steps,
                inference_elapsed_s=elapsed_s,
                inference_start_action_index=snapshot.action_index,
                prefix_attention_horizon=snapshot.remaining_steps,
            )
            chunks += 1
            if stop_on_done and bool(output.chunk.metadata.get("done", False)):
                stop_reason = "policy_done"

        # Give the final generation the same configured execution horizon.
        executor.wait_replan_boundary()
    finally:
        try:
            executor.stop()
        finally:
            current_status = _async_status(
                executor,
                config=config,
                latency=latency,
                delay=delay,
                replacement_diagnostics=replacement_diagnostics,
            )
            if status_out is not None:
                status_out.clear()
                status_out.update(current_status)

    status = _async_status(
        executor,
        config=config,
        latency=latency,
        delay=delay,
        replacement_diagnostics=replacement_diagnostics,
    )
    return chunks, executor.sent_action_count, stop_reason, status


def _record_async_policy_chunk(
    data_session: Any,
    chunk: Any,
    *,
    chunk_index: int,
    estimated_delay_steps: int,
    actual_delay_steps: int,
    inference_elapsed_s: float,
    inference_start_action_index: int,
    prefix_attention_horizon: int,
) -> None:
    data_session.record_policy_action_chunk(
        chunk,
        metadata={
            "chunk_index": int(chunk_index),
            "async_inference": True,
            "estimated_delay_steps": int(estimated_delay_steps),
            "actual_delay_steps": int(actual_delay_steps),
            "inference_elapsed_s": float(inference_elapsed_s),
            "inference_start_action_index": int(inference_start_action_index),
            "prefix_attention_horizon": int(prefix_attention_horizon),
        },
    )


def _async_status(
    executor: AsyncChunkExecutor,
    *,
    config: Any,
    latency: InferenceLatencyTracker,
    delay: InferenceDelayTracker,
    replacement_diagnostics: Any,
) -> dict[str, Any]:
    status = executor.status()
    status["config"] = {
        "initial_delay_steps": config.initial_delay_steps,
        "latency_history": config.latency_history,
        "join_timeout_s": config.join_timeout_s,
        "rtc": dict(config.rtc),
    }
    status["latency"] = latency.status()
    status["delay"] = delay.status()
    status["replacement_diagnostics"] = replacement_diagnostics.status()
    return status


class _ReplacementDiagnosticLog:
    def __init__(self, path: str | os.PathLike[str] | None):
        self.path = None if path is None else Path(path)
        self.records_written = 0
        self.write_errors = 0
        self.last_error: str | None = None

    def append(self, record: dict[str, Any]) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
            self.records_written += 1
        except Exception as exc:
            self.write_errors += 1
            self.last_error = repr(exc)
            print(
                "RTC queue replacement diagnostic log failed: "
                f"{self.last_error}",
                flush=True,
            )

    def status(self) -> dict[str, Any]:
        return {
            "path": None if self.path is None else str(self.path),
            "records_written": int(self.records_written),
            "write_errors": int(self.write_errors),
            "last_error": self.last_error,
        }


def _stop_data_and_hardware(data_session: Any, hardware: Any) -> None:
    errors: list[str] = []

    def _run(label: str, fn: Any) -> None:
        try:
            fn()
        except Exception as exc:
            errors.append(f"{label}: {exc}")

    data_thread = threading.Thread(
        target=_run,
        args=("data.stop", data_session.stop),
        name="rollout_data_stop",
        daemon=False,
    )
    hardware_thread = threading.Thread(
        target=_run,
        args=("hardware.stop", hardware.stop),
        name="rollout_hardware_stop",
        daemon=False,
    )
    data_thread.start()
    hardware_thread.start()
    data_thread.join()
    hardware_thread.join()
    _run("data.close", data_session.close)
    _run("hardware.close", hardware.close)
    if errors:
        print("cleanup errors: " + "; ".join(errors), flush=True)


def _reset_robot_home(robot_client: Any, *, timeout_s: float) -> None:
    if robot_client is None:
        raise RuntimeError("robot client is unavailable for reset_home")
    reset_home = getattr(robot_client, "reset_home", None)
    if not callable(reset_home):
        raise RuntimeError("robot client does not implement reset_home()")
    result = reset_home(timeout=float(timeout_s), wait=True)
    if not isinstance(result, dict) or not bool(result.get("accepted", False)):
        raise RuntimeError(f"robot reset_home rejected: {result}")


def _ensure_startup_home(
    robot_client: Any,
    data_session: Any,
    *,
    state_stream: str,
    target: Any,
    tolerance: Any,
    timeout_s: float,
    settle_s: float,
) -> tuple[bool, list[float], list[float]]:
    """Reset only when measured qpos is outside the configured home envelope."""
    target_array = np.asarray(target, dtype=np.float32).reshape(-1)
    tolerance_array = np.asarray(tolerance, dtype=np.float32).reshape(-1)
    if target_array.shape != (14,) or tolerance_array.shape != (14,):
        raise ValueError("startup home target and tolerance must both have 14 values")
    if not np.all(np.isfinite(target_array)) or not np.all(np.isfinite(tolerance_array)):
        raise ValueError("startup home target and tolerance must be finite")
    if np.any(tolerance_array <= 0):
        raise ValueError("startup home tolerance must be positive")
    if float(settle_s) < 0:
        raise ValueError("startup_home_settle_s must be non-negative")

    before, before_stamp_ns = _wait_robot_qpos_numpy(
        robot_client,
        data_session,
        state_stream=state_stream,
        timeout_s=timeout_s,
    )
    outside = np.abs(before - target_array) > tolerance_array
    if not np.any(outside):
        print(
            "[rollout_sync] startup_home_check already_home=true "
            f"qpos={before.tolist()}",
            flush=True,
        )
        values = before.tolist()
        return False, values, values

    indexes = np.flatnonzero(outside).tolist()
    print(
        "[rollout_sync] startup_home_check already_home=false; "
        f"dimensions={indexes}, qpos={before.tolist()}; starting reset_home",
        flush=True,
    )
    _reset_robot_home(robot_client, timeout_s=timeout_s)
    deadline = time.monotonic() + float(settle_s)
    while time.monotonic() < deadline:
        robot_client.spin_once(min(0.01, deadline - time.monotonic()))
    after, _ = _wait_robot_qpos_numpy(
        robot_client,
        data_session,
        state_stream=state_stream,
        timeout_s=timeout_s,
        newer_than_ns=before_stamp_ns,
    )
    outside_after = np.abs(after - target_array) > tolerance_array
    if np.any(outside_after):
        indexes = np.flatnonzero(outside_after).tolist()
        raise RuntimeError(
            "reset_home completed but measured qpos is outside startup home "
            f"tolerance; dimensions={indexes}, qpos={after.tolist()}"
        )
    print(
        "[rollout_sync] startup reset_home completed and measured home confirmed; "
        f"qpos={after.tolist()}",
        flush=True,
    )
    return True, before.tolist(), after.tolist()


def _wait_robot_qpos_numpy(
    robot_client: Any,
    data_session: Any,
    *,
    state_stream: str,
    timeout_s: float,
    newer_than_ns: int = 0,
) -> tuple[np.ndarray, int]:
    deadline = time.monotonic() + float(timeout_s)
    last_error = ""
    while time.monotonic() < deadline:
        try:
            sample = data_session.latest_numpy([state_stream])[state_stream]
            stamp_ns = int(sample.stamp_ns)
            if stamp_ns <= int(newer_than_ns):
                raise ValueError("latest robot state is not newer than reset request")
            position = np.asarray(sample.data["position"], dtype=np.float32).reshape(-1)
            if position.shape != (14,) or not np.all(np.isfinite(position)):
                raise ValueError("robot state position must contain 14 finite values")
            return position, stamp_ns
        except (KeyError, TypeError, ValueError) as exc:
            last_error = str(exc)
        robot_client.spin_once(min(0.01, max(0.0, deadline - time.monotonic())))
    raise TimeoutError(
        f"timed out waiting for decoded {state_stream!r} qpos: {last_error}"
    )


def _wait_policy_inputs_ready(data_session: Any, policy: Any, timeout_s: float) -> None:
    anchor = getattr(policy, "anchor", None)
    stream_names = getattr(policy, "stream_names", None)
    camera_streams = getattr(policy, "camera_streams", None)
    window_size = int(getattr(policy, "window_size", 1))
    stride = int(getattr(policy, "stride", 1))
    if not anchor or not stream_names:
        return
    camera_names = tuple(
        str(name) for name in dict(camera_streams or {}).values()
    )
    startup_camera_frames = int(
        getattr(policy, "startup_camera_frames", 1)
    )
    if startup_camera_frames <= 0:
        raise ValueError("policy startup_camera_frames must be positive")
    needed = (window_size - 1) * stride + 1
    deadline = time.monotonic() + float(timeout_s)
    last_error = ""
    while time.monotonic() < deadline:
        numpy_counts = data_session.numpy_counts()
        cameras_warm = (
            not camera_names
            or (
                data_session.ready(camera_names, count=startup_camera_frames)
                and all(
                    int(numpy_counts.get(name, 0)) >= startup_camera_frames
                    for name in camera_names
                )
            )
        )
        if (
            cameras_warm
            and data_session.ready([anchor], count=needed)
            and data_session.ready(stream_names, count=1)
        ):
            try:
                frames = data_session.window_numpy(
                    anchor=anchor,
                    names=stream_names,
                    count=window_size,
                    stride=stride,
                    slop_ms=float(getattr(policy, "slop_ms", 100.0)),
                )
                _validate_policy_window(policy, frames)
                if camera_names and startup_camera_frames > 1:
                    print(
                        "[rollout_sync] camera warmup complete before first inference: "
                        + ", ".join(
                            f"{name}={int(numpy_counts.get(name, 0))}"
                            for name in camera_names
                        ),
                        flush=True,
                    )
                return
            except Exception as exc:
                last_error = str(exc)
        time.sleep(0.05)
    skew_hint = ""
    try:
        skew = data_session.latest_skew_ms(anchor=anchor, names=stream_names)
        skew_hint = f" latest_header_skew_ms={skew}"
    except Exception:
        pass
    raise TimeoutError(
        "timed out waiting for policy input window: "
        f"anchor={anchor!r} needed={needed} "
        f"startup_camera_frames={startup_camera_frames} "
        f"camera_names={camera_names} counts={data_session.counts()} "
        f"numpy_counts={data_session.numpy_counts()} "
        f"last_error={last_error!r}{skew_hint}"
    )


def _validate_policy_window(policy: Any, frames: list[Any]) -> None:
    runtime = getattr(policy, "runtime", None)
    if runtime is None or not runtime.__class__.__module__.startswith("infer."):
        return
    from infer.preprocess import parse_preprocess_config
    from prometheus.policy.numpy_preprocess import build_obs_from_numpy_frames

    preprocess_cfg = parse_preprocess_config(
        runtime.cfg,
        robot=getattr(policy, "robot_cfg", {}),
    )
    build_obs_from_numpy_frames(
        frames,
        preprocess_cfg,
        runtime.normalizer,
        window_size=int(getattr(runtime, "window_size")),
    )


if __name__ == "__main__":
    main()
