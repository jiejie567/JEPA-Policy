from __future__ import annotations

import os
import queue
import select
import shutil
import sys
import termios
import threading
import time
import traceback
import tty
from pathlib import Path
from typing import Any

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
from prometheus.workflows.rollout_sync import (
    _check_startup_action_semantics,
    _ensure_startup_home,
    _record_action,
    _reset_robot_home,
    _stop_data_and_hardware,
    _wait_policy_inputs_ready,
    _wait_recording_ready,
)


class TerminalCommands:
    """Single-key controls that remain responsive while inference is running."""

    def __init__(self) -> None:
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            raise RuntimeError("persistent paper evaluation requires an interactive terminal")
        self._fd = sys.stdin.fileno()
        self._saved = termios.tcgetattr(self._fd)
        self._commands: queue.Queue[str] = queue.Queue()
        self._closed = threading.Event()
        tty.setcbreak(self._fd)
        self._thread = threading.Thread(
            target=self._read_loop,
            name="paper_eval_keyboard",
            daemon=True,
        )
        self._thread.start()

    def _read_loop(self) -> None:
        while not self._closed.is_set():
            try:
                readable, _, _ = select.select([self._fd], [], [], 0.1)
                if not readable:
                    continue
                raw = os.read(self._fd, 1)
            except OSError:
                return
            if raw == b" ":
                self._commands.put("start")
            elif raw in {b"s", b"S"}:
                self._commands.put("success")
            elif raw in {b"f", b"F"}:
                self._commands.put("failure")
            elif raw in {b"a", b"A"}:
                self._commands.put("skip")
            elif raw in {b"r", b"R"}:
                self._commands.put("reset")
            elif raw in {b"h", b"H"}:
                self._commands.put("help")

    def get(self, timeout_s: float = 0.05) -> str | None:
        try:
            return self._commands.get(timeout=max(0.0, float(timeout_s)))
        except queue.Empty:
            return None

    def poll(self) -> str | None:
        return self.get(0.0)

    def discard(self) -> None:
        while self.poll() is not None:
            pass

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._thread.join(timeout=0.3)
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)


def _print_controls() -> None:
    print(
        "[paper_eval] controls:\n"
        "  Space             : start while idle; finish recovery countdown early\n"
        "  S                 : mark active episode SUCCESS, save, and reset Home\n"
        "  F                 : mark active episode FAILURE, save, and reset Home\n"
        "  A                 : skip active episode, delete its data, and reset Home\n"
        "  R                 : emergency discard/reset active episode; reset Home while idle\n"
        "  Enter             : ignored\n"
        "  H                 : show controls\n"
        "  Ctrl+C            : reset Home, close the persistent session, and exit",
        flush=True,
    )


def _service_wait(robot_client: Any, duration_s: float = 0.05) -> None:
    robot_client.spin_once(max(0.0, float(duration_s)))


def _wait_ready_command(commands: Any, robot_client: Any, episode_index: int) -> str:
    print(
        f"[paper_eval] READY episode={episode_index:04d}. "
        "Press Space to start, R to reset, Ctrl+C to end.",
        flush=True,
    )
    while True:
        command = commands.get(0.05)
        _service_wait(robot_client, 0.0)
        if command == "help":
            _print_controls()
        elif command in {"start", "reset"}:
            return command


def _wait_recovery(commands: Any, robot_client: Any, seconds: int) -> None:
    if seconds <= 0:
        return
    deadline = time.monotonic() + seconds
    print("[paper_eval] Home reset complete. Restore the scene now.", flush=True)
    last_remaining = None
    while True:
        remaining = max(0, int(deadline - time.monotonic() + 0.999))
        if remaining != last_remaining:
            print(
                f"\r[paper_eval] scene recovery: {remaining:3d}s remaining "
                "(Space to finish) ",
                end="",
                flush=True,
            )
            last_remaining = remaining
        if time.monotonic() >= deadline:
            print("\r[paper_eval] scene recovery: ready                       ", flush=True)
            return
        command = commands.get(0.05)
        _service_wait(robot_client, 0.0)
        if command == "start":
            print("\r[paper_eval] scene recovery: finished early              ", flush=True)
            return
        if command == "help":
            print()
            _print_controls()
        # Reset during recovery is redundant and intentionally ignored.


def _reset_episode_state(policy: Any, scheduler: Any) -> int:
    scheduler.reset()
    scheduler.history.clear()
    reset_episode = getattr(policy, "reset_episode", None)
    if callable(reset_episode):
        reset_episode()
    require_fresh = getattr(policy, "require_observation_after", None)
    barrier_ns = time.time_ns()
    if callable(require_fresh):
        require_fresh(barrier_ns)
    return barrier_ns


def _write_episode_row(path: Path, payload: dict[str, Any]) -> None:
    fields = (
        payload["episode_id"],
        payload["outcome"],
        payload["started_at"],
        payload["ended_at"],
        payload["status"],
        payload["stop_reason"],
        payload["chunks"],
        payload["actions_sent"],
        payload.get("data_dir", ""),
    )
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\t".join(str(value) for value in fields) + "\n")


def _write_summary(
    path: Path,
    *,
    success_count: int,
    failure_count: int,
    skipped_count: int,
    unjudged_count: int,
) -> None:
    evaluated = int(success_count) + int(failure_count)
    write_json(
        path,
        {
            "evaluated_episodes": evaluated,
            "success_count": int(success_count),
            "failure_count": int(failure_count),
            "skipped_count": int(skipped_count),
            "unjudged_count": int(unjudged_count),
            "success_rate": (
                None if evaluated == 0 else float(success_count) / float(evaluated)
            ),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )


def run_from_config(cfg: Any, *, commands: Any | None = None) -> int:
    data = config_dict(cfg, "rollout_paper_eval")
    run_cfg = section(data, "run")
    robot_cfg = section(data, "robot")
    task_cfg = section(data, "task")
    hardware_cfg = section(data, "hardware")
    data_cfg = section(data, "data")
    policy_cfg = section(data, "policy")
    policy_session_cfg = section(data, "policy_session")
    scheduler_cfg = section(data, "scheduler")
    workflow_cfg = section(data, "workflow")
    paper_cfg = dict(workflow_cfg.get("paper_eval", {}))
    recovery_seconds = int(paper_cfg.get("recovery_seconds", 30))
    max_episodes = int(paper_cfg.get("max_episodes", 0))
    if recovery_seconds < 0 or max_episodes < 0:
        raise ValueError("paper_eval recovery_seconds/max_episodes must be non-negative")

    run = prepare_run(run_cfg)
    run.runtime_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PROMETHEUS_CAMERA_TIMING_TRACE"] = str(
        run.runtime_dir / "camera_publish_timing.jsonl"
    )
    paper_dir = run.output_dir / "paper_eval"
    episodes_dir = paper_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    index_path = paper_dir / "episodes.tsv"
    index_path.write_text(
        "episode\toutcome\tstarted_at\tended_at\tstatus\tstop_reason\tchunks\t"
        "actions_sent\tdata_dir\n",
        encoding="utf-8",
    )
    summary_path = paper_dir / "summary.json"

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
    owned_commands = commands is None
    terminal = None
    episodes: list[dict[str, Any]] = []
    status = "starting"
    failure = ""
    reset_home_succeeded = False
    return_code = 0
    wait_ready_timeout_s = float(run_cfg["wait_ready_timeout_s"])
    action_steps_per_chunk = workflow_cfg.get("action_steps_per_chunk")
    if action_steps_per_chunk is not None:
        action_steps_per_chunk = int(action_steps_per_chunk)
    post_chunk_delay_s = float(workflow_cfg.get("post_chunk_observation_delay_s", 0.0))

    try:
        hardware.start()
        data_session.start()
        policy_session.start()
        hardware.wait_ready(wait_ready_timeout_s)
        robot_client = hardware.robot_client
        data_session.wait_ready(wait_ready_timeout_s)
        policy_session.wait_ready(wait_ready_timeout_s)
        if bool(getattr(policy_session.policy, "async_inference_enabled", False)):
            raise RuntimeError("persistent paper evaluation currently requires synchronous inference")
        if bool(workflow_cfg.get("ensure_home_before_rollout", False)):
            attempted, _, _ = _ensure_startup_home(
                robot_client,
                data_session,
                state_stream=str(workflow_cfg.get("startup_home_state_stream", "robot_state")),
                target=workflow_cfg.get("startup_home_target"),
                tolerance=workflow_cfg.get("startup_home_tolerance"),
                timeout_s=float(workflow_cfg.get("reset_home_timeout_s", 30.0)),
                settle_s=float(workflow_cfg.get("startup_home_settle_s", 0.2)),
            )
            reset_home_succeeded = attempted
        _wait_policy_inputs_ready(data_session, policy_session.policy, wait_ready_timeout_s)
        scheduler = instantiate_config(scheduler_cfg, "scheduler", robot_client=robot_client)
        if owned_commands:
            terminal = TerminalCommands()
            commands = terminal
        assert commands is not None

        status = "ready"
        print(
            "[paper_eval] PERSISTENT SESSION READY: checkpoint/CUDA/cameras/robot "
            "will remain loaded across episodes",
            flush=True,
        )
        print(f"[paper_eval] output_dir={run.output_dir}", flush=True)
        _print_controls()
        episode_index = 0
        attempt_count = 0
        success_count = 0
        failure_count = 0
        skipped_count = 0
        unjudged_count = 0
        _write_summary(
            summary_path,
            success_count=success_count,
            failure_count=failure_count,
            skipped_count=skipped_count,
            unjudged_count=unjudged_count,
        )
        while max_episodes == 0 or episode_index < max_episodes:
            command = _wait_ready_command(commands, robot_client, episode_index)
            if command == "reset":
                print("[paper_eval] idle Home reset requested", flush=True)
                _reset_robot_home(
                    robot_client,
                    timeout_s=float(workflow_cfg.get("reset_home_timeout_s", 30.0)),
                )
                reset_home_succeeded = True
                commands.discard()
                _wait_recovery(commands, robot_client, recovery_seconds)
                continue

            episode_id = f"{episode_index:04d}"
            started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            episode = {
                "episode_id": episode_id,
                "attempt_index": attempt_count,
                "outcome": "pending",
                "started_at": started_at,
                "ended_at": "",
                "status": "running",
                "stop_reason": "not_started",
                "chunks": 0,
                "actions_sent": 0,
                "data_dir": "",
                "observation_barrier_targets_ns": [],
                "startup_action_guard_checks": [],
            }
            episode["initial_observation_barrier_ns"] = _reset_episode_state(
                policy_session.policy, scheduler
            )
            if data_session.recording_enabled:
                episode_path = data_session.start_episode(
                    episode_id,
                    metadata={
                        "workflow": "rollout_paper_eval",
                        "run_id": run.run_id,
                        "episode_id": episode_id,
                        "task": task_cfg,
                    },
                )
                episode["data_dir"] = str(episode_path)
                _wait_recording_ready(data_session, max(wait_ready_timeout_s, 30.0))
            print(
                f"[paper_eval] starting episode {episode_id}: manual S/F/A decision, "
                f"A{action_steps_per_chunk if action_steps_per_chunk is not None else 'all'} "
                "per inference chunk",
                flush=True,
            )
            stop_requested = False
            operator_stop_reason = ""
            operator_outcome = ""

            def should_stop() -> bool:
                nonlocal operator_outcome, operator_stop_reason, stop_requested
                while True:
                    active_command = commands.poll()
                    if active_command is None:
                        break
                    if active_command == "reset":
                        stop_requested = True
                        operator_stop_reason = "operator_reset"
                        operator_outcome = "skip"
                        break
                    elif active_command in {"success", "failure", "skip"}:
                        stop_requested = True
                        operator_outcome = active_command
                        operator_stop_reason = f"operator_{active_command}"
                        break
                    elif active_command == "help":
                        _print_controls()
                return stop_requested

            try:
                while not should_stop():
                    robot_client.spin_once(0.0)
                    chunk = policy_session.policy.infer(data_session)
                    if should_stop():
                        break
                    guard_check = _check_startup_action_semantics(
                        chunk,
                        workflow_cfg.get("startup_action_guard", {}),
                        chunk_index=int(episode["chunks"]),
                    )
                    if guard_check is not None:
                        episode["startup_action_guard_checks"].append(guard_check)
                        if bool(guard_check["rejected"]):
                            raise SafetyGuardError(
                                "startup_action_guard rejected policy chunk "
                                f"{episode['chunks']}"
                            )
                    chunk_recorded = data_session.record_policy_action_chunk(
                        chunk,
                        metadata={"chunk_index": int(episode["chunks"])},
                    )
                    if data_session.recording_enabled and not chunk_recorded:
                        raise RuntimeError("failed to enqueue policy action chunk for recording")
                    sent = scheduler.execute(
                        chunk,
                        steps=action_steps_per_chunk,
                        on_action=(
                            _record_action(data_session, int(episode["chunks"]))
                            if data_session.recording_enabled
                            else None
                        ),
                        should_stop=should_stop,
                    )
                    episode["actions_sent"] += sent
                    if sent:
                        episode["chunks"] += 1
                    if stop_requested:
                        break
                    require_fresh = getattr(
                        policy_session.policy, "require_observation_after", None
                    )
                    if callable(require_fresh):
                        target_ns = time.time_ns() + int(
                            round(float(chunk.dt) * 1_000_000_000)
                        )
                        require_fresh(target_ns)
                        episode["observation_barrier_targets_ns"].append(target_ns)
                    if post_chunk_delay_s > 0:
                        scheduler.idle(post_chunk_delay_s)
                if stop_requested:
                    episode["outcome"] = operator_outcome
                    episode["status"] = (
                        "completed" if operator_outcome in {"success", "failure"} else "skipped"
                    )
                    episode["stop_reason"] = operator_stop_reason
                else:
                    raise RuntimeError("manual evaluation ended without an S/F/A decision")
            except KeyboardInterrupt:
                episode["status"] = "interrupted"
                episode["outcome"] = "unjudged"
                episode["stop_reason"] = "session_ctrl_c"
                raise
            except SafetyGuardError as exc:
                # A rejected command is a policy/evaluation failure, not a
                # reason to tear down the persistent checkpoint, cameras and
                # robot session. No rejected action was dispatched. Preserve
                # the episode as a failure, reset Home through the normal path
                # below, then let the operator prepare the next attempt.
                episode["status"] = "completed"
                episode["outcome"] = "failure"
                episode["stop_reason"] = "safety_guard"
                episode["failure"] = f"{type(exc).__name__}: {exc}"
                print(
                    "[paper_eval] safety guard stopped this episode; "
                    "recording FAILURE and keeping the persistent session alive",
                    flush=True,
                )
            except Exception as exc:
                episode["status"] = "error"
                episode["outcome"] = "error"
                episode["stop_reason"] = "exception"
                episode["failure"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                keep_episode = episode["outcome"] in {"success", "failure"}
                if data_session.recording_enabled:
                    data_session.stop_episode(accept=keep_episode)
                episode["ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                episode_path = Path(episode["data_dir"]) if episode["data_dir"] else None
                if keep_episode:
                    outcome = str(episode["outcome"])
                    if episode_path is not None:
                        classified_path = episode_path.parent / outcome / episode_id
                        classified_path.parent.mkdir(parents=True, exist_ok=True)
                        if classified_path.exists():
                            raise FileExistsError(
                                f"classified episode already exists: {classified_path}"
                            )
                        episode_path.replace(classified_path)
                        episode["data_dir"] = str(classified_path)
                    metadata_path = episodes_dir / outcome / f"{episode_id}.json"
                    metadata_path.parent.mkdir(parents=True, exist_ok=True)
                    write_json(metadata_path, episode)
                    _write_episode_row(index_path, episode)
                    episodes.append(dict(episode))
                    if outcome == "success":
                        success_count += 1
                    else:
                        failure_count += 1
                elif episode["outcome"] in {"skip", "unjudged"}:
                    if episode_path is not None and episode_path.exists():
                        shutil.rmtree(episode_path)
                    if episode["outcome"] == "skip":
                        skipped_count += 1
                    else:
                        unjudged_count += 1
                else:
                    metadata_path = episodes_dir / "error" / f"attempt_{attempt_count:04d}.json"
                    metadata_path.parent.mkdir(parents=True, exist_ok=True)
                    write_json(metadata_path, episode)
                _write_summary(
                    summary_path,
                    success_count=success_count,
                    failure_count=failure_count,
                    skipped_count=skipped_count,
                    unjudged_count=unjudged_count,
                )
                attempt_count += 1

            print(
                f"[paper_eval] episode {episode_id} outcome={episode['outcome']}; "
                "starting ordered Home reset",
                flush=True,
            )
            _reset_robot_home(
                robot_client,
                timeout_s=float(workflow_cfg.get("reset_home_timeout_s", 30.0)),
            )
            reset_home_succeeded = True
            if episode["outcome"] in {"success", "failure"}:
                episode_index += 1
            commands.discard()
            if max_episodes and episode_index >= max_episodes:
                break
            _wait_recovery(commands, robot_client, recovery_seconds)

        status = "completed"
        print(
            f"[paper_eval] requested evaluated episode count completed: {episode_index} "
            f"(success={success_count}, failure={failure_count}, skipped={skipped_count})",
            flush=True,
        )
    except KeyboardInterrupt:
        status = "interrupted"
        return_code = 130
        print(
            "\n[paper_eval] Ctrl+C received; resetting Home before persistent session cleanup",
            flush=True,
        )
        if robot_client is not None:
            try:
                _reset_robot_home(
                    robot_client,
                    timeout_s=float(workflow_cfg.get("reset_home_timeout_s", 30.0)),
                )
                reset_home_succeeded = True
                print("[paper_eval] Home reset complete; entering damping cleanup", flush=True)
            except KeyboardInterrupt:
                failure = "Home reset aborted by second Ctrl+C"
                print(f"[paper_eval] {failure}; entering damping cleanup", flush=True)
            except Exception as reset_exc:
                failure = f"Home reset failed: {type(reset_exc).__name__}: {reset_exc}"
                print(f"[paper_eval] {failure}; entering damping cleanup", flush=True)
    except Exception as exc:
        status = "error"
        return_code = 1
        failure = f"{type(exc).__name__}: {exc}"
        print(f"[paper_eval] fatal: {failure}", flush=True)
        traceback.print_exc()
        if robot_client is not None and bool(workflow_cfg.get("reset_home_on_error", False)):
            try:
                _reset_robot_home(
                    robot_client,
                    timeout_s=float(workflow_cfg.get("reset_home_timeout_s", 30.0)),
                )
                reset_home_succeeded = True
            except Exception as reset_exc:
                failure += f"; reset failed: {type(reset_exc).__name__}: {reset_exc}"
    finally:
        if terminal is not None:
            terminal.close()
        try:
            if data_session.recording_enabled:
                data_session.stop_episode(accept=False)
        except Exception:
            pass
        try:
            policy_session.stop()
        finally:
            policy_session.close()
        _stop_data_and_hardware(data_session, hardware)
        write_json(
            run.manifest_path,
            {
                "run_id": run.run_id,
                "workflow": "rollout_paper_eval",
                "persistent_resources": True,
                "status": status,
                "failure": failure,
                "episode_count": len(episodes),
                "evaluated_episodes": len(episodes),
                "success_count": sum(item.get("outcome") == "success" for item in episodes),
                "failure_count": sum(item.get("outcome") == "failure" for item in episodes),
                "skipped_count": skipped_count if "skipped_count" in locals() else 0,
                "unjudged_count": unjudged_count if "unjudged_count" in locals() else 0,
                "success_rate": (
                    None
                    if not episodes
                    else sum(item.get("outcome") == "success" for item in episodes)
                    / len(episodes)
                ),
                "episodes": episodes,
                "checkpoint_loads": 1,
                "reset_home_succeeded": reset_home_succeeded,
                "output_dir": str(run.output_dir),
                "original_camera_recording": str(run.output_dir / "camera_original"),
                "per_episode_data": str(run.output_dir / "data"),
                "hardware": hardware.status().as_dict(),
                "data": data_session.status().as_dict(),
                "policy": policy_session.status().as_dict(),
                "robot_client": None if robot_client is None else robot_client.status(),
                "time_ns": time.time_ns(),
            },
        )
    return return_code


def main() -> None:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../configs", config_name="rollout_jepa_safe_sync")
    def _main(cfg: DictConfig) -> None:
        raise SystemExit(run_from_config(cfg))

    _main()


if __name__ == "__main__":
    main()
