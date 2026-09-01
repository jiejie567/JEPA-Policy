from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from prometheus.dagger.controller import DAggerConfig, DAggerController
from prometheus.dagger.phase import DAggerPhase
from prometheus.policy.scheduler import ActionChunk
from prometheus.sessions.event import semantics_by_id, semantics_by_name
from prometheus.sessions.policy import PolicySession
from prometheus.utils.workflow import (
    build_data_session,
    build_event_session,
    build_hardware_session,
    config_dict,
    instantiate_config,
    prepare_run,
    section,
    write_json,
)
from prometheus.visualization.preview import (
    close_realtime_vis,
    poll_realtime_vis,
    start_realtime_vis,
    sync_dagger_preview,
)
from prometheus.workflows.record_drag import _next_episode_index
from prometheus.visualization.preview import (
    close_preview,
    poll_preview,
    set_preview_status,
    start_collection_preview,
)
from prometheus.workflows.rollout_sync import (
    _stop_data_and_hardware,
    _wait_policy_inputs_ready,
)


class ChunkRunner:
    """Execute one cached policy chunk action per scheduler step."""
    def __init__(self) -> None:
        self._chunk: ActionChunk | None = None
        self._index = 0

    def reset(self) -> None:
        self._chunk = None
        self._index = 0

    def step(self, scheduler: Any, *, infer_fn: Any) -> tuple[np.ndarray | None, int]:
        if self._chunk is None or self._index >= self._chunk.horizon:
            self._chunk = infer_fn()
            self._index = 0
        action = self._chunk.actions[self._index].copy()
        sub = ActionChunk(
            actions=action.reshape(1, -1),
            action_space=self._chunk.action_space,
            hz=self._chunk.hz,
            metadata=self._chunk.metadata,
        )
        sent = int(scheduler.execute(sub, steps=1))
        self._index += 1
        return action, sent


@dataclass
class ScriptedEventSchedule:
    events: list[tuple[int, Any, int]]
    _emitted: set[int] = field(default_factory=set)

    @classmethod
    def from_config(cls, raw_events: Any, event_cfg: dict[str, Any]) -> ScriptedEventSchedule:
        if not isinstance(raw_events, list):
            raise ValueError("workflow.scripted_events must be a list")
        name_to_id = semantics_by_name(event_cfg)
        events: list[tuple[int, Any, int]] = []
        for item in raw_events:
            if not isinstance(item, dict):
                raise ValueError("workflow.scripted_events items must be mappings")
            if "name" not in item or "after_ticks" not in item:
                raise ValueError("workflow.scripted_events items require name and after_ticks")
            name = str(item["name"])
            if name not in name_to_id:
                raise ValueError(f"workflow.scripted_events name {name!r} is not defined in event.semantics")
            events.append((name_to_id[name], item.get("value", True), int(item["after_ticks"])))
        return cls(events=events)

    def tick(self, loop_tick: int, event_session: Any) -> None:
        for index, (event_id, value, after_ticks) in enumerate(self.events):
            if index in self._emitted:
                continue
            if loop_tick >= after_ticks:
                event_session.emit(event_id, value)
                self._emitted.add(index)


def run_from_config(cfg: Any) -> int:
    # step1: hydra config
    data = config_dict(cfg, "rollout_dagger")
    _apply_ros_config(data.get("ros", {}))
    run_cfg = section(data, "run")
    robot_cfg = section(data, "robot")
    task_cfg = section(data, "task")
    hardware_cfg = section(data, "hardware")
    data_cfg = section(data, "data")
    policy_cfg = section(data, "policy")
    policy_session_cfg = section(data, "policy_session")
    scheduler_cfg = section(data, "scheduler")
    event_cfg = section(data, "event")
    workflow_cfg = section(data, "workflow")
    dagger_cfg = _dagger_cfg(data.get("dagger", {}))
    run = prepare_run(run_cfg)
    run.runtime_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PROMETHEUS_CAMERA_TIMING_TRACE"] = str(run.runtime_dir / "camera_publish_timing.jsonl")

    if not section(data_cfg, "recording").get("enabled", False):
        raise ValueError("rollout_dagger requires data.recording.enabled=true")

    control_hz = float(dagger_cfg.get("control_hz", 30.0))
    max_cycles = workflow_cfg.get("max_cycles")
    max_ticks = workflow_cfg.get("max_ticks")
    interrupt_accept_active_episode = bool(workflow_cfg.get("interrupt_accept_active_episode", False))
    scripted_schedule = _scripted_schedule(workflow_cfg.get("scripted_events"), event_cfg)
    event_semantics = semantics_by_id(event_cfg)

    # step2: workflow-owned sessions and policy object
    hardware = build_hardware_session(run=run, hardware_cfg=hardware_cfg, robot_cfg=robot_cfg)
    data_session = build_data_session(data_cfg=data_cfg, hardware=hardware)
    event_session = build_event_session(event_cfg)
    policy = instantiate_config(policy_cfg, "policy")
    policy_session = PolicySession(
        policy=policy,
        name=str(policy_session_cfg["name"]),
        mode=str(policy_session_cfg["mode"]),
    )

    robot_client = None
    scheduler = None
    controller = None

    status = "running"
    stop_reason = "not_started"
    cycles_completed = 0
    actions_sent = 0
    chunk_steps = 0
    loop_ticks = 0
    return_code = 0
    preview = None

    try:
        # step3: start hardware and policy resources, then wait for readiness
        hardware.start()
        data_session.start()
        policy_session.start()
        event_session.start()
        wait_ready_timeout_s = float(run_cfg["wait_ready_timeout_s"])
        hardware.wait_ready(wait_ready_timeout_s)
        data_session.wait_ready(wait_ready_timeout_s)
        policy_session.wait_ready(wait_ready_timeout_s)
        event_session.wait_ready(wait_ready_timeout_s)
        _wait_policy_inputs_ready(data_session, policy_session.policy, wait_ready_timeout_s)

        # step4: bind the robot client and scheduler after hardware is ready
        robot_client = hardware.robot_client
        scheduler = instantiate_config(scheduler_cfg, "scheduler", robot_client=robot_client)
        scheduler.reset()

        # step5: start the controller
        controller = DAggerController(
            robot_client=robot_client,
            data_session=data_session,
            config=DAggerConfig(
                cycle_wait_s=float(dagger_cfg.get("cycle_wait_s", 30.0)),
                reset_home_timeout_s=float(dagger_cfg.get("reset_home_timeout_s", 30.0)),
                event_name=str(dagger_cfg.get("event_name", "event_label")),
            ),
            run_id=run.run_id,
            get_robot_stamp_ns=lambda: _latest_robot_stamp_ns(data_session),
            episode_index_start=_next_episode_index(data_cfg),
            task_metadata={"task": task_cfg},
        )
        controller.bootstrap_autonomous()
        preview = start_realtime_vis(data_session, workflow_cfg)
        sync_dagger_preview(preview, controller, data_session)

        chunk_runner = ChunkRunner()
        tick_s = 1.0 / control_hz

        while True:
            tick_start = time.monotonic()
            loop_ticks += 1
            scripted_schedule.tick(loop_ticks, event_session)

            polled = event_session.poll()
            if polled is not None:
                event_id, _value = polled
                prev_phase = controller.phase
                controller.apply_semantic(event_semantics[event_id])
                if controller.phase != prev_phase and controller.phase in {
                    DAggerPhase.CYCLE_WAIT,
                    DAggerPhase.CORRECTING,
                }:
                    chunk_runner.reset()

            robot_client.spin_once(0.0)

            if controller.phase == DAggerPhase.CYCLE_WAIT:
                if controller.tick_cycle_wait():
                    chunk_runner.reset()
                    cycles_completed += 1
                    if max_cycles is not None and cycles_completed >= max_cycles:
                        stop_reason = "max_cycles"
                        break
            elif controller.phase == DAggerPhase.AUTONOMOUS:
                if controller.can_infer or controller.pending_pause:
                    _action, sent = chunk_runner.step(
                        scheduler,
                        infer_fn=lambda: policy_session.policy.infer(data_session),
                    )
                    actions_sent += sent
                    chunk_steps += sent
                    if controller.pending_pause:
                        controller.enter_paused(_action)
            elif controller.should_hold and controller.hold_action is not None:
                robot_client.send_action(controller.hold_action)

            if max_ticks is not None and loop_ticks >= max_ticks:
                stop_reason = "max_ticks"
                break

            sync_dagger_preview(preview, controller, data_session)
            poll_realtime_vis(preview)

            delay = tick_s - (time.monotonic() - tick_start)
            if delay > 0:
                time.sleep(delay)
        else:
            stop_reason = "loop_exit"

        status = "completed"
        return 0
    except KeyboardInterrupt:
        status = "interrupted"
        stop_reason = "keyboard_interrupt"
        return_code = 130
        return return_code
    except Exception:
        status = "error"
        stop_reason = "exception"
        return_code = 1
        raise
    finally:
        close_realtime_vis(preview)
        try:
            data_session.stop_ingress()
        except Exception:
            pass
        if data_session.recording_enabled and data_session.recorder.recording:
            accept = status == "completed" or (
                status == "interrupted" and interrupt_accept_active_episode
            )
            data_session.stop_episode(accept=accept)
        try:
            event_session.stop()
        finally:
            event_session.close()
        try:
            policy_session.stop()
        finally:
            policy_session.close()
        _stop_data_and_hardware(data_session, hardware)
        write_json(
            run.manifest_path,
            {
                "run_id": run.run_id,
                "workflow": "rollout_dagger",
                "status": status,
                "stop_reason": stop_reason,
                "cycles_completed": cycles_completed,
                "actions_sent": actions_sent,
                "chunk_steps": chunk_steps,
                "loop_ticks": loop_ticks,
                "time_ns": time.time_ns(),
                "output_dir": str(run.output_dir),
                "runtime_dir": str(run.runtime_dir),
                "task": task_cfg,
                "controller": None if controller is None else controller.status(),
                "hardware": hardware.status().as_dict(),
                "data": data_session.status().as_dict(),
                "policy": policy_session.status().as_dict(),
                "event": event_session.status().as_dict(),
                "robot_client": None if robot_client is None else robot_client.status(),
                "scheduler": {
                    "dry_run": None if scheduler is None else scheduler.dry_run,
                    "history": 0 if scheduler is None else len(scheduler.history),
                    "filters": [] if scheduler is None else [item.name for item in scheduler.filters],
                },
            },
        )
        if return_code:
            return return_code


def main() -> None:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../configs", config_name="dagger/rollout_fm_dagger")
    def _main(cfg: DictConfig) -> None:
        raise SystemExit(run_from_config(cfg))

    _main()


def _dagger_cfg(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("dagger must be a mapping")
    return dict(raw)


def _scripted_schedule(raw_events: Any, event_cfg: dict[str, Any]) -> ScriptedEventSchedule:
    if not raw_events:
        return ScriptedEventSchedule(events=[])
    return ScriptedEventSchedule.from_config(raw_events, event_cfg)


def _latest_robot_stamp_ns(data_session: Any) -> int | None:
    try:
        sample = data_session.latest(["robot_state"])["robot_state"]
    except (KeyError, TypeError):
        return None
    stamp_ns = getattr(sample, "stamp_ns", None)
    return None if stamp_ns is None else int(stamp_ns)


def _apply_ros_config(raw_cfg: Any) -> None:
    if raw_cfg is None:
        return
    if not isinstance(raw_cfg, dict):
        raise ValueError("ros must be a mapping")
    if "localhost_only" in raw_cfg:
        enabled = bool(raw_cfg["localhost_only"])
        os.environ["ROS_LOCALHOST_ONLY"] = "1" if enabled else "0"
        if enabled:
            os.environ.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")


if __name__ == "__main__":
    main()
