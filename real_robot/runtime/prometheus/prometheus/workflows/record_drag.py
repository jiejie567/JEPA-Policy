from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any

from prometheus.sessions.event import semantics_by_id, semantics_by_name
from prometheus.utils.workflow import (
    build_data_session,
    build_event_session,
    build_hardware_session,
    config_dict,
    prepare_run,
    section,
    write_json,
)
from prometheus.visualization.preview import (
    close_preview,
    poll_preview,
    set_preview_status,
    start_collection_preview,
)


def run_from_config(cfg: Any) -> int:
    data = config_dict(cfg, "record_drag")
    _apply_ros_config(data.get("ros", {}))
    run_cfg = section(data, "run")
    robot_cfg = section(data, "robot")
    task_cfg = section(data, "task")
    hardware_cfg = section(data, "hardware")
    data_cfg = section(data, "data")
    event_cfg = section(data, "event")
    workflow_cfg = section(data, "workflow")
    run = prepare_run(run_cfg)

    hardware = build_hardware_session(run=run, hardware_cfg=hardware_cfg, robot_cfg=robot_cfg)
    data_session = build_data_session(data_cfg=data_cfg, hardware=hardware)
    event_session = build_event_session(event_cfg)
    event_semantics = semantics_by_id(event_cfg)

    if not data_session.recording_enabled:
        raise ValueError("record_drag requires data.recording.enabled=true")

    wait_ready_timeout_s = float(run_cfg["wait_ready_timeout_s"])
    event_wait_timeout_s = float(workflow_cfg["event_wait_timeout_s"])
    max_episodes = int(workflow_cfg["max_episodes"])
    scripted_events = _scripted_events(workflow_cfg["events"], event_cfg)
    if max_episodes <= 0:
        raise ValueError("workflow.max_episodes must be positive")
    if event_wait_timeout_s <= 0:
        raise ValueError("workflow.event_wait_timeout_s must be positive")
    if not scripted_events and not event_cfg["input_source"]:
        raise ValueError("record_drag needs event.input_source or workflow.events")
    if scripted_events and not event_cfg["input_source"]:
        stop_id = semantics_by_name(event_cfg)["record_stop"]
        stops = sum(1 for event_id, _value in scripted_events if event_id == stop_id)
        if stops < max_episodes:
            raise ValueError("scripted record_drag needs one record_stop per requested episode")

    status = "running"
    stop_reason = "not_started"
    next_episode_index = _next_episode_index(data_cfg)
    started = 0
    saved = 0
    rejected = 0
    ignored = 0
    current_episode_id = ""
    current_episode_dir = None
    preview = None
    cleanup_errors: list[str] = []

    try:
        hardware.start()
        data_session.start()
        event_session.start()
        hardware.wait_ready(wait_ready_timeout_s)
        data_session.wait_ready(wait_ready_timeout_s)
        event_session.wait_ready(wait_ready_timeout_s)
        preview = start_collection_preview(data_session, workflow_cfg)
        _print_ready_message(run, task_cfg, workflow_cfg, data_session)
        _emit_scripted_events(event_session, scripted_events)

        while saved + rejected < max_episodes:
            poll_preview(preview)
            event = event_session.wait(timeout_s=event_wait_timeout_s)
            if event is None:
                continue
            poll_preview(preview)
            event_id, value = event
            name = event_semantics[event_id]

            if name == "record_start":
                if current_episode_id:
                    ignored += 1
                    continue
                current_episode_id = f"{next_episode_index:04d}"
                next_episode_index += 1
                started += 1
                current_episode_dir = data_session.start_episode(
                    current_episode_id,
                    metadata={"workflow": "record_drag", "run_id": run.run_id, "task": task_cfg},
                )
                set_preview_status(preview, "running", f"episode={current_episode_id}")
                data_session.record_event("record_start", value, metadata={"episode_id": current_episode_id})
                print(f"[record] started episode={current_episode_id} dir={current_episode_dir}", flush=True)
                continue

            if name == "record_stop":
                if not current_episode_id:
                    ignored += 1
                    continue
                data_session.record_event("record_stop", value, metadata={"episode_id": current_episode_id})
                accept = bool(value)
                data_session.stop_episode(accept=accept)
                set_preview_status(preview, "saved" if accept else "rejected", f"episode={current_episode_id}")
                print(
                    f"[record] {'saved' if accept else 'rejected'} episode={current_episode_id} dir={current_episode_dir}",
                    flush=True,
                )
                episode_id = current_episode_id
                saved += int(accept)
                rejected += int(not accept)
                current_episode_id = ""
                current_episode_dir = None
                _reset_robot_after_episode(hardware, workflow_cfg, episode_id)
                stop_reason = "max_episodes" if saved + rejected >= max_episodes else "record_stop"
                continue

            if not current_episode_id:
                ignored += 1
                continue
            data_session.record_event(name, value, metadata={"episode_id": current_episode_id})
        else:
            stop_reason = "max_episodes"

        status = "completed"
        return 0
    except KeyboardInterrupt:
        status = "interrupted"
        stop_reason = "keyboard_interrupt"
        print("[record] Ctrl-C received; saving any active episode before shutdown.", flush=True)
        return 130
    except Exception:
        status = "error"
        stop_reason = "exception"
        raise
    finally:
        if current_episode_id:
            data_session.record_event("record_stop", metadata={"reason": stop_reason})
            data_session.stop_episode(accept=status in {"completed", "interrupted"})
            if status in {"completed", "interrupted"}:
                saved += 1
                set_preview_status(preview, "saved", f"episode={current_episode_id}")
                print(f"[record] saved episode={current_episode_id} dir={current_episode_dir}", flush=True)
                try:
                    _reset_robot_after_episode(hardware, workflow_cfg, current_episode_id)
                except Exception as exc:
                    cleanup_errors.append(f"reset_after_episode: {exc}")
                    print(f"[robot] reset_after_episode failed during cleanup: {exc}", flush=True)
            else:
                rejected += 1
                set_preview_status(preview, "rejected", f"episode={current_episode_id}")
        close_preview(preview)
        try:
            event_session.stop()
        finally:
            event_session.close()
        try:
            data_session.stop()
        finally:
            data_session.close()
        try:
            hardware.stop()
        finally:
            hardware.close()
        if cleanup_errors:
            print("cleanup errors: " + "; ".join(cleanup_errors), flush=True)
        write_json(
            run.manifest_path,
            {
                "run_id": run.run_id,
                "status": status,
                "stop_reason": stop_reason,
                "episodes_started": started,
                "episodes_saved": saved,
                "episodes_rejected": rejected,
                "events_ignored": ignored,
                "time_ns": time.time_ns(),
                "output_dir": str(run.output_dir),
                "runtime_dir": str(run.runtime_dir),
                "task": task_cfg,
                "hardware": hardware.status().as_dict(),
                "data": data_session.status().as_dict(),
                "event": event_session.status().as_dict(),
            },
        )


def _emit_scripted_events(event_session: Any, events: list[tuple[int, Any]]) -> None:
    for event_id, value in events:
        event_session.emit(event_id, value)


def _scripted_events(raw_events: Any, event_cfg: dict[str, Any]) -> list[tuple[int, Any]]:
    if not isinstance(raw_events, list):
        raise ValueError("workflow.events must be a list")
    name_to_id = semantics_by_name(event_cfg)
    events: list[tuple[int, Any]] = []
    for item in raw_events:
        if not isinstance(item, dict):
            raise ValueError("workflow.events items must be mappings with name and value")
        if "name" not in item or "value" not in item:
            raise ValueError("workflow.events items require name and value")
        name = str(item["name"])
        if name not in name_to_id:
            raise ValueError(f"workflow.events name {name!r} is not defined in event.semantics")
        events.append((name_to_id[name], item["value"]))
    return events


def _reset_robot_after_episode(hardware: Any, workflow_cfg: dict[str, Any], episode_id: str) -> None:
    reset_cfg = workflow_cfg.get("reset_after_episode", True)
    if isinstance(reset_cfg, dict):
        enabled = bool(reset_cfg.get("enabled", True))
        timeout_s = float(reset_cfg.get("timeout_s", 30.0))
    else:
        enabled = bool(reset_cfg)
        timeout_s = 30.0
    if not enabled:
        return

    client = getattr(hardware, "robot_client", None)
    reset_home = getattr(client, "reset_home", None)
    if not callable(reset_home):
        print("[robot] reset_after_episode skipped: robot client has no reset_home()", flush=True)
        return

    print(f"[robot] episode={episode_id} recorded; resetting robot home.", flush=True)
    result = reset_home(timeout=timeout_s, wait=True)
    if isinstance(result, dict) and not bool(result.get("accepted", False)):
        raise RuntimeError(f"robot reset_home rejected: {result}")


def _next_episode_index(data_cfg: dict[str, Any]) -> int:
    recording_cfg = data_cfg.get("recording", {})
    if not isinstance(recording_cfg, dict):
        return 0
    output_dir = Path(str(recording_cfg.get("output_dir", ""))).expanduser()
    if not output_dir.exists():
        return 0
    indices = [
        int(path.name)
        for path in output_dir.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]
    return max(indices, default=-1) + 1


def _print_ready_message(run: Any, task_cfg: dict[str, Any], workflow_cfg: dict[str, Any], data_session: Any) -> None:
    preview_cfg = workflow_cfg.get("preview", {})
    preview_mode = preview_cfg.get("mode", "local") if isinstance(preview_cfg, dict) else "disabled"
    print("", flush=True)
    print("[ready] Prometheus data collection is ready.", flush=True)
    print(f"[ready] task={task_cfg.get('id')} run_id={run.run_id}", flush=True)
    print(f"[ready] recording_root={data_session.recorder.config.get('output_dir', '')}", flush=True)
    print(f"[ready] preview={preview_mode}", flush=True)
    print(f"[ready] ROS_LOCALHOST_ONLY={os.environ.get('ROS_LOCALHOST_ONLY', '')}", flush=True)
    print("[ready] controls: aa=start recording, bb=stop/save, cc=event mark, Ctrl-C=save active episode and exit", flush=True)
    print("", flush=True)


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


def main() -> None:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../configs", config_name="record_drag")
    def _main(cfg: DictConfig) -> None:
        raise SystemExit(run_from_config(cfg))

    _main()


if __name__ == "__main__":
    main()
