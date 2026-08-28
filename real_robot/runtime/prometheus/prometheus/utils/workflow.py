from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from hydra.utils import instantiate

from prometheus.sessions.data import DataSession
from prometheus.sessions.event import EventSession
from prometheus.sessions.hardware import HardwareSession


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    output_dir: Path
    runtime_dir: Path
    manifest_path: Path


def config_dict(cfg: Any, workflow_name: str) -> dict[str, Any]:
    if cfg.__class__.__module__.startswith("omegaconf."):
        from omegaconf import OmegaConf

        cfg = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg, Mapping):
        raise ValueError(f"{workflow_name} config must resolve to a mapping")
    return dict(cfg)


def section(data: Mapping[str, Any], name: str) -> dict[str, Any]:
    if name not in data:
        raise ValueError(f"{name} section is required")
    value = data[name]
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def prepare_run(run_cfg: Mapping[str, Any]) -> RunPaths:
    run_id = str(run_cfg["id"]).strip()
    output_dir_value = str(run_cfg["output_dir"]).strip()
    if not run_id:
        raise ValueError("run.id must be non-empty")
    if not output_dir_value:
        raise ValueError("run.output_dir must be non-empty")
    output_dir = Path(output_dir_value).expanduser()
    runtime_dir = Path(str(run_cfg["runtime_dir"])).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        run_id=run_id,
        output_dir=output_dir,
        runtime_dir=runtime_dir,
        manifest_path=output_dir / "workflow_manifest.json",
    )


def build_hardware_session(
    *,
    run: RunPaths,
    hardware_cfg: Mapping[str, Any],
    robot_cfg: Mapping[str, Any],
) -> HardwareSession:
    robot = instantiate_config(robot_cfg, "robot", _recursive_=False)
    if not isinstance(robot, Mapping):
        raise ValueError("robot resolver must return a mapping")
    robot = dict(robot)

    sensors = hardware_cfg["sensors"]
    if sensors == "auto":
        sensors = dict(robot["sensors"])
    else:
        if not isinstance(sensors, Mapping):
            raise ValueError("hardware.sensors must be a mapping or 'auto'")
        sensors = dict(sensors)

    streams = hardware_cfg["streams"]
    if streams == "auto":
        streams = dict(robot["streams"])
    else:
        if not isinstance(streams, Mapping):
            raise ValueError("hardware.streams must be a mapping or 'auto'")
        streams = dict(streams)

    return HardwareSession(
        run_id=run.run_id,
        runtime_dir=run.runtime_dir,
        name=str(hardware_cfg["name"]),
        mode=str(hardware_cfg["mode"]),
        robot=robot,
        sensors=sensors,
        streams=streams,
    )


def build_data_session(
    *,
    data_cfg: Mapping[str, Any],
    hardware: HardwareSession,
) -> DataSession:
    recording = section(data_cfg, "recording")
    return DataSession(
        hardware=hardware,
        name=str(data_cfg["name"]),
        mode=str(data_cfg["mode"]),
        history=int(data_cfg["history"]),
        bridge=section(data_cfg, "bridge"),
        numpy=dict(data_cfg.get("numpy", {})),
        recording=recording,
        timing=dict(data_cfg.get("timing", {})),
    )


def build_event_session(event_cfg: Mapping[str, Any]) -> EventSession:
    return EventSession(
        name=str(event_cfg["name"]),
        mode=str(event_cfg["mode"]),
        input_source=event_cfg.get("input_source"),
        event_cfg=dict(event_cfg),
    )


def instantiate_config(cfg: Mapping[str, Any], name: str, **kwargs: Any) -> Any:
    if not str(cfg.get("_target_", "")).strip():
        raise ValueError(f"{name}._target_ must be set")

    return instantiate(dict(cfg), **kwargs, _convert_="all")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)
