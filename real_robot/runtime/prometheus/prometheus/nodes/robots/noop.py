from __future__ import annotations

from typing import Any, Mapping


def resolve_robot(**robot: Any) -> dict[str, Any]:
    config = dict(robot)
    config["type"] = "noop"
    config.update(runtime_config(config))
    config["sensors"] = sensor_config(config)
    config["streams"] = streams(config)
    return config


def runtime_config(robot: Mapping[str, Any]) -> dict[str, Any]:
    return {}


def sensor_config(robot: Mapping[str, Any]) -> dict[str, Any]:
    return dict(robot.get("sensors", {}))


def streams(robot: Mapping[str, Any]) -> dict[str, Any]:
    return dict(robot.get("streams", {}))
