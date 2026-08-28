from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


CONTROL_FIELDS = {
    "send_action": frozenset(
        {
            "request_id",
            "client_stamp_ns",
            "command",
            "command_source",
            "action_space",
            "action",
        }
    ),
    "release_control": frozenset(
        {"request_id", "client_stamp_ns", "command", "command_source"}
    ),
    "reset_home": frozenset(
        {"request_id", "client_stamp_ns", "command", "command_source"}
    ),
    "set_teach_mode": frozenset(
        {"request_id", "client_stamp_ns", "command", "command_source"}
    ),
    "set_position_mode": frozenset(
        {"request_id", "client_stamp_ns", "command", "command_source"}
    ),
}


def json_dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), separators=(",", ":"), ensure_ascii=True)


def default_command_source() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def stamp_to_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


@dataclass(frozen=True)
class RobotTopics:
    state: str
    control: str
    result: str
    status: str


def robot_topics_from_mapping(value: Mapping[str, Any]) -> RobotTopics:
    missing = [name for name in RobotTopics.__dataclass_fields__ if name not in value]
    if missing:
        raise ValueError(f"robot topics missing fields: {', '.join(missing)}")
    return RobotTopics(
        state=str(value["state"]),
        control=str(value["control"]),
        result=str(value["result"]),
        status=str(value["status"]),
    )


def validate_control_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("control payload must be a JSON object")
    command = payload.get("command")
    if command not in CONTROL_FIELDS:
        raise ValueError(f"unsupported command {command!r}")
    required = CONTROL_FIELDS[command]
    keys = set(payload)
    missing = required.difference(keys)
    unknown = keys.difference(required)
    if missing:
        raise ValueError(f"{command} missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"{command} contains unknown fields: {', '.join(sorted(unknown))}")
    if not isinstance(payload["request_id"], str) or not payload["request_id"]:
        raise ValueError("request_id must be a non-empty string")
    if not isinstance(payload["client_stamp_ns"], int):
        raise ValueError("client_stamp_ns must be an integer")
    if not isinstance(payload["command_source"], str) or not payload["command_source"]:
        raise ValueError("command_source must be a non-empty string")
    if command == "send_action" and not isinstance(payload["action_space"], str):
        raise ValueError("action_space must be a string")
    return payload


def checked_action_array(
    action: Any,
    *,
    action_space: str,
    action_dims: Mapping[str, int] | None = None,
) -> np.ndarray:
    action_space = str(action_space)
    if not action_space:
        raise ValueError("action_space must be non-empty")

    values = np.asarray(action, dtype=np.float32).reshape(-1)
    expected_dim = dict(action_dims or {}).get(action_space)
    if expected_dim is not None and values.size != int(expected_dim):
        raise ValueError(f"{action_space} action must have {expected_dim} values, got {values.size}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{action_space} action contains NaN or inf values")
    return values
