from __future__ import annotations

from typing import Any

import numpy as np


class RobotClientActionTarget:
    """Adapter from RobotControlClient-like objects to ActionScheduler targets."""

    def __init__(self, robot_client: Any, *, timeout_s: float, wait: bool):
        self.robot_client = robot_client
        self.action_space = str(robot_client.action_space)
        dims = dict(getattr(robot_client, "action_dims", {}) or {})
        fallback_dim = getattr(robot_client, "action_dim", None)
        self.action_dim = int(dims.get(self.action_space, fallback_dim or 0))
        if self.action_dim <= 0:
            raise ValueError("robot client must expose positive action_dim")
        self.timeout_s = float(timeout_s)
        self.wait = bool(wait)

    def send_action(self, action: np.ndarray) -> None:
        result = self.robot_client.send_action(action, timeout=self.timeout_s, wait=self.wait)
        if self.wait:
            if not isinstance(result, dict):
                raise RuntimeError("robot client send_action must return a dict when wait=True")
            if not bool(result.get("accepted", False)):
                raise RuntimeError(str(result.get("message", "robot rejected action")))
        else:
            spin_once = getattr(self.robot_client, "spin_once", None)
            if callable(spin_once):
                spin_once(0.0)
