from __future__ import annotations

import time
from collections import deque
from typing import Any

import numpy as np

from prometheus.nodes.robots.robot_node import RobotState
from prometheus.policy.scheduler import ActionChunk


class NoopPolicy:
    """Hardware-independent policy for workflow and scheduler tests."""

    def __init__(
        self,
        *,
        action_space: str,
        action_dim: int,
        hz: float = 10.0,
        horizon: int = 1,
        value: float = 0.0,
        stop_after: int | None = None,
        interrupt_after: int | None = None,
    ):
        if not action_space:
            raise ValueError("NoopPolicy requires action_space")
        if int(action_dim) <= 0:
            raise ValueError("NoopPolicy requires positive action_dim")
        if int(horizon) <= 0:
            raise ValueError("NoopPolicy horizon must be positive")
        if float(hz) <= 0:
            raise ValueError("NoopPolicy hz must be positive")
        if stop_after is not None and int(stop_after) <= 0:
            raise ValueError("NoopPolicy stop_after must be positive when set")
        if interrupt_after is not None and int(interrupt_after) <= 0:
            raise ValueError("NoopPolicy interrupt_after must be positive when set")

        self.action_space = str(action_space)
        self.action_dim = int(action_dim)
        self.hz = float(hz)
        self.horizon = int(horizon)
        self.value = float(value)
        self.stop_after = None if stop_after is None else int(stop_after)
        self.interrupt_after = (
            None if interrupt_after is None else int(interrupt_after)
        )
        self.infer_count = 0
        self.closed = False

    def infer(self, data: Any) -> ActionChunk:
        if self.closed:
            raise RuntimeError("NoopPolicy is closed")
        self.infer_count += 1
        if (
            self.interrupt_after is not None
            and self.infer_count >= self.interrupt_after
        ):
            raise KeyboardInterrupt
        actions = np.full((self.horizon, self.action_dim), self.value, dtype=np.float32)
        metadata: dict[str, Any] = {"policy": "noop", "infer_count": self.infer_count}
        if self.stop_after is not None and self.infer_count >= self.stop_after:
            metadata["done"] = True
        return ActionChunk(
            actions=actions,
            action_space=self.action_space,
            hz=self.hz,
            metadata=metadata,
        )

    def status(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "action_space": self.action_space,
            "action_dim": self.action_dim,
            "hz": self.hz,
            "horizon": self.horizon,
            "infer_count": self.infer_count,
            "interrupt_after": self.interrupt_after,
            "closed": self.closed,
        }

    def close(self) -> None:
        self.closed = True


class NoopRobotClient:
    """Robot-control stand-in for hardware-independent rollout tests."""

    def __init__(self, *, action_space: str, action_dim: int, history: int = 1024):
        if not action_space:
            raise ValueError("NoopRobotClient requires action_space")
        if int(action_dim) <= 0:
            raise ValueError("NoopRobotClient requires positive action_dim")
        if int(history) <= 0:
            raise ValueError("NoopRobotClient history must be positive")
        self.action_space = str(action_space)
        self.action_dim = int(action_dim)
        self.history: deque[np.ndarray] = deque(maxlen=int(history))
        self.closed = False
        self.released = False
        self.reset_count = 0
        self.teach_count = 0
        self.position_count = 0
        self.events: list[str] = []

    def send_action(self, action: Any, *, timeout: float = 3.0, wait: bool = True) -> dict[str, Any]:
        if self.closed:
            raise RuntimeError("NoopRobotClient is closed")
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if values.shape != (self.action_dim,):
            raise ValueError(f"noop action must have shape {(self.action_dim,)}, got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("noop action contains NaN or inf")
        self.history.append(values.copy())
        return {
            "accepted": True,
            "message": "",
            "action_space": self.action_space,
            "action_dim": self.action_dim,
            "timeout": float(timeout),
            "wait": bool(wait),
        }

    def spin_once(self, timeout_s: float = 0.0) -> None:
        if timeout_s > 0:
            time.sleep(float(timeout_s))

    def release_control(self, *, timeout: float = 3.0, wait: bool = True) -> dict[str, Any]:
        self.released = True
        self.events.append("release_control")
        return {
            "accepted": True,
            "message": "",
            "timeout": float(timeout),
            "wait": bool(wait),
        }

    def reset_home(self, *, timeout: float = 30.0, wait: bool = True) -> dict[str, Any]:
        self.reset_count += 1
        self.events.append("reset_home")
        return {
            "accepted": True,
            "message": "",
            "timeout": float(timeout),
            "wait": bool(wait),
        }

    def set_teach_mode(self, *, timeout: float = 5.0, wait: bool = True) -> dict[str, Any]:
        self.teach_count += 1
        return {
            "accepted": True,
            "message": "",
            "timeout": float(timeout),
            "wait": bool(wait),
        }

    def set_position_mode(self, *, timeout: float = 5.0, wait: bool = True) -> dict[str, Any]:
        self.position_count += 1
        return {
            "accepted": True,
            "message": "",
            "timeout": float(timeout),
            "wait": bool(wait),
        }

    def close(self) -> None:
        self.events.append("close")
        self.closed = True

    def status(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "action_space": self.action_space,
            "action_dim": self.action_dim,
            "sent_actions": len(self.history),
            "released": self.released,
            "closed": self.closed,
            "reset_count": self.reset_count,
            "teach_count": self.teach_count,
            "position_count": self.position_count,
            "events": list(self.events),
        }


class NoopRobotDriver:
    """RobotDriver test adapter with no ROS or hardware dependency."""

    def __init__(self, *, action_space: str = "noop_action", action_dim: int = 2):
        self.action_space = str(action_space)
        self.action_dim = int(action_dim)
        self.connected = False
        self.closed = False
        self.actions: list[list[float]] = []

    def connect(self) -> None:
        self.connected = True

    def read_state(self) -> RobotState:
        return RobotState(
            joint_names=[f"joint_{index}" for index in range(self.action_dim)],
            qpos=np.zeros(self.action_dim, dtype=np.float32),
        )

    def send_action(self, action_space: str, action: np.ndarray) -> None:
        if action_space != self.action_space:
            raise ValueError(f"expected action_space {self.action_space!r}, got {action_space!r}")
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if values.shape != (self.action_dim,):
            raise ValueError(f"expected action shape {(self.action_dim,)}, got {values.shape}")
        self.actions.append([float(value) for value in values])

    def close(self) -> None:
        self.closed = True

    def status(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "connected": self.connected,
            "closed": self.closed,
            "sent_actions": len(self.actions),
        }


class NoopRobotOwner:
    """Owner runtime test double that wraps a NoopRobotDriver-like object."""

    def __init__(self, *, driver: Any, robot_id: str = "noop"):
        self.driver = driver
        self.robot_id = str(robot_id)
        self.started = False
        self.ready = False
        self.stopped = False
        self.closed = False

    def start(self) -> None:
        self.driver.connect()
        self.started = True
        self.ready = True

    def wait_ready(self, timeout_s: float) -> None:
        if not self.ready:
            raise TimeoutError(f"{self.robot_id} owner is not ready")

    def stop(self) -> None:
        self.stopped = True
        self.ready = False

    def close(self) -> None:
        self.driver.close()
        self.closed = True

    def status(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "robot_id": self.robot_id,
            "started": self.started,
            "ready": self.ready,
            "stopped": self.stopped,
            "closed": self.closed,
        }


class NoopRuntimeComponent:
    """Generic lifecycle component for session tests."""

    def __init__(self, *, name: str, data: Any | None = None):
        self.name = str(name)
        self.data = data
        self.started = False
        self.ready = False
        self.stopped = False
        self.closed = False

    def start(self) -> None:
        self.started = True
        self.ready = True

    def wait_ready(self, timeout_s: float) -> None:
        if not self.ready:
            raise TimeoutError(f"{self.name} is not ready")

    def stop(self) -> None:
        self.stopped = True
        self.ready = False

    def close(self) -> None:
        self.closed = True

    def status(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "name": self.name,
            "has_data": self.data is not None,
            "started": self.started,
            "ready": self.ready,
            "stopped": self.stopped,
            "closed": self.closed,
        }
