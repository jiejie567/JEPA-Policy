from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class RobotState:
    joint_names: list[str]
    qpos: np.ndarray
    qvel: np.ndarray | None = None
    effort: np.ndarray | None = None


@runtime_checkable
class RobotDriver(Protocol):
    """Minimal hardware adapter interface for one robot."""

    def connect(self) -> None:
        ...

    def read_state(self) -> RobotState:
        ...

    def send_action(self, action_space: str, action: np.ndarray) -> None:
        ...

    def close(self) -> None:
        ...

    def status(self) -> Mapping[str, Any]:
        ...
