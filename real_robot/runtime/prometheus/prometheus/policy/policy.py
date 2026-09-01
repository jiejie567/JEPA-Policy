from __future__ import annotations

from typing import Any, Protocol

from prometheus.policy.scheduler import ActionChunk


class Policy(Protocol):
    """A policy reads workflow data and returns an executable action chunk."""

    def infer(self, data: Any) -> ActionChunk:
        ...
