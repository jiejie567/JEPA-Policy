from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any, Mapping, Protocol, runtime_checkable


class SessionMode(str, Enum):
    """How a session obtains the resource it represents."""

    OWNED = "owned"
    ATTACHED = "attached"


class SessionState(str, Enum):
    """Common lifecycle states for asynchronously managed resources."""

    NEW = "new"
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    CLOSED = "closed"


@dataclass(frozen=True)
class SessionStatus:
    """Structured status snapshot returned by every session."""

    name: str
    mode: SessionMode
    state: SessionState
    ready: bool = False
    message: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)
    started_at_monotonic_s: float | None = None
    updated_at_monotonic_s: float = field(default_factory=time.monotonic)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode.value,
            "state": self.state.value,
            "ready": self.ready,
            "message": self.message,
            "details": dict(self.details),
            "started_at_monotonic_s": self.started_at_monotonic_s,
            "updated_at_monotonic_s": self.updated_at_monotonic_s,
        }


class SessionError(RuntimeError):
    """Base error for session lifecycle failures."""


class SessionStateError(SessionError):
    """Raised when a lifecycle operation is invalid for the current state."""


class SessionTimeoutError(SessionError, TimeoutError):
    """Raised when a session does not become ready in time."""


@runtime_checkable
class Session(Protocol):
    """Runtime contract implemented by workflow-managed resources."""

    @property
    def name(self) -> str:
        ...

    @property
    def mode(self) -> SessionMode:
        ...

    def start(self) -> None:
        ...

    def wait_ready(self, timeout_s: float) -> SessionStatus:
        ...

    def status(self) -> SessionStatus:
        ...

    def stop(self) -> None:
        ...

    def close(self) -> None:
        ...


class BaseSession(ABC):
    """Small base class for sessions that want shared status bookkeeping."""

    def __init__(self, *, name: str, mode: SessionMode):
        if not name:
            raise ValueError("session name must be non-empty")
        self._name = str(name)
        self._mode = SessionMode(mode)
        self._state = SessionState.NEW
        self._ready = False
        self._message = ""
        self._details: dict[str, Any] = {}
        self._started_at_monotonic_s: float | None = None
        self._updated_at_monotonic_s = time.monotonic()
        self._lock = RLock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def mode(self) -> SessionMode:
        return self._mode

    def status(self) -> SessionStatus:
        with self._lock:
            return SessionStatus(
                name=self._name,
                mode=self._mode,
                state=self._state,
                ready=self._ready,
                message=self._message,
                details=dict(self._details),
                started_at_monotonic_s=self._started_at_monotonic_s,
                updated_at_monotonic_s=self._updated_at_monotonic_s,
            )

    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def wait_ready(self, timeout_s: float) -> SessionStatus:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    def _set_status(
        self,
        *,
        state: SessionState | None = None,
        ready: bool | None = None,
        message: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> SessionStatus:
        with self._lock:
            if state is not None:
                self._state = SessionState(state)
                if self._state == SessionState.STARTING and self._started_at_monotonic_s is None:
                    self._started_at_monotonic_s = time.monotonic()
            if ready is not None:
                self._ready = bool(ready)
            elif state is not None and self._state == SessionState.READY:
                self._ready = True
            elif state is not None and self._state in {
                SessionState.NEW,
                SessionState.STARTING,
                SessionState.STOPPING,
                SessionState.STOPPED,
                SessionState.FAILED,
                SessionState.CLOSED,
            }:
                self._ready = False
            if message is not None:
                self._message = str(message)
            if details is not None:
                self._details = dict(details)
            self._updated_at_monotonic_s = time.monotonic()
            return self.status()

    def _mark_failed(self, message: str, *, details: Mapping[str, Any] | None = None) -> SessionStatus:
        return self._set_status(
            state=SessionState.FAILED,
            ready=False,
            message=message,
            details=details,
        )
