from __future__ import annotations

import time
from typing import Any, Mapping

from prometheus.policy.policy import Policy
from prometheus.sessions.session import (
    BaseSession,
    SessionMode,
    SessionState,
    SessionStateError,
    SessionStatus,
    SessionTimeoutError,
)


class PolicySession(BaseSession):
    """Lifecycle wrapper for policies with optional long-lived resources."""

    def __init__(
        self,
        *,
        policy: Policy,
        mode: str | SessionMode = SessionMode.OWNED,
        name: str = "policy",
        details: Mapping[str, Any] | None = None,
    ):
        super().__init__(name=name, mode=SessionMode(mode))
        if policy is None:
            raise ValueError("policy must not be None")
        self._policy = policy
        self._extra_details = dict(details or {})

    @property
    def policy(self) -> Policy:
        status = self.status()
        if status.state not in {SessionState.READY, SessionState.STOPPING, SessionState.STOPPED}:
            raise SessionStateError(f"policy session is not ready: {status.state.value}")
        return self._policy

    def start(self) -> None:
        status = self.status()
        if status.state == SessionState.READY:
            return
        if status.state in {SessionState.STOPPING, SessionState.STOPPED, SessionState.CLOSED}:
            raise SessionStateError(f"cannot start policy session from state {status.state.value}")
        self._set_status(
            state=SessionState.STARTING,
            message="starting policy session",
            details=self._status_details(),
        )
        try:
            start = getattr(self._policy, "start", None)
            if callable(start):
                start()
            self._set_status(
                state=SessionState.READY,
                message="policy session ready",
                details=self._status_details(),
            )
        except Exception as exc:
            self._mark_failed(str(exc), details=self._status_details())
            raise

    def wait_ready(self, timeout_s: float) -> SessionStatus:
        wait_ready = getattr(self._policy, "wait_ready", None)
        if callable(wait_ready):
            wait_ready(timeout_s=float(timeout_s))
            self._set_status(
                state=SessionState.READY,
                message="policy session ready",
                details=self._status_details(),
            )
            return self.status()

        deadline = time.monotonic() + float(timeout_s)
        while True:
            status = self.status()
            if status.ready:
                return status
            if status.state == SessionState.FAILED:
                raise SessionStateError(status.message or "policy session failed")
            if time.monotonic() >= deadline:
                raise SessionTimeoutError(
                    f"timed out waiting for policy session {self.name!r}"
                )
            time.sleep(0.01)

    def stop(self) -> None:
        status = self.status()
        if status.state in {SessionState.NEW, SessionState.STOPPED, SessionState.CLOSED}:
            return
        if status.state == SessionState.STOPPING:
            return
        self._set_status(
            state=SessionState.STOPPING,
            message="stopping policy session",
            details=self._status_details(),
        )
        stop = getattr(self._policy, "stop", None)
        if callable(stop):
            stop()
        self._set_status(
            state=SessionState.STOPPED,
            message="policy session stopped",
            details=self._status_details(),
        )

    def close(self) -> None:
        status = self.status()
        if status.state == SessionState.CLOSED:
            return
        if status.state not in {SessionState.NEW, SessionState.STOPPED}:
            self.stop()
        close = getattr(self._policy, "close", None)
        if callable(close):
            close()
        self._set_status(
            state=SessionState.CLOSED,
            message="policy session closed",
            details=self._status_details(),
        )

    def _status_details(self) -> dict[str, Any]:
        payload = {
            "policy_type": type(self._policy).__name__,
            **self._extra_details,
        }
        status = getattr(self._policy, "status", None)
        if callable(status):
            payload["policy"] = status()
        return payload
