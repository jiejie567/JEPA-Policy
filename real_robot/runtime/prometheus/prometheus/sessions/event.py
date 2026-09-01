from __future__ import annotations

import os
import select
import struct
import sys
import termios
import time
import tty
from collections import deque
from threading import Condition, Event as StopEvent, Thread
from typing import Any, Mapping

from prometheus.sessions.session import BaseSession, SessionMode, SessionState, SessionStateError, SessionStatus

Event = tuple[int, Any]

PEDAL_DEVICE = "/dev/input/prometheus-pedal"
PEDAL_EDGE_BY_VALUE = {0: "up", 1: "down", 2: "down"}
DEFAULT_PEDAL_MAP = {30: 0, 48: 1, 46: 2}
DEFAULT_KEY_MAP = {"a": 0, "b": 1, "c": 2}
DEFAULT_RECORD_DRAG_SEMANTICS = {0: "record_start", 1: "record_stop", 2: "event_mark"}
_EVDEV_EVENT = struct.Struct("llHHI")


class EventSession(BaseSession):
    """Workflow-local event queue with an optional real input source."""

    def __init__(
        self,
        *,
        mode: str | SessionMode = SessionMode.OWNED,
        name: str = "event",
        input_source: str | None = None,
        event_cfg: Mapping[str, Any] | None = None,
    ):
        super().__init__(name=name, mode=SessionMode(mode))
        self.event_cfg = dict(event_cfg or {})
        self._events: deque[Event] = deque()
        self._condition = Condition(self._lock)
        self._input_source_name, self._input_source = _build_input_source(
            self,
            input_source,
            self.event_cfg,
        )
        self._details = self._details_snapshot()

    def emit(self, event_id: int | str, value: Any = True) -> None:
        with self._condition:
            self._require_ready()
            self._events.append((normalize_event_id(event_id), value))
            self._set_status(details=self._details_snapshot())
            self._condition.notify()

    def poll(self) -> Event | None:
        return self.wait(timeout_s=0.0)

    def wait(self, timeout_s: float | None = None) -> Event | None:
        deadline = None if timeout_s is None else time.monotonic() + float(timeout_s)
        with self._condition:
            self._require_ready()
            while not self._events:
                if deadline is None:
                    self._condition.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._condition.wait(remaining)
                self._require_ready()
            event = self._events.popleft()
            self._set_status(details=self._details_snapshot())
            return event

    def start(self) -> None:
        if self._state == SessionState.READY:
            return
        if self._state != SessionState.NEW:
            raise SessionStateError(f"cannot start event session from state {self._state.value}")
        self._notify(state=SessionState.READY, message="event session ready")
        try:
            if self._input_source is not None:
                self._input_source.start()
                self._set_status(details=self._details_snapshot())
        except Exception as exc:
            self._mark_failed(str(exc), details=self._details_snapshot())
            raise

    def wait_ready(self, timeout_s: float) -> SessionStatus:
        status = self.status()
        if status.ready:
            return status
        raise SessionStateError(status.message or f"event session is {status.state.value}")

    def stop(self) -> None:
        if self._state in {SessionState.NEW, SessionState.STOPPED, SessionState.CLOSED}:
            return
        if self._input_source is not None:
            self._input_source.close()
        self._notify(state=SessionState.STOPPED, message="event session stopped")

    def close(self) -> None:
        if self._state == SessionState.CLOSED:
            return
        if self._state not in {SessionState.NEW, SessionState.STOPPED}:
            self.stop()
        self._notify(state=SessionState.CLOSED, message="event session closed")

    def _notify(self, *, state: SessionState, message: str) -> None:
        self._set_status(state=state, message=message, details=self._details_snapshot())
        with self._condition:
            self._condition.notify_all()

    def _require_ready(self) -> None:
        if self._state != SessionState.READY:
            raise SessionStateError(f"event session is not ready: {self._state.value}")

    def _details_snapshot(self) -> dict[str, Any]:
        source = None
        if self._input_source is not None:
            source = {"name": self._input_source_name, **self._input_source.status()}
        return {"pending_events": len(self._events), "input_source": source}


def normalize_event_id(value: int | str) -> int:
    if isinstance(value, bool):
        raise ValueError("event id must be an integer, not bool")
    if isinstance(value, int):
        return int(value)
    text = str(value).strip()
    if not text:
        raise ValueError("event id must be non-empty")
    if not text.lstrip("-").isdigit():
        raise ValueError(f"event id must be an integer, got {value!r}")
    return int(text)


def int_key_map(raw: Any, default: Mapping[int, int]) -> dict[int, int]:
    if raw is None:
        return {int(key): int(value) for key, value in default.items()}
    if not isinstance(raw, Mapping):
        raise ValueError("event map must be a mapping")
    return {int(key): int(value) for key, value in raw.items()}


def str_key_map(raw: Any, default: Mapping[str, int]) -> dict[str, int]:
    if raw is None:
        return {str(key): int(value) for key, value in default.items()}
    if not isinstance(raw, Mapping):
        raise ValueError("event map must be a mapping")
    return {str(key): int(value) for key, value in raw.items()}


def semantics_by_id(event_cfg: Mapping[str, Any] | None) -> dict[int, str]:
    raw = dict(event_cfg or {}).get("semantics")
    if raw is None:
        return dict(DEFAULT_RECORD_DRAG_SEMANTICS)
    if not isinstance(raw, Mapping):
        raise ValueError("event.semantics must be a mapping")
    return {normalize_event_id(key): str(value) for key, value in raw.items()}


def semantics_by_name(event_cfg: Mapping[str, Any] | None) -> dict[str, int]:
    return {name: event_id for event_id, name in semantics_by_id(event_cfg).items()}


def event_name(event_cfg: Mapping[str, Any] | None, event_id: int) -> str:
    normalized = normalize_event_id(event_id)
    semantics = semantics_by_id(event_cfg)
    if normalized not in semantics:
        raise KeyError(f"unknown event id {normalized!r}")
    return semantics[normalized]


class PedalInput:
    """Read the pcsensor foot pedal and emit integer event ids."""

    def __init__(
        self,
        events: EventSession,
        *,
        device: str = PEDAL_DEVICE,
        pedal_map: Mapping[int, int] | None = None,
        confirm_presses: int = 2,
        confirm_window_s: float = 1.0,
        debounce_s: float = 0.25,
        poll_timeout_s: float = 0.02,
    ):
        self.events = events
        self.device = str(device)
        self.pedal_map = int_key_map(pedal_map, DEFAULT_PEDAL_MAP)
        self.confirm_presses = int(confirm_presses)
        self.confirm_window_s = float(confirm_window_s)
        self.debounce_s = float(debounce_s)
        self.poll_timeout_s = float(poll_timeout_s)
        if self.confirm_presses <= 0:
            raise ValueError("confirm_presses must be positive")
        if self.confirm_window_s <= 0:
            raise ValueError("confirm_window_s must be positive")
        if self.debounce_s < 0:
            raise ValueError("debounce_s must be non-negative")
        if self.poll_timeout_s <= 0:
            raise ValueError("poll_timeout_s must be positive")
        self._candidate: tuple[int, int, float] | None = None
        self._locked_event: int | None = None
        self._last_emit_at: dict[int, float] = {}
        self._fd: int | None = None
        self._pending = b""
        self._stop = StopEvent()
        self._thread: Thread | None = None
        self._error = ""

    def start(self) -> None:
        if self._thread is not None:
            return
        self._fd = os.open(self.device, os.O_RDONLY | os.O_NONBLOCK)
        self._stop.clear()
        self._thread = Thread(target=self._run, name="prometheus-pedal", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.poll_timeout_s * 4.0))
            self._thread = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    close = stop

    def status(self) -> dict[str, Any]:
        started = self._thread is not None
        return {
            "type": type(self).__name__,
            "device": self.device,
            "pedal_map": dict(self.pedal_map),
            "confirm_presses": self.confirm_presses,
            "started": started,
            "error": self._error,
        }

    def _run(self) -> None:
        if self._fd is None:
            raise RuntimeError("pedal input file descriptor is not open")
        try:
            while not self._stop.is_set():
                readable, _, _ = select.select([self._fd], [], [], self.poll_timeout_s)
                if not readable:
                    continue
                try:
                    self._process_evdev_bytes(os.read(self._fd, _EVDEV_EVENT.size * 64))
                except BlockingIOError:
                    pass
        except Exception as exc:
            self._error = str(exc)
            raise

    def _process_evdev_bytes(self, data: bytes) -> list[Event]:
        emitted: list[Event] = []
        self._pending += data
        while len(self._pending) >= _EVDEV_EVENT.size:
            raw, self._pending = self._pending[: _EVDEV_EVENT.size], self._pending[_EVDEV_EVENT.size :]
            _sec, _usec, event_type, code, value = _EVDEV_EVENT.unpack(raw)
            edge = PEDAL_EDGE_BY_VALUE.get(value)
            event_id = self.pedal_map.get(code)
            if event_type != 1 or event_id is None or edge is None:
                continue
            event = self._process_event_edge(event_id, edge)
            if event is not None:
                emitted.append(event)
        return emitted

    def _process_event_edge(self, event_id: int, edge: str, *, now: float | None = None) -> Event | None:
        event_id = normalize_event_id(event_id)
        edge = str(edge).strip().lower()
        if edge == "up":
            if self._locked_event == event_id:
                self._locked_event = None
            return None
        if edge != "down" or self._locked_event == event_id:
            return None

        timestamp = time.monotonic() if now is None else float(now)
        previous = self._last_emit_at.get(event_id)
        if previous is not None and timestamp - previous < self.debounce_s:
            return None

        candidate_id, count, started_at = self._candidate or (-1, 0, timestamp)
        if candidate_id != event_id or timestamp - started_at > self.confirm_window_s:
            count, started_at = 0, timestamp
        count += 1
        if count < self.confirm_presses:
            self._candidate = (event_id, count, started_at)
            return None

        self._candidate = None
        self._locked_event = event_id
        self._last_emit_at[event_id] = timestamp
        self.events.emit(event_id)
        return (event_id, True)


class TtyInput:
    """Read a/b/c from the controlling terminal and emit integer event ids."""

    def __init__(
        self,
        events: EventSession,
        *,
        key_map: Mapping[str, int] | None = None,
        confirm_presses: int = 2,
        confirm_window_s: float = 1.0,
        debounce_s: float = 0.25,
        poll_timeout_s: float = 0.02,
    ):
        self.events = events
        self.key_map = str_key_map(key_map, DEFAULT_KEY_MAP)
        self.confirm_presses = int(confirm_presses)
        self.confirm_window_s = float(confirm_window_s)
        self.debounce_s = float(debounce_s)
        self.poll_timeout_s = float(poll_timeout_s)
        self._candidate: tuple[int, int, float] | None = None
        self._last_emit_at: dict[int, float] = {}
        self._stop = StopEvent()
        self._thread: Thread | None = None
        self._old_termios: Any | None = None
        self._error = ""

    def start(self) -> None:
        if self._thread is not None:
            return
        if not sys.stdin.isatty():
            raise RuntimeError("tty input source requires an interactive terminal")
        self._old_termios = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())
        self._stop.clear()
        self._thread = Thread(target=self._run, name="prometheus-tty-events", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.poll_timeout_s * 4.0))
            self._thread = None
        if self._old_termios is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_termios)
            self._old_termios = None

    close = stop

    def status(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "key_map": dict(self.key_map),
            "confirm_presses": self.confirm_presses,
            "started": self._thread is not None,
            "error": self._error,
        }

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                readable, _, _ = select.select([sys.stdin], [], [], self.poll_timeout_s)
                if not readable:
                    continue
                char = sys.stdin.read(1).lower()
                event_id = self.key_map.get(char)
                if event_id is not None:
                    self._process_key_event(event_id)
        except Exception as exc:
            self._error = str(exc)
            raise

    def _process_key_event(self, event_id: int, *, now: float | None = None) -> Event | None:
        event_id = normalize_event_id(event_id)
        timestamp = time.monotonic() if now is None else float(now)
        previous = self._last_emit_at.get(event_id)
        if previous is not None and timestamp - previous < self.debounce_s:
            return None
        candidate_id, count, started_at = self._candidate or (-1, 0, timestamp)
        if candidate_id != event_id or timestamp - started_at > self.confirm_window_s:
            count, started_at = 0, timestamp
        count += 1
        if count < self.confirm_presses:
            self._candidate = (event_id, count, started_at)
            return None
        self._candidate = None
        self._last_emit_at[event_id] = timestamp
        self.events.emit(event_id)
        return (event_id, True)


class CompositeInput:
    def __init__(self, sources: list[tuple[str, Any]]):
        self.sources = list(sources)

    def start(self) -> None:
        started: list[Any] = []
        try:
            for _name, source in self.sources:
                source.start()
                started.append(source)
        except Exception:
            for source in reversed(started):
                source.close()
            raise

    def close(self) -> None:
        for _name, source in reversed(self.sources):
            source.close()

    stop = close

    def status(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "sources": [
                {"name": name, **source.status()}
                for name, source in self.sources
            ],
        }


SUPPORTED_INPUT_SOURCES = frozenset({"pedal", "tty"})


def _input_section(event_cfg: Mapping[str, Any], name: str) -> dict[str, Any]:
    raw = event_cfg.get(name, {})
    if not isinstance(raw, Mapping):
        raise ValueError(f"event.{name} must be a mapping")
    return dict(raw)


def _create_pedal_input(events: EventSession, event_cfg: Mapping[str, Any]) -> PedalInput:
    pedal_cfg = _input_section(event_cfg, "pedal")
    return PedalInput(
        events,
        device=str(pedal_cfg.get("device", PEDAL_DEVICE)),
        pedal_map=pedal_cfg.get("pedal_map"),
        confirm_presses=int(pedal_cfg.get("confirm_presses", 2)),
        confirm_window_s=float(pedal_cfg.get("confirm_window_s", 1.0)),
        debounce_s=float(pedal_cfg.get("debounce_s", 0.25)),
        poll_timeout_s=float(pedal_cfg.get("poll_timeout_s", 0.02)),
    )


def _create_tty_input(events: EventSession, event_cfg: Mapping[str, Any]) -> TtyInput:
    tty_cfg = _input_section(event_cfg, "tty")
    return TtyInput(
        events,
        key_map=tty_cfg.get("key_map"),
        confirm_presses=int(tty_cfg.get("confirm_presses", 2)),
        confirm_window_s=float(tty_cfg.get("confirm_window_s", 1.0)),
        debounce_s=float(tty_cfg.get("debounce_s", 0.25)),
        poll_timeout_s=float(tty_cfg.get("poll_timeout_s", 0.02)),
    )


def _create_input_source(name: str, events: EventSession, event_cfg: Mapping[str, Any]) -> Any:
    if name == "pedal":
        return _create_pedal_input(events, event_cfg)
    if name == "tty":
        return _create_tty_input(events, event_cfg)
    raise ValueError(f"unsupported input source {name!r}")


def _build_input_source(
    events: EventSession,
    input_source: str | None,
    event_cfg: Mapping[str, Any],
) -> tuple[str | None, Any | None]:
    raw = "" if input_source is None else str(input_source).strip()
    if not raw:
        return None, None
    names = [item.strip() for item in raw.replace("+", ",").split(",") if item.strip()]
    unknown = [name for name in names if name not in SUPPORTED_INPUT_SOURCES]
    if unknown:
        supported = ", ".join(sorted(SUPPORTED_INPUT_SOURCES))
        raise ValueError(f"unsupported input source {unknown[0]!r}; supported: {supported}")
    sources = [(name, _create_input_source(name, events, event_cfg)) for name in names]
    if len(sources) == 1:
        return sources[0]
    return ",".join(name for name, _source in sources), CompositeInput(sources)
