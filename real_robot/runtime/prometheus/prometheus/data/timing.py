from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class TimingTraceRecorder:
    """Small JSONL timing trace for live data-path latency checks."""

    def __init__(self, config: Mapping[str, Any] | None):
        config = dict(config or {})
        self.enabled = bool(config.get("enabled", False))
        self.path = Path(str(config.get("path", "/tmp/prometheus_data_timing.jsonl"))).expanduser()
        self._lock = threading.Lock()
        self._file: Any | None = None
        self._error = ""
        self._events = 0

    def start(self) -> None:
        if not self.enabled or self._file is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None

    def record(self, event: str, *, name: str, stamp_ns: int, fields: Mapping[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        payload = {
            "event": str(event),
            "name": str(name),
            "stamp_ns": int(stamp_ns),
        }
        if fields:
            payload.update(_jsonable_dict(fields))
        try:
            line = json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"
            with self._lock:
                if self._file is None:
                    return
                self._file.write(line)
                self._file.flush()
                self._events += 1
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "path": str(self.path) if self.enabled else "",
            "events": self._events,
            "error": self._error,
        }


def _jsonable_dict(fields: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _jsonable(value) for key, value in fields.items()}


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return _jsonable_dict(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)
