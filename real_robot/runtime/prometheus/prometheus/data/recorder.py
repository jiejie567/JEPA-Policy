from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from prometheus.data.raw_episode import (
    EPISODE_DIRS,
    RawEpisodeWriter,
    action_input_stamp_ns,
    atomic_json_dump,
    json_safe,
    parse_camera_image_name,
    parse_camera_info_name,
    parse_tactile_name,
    safe_name,
)
from prometheus.data.types import DataSample, DataStream


STOP = object()


@dataclass(frozen=True)
class RecordItem:
    kind: str
    payload: Any
    stamp_ns: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kind = str(self.kind).strip()
        if not kind:
            raise ValueError("record item kind must be non-empty")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "stamp_ns", int(self.stamp_ns))
        object.__setattr__(self, "metadata", dict(self.metadata))


class DataRecorder:
    def __init__(
        self,
        *,
        streams: Mapping[str, DataStream],
        config: Mapping[str, Any] | None,
        name: str,
        on_failure: Callable[[str], None] | None = None,
    ):
        self.streams = dict(streams)
        self.config = dict(config or {})
        self.name = str(name)
        self.on_failure = on_failure
        self._queues: dict[str, queue.Queue[Any]] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._episode_dir: Path | None = None
        self._recording = False
        self._failure = ""
        self._metadata: dict[str, Any] = {}
        self._counts: dict[str, int] = {}
        self._writer: RawEpisodeWriter | None = None
        self._lock = threading.RLock()
        self._paused = False

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("enabled", False))

    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def paused(self) -> bool:
        return self._paused

    def pause(self) -> None:
        if not self._recording:
            raise RuntimeError("data recorder has no active episode to pause")
        self._paused = True

    def resume(self) -> None:
        if not self._recording:
            raise RuntimeError("data recorder has no active episode to resume")
        self._paused = False

    def start(self) -> None:
        if not self.enabled or self._threads:
            return
        queue_size = int(self.config["queue_size"])
        if queue_size <= 0:
            raise ValueError("data.recording.queue_size must be positive")
        for group in self._queue_groups():
            item_queue: queue.Queue[Any] = queue.Queue(maxsize=queue_size)
            thread = threading.Thread(target=self._write_loop, args=(group,), name=f"{self.name}_{group}_writer", daemon=True)
            self._queues[group] = item_queue
            self._threads[group] = thread
            thread.start()

    def close(self) -> None:
        if self._recording:
            self.stop_episode(accept=False)
        for item_queue in self._queues.values():
            item_queue.put(STOP)
        for thread in self._threads.values():
            thread.join(timeout=2.0)
        self._threads.clear()
        self._queues.clear()

    def start_episode(self, episode_id: str, metadata: Mapping[str, Any] | None = None) -> Path:
        if not self.enabled:
            raise RuntimeError("data recording is disabled")
        if self._recording:
            raise RuntimeError("data recorder already has an active episode")

        episode_id = safe_name(str(episode_id).strip())
        if not episode_id:
            raise ValueError("episode_id must be non-empty")
        output_dir = Path(self.config["output_dir"]).expanduser()
        episode_dir = output_dir / episode_id
        episode_dir.mkdir(parents=True, exist_ok=False)
        for name in EPISODE_DIRS:
            (episode_dir / name).mkdir()

        self._episode_dir = episode_dir
        with self._lock:
            self._recording = True
            self._paused = False
            self._failure = ""
            self._metadata = dict(metadata or {})
            self._counts = {"items": 0, "dropped": 0}
            self._writer = RawEpisodeWriter(
                episode_dir,
                self.streams,
                fps=float(self.config["fps"]),
                enable_plots=bool(self.config.get("enable_plots", True)),
                joint_plot_ranges=self.config.get("joint_plot_ranges"),
                default_event_value=self.config.get("default_event_value"),
            )
        self.write_manifests("recording")
        return episode_dir

    def record_sample(self, sample: DataSample) -> bool:
        return self._append(RecordItem("sample", sample, sample.stamp_ns))

    def record_action_chunk(self, chunk: Any, metadata: Mapping[str, Any] | None = None) -> bool:
        return self._append(
            RecordItem(
                "action_chunk",
                chunk,
                action_input_stamp_ns(chunk),
                metadata=dict(metadata or {}),
            )
        )

    def record_action(
        self,
        action: Any,
        *,
        action_space: str,
        hz: float,
        stamp_ns: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        now_ns = time.time_ns()
        return self._append(
            RecordItem(
                "action",
                {"action": action, "action_space": str(action_space), "hz": float(hz)},
                now_ns if stamp_ns is None else int(stamp_ns),
                metadata=dict(metadata or {}),
            )
        )

    def record_event(
        self,
        name: str,
        value: Any = True,
        *,
        stamp_ns: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        now_ns = time.time_ns()
        return self._append(
            RecordItem(
                "event",
                {"name": str(name), "value": value},
                now_ns if stamp_ns is None else int(stamp_ns),
                metadata=dict(metadata or {}),
            )
        )

    def stop_episode(self, *, accept: bool = True) -> str:
        if not self._recording:
            return "idle"
        self._recording = False
        self._paused = False
        for item_queue in self._queues.values():
            item_queue.join()
        self._finalize_writer()
        status = "failed" if self._failure else ("saved" if accept else "rejected")
        self.write_manifests(status)
        return status

    def details(self) -> dict[str, Any]:
        return {
            "recording_enabled": self.enabled,
            "recording": self._recording,
            "paused": self._paused,
            "episode_dir": None if self._episode_dir is None else str(self._episode_dir),
            "failure": self._failure,
            "queue_size": sum(item_queue.qsize() for item_queue in self._queues.values()),
            "queues": {group: item_queue.qsize() for group, item_queue in self._queues.items()},
            "record_counts": dict(self._counts),
        }

    def write_manifests(self, status: str) -> None:
        if self._episode_dir is None:
            raise RuntimeError("data session has no active episode")
        common = {
            "status": status,
            "schema": "prometheus_raw_episode_v1",
            "updated_at_unix": time.time(),
            "episode_dir": str(self._episode_dir),
            "failure": self._failure,
            "counts": dict(self._counts),
            "metadata": json_safe(self._metadata),
        }
        with self._lock:
            manifests = self._require_writer().manifests(status, common)
        for name, payload in manifests.items():
            atomic_json_dump(payload, self._episode_dir / "manifests" / f"{name}.json")

    def _append(self, item: RecordItem) -> bool:
        if not self._recording or self._failure or self._paused:
            return False
        try:
            self._queue_for_item(item).put_nowait(item)
            return True
        except queue.Full:
            with self._lock:
                self._counts["dropped"] = self._counts.get("dropped", 0) + 1
            self._fail("data recording queue overflow")
            return False

    def _write_loop(self, group: str) -> None:
        item_queue = self._queues[group]
        while True:
            item = item_queue.get()
            try:
                if item is STOP:
                    return
                if not isinstance(item, RecordItem):
                    raise ValueError(f"data writer expected RecordItem, got {type(item).__name__}")
                self._write_item(item)
            except Exception as exc:
                self._fail(str(exc))
            finally:
                item_queue.task_done()

    def _write_item(self, item: RecordItem) -> None:
        if self._paused:
            return
        writer = self._require_writer()
        if item.kind == "sample":
            count_key = writer.write_sample(item.payload)
        elif item.kind == "action_chunk":
            count_key = writer.write_action_chunk(item.payload, stamp_ns=item.stamp_ns, metadata=item.metadata)
        elif item.kind == "action":
            payload = item.payload
            count_key = writer.write_action(
                payload["action"],
                stamp_ns=item.stamp_ns,
                action_space=payload["action_space"],
                hz=payload["hz"],
                metadata=item.metadata,
            )
        elif item.kind == "event":
            count_key = writer.write_event(item.payload, stamp_ns=item.stamp_ns, metadata=item.metadata)
        else:
            raise ValueError(f"unknown record item kind {item.kind!r}")
        with self._lock:
            self._counts[count_key] = self._counts.get(count_key, 0) + 1
            self._counts["items"] = self._counts.get("items", 0) + 1

    def _finalize_writer(self) -> None:
        try:
            with self._lock:
                self._require_writer().finalize()
        except Exception as exc:
            self._fail(str(exc))

    def _fail(self, message: str) -> None:
        with self._lock:
            if self._failure:
                return
            self._failure = str(message)
        if self.on_failure is not None:
            self.on_failure(self._failure)

    def _queue_groups(self) -> list[str]:
        groups = {"robot", "event", "action"}
        for name in self.streams:
            camera = parse_camera_image_name(name) or parse_camera_info_name(name)
            if camera is not None:
                groups.add(f"camera:{camera[0]}")
                continue
            tactile = parse_tactile_name(name)
            if tactile is not None:
                groups.add(f"tactile:{tactile[0]}")
        return sorted(groups)

    def _queue_for_item(self, item: RecordItem) -> queue.Queue[Any]:
        group = self._group_for_item(item)
        item_queue = self._queues.get(group)
        if item_queue is None:
            raise RuntimeError(f"data recording queue {group!r} has not been started")
        return item_queue

    def _group_for_item(self, item: RecordItem) -> str:
        if item.kind in {"action", "action_chunk"}:
            return "action"
        if item.kind == "event":
            return "event"
        if item.kind != "sample":
            return "event"
        name = str(getattr(item.payload, "name", ""))
        camera = parse_camera_image_name(name) or parse_camera_info_name(name)
        if camera is not None:
            return f"camera:{camera[0]}"
        tactile = parse_tactile_name(name)
        if tactile is not None:
            return f"tactile:{tactile[0]}"
        return "robot"

    def _require_writer(self) -> RawEpisodeWriter:
        if self._writer is None:
            raise RuntimeError("data recording episode has not been started")
        return self._writer
