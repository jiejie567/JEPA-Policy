from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from threading import RLock
from typing import Any, Mapping

from prometheus.data.buffer import DataBuffer
from prometheus.data.numpy_buffer import AsyncNumpyDecoder
from prometheus.data.recorder import DataRecorder
from prometheus.data.timing import TimingTraceRecorder
from prometheus.data.types import DataFrame, DataSample, DataStream, msg_stamp_ns, normalize_streams
from prometheus.sessions.session import BaseSession, SessionMode, SessionState, SessionStateError, SessionStatus


DEFAULT_DATA_HISTORY = 1
STAMP_SOURCES = frozenset({"header", "recv"})


class DataSession(BaseSession):
    """Workflow data plane: stream ingress, online views, recording, and visualization hooks."""

    def __init__(
        self,
        streams: Mapping[str, Any] | None = None,
        *,
        hardware: Any | None = None,
        history: int = DEFAULT_DATA_HISTORY,
        name: str = "data",
        mode: str | SessionMode = SessionMode.OWNED,
        bridge: Mapping[str, Any] | None = None,
        numpy: Mapping[str, Any] | None = None,
        recording: Mapping[str, Any] | None = None,
        timing: Mapping[str, Any] | None = None,
    ):
        super().__init__(name=name, mode=SessionMode(mode))
        if self.mode != SessionMode.OWNED:
            raise ValueError("DataSession currently supports only mode='owned'")

        if streams is not None:
            raw_streams = streams
        elif hardware is not None:
            raw_streams = hardware.streams
        else:
            raise ValueError("DataSession requires streams or hardware")

        self.streams = normalize_streams(raw_streams)
        self.buffer = DataBuffer(self.streams, history=history)
        self.bridge_config = dict(bridge or {})
        self.numpy_config = dict(numpy or {})
        self.timing = TimingTraceRecorder(timing)
        stamp_source = str(self.bridge_config.get("stamp_source", "header")).strip().lower()
        if stamp_source not in STAMP_SOURCES:
            raise ValueError(f"data bridge stamp_source must be one of {sorted(STAMP_SOURCES)}")
        self.stamp_source = stamp_source
        self.recorder = DataRecorder(
            streams=self.streams,
            config=recording,
            name=self.name,
            on_failure=self._recording_failed,
        )
        self.numpy = AsyncNumpyDecoder(
            self.streams,
            history=int(self.numpy_config.get("history", history)),
            enabled=bool(self.numpy_config.get("enabled", True)),
            recording=self.numpy_config.get("recording"),
            timing=self.timing,
        )
        self._lock = RLock()
        self._bridge: Any | None = None
        self._closed = False

    @property
    def history(self) -> int:
        return self.buffer.history

    @property
    def recording_enabled(self) -> bool:
        return self.recorder.enabled

    def start(self) -> None:
        status = self.status()
        if status.state == SessionState.READY:
            return
        if status.state != SessionState.NEW:
            raise SessionStateError(f"cannot start data session from state {status.state.value}")
        self._set_status(state=SessionState.STARTING, message="starting data session", details=self._status_details())
        try:
            self.recorder.start()
            self.timing.start()
            self.numpy.start()
            if self.streams and bool(self.bridge_config["enabled"]):
                from prometheus.nodes.data_bridge import RosDataBridge

                self._bridge = RosDataBridge(
                    data=self,
                    name=str(self.bridge_config["name"]),
                    wait_all_streams=bool(self.bridge_config["wait_all_streams"]),
                    qos_depth=int(self.bridge_config["qos_depth"]),
                    sensor_qos=str(self.bridge_config.get("sensor_qos", "sensor")),
                    print_camera_timestamps=bool(self.bridge_config.get("print_camera_timestamps", False)),
                    print_camera_rates=bool(self.bridge_config.get("print_camera_rates", False)),
                    camera_rate_report_s=float(self.bridge_config.get("camera_rate_report_s", 2.0)),
                )
                self._bridge.start()
            self._set_status(state=SessionState.READY, message="data session ready", details=self._status_details())
        except Exception as exc:
            self.stop()
            self._mark_failed(str(exc), details=self._status_details())
            raise

    def wait_ready(self, timeout_s: float) -> SessionStatus:
        status = self.status()
        if status.state == SessionState.NEW:
            raise SessionStateError(f"data session {self.name!r} has not been started")
        if status.state == SessionState.FAILED:
            raise SessionStateError(status.message or "data session failed")
        if self._bridge is not None:
            self._bridge.wait_ready(float(timeout_s))
        self._set_status(state=SessionState.READY, message="data session ready", details=self._status_details())
        return self.status()

    def stop(self) -> None:
        status = self.status()
        if status.state in {SessionState.NEW, SessionState.STOPPED, SessionState.CLOSED}:
            return
        if status.state == SessionState.STOPPING:
            return
        self._set_status(state=SessionState.STOPPING, message="stopping data session", details=self._status_details())
        self.stop_ingress()
        self.recorder.close()
        self.numpy.stop()
        self.timing.close()
        self._set_status(state=SessionState.STOPPED, message="data session stopped", details=self._status_details())

    def stop_ingress(self) -> None:
        if self._bridge is not None:
            self._bridge.close()
            self._bridge = None

    def close(self) -> None:
        status = self.status()
        if status.state == SessionState.CLOSED:
            return
        if status.state not in {SessionState.NEW, SessionState.STOPPED}:
            self.stop()
        with self._lock:
            self._closed = True
        self._set_status(state=SessionState.CLOSED, message="data session closed", details=self._status_details())

    def callback(self, name: str):
        self._stream_name(name)

        def _callback(msg: Any) -> DataSample:
            return self.ingest(name, msg)

        return _callback

    def ingest(
        self,
        name: str,
        msg: Any,
        *,
        stamp_ns: int | None = None,
        recv_ns: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> DataSample:
        name = self._stream_name(name)
        if self._closed:
            raise RuntimeError("data session is closed")
        ingest_start_ns = time.time_ns()
        recv = ingest_start_ns if recv_ns is None else int(recv_ns)
        if stamp_ns is None:
            if self.stamp_source == "recv":
                stamp = recv
            else:
                stamp = msg_stamp_ns(msg)
                if stamp <= 0:
                    stamp = recv
        else:
            stamp = int(stamp_ns)
        sample_metadata = dict(metadata or {})
        sample_metadata.setdefault("subscriber_recv_ns", recv)
        sample_metadata.setdefault("ingest_start_ns", ingest_start_ns)
        sample_metadata.setdefault("publish_ros_ns", msg_stamp_ns(msg))
        sample = DataSample(
            name=name,
            msg=msg,
            stamp_ns=stamp,
            recv_ns=recv,
            stream=self.streams[name],
            metadata=sample_metadata,
        )
        self.buffer.append(sample)
        buffer_append_ns = time.time_ns()
        sample.metadata["buffer_append_ns"] = buffer_append_ns
        self.timing.record(
            "buffer_append",
            name=sample.name,
            stamp_ns=sample.stamp_ns,
            fields={
                **dict(sample.metadata),
                "recv_to_buffer_ms": (buffer_append_ns - sample.recv_ns) / 1_000_000.0,
                "publish_ros_to_recv_ms": (
                    (sample.recv_ns - int(sample.metadata["publish_ros_ns"])) / 1_000_000.0
                    if int(sample.metadata.get("publish_ros_ns", 0)) > 0
                    else None
                ),
            },
        )
        self.numpy.enqueue(sample)
        if self.recorder.recording:
            self.recorder.record_sample(sample)
        return sample

    def ready(self, names: Sequence[str] | None = None, *, count: int = 1) -> bool:
        return self.buffer.ready(names, count=count)

    def latest(self, names: Sequence[str] | None = None) -> dict[str, DataSample]:
        return self.buffer.latest(names)

    def latest_numpy(self, names: Sequence[str] | None = None) -> dict[str, Any]:
        return self.numpy.buffer.latest(names)

    def frame(
        self,
        *,
        anchor: str,
        names: Sequence[str] | None = None,
        slop_ms: float,
        anchor_index: int = -1,
    ) -> DataFrame:
        return self.buffer.frame(anchor=anchor, names=names, slop_ms=slop_ms, anchor_index=anchor_index)

    def frame_numpy(
        self,
        *,
        anchor: str,
        names: Sequence[str] | None = None,
        slop_ms: float,
        anchor_index: int = -1,
    ) -> Any:
        return self.numpy.buffer.frame(anchor=anchor, names=names, slop_ms=slop_ms, anchor_index=anchor_index)

    def window(
        self,
        *,
        anchor: str,
        names: Sequence[str] | None = None,
        count: int,
        stride: int = 1,
        slop_ms: float,
    ) -> list[DataFrame]:
        return self.buffer.window(anchor=anchor, names=names, count=count, stride=stride, slop_ms=slop_ms)

    def window_numpy(
        self,
        *,
        anchor: str,
        names: Sequence[str] | None = None,
        count: int,
        stride: int = 1,
        slop_ms: float,
        wait_latest: bool = False,
        timeout_ms: float = 0.0,
        min_anchor_stamp_ns: int | None = None,
    ) -> list[Any]:
        raw_anchor_stamp_ns = 0
        wait_ms = 0.0
        minimum_stamp_wait_ms = 0.0
        used_stale_window = False
        raw_sample_stamps: dict[str, int] = {}
        if min_anchor_stamp_ns is not None:
            minimum_stamp = int(min_anchor_stamp_ns)
            if minimum_stamp <= 0:
                raise ValueError("min_anchor_stamp_ns must be positive")
            reached, minimum_stamp_wait_ms = self.numpy.buffer.wait_until_stamp(
                anchor,
                minimum_stamp,
                timeout_ms=timeout_ms,
            )
            if not reached:
                latest_stamp_ns = 0
                try:
                    latest_stamp_ns = self.numpy.buffer.latest_stamp_ns(anchor)
                except ValueError:
                    pass
                raise TimeoutError(
                    f"numpy anchor {anchor!r} did not reach required stamp "
                    f"{minimum_stamp} within {float(timeout_ms):.1f} ms; "
                    f"latest={latest_stamp_ns}"
                )
        if wait_latest:
            wait_start = time.perf_counter()
            deadline = time.monotonic() + max(0.0, float(timeout_ms)) / 1000.0
            try:
                raw_frame = self.buffer.frame(anchor=anchor, names=names, slop_ms=slop_ms)
                raw_anchor_stamp_ns = int(raw_frame.stamp_ns)
                raw_sample_stamps = {
                    sample_name: int(sample.stamp_ns)
                    for sample_name, sample in raw_frame.samples.items()
                }
                for sample_name, target_stamp_ns in raw_sample_stamps.items():
                    remaining_ms = max(0.0, (deadline - time.monotonic()) * 1000.0)
                    decoded, _sample_wait_ms = self.numpy.buffer.wait_until_stamp(
                        sample_name,
                        target_stamp_ns,
                        timeout_ms=remaining_ms,
                    )
                    if not decoded:
                        used_stale_window = True
                        break
                wait_ms = (time.perf_counter() - wait_start) * 1000.0
            except ValueError:
                used_stale_window = True
                wait_ms = (time.perf_counter() - wait_start) * 1000.0
        frames = self.numpy.buffer.window(anchor=anchor, names=names, count=count, stride=stride, slop_ms=slop_ms)
        if wait_latest:
            latest_numpy_stamp_ns = int(frames[-1].samples[anchor].stamp_ns)
            latest_sample_stamps = {
                sample_name: int(sample.stamp_ns)
                for sample_name, sample in frames[-1].samples.items()
            }
            self.timing.record(
                "window_numpy",
                name=str(anchor),
                stamp_ns=latest_numpy_stamp_ns,
                fields={
                    "raw_anchor_stamp_ns": raw_anchor_stamp_ns,
                    "raw_sample_stamps": raw_sample_stamps,
                    "latest_numpy_stamp_ns": latest_numpy_stamp_ns,
                    "latest_sample_stamps": latest_sample_stamps,
                    "numpy_wait_ms": wait_ms,
                    "minimum_anchor_stamp_ns": min_anchor_stamp_ns,
                    "minimum_stamp_wait_ms": minimum_stamp_wait_ms,
                    "raw_to_numpy_gap_ms": (
                        (raw_anchor_stamp_ns - latest_numpy_stamp_ns) / 1_000_000.0
                        if raw_anchor_stamp_ns > 0
                        else None
                    ),
                    "used_stale_window": used_stale_window,
                    "window_count": int(count),
                    "window_stride": int(stride),
                },
            )
        return frames

    def counts(self) -> dict[str, int]:
        return self.buffer.counts()

    def numpy_counts(self) -> dict[str, int]:
        return self.numpy.buffer.counts()

    def image_png(self, stream: str, *, t_ms: int | None = None, index: int | None = None) -> bytes:
        if t_ms is not None:
            raise NotImplementedError("live numpy image lookup by timestamp is not implemented")
        from prometheus.data.visualizer import encode_png
        if self.numpy.enabled:
            try:
                sample = self.numpy.buffer.latest([stream])[stream] if index is None else self.numpy.buffer.sample_at(stream, index)
                return encode_png(sample.data)
            except Exception:
                pass
        from prometheus.data.raw_episode import image_to_array

        sample = self.latest([stream])[stream]
        image, encoding = image_to_array(sample.msg)
        if encoding == "bgr8":
            image = image[:, :, ::-1]
        return encode_png(image)

    def latest_skew_ms(
        self,
        *,
        anchor: str,
        names: Sequence[str] | None = None,
    ) -> dict[str, float]:
        anchor_sample = self.latest([anchor])[anchor]
        selected = tuple(self.stream_names()) if names is None else tuple(names)
        skew: dict[str, float] = {}
        for name in selected:
            sample = self.latest([name])[name]
            skew[name] = (sample.stamp_ns - anchor_sample.stamp_ns) / 1_000_000.0
        return skew

    def stream_names(self) -> tuple[str, ...]:
        return tuple(self.streams)

    def start_episode(self, episode_id: str, metadata: Mapping[str, Any] | None = None) -> Path:
        if not self.recording_enabled:
            raise SessionStateError("data recording is disabled")
        if self.recorder.recording:
            raise SessionStateError("data session already has an active episode")
        status = self.status()
        if not status.ready:
            raise SessionStateError(status.message or f"data session is {status.state.value}")
        episode_dir = self.recorder.start_episode(episode_id, metadata=metadata)
        self._set_status(message=f"recording {episode_dir.name}", details=self._status_details())
        return episode_dir

    def record_action_chunk(self, chunk: Any, metadata: Mapping[str, Any] | None = None) -> bool:
        return self.recorder.record_action_chunk(chunk, metadata=metadata)

    def record_policy_action_chunk(self, chunk: Any, metadata: Mapping[str, Any] | None = None) -> bool:
        if self.recorder.recording:
            return self.recorder.record_action_chunk(chunk, metadata=metadata)
        return self.numpy.recorder.record_action_chunk(chunk, metadata=metadata)

    def record_action(
        self,
        action: Any,
        *,
        action_space: str,
        hz: float,
        stamp_ns: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        return self.recorder.record_action(
            action,
            action_space=action_space,
            hz=hz,
            stamp_ns=stamp_ns,
            metadata=metadata,
        )

    def record_event(
        self,
        name: str,
        value: Any = True,
        *,
        stamp_ns: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        return self.recorder.record_event(name, value, stamp_ns=stamp_ns, metadata=metadata)

    def stop_episode(self, *, accept: bool = True) -> None:
        status = self.recorder.stop_episode(accept=accept)
        if status != "idle":
            self._set_status(message=f"episode {status}", details=self._status_details())

    def visualize(self, **kwargs: Any) -> Any:
        from prometheus.data.visualizer import DataVisualizer

        return DataVisualizer(self, **kwargs)

    @staticmethod
    def open_episode(episode_dir: str | Path) -> Any:
        from prometheus.data.episode import RawEpisodeReader

        return RawEpisodeReader(episode_dir)

    def _stream_name(self, name: str) -> str:
        if name not in self.streams:
            raise KeyError(f"unknown data stream name: {name!r}")
        return name

    def _recording_failed(self, message: str) -> None:
        self._mark_failed(message, details=self._status_details())

    def _status_details(self) -> dict[str, Any]:
        details = {
            "history": self.history,
            "stamp_source": self.stamp_source,
            "streams": {name: stream.as_dict() for name, stream in self.streams.items()},
            "counts": self.counts(),
            "numpy": self.numpy.status(),
            "timing": self.timing.status(),
            "bridge": None if self._bridge is None else self._bridge.status(),
            "closed": self._closed,
        }
        details.update(self.recorder.details())
        return details
