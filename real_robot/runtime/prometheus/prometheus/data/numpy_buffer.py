from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import json
import numpy as np

from prometheus.data.types import DataStream
from prometheus.data.timing import TimingTraceRecorder
from prometheus.data.types import DataSample


@dataclass(frozen=True)
class NumpySample:
    name: str
    data: Any
    stamp_ns: int
    recv_ns: int
    stream: DataStream
    encoding: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NumpyFrame:
    stamp_ns: int
    samples: dict[str, NumpySample]
    skew_ms: dict[str, float]


class NumpyBuffer:
    def __init__(self, streams: dict[str, DataStream], *, history: int):
        if int(history) <= 0:
            raise ValueError("numpy buffer history must be positive")
        self.history = int(history)
        self.streams = dict(streams)
        self.buffers = {name: deque(maxlen=self.history) for name in self.streams}
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)

    def append(self, sample: NumpySample) -> None:
        with self._condition:
            self.buffers[self._stream_name(sample.name)].append(sample)
            self._condition.notify_all()

    def ready(self, names: Sequence[str] | None = None, *, count: int = 1) -> bool:
        if count <= 0:
            raise ValueError("count must be positive")
        with self._lock:
            return all(len(self.buffers[name]) >= count for name in self._names(names))

    def latest(self, names: Sequence[str] | None = None) -> dict[str, NumpySample]:
        with self._lock:
            return {name: self._sample_at(name, -1) for name in self._names(names)}

    def sample_at(self, name: str, index: int) -> NumpySample:
        with self._lock:
            return self._sample_at(name, index)

    def frame(
        self,
        *,
        anchor: str,
        names: Sequence[str] | None = None,
        slop_ms: float,
        anchor_index: int = -1,
    ) -> NumpyFrame:
        with self._lock:
            anchor_sample = self._sample_at(anchor, anchor_index)
            return self._frame_at(anchor_sample.stamp_ns, names=self._names(names), slop_ms=slop_ms)

    def window(
        self,
        *,
        anchor: str,
        names: Sequence[str] | None = None,
        count: int,
        stride: int = 1,
        slop_ms: float,
    ) -> list[NumpyFrame]:
        if count <= 0:
            raise ValueError("count must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")
        with self._lock:
            anchor = self._stream_name(anchor)
            anchor_samples = list(self.buffers[anchor])
            needed = (count - 1) * stride + 1
            if len(anchor_samples) < needed:
                raise ValueError(f"numpy anchor {anchor!r} needs {needed} samples, got {len(anchor_samples)}")
            selected = list(reversed(anchor_samples[-1 : -needed - 1 : -stride]))
            selected_names = self._names(names)
            return [
                self._frame_at(sample.stamp_ns, names=selected_names, slop_ms=slop_ms)
                for sample in selected
            ]

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {name: len(buffer) for name, buffer in self.buffers.items()}

    def latest_stamp_ns(self, name: str) -> int:
        with self._lock:
            return int(self._sample_at(name, -1).stamp_ns)

    def wait_until_stamp(self, name: str, stamp_ns: int, *, timeout_ms: float) -> tuple[bool, float]:
        deadline = time.monotonic() + max(0.0, float(timeout_ms)) / 1000.0
        start = time.perf_counter()
        with self._condition:
            name = self._stream_name(name)
            while True:
                latest = self.buffers[name][-1].stamp_ns if self.buffers[name] else 0
                if int(latest) >= int(stamp_ns):
                    return True, (time.perf_counter() - start) * 1000.0
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False, (time.perf_counter() - start) * 1000.0
                self._condition.wait(timeout=remaining)

    def _frame_at(self, stamp_ns: int, *, names: tuple[str, ...], slop_ms: float) -> NumpyFrame:
        slop_ns = int(float(slop_ms) * 1_000_000)
        if slop_ns < 0:
            raise ValueError("slop_ms must be non-negative")
        samples = {name: self._nearest(name, stamp_ns, slop_ns) for name in names}
        skew_ms = {name: (sample.stamp_ns - stamp_ns) / 1_000_000.0 for name, sample in samples.items()}
        return NumpyFrame(stamp_ns=stamp_ns, samples=samples, skew_ms=skew_ms)

    def _nearest(self, name: str, stamp_ns: int, slop_ns: int) -> NumpySample:
        candidates = [sample for sample in self.buffers[self._stream_name(name)] if sample.stamp_ns > 0]
        if not candidates:
            raise ValueError(f"numpy stream {name!r} has no timestamped samples")
        sample = min(candidates, key=lambda item: abs(item.stamp_ns - stamp_ns))
        skew_ns = abs(sample.stamp_ns - stamp_ns)
        if skew_ns > slop_ns:
            raise ValueError(
                f"numpy stream {name!r} nearest sample skew {skew_ns / 1_000_000.0:.3f} ms "
                f"exceeds slop {slop_ns / 1_000_000.0:.3f} ms"
            )
        return sample

    def _sample_at(self, name: str, index: int) -> NumpySample:
        name = self._stream_name(name)
        if not self.buffers[name]:
            raise ValueError(f"numpy stream {name!r} has no samples")
        try:
            return self.buffers[name][index]
        except IndexError as exc:
            raise ValueError(f"numpy stream {name!r} has {len(self.buffers[name])} samples, cannot read index {index}") from exc

    def _stream_name(self, name: str) -> str:
        if name not in self.buffers:
            raise KeyError(f"unknown numpy stream name: {name!r}")
        return name

    def _names(self, names: Sequence[str] | None) -> tuple[str, ...]:
        if isinstance(names, str):
            raise TypeError("names must be an iterable of stream names, not a string")
        selected = tuple(self.buffers) if names is None else tuple(names)
        missing = [name for name in selected if name not in self.buffers]
        if missing:
            raise KeyError(f"unknown numpy stream names: {missing}")
        return selected


class AsyncNumpyDecoder:
    def __init__(
        self,
        streams: dict[str, DataStream],
        *,
        history: int,
        enabled: bool = False,
        recording: Mapping[str, Any] | None = None,
        timing: TimingTraceRecorder | None = None,
    ):
        self.enabled = bool(enabled)
        self.buffer = NumpyBuffer(streams, history=history)
        self.recorder = NumpyRecorder(recording, streams=streams)
        self.timing = timing
        self._pending: dict[str, deque[Any]] = {}
        self._condition = threading.Condition()
        self._stop = False
        self._thread: threading.Thread | None = None
        self._error = ""
        self._decoded = 0
        self._dropped = 0

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self.recorder.start()
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="data_numpy_decoder", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self.recorder.close()

    def enqueue(self, sample: Any) -> None:
        if not self.enabled or not _can_decode(sample.msg):
            return
        with self._condition:
            pending = self._pending.setdefault(sample.name, deque(maxlen=self.buffer.history))
            before = len(pending)
            pending.append(sample)
            if before == self.buffer.history:
                self._dropped += 1
            self._condition.notify()

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "error": self._error,
            "decoded": self._decoded,
            "dropped": self._dropped,
            "counts": self.buffer.counts(),
            "pending": sum(len(items) for items in self._pending.values()),
            "recording": self.recorder.status(),
        }

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._stop:
                    self._condition.wait()
                if self._stop and not self._pending:
                    return
                items = [sample for pending in self._pending.values() for sample in pending]
                self._pending.clear()
            for sample in items:
                try:
                    decode_start_ns = time.time_ns()
                    decoded = decode_image_sample(sample)
                    decode_done_ns = time.time_ns()
                    decoded = NumpySample(
                        name=decoded.name,
                        data=decoded.data,
                        stamp_ns=decoded.stamp_ns,
                        recv_ns=decoded.recv_ns,
                        stream=decoded.stream,
                        encoding=decoded.encoding,
                        metadata={
                            **dict(decoded.metadata),
                            "decode_start_ns": decode_start_ns,
                            "decode_done_ns": decode_done_ns,
                        },
                    )
                    self.buffer.append(decoded)
                    self.recorder.record(decoded)
                    self._decoded += 1
                    if self.timing is not None:
                        self.timing.record(
                            "numpy_decode",
                            name=decoded.name,
                            stamp_ns=decoded.stamp_ns,
                            fields={
                                **dict(decoded.metadata),
                                "decode_latency_ms": (decode_done_ns - int(decoded.recv_ns)) / 1_000_000.0,
                                "decode_cost_ms": (decode_done_ns - decode_start_ns) / 1_000_000.0,
                            },
                        )
                except Exception as exc:
                    self._error = f"{sample.name}: {type(exc).__name__}: {exc}"


def decode_image_sample(sample: Any) -> NumpySample:
    data, encoding = msg_to_numpy(sample.msg)
    return NumpySample(
        name=str(sample.name),
        data=data,
        stamp_ns=int(sample.stamp_ns),
        recv_ns=int(sample.recv_ns),
        stream=sample.stream,
        encoding=encoding,
        metadata=dict(sample.metadata),
    )


def image_msg_to_numpy(msg: Any) -> tuple[np.ndarray, str]:
    encoding = str(msg.encoding).lower()
    dtype, channels = _image_layout(encoding)
    height, width = int(msg.height), int(msg.width)
    row_values = int(msg.step) // np.dtype(dtype).itemsize
    raw = np.frombuffer(bytes(msg.data), dtype=dtype).reshape(height, row_values)
    if channels == 1:
        return raw[:, :width].copy(), encoding
    image = raw[:, : width * channels].reshape(height, width, channels)
    if encoding == "bgr8":
        return image[:, :, ::-1].copy(), "rgb8"
    if encoding == "bgra8":
        return image[:, :, 2::-1].copy(), "rgb8"
    if encoding == "rgba8":
        return image[:, :, :3].copy(), "rgb8"
    return image.copy(), encoding


def msg_to_numpy(msg: Any) -> tuple[Any, str]:
    if _looks_like_image(msg):
        return image_msg_to_numpy(msg)
    if _looks_like_pointcloud2(msg):
        return pointcloud2_to_numpy(msg), "pointcloud2"
    if _looks_like_joint_state(msg):
        return joint_state_to_numpy(msg), "joint_state"
    if _looks_like_pose_stamped(msg):
        return pose_stamped_to_numpy(msg), "pose_stamped"
    raise ValueError(f"unsupported message for numpy decode: {type(msg).__name__}")


def joint_state_to_numpy(msg: Any) -> dict[str, np.ndarray]:
    return {
        "position": np.asarray(getattr(msg, "position", ()), dtype=np.float32).copy(),
        "velocity": np.asarray(getattr(msg, "velocity", ()), dtype=np.float32).copy(),
        "effort": np.asarray(getattr(msg, "effort", ()), dtype=np.float32).copy(),
        "name": np.asarray([str(item) for item in getattr(msg, "name", ())], dtype=object),
    }


def pose_stamped_to_numpy(msg: Any) -> np.ndarray:
    pose = msg.pose
    return np.asarray(
        [
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.w,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
        ],
        dtype=np.float32,
    )


def pointcloud2_to_numpy(msg: Any) -> np.ndarray:
    field_names = tuple(field.name for field in getattr(msg, "fields", ()))
    selected = tuple(name for name in ("x", "y", "z", "dx", "dy", "dz") if name in field_names)
    if not selected:
        raise ValueError("PointCloud2 has no supported fields")
    import sensor_msgs_py.point_cloud2 as pc2

    points = pc2.read_points(msg, field_names=selected, skip_nans=False)
    if isinstance(points, np.ndarray) and points.dtype.fields:
        array = np.column_stack([points[field] for field in selected]).astype(np.float32)
    else:
        array = np.asarray(points if isinstance(points, np.ndarray) else list(points), dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(0, len(selected)) if array.size == 0 else array.reshape(-1, len(selected))
    return np.ascontiguousarray(array, dtype=np.float32)


def _image_layout(encoding: str) -> tuple[Any, int]:
    if encoding in {"rgb8", "bgr8"}:
        return np.uint8, 3
    if encoding in {"rgba8", "bgra8"}:
        return np.uint8, 4
    if encoding in {"mono8", "8uc1"}:
        return np.uint8, 1
    if encoding in {"16uc1", "mono16"}:
        return np.uint16, 1
    if encoding == "32fc1":
        return np.float32, 1
    raise ValueError(f"unsupported image encoding {encoding!r}")


def _looks_like_image(msg: Any) -> bool:
    return all(hasattr(msg, name) for name in ("height", "width", "encoding", "step", "data"))


def _looks_like_joint_state(msg: Any) -> bool:
    return all(hasattr(msg, name) for name in ("position", "velocity", "effort"))


def _looks_like_pose_stamped(msg: Any) -> bool:
    pose = getattr(msg, "pose", None)
    return hasattr(pose, "position") and hasattr(pose, "orientation")


def _looks_like_pointcloud2(msg: Any) -> bool:
    return all(hasattr(msg, name) for name in ("fields", "point_step", "row_step", "data"))


def _can_decode(msg: Any) -> bool:
    return _looks_like_image(msg) or _looks_like_pointcloud2(msg) or _looks_like_joint_state(msg) or _looks_like_pose_stamped(msg)


class NumpyRecorder:
    def __init__(self, config: Mapping[str, Any] | None, *, streams: Mapping[str, DataStream]):
        config = dict(config or {})
        self.enabled = bool(config.get("enabled", False))
        self.output_dir = Path(str(config.get("output_dir", "/tmp/prometheus_numpy_buffer"))).expanduser()
        self.run_id = str(config.get("run_id", f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}"))
        self.format = str(config.get("format", "npz")).strip().lower()
        if self.format in {"mp4_pickle", "raw", "raw_episode"}:
            self.format = "raw_episode"
        elif self.format != "npz":
            raise ValueError("numpy recording format must be 'npz' or 'raw_episode'")
        self.root = self.output_dir / self.run_id
        self.streams = dict(streams)
        self.fps = float(config.get("fps", 30.0))
        self.enable_plots = bool(config.get("enable_plots", True))
        self.joint_plot_ranges = config.get("joint_plot_ranges")
        self.default_event_value = config.get("default_event_value", 0)
        self._counts: dict[str, int] = {}
        self._timestamps_file: Any | None = None
        self._manifest_path = self.root / "manifest.json"
        self._error = ""
        self._raw_writer: Any | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        if self.format == "raw_episode":
            from prometheus.data.raw_episode import EPISODE_DIRS, RawEpisodeWriter

            for name in EPISODE_DIRS:
                (self.root / name).mkdir(exist_ok=True)
            self._raw_writer = RawEpisodeWriter(
                self.root,
                self.streams,
                fps=self.fps,
                enable_plots=self.enable_plots,
                joint_plot_ranges=self.joint_plot_ranges,
                default_event_value=self.default_event_value,
            )
        else:
            self._timestamps_file = (self.root / "timestamps.jsonl").open("a", encoding="utf-8")
        self._write_manifest(status="running")

    def record(self, sample: NumpySample) -> None:
        if not self.enabled:
            return
        try:
            index = self._counts.get(sample.name, 0)
            self._counts[sample.name] = index + 1
            if self.format == "raw_episode":
                self._record_raw_episode(sample)
                return
            stream_dir = self.root / sample.name
            stream_dir.mkdir(parents=True, exist_ok=True)
            rel_path = Path(sample.name) / f"{index:06d}.npz"
            abs_path = self.root / rel_path
            if isinstance(sample.data, Mapping):
                arrays = {str(key): np.asarray(value) for key, value in sample.data.items()}
                np.savez_compressed(abs_path, **arrays)
            else:
                np.savez_compressed(abs_path, data=np.asarray(sample.data))
            entry = {
                "stream": sample.name,
                "index": index,
                "file": str(rel_path),
                "stamp_ns": int(sample.stamp_ns),
                "recv_ns": int(sample.recv_ns),
                "encoding": sample.encoding,
                "summary": _array_summary(sample.data),
            }
            if self._timestamps_file is not None:
                self._timestamps_file.write(json.dumps(entry, separators=(",", ":")) + "\n")
                self._timestamps_file.flush()
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"

    def record_action_chunk(self, chunk: Any, metadata: Mapping[str, Any] | None = None) -> bool:
        if not self.enabled:
            return False
        try:
            input_stamp_ns = _action_input_stamp_ns(chunk)
            action_dir = self.root / "action"
            action_dir.mkdir(parents=True, exist_ok=True)
            actions = np.asarray(chunk.actions, dtype=np.float32).copy()
            by_input_path = action_dir / "action_chunk_by_input_timestamp.pkl"
            metadata_path = action_dir / "action_chunk_metadata.pkl"
            by_input = _read_pickle_mapping(by_input_path)
            meta_by_input = _read_pickle_mapping(metadata_path)
            by_input[int(input_stamp_ns)] = actions
            meta_by_input[int(input_stamp_ns)] = {
                "input_stamp_ns": int(input_stamp_ns),
                "action_space": str(chunk.action_space),
                "hz": float(chunk.hz),
                "horizon": int(actions.shape[0]),
                "action_dim": int(actions.shape[1]),
                "chunk_metadata": _json_safe(dict(getattr(chunk, "metadata", {}))),
                "metadata": _json_safe(dict(metadata or {})),
            }
            _atomic_pickle_dump(by_input, by_input_path)
            _atomic_pickle_dump(meta_by_input, metadata_path)
            if self._raw_writer is not None:
                self._raw_writer.write_action_chunk(chunk, stamp_ns=int(input_stamp_ns), metadata=dict(metadata or {}))
            self._counts["policy_action_chunks"] = self._counts.get("policy_action_chunks", 0) + 1
            return True
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            return False

    def close(self) -> None:
        if self._timestamps_file is not None:
            self._timestamps_file.close()
            self._timestamps_file = None
        if self._raw_writer is not None:
            try:
                self._raw_writer.finalize()
                self._write_raw_manifests(status="complete")
            except Exception as exc:
                self._error = f"{type(exc).__name__}: {exc}"
                self._write_raw_manifests(status="error")
            finally:
                self._raw_writer = None
        if self.enabled:
            self._write_manifest(status="error" if self._error else "complete")

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "output_dir": str(self.root) if self.enabled else "",
            "format": self.format,
            "counts": dict(self._counts),
            "error": self._error,
        }

    def _record_raw_episode(self, sample: NumpySample) -> None:
        if self._raw_writer is None:
            return
        msg = _numpy_sample_to_msg(sample)
        data_sample = DataSample(
            name=sample.name,
            msg=msg,
            stamp_ns=sample.stamp_ns,
            recv_ns=sample.recv_ns,
            stream=sample.stream,
            metadata=dict(sample.metadata),
        )
        self._raw_writer.write_sample(data_sample)

    def _write_raw_manifests(self, *, status: str) -> None:
        if self._raw_writer is None:
            return
        from prometheus.data.raw_episode import atomic_json_dump, json_safe

        common = {
            "status": status,
            "schema": "prometheus_raw_episode_v1",
            "updated_at_unix": time.time(),
            "episode_dir": str(self.root),
            "failure": self._error,
            "counts": dict(self._counts),
            "metadata": {"source": "DataSession.numpy_recording", "format": self.format},
        }
        for name, manifest in self._raw_writer.manifests(status, common).items():
            atomic_json_dump(json_safe(manifest), self.root / "manifests" / f"{name}.json")

    def _write_manifest(self, *, status: str) -> None:
        payload = {
            "status": status,
            "format": "prometheus_raw_episode_v1" if self.format == "raw_episode" else "temporary_numpy_npz_v1",
            "counts": dict(self._counts),
            "timestamps": "camera/timestamp.pkl" if self.format == "raw_episode" else "timestamps.jsonl",
        }
        self._manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _numpy_sample_to_msg(sample: NumpySample) -> Any:
    if isinstance(sample.data, Mapping) and {"position", "velocity", "effort"}.issubset(sample.data):
        return SimpleNamespace(
            header=_header(sample.stamp_ns),
            name=[str(item) for item in np.asarray(sample.data.get("name", ())).tolist()],
            position=np.asarray(sample.data["position"], dtype=np.float32).tolist(),
            velocity=np.asarray(sample.data["velocity"], dtype=np.float32).tolist(),
            effort=np.asarray(sample.data["effort"], dtype=np.float32).tolist(),
        )
    if str(sample.encoding) == "pose_stamped":
        return _pose_array_to_msg(sample)
    if str(sample.encoding) == "pointcloud2":
        return np.asarray(sample.data, dtype=np.float32)
    if isinstance(sample.data, np.ndarray):
        return _image_array_to_msg(sample)
    array = np.asarray(sample.data, dtype=np.float32).reshape(-1)
    if array.size == 7:
        return _pose_array_to_msg(sample)
    raise ValueError(f"cannot convert numpy sample {sample.name!r} with encoding {sample.encoding!r} to raw episode msg")


def _pose_array_to_msg(sample: NumpySample) -> Any:
    array = np.asarray(sample.data, dtype=np.float32).reshape(-1)
    if array.size != 7:
        raise ValueError(f"pose sample {sample.name!r} must have 7 values, got {array.size}")
    return SimpleNamespace(
        header=_header(sample.stamp_ns),
        pose=SimpleNamespace(
            position=SimpleNamespace(x=float(array[0]), y=float(array[1]), z=float(array[2])),
            orientation=SimpleNamespace(w=float(array[3]), x=float(array[4]), y=float(array[5]), z=float(array[6])),
        ),
    )


def _image_array_to_msg(sample: NumpySample) -> Any:
    image = np.asarray(sample.data)
    encoding = str(sample.encoding).lower()
    if image.dtype != np.uint8:
        image = image.astype(np.uint8, copy=False)
    if image.ndim == 2:
        encoding = "mono8"
        step = int(image.shape[1])
    elif image.ndim == 3 and image.shape[2] == 3:
        if encoding not in {"rgb8", "bgr8"}:
            encoding = "rgb8"
        step = int(image.shape[1] * 3)
    else:
        raise ValueError(f"image sample {sample.name!r} must be HxW or HxWx3 uint8, got {image.shape}")
    return SimpleNamespace(
        header=_header(sample.stamp_ns),
        height=int(image.shape[0]),
        width=int(image.shape[1]),
        encoding=encoding,
        step=step,
        data=np.ascontiguousarray(image).tobytes(),
    )


def _header(stamp_ns: int) -> Any:
    sec, nanosec = divmod(int(stamp_ns), 1_000_000_000)
    return SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nanosec))


def _action_input_stamp_ns(chunk: Any) -> int:
    timestamp = dict(getattr(chunk, "metadata", {}).get("timestamp", {}) or {})
    frame = timestamp.get("frame")
    if frame is not None:
        values = np.asarray(frame, dtype=np.int64).reshape(-1)
        if values.size:
            return int(values[-1])
    return time.time_ns()


def _read_pickle_mapping(path: Path) -> dict[int, Any]:
    if not path.exists():
        return {}
    import pickle

    with path.open("rb") as file:
        value = pickle.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a dict")
    return value


def _atomic_pickle_dump(value: Any, path: Path) -> None:
    import os
    import pickle

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as file:
        pickle.dump(value, file)
    os.replace(tmp_path, path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"type": f"{type(value).__module__}.{type(value).__name__}"}


def _array_summary(data: Any) -> Any:
    if isinstance(data, Mapping):
        return {str(key): _array_summary(value) for key, value in data.items()}
    arr = np.asarray(data)
    return {"shape": list(arr.shape), "dtype": str(arr.dtype)}
