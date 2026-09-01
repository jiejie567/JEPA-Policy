from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from prometheus.data.types import DataSample, DataStream, normalize_streams


EPISODE_DIRS = ("camera", "tactile", "robot", "event", "manifests")
CAMERA_IMAGE_SUFFIXES = ("color", "infra1", "infra2")
CAMERA_INFO_SUFFIXES = ("color_info", "infra1_info", "infra2_info")
CAMERA_RECORD_SUFFIXES = {
    "color": "rgb",
    "infra1": "ir_left",
    "infra2": "ir_right",
}
TACTILE_POINT_FIELDS = ("x", "y", "z", "dx", "dy", "dz")
TACTILE_FLOW_WIDTH = 35
TACTILE_FLOW_HEIGHT = 20
TACTILE_IMAGE_SUFFIXES = ("tactile_difference", "tactile_raw")
BUNDLE_TIMEOUT_S = 0.5


@dataclass
class PendingBundle:
    created_at: float
    values: dict[str, DataSample] = field(default_factory=dict)


class RawEpisodeWriter:
    def __init__(
        self,
        episode_dir: Path,
        streams: Mapping[str, Any],
        *,
        fps: float,
        enable_plots: bool = True,
        joint_plot_ranges: list[list[float]] | None = None,
        default_event_value: int | None = None,
    ):
        self.episode_dir = episode_dir
        self.streams = normalize_streams(streams)
        self.camera = CameraWriter(episode_dir, self.streams, fps=fps)
        self.tactile = TactileWriter(episode_dir, self.streams, fps=fps)
        self.robot = RobotWriter(episode_dir, enable_plots=enable_plots, joint_plot_ranges=joint_plot_ranges)
        self.action = ActionWriter(episode_dir)
        self.event = EventWriter(episode_dir, default_value=default_event_value)

    def write_sample(self, sample: DataSample) -> str:
        if sample.name == "robot_state":
            self.robot.write_state(sample)
            return "robot.robot_state"
        if sample.name == "left_eef":
            self.robot.write_eef("left", sample)
            return "robot.left_eef"
        if sample.name == "right_eef":
            self.robot.write_eef("right", sample)
            return "robot.right_eef"
        if parse_camera_image_name(sample.name) is not None:
            return f"camera.{self.camera.write_image(sample)}"
        if parse_camera_info_name(sample.name) is not None:
            return f"camera.{self.camera.write_info(sample)}"
        if parse_tactile_name(sample.name) is not None:
            return f"tactile.{self.tactile.write(sample)}"
        raise ValueError(f"raw episode writer does not know how to store data stream {sample.name!r}")

    def write_action_chunk(self, chunk: Any, *, stamp_ns: int, metadata: Mapping[str, Any]) -> str:
        self.action.write_chunk(chunk, stamp_ns=stamp_ns, metadata=metadata)
        return "action.policy_chunk"

    def write_action(
        self,
        action: Any,
        *,
        stamp_ns: int,
        action_space: str,
        hz: float,
        metadata: Mapping[str, Any],
    ) -> str:
        self.action.write_action(action, stamp_ns=stamp_ns, action_space=action_space, hz=hz, metadata=metadata)
        return "action.executed"

    def write_event(self, event: Mapping[str, Any], *, stamp_ns: int, metadata: Mapping[str, Any]) -> str:
        self.event.write(event, stamp_ns=stamp_ns, metadata=metadata)
        return "event.workflow"

    def finalize(self) -> None:
        errors = []
        for name, writer in (("camera", self.camera), ("tactile", self.tactile), ("robot", self.robot)):
            try:
                writer.finalize()
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
        self.event.ensure_default_from_robot(self.robot.robot_state)
        for name, writer in (("action", self.action), ("event", self.event)):
            try:
                writer.finalize()
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
        if errors:
            raise RuntimeError("; ".join(errors))

    def manifests(self, status: str, common: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
        manifests = {
            "camera": self.camera.manifest(status, common),
            "tactile": self.tactile.manifest(status, common),
            "robot": self.robot.manifest(status, common),
            "event": self.event.manifest(status, common),
        }
        if self.action.has_data:
            manifests["action"] = self.action.manifest(status, common)
        return manifests


class CameraWriter:
    def __init__(self, episode_dir: Path, topics: Mapping[str, DataStream], *, fps: float):
        self.episode_dir = episode_dir
        self.topics = dict(topics)
        self.fps = float(fps)
        self.expected: dict[str, set[str]] = defaultdict(set)
        self.stream_kind: dict[str, str] = {}
        self.record_name: dict[str, str] = {}
        self.info_kind: dict[str, tuple[str, str]] = {}
        self.info_record_name: dict[str, str] = {}
        for name in topics:
            image = parse_camera_image_name(name)
            if image is not None:
                camera_name, kind = image
                self.expected[camera_name].add(name)
                self.stream_kind[name] = kind
                self.record_name[name] = camera_record_image_name(camera_name, kind)
                continue
            info = parse_camera_info_name(name)
            if info is not None:
                camera_name, kind = info
                self.info_kind[name] = info
                self.info_record_name[name] = camera_record_info_name(camera_name, kind)

        self.pending: dict[tuple[str, int], PendingBundle] = {}
        self.timestamps: dict[str, list[int]] = {name: [] for name in self.expected}
        self.image_counts: dict[str, int] = {
            self.record_name[name]: 0
            for names in self.expected.values()
            for name in names
        }
        self.image_encodings: dict[str, str] = {}
        self.video_metadata: dict[str, dict[str, Any]] = {}
        self.camera_info: dict[str, Any] = {}
        self.writers: dict[str, Any] = {}
        self.dropped_incomplete = 0
        self.duplicate_messages = 0
        self._locks: dict[str, threading.RLock] = defaultdict(threading.RLock)

    def write_image(self, sample: DataSample) -> str:
        parsed = parse_camera_image_name(sample.name)
        if parsed is None:
            raise ValueError(f"{sample.name!r} is not a camera image stream")
        camera_name, _kind = parsed
        if camera_name not in self.expected or sample.name not in self.expected[camera_name]:
            raise ValueError(f"camera stream {sample.name!r} is not declared")

        with self._locks[camera_name]:
            self._expire_pending(now=time.monotonic(), sensor_name=camera_name)
            key = (camera_name, int(sample.stamp_ns))
            bundle = self.pending.setdefault(key, PendingBundle(created_at=time.monotonic()))
            if sample.name in bundle.values:
                self.duplicate_messages += 1
            bundle.values[sample.name] = sample
            if self.expected[camera_name].issubset(bundle.values):
                self.pending.pop(key)
                self._commit_bundle(camera_name, int(sample.stamp_ns), bundle.values)
        return self.record_name[sample.name]

    def write_info(self, sample: DataSample) -> str:
        parsed = parse_camera_info_name(sample.name)
        if parsed is None:
            raise ValueError(f"{sample.name!r} is not a camera info stream")
        camera_name, _kind = parsed
        with self._locks[camera_name]:
            self.camera_info[self.info_record_name[sample.name]] = message_to_dict(sample.msg)
        return self.info_record_name[sample.name]

    def finalize(self) -> None:
        for camera_name in sorted(self.expected):
            with self._locks[camera_name]:
                self._expire_pending(now=time.monotonic(), force=True, sensor_name=camera_name)
        errors = []
        for writer in list(self.writers.values()):
            try:
                writer.release()
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
        if self.timestamps:
            atomic_pickle_dump(self.timestamps, self.episode_dir / "camera" / "timestamp.pkl")
        if self.video_metadata or self.camera_info:
            atomic_json_dump(self.metadata(), self.episode_dir / "camera" / "metadata.json")
        if errors:
            raise RuntimeError("; ".join(errors))

    def manifest(self, status: str, common: Mapping[str, Any]) -> dict[str, Any]:
        entries: dict[str, Any] = {}
        for camera_name in sorted(self.expected):
            record_id = f"prometheus_camera_{safe_name(camera_name)}_recorder"
            payload = dict(common)
            payload.update(
                {
                    "record_id": record_id,
                    "record_mode": "sensor",
                    "sensor_kind": "camera",
                    "sensor_name": camera_name,
                    "camera_frames": {
                        self.record_name[name]: self.image_counts.get(self.record_name[name], 0)
                        for name in sorted(self.expected[camera_name])
                    },
                    "camera_timestamp_samples": len(self.timestamps.get(camera_name, [])),
                    "complete_sensor_bundles": len(self.timestamps.get(camera_name, [])),
                    "dropped_incomplete_sensor_bundles": self.dropped_incomplete,
                    "duplicate_sensor_messages": self.duplicate_messages,
                }
            )
            entries[record_id] = payload
        return entries

    def metadata(self) -> dict[str, Any]:
        payload = {}
        for name in sorted(self.stream_kind, key=self.record_name.__getitem__):
            camera_name, kind = require_camera_image_name(name)
            record_name = self.record_name[name]
            entry = {
                "topic": self.topics[name].topic,
                "encoding": self.image_encodings.get(record_name, ""),
                "frames": self.image_counts.get(record_name, 0),
                "timestamps": f"camera/timestamp.pkl::{camera_name}",
                "timestamp_policy": "shared_by_complete_physical_camera_bundle",
            }
            entry.update(self.video_metadata.get(record_name, {}))
            info_name = camera_record_info_name(camera_name, kind)
            if info_name in self.camera_info:
                entry["camera_info"] = self.camera_info[info_name]
            payload[record_name] = entry
        return payload

    def _commit_bundle(self, camera_name: str, stamp_ns: int, bundle: Mapping[str, DataSample]) -> None:
        for name in sorted(self.expected[camera_name]):
            sample = bundle[name]
            record_name = self.record_name[name]
            image, encoding = image_to_array(sample.msg)
            writer = self.writers.get(record_name)
            if writer is None:
                writer, metadata = open_camera_video_writer(
                    record_name,
                    self._video_path(record_name, image),
                    image,
                    encoding,
                    fps=self.fps,
                )
                self.writers[record_name] = writer
                self.video_metadata[record_name] = metadata
            writer.write(image)
            self.image_encodings[record_name] = encoding
            self.image_counts[record_name] = self.image_counts.get(record_name, 0) + 1
        self.timestamps.setdefault(camera_name, []).append(stamp_ns // 1_000_000)

    def _video_path(self, name: str, image: np.ndarray) -> Path:
        suffix = ".mkv" if image.ndim == 2 else ".mp4"
        return self.episode_dir / "camera" / f"{name}{suffix}"

    def _expire_pending(self, *, now: float, force: bool = False, sensor_name: str | None = None) -> None:
        expired = [
            key
            for key, bundle in self.pending.items()
            if (sensor_name is None or key[0] == sensor_name)
            and (force or now - bundle.created_at >= BUNDLE_TIMEOUT_S)
        ]
        for key in expired:
            self.pending.pop(key, None)
        self.dropped_incomplete += len(expired)


class TactileWriter:
    def __init__(self, episode_dir: Path, topics: Mapping[str, DataStream], *, fps: float):
        self.episode_dir = episode_dir
        self.topics = dict(topics)
        self.fps = float(fps)
        self.expected: dict[str, set[str]] = defaultdict(set)
        self.stream_kind: dict[str, str] = {}
        for name in topics:
            parsed = parse_tactile_name(name)
            if parsed is None:
                continue
            sensor_name, kind = parsed
            self.expected[sensor_name].add(name)
            self.stream_kind[name] = kind

        self.pending: dict[tuple[str, int], PendingBundle] = {}
        self.timestamps: dict[str, list[int]] = {name: [] for name in self.expected}
        self.pointcloud: dict[str, dict[str, Any]] = {
            name: {"data": []}
            for name, streams in self.expected.items()
            if f"{name}_tactile_flow" in streams
        }
        self.image_counts: dict[str, int] = {
            name: 0
            for name, kind in self.stream_kind.items()
            if kind in {"difference", "raw"}
        }
        self.image_encodings: dict[str, str] = {}
        self.video_metadata: dict[str, dict[str, Any]] = {}
        self.writers: dict[str, Any] = {}
        self.dropped_incomplete = 0
        self.duplicate_messages = 0
        self._locks: dict[str, threading.RLock] = defaultdict(threading.RLock)

    def write(self, sample: DataSample) -> str:
        parsed = parse_tactile_name(sample.name)
        if parsed is None:
            raise ValueError(f"{sample.name!r} is not a tactile stream")
        sensor_name, _kind = parsed
        if sensor_name not in self.expected or sample.name not in self.expected[sensor_name]:
            raise ValueError(f"tactile stream {sample.name!r} is not declared")

        with self._locks[sensor_name]:
            self._expire_pending(now=time.monotonic(), sensor_name=sensor_name)
            key = (sensor_name, int(sample.stamp_ns))
            bundle = self.pending.setdefault(key, PendingBundle(created_at=time.monotonic()))
            if sample.name in bundle.values:
                self.duplicate_messages += 1
            bundle.values[sample.name] = sample
            if self.expected[sensor_name].issubset(bundle.values):
                self.pending.pop(key)
                self._commit_bundle(sensor_name, int(sample.stamp_ns), bundle.values)
        return tactile_record_name(sample.name)

    def finalize(self) -> None:
        for sensor_name in sorted(self.expected):
            with self._locks[sensor_name]:
                self._expire_pending(now=time.monotonic(), force=True, sensor_name=sensor_name)
        for writer in list(self.writers.values()):
            writer.release()
        if self.timestamps:
            atomic_pickle_dump(self.timestamps, self.episode_dir / "tactile" / "timestamp.pkl")
        if self.pointcloud:
            atomic_pickle_dump(self.pointcloud, self.episode_dir / "tactile" / "tactile_pointcloud_dict.pkl")
        if self.expected:
            atomic_json_dump(self.metadata(), self.episode_dir / "tactile" / "metadata.json")

    def manifest(self, status: str, common: Mapping[str, Any]) -> dict[str, Any]:
        entries: dict[str, Any] = {}
        for sensor_name in sorted(self.expected):
            record_id = f"prometheus_tactile_{safe_name(sensor_name)}_recorder"
            payload = dict(common)
            payload.update(
                {
                    "record_id": record_id,
                    "record_mode": "sensor",
                    "sensor_kind": "tactile",
                    "sensor_name": sensor_name,
                    "tactile_samples": len(self.pointcloud.get(sensor_name, {}).get("data", [])),
                    "tactile_image_frames": {
                        tactile_record_name(name): self.image_counts.get(name, 0)
                        for name in sorted(self.expected[sensor_name])
                        if self.stream_kind.get(name) in {"difference", "raw"}
                    },
                    "tactile_timestamp_samples": len(self.timestamps.get(sensor_name, [])),
                    "complete_sensor_bundles": len(self.timestamps.get(sensor_name, [])),
                    "dropped_incomplete_sensor_bundles": self.dropped_incomplete,
                    "duplicate_sensor_messages": self.duplicate_messages,
                }
            )
            entries[record_id] = payload
        return entries

    def metadata(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for sensor_name in sorted(self.expected):
            images: dict[str, Any] = {}
            for name in sorted(self.expected[sensor_name]):
                kind = self.stream_kind[name]
                if kind not in {"difference", "raw"}:
                    continue
                record_name = tactile_record_name(name)
                entry = {
                    "topic": self.topics[name].topic,
                    "encoding": self.image_encodings.get(name, ""),
                    "frames": self.image_counts.get(name, 0),
                    "video": f"tactile/{record_name}.mkv",
                    "container": "matroska",
                    "codec": "ffv1",
                    "pixel_format": "bgr0",
                    "lossless": True,
                    "timestamps": f"tactile/timestamp.pkl::{sensor_name}",
                    "timestamp_policy": "shared_by_complete_physical_tactile_bundle",
                }
                entry.update(self.video_metadata.get(record_name, {}))
                images[kind] = entry
            sensor_payload: dict[str, Any] = {"images": images}
            if sensor_name in self.pointcloud:
                sensor_payload["pointcloud"] = {
                    "topic": self.topics[f"{sensor_name}_tactile_flow"].topic,
                    "samples": len(self.pointcloud[sensor_name]["data"]),
                    "width": TACTILE_FLOW_WIDTH,
                    "height": TACTILE_FLOW_HEIGHT,
                    "sample_shape": [
                        TACTILE_FLOW_HEIGHT * TACTILE_FLOW_WIDTH,
                        len(TACTILE_POINT_FIELDS),
                    ],
                    "storage": f"tactile/tactile_pointcloud_dict.pkl::{sensor_name}",
                    "timestamps": f"tactile/timestamp.pkl::{sensor_name}",
                    "timestamp_policy": "shared_by_complete_physical_tactile_bundle",
                }
            payload[sensor_name] = sensor_payload
        return payload

    def _commit_bundle(self, sensor_name: str, stamp_ns: int, bundle: Mapping[str, DataSample]) -> None:
        for name in sorted(self.expected[sensor_name]):
            sample = bundle[name]
            kind = self.stream_kind[name]
            if kind == "flow":
                self.pointcloud[sensor_name]["data"].append(pointcloud_to_array(sample.msg).copy())
                continue
            record_name = tactile_record_name(name)
            image, encoding = image_to_array(sample.msg)
            writer = self.writers.get(record_name)
            if writer is None:
                writer, metadata = open_tactile_video_writer(
                    self.episode_dir / "tactile" / f"{record_name}.mkv",
                    image,
                    encoding,
                    fps=self.fps,
                )
                self.writers[record_name] = writer
                self.video_metadata[record_name] = metadata
            writer.write(image)
            self.image_encodings[name] = encoding
            self.image_counts[name] = self.image_counts.get(name, 0) + 1
        self.timestamps.setdefault(sensor_name, []).append(stamp_ns // 1_000_000)

    def _expire_pending(self, *, now: float, force: bool = False, sensor_name: str | None = None) -> None:
        expired = [
            key
            for key, bundle in self.pending.items()
            if (sensor_name is None or key[0] == sensor_name)
            and (force or now - bundle.created_at >= BUNDLE_TIMEOUT_S)
        ]
        for key in expired:
            self.pending.pop(key, None)
        self.dropped_incomplete += len(expired)


class RobotWriter:
    def __init__(
        self,
        episode_dir: Path,
        *,
        enable_plots: bool = True,
        joint_plot_ranges: list[list[float]] | None = None,
    ):
        self.episode_dir = episode_dir
        self.robot_state = {
            "left": robot_side_series(),
            "right": robot_side_series(),
        }
        self.sample_indices: dict[int, int] = {}
        self.latest_eef: dict[str, np.ndarray | None] = {"left": None, "right": None}
        self.enable_plots = bool(enable_plots)
        self.joint_plot_ranges = joint_plot_ranges
        self.plot_outputs: list[str] = []
        self.plot_error = ""

    def write_state(self, sample: DataSample) -> None:
        joint = joint_field(getattr(sample.msg, "position", ()), "position")
        vel = joint_field(getattr(sample.msg, "velocity", ()), "velocity")
        effort = joint_field(getattr(sample.msg, "effort", ()), "effort")
        self._append_state(sample.stamp_ns, joint=joint, vel=vel, effort=effort)

    def write_eef(self, side: str, sample: DataSample) -> None:
        eef = pose_to_array(sample.msg)
        self.latest_eef[str(side)] = None if eef is None else eef.copy()
        committed_index = self.sample_indices.get(int(sample.stamp_ns))
        if committed_index is not None:
            self.robot_state[str(side)]["eef"][committed_index] = None if eef is None else eef.copy()

    def finalize(self) -> None:
        self._sort_by_timestamp()
        if self.robot_state["left"]["timestamps"] or self.robot_state["right"]["timestamps"]:
            atomic_pickle_dump(self.robot_state, self.episode_dir / "robot" / "robot_state_dict.pkl")
            self._write_episode_plots()

    def manifest(self, status: str, common: Mapping[str, Any]) -> dict[str, Any]:
        record_id = "prometheus_robot_recorder"
        payload = dict(common)
        payload.update(
            {
                "record_id": record_id,
                "record_mode": "robot",
                "robot_samples": {
                    side: len(stream["timestamps"])
                    for side, stream in self.robot_state.items()
                },
                "incomplete_robot_samples": self.incomplete_counts(),
                "pending_robot_samples": 0,
                "plot_outputs": list(self.plot_outputs),
            }
        )
        if self.plot_error:
            payload["plot_error"] = self.plot_error
        return {record_id: payload}

    def _write_episode_plots(self) -> None:
        if not self.enable_plots:
            return
        try:
            from prometheus.data.episode_plots import plot_episode_state_isolated

            self.plot_outputs = plot_episode_state_isolated(
                self.episode_dir,
                self.episode_dir / "robot" / "robot_state_dict.pkl",
                output_dir=self.episode_dir / "robot",
                joint_plot_ranges=self.joint_plot_ranges,
            )
        except Exception as exc:
            self.plot_error = str(exc)

    def incomplete_counts(self) -> dict[str, int]:
        result = {}
        for side, stream in self.robot_state.items():
            count = 0
            for index in range(len(stream["timestamps"])):
                if any(stream[field][index] is None for field in ("joint", "vel", "effort", "eef")):
                    count += 1
            result[side] = count
        return result

    def _append_state(
        self,
        stamp_ns: int,
        *,
        joint: np.ndarray | None,
        vel: np.ndarray | None,
        effort: np.ndarray | None,
    ) -> None:
        sample_index = len(self.robot_state["left"]["timestamps"])
        timestamp_ms = int(stamp_ns) // 1_000_000
        for side, start in (("left", 0), ("right", 7)):
            target = self.robot_state[side]
            for field_name, value in (("joint", joint), ("vel", vel), ("effort", effort)):
                target[field_name].append(None if value is None else value[start : start + 7].copy())
            eef = self.latest_eef.get(side)
            target["eef"].append(None if eef is None else eef.copy())
            target["timestamps"].append(timestamp_ms)
        self.sample_indices[int(stamp_ns)] = sample_index

    def _sort_by_timestamp(self) -> None:
        timestamps = self.robot_state["left"]["timestamps"]
        if len(timestamps) < 2:
            return
        order = sorted(range(len(timestamps)), key=timestamps.__getitem__)
        if order == list(range(len(timestamps))):
            return
        for side in ("left", "right"):
            stream = self.robot_state[side]
            for field_name in ("joint", "vel", "effort", "eef", "timestamps"):
                stream[field_name] = [stream[field_name][index] for index in order]


class ActionWriter:
    def __init__(self, episode_dir: Path):
        self.episode_dir = episode_dir
        self.chunks: dict[str, Any] = {"chunks": [], "timestamps": []}
        self.chunks_by_input_timestamp: dict[int, np.ndarray] = {}
        self.chunk_metadata_by_input_timestamp: dict[int, Any] = {}
        self.actions: dict[str, Any] = {
            "actions": [],
            "action_space": [],
            "hz": [],
            "timestamps": [],
            "metadata": [],
        }

    @property
    def has_data(self) -> bool:
        return bool(self.chunks["chunks"] or self.actions["actions"])

    def write_chunk(self, chunk: Any, *, stamp_ns: int, metadata: Mapping[str, Any]) -> None:
        actions = np.asarray(chunk.actions, dtype=np.float32)
        input_stamp_ns = action_input_stamp_ns(chunk, fallback_ns=stamp_ns)
        self._ensure_dir()
        self.chunks["chunks"].append(
            {
                "action_space": str(chunk.action_space),
                "hz": float(chunk.hz),
                "horizon": int(actions.shape[0]),
                "action_dim": int(actions.shape[1]),
                "actions": actions.copy(),
                "policy_metadata": json_safe(dict(getattr(chunk, "metadata", {}))),
                "metadata": json_safe(dict(metadata)),
                "input_stamp_ns": int(input_stamp_ns),
            }
        )
        self.chunks["timestamps"].append(int(input_stamp_ns) // 1_000_000)
        self.chunks_by_input_timestamp[int(input_stamp_ns)] = actions.copy()
        self.chunk_metadata_by_input_timestamp[int(input_stamp_ns)] = {
            "input_stamp_ns": int(input_stamp_ns),
            "action_space": str(chunk.action_space),
            "hz": float(chunk.hz),
            "horizon": int(actions.shape[0]),
            "action_dim": int(actions.shape[1]),
            "chunk_metadata": json_safe(dict(getattr(chunk, "metadata", {}))),
            "metadata": json_safe(dict(metadata)),
        }

    def write_action(
        self,
        action: Any,
        *,
        stamp_ns: int,
        action_space: str,
        hz: float,
        metadata: Mapping[str, Any],
    ) -> None:
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if values.size <= 0 or not np.all(np.isfinite(values)):
            raise ValueError("executed action must be finite and non-empty")
        if not str(action_space):
            raise ValueError("executed action_space must be non-empty")
        if float(hz) <= 0:
            raise ValueError("executed action hz must be positive")
        self._ensure_dir()
        self.actions["actions"].append(values.copy())
        self.actions["action_space"].append(str(action_space))
        self.actions["hz"].append(float(hz))
        self.actions["timestamps"].append(int(stamp_ns) // 1_000_000)
        self.actions["metadata"].append(json_safe(dict(metadata)))

    def finalize(self) -> None:
        if self.chunks["chunks"]:
            atomic_pickle_dump(self.chunks, self.episode_dir / "action" / "action_chunk_dict.pkl")
            atomic_pickle_dump(
                self.chunks_by_input_timestamp,
                self.episode_dir / "action" / "action_chunk_by_input_timestamp.pkl",
            )
            atomic_pickle_dump(
                self.chunk_metadata_by_input_timestamp,
                self.episode_dir / "action" / "action_chunk_metadata.pkl",
            )
        if self.actions["actions"]:
            atomic_pickle_dump(self.actions, self.episode_dir / "action" / "executed_action_dict.pkl")

    def manifest(self, status: str, common: Mapping[str, Any]) -> dict[str, Any]:
        record_id = "prometheus_action_recorder"
        payload = dict(common)
        payload.update(
            {
                "record_id": record_id,
                "record_mode": "action",
                "policy_action_chunks": len(self.chunks["chunks"]),
                "executed_actions": len(self.actions["actions"]),
            }
        )
        return {record_id: payload}

    def _ensure_dir(self) -> None:
        (self.episode_dir / "action").mkdir(exist_ok=True)


def carry_forward_event_values(
    robot_timestamps_ms: Sequence[int],
    explicit_events: Sequence[tuple[int, Any]],
    *,
    default_value: int,
) -> list[int]:
    robot_ts = sorted(int(ts) for ts in robot_timestamps_ms)
    toggles = sorted((int(ts), _event_value_int(value)) for ts, value in explicit_events)

    current = int(default_value)
    toggle_idx = 0
    values: list[int] = []
    for ts in robot_ts:
        while toggle_idx < len(toggles) and toggles[toggle_idx][0] <= ts:
            current = toggles[toggle_idx][1]
            toggle_idx += 1
        values.append(current)
    return values


def _event_value_int(value: Any) -> int:
    return int(value)


class EventWriter:
    def __init__(self, episode_dir: Path, *, default_value: int | None = 0):
        self.episode_dir = episode_dir
        self.default_value = None if default_value is None else int(default_value)
        self.data: dict[str, Any] = {"data": [], "timestamps": [], "names": [], "metadata": []}

    def write(self, event: Mapping[str, Any], *, stamp_ns: int, metadata: Mapping[str, Any]) -> None:
        self.data["names"].append(str(event["name"]))
        self.data["data"].append(event.get("value", True))
        self.data["timestamps"].append(int(stamp_ns) // 1_000_000)
        self.data["metadata"].append(json_safe(dict(metadata)))

    def ensure_default_from_robot(self, robot_state: Mapping[str, Any]) -> None:
        if self.default_value is None:
            return
        left = robot_state.get("left", {}) if isinstance(robot_state, Mapping) else {}
        right = robot_state.get("right", {}) if isinstance(robot_state, Mapping) else {}
        robot_ts = sorted(
            set(int(item) for item in left.get("timestamps", []))
            | set(int(item) for item in right.get("timestamps", []))
        )
        if not robot_ts:
            return
        explicit_events = list(zip(self.data["timestamps"], self.data["data"]))
        explicit_at = {int(ts) for ts, _value in explicit_events}
        self.data["timestamps"] = robot_ts
        self.data["data"] = carry_forward_event_values(
            robot_ts,
            explicit_events,
            default_value=self.default_value,
        )
        self.data["names"] = ["event_label" for _ in robot_ts]
        self.data["metadata"] = [
            {"source": "explicit_toggle" if timestamp in explicit_at else "carry_forward"}
            for timestamp in robot_ts
        ]

    def finalize(self) -> None:
        if self.data["data"]:
            atomic_pickle_dump(self.data, self.episode_dir / "event" / "event_dict.pkl")

    def manifest(self, status: str, common: Mapping[str, Any]) -> dict[str, Any]:
        record_id = "prometheus_event_recorder"
        payload = dict(common)
        payload.update(
            {
                "record_id": record_id,
                "record_mode": "event",
                "event_samples": len(self.data["data"]),
            }
        )
        return {record_id: payload}


def parse_camera_image_name(name: str) -> tuple[str, str] | None:
    for suffix in CAMERA_IMAGE_SUFFIXES:
        marker = f"_{suffix}"
        if name.endswith(marker):
            camera_name = name[: -len(marker)]
            if camera_name:
                return camera_name, suffix
    return None


def parse_camera_info_name(name: str) -> tuple[str, str] | None:
    for suffix in CAMERA_INFO_SUFFIXES:
        marker = f"_{suffix}"
        if name.endswith(marker):
            camera_name = name[: -len(marker)]
            if camera_name:
                return camera_name, suffix.removesuffix("_info")
    return None


def require_camera_image_name(name: str) -> tuple[str, str]:
    parsed = parse_camera_image_name(name)
    if parsed is None:
        raise ValueError(f"{name!r} is not a camera image stream name")
    return parsed


def parse_tactile_name(name: str) -> tuple[str, str] | None:
    for suffix in TACTILE_IMAGE_SUFFIXES:
        marker = f"_{suffix}"
        if name.endswith(marker):
            sensor_name = name[: -len(marker)]
            if sensor_name:
                return sensor_name, suffix.removeprefix("tactile_")
    marker = "_tactile_flow"
    if name.endswith(marker):
        sensor_name = name[: -len(marker)]
        if sensor_name:
            return sensor_name, "flow"
    return None


def tactile_record_name(name: str) -> str:
    parsed = parse_tactile_name(name)
    if parsed is None:
        raise ValueError(f"{name!r} is not a tactile stream name")
    sensor_name, kind = parsed
    return f"{sensor_name}_{kind}"


def camera_record_image_name(camera_name: str, kind: str) -> str:
    return f"{camera_name}_{CAMERA_RECORD_SUFFIXES[kind]}"


def camera_record_info_name(camera_name: str, kind: str) -> str:
    return f"{camera_record_image_name(camera_name, kind)}_info"


def joint_field(values: Any, field_name: str) -> np.ndarray | None:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0 and field_name in {"velocity", "effort"}:
        return None
    if array.size != 14:
        raise ValueError(f"Robot JointState {field_name} must have 14 values, got {array.size}")
    array = array.reshape(14).copy()
    if field_name in {"velocity", "effort"} and not np.isfinite(array).any():
        return None
    return array


def pose_to_array(msg: Any) -> np.ndarray | None:
    pose = getattr(msg, "pose")
    array = np.array(
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
    return None if not np.isfinite(array).any() else array


def robot_side_series() -> dict[str, list[Any]]:
    return {"joint": [], "vel": [], "effort": [], "eef": [], "timestamps": []}


def action_input_stamp_ns(chunk: Any, *, fallback_ns: int | None = None) -> int:
    timestamp = dict(getattr(chunk, "metadata", {}).get("timestamp", {}) or {})
    frame = timestamp.get("frame")
    if frame is not None:
        values = np.asarray(frame, dtype=np.int64).reshape(-1)
        if values.size:
            return int(values[-1])
    return time.time_ns() if fallback_ns is None else int(fallback_ns)


def image_to_array(msg: Any) -> tuple[np.ndarray, str]:
    if isinstance(msg, np.ndarray):
        image = np.asarray(msg, dtype=np.uint8)
        if image.ndim == 2:
            return np.ascontiguousarray(image), "mono8"
        if image.ndim == 3 and image.shape[2] == 3:
            return np.ascontiguousarray(image), "bgr8"
        raise ValueError(f"numpy image must be HxW or HxWx3 uint8, got {image.shape}")

    encoding = str(getattr(msg, "encoding", "")).lower()
    height = int(getattr(msg, "height"))
    width = int(getattr(msg, "width"))
    channels = {"mono8": 1, "8uc1": 1, "rgb8": 3, "bgr8": 3}.get(encoding)
    if channels is None:
        raise ValueError(f"unsupported ROS image encoding {encoding!r}")
    step = int(getattr(msg, "step", width * channels))
    data = getattr(msg, "data")
    raw = np.frombuffer(data if isinstance(data, (bytes, bytearray, memoryview)) else bytes(data), dtype=np.uint8)
    rows = raw.reshape(height, step)
    trimmed = rows[:, : width * channels]
    if channels == 1:
        return np.ascontiguousarray(trimmed.reshape(height, width)), "mono8"
    return np.ascontiguousarray(trimmed.reshape(height, width, channels)), encoding


def pointcloud_to_array(msg: Any) -> np.ndarray:
    if isinstance(msg, np.ndarray):
        array = np.asarray(msg, dtype=np.float32)
    else:
        fields = tuple(field.name for field in getattr(msg, "fields", ()))
        if fields != TACTILE_POINT_FIELDS:
            raise ValueError(f"Tactile PointCloud2 fields must be {TACTILE_POINT_FIELDS}, got {fields}")
        import sensor_msgs_py.point_cloud2 as pc2

        points = pc2.read_points(msg, field_names=TACTILE_POINT_FIELDS, skip_nans=False)
        if isinstance(points, np.ndarray) and points.dtype.fields:
            array = np.column_stack([points[field] for field in TACTILE_POINT_FIELDS]).astype(np.float32)
        else:
            array = np.asarray(points if isinstance(points, np.ndarray) else list(points), dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(0, len(TACTILE_POINT_FIELDS)) if array.size == 0 else array.reshape(-1, len(TACTILE_POINT_FIELDS))
    expected_shape = (TACTILE_FLOW_HEIGHT * TACTILE_FLOW_WIDTH, len(TACTILE_POINT_FIELDS))
    if array.shape != expected_shape:
        raise ValueError(f"Tactile PointCloud2 must contain {expected_shape} float32 values, got {array.shape}")
    return np.ascontiguousarray(array, dtype=np.float32)


def open_camera_video_writer(
    name: str,
    path: Path,
    image: np.ndarray,
    encoding: str,
    *,
    fps: float,
) -> tuple[Any, dict[str, Any]]:
    height, width = image.shape[:2]
    if image.ndim == 2:
        writer = FFmpegPipeWriter(
            path,
            width=width,
            height=height,
            fps=fps,
            input_pixel_format="gray",
            output_args=["-c:v", "ffv1", "-level", "3", "-coder", "1", "-context", "1", "-g", "1", "-slicecrc", "1", "-pix_fmt", "gray"],
            channels=1,
        )
        return writer, {
            "video": f"camera/{path.name}",
            "container": "matroska",
            "codec": "ffv1",
            "pixel_format": "gray",
            "width": width,
            "height": height,
            "lossless": True,
        }

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"camera image {name!r} must be mono8, rgb8, or bgr8; got shape {image.shape}")
    input_pixel_format = {"rgb8": "rgb24", "bgr8": "bgr24"}.get(encoding)
    if input_pixel_format is None:
        raise ValueError(f"camera image {name!r} unsupported encoding {encoding!r}")
    # NVIDIA's HEVC encoder rejects the checkpoint-native 128x128 policy
    # stream on this host. Keep NVENC for original-resolution camera video and
    # use software H.264 only for small policy images.
    use_nvenc = width >= 145 and height >= 145
    if use_nvenc:
        output_args = [
            "-c:v",
            "hevc_nvenc",
            "-preset",
            "p6",
            "-tune",
            "uhq",
            "-rc",
            "vbr",
            "-cq",
            "14",
            "-b:v",
            "0",
            "-spatial_aq",
            "1",
            "-temporal_aq",
            "1",
            "-rc-lookahead",
            "20",
            "-pix_fmt",
            "yuv444p",
            "-tag:v",
            "hvc1",
        ]
        video_metadata = {
            "container": "mp4",
            "codec": "hevc",
            "encoder": "hevc_nvenc",
            "pixel_format": "yuv444p",
            "lossless": False,
            "quality": {"mode": "cq", "value": 14},
        }
    else:
        output_args = [
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "14",
            "-pix_fmt",
            "yuv444p",
            "-tag:v",
            "avc1",
        ]
        video_metadata = {
            "container": "mp4",
            "codec": "h264",
            "encoder": "libx264",
            "pixel_format": "yuv444p",
            "lossless": False,
            "quality": {"mode": "crf", "value": 14},
        }
    writer = FFmpegPipeWriter(
        path,
        width=width,
        height=height,
        fps=fps,
        input_pixel_format=input_pixel_format,
        output_args=output_args,
        channels=3,
    )
    return writer, {
        "video": f"camera/{path.name}",
        "width": width,
        "height": height,
        **video_metadata,
    }


def open_tactile_video_writer(
    path: Path,
    image: np.ndarray,
    encoding: str,
    *,
    fps: float,
) -> tuple[Any, dict[str, Any]]:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"tactile image must be HxWx3 uint8, got shape {image.shape}")
    input_pixel_format = {"rgb8": "rgb24", "bgr8": "bgr24"}.get(str(encoding).lower())
    if input_pixel_format is None:
        raise ValueError(f"tactile image unsupported encoding {encoding!r}")
    height, width = image.shape[:2]
    writer = FFmpegPipeWriter(
        path,
        width=width,
        height=height,
        fps=fps,
        input_pixel_format=input_pixel_format,
        output_args=[
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-coder",
            "1",
            "-context",
            "1",
            "-g",
            "1",
            "-slicecrc",
            "1",
            "-pix_fmt",
            "bgr0",
        ],
        channels=3,
    )
    return writer, {
        "width": width,
        "height": height,
        "video": f"tactile/{path.name}",
        "container": "matroska",
        "codec": "ffv1",
        "pixel_format": "bgr0",
        "lossless": True,
    }


class FFmpegPipeWriter:
    def __init__(
        self,
        path: Path,
        *,
        width: int,
        height: int,
        input_pixel_format: str,
        output_args: list[str],
        channels: int,
        fps: float,
    ):
        ffmpeg = find_ffmpeg()
        if ffmpeg is None:
            raise RuntimeError("ffmpeg executable is required for video recording")
        self.path = path
        self.width = int(width)
        self.height = int(height)
        self.channels = int(channels)
        self.process = subprocess.Popen(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                input_pixel_format,
                "-video_size",
                f"{self.width}x{self.height}",
                "-framerate",
                str(float(fps)),
                "-i",
                "pipe:0",
                "-an",
                *output_args,
                str(path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            # Keep terminal Ctrl+C scoped to the rollout/camera workers. The
            # writers finalize when their stdin is closed during ordered
            # cleanup; receiving SIGINT themselves produced valid MP4 files
            # but an erroneous ffmpeg return code 255 in the run manifest.
            start_new_session=True,
        )
        if self.process.stdin is None:
            raise RuntimeError(f"failed to open ffmpeg stdin for {path}")
        self.closed = False

    def write(self, image: np.ndarray) -> None:
        if self.closed:
            raise RuntimeError(f"attempted to write closed video {self.path}")
        expected = (self.height, self.width) if self.channels == 1 else (self.height, self.width, self.channels)
        frame = np.asarray(image)
        if frame.dtype != np.uint8 or frame.shape != expected:
            raise ValueError(f"video frame for {self.path} must be shape={expected} uint8, got shape={frame.shape} dtype={frame.dtype}")
        try:
            self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError(self._failure_detail()) from exc

    def release(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.process.stdin is not None:
            self.process.stdin.close()
            self.process.stdin = None
        try:
            _stdout, stderr = self.process.communicate(timeout=30)
        except subprocess.TimeoutExpired as exc:
            self.process.kill()
            _stdout, stderr = self.process.communicate()
            raise RuntimeError(f"timed out finalizing video {self.path}: {stderr.decode(errors='replace').strip()}") from exc
        if self.process.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {self.path} with returncode {self.process.returncode}: {stderr.decode(errors='replace').strip()}")

    def _failure_detail(self) -> str:
        returncode = self.process.poll()
        stderr = b""
        if returncode is not None and self.process.stderr is not None:
            stderr = self.process.stderr.read()
        return f"ffmpeg writer failed for {self.path} with returncode={returncode}: {stderr.decode(errors='replace').strip()}"


def message_to_dict(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [message_to_dict(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): message_to_dict(item) for key, item in value.items()}
    if hasattr(value, "__slots__"):
        return {
            name: message_to_dict(getattr(value, name))
            for name in value.__slots__
            if not name.startswith("_")
        }
    if hasattr(value, "__dict__"):
        return {
            str(key): message_to_dict(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return {"type": f"{type(value).__module__}.{type(value).__name__}"}


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"type": "bytes", "size": len(value)}
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"type": f"{type(value).__module__}.{type(value).__name__}"}


def atomic_pickle_dump(value: Any, path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as file:
        pickle.dump(value, file)
    os.replace(tmp_path, path)


def atomic_json_dump(value: Mapping[str, Any], path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(json_safe(dict(value)), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in str(value)).strip("._") or "item"


def find_ffmpeg() -> str | None:
    environment_ffmpeg = Path(sys.executable).with_name("ffmpeg")
    if environment_ffmpeg.is_file():
        return str(environment_ffmpeg)
    return shutil.which("ffmpeg")
