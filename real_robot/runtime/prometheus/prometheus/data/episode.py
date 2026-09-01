from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from prometheus.data.raw_episode import find_ffmpeg
from prometheus.data.types import DataStream


class RawEpisodeReader:
    def __init__(self, episode_dir: str | Path):
        self.episode_dir = Path(episode_dir).expanduser()
        if not self.episode_dir.is_dir():
            raise FileNotFoundError(f"episode directory does not exist: {self.episode_dir}")
        self.manifests = _read_manifests(self.episode_dir / "manifests")
        self.camera_metadata = _read_json(self.episode_dir / "camera" / "metadata.json")
        self.camera_timestamps = _read_pickle(self.episode_dir / "camera" / "timestamp.pkl", default={})
        self.tactile_metadata = _read_json(self.episode_dir / "tactile" / "metadata.json")
        self.tactile_timestamps = _read_pickle(self.episode_dir / "tactile" / "timestamp.pkl", default={})
        self.tactile_pointcloud = _read_pickle(self.episode_dir / "tactile" / "tactile_pointcloud_dict.pkl", default={})
        self.robot = _read_pickle(self.episode_dir / "robot" / "robot_state_dict.pkl", default={})
        self.events = _read_pickle(self.episode_dir / "event" / "event_dict.pkl", default={})
        self.action_chunks = _read_pickle(self.episode_dir / "action" / "action_chunk_dict.pkl", default={})
        self.executed_actions = _read_pickle(self.episode_dir / "action" / "executed_action_dict.pkl", default={})
        self.camera_streams = _camera_streams(self.episode_dir, self.camera_metadata, self.camera_timestamps)
        self.tactile_image_streams = _tactile_image_streams(self.episode_dir, self.tactile_metadata, self.tactile_timestamps)
        self.visual_image_streams = {**self.camera_streams, **self.tactile_image_streams}
        self.streams = {
            name: DataStream(
                name=name,
                topic=str(spec["topic"]),
                msg_type="prometheus.data.VideoFrame",
                domain="offline",
            )
            for name, spec in self.visual_image_streams.items()
        }

    @property
    def name(self) -> str:
        return self.episode_dir.name

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": "offline",
            "state": "ready",
            "ready": True,
            "message": "episode loaded",
            "details": {
                "episode_dir": str(self.episode_dir),
                "counts": self.counts(),
                "manifests": self.manifests,
            },
            "updated_at_monotonic_s": time.monotonic(),
        }

    def counts(self) -> dict[str, int]:
        counts = {name: int(spec["frames"]) for name, spec in self.visual_image_streams.items()}
        for name, item in self.tactile_pointcloud.items():
            if isinstance(item, Mapping):
                counts[f"{name}_tactile_flow"] = len(item.get("data", []))
        counts["events"] = len(self.events.get("timestamps", []))
        counts["policy_action_chunks"] = len(self.action_chunks.get("timestamps", []))
        counts["executed_actions"] = len(self.executed_actions.get("timestamps", []))
        for side, stream in self.robot.items():
            if isinstance(stream, Mapping):
                counts[f"robot_{side}"] = len(stream.get("timestamps", []))
        return counts

    def timeline(self) -> dict[str, Any]:
        timestamps = sorted(
            set().union(
                *[set(spec["timestamps_ms"]) for spec in self.camera_streams.values()],
                *[set(spec["timestamps_ms"]) for spec in self.tactile_image_streams.values()],
                set(self.events.get("timestamps", [])),
                set(self.action_chunks.get("timestamps", [])),
                set(self.executed_actions.get("timestamps", [])),
                *[set(stream.get("timestamps", [])) for stream in self.robot.values() if isinstance(stream, Mapping)],
            )
        )
        return {
            "offline": True,
            "timestamps_ms": timestamps,
            "start_ms": timestamps[0] if timestamps else None,
            "end_ms": timestamps[-1] if timestamps else None,
            "streams": [
                {
                    "name": name,
                    "frames": spec["frames"],
                    "video": spec["video"],
                    "container": spec.get("container", ""),
                    "codec": spec.get("codec", ""),
                    "timestamps_ms": spec["timestamps_ms"],
                }
                for name, spec in self.visual_image_streams.items()
            ],
            "events": _event_items(self.events),
            "actions": {
                "policy_chunks": self.action_chunks.get("timestamps", []),
                "executed": self.executed_actions.get("timestamps", []),
            },
        }

    def visual_streams(self, *, t_ms: int | None = None) -> list[dict[str, Any]]:
        result = []
        for name, spec in self.visual_image_streams.items():
            index, stamp_ms = _nearest_index(spec["timestamps_ms"], t_ms)
            result.append(
                {
                    "name": name,
                    "topic": spec["topic"],
                    "msg_type": "prometheus.data.VideoFrame",
                    "count": spec["frames"],
                    "image": True,
                    "offline": True,
                    "index": index,
                    "stamp_ms": stamp_ms,
                    "stamp_ns": None if stamp_ms is None else int(stamp_ms) * 1_000_000,
                    "skew_ms": None if stamp_ms is None or t_ms is None else int(stamp_ms) - int(t_ms),
                    "video": spec["video"],
                    "container": spec.get("container", ""),
                    "codec": spec.get("codec", ""),
                }
            )
        return result

    def image_png(self, stream: str, *, t_ms: int | None = None, index: int | None = None) -> bytes:
        if stream not in self.visual_image_streams:
            raise KeyError(f"unknown episode image stream: {stream!r}")
        spec = self.visual_image_streams[stream]
        frame_index = int(index) if index is not None else _nearest_index(spec["timestamps_ms"], t_ms)[0]
        if frame_index < 0 or frame_index >= int(spec["frames"]):
            raise IndexError(f"stream {stream!r} frame index {frame_index} is out of range")
        ffmpeg = find_ffmpeg()
        if ffmpeg is None:
            raise RuntimeError("ffmpeg executable is required for offline video preview")
        command = [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(spec["path"]),
            "-vf",
            f"select=eq(n\\,{frame_index})",
            "-vframes",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "pipe:1",
        ]
        result = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0 or not result.stdout:
            detail = result.stderr.decode(errors="replace").strip()
            raise RuntimeError(f"failed to extract frame {frame_index} from {spec['video']}: {detail}")
        return result.stdout

    def visualize(self, **kwargs: Any) -> Any:
        from prometheus.data.visualizer import DataVisualizer

        return DataVisualizer(self, **kwargs)


def _camera_streams(episode_dir: Path, metadata: Mapping[str, Any], timestamps: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    streams = {}
    for name, spec in metadata.items():
        if not isinstance(spec, Mapping) or "video" not in spec:
            continue
        timestamp_ref = str(spec["timestamps"])
        if "::" not in timestamp_ref:
            raise ValueError(f"camera metadata {name!r} has invalid timestamps ref {timestamp_ref!r}")
        timestamp_key = timestamp_ref.rsplit("::", 1)[1]
        stream_timestamps = [int(value) for value in timestamps.get(timestamp_key, [])]
        streams[str(name)] = {
            "topic": str(spec.get("topic", "")),
            "video": str(spec["video"]),
            "path": episode_dir / str(spec["video"]),
            "frames": int(spec.get("frames", len(stream_timestamps))),
            "timestamps_ms": stream_timestamps,
            "container": str(spec.get("container", "")),
            "codec": str(spec.get("codec", "")),
            "pixel_format": str(spec.get("pixel_format", "")),
        }
    return streams


def _tactile_image_streams(episode_dir: Path, metadata: Mapping[str, Any], timestamps: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    streams = {}
    for sensor_name, sensor_spec in metadata.items():
        if not isinstance(sensor_spec, Mapping):
            continue
        images = sensor_spec.get("images", {})
        if not isinstance(images, Mapping):
            continue
        for kind, spec in images.items():
            if not isinstance(spec, Mapping) or "video" not in spec:
                continue
            timestamp_ref = str(spec["timestamps"])
            if "::" not in timestamp_ref:
                raise ValueError(f"tactile metadata {sensor_name!r}/{kind!r} has invalid timestamps ref {timestamp_ref!r}")
            timestamp_key = timestamp_ref.rsplit("::", 1)[1]
            stream_timestamps = [int(value) for value in timestamps.get(timestamp_key, [])]
            name = f"{sensor_name}_tactile_{kind}"
            streams[name] = {
                "topic": str(spec.get("topic", "")),
                "video": str(spec["video"]),
                "path": episode_dir / str(spec["video"]),
                "frames": int(spec.get("frames", len(stream_timestamps))),
                "timestamps_ms": stream_timestamps,
                "container": str(spec.get("container", "")),
                "codec": str(spec.get("codec", "")),
                "pixel_format": str(spec.get("pixel_format", "")),
            }
    return streams


def _nearest_index(values: list[int], target: int | None) -> tuple[int, int | None]:
    if not values:
        return 0, None
    if target is None:
        return len(values) - 1, values[-1]
    index = min(range(len(values)), key=lambda item: abs(values[item] - int(target)))
    return index, values[index]


def _event_items(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    timestamps = data.get("timestamps", [])
    names = data.get("names", [])
    values = data.get("data", [])
    result = []
    for index, stamp_ms in enumerate(timestamps):
        result.append(
            {
                "timestamp_ms": int(stamp_ms),
                "name": str(names[index]) if index < len(names) else "event",
                "value": values[index] if index < len(values) else True,
            }
        )
    return result


def _read_manifests(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        return {}
    return {
        item.stem: _read_json(item)
        for item in sorted(path.glob("*.json"))
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_pickle(path: Path, *, default: Any) -> Any:
    if not path.is_file():
        return default
    with path.open("rb") as file:
        return pickle.load(file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Open a Prometheus raw episode in the DataSession visualizer.")
    parser.add_argument("episode_dir")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    reader = RawEpisodeReader(args.episode_dir)
    viewer = reader.visualize(host=args.host, port=args.port)
    print(viewer.url, flush=True)
    viewer.start(block=True)


if __name__ == "__main__":
    main()
