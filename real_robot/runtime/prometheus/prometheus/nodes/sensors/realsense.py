from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import signal
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


CAMERA_TOPIC_PREFIX = "/prometheus/sensors/cameras"
LOG_HISTORY = 80


@dataclass(frozen=True)
class CameraResize:
    scale: float | None = None
    width: int | None = None
    height: int | None = None

    @property
    def enabled(self) -> bool:
        return self.width is not None or self.height is not None or self.scale not in (None, 1.0)


@dataclass(frozen=True)
class OriginalColorRecording:
    enabled: bool = False
    output_dir: str = ""


class RealsenseSensorNode:
    """HardwareSession-owned pyrealsense2 camera publisher.

    Each physical camera runs in its own process with its own pyrealsense2
    pipeline and ROS node. The process publishes the same topic contract that
    the RealSense ROS wrapper used to expose.
    """

    def __init__(
        self,
        *,
        name: str = "realsense",
        cameras: Mapping[str, str],
        camera_namespace: str = CAMERA_TOPIC_PREFIX,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        publish_hz: float | None = None,
        enable_infra: bool = True,
        ir: bool | None = None,
        resize: Any = 1.0,
        qos: str = "sensor",
        camera_info: bool = True,
        original_color_recording: Mapping[str, Any] | None = None,
        ros_domain_id: int | None = None,
    ):
        self.name = str(name)
        self.cameras = normalize_camera_map(cameras)
        if not self.cameras:
            raise ValueError("RealsenseSensorNode requires at least one camera")
        self.camera_namespace = str(camera_namespace).rstrip("/")
        self.width = _positive_int(width, "width")
        self.height = _positive_int(height, "height")
        self.fps = _positive_int(fps, "fps")
        self.publish_hz = _positive_float(self.fps if publish_hz is None else publish_hz, "publish_hz")
        self.enable_infra = bool(enable_infra if ir is None else ir)
        self.resize = normalize_resize(resize)
        self.qos = normalize_qos(qos)
        self.camera_info = bool(camera_info)
        self.original_color_recording = normalize_original_color_recording(
            original_color_recording
        )
        self.ros_domain_id = None if ros_domain_id is None else int(ros_domain_id)
        self._ctx = mp.get_context("spawn")
        self._ready_queue: Any | None = None
        self._event_queue: Any | None = None
        self._procs: dict[str, mp.Process] = {}
        self._events: dict[str, list[str]] = {}
        self._ready_names: set[str] = set()
        self._started = False
        self._closed = False

    def start(self) -> None:
        if self._started:
            return
        self._ready_queue = self._ctx.Queue()
        self._event_queue = self._ctx.Queue()
        self._ready_names.clear()
        self._events = {camera_name: [] for camera_name in self.cameras}

        for camera_name, serial in self.cameras.items():
            topics = camera_topics(
                camera_name,
                camera_namespace=self.camera_namespace,
                enable_infra=self.enable_infra,
                camera_info=self.camera_info,
            )
            proc = self._ctx.Process(
                target=_camera_worker,
                name=f"{self.name}_{camera_name}",
                kwargs={
                    "camera_name": camera_name,
                    "serial": serial,
                    "topics": topics,
                    "width": self.width,
                    "height": self.height,
                    "fps": self.fps,
                    "publish_hz": self.publish_hz,
                    "enable_infra": self.enable_infra,
                    "resize": self.resize,
                    "qos": self.qos,
                    "camera_info": self.camera_info,
                    "original_color_recording": self.original_color_recording,
                    "ros_domain_id": self.ros_domain_id,
                    "ready_queue": self._ready_queue,
                    "event_queue": self._event_queue,
                },
            )
            proc.start()
            self._procs[camera_name] = proc
            self._events[camera_name].append(f"started pid={proc.pid}")

        self._started = True
        self._closed = False

    def wait_ready(self, timeout_s: float) -> None:
        if not self._started:
            raise RuntimeError("RealSense sensor has not been started")
        deadline = time.monotonic() + float(timeout_s)
        expected = set(self.cameras)
        while time.monotonic() < deadline:
            self._drain_queues()
            self._raise_on_launch_exit()
            if set(self._ready_names) == expected:
                return
            time.sleep(0.05)
        raise TimeoutError(
            "timed out waiting for RealSense cameras: "
            f"ready={sorted(self._ready_names)} expected={sorted(expected)} events={self._recent_logs()}"
        )

    def stop(self) -> None:
        for proc in self._procs.values():
            if proc.is_alive():
                proc.terminate()
        for proc in self._procs.values():
            proc.join(timeout=10.0)
        for proc in self._procs.values():
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=1.0)
        self._procs.clear()
        self._ready_names.clear()
        self._started = False

    def close(self) -> None:
        self.stop()
        self._closed = True

    def status(self) -> dict[str, Any]:
        self._drain_queues()
        ready_names = sorted(self._ready_names)
        return {
            "type": type(self).__name__,
            "name": self.name,
            "started": self._started,
            "closed": self._closed,
            "ready": set(ready_names) == set(self.cameras),
            "ready_names": ready_names,
            "cameras": dict(self.cameras),
            "camera_namespace": self.camera_namespace,
            "ros_domain_id": self.ros_domain_id,
            "fps": self.fps,
            "publish_hz": self.publish_hz,
            "enable_infra": self.enable_infra,
            "resize": resize_status(self.resize),
            "qos": self.qos,
            "camera_info": self.camera_info,
            "original_color_recording": {
                "enabled": self.original_color_recording.enabled,
                "output_dir": self.original_color_recording.output_dir,
                "capture_width": self.width,
                "capture_height": self.height,
            },
            "processes": {
                name: {"pid": proc.pid, "exitcode": proc.exitcode}
                for name, proc in self._procs.items()
            },
        }

    def _drain_queues(self) -> None:
        if self._ready_queue is not None:
            while True:
                try:
                    self._ready_names.add(str(self._ready_queue.get_nowait()))
                except queue.Empty:
                    break
        if self._event_queue is not None:
            while True:
                try:
                    camera_name, message = self._event_queue.get_nowait()
                except queue.Empty:
                    break
                events = self._events.setdefault(str(camera_name), [])
                events.append(str(message))
                del events[:-LOG_HISTORY]

    def _raise_on_launch_exit(self) -> None:
        dead = {
            name: proc.exitcode
            for name, proc in self._procs.items()
            if proc.exitcode is not None and name not in self._ready_names
        }
        if dead:
            raise RuntimeError(f"RealSense process exited before ready: {dead}; events={self._recent_logs()}")

    def _recent_logs(self) -> dict[str, list[str]]:
        self._drain_queues()
        return {name: list(lines)[-10:] for name, lines in self._events.items()}


def _camera_worker(
    *,
    camera_name: str,
    serial: str,
    topics: dict[str, str],
    width: int,
    height: int,
    fps: int,
    publish_hz: float,
    enable_infra: bool,
    resize: CameraResize,
    qos: str,
    camera_info: bool,
    original_color_recording: OriginalColorRecording,
    ros_domain_id: int | None,
    ready_queue: Any,
    event_queue: Any,
) -> None:
    import numpy as np
    import pyrealsense2 as rs
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.signals import SignalHandlerOptions
    from sensor_msgs.msg import CameraInfo, Image

    shutdown_requested = False

    def _request_shutdown(_signum: int, _frame: Any) -> None:
        nonlocal shutdown_requested
        shutdown_requested = True

    signal.signal(signal.SIGINT, _request_shutdown)
    signal.signal(signal.SIGTERM, _request_shutdown)

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(str(serial))
    cfg.enable_stream(rs.stream.color, int(width), int(height), rs.format.bgr8, int(fps))
    if enable_infra:
        cfg.enable_stream(rs.stream.infrared, 1, int(width), int(height), rs.format.y8, int(fps))
        cfg.enable_stream(rs.stream.infrared, 2, int(width), int(height), rs.format.y8, int(fps))

    rclpy.init(args=None, domain_id=ros_domain_id, signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node(f"prometheus_realsense_{safe_node_name(camera_name)}")
    bridge = CvBridge()
    timing_trace = _PublishTimingTrace(os.environ.get("PROMETHEUS_CAMERA_TIMING_TRACE", ""), camera_name)
    original_recorder = _OriginalColorRecorder(
        camera_name=camera_name,
        config=original_color_recording,
        fps=float(fps),
    )
    image_qos = qos_profile_sensor_data if qos == "sensor" else 10
    publishers = {
        name: node.create_publisher(Image, topic, image_qos)
        for name, topic in topics.items()
        if not name.endswith("_info")
    }
    info_publishers = {
        name: node.create_publisher(CameraInfo, topic, image_qos)
        for name, topic in topics.items()
        if camera_info and name.endswith("_info")
    }

    try:
        profile = pipeline.start(cfg)
        event_queue.put((camera_name, f"pipeline started serial={serial} rgb={width}x{height}@{fps} publish_hz={publish_hz}"))
        info_templates = (
            _camera_info_templates(profile, rs, camera_name=camera_name, enable_infra=enable_infra, resize=resize)
            if camera_info
            else {}
        )
        period = 1.0 / max(1.0, float(publish_hz))
        next_time = time.monotonic()
        ready_reported = False

        while rclpy.ok() and not shutdown_requested:
            frames = pipeline.wait_for_frames()
            stamp = node.get_clock().now().to_msg()
            published_streams: set[str] = set()

            color = frames.get_color_frame()
            if color:
                color_image = np.asanyarray(color.get_data())
                original_recorder.write(
                    color_image,
                    ros_stamp_ns=_stamp_msg_to_ns(stamp),
                    device_frame_number=int(color.get_frame_number()),
                    device_timestamp_ms=float(color.get_timestamp()),
                )
                _publish_image(
                    bridge,
                    publishers["color"],
                    color_image,
                    "bgr8",
                    stamp,
                    f"{camera_name}_color_frame",
                    resize,
                    timing_trace,
                    "color",
                )
                if camera_info:
                    _publish_info(info_publishers.get("color_info"), info_templates.get("color_info"), stamp)
                published_streams.add("color")

            if enable_infra:
                infra1 = frames.get_infrared_frame(1)
                if infra1:
                    _publish_image(
                        bridge,
                        publishers["infra1"],
                        np.asanyarray(infra1.get_data()),
                        "mono8",
                        stamp,
                        f"{camera_name}_infra1_frame",
                        resize,
                        timing_trace,
                        "infra1",
                    )
                    if camera_info:
                        _publish_info(info_publishers.get("infra1_info"), info_templates.get("infra1_info"), stamp)
                    published_streams.add("infra1")
                infra2 = frames.get_infrared_frame(2)
                if infra2:
                    _publish_image(
                        bridge,
                        publishers["infra2"],
                        np.asanyarray(infra2.get_data()),
                        "mono8",
                        stamp,
                        f"{camera_name}_infra2_frame",
                        resize,
                        timing_trace,
                        "infra2",
                    )
                    if camera_info:
                        _publish_info(info_publishers.get("infra2_info"), info_templates.get("infra2_info"), stamp)
                    published_streams.add("infra2")

            expected_streams = {"color", *(("infra1", "infra2") if enable_infra else ())}
            if not ready_reported and expected_streams.issubset(published_streams):
                ready_queue.put(camera_name)
                ready_reported = True

            rclpy.spin_once(node, timeout_sec=0.0)
            next_time += period
            time.sleep(max(0.0, next_time - time.monotonic()))
    except Exception as exc:
        event_queue.put((camera_name, f"error: {type(exc).__name__}: {exc}"))
        raise
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        timing_trace.close()
        recording_failure: Exception | None = None
        try:
            original_recorder.close()
        except Exception as exc:
            recording_failure = exc
            event_queue.put(
                (
                    camera_name,
                    "original recording finalize error: "
                    f"{type(exc).__name__}: {exc}",
                )
            )
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()
        if recording_failure is not None:
            raise recording_failure


def _publish_image(
    bridge: Any,
    publisher: Any,
    image: Any,
    encoding: str,
    stamp: Any,
    frame_id: str,
    resize: CameraResize,
    timing_trace: Any | None = None,
    stream_name: str = "",
) -> None:
    image = _resize_image(image, resize)
    msg = bridge.cv2_to_imgmsg(image, encoding=encoding)
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    publish_start_ns = time.time_ns()
    publisher.publish(msg)
    publish_done_ns = time.time_ns()
    if timing_trace is not None:
        timing_trace.record(
            stream_name,
            publish_ros_ns=_stamp_msg_to_ns(stamp),
            publish_start_ns=publish_start_ns,
            publish_done_ns=publish_done_ns,
            encoding=encoding,
            height=int(getattr(msg, "height", 0)),
            width=int(getattr(msg, "width", 0)),
            bytes_len=len(getattr(msg, "data", b"")),
        )


def _publish_info(publisher: Any | None, template: Any | None, stamp: Any) -> None:
    if publisher is None or template is None:
        return
    msg = template
    msg.header.stamp = stamp
    publisher.publish(msg)


def _resize_image(image: Any, resize: CameraResize) -> Any:
    if not resize.enabled:
        return image
    import cv2

    target_width, target_height = _target_size(image.shape[1], image.shape[0], resize)
    if (image.shape[1], image.shape[0]) == (target_width, target_height):
        return image
    interpolation = cv2.INTER_AREA if target_width <= image.shape[1] and target_height <= image.shape[0] else cv2.INTER_LINEAR
    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)


def _camera_info_templates(
    profile: Any,
    rs: Any,
    *,
    camera_name: str,
    enable_infra: bool,
    resize: CameraResize,
) -> dict[str, Any]:
    templates = {
        "color_info": _camera_info_from_profile(
            profile.get_stream(rs.stream.color).as_video_stream_profile(),
            f"{camera_name}_color_frame",
            resize,
        ),
    }
    if enable_infra:
        templates["infra1_info"] = _camera_info_from_profile(
            profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile(),
            f"{camera_name}_infra1_frame",
            resize,
        )
        templates["infra2_info"] = _camera_info_from_profile(
            profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile(),
            f"{camera_name}_infra2_frame",
            resize,
        )
    return templates


def _camera_info_from_profile(video_profile: Any, frame_id: str, resize: CameraResize) -> Any:
    from sensor_msgs.msg import CameraInfo

    intr = video_profile.get_intrinsics()
    width, height = _target_size(int(intr.width), int(intr.height), resize)
    sx = width / float(intr.width)
    sy = height / float(intr.height)

    msg = CameraInfo()
    msg.header.frame_id = frame_id
    msg.width = width
    msg.height = height
    fx = float(intr.fx) * sx
    fy = float(intr.fy) * sy
    ppx = float(intr.ppx) * sx
    ppy = float(intr.ppy) * sy
    msg.k = [fx, 0.0, ppx, 0.0, fy, ppy, 0.0, 0.0, 1.0]
    msg.p = [fx, 0.0, ppx, 0.0, 0.0, fy, ppy, 0.0, 0.0, 0.0, 1.0, 0.0]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.d = [float(value) for value in intr.coeffs]
    msg.distortion_model = str(intr.model).split(".")[-1]
    return msg


def _target_size(width: int, height: int, resize: CameraResize) -> tuple[int, int]:
    if resize.width is not None and resize.height is not None:
        return int(resize.width), int(resize.height)
    if resize.scale is not None:
        return max(1, int(round(width * resize.scale))), max(1, int(round(height * resize.scale)))
    return int(width), int(height)


def camera_topics(
    camera_name: str,
    *,
    camera_namespace: str = CAMERA_TOPIC_PREFIX,
    enable_infra: bool = True,
    camera_info: bool = True,
) -> dict[str, str]:
    base = f"{camera_namespace.rstrip('/')}/{camera_name}"
    topics = {
        "color": f"{base}/color/image_raw",
    }
    if camera_info:
        topics["color_info"] = f"{base}/color/camera_info"
    if enable_infra:
        topics["infra1"] = f"{base}/infra1/image_rect_raw"
        topics["infra2"] = f"{base}/infra2/image_rect_raw"
        if camera_info:
            topics["infra1_info"] = f"{base}/infra1/camera_info"
            topics["infra2_info"] = f"{base}/infra2/camera_info"
    return topics


def parse_camera_map(value: str) -> dict[str, str]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("camera map must be a JSON object: name -> RealSense serial")
    return normalize_camera_map(parsed)


def normalize_camera_map(value: Mapping[str, Any]) -> dict[str, str]:
    cameras = {}
    serials: dict[str, str] = {}
    for key, serial in value.items():
        name = str(key).strip()
        if not name:
            raise ValueError("RealSense camera name must be non-empty")
        if name != safe_node_name(name):
            raise ValueError(f"RealSense camera name {name!r} must be a ROS-safe name")
        if isinstance(serial, Mapping):
            serial = serial.get("serial_no", "")
        serial = str(serial).strip()
        if not serial:
            raise ValueError(f"RealSense camera {name!r} serial must be non-empty")
        if serial in serials:
            raise ValueError(f"RealSense serial {serial!r} is used by {serials[serial]!r} and {name!r}")
        cameras[name] = serial
        serials[serial] = name
    return cameras


def normalize_resize(value: Any) -> CameraResize:
    if value is None:
        return CameraResize(scale=1.0)
    if isinstance(value, Mapping):
        if "scale" in value:
            return CameraResize(scale=_positive_float(value["scale"], "resize.scale"))
        if "width" in value and "height" in value:
            return CameraResize(width=_positive_int(value["width"], "resize.width"), height=_positive_int(value["height"], "resize.height"))
        raise ValueError("resize mapping must contain scale or width/height")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ValueError("resize sequence must be [width, height]")
        return CameraResize(width=_positive_int(value[0], "resize.width"), height=_positive_int(value[1], "resize.height"))
    return CameraResize(scale=_positive_float(value, "resize"))


def resize_status(resize: CameraResize) -> dict[str, Any]:
    return {
        "scale": resize.scale,
        "width": resize.width,
        "height": resize.height,
    }


def safe_node_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(name))


def camera_stream_names(camera_name: str, *, enable_infra: bool = True, camera_info: bool = True) -> dict[str, str]:
    base = str(camera_name)
    names = {
        "color": f"{base}_color",
    }
    if camera_info:
        names["color_info"] = f"{base}_color_info"
    if enable_infra:
        names["infra1"] = f"{base}_infra1"
        names["infra2"] = f"{base}_infra2"
        if camera_info:
            names["infra1_info"] = f"{base}_infra1_info"
            names["infra2_info"] = f"{base}_infra2_info"
    return names


def camera_streams(
    camera_name: str,
    ros_domain_id: int,
    *,
    camera_namespace: str = CAMERA_TOPIC_PREFIX,
    enable_infra: bool = True,
    camera_info: bool = True,
) -> dict[str, dict[str, Any]]:
    names = camera_stream_names(camera_name, enable_infra=enable_infra, camera_info=camera_info)
    topics = camera_topics(camera_name, camera_namespace=camera_namespace, enable_infra=enable_infra, camera_info=camera_info)
    return {names[key]: _topic(topic, _msg_type_for_stream(key), ros_domain_id) for key, topic in topics.items()}


def runtime_config(robot: Mapping[str, Any]) -> dict[str, Any]:
    camera_config = dict(robot.get("cameras", {}))
    cameras = camera_serials(camera_config)
    if not cameras:
        return {}
    required = ("capture_hz", "capture_width", "capture_height", "publish_hz", "ir", "prepub_resize")
    missing = [name for name in required if name not in camera_config]
    if missing:
        raise ValueError(f"robot.cameras is missing required keys: {missing}")
    enable_infra = bool(camera_config["ir"])
    resize = camera_config["prepub_resize"]
    fps = int(camera_config["capture_hz"])
    publish_hz = float(camera_config["publish_hz"])
    qos = normalize_qos(camera_config.get("qos", "sensor"))
    camera_info = bool(camera_config.get("camera_info", True))
    original_color_recording = camera_config.get("original_color_recording")
    return {
        "realsense": {
            "_target_": "prometheus.nodes.sensors.realsense.RealsenseSensorNode",
            "name": "realsense",
            "cameras": cameras,
            "camera_namespace": CAMERA_TOPIC_PREFIX,
            "width": int(camera_config["capture_width"]),
            "height": int(camera_config["capture_height"]),
            "fps": fps,
            "publish_hz": publish_hz,
            "enable_infra": enable_infra,
            "resize": resize,
            "qos": qos,
            "camera_info": camera_info,
            "original_color_recording": original_color_recording,
            "ros_domain_id": int(robot["sensor_domain_id"]),
        }
    }


def streams(robot: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    camera_config = dict(robot.get("cameras", {}))
    cameras = camera_serials(camera_config)
    if "ir" not in camera_config:
        raise ValueError("robot.cameras is missing required key: ir")
    enable_infra = bool(camera_config["ir"])
    camera_info = bool(camera_config.get("camera_info", True))
    items: dict[str, dict[str, Any]] = {}
    for camera_name in cameras:
        items.update(
            camera_streams(
                camera_name,
                int(robot["sensor_domain_id"]),
                enable_infra=enable_infra,
                camera_info=camera_info,
            )
        )
    return items


def camera_serials(camera_config: Mapping[str, Any]) -> dict[str, str]:
    serial_no = camera_config.get("serial_no")
    if not isinstance(serial_no, Mapping):
        raise ValueError("robot.cameras.serial_no must be a mapping of camera name to RealSense serial")
    return normalize_camera_map(serial_no)


def _msg_type_for_stream(key: str) -> str:
    if key.endswith("_info") or key in {"color_info", "infra1_info", "infra2_info"}:
        return "sensor_msgs.msg.CameraInfo"
    return "sensor_msgs.msg.Image"


def _topic(topic: str, msg_type: str, ros_domain_id: int) -> dict[str, Any]:
    return {
        "topic": topic,
        "msg_type": msg_type,
        "domain": "sensor",
        "ros_domain_id": int(ros_domain_id),
    }


def normalize_qos(value: Any) -> str:
    qos = str(value).strip().lower()
    if qos in {"sensor", "sensor_data", "best_effort", "besteffort"}:
        return "sensor"
    if qos in {"reliable", "default"}:
        return "reliable"
    raise ValueError("camera qos must be 'sensor' or 'reliable'")


def normalize_original_color_recording(
    value: Mapping[str, Any] | None,
) -> OriginalColorRecording:
    config = dict(value or {})
    enabled = bool(config.get("enabled", False))
    output_dir = str(config.get("output_dir", "")).strip()
    if enabled and not output_dir:
        raise ValueError(
            "original_color_recording.output_dir is required when enabled"
        )
    return OriginalColorRecording(enabled=enabled, output_dir=output_dir)


class _OriginalColorRecorder:
    """Record the capture frame before the ROS/policy resize in each worker."""

    def __init__(
        self,
        *,
        camera_name: str,
        config: OriginalColorRecording,
        fps: float,
    ) -> None:
        self.camera_name = str(camera_name)
        self.config = config
        self.fps = float(fps)
        self.frame_count = 0
        self._writer: Any | None = None
        self._timestamps: Any | None = None
        self._video_metadata: dict[str, Any] = {}
        self._output_dir: Path | None = None
        if not config.enabled:
            return
        self._output_dir = Path(config.output_dir).expanduser()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        timestamps_path = self._output_dir / f"{self.camera_name}_timestamps.jsonl"
        self._timestamps = timestamps_path.open("w", encoding="utf-8")

    def write(
        self,
        image: Any,
        *,
        ros_stamp_ns: int,
        device_frame_number: int,
        device_timestamp_ms: float,
    ) -> None:
        if not self.config.enabled:
            return
        if self._output_dir is None or self._timestamps is None:
            raise RuntimeError("original camera recorder was not initialized")
        if self._writer is None:
            from prometheus.data.raw_episode import open_camera_video_writer

            video_path = self._output_dir / f"{self.camera_name}_original_rgb.mp4"
            self._writer, self._video_metadata = open_camera_video_writer(
                f"{self.camera_name}_original_rgb",
                video_path,
                image,
                "bgr8",
                fps=self.fps,
            )
            self._video_metadata["video"] = video_path.name
            self._video_metadata["source"] = "realsense_capture_before_prepub_resize"
            self._video_metadata["timestamps"] = (
                f"{self.camera_name}_timestamps.jsonl"
            )
        self._writer.write(image)
        payload = {
            "frame_index": self.frame_count,
            "ros_stamp_ns": int(ros_stamp_ns),
            "host_write_ns": time.time_ns(),
            "device_frame_number": int(device_frame_number),
            "device_timestamp_ms": float(device_timestamp_ms),
        }
        self._timestamps.write(
            json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"
        )
        self.frame_count += 1
        if self.frame_count % max(1, int(round(self.fps))) == 0:
            self._timestamps.flush()

    def close(self) -> None:
        if not self.config.enabled:
            return
        failure: Exception | None = None
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception as exc:
                failure = exc
        if self._timestamps is not None:
            self._timestamps.close()
            self._timestamps = None
        if self._output_dir is not None:
            metadata = {
                "camera": self.camera_name,
                "frame_count": self.frame_count,
                "fps": self.fps,
                **self._video_metadata,
            }
            metadata_path = (
                self._output_dir / f"{self.camera_name}_original_metadata.json"
            )
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        self._writer = None
        if failure is not None:
            raise failure


class _PublishTimingTrace:
    def __init__(self, path_value: str, camera_name: str):
        self.enabled = bool(str(path_value).strip())
        self.camera_name = str(camera_name)
        self._file: Any | None = None
        if not self.enabled:
            return
        path = str(path_value).strip()
        if "{camera}" in path:
            path = path.format(camera=safe_node_name(camera_name))
        else:
            base, ext = os.path.splitext(path)
            ext = ext or ".jsonl"
            path = f"{base}_{safe_node_name(camera_name)}{ext}"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._file = open(path, "a", encoding="utf-8")

    def record(
        self,
        stream_name: str,
        *,
        publish_ros_ns: int,
        publish_start_ns: int,
        publish_done_ns: int,
        encoding: str,
        height: int,
        width: int,
        bytes_len: int,
    ) -> None:
        if self._file is None:
            return
        payload = {
            "event": "ros_publish",
            "camera": self.camera_name,
            "stream": str(stream_name),
            "name": f"{self.camera_name}_{stream_name}",
            "publish_ros_ns": int(publish_ros_ns),
            "publish_start_ns": int(publish_start_ns),
            "publish_done_ns": int(publish_done_ns),
            "publish_call_ms": (int(publish_done_ns) - int(publish_start_ns)) / 1_000_000.0,
            "encoding": str(encoding),
            "height": int(height),
            "width": int(width),
            "bytes": int(bytes_len),
        }
        self._file.write(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def _stamp_msg_to_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _positive_int(value: Any, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_float(value: Any, name: str) -> float:
    value = float(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value
