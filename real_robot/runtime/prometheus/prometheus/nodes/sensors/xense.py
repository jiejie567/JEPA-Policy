from __future__ import annotations

import ctypes
import multiprocessing as mp
import os
import queue
import signal
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TACTILE_TOPIC_PREFIX = "/prometheus/sensors/tactile"
TACTILE_POINT_FIELDS = ("x", "y", "z", "dx", "dy", "dz")
TACTILE_IMAGE_WIDTH = 700
TACTILE_IMAGE_HEIGHT = 400
TACTILE_FLOW_WIDTH = 35
TACTILE_FLOW_HEIGHT = 20
LOG_HISTORY = 80

XENSE_GPU_LIBRARIES = (
    "libcudart.so.12",
    "libcublasLt.so.12",
    "libcublas.so.12",
    "libcurand.so.10",
    "libcufft.so.11",
    "libnvrtc.so.12",
    "libcudnn.so.9",
    "libcudnn_ops.so.9",
    "libcudnn_graph.so.9",
    "libcudnn_heuristic.so.9",
    "libcudnn_engines_runtime_compiled.so.9",
    "libcudnn_engines_precompiled.so.9",
    "libcudnn_adv.so.9",
    "libcudnn_cnn.so.9",
)
XENSE_FRAME_FIELDS = ("raw", "difference", "mesh", "flow", "sdk_timestamp")


@dataclass(frozen=True)
class TactileResize:
    scale: float | None = None
    width: int | None = None
    height: int | None = None

    @property
    def enabled(self) -> bool:
        return self.width is not None or self.height is not None or self.scale not in (None, 1.0)


class XenseTactileSensorNode:
    """HardwareSession-owned multiprocess Xense tactile publisher."""

    def __init__(
        self,
        *,
        name: str = "xense_tactile",
        sensors: Mapping[str, str],
        topic_prefix: str = TACTILE_TOPIC_PREFIX,
        publish_hz: float = 30.0,
        prepub_resize: Any = 1.0,
        publish_flow: bool = True,
        publish_difference: bool = False,
        publish_raw: bool = False,
        require_gpu: bool = True,
        qos_depth: int = 20,
        ros_domain_id: int | None = None,
    ):
        self.name = str(name)
        self.sensors = normalize_tactile_map(sensors)
        if not self.sensors:
            raise ValueError("XenseTactileSensorNode requires at least one tactile sensor")
        self.topic_prefix = str(topic_prefix).rstrip("/")
        self.publish_hz = _positive_float(publish_hz, "publish_hz")
        self.prepub_resize = normalize_resize(prepub_resize)
        self.publish_flow = bool(publish_flow)
        self.publish_difference = bool(publish_difference)
        self.publish_raw = bool(publish_raw)
        self.require_gpu = bool(require_gpu)
        self.qos_depth = _positive_int(qos_depth, "qos_depth")
        self.ros_domain_id = None if ros_domain_id is None else int(ros_domain_id)
        if not (self.publish_flow or self.publish_difference or self.publish_raw):
            raise ValueError("Xense tactile requires at least one enabled output")
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
        self._events = {name: [] for name in self.sensors}
        for sensor_name, serial in self.sensors.items():
            proc = self._ctx.Process(
                target=_tactile_worker,
                name=f"{self.name}_{sensor_name}",
                kwargs={
                    "sensor_name": sensor_name,
                    "serial": serial,
                    "topic_prefix": self.topic_prefix,
                    "publish_hz": self.publish_hz,
                    "prepub_resize": self.prepub_resize,
                    "publish_flow": self.publish_flow,
                    "publish_difference": self.publish_difference,
                    "publish_raw": self.publish_raw,
                    "require_gpu": self.require_gpu,
                    "qos_depth": self.qos_depth,
                    "ros_domain_id": self.ros_domain_id,
                    "ready_queue": self._ready_queue,
                    "event_queue": self._event_queue,
                },
            )
            proc.start()
            self._procs[sensor_name] = proc
            self._events[sensor_name].append(f"started pid={proc.pid}")
        self._started = True
        self._closed = False

    def wait_ready(self, timeout_s: float) -> None:
        if not self._started:
            raise RuntimeError("Xense tactile sensor has not been started")
        deadline = time.monotonic() + float(timeout_s)
        expected = set(self.sensors)
        while time.monotonic() < deadline:
            self._drain_queues()
            self._raise_on_launch_exit()
            if set(self._ready_names) == expected:
                return
            time.sleep(0.05)
        raise TimeoutError(
            "timed out waiting for Xense tactile sensors: "
            f"ready={sorted(self._ready_names)} expected={sorted(expected)} events={self._recent_logs()}"
        )

    def stop(self) -> None:
        for proc in self._procs.values():
            if proc.is_alive():
                proc.terminate()
        for proc in self._procs.values():
            proc.join(timeout=3.0)
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
        return {
            "type": type(self).__name__,
            "name": self.name,
            "started": self._started,
            "closed": self._closed,
            "ready": set(self._ready_names) == set(self.sensors),
            "ready_names": sorted(self._ready_names),
            "sensors": dict(self.sensors),
            "topic_prefix": self.topic_prefix,
            "ros_domain_id": self.ros_domain_id,
            "publish_hz": self.publish_hz,
            "prepub_resize": resize_status(self.prepub_resize),
            "publish_flow": self.publish_flow,
            "publish_difference": self.publish_difference,
            "publish_raw": self.publish_raw,
            "require_gpu": self.require_gpu,
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
                    sensor_name, message = self._event_queue.get_nowait()
                except queue.Empty:
                    break
                events = self._events.setdefault(str(sensor_name), [])
                events.append(str(message))
                del events[:-LOG_HISTORY]

    def _raise_on_launch_exit(self) -> None:
        dead = {
            name: proc.exitcode
            for name, proc in self._procs.items()
            if proc.exitcode is not None and name not in self._ready_names
        }
        if dead:
            raise RuntimeError(f"Xense tactile process exited before ready: {dead}; events={self._recent_logs()}")

    def _recent_logs(self) -> dict[str, list[str]]:
        self._drain_queues()
        return {name: list(lines)[-10:] for name, lines in self._events.items()}


def _tactile_worker(
    *,
    sensor_name: str,
    serial: str,
    topic_prefix: str,
    publish_hz: float,
    prepub_resize: TactileResize,
    publish_flow: bool,
    publish_difference: bool,
    publish_raw: bool,
    require_gpu: bool,
    qos_depth: int,
    ros_domain_id: int | None,
    ready_queue: Any,
    event_queue: Any,
) -> None:
    preload_xense_gpu_libraries(required=require_gpu)

    import numpy as np
    import rclpy
    import sensor_msgs_py.point_cloud2 as pc2
    from cv_bridge import CvBridge
    from rclpy.signals import SignalHandlerOptions
    from sensor_msgs.msg import Image, PointCloud2, PointField
    from std_msgs.msg import Header
    from xensesdk import Sensor

    shutdown_requested = False

    def _request_shutdown(_signum: int, _frame: Any) -> None:
        nonlocal shutdown_requested
        shutdown_requested = True

    signal.signal(signal.SIGINT, _request_shutdown)
    signal.signal(signal.SIGTERM, _request_shutdown)

    sensor = Sensor.create(str(serial))
    using_gpu = bool(getattr(sensor, "infer_engine_using_gpu", False))
    if require_gpu and not using_gpu:
        release = getattr(sensor, "release", None)
        if callable(release):
            release()
        raise RuntimeError(f"Xense {sensor_name} serial={serial} fell back to CPU while require_gpu=true")

    rclpy.init(args=None, domain_id=ros_domain_id, signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node(f"prometheus_xense_{safe_node_name(sensor_name)}")
    bridge = CvBridge()
    topics = tactile_topics(
        sensor_name,
        topic_prefix=topic_prefix,
        publish_flow=publish_flow,
        publish_difference=publish_difference,
        publish_raw=publish_raw,
    )
    point_fields = [
        PointField(name=name, offset=index * 4, datatype=PointField.FLOAT32, count=1)
        for index, name in enumerate(TACTILE_POINT_FIELDS)
    ]
    publishers = {}
    if publish_flow:
        publishers["flow"] = node.create_publisher(PointCloud2, topics["flow"], qos_depth)
    if publish_difference:
        publishers["difference"] = node.create_publisher(Image, topics["difference"], qos_depth)
    if publish_raw:
        publishers["raw"] = node.create_publisher(Image, topics["raw"], qos_depth)

    event_queue.put((sensor_name, f"sensor created serial={serial} inference={'GPU' if using_gpu else 'CPU'} publish_hz={publish_hz}"))
    period = 1.0 / max(1.0, float(publish_hz))
    next_time = time.monotonic()
    ready_reported = False
    try:
        while rclpy.ok() and not shutdown_requested:
            outputs = read_xense_frame(sensor, Sensor.OutputType)
            stamp = node.get_clock().now().to_msg()
            header = Header()
            header.stamp = stamp
            header.frame_id = f"{sensor_name}_tactile_frame"
            published: set[str] = set()

            if publish_flow:
                mesh = np.asarray(outputs["mesh"], dtype=np.float32).reshape(-1, 3)
                flow = np.asarray(outputs["flow"], dtype=np.float32).reshape(-1, 3)
                expected_points = TACTILE_FLOW_WIDTH * TACTILE_FLOW_HEIGHT
                if mesh.shape != (expected_points, 3) or flow.shape != (expected_points, 3):
                    raise ValueError(
                        f"Xense {sensor_name} flow must be {expected_points}x3 mesh and flow, "
                        f"got mesh={mesh.shape} flow={flow.shape}"
                    )
                points = np.concatenate((mesh, flow), axis=1).astype(np.float32, copy=False)
                publishers["flow"].publish(pc2.create_cloud(header, point_fields, points))
                published.add("flow")
            if publish_difference:
                _publish_tactile_image(
                    bridge,
                    publishers["difference"],
                    sensor_name,
                    "difference",
                    outputs["difference"],
                    stamp,
                    prepub_resize,
                )
                published.add("difference")
            if publish_raw:
                _publish_tactile_image(
                    bridge,
                    publishers["raw"],
                    sensor_name,
                    "raw",
                    outputs["raw"],
                    stamp,
                    prepub_resize,
                )
                published.add("raw")

            expected = {name for name, enabled in (("flow", publish_flow), ("difference", publish_difference), ("raw", publish_raw)) if enabled}
            if not ready_reported and expected.issubset(published):
                ready_queue.put(sensor_name)
                ready_reported = True

            rclpy.spin_once(node, timeout_sec=0.0)
            next_time += period
            time.sleep(max(0.0, next_time - time.monotonic()))
    except Exception as exc:
        event_queue.put((sensor_name, f"error: {type(exc).__name__}: {exc}"))
        raise
    finally:
        release = getattr(sensor, "release", None)
        if callable(release):
            try:
                release()
            except Exception:
                pass
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


def preload_xense_gpu_libraries(*, library_dir: str | Path | None = None, required: bool) -> tuple[str, ...]:
    root = Path(library_dir) if library_dir is not None else Path(sys.prefix) / "lib"
    loaded: list[str] = []
    errors: list[str] = []
    for library_name in XENSE_GPU_LIBRARIES:
        path = root / library_name
        try:
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            errors.append(f"{library_name}: {exc}")
        else:
            loaded.append(library_name)
    if required and errors:
        raise RuntimeError(f"Xense GPU runtime preload failed under {root}: " + "; ".join(errors))
    return tuple(loaded)


def read_xense_frame(sensor: Any, output_type: Any) -> dict[str, Any]:
    values = tuple(
        sensor.selectSensorInfo(
            output_type.Rectify,
            output_type.Difference,
            output_type.Mesh3D,
            output_type.Mesh3DFlow,
            output_type.TimeStamp,
        )
    )
    if len(values) != len(XENSE_FRAME_FIELDS):
        raise RuntimeError(f"Xense selectSensorInfo returned {len(values)} values; expected {len(XENSE_FRAME_FIELDS)}")
    return dict(zip(XENSE_FRAME_FIELDS, values, strict=True))


def _publish_tactile_image(
    bridge: Any,
    publisher: Any,
    sensor_name: str,
    stream: str,
    image: Any,
    stamp: Any,
    resize: TactileResize,
) -> None:
    import numpy as np

    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        arr = arr[:, :, :3]
    else:
        raise ValueError(f"unsupported Xense {stream} image shape for {sensor_name}: {arr.shape}")
    if arr.shape[:2] == (TACTILE_IMAGE_WIDTH, TACTILE_IMAGE_HEIGHT):
        arr = np.swapaxes(arr, 0, 1)
    if arr.shape[:2] != (TACTILE_IMAGE_HEIGHT, TACTILE_IMAGE_WIDTH):
        raise ValueError(
            f"Xense {sensor_name} {stream} must be {TACTILE_IMAGE_WIDTH}x{TACTILE_IMAGE_HEIGHT}, "
            f"got {arr.shape[1]}x{arr.shape[0]}"
        )
    arr = _resize_image(arr, resize)
    msg = bridge.cv2_to_imgmsg(arr, encoding="rgb8")
    msg.header.stamp = stamp
    msg.header.frame_id = f"{sensor_name}_{stream}_frame"
    publisher.publish(msg)


def _resize_image(image: Any, resize: TactileResize) -> Any:
    if not resize.enabled:
        return image
    import cv2

    target_width, target_height = _target_size(image.shape[1], image.shape[0], resize)
    if (image.shape[1], image.shape[0]) == (target_width, target_height):
        return image
    interpolation = cv2.INTER_AREA if target_width <= image.shape[1] and target_height <= image.shape[0] else cv2.INTER_LINEAR
    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)


def tactile_topics(
    sensor_name: str,
    *,
    topic_prefix: str = TACTILE_TOPIC_PREFIX,
    publish_flow: bool = True,
    publish_difference: bool = False,
    publish_raw: bool = False,
) -> dict[str, str]:
    root = f"{topic_prefix.rstrip('/')}/{sensor_name}"
    topics = {}
    if publish_flow:
        topics["flow"] = f"{root}/pointcloud"
    if publish_difference:
        topics["difference"] = f"{root}/difference"
    if publish_raw:
        topics["raw"] = f"{root}/raw"
    return topics


def streams(robot: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    cfg = dict(robot.get("tactile") or {})
    serials = normalize_tactile_map(cfg.get("serial_no", {}))
    if not serials:
        return {}
    sensor_domain_id = int(robot["sensor_domain_id"])
    topic_prefix = str(cfg.get("topic_prefix", TACTILE_TOPIC_PREFIX))
    publish_flow = bool(cfg.get("publish_flow", True))
    publish_difference = bool(cfg.get("publish_difference", False))
    publish_raw = bool(cfg.get("publish_raw", False))
    items: dict[str, dict[str, Any]] = {}
    for sensor_name in serials:
        topics = tactile_topics(
            sensor_name,
            topic_prefix=topic_prefix,
            publish_flow=publish_flow,
            publish_difference=publish_difference,
            publish_raw=publish_raw,
        )
        if "flow" in topics:
            items[f"{sensor_name}_tactile_flow"] = {
                "topic": topics["flow"],
                "msg_type": "sensor_msgs.msg.PointCloud2",
                "domain": "sensor",
                "ros_domain_id": sensor_domain_id,
            }
        if "difference" in topics:
            items[f"{sensor_name}_tactile_difference"] = {
                "topic": topics["difference"],
                "msg_type": "sensor_msgs.msg.Image",
                "domain": "sensor",
                "ros_domain_id": sensor_domain_id,
            }
        if "raw" in topics:
            items[f"{sensor_name}_tactile_raw"] = {
                "topic": topics["raw"],
                "msg_type": "sensor_msgs.msg.Image",
                "domain": "sensor",
                "ros_domain_id": sensor_domain_id,
            }
    return items


def runtime_config(robot: Mapping[str, Any]) -> dict[str, Any]:
    cfg = dict(robot.get("tactile") or {})
    serials = normalize_tactile_map(cfg.get("serial_no", {}))
    if not serials:
        return {}
    return {
        "tactile": {
            "_target_": "prometheus.nodes.sensors.xense.XenseTactileSensorNode",
            "name": "xense_tactile",
            "sensors": serials,
            "topic_prefix": str(cfg.get("topic_prefix", TACTILE_TOPIC_PREFIX)),
            "publish_hz": float(cfg.get("publish_hz", 30.0)),
            "prepub_resize": cfg.get("prepub_resize", 1.0),
            "publish_flow": bool(cfg.get("publish_flow", True)),
            "publish_difference": bool(cfg.get("publish_difference", False)),
            "publish_raw": bool(cfg.get("publish_raw", False)),
            "require_gpu": bool(cfg.get("require_gpu", True)),
            "qos_depth": int(cfg.get("qos_depth", 20)),
            "ros_domain_id": int(robot["sensor_domain_id"]),
        }
    }


def normalize_tactile_map(value: Mapping[str, Any]) -> dict[str, str]:
    sensors: dict[str, str] = {}
    serials: dict[str, str] = {}
    for key, serial in value.items():
        name = str(key).strip()
        if not name:
            raise ValueError("Xense tactile sensor name must be non-empty")
        if name != safe_node_name(name):
            raise ValueError(f"Xense tactile sensor name {name!r} must be a ROS-safe name")
        if isinstance(serial, Mapping):
            serial = serial.get("serial_no", "")
        serial = str(serial).strip()
        if not serial:
            raise ValueError(f"Xense tactile sensor {name!r} serial must be non-empty")
        if serial in serials:
            raise ValueError(f"Xense tactile serial {serial!r} is used by {serials[serial]!r} and {name!r}")
        sensors[name] = serial
        serials[serial] = name
    return sensors


def normalize_resize(value: Any) -> TactileResize:
    if value is None:
        return TactileResize(scale=1.0)
    if isinstance(value, Mapping):
        if "scale" in value:
            return TactileResize(scale=_positive_float(value["scale"], "prepub_resize.scale"))
        if "width" in value and "height" in value:
            return TactileResize(width=_positive_int(value["width"], "prepub_resize.width"), height=_positive_int(value["height"], "prepub_resize.height"))
        raise ValueError("prepub_resize mapping must contain scale or width/height")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ValueError("prepub_resize sequence must be [width, height]")
        return TactileResize(width=_positive_int(value[0], "prepub_resize.width"), height=_positive_int(value[1], "prepub_resize.height"))
    return TactileResize(scale=_positive_float(value, "prepub_resize"))


def resize_status(resize: TactileResize) -> dict[str, Any]:
    return {"scale": resize.scale, "width": resize.width, "height": resize.height, "enabled": resize.enabled}


def _target_size(width: int, height: int, resize: TactileResize) -> tuple[int, int]:
    if resize.width is not None and resize.height is not None:
        return int(resize.width), int(resize.height)
    if resize.scale is not None:
        return max(1, int(round(width * resize.scale))), max(1, int(round(height * resize.scale)))
    return int(width), int(height)


def _positive_int(value: Any, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_float(value: Any, name: str) -> float:
    result = float(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def safe_node_name(name: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in str(name))
