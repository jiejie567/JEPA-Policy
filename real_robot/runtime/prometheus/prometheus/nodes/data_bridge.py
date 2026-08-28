from __future__ import annotations

import importlib
import threading
import time
from dataclasses import dataclass
from typing import Any

from prometheus.data.types import DataStream
from prometheus.data.types import msg_stamp_ns
from prometheus.sessions.data import DataSession


SPIN_TIMEOUT_S = 0.02
DEFAULT_CAMERA_TIMESTAMP_STREAMS = (
    "base_0_color",
    "left_wrist_0_color",
    "right_wrist_0_color",
)


@dataclass
class _DomainRuntime:
    domain_id: int | None
    rclpy: Any
    context: Any
    node: Any
    executor: Any
    subscriptions: list[Any]
    thread: threading.Thread


class RosDataBridge:
    """Ingress ROS topic samples into one DataSession."""

    def __init__(
        self,
        *,
        data: DataSession,
        name: str = "ros_data",
        wait_all_streams: bool = True,
        qos_depth: int = 10,
        sensor_qos: str = "sensor",
        print_camera_timestamps: bool = False,
        camera_timestamp_streams: tuple[str, ...] = DEFAULT_CAMERA_TIMESTAMP_STREAMS,
        print_camera_rates: bool = False,
        camera_rate_report_s: float = 2.0,
    ):
        self.data = data
        self.name = str(name)
        self.wait_all_streams = bool(wait_all_streams)
        self.qos_depth = int(qos_depth)
        self.sensor_qos = _normalize_sensor_qos(sensor_qos)
        self.print_camera_timestamps = bool(print_camera_timestamps)
        self.print_camera_rates = bool(print_camera_rates)
        self._camera_timestamp_printer = _CameraTimestampPrinter(camera_timestamp_streams)
        self._camera_rate_printer = _CameraRatePrinter(
            camera_timestamp_streams,
            report_s=float(camera_rate_report_s),
        )
        if not self.name:
            raise ValueError("bridge name must be non-empty")
        if self.qos_depth <= 0:
            raise ValueError("qos_depth must be positive")
        self._stream_groups = _streams_by_domain(self.data.streams.values())
        self._runtimes: list[_DomainRuntime] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._error = ""
        self._started = False
        self._closed = False

    def start(self) -> None:
        if self._started:
            return
        if not self._stream_groups:
            raise ValueError("RosDataBridge requires at least one data stream")
        self._stop.clear()
        self._error = ""
        try:
            for domain_id, streams in self._stream_groups.items():
                self._runtimes.append(self._start_domain(domain_id, streams))
            self._started = True
            self._closed = False
        except Exception:
            self.close()
            raise

    def wait_ready(self, timeout_s: float) -> None:
        if not self._started:
            raise RuntimeError("RosDataBridge has not been started")
        deadline = time.monotonic() + float(timeout_s)
        names = tuple(self.data.streams)
        while time.monotonic() < deadline:
            self._raise_if_failed()
            if not self.wait_all_streams or self.data.ready(names):
                return
            time.sleep(SPIN_TIMEOUT_S)
        counts = self.data.counts()
        raise TimeoutError(f"timed out waiting for ROS data streams: counts={counts}")

    def stop(self) -> None:
        if not self._started and not self._runtimes:
            return
        self._stop.set()
        for runtime in self._runtimes:
            runtime.thread.join(timeout=1.0)
        for runtime in reversed(self._runtimes):
            self._close_domain(runtime)
        self._runtimes.clear()
        self._started = False

    def close(self) -> None:
        self.stop()
        self._closed = True

    def status(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "name": self.name,
            "started": self._started,
            "closed": self._closed,
            "wait_all_streams": self.wait_all_streams,
            "sensor_qos": self.sensor_qos,
            "print_camera_timestamps": self.print_camera_timestamps,
            "print_camera_rates": self.print_camera_rates,
            "error": self._error,
            "domains": {
                _domain_key(domain_id): [stream.name for stream in streams]
                for domain_id, streams in self._stream_groups.items()
            },
            "counts": self.data.counts(),
        }

    def _start_domain(self, domain_id: int | None, streams: list[DataStream]) -> _DomainRuntime:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data

        context = rclpy.Context()
        rclpy.init(args=None, context=context, domain_id=domain_id)
        node = Node(_node_name(self.name, domain_id), context=context)
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)
        subscriptions = []

        try:
            for stream in streams:
                msg_type = import_message_type(stream.msg_type)
                qos = (
                    (qos_profile_sensor_data if self.sensor_qos == "sensor" else self.qos_depth)
                    if stream.domain == "sensor"
                    else self.qos_depth
                )
                subscriptions.append(
                    node.create_subscription(msg_type, stream.topic, self._callback(stream.name), qos)
                )
        except Exception:
            executor.remove_node(node)
            executor.shutdown()
            node.destroy_node()
            if context.ok():
                rclpy.shutdown(context=context)
            raise

        thread = threading.Thread(
            target=self._spin,
            args=(domain_id, context, executor),
            name=f"{self.name}_{_domain_key(domain_id)}",
            daemon=True,
        )
        thread.start()
        return _DomainRuntime(domain_id, rclpy, context, node, executor, subscriptions, thread)

    def _spin(self, domain_id: int | None, context: Any, executor: Any) -> None:
        try:
            while not self._stop.is_set() and context.ok():
                executor.spin_once(timeout_sec=SPIN_TIMEOUT_S)
        except Exception as exc:
            with self._lock:
                self._error = f"domain {_domain_key(domain_id)}: {exc}"

    def _close_domain(self, runtime: _DomainRuntime) -> None:
        runtime.executor.remove_node(runtime.node)
        runtime.executor.shutdown()
        runtime.node.destroy_node()
        if runtime.context.ok():
            runtime.rclpy.shutdown(context=runtime.context)

    def _raise_if_failed(self) -> None:
        with self._lock:
            error = self._error
        if error:
            raise RuntimeError(error)

    def _callback(self, name: str):
        def _wrapped(msg: Any) -> Any:
            callback_ns = time.time_ns()
            header_stamp_ns = msg_stamp_ns(msg)
            sample = self.data.ingest(
                name,
                msg,
                recv_ns=callback_ns,
                metadata={
                    "bridge_callback_ns": callback_ns,
                    "subscriber_recv_ns": callback_ns,
                    "publish_ros_ns": header_stamp_ns,
                    "publish_wall_estimate_ns": header_stamp_ns,
                    "publish_wall_estimate_source": "msg.header.stamp",
                },
            )
            if self.print_camera_timestamps:
                self._camera_timestamp_printer.update(sample)
            if self.print_camera_rates:
                self._camera_rate_printer.update(sample)
            return sample

        return _wrapped


class _CameraTimestampPrinter:
    def __init__(self, streams: tuple[str, ...]):
        self.streams = tuple(str(stream) for stream in streams)
        self.latest: dict[str, tuple[int, float | None]] = {}
        self.previous_stamp_ns: dict[str, int] = {}
        self._lock = threading.Lock()

    def update(self, sample: Any) -> None:
        name = str(sample.name)
        if name not in self.streams:
            return
        with self._lock:
            stamp_ns = int(sample.stamp_ns)
            previous = self.previous_stamp_ns.get(name)
            dt_ms = None if previous is None else (stamp_ns - previous) / 1_000_000.0
            self.previous_stamp_ns[name] = stamp_ns
            self.latest[name] = (stamp_ns, dt_ms)
            if not all(stream in self.latest for stream in self.streams):
                return
            print(", ".join(self._format(stream) for stream in self.streams), flush=True)

    def _format(self, stream: str) -> str:
        stamp_ns, dt_ms = self.latest[stream]
        camera = stream.removesuffix("_color")
        dt = "NA" if dt_ms is None else f"{dt_ms:.3f}ms"
        return f"'{camera}_camera: {stamp_ns}, {dt}'"


class _CameraRatePrinter:
    def __init__(self, streams: tuple[str, ...], *, report_s: float):
        self.streams = tuple(str(stream) for stream in streams)
        self.report_s = max(0.1, float(report_s))
        self.times: dict[str, list[int]] = {stream: [] for stream in self.streams}
        self.last_report = time.monotonic()
        self._lock = threading.Lock()

    def update(self, sample: Any) -> None:
        name = str(sample.name)
        if name not in self.times:
            return
        with self._lock:
            self.times[name].append(int(sample.recv_ns))
            now = time.monotonic()
            if now - self.last_report < self.report_s:
                return
            self.last_report = now
            print("record_camera_rates: " + ", ".join(self._format(stream) for stream in self.streams), flush=True)
            self.times = {stream: [] for stream in self.streams}

    def _format(self, stream: str) -> str:
        values = self.times.get(stream, [])
        camera = stream.removesuffix("_color")
        if len(values) < 2:
            return f"{camera}=NA"
        elapsed_s = (values[-1] - values[0]) / 1_000_000_000.0
        hz = (len(values) - 1) / elapsed_s if elapsed_s > 0 else 0.0
        dt_ms = [
            (current - previous) / 1_000_000.0
            for previous, current in zip(values, values[1:])
        ]
        return (
            f"{camera}={hz:.2f}Hz"
            f"(n={len(values)},dt_avg={sum(dt_ms) / len(dt_ms):.2f}ms,max={max(dt_ms):.2f}ms)"
        )


def import_message_type(path: str) -> Any:
    module_name, class_name = str(path).rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _normalize_sensor_qos(value: Any) -> str:
    qos = str(value).strip().lower()
    if qos in {"sensor", "sensor_data", "best_effort", "besteffort"}:
        return "sensor"
    if qos in {"reliable", "default"}:
        return "reliable"
    raise ValueError("data bridge sensor_qos must be 'sensor' or 'reliable'")


def _streams_by_domain(streams: Any) -> dict[int | None, list[DataStream]]:
    groups: dict[int | None, list[DataStream]] = {}
    for stream in streams:
        groups.setdefault(stream.ros_domain_id, []).append(stream)
    return {
        domain_id: sorted(items, key=lambda item: item.name)
        for domain_id, items in sorted(groups.items(), key=lambda item: -1 if item[0] is None else item[0])
    }


def _domain_key(domain_id: int | None) -> str:
    return "default" if domain_id is None else str(domain_id)


def _node_name(name: str, domain_id: int | None) -> str:
    raw = f"{name}_{_domain_key(domain_id)}"
    return "".join(char if char.isalnum() else "_" for char in raw)
