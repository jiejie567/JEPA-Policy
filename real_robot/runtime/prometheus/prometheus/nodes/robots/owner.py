from __future__ import annotations

import json
import os
import socket
import threading
import time
from collections import OrderedDict
from typing import Any, Mapping

import numpy as np

from prometheus.nodes.robots.control import (
    RobotTopics,
    checked_action_array,
    json_dumps,
    robot_topics_from_mapping,
    stamp_to_ns,
    validate_control_payload,
)
from prometheus.nodes.robots.robot_node import RobotDriver


class RobotOwnerNode:
    """ROS owner runtime that wraps a small RobotDriver."""

    def __init__(
        self,
        *,
        driver: RobotDriver,
        robot_id: str,
        topics: RobotTopics | Mapping[str, Any],
        hz: float,
        action_dims: Mapping[str, int],
        node: Any | None = None,
        node_name: str | None = None,
        ros_domain_id: int | None = None,
        qos_depth: int = 50,
        status_qos_depth: int = 10,
        command_source_timeout_s: float = 1.0,
    ):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String

        class _Node(Node):
            pass

        if not robot_id:
            raise ValueError("robot_id must be non-empty")
        if float(hz) <= 0:
            raise ValueError("hz must be positive")

        self.rclpy = rclpy
        self.JointState = JointState
        self.String = String
        self.driver = driver
        self.robot_id = str(robot_id)
        self.ros_domain_id = None if ros_domain_id is None else int(ros_domain_id)
        self.topics = topics if isinstance(topics, RobotTopics) else robot_topics_from_mapping(topics)
        self.hz = float(hz)
        self.action_dims = dict(action_dims)
        self.qos_depth = int(qos_depth)
        self.status_qos_depth = int(status_qos_depth)
        self.command_source_timeout_ns = int(float(command_source_timeout_s) * 1_000_000_000)
        self.hostname = socket.gethostname()
        self.owner_pid = os.getpid()
        self.state_count = 0
        self.active_command_source: str | None = None
        self.active_command_source_stamp_ns = 0
        self.control_result_cache: OrderedDict[str, tuple[str, dict[str, Any]]] = OrderedDict()
        self._owns_node = node is None
        self._context: Any | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._started = False
        self._closed = False
        self._error = ""

        if node is None:
            self._context = rclpy.Context()
            rclpy.init(args=None, context=self._context, domain_id=self.ros_domain_id)
            self.node = _Node(node_name or f"{self.robot_id}_robot_owner", context=self._context)
        else:
            self.node = node
            self._context = getattr(node, "context", None)
        self.executor = SingleThreadedExecutor(context=self._context)
        self.executor.add_node(self.node)

        self.state_pub = self.node.create_publisher(self.JointState, self.topics.state, self.qos_depth)
        self.status_pub = self.node.create_publisher(self.String, self.topics.status, self.status_qos_depth)
        self.result_pub = self.node.create_publisher(self.String, self.topics.result, self.status_qos_depth)
        setup_ros = getattr(self.driver, "setup_ros", None)
        if callable(setup_ros):
            setup_ros(node=self.node, qos_depth=self.qos_depth)
        self.node.create_subscription(self.String, self.topics.control, self._control_cb, self.status_qos_depth)
        self.timer = self.node.create_timer(1.0 / self.hz, self._publish_state)
        self.status_timer = self.node.create_timer(1.0, self._publish_status_timer)

    def start(self) -> None:
        if self._started:
            return
        self.driver.connect()
        self._stop.clear()
        self._error = ""
        self._thread = threading.Thread(target=self._spin, name=f"{self.robot_id}_owner", daemon=True)
        self._thread.start()
        self._started = True
        self._closed = False
        self._publish_status_safely("running")

    def wait_ready(self, timeout_s: float) -> None:
        if not self._started:
            raise RuntimeError("robot owner has not been started")
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            if self._error:
                raise RuntimeError(self._error)
            if self._thread is not None and self._thread.is_alive():
                return
            time.sleep(0.01)
        raise TimeoutError(f"timed out waiting for robot owner {self.robot_id!r}")

    def stop(self) -> None:
        if not self._started:
            return
        self._publish_status_safely("stopping")
        self.active_command_source = None
        self.active_command_source_stamp_ns = 0
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._started = False

    def spin_once(self, timeout_s: float = 0.0) -> None:
        self.executor.spin_once(timeout_sec=float(timeout_s))

    def status(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "robot_id": self.robot_id,
            "ros_domain_id": self.ros_domain_id,
            "started": self._started,
            "closed": self._closed,
            "error": self._error,
            "active_command_source": self.active_command_source,
            "state_count": self.state_count,
            "topics": {
                "state": self.topics.state,
                "control": self.topics.control,
                "result": self.topics.result,
                "status": self.topics.status,
            },
            "driver": dict(self.driver.status()),
        }

    def close(self) -> None:
        if self._closed:
            return
        self.stop()
        self._publish_status_safely("closing")
        try:
            self.driver.close()
        finally:
            try:
                self.executor.remove_node(self.node)
            finally:
                self.executor.shutdown()
            if self._owns_node:
                self.node.destroy_node()
            if self._owns_node and self._context is not None and self._context.ok():
                self.rclpy.shutdown(context=self._context)
            self._closed = True

    def _spin(self) -> None:
        try:
            while not self._stop.is_set() and self._context_ok():
                self.executor.spin_once(timeout_sec=0.02)
        except Exception as exc:
            self._error = str(exc)

    def _context_ok(self) -> bool:
        if self._context is not None:
            return bool(self._context.ok())
        return bool(self.rclpy.ok())

    def _publish_state(self) -> None:
        state = self.driver.read_state()
        stamp = self.node.get_clock().now().to_msg()
        self.state_pub.publish(self._state_msg(stamp, state))
        publish_ros_state = getattr(self.driver, "publish_ros_state", None)
        if callable(publish_ros_state):
            publish_ros_state(node=self.node, stamp=stamp, state=state)
        self.state_count += 1

    def _publish_status_timer(self) -> None:
        self._expire_command_source(self._now_ns())
        self._publish_status("running")

    def _control_cb(self, msg: Any) -> None:
        request_stamp_ns = self._now_ns()
        try:
            payload = validate_control_payload(json.loads(msg.data))
        except Exception as exc:
            self._publish_result("", False, "invalid_request", str(exc), request_stamp_ns)
            return

        request_id = payload["request_id"]
        request_signature = json_dumps(payload)
        cached = self.control_result_cache.get(request_id)
        if cached is not None:
            cached_signature, cached_result = cached
            if cached_signature != request_signature:
                self._publish_result(
                    request_id,
                    False,
                    "duplicate_request_id",
                    "request_id was already used for a different payload",
                    request_stamp_ns,
                )
                return
            self._publish_result_payload(cached_result)
            return

        try:
            self._handle_payload(payload, request_stamp_ns, request_signature)
        except Exception as exc:
            self._publish_result(
                request_id,
                False,
                "error",
                str(exc),
                request_stamp_ns,
                request_signature=request_signature,
            )

    def _handle_payload(
        self,
        payload: dict[str, Any],
        request_stamp_ns: int,
        request_signature: str,
    ) -> None:
        command = payload["command"]
        if command == "send_action":
            self._handle_send_action(payload, request_stamp_ns, request_signature)
            return
        if command == "release_control":
            self._handle_release_control(payload, request_stamp_ns, request_signature)
            return
        if command == "reset_home":
            self._handle_reset_home(payload, request_stamp_ns, request_signature)
            return
        if command == "set_teach_mode":
            self._handle_set_teach_mode(payload, request_stamp_ns, request_signature)
            return
        if command == "set_position_mode":
            self._handle_set_position_mode(payload, request_stamp_ns, request_signature)
            return
        raise ValueError(f"unsupported command {command!r}")

    def _handle_send_action(
        self,
        payload: dict[str, Any],
        request_stamp_ns: int,
        request_signature: str,
    ) -> None:
        request_id = payload["request_id"]
        command_source = str(payload["command_source"])
        action_space = str(payload["action_space"])
        action = checked_action_array(
            payload["action"],
            action_space=action_space,
            action_dims=self.action_dims,
        )
        self._check_command_source(command_source, request_stamp_ns)

        self.driver.send_action(action_space, action)
        self.active_command_source = command_source
        self.active_command_source_stamp_ns = int(request_stamp_ns)
        self._publish_result(
            request_id,
            True,
            "action_sent",
            "",
            request_stamp_ns,
            request_signature=request_signature,
            command_source=command_source,
            action_space=action_space,
        )

    def _handle_release_control(
        self,
        payload: dict[str, Any],
        request_stamp_ns: int,
        request_signature: str,
    ) -> None:
        request_id = payload["request_id"]
        command_source = str(payload["command_source"])
        if self.active_command_source not in (None, command_source):
            self._publish_result(
                request_id,
                False,
                "wrong_command_source",
                f"active command_source is {self.active_command_source!r}",
                request_stamp_ns,
                request_signature=request_signature,
                command_source=command_source,
            )
            return

        self.active_command_source = None
        self.active_command_source_stamp_ns = 0
        self._publish_result(
            request_id,
            True,
            "control_released",
            "",
            request_stamp_ns,
            request_signature=request_signature,
            command_source=command_source,
        )

    def _handle_reset_home(
        self,
        payload: dict[str, Any],
        request_stamp_ns: int,
        request_signature: str,
    ) -> None:
        request_id = payload["request_id"]
        command_source = str(payload["command_source"])
        self._check_command_source(command_source, request_stamp_ns)

        reset = getattr(self.driver, "reset_home", None)
        if not callable(reset):
            reset = getattr(self.driver, "reset_to_home", None)
        if not callable(reset):
            raise RuntimeError(f"{type(self.driver).__name__} does not support reset_home")

        reset()
        self.active_command_source = None
        self.active_command_source_stamp_ns = 0
        self._publish_result(
            request_id,
            True,
            "home_reset",
            "",
            request_stamp_ns,
            request_signature=request_signature,
            command_source=command_source,
        )

    def _handle_set_teach_mode(
        self,
        payload: dict[str, Any],
        request_stamp_ns: int,
        request_signature: str,
    ) -> None:
        self._handle_driver_mode(
            payload,
            request_stamp_ns,
            request_signature,
            method_name="set_teach_mode",
            success_status="teach_mode_set",
        )

    def _handle_set_position_mode(
        self,
        payload: dict[str, Any],
        request_stamp_ns: int,
        request_signature: str,
    ) -> None:
        self._handle_driver_mode(
            payload,
            request_stamp_ns,
            request_signature,
            method_name="set_position_mode",
            success_status="position_mode_set",
        )

    def _handle_driver_mode(
        self,
        payload: dict[str, Any],
        request_stamp_ns: int,
        request_signature: str,
        *,
        method_name: str,
        success_status: str,
    ) -> None:
        request_id = payload["request_id"]
        command_source = str(payload["command_source"])
        self._check_command_source(command_source, request_stamp_ns)

        mode_switch = getattr(self.driver, method_name, None)
        if not callable(mode_switch):
            raise RuntimeError(f"{type(self.driver).__name__} does not support {method_name}")

        mode_switch()
        self.active_command_source_stamp_ns = int(request_stamp_ns)
        self._publish_result(
            request_id,
            True,
            success_status,
            "",
            request_stamp_ns,
            request_signature=request_signature,
            command_source=command_source,
        )

    def _check_command_source(self, command_source: str, stamp_ns: int) -> None:
        if not command_source:
            raise ValueError("command_source must be non-empty")
        self._expire_command_source(stamp_ns)
        if self.active_command_source in (None, command_source):
            return
        raise RuntimeError(f"active command_source is {self.active_command_source!r}")

    def _expire_command_source(self, now_ns: int) -> None:
        if self.active_command_source is None or self.command_source_timeout_ns <= 0:
            return
        if now_ns - self.active_command_source_stamp_ns > self.command_source_timeout_ns:
            self.active_command_source = None
            self.active_command_source_stamp_ns = 0

    def _state_msg(self, stamp: Any, state: Any) -> Any:
        names = list(state.joint_names)
        size = len(names)
        msg = self.JointState()
        msg.header.stamp = stamp
        msg.header.frame_id = f"{self.robot_id}/state"
        msg.name = names
        msg.position = [float(value) for value in np.asarray(state.qpos, dtype=np.float32).reshape(size)]
        if state.qvel is not None:
            msg.velocity = [float(value) for value in np.asarray(state.qvel, dtype=np.float32).reshape(size)]
        if state.effort is not None:
            msg.effort = [float(value) for value in np.asarray(state.effort, dtype=np.float32).reshape(size)]
        return msg

    def _publish_status(self, state: str) -> None:
        payload = self.status()
        payload.update(
            {
                "state": state,
                "owner_pid": self.owner_pid,
                "owner_host": self.hostname,
                "stamp_ns": time.time_ns(),
            }
        )
        msg = self.String()
        msg.data = json_dumps(payload)
        self.status_pub.publish(msg)

    def _publish_status_safely(self, state: str) -> None:
        try:
            self._publish_status(state)
        except Exception as exc:
            self._error = str(exc)

    def _publish_result(
        self,
        request_id: str,
        accepted: bool,
        status: str,
        message: str,
        request_stamp_ns: int,
        request_signature: str = "",
        **extra: Any,
    ) -> None:
        payload = {
            "request_id": request_id,
            "accepted": bool(accepted),
            "status": status,
            "message": message,
            "request_stamp_ns": int(request_stamp_ns),
            "result_stamp_ns": self._now_ns(),
            "active_command_source": self.active_command_source,
        }
        payload.update(extra)
        if request_id and request_signature:
            self.control_result_cache[request_id] = (request_signature, dict(payload))
            self.control_result_cache.move_to_end(request_id)
            while len(self.control_result_cache) > 256:
                self.control_result_cache.popitem(last=False)
        self._publish_result_payload(payload)

    def _publish_result_payload(self, payload: Mapping[str, Any]) -> None:
        msg = self.String()
        msg.data = json_dumps(payload)
        self.result_pub.publish(msg)

    def _now_ns(self) -> int:
        return stamp_to_ns(self.node.get_clock().now().to_msg())
