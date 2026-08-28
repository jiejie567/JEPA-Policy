from __future__ import annotations

import json
import time
from typing import Any

from prometheus.nodes.robots.control import (
    checked_action_array,
    default_command_source,
    json_dumps,
    validate_control_payload,
)


class RobotControlClient:
    """Small ROS client for sending actions to a robot owner."""

    def __init__(
        self,
        *,
        action_space: str,
        action_dim: int,
        control_topic: str,
        result_topic: str,
        command_source: str | None = None,
        node: Any | None = None,
        node_name: str = "prometheus_robot_control_client",
        ros_domain_id: int | None = None,
        qos_depth: int = 10,
        retry_interval_s: float = 0.25,
    ):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from std_msgs.msg import String

        class _Node(Node):
            pass

        if not action_space:
            raise ValueError("action_space must be non-empty")
        if int(action_dim) <= 0:
            raise ValueError("action_dim must be positive")
        if not control_topic:
            raise ValueError("control_topic must be non-empty")
        if not result_topic:
            raise ValueError("result_topic must be non-empty")

        self.rclpy = rclpy
        self.String = String
        self.action_space = str(action_space)
        self.action_dims = {self.action_space: int(action_dim)}
        self.control_topic = str(control_topic)
        self.result_topic = str(result_topic)
        self.ros_domain_id = None if ros_domain_id is None else int(ros_domain_id)
        self.command_source = command_source or default_command_source()
        self.retry_interval_s = float(retry_interval_s)
        self.request_id = ""
        self.result: dict[str, Any] | None = None
        self._seq = 0
        self._owns_node = node is None
        self._context: Any | None = None
        self._closed = False

        if node is None:
            self._context = rclpy.Context()
            rclpy.init(args=None, context=self._context, domain_id=self.ros_domain_id)
            self.node = _Node(node_name, context=self._context)
        else:
            self.node = node
            self._context = getattr(node, "context", None)
        self.executor = SingleThreadedExecutor(context=self._context)
        self.executor.add_node(self.node)

        self.pub = self.node.create_publisher(String, self.control_topic, int(qos_depth))
        self.node.create_subscription(String, self.result_topic, self._result_cb, int(qos_depth))

    def __enter__(self) -> RobotControlClient:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def send_action(
        self,
        action: Any,
        *,
        timeout: float = 3.0,
        wait: bool = True,
    ) -> dict[str, Any] | None:
        values = checked_action_array(
            action,
            action_space=self.action_space,
            action_dims=self.action_dims,
        )
        return self._request(
            {
                "command": "send_action",
                "command_source": self.command_source,
                "action_space": self.action_space,
                "action": values.tolist(),
            },
            timeout=timeout,
            wait=wait,
        )

    def release_control(
        self,
        *,
        timeout: float = 3.0,
        wait: bool = True,
    ) -> dict[str, Any] | None:
        return self._request(
            {
                "command": "release_control",
                "command_source": self.command_source,
            },
            timeout=timeout,
            wait=wait,
        )

    def reset_home(
        self,
        *,
        timeout: float = 30.0,
        wait: bool = True,
    ) -> dict[str, Any] | None:
        return self._request(
            {
                "command": "reset_home",
                "command_source": self.command_source,
            },
            timeout=timeout,
            wait=wait,
        )

    def set_teach_mode(
        self,
        *,
        timeout: float = 5.0,
        wait: bool = True,
    ) -> dict[str, Any] | None:
        return self._request(
            {
                "command": "set_teach_mode",
                "command_source": self.command_source,
            },
            timeout=timeout,
            wait=wait,
        )

    def set_position_mode(
        self,
        *,
        timeout: float = 5.0,
        wait: bool = True,
    ) -> dict[str, Any] | None:
        return self._request(
            {
                "command": "set_position_mode",
                "command_source": self.command_source,
            },
            timeout=timeout,
            wait=wait,
        )

    def wait_for_owner(self, timeout_s: float = 5.0) -> None:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            if self.pub.get_subscription_count() > 0:
                return
            self.spin_once(0.05)
        raise TimeoutError(f"no robot owner ready on {self.control_topic}")

    def spin_once(self, timeout_s: float = 0.0) -> None:
        self.executor.spin_once(timeout_sec=float(timeout_s))

    def status(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "action_space": self.action_space,
            "ros_domain_id": self.ros_domain_id,
            "control_topic": self.control_topic,
            "result_topic": self.result_topic,
            "command_source": self.command_source,
            "request_id": self.request_id,
            "closed": self._closed,
        }

    def close(self) -> None:
        if self._closed:
            return
        self.executor.remove_node(self.node)
        self.executor.shutdown()
        if self._owns_node:
            self.node.destroy_node()
        if self._owns_node and self._context is not None and self._context.ok():
            self.rclpy.shutdown(context=self._context)
        self._closed = True

    def _request(
        self,
        payload: dict[str, Any],
        *,
        timeout: float,
        wait: bool,
    ) -> dict[str, Any] | None:
        self._seq += 1
        request = {
            "request_id": f"{self.command_source}:{self._seq}",
            "client_stamp_ns": time.time_ns(),
            **payload,
        }
        validate_control_payload(request)

        self.request_id = str(request["request_id"])
        self.result = None
        msg = self.String()
        msg.data = json_dumps(request)

        if not wait:
            self.pub.publish(msg)
            return None

        self.wait_for_owner(timeout)
        deadline = time.monotonic() + max(0.1, float(timeout))
        next_publish = 0.0
        while self._context_ok() and self.result is None and time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_publish:
                self.pub.publish(msg)
                next_publish = now + self.retry_interval_s
            self.spin_once(0.05)
        if self.result is None:
            raise TimeoutError(f"timed out waiting for robot control result: {self.request_id}")
        return self.result

    def _result_cb(self, msg: Any) -> None:
        payload = json.loads(msg.data)
        if str(payload.get("request_id", "")) == self.request_id:
            self.result = payload

    def _context_ok(self) -> bool:
        if self._context is not None:
            return bool(self._context.ok())
        return bool(self.rclpy.ok())
