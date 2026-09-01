from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from prometheus.policy.scheduler import ActionChunk
from prometheus.utils.remote import DEFAULT_MAX_SIZE, RemoteInferenceClient

DEFAULT_COLOR_STREAM_NAME = "d405_color"
DEFAULT_COLOR_INFO_STREAM_NAME = "d405_color_info"
DEFAULT_INFRA1_STREAM_NAME = "d405_infra1"
DEFAULT_INFRA2_STREAM_NAME = "d405_infra2"
DEFAULT_INFRA1_INFO_STREAM_NAME = "d405_infra1_info"
DEFAULT_INFRA2_INFO_STREAM_NAME = "d405_infra2_info"
DEFAULT_BASE_STREAM_NAMES = (
    "robot_state",
    "left_eef",
    "right_eef",
    DEFAULT_COLOR_STREAM_NAME,
    DEFAULT_COLOR_INFO_STREAM_NAME,
)
DEFAULT_INIT_STREAM_NAMES = DEFAULT_BASE_STREAM_NAMES + (
    DEFAULT_INFRA1_STREAM_NAME,
    DEFAULT_INFRA2_STREAM_NAME,
    DEFAULT_INFRA1_INFO_STREAM_NAME,
    DEFAULT_INFRA2_INFO_STREAM_NAME,
)


class HumanEgoPolicy:
    """Client-side adapter for a remote HumanEgo policy server."""

    def __init__(
        self,
        *,
        remote: Mapping[str, Any],
        inputs: Mapping[str, Any] | None = None,
        robot: Mapping[str, Any] | None = None,
        client: Any | None = None,
    ):
        self.remote = dict(remote)
        self.inputs = dict(inputs or {})
        self.robot = dict(robot or {})
        self.anchor = str(self.inputs.get("anchor", DEFAULT_COLOR_STREAM_NAME))
        self.slop_ms = float(self.inputs.get("slop_ms", 100.0))
        self.base_names = _names(self.inputs.get("base_names", DEFAULT_BASE_STREAM_NAMES))
        self.init_names = _names(self.inputs.get("init_names", DEFAULT_INIT_STREAM_NAMES))
        self.camera_prefix = _camera_prefix_from_color_name(self.anchor)
        self.color_name = self.anchor
        self.color_info_name = f"{self.camera_prefix}_color_info"
        self.infra1_name = f"{self.camera_prefix}_infra1"
        self.infra2_name = f"{self.camera_prefix}_infra2"
        self.infra1_info_name = f"{self.camera_prefix}_infra1_info"
        self.infra2_info_name = f"{self.camera_prefix}_infra2_info"
        self.gripper_width_m = float(self.robot.get("gripper_width_m", 0.082))
        self.gripper_names = dict(self.robot.get("gripper_names", {"left": "left_gripper", "right": "right_gripper"}))
        self.first_infer = True
        self.client = client or RemoteInferenceClient(
            str(self.remote["server"]),
            max_size=int(self.remote.get("max_size", DEFAULT_MAX_SIZE)),
            ssh_tunnel=self.remote.get("ssh_tunnel"),
        )

    def infer(self, data: Any) -> ActionChunk:
        request = {"cmd": "infer", "observation": self._observation_payload(data)}
        if self.first_infer:
            request["reset_episode"] = True
        response = self.client.request(request)
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error", "HumanEgo remote request failed")))
        self.first_infer = False
        chunk = dict(response["action_chunk"])
        return ActionChunk(
            chunk["actions"],
            action_space=str(chunk["action_space"]),
            hz=float(chunk["hz"]),
            metadata=dict(chunk.get("metadata", {})),
        )

    def close(self) -> None:
        self.client.close()

    def status(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "server": self.remote.get("server"),
            "first_infer": self.first_infer,
            "anchor": self.anchor,
            "slop_ms": self.slop_ms,
        }

    def _observation_payload(self, data: Any) -> dict[str, Any]:
        names = self.init_names if self.first_infer else self.base_names
        frame = data.frame(anchor=self.anchor, names=names, slop_ms=self.slop_ms)
        samples = frame.samples
        robot_state = samples["robot_state"].msg
        payload = {
            "rgb_bgr": image_to_bgr(samples[self.color_name].msg),
            "K": camera_info_K(samples[self.color_info_name].msg),
            "eef_in_base": {
                "left": pose_stamped_to_T(samples["left_eef"].msg),
                "right": pose_stamped_to_T(samples["right_eef"].msg),
            },
            "gripper_widths": {
                "left": gripper_opening_m(robot_state, self.gripper_names["left"], self.gripper_width_m),
                "right": gripper_opening_m(robot_state, self.gripper_names["right"], self.gripper_width_m),
            },
            "gripper_width_m": self.gripper_width_m,
            "stamp_ns": int(samples[self.color_name].stamp_ns),
        }
        if self.first_infer:
            payload["stereo"] = {
                "left": image_to_mono8(samples[self.infra1_name].msg),
                "right": image_to_mono8(samples[self.infra2_name].msg),
                "K": camera_info_K(samples[self.infra1_info_name].msg),
                "baseline_m": camera_info_baseline_m(samples[self.infra2_info_name].msg),
                "stamp_ns": int(samples[self.infra1_name].stamp_ns),
            }
        return payload


def gripper_opening_m(msg: Any, joint_name: str, gripper_width_m: float) -> float:
    names = list(getattr(msg, "name", []))
    positions = list(getattr(msg, "position", []))
    if joint_name not in names:
        raise ValueError(f"robot_state does not contain gripper joint {joint_name!r}")
    return float(np.clip(positions[names.index(joint_name)], 0.0, gripper_width_m))


def image_to_bgr(msg: Any) -> np.ndarray:
    image = image_to_array(msg)
    encoding = str(msg.encoding).lower()
    if encoding == "rgb8":
        return image[:, :, ::-1].copy()
    if encoding == "rgba8":
        return image[:, :, 2::-1].copy()
    if encoding == "bgra8":
        return image[:, :, :3].copy()
    if encoding == "bgr8":
        return image
    raise ValueError(f"unsupported RGB image encoding {msg.encoding!r}")


def image_to_mono8(msg: Any) -> np.ndarray:
    image = image_to_array(msg)
    encoding = str(msg.encoding).lower()
    if encoding not in {"mono8", "8uc1"}:
        raise ValueError(f"unsupported mono image encoding {msg.encoding!r}")
    if image.dtype != np.uint8:
        raise ValueError(f"mono image must be uint8, got {image.dtype}")
    return image


def image_to_array(msg: Any) -> np.ndarray:
    dtype, channels = _image_layout(str(msg.encoding).lower())
    height, width = int(msg.height), int(msg.width)
    row_values = int(msg.step) // np.dtype(dtype).itemsize
    data = np.frombuffer(bytes(msg.data), dtype=dtype).reshape(height, row_values)
    if channels == 1:
        return data[:, :width].copy()
    return data[:, : width * channels].reshape(height, width, channels).copy()


def camera_info_K(msg: Any) -> np.ndarray:
    values = getattr(msg, "k", None)
    if values is None:
        values = getattr(msg, "K")
    return np.asarray(values, dtype=np.float64).reshape(3, 3)


def camera_info_baseline_m(msg: Any) -> float:
    values = getattr(msg, "p", None)
    if values is None:
        values = getattr(msg, "P")
    P = np.asarray(values, dtype=np.float64).reshape(3, 4)
    fx, tx = float(P[0, 0]), float(P[0, 3])
    if fx == 0.0 or tx == 0.0:
        raise ValueError(f"CameraInfo.P does not encode stereo baseline: fx={fx}, tx={tx}")
    return abs(tx / fx)


def pose_stamped_to_T(msg: Any) -> np.ndarray:
    pose = msg.pose
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = [float(pose.position.x), float(pose.position.y), float(pose.position.z)]
    T[:3, :3] = quat_wxyz_to_matrix(
        [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z]
    )
    if not np.all(np.isfinite(T)):
        raise ValueError("PoseStamped contains NaN or inf")
    return T


def quat_wxyz_to_matrix(q: Any) -> np.ndarray:
    w, x, y, z = _normalize_quat(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


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


def _normalize_quat(q: Any) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm <= 0.0:
        raise ValueError("zero quaternion")
    return q / norm


def _names(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        raise TypeError("input stream names must be a sequence, not a string")
    return tuple(str(item) for item in value)


def _camera_prefix_from_color_name(name: str) -> str:
    suffix = "_color"
    if not name.endswith(suffix):
        raise ValueError(f"HumanEgo inputs.anchor must be an official RealSense color name, got {name!r}")
    return name[: -len(suffix)]
