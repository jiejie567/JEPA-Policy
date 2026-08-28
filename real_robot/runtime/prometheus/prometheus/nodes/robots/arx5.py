from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import wraps
import time
from typing import Any, Callable, Mapping

import numpy as np

from prometheus.nodes.robots.robot_node import RobotState
from prometheus.utils.robot import rpy_to_quaternion_wxyz


ARX5_ACTION_SPACES = ("abs_qpos", "delta_qpos", "abs_eef", "delta_eef")
QPOS_ACTION_SPACES = ("abs_qpos", "delta_qpos")
EEF_ACTION_SPACES = ("abs_eef", "delta_eef")
ACTION_DIMS = {action_space: 14 for action_space in ARX5_ACTION_SPACES}
BACKGROUND_SEND_RECV = True
GRAVITY_COMPENSATION = True
TEACH_KD_SCALE = 0.1
POSITION_KP_SCALE = 1.0
POSITION_KD_SCALE = 1.0
GRIPPER_KP_SCALE = 1.5
GRIPPER_KD_SCALE = 1.5
GRIPPER_COMMAND_MARGIN_M = 0.002
# Calibration used for all reported real-robot evaluations. The model-space
# opening is shifted 5 mm toward closing before the physical-range clip.
GRIPPER_COMMAND_BIAS_M = -0.005
GRIPPER_WIDTH_M = 0.082
DEFAULT_PREVIEW_TIME_S = 0.08
RESET_HOME_FINAL_WAYPOINT_S = 0.6
LOG_LEVEL = "INFO"
MODEL = "X5"
ROBOT_TOPIC_SUFFIXES = {
    "state": "state",
    "left_eef": "left_eef_pose",
    "right_eef": "right_eef_pose",
    "control": "control",
    "result": "control_result",
    "status": "status",
}
SIDE_VECTOR_START = {"left": 0, "right": 7}
SIDE_EEF6_START = {"left": 0, "right": 6}


@dataclass(frozen=True, kw_only=True)
class Arx5RobotState(RobotState):
    eef_pose_6d: np.ndarray
    eef_wxyz: np.ndarray


@dataclass
class Arx5Config:
    left_can: str = "can3"
    right_can: str = "can1"
    hz: float = 120.0
    gripper_width_m: float = GRIPPER_WIDTH_M
    gripper_open_readout: float = -3.4
    left_gripper_open_readout: float = -3.4
    right_gripper_open_readout: float = -3.4
    action_space: str = "abs_qpos"
    reset_home_on_close: bool = True

    @property
    def preview_time_s(self) -> float:
        return DEFAULT_PREVIEW_TIME_S


def requires_connected(method: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not self.is_connected:
            raise RuntimeError("ARX5 controller is not connected")
        return method(self, *args, **kwargs)

    return wrapper


def _resolve_gripper_open_readouts(
    gripper_open_readout: float,
    left_gripper_open_readout: float | None,
    right_gripper_open_readout: float | None,
) -> tuple[float, float]:
    if left_gripper_open_readout is None and right_gripper_open_readout is None:
        fallback = float(gripper_open_readout)
        return fallback, fallback
    if left_gripper_open_readout is None or right_gripper_open_readout is None:
        raise ValueError(
            "left_gripper_open_readout and right_gripper_open_readout must be provided together; "
            "gripper_open_readout is only used when both side-specific values are absent"
        )
    return float(left_gripper_open_readout), float(right_gripper_open_readout)


class Arx5Driver:
    """ARX5 bimanual hardware driver with no workflow ownership."""

    sides = ("left", "right")
    arm_dof = 6
    action_dim = 14

    def __init__(
        self,
        *,
        robot_id: str = "arx5",
        left_can: str = "can3",
        right_can: str = "can1",
        hz: float = 120.0,
        gripper_width_m: float = GRIPPER_WIDTH_M,
        gripper_open_readout: float = -3.4,
        left_gripper_open_readout: float | None = None,
        right_gripper_open_readout: float | None = None,
        action_space: str = "abs_qpos",
        reset_home_on_close: bool = True,
        eef_topics: Mapping[str, str] | None = None,
    ):
        if action_space not in ARX5_ACTION_SPACES:
            raise ValueError(f"ARX5 action_space must be one of {ARX5_ACTION_SPACES}")
        if float(hz) <= 0:
            raise ValueError("ARX5 hz must be positive")
        left_open_readout, right_open_readout = _resolve_gripper_open_readouts(
            gripper_open_readout,
            left_gripper_open_readout,
            right_gripper_open_readout,
        )
        self.robot_id = str(robot_id)
        self.config = Arx5Config(
            left_can=str(left_can),
            right_can=str(right_can),
            hz=float(hz),
            gripper_width_m=float(gripper_width_m),
            gripper_open_readout=float(gripper_open_readout),
            left_gripper_open_readout=left_open_readout,
            right_gripper_open_readout=right_open_readout,
            action_space=str(action_space),
            reset_home_on_close=bool(reset_home_on_close),
        )
        self.eef_topics = dict(eef_topics or {})
        self.PoseStamped: Any | None = None
        self.eef_pubs: dict[str, Any] = {}
        self._arx5: Any | None = None
        self._controllers: dict[str, Any] = {}
        self._robot_configs: dict[str, Any] = {}
        self._controller_configs: dict[str, Any] = {}
        self._fk_solvers: dict[str, Any] = {}
        self._side_executor: ThreadPoolExecutor | None = None
        self._closed = True
        self.control_mode = ""

    @property
    def joint_names(self) -> list[str]:
        names = []
        for side in self.sides:
            names.extend([f"{side}_joint_{idx + 1}" for idx in range(self.arm_dof)])
            names.append(f"{side}_gripper")
        return names

    @property
    def is_connected(self) -> bool:
        return len(self._controllers) == 2 and not self._closed

    @property
    def controller_type(self) -> str:
        return controller_type_for_action_space(self.config.action_space)

    @property
    def command_kind(self) -> str:
        return command_kind_for_action_space(self.config.action_space)

    def connect(self) -> None:
        if self.is_connected:
            return
        self._arx5 = self._import_arx5()
        self._side_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix=f"{self.robot_id}_arx5",
        )
        self._closed = False
        try:
            for side in self.sides:
                robot_cfg = self._arx5.RobotConfigFactory.get_instance().get_config(MODEL)
                robot_cfg.gripper_open_readout = (
                    self.config.left_gripper_open_readout
                    if side == "left"
                    else self.config.right_gripper_open_readout
                )
                robot_cfg.gripper_width = float(self.config.gripper_width_m)
                can_name = self.config.left_can if side == "left" else self.config.right_can

                ctrl_cfg = self._arx5.ControllerConfigFactory.get_instance().get_config(
                    self.controller_type,
                    robot_cfg.joint_dof,
                )
                ctrl_cfg.background_send_recv = BACKGROUND_SEND_RECV
                ctrl_cfg.gravity_compensation = GRAVITY_COMPENSATION
                ctrl_cfg.default_preview_time = float(self.config.preview_time_s)

                if self.controller_type == "joint_controller":
                    controller = self._arx5.Arx5JointController(
                        robot_cfg,
                        ctrl_cfg,
                        can_name,
                    )
                elif self.controller_type == "cartesian_controller":
                    controller = self._arx5.Arx5CartesianController(
                        robot_cfg,
                        ctrl_cfg,
                        can_name,
                    )
                else:
                    raise ValueError(f"unsupported ARX5 controller type {self.controller_type!r}")
                controller.set_log_level(getattr(self._arx5.LogLevel, LOG_LEVEL))
                self._controllers[side] = controller
                self._robot_configs[side] = robot_cfg
                self._controller_configs[side] = ctrl_cfg
                self._fk_solvers[side] = self._arx5.Arx5Solver(
                    robot_cfg.urdf_path,
                    robot_cfg.joint_dof,
                    robot_cfg.joint_pos_min,
                    robot_cfg.joint_pos_max,
                    robot_cfg.base_link_name,
                    robot_cfg.eef_link_name,
                    robot_cfg.gravity_vector,
                )
            self.set_teach_mode()
        except Exception:
            self.close()
            raise

    @requires_connected
    def read_state(self) -> Arx5RobotState:
        def read_side_state(side: str) -> dict[str, np.ndarray]:
            controller = self._controllers[side]
            state = controller.get_joint_state()
            arm_pos64 = np.asarray(state.pos(), dtype=np.float64).copy()
            arm_pos = arm_pos64.astype(np.float32)
            arm_vel = np.asarray(state.vel(), dtype=np.float32).copy()
            arm_effort = np.asarray(state.torque(), dtype=np.float32).copy()
            gripper_opening = float(np.clip(float(state.gripper_pos), 0.0, self.config.gripper_width_m))
            eef_pose_6d = self._eef_pose_6d(side, controller, arm_pos64)
            if np.all(np.isfinite(eef_pose_6d)):
                eef_wxyz = np.concatenate(
                    [eef_pose_6d[:3], rpy_to_quaternion_wxyz(eef_pose_6d[3], eef_pose_6d[4], eef_pose_6d[5])]
                ).astype(np.float32)
            else:
                eef_wxyz = np.full(7, np.nan, dtype=np.float32)
            return {
                "qpos": np.concatenate([arm_pos, [gripper_opening]]).astype(np.float32),
                "qvel": np.concatenate(
                    [arm_vel, [_optional_float(getattr(state, "gripper_vel", None))]]
                ).astype(np.float32),
                "effort": np.concatenate(
                    [arm_effort, [_optional_float(getattr(state, "gripper_torque", None))]]
                ).astype(np.float32),
                "eef_pose_6d": eef_pose_6d,
                "eef_wxyz": eef_wxyz,
            }

        side_states = self._run_sides(read_side_state)
        return Arx5RobotState(
            joint_names=self.joint_names,
            qpos=np.concatenate([side_states[side]["qpos"] for side in self.sides]).astype(np.float32),
            qvel=np.concatenate([side_states[side]["qvel"] for side in self.sides]).astype(np.float32),
            effort=np.concatenate([side_states[side]["effort"] for side in self.sides]).astype(np.float32),
            eef_pose_6d=np.concatenate(
                [side_states[side]["eef_pose_6d"] for side in self.sides]
            ).astype(np.float32),
            eef_wxyz=np.concatenate([side_states[side]["eef_wxyz"] for side in self.sides]).astype(np.float32),
        )

    @requires_connected
    def send_action(self, action_space: str, action: np.ndarray) -> None:
        if self.control_mode != "position":
            self.set_position_mode()
        action_space = str(action_space)
        if action_space == "abs_qpos":
            self.send_joint_positions(action)
            return
        if action_space == "delta_qpos":
            current = np.asarray(self.read_state().qpos, dtype=np.float32).reshape(14)
            self.send_joint_positions(current + np.asarray(action, dtype=np.float32).reshape(14))
            return
        if action_space == "abs_eef":
            self.send_eef_positions(action)
            return
        if action_space == "delta_eef":
            current = eef_action_from_state(self.read_state())
            self.send_eef_positions(current + np.asarray(action, dtype=np.float32).reshape(14))
            return
        raise ValueError(f"unknown ARX5 action_space {action_space!r}; expected {ARX5_ACTION_SPACES}")

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self.is_connected:
                try:
                    if self.config.reset_home_on_close:
                        self.reset_to_home()
                finally:
                    self.set_teach_mode()
        finally:
            self._controllers.clear()
            self._robot_configs.clear()
            self._controller_configs.clear()
            self._fk_solvers.clear()
            if self._side_executor is not None:
                self._side_executor.shutdown(wait=True)
                self._side_executor = None
            self._closed = True

    def status(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "robot_id": self.robot_id,
            "connected": self.is_connected,
            "action_space": self.config.action_space,
            "command_kind": self.command_kind,
            "controller_type": self.controller_type,
            "control_mode": self.control_mode,
            "hz": self.config.hz,
            "preview_time_s": self.config.preview_time_s,
            "left_can": self.config.left_can,
            "right_can": self.config.right_can,
            "reset_home_on_close": self.config.reset_home_on_close,
            "eef_topics": dict(self.eef_topics),
        }

    @requires_connected
    def set_teach_mode(self) -> None:
        def set_side_teach_mode(side: str) -> None:
            controller = self._controllers[side]
            controller.set_to_damping()
            gain = controller.get_gain()
            gain.kd()[:] *= TEACH_KD_SCALE
            gain.gripper_kd *= TEACH_KD_SCALE
            controller.set_gain(gain)

        self._run_sides(set_side_teach_mode)
        self.control_mode = "teach"

    @requires_connected
    def set_position_mode(self) -> None:
        def set_side_position_mode(side: str) -> None:
            controller = self._controllers[side]
            state = controller.get_joint_state()
            if self.command_kind == "qpos":
                cmd = self._arx5.JointState(self._robot_configs[side].joint_dof)
                cmd.pos()[:] = np.asarray(state.pos(), dtype=np.float64)
                cmd.gripper_pos = self._clip_gripper_command(float(state.gripper_pos))
                controller.set_joint_cmd(cmd)
            elif self.command_kind == "eef":
                pose_6d = self._eef_pose_6d(
                    side,
                    controller,
                    np.asarray(state.pos(), dtype=np.float64),
                )
                check_finite("current eef pose", pose_6d)
                cmd = self._arx5.EEFState(
                    np.asarray(pose_6d, dtype=np.float64),
                    self._clip_gripper_command(float(state.gripper_pos)),
                )
                controller.set_eef_cmd(cmd)

            ctrl_cfg = self._controller_configs[side]
            gain = self._arx5.Gain(self._robot_configs[side].joint_dof)
            gain.kp()[:] = np.asarray(ctrl_cfg.default_kp, dtype=np.float64) * POSITION_KP_SCALE
            gain.kd()[:] = np.asarray(ctrl_cfg.default_kd, dtype=np.float64) * POSITION_KD_SCALE
            gain.gripper_kp = float(ctrl_cfg.default_gripper_kp) * GRIPPER_KP_SCALE
            gain.gripper_kd = float(ctrl_cfg.default_gripper_kd) * GRIPPER_KD_SCALE
            controller.set_gain(gain)

        self._run_sides(set_side_position_mode)
        self.control_mode = "position"

    @requires_connected
    def reset_home(self) -> None:
        self.reset_to_home()
        # The SDK returns immediately after scheduling its final 0.5-second
        # home waypoint.  Do not acknowledge the reset until that waypoint has
        # had time to complete and the controller is holding the target.
        time.sleep(RESET_HOME_FINAL_WAYPOINT_S)
        self.control_mode = "position"
        # Keep the controller holding home until the owning workflow verifies
        # the measured pose.  close() performs the ordered transition to
        # teach/damping after verification or rollout cleanup.

    @requires_connected
    def reset_to_home(self) -> None:
        self._run_sides(lambda side: self._controllers[side].reset_to_home())

    @requires_connected
    def send_joint_positions(self, qpos: np.ndarray) -> np.ndarray:
        if self.command_kind != "qpos":
            raise RuntimeError(
                f"ARX5 action_space={self.config.action_space!r} uses "
                f"{self.command_kind!r} commands, not 'qpos'"
            )
        qpos = np.asarray(qpos, dtype=np.float32).reshape(14)
        check_finite("qpos command", qpos)

        def send_side_joint_position(side: str) -> None:
            target = _side_vector(qpos, side)
            cmd = self._arx5.JointState(self.arm_dof)
            cmd.pos()[:] = target[:6]
            cmd.gripper_pos = self._gripper_command(float(target[6]))
            self._controllers[side].set_joint_cmd(cmd)

        self._run_sides(send_side_joint_position)
        return qpos

    @requires_connected
    def send_eef_positions(self, eef_action: np.ndarray) -> np.ndarray:
        if self.command_kind != "eef":
            raise RuntimeError(
                f"ARX5 action_space={self.config.action_space!r} uses "
                f"{self.command_kind!r} commands, not 'eef'"
            )
        command = np.asarray(eef_action, dtype=np.float32).reshape(14).copy()
        check_finite("eef command", command)

        def send_side_eef_position(side: str) -> None:
            target = _side_vector(command, side)
            target[6] = self._gripper_command(float(target[6]))
            cmd = self._arx5.EEFState(np.asarray(target[:6], dtype=np.float64), float(target[6]))
            self._controllers[side].set_eef_cmd(cmd)

        self._run_sides(send_side_eef_position)
        return command.astype(np.float32)

    def setup_ros(self, *, node: Any, qos_depth: int) -> None:
        if not self.eef_topics:
            return
        from geometry_msgs.msg import PoseStamped

        missing = {"left", "right"}.difference(self.eef_topics)
        if missing:
            raise ValueError(f"ARX5 eef_topics missing sides: {sorted(missing)}")
        self.PoseStamped = PoseStamped
        self.eef_pubs = {
            side: node.create_publisher(PoseStamped, topic, int(qos_depth))
            for side, topic in self.eef_topics.items()
        }

    def publish_ros_state(self, *, node: Any, stamp: Any, state: RobotState) -> None:
        if not self.eef_pubs:
            return
        eef = np.asarray(getattr(state, "eef_wxyz"), dtype=np.float32).reshape(14)
        for side in self.sides:
            if self.PoseStamped is None:
                raise RuntimeError("ARX5 EEF publishers are not initialized")
            pose = _side_vector(eef, side)
            if not np.all(np.isfinite(pose)):
                continue
            msg = self.PoseStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = f"{self.robot_id}/{side}_eef"
            msg.pose.position.x = float(pose[0])
            msg.pose.position.y = float(pose[1])
            msg.pose.position.z = float(pose[2])
            msg.pose.orientation.w = float(pose[3])
            msg.pose.orientation.x = float(pose[4])
            msg.pose.orientation.y = float(pose[5])
            msg.pose.orientation.z = float(pose[6])
            self.eef_pubs[side].publish(msg)

    def _eef_pose_6d(self, side: str, controller: Any, joint_pos: np.ndarray) -> np.ndarray:
        try:
            return np.asarray(controller.get_eef_state().pose_6d(), dtype=np.float32).reshape(6)
        except Exception:
            pass
        solver = self._fk_solvers.get(side)
        if solver is not None:
            try:
                return np.asarray(
                    solver.forward_kinematics(
                        np.asarray(joint_pos, dtype=np.float64).reshape(self.arm_dof)
                    ),
                    dtype=np.float32,
                ).reshape(6)
            except Exception:
                pass
        return np.full(6, np.nan, dtype=np.float32)

    def _clip_gripper_command(self, gripper_pos: float) -> float:
        safe_max = self.config.gripper_width_m - GRIPPER_COMMAND_MARGIN_M
        safe_max = max(0.0, min(float(self.config.gripper_width_m), safe_max))
        return float(np.clip(gripper_pos, 0.0, safe_max))

    def _gripper_command(self, gripper_pos: float) -> float:
        return self._clip_gripper_command(float(gripper_pos) + GRIPPER_COMMAND_BIAS_M)

    def _run_sides(self, fn: Callable[[str], Any]) -> dict[str, Any]:
        executor = self._side_executor
        if executor is None:
            return {side: fn(side) for side in self.sides}

        futures = {side: executor.submit(fn, side) for side in self.sides}
        results: dict[str, Any] = {}
        first_error: Exception | None = None
        for side in self.sides:
            try:
                results[side] = futures[side].result()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error
        return results

    @staticmethod
    def _import_arx5() -> Any:
        try:
            import arx5_interface as arx5
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "arx5_interface is required for ARX5 hardware. Install the ARX5 "
                "SDK into this environment and verify with "
                "`python -c \"import arx5_interface\"`."
            ) from exc
        except ImportError as exc:
            raise ImportError(
                "arx5_interface was found but one of its native dependencies "
                f"could not be loaded: {exc}"
            ) from exc
        return arx5


def controller_type_for_action_space(action_space: str) -> str:
    if action_space in QPOS_ACTION_SPACES:
        return "joint_controller"
    if action_space in EEF_ACTION_SPACES:
        return "cartesian_controller"
    raise ValueError(f"unknown action_space {action_space!r}; expected {ARX5_ACTION_SPACES}")


def command_kind_for_action_space(action_space: str) -> str:
    if action_space in QPOS_ACTION_SPACES:
        return "qpos"
    if action_space in EEF_ACTION_SPACES:
        return "eef"
    raise ValueError(f"unknown action_space {action_space!r}; expected {ARX5_ACTION_SPACES}")


def arx5_topics(robot_id: str) -> dict[str, str]:
    prefix = f"/prometheus/robots/{robot_id}"
    return {
        name: f"{prefix}/{suffix}"
        for name, suffix in ROBOT_TOPIC_SUFFIXES.items()
    }


def resolve_robot(**robot: Any) -> dict[str, Any]:
    config = dict(robot)
    config["type"] = "arx5"
    config.update(runtime_config(config))
    config["sensors"] = sensor_config(config)
    config["streams"] = streams(config)
    return config


def runtime_config(robot: Mapping[str, Any]) -> dict[str, Any]:
    robot_id = str(robot["id"])
    robot_cfg = _robot_config(robot)
    can = _side_mapping(robot_cfg, "can")
    topics = arx5_topics(robot_id)
    action_space = str(robot_cfg["action_space"])
    state_publish_hz = float(robot_cfg["state_publish_hz"])
    gripper_width_m = float(robot_cfg.get("gripper_width_m", GRIPPER_WIDTH_M))
    gripper_open_readout = float(robot_cfg.get("gripper_open_readout", -3.4))
    control_domain_id = int(robot["control_domain_id"])
    return {
        "driver": {
            "_target_": "prometheus.nodes.robots.arx5.Arx5Driver",
            "robot_id": robot_id,
            "left_can": can["left"],
            "right_can": can["right"],
            "hz": state_publish_hz,
            "gripper_width_m": gripper_width_m,
            "gripper_open_readout": gripper_open_readout,
            "left_gripper_open_readout": robot_cfg.get("left_gripper_open_readout", gripper_open_readout),
            "right_gripper_open_readout": robot_cfg.get("right_gripper_open_readout", gripper_open_readout),
            "action_space": action_space,
            "reset_home_on_close": bool(robot_cfg.get("reset_home_on_close", True)),
            "eef_topics": {
                "left": topics["left_eef"],
                "right": topics["right_eef"],
            },
        },
        "owner": {
            "_target_": "prometheus.nodes.robots.owner.RobotOwnerNode",
            "robot_id": robot_id,
            "topics": {
                "state": topics["state"],
                "control": topics["control"],
                "result": topics["result"],
                "status": topics["status"],
            },
            "hz": state_publish_hz,
            "ros_domain_id": control_domain_id,
            "action_dims": ACTION_DIMS,
        },
        "client": {
            "_target_": "prometheus.nodes.robots.client.RobotControlClient",
            "action_space": action_space,
            "action_dim": 14,
            "control_topic": topics["control"],
            "result_topic": topics["result"],
            "command_source": f"{robot_id}_workflow",
            "ros_domain_id": control_domain_id,
        },
    }


def _robot_config(robot: Mapping[str, Any]) -> dict[str, Any]:
    value = robot.get("robot")
    if not isinstance(value, Mapping):
        raise ValueError("ARX5 config requires a robot mapping")
    config = dict(value)
    missing = {"state_publish_hz", "action_space", "can"}.difference(config)
    if missing:
        raise ValueError(f"ARX5 robot config missing required keys: {sorted(missing)}")
    return config


def sensor_config(robot: Mapping[str, Any]) -> dict[str, Any]:
    from prometheus.nodes.sensors.realsense import runtime_config as build_sensor_runtime
    from prometheus.nodes.sensors.xense import runtime_config as build_tactile_runtime

    sensors = build_sensor_runtime(robot)
    sensors.update(build_tactile_runtime(robot))
    return sensors


def streams(robot: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    from prometheus.nodes.sensors.realsense import streams as build_sensor_streams
    from prometheus.nodes.sensors.xense import streams as build_tactile_streams

    items = robot_streams(robot)
    items.update(build_sensor_streams(robot))
    items.update(build_tactile_streams(robot))
    return items


def robot_streams(robot: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    control_domain_id = int(robot["control_domain_id"])
    topics = arx5_topics(str(robot["id"]))
    return {
        "robot_state": {
            "topic": topics["state"],
            "msg_type": "sensor_msgs.msg.JointState",
            "domain": "robot",
            "ros_domain_id": control_domain_id,
        },
        "left_eef": {
            "topic": topics["left_eef"],
            "msg_type": "geometry_msgs.msg.PoseStamped",
            "domain": "robot",
            "ros_domain_id": control_domain_id,
        },
        "right_eef": {
            "topic": topics["right_eef"],
            "msg_type": "geometry_msgs.msg.PoseStamped",
            "domain": "robot",
            "ros_domain_id": control_domain_id,
        },
    }


def eef_action_from_state(state: Arx5RobotState) -> np.ndarray:
    parts = []
    for side in Arx5Driver.sides:
        eef_start = SIDE_EEF6_START[side]
        parts.append(
            np.concatenate(
                [
                    state.eef_pose_6d[eef_start : eef_start + 6],
                    [state.qpos[SIDE_VECTOR_START[side] + 6]],
                ]
            ).astype(np.float32)
        )
    return np.concatenate(parts).astype(np.float32)


def _side_vector(values: np.ndarray, side: str) -> np.ndarray:
    start = SIDE_VECTOR_START[side]
    return values[start : start + 7]


def check_finite(name: str, values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains NaN or inf values")


def _optional_float(value: Any) -> float:
    return np.nan if value is None else float(value)


def _side_mapping(robot: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = robot.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"ARX5 robot config requires {name}.left and {name}.right")
    missing = {"left", "right"}.difference(value)
    if missing:
        raise ValueError(f"ARX5 robot config {name} missing sides: {sorted(missing)}")
    return {"left": value["left"], "right": value["right"]}
