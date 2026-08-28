from __future__ import annotations

import numpy as np
import pytest

from prometheus.nodes.robots.arx5 import (
    Arx5Driver,
    Arx5RobotState,
    command_kind_for_action_space,
    controller_type_for_action_space,
    eef_action_from_state,
)
from prometheus.utils.robot import rpy_to_quaternion_wxyz


def test_arx5_driver_contract_without_sdk_import():
    driver = Arx5Driver(
        robot_id="x5_test",
        action_space="abs_qpos",
        hz=120,
        left_gripper_open_readout=-3.4,
        right_gripper_open_readout=-3.4,
    )

    assert driver.joint_names == [
        "left_joint_1",
        "left_joint_2",
        "left_joint_3",
        "left_joint_4",
        "left_joint_5",
        "left_joint_6",
        "left_gripper",
        "right_joint_1",
        "right_joint_2",
        "right_joint_3",
        "right_joint_4",
        "right_joint_5",
        "right_joint_6",
        "right_gripper",
    ]
    assert driver.status()["connected"] is False
    assert driver.status()["preview_time_s"] == pytest.approx(0.08)


def test_arx5_gripper_open_readout_is_only_pair_fallback():
    driver = Arx5Driver(robot_id="x5_test", gripper_open_readout=-3.4)

    assert driver.config.left_gripper_open_readout == pytest.approx(-3.4)
    assert driver.config.right_gripper_open_readout == pytest.approx(-3.4)

    driver = Arx5Driver(
        robot_id="x5_test",
        gripper_open_readout=-3.4,
        left_gripper_open_readout=-3.46,
        right_gripper_open_readout=-3.47,
    )

    assert driver.config.left_gripper_open_readout == pytest.approx(-3.46)
    assert driver.config.right_gripper_open_readout == pytest.approx(-3.47)

    with pytest.raises(ValueError, match="provided together"):
        Arx5Driver(
            robot_id="x5_test",
            gripper_open_readout=-3.4,
            left_gripper_open_readout=-3.46,
        )


def test_arx5_action_space_maps_to_controller_kind():
    assert controller_type_for_action_space("abs_qpos") == "joint_controller"
    assert controller_type_for_action_space("delta_qpos") == "joint_controller"
    assert controller_type_for_action_space("abs_eef") == "cartesian_controller"
    assert command_kind_for_action_space("delta_eef") == "eef"

    with pytest.raises(ValueError, match="unknown action_space"):
        controller_type_for_action_space("bad")


def test_arx5_eef_pose_6d_prefers_controller_eef_state():
    driver = Arx5Driver(
        robot_id="x5_test",
        action_space="abs_eef",
        left_gripper_open_readout=-3.4,
        right_gripper_open_readout=-3.4,
    )

    class FakeEEFState:
        def pose_6d(self) -> np.ndarray:
            return np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64)

    class FakeController:
        def get_eef_state(self) -> FakeEEFState:
            return FakeEEFState()

    pose = driver._eef_pose_6d("left", FakeController(), np.zeros(6, dtype=np.float64))

    assert pose.shape == (6,)
    assert np.allclose(pose, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])


def test_arx5_eef_pose_6d_falls_back_to_solver():
    driver = Arx5Driver(
        robot_id="x5_test",
        action_space="abs_eef",
        left_gripper_open_readout=-3.4,
        right_gripper_open_readout=-3.4,
    )

    class BrokenController:
        def get_eef_state(self):
            raise RuntimeError("eef state unavailable")

    class FakeSolver:
        def forward_kinematics(self, joint_pos: np.ndarray) -> np.ndarray:
            assert joint_pos.shape == (6,)
            return np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], dtype=np.float64)

    driver._fk_solvers["left"] = FakeSolver()
    pose = driver._eef_pose_6d("left", BrokenController(), np.arange(6, dtype=np.float64))

    assert np.allclose(pose, [1.0, 2.0, 3.0, 0.1, 0.2, 0.3])

    state = Arx5RobotState(
        joint_names=[],
        qpos=np.array([0, 0, 0, 0, 0, 0, 0.01, 0, 0, 0, 0, 0, 0, 0.02], dtype=np.float32),
        qvel=None,
        effort=None,
        eef_pose_6d=np.arange(12, dtype=np.float32),
        eef_wxyz=np.zeros(14, dtype=np.float32),
    )

    action = eef_action_from_state(state)

    assert action.shape == (14,)
    assert action[6] == pytest.approx(0.01)
    assert action[13] == pytest.approx(0.02)


def test_arx5_quaternion_is_normalized():
    quat = rpy_to_quaternion_wxyz(0.1, -0.2, 0.3)

    assert quat.shape == (4,)
    assert np.linalg.norm(quat) == pytest.approx(1.0)


def test_arx5_gripper_command_applies_shared_close_bias():
    driver = Arx5Driver(robot_id="x5_test", action_space="abs_eef")

    assert driver._gripper_command(0.02) == pytest.approx(0.015)
    assert driver._gripper_command(0.005) == pytest.approx(0.0)
    assert driver._gripper_command(0.0) == pytest.approx(0.0)
    assert driver._clip_gripper_command(0.082) == pytest.approx(0.08)


def test_arx5_close_can_skip_reset_without_changing_default():
    class TrackingDriver(Arx5Driver):
        def __init__(self, *, reset_home_on_close: bool):
            super().__init__(reset_home_on_close=reset_home_on_close)
            self._closed = False
            self._controllers = {"left": object(), "right": object()}
            self.reset_count = 0
            self.teach_count = 0

        def reset_to_home(self) -> None:
            self.reset_count += 1

        def set_teach_mode(self) -> None:
            self.teach_count += 1

    safe = TrackingDriver(reset_home_on_close=False)
    safe.close()
    assert safe.reset_count == 0
    assert safe.teach_count == 1

    original = TrackingDriver(reset_home_on_close=True)
    original.close()
    assert original.reset_count == 1
    assert original.teach_count == 1


def test_arx5_reset_home_holds_until_ordered_cleanup(monkeypatch):
    sleeps = []
    monkeypatch.setattr("prometheus.nodes.robots.arx5.time.sleep", sleeps.append)

    class TrackingDriver(Arx5Driver):
        def __init__(self):
            super().__init__(reset_home_on_close=False)
            self._closed = False
            self._controllers = {"left": object(), "right": object()}
            self.reset_count = 0
            self.teach_count = 0

        def reset_to_home(self) -> None:
            self.reset_count += 1

        def set_teach_mode(self) -> None:
            self.teach_count += 1

    driver = TrackingDriver()
    driver.reset_home()

    assert driver.reset_count == 1
    assert driver.teach_count == 0
    assert driver.control_mode == "position"
    assert sleeps == [pytest.approx(0.6)]

    driver.close()
    assert driver.teach_count == 1
