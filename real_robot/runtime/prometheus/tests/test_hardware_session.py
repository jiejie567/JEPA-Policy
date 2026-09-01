from __future__ import annotations

from prometheus.sessions.hardware import HardwareSession


def test_hardware_session_composes_runtime_components(tmp_path):
    session = HardwareSession(
        run_id="hardware_noop",
        runtime_dir=tmp_path / "runtime",
        robot={
            "id": "noop",
            "driver": {
                "_target_": "tests.noop.NoopRobotDriver",
                "action_space": "noop_action",
                "action_dim": 2,
            },
            "owner": {
                "_target_": "tests.noop.NoopRobotOwner",
                "robot_id": "noop",
            },
            "client": {
                "_target_": "tests.noop.NoopRobotClient",
                "action_space": "noop_action",
                "action_dim": 2,
            },
        },
        sensors={
            "camera": {
                "_target_": "tests.noop.NoopRuntimeComponent",
                "name": "camera",
            }
        },
        streams={
            "robot_state": {
                "topic": "/robot/state",
                "msg_type": "sensor_msgs.msg.JointState",
                "domain": "robot",
                "ros_domain_id": 0,
            },
        },
    )

    session.start()
    status = session.wait_ready(timeout_s=1.0)

    assert status.ready
    assert session.robot_driver.status()["connected"]
    assert session.robot_owner.status()["ready"]
    assert session.robot_client.send_action([0.0, 1.0])["accepted"]
    assert session.sensor_nodes["camera"].status()["ready"]
    assert session.streams["robot_state"].topic == "/robot/state"
    assert not session.sensor_nodes["camera"].status()["has_data"]

    session.stop()
    session.close()

    details = session.status().details
    assert not (tmp_path / "runtime" / "hardware_manifest.json").exists()
    assert details["robot_client"]["released"]
    assert details["robot_client"]["closed"]
    assert details["robot_owner"]["stopped"]
    assert details["robot_owner"]["closed"]
    assert details["sensors"]["camera"]["closed"]
