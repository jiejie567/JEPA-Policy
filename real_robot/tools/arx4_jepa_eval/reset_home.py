#!/usr/bin/env python3
"""Reset the deployed X5 bimanual robot to home, then leave it in teach mode."""

from __future__ import annotations

import json

from prometheus.nodes.robots.arx5 import Arx5Driver


def main() -> int:
    robot = Arx5Driver(
        robot_id="arx5",
        left_can="can3",
        right_can="can1",
        hz=120.0,
        gripper_width_m=0.082,
        gripper_open_readout=-3.4,
        left_gripper_open_readout=-3.2866,
        right_gripper_open_readout=-3.27058,
        action_space="abs_qpos",
        # An exception must not cause a second reset attempt during cleanup.
        reset_home_on_close=False,
    )
    try:
        robot.connect()
        print(
            "[reset_home] before=" + json.dumps(robot.read_state().qpos.tolist()),
            flush=True,
        )
        robot.reset_home()
        print(
            "[reset_home] after=" + json.dumps(robot.read_state().qpos.tolist()),
            flush=True,
        )
    finally:
        robot.close()
        print("[reset_home] closed_in_teach=true", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
