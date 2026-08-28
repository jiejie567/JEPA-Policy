from __future__ import annotations

import numpy as np
import pytest

import prometheus.policy.scheduler as scheduler_module
from prometheus.policy.scheduler import (
    ActionChunk,
    ActionScheduler,
    BoundsGuard,
    GripperCloseHeightGateFilter,
    GripperCloseLeadFilter,
    HoldDimensionsFilter,
    MaxDeltaFilter,
    MaxDeltaGuard,
    SafetyGuardError,
    StatefulDynamicsFilter,
)


class RejectingClient:
    action_space = "abs_qpos"

    def send_action(self, *_args, **_kwargs):
        raise AssertionError("dry-run must not call send_action")

    def spin_once(self, _timeout_s=0.0):
        return None


class TimedClient:
    action_space = "abs_qpos"

    def __init__(self, clock: list[float]):
        self.clock = clock
        self.sent_at: list[float] = []

    def send_action(self, *_args, **_kwargs):
        self.sent_at.append(self.clock[0])
        return {"accepted": True}

    def spin_once(self, timeout_s=0.0):
        self.clock[0] += float(timeout_s)


def _chunk(action: np.ndarray, current: np.ndarray) -> ActionChunk:
    return ActionChunk(
        actions=np.asarray(action, dtype=np.float32).reshape(1, -1),
        action_space="abs_qpos",
        hz=10.0,
        metadata={
            "current_action": np.asarray(current, dtype=np.float32),
            "current_velocity": np.zeros_like(current, dtype=np.float32),
        },
    )


def test_safety_guards_accept_a_finite_in_bounds_small_step():
    scheduler = ActionScheduler(
        RejectingClient(),
        dry_run=True,
        filters=[
            BoundsGuard(lower=np.zeros(2), upper=np.ones(2)),
            MaxDeltaGuard(max_delta=np.full(2, 0.2)),
        ],
    )

    assert scheduler.execute(_chunk(np.array([0.55, 0.45]), np.array([0.5, 0.5]))) == 1


def test_hold_dimensions_filter_keeps_task_inactive_arm_at_measured_qpos():
    current = np.array([0.2, -0.1, 0.5], dtype=np.float32)
    chunk = ActionChunk(
        actions=np.array([[1.0, 1.0, 0.6], [-1.0, -1.0, 0.7]], dtype=np.float32),
        action_space="abs_qpos",
        hz=10.0,
        metadata={"current_action": current},
    )
    scheduler = ActionScheduler(
        RejectingClient(),
        dry_run=True,
        filters=[HoldDimensionsFilter(dimensions=[0, 1])],
    )

    assert scheduler.execute(chunk) == 2
    np.testing.assert_allclose(
        np.asarray(scheduler.history),
        np.array([[0.2, -0.1, 0.6], [0.2, -0.1, 0.7]], dtype=np.float32),
    )


def test_scheduler_accounts_for_sent_action_before_recording_callback_failure():
    clock = [10.0]
    client = TimedClient(clock)
    scheduler = ActionScheduler(client, dry_run=False, wait=True)

    def fail_recording(_action, _context):
        raise RuntimeError("recording failed")

    with pytest.raises(RuntimeError, match="recording failed"):
        scheduler.execute(
            _chunk(np.array([0.1]), np.array([0.0])),
            on_action=fail_recording,
        )

    assert len(client.sent_at) == 1
    assert len(scheduler.history) == 1


def test_scheduler_can_stop_cooperatively_between_actions():
    clock = [10.0]
    client = TimedClient(clock)
    scheduler = ActionScheduler(client, dry_run=False, wait=True)
    chunk = ActionChunk(
        actions=np.zeros((4, 2), dtype=np.float32),
        action_space="abs_qpos",
        hz=10.0,
        metadata={"current_action": np.zeros(2, dtype=np.float32)},
    )

    assert scheduler.execute(
        chunk,
        should_stop=lambda: len(client.sent_at) >= 2,
    ) == 2
    assert len(client.sent_at) == 2


def test_bounds_guard_rejects_instead_of_clipping():
    scheduler = ActionScheduler(
        RejectingClient(),
        dry_run=True,
        filters=[BoundsGuard(lower=np.zeros(2), upper=np.ones(2))],
    )

    with pytest.raises(SafetyGuardError, match="out-of-bounds dimensions"):
        scheduler.execute(_chunk(np.array([1.1, 0.5]), np.array([0.5, 0.5])))


def test_bounds_guard_clips_only_within_explicit_tolerance():
    scheduler = ActionScheduler(
        RejectingClient(),
        dry_run=True,
        filters=[
            BoundsGuard(
                lower=np.zeros(2),
                upper=np.ones(2),
                clip_tolerance=np.full(2, 0.005),
                name="raw_bounds_guard",
            )
        ],
    )

    assert scheduler.execute(_chunk(np.array([-0.001314, 0.5]), np.array([0.5, 0.5]))) == 1
    assert scheduler.history[-1][0] == pytest.approx(0.0)

    with pytest.raises(SafetyGuardError, match="out-of-bounds dimensions"):
        scheduler.execute(_chunk(np.array([-0.006, 0.5]), np.array([0.5, 0.5])))


def test_raw_gripper_undershoot_is_clipped_before_filtered_bounds():
    chunk = ActionChunk(
        actions=np.array([[-0.0004795]], dtype=np.float32),
        action_space="abs_qpos",
        hz=10.0,
        metadata={
            "current_action": np.array([0.002], dtype=np.float32),
            "current_velocity": np.array([0.0], dtype=np.float32),
        },
    )
    scheduler = ActionScheduler(
        RejectingClient(),
        dry_run=True,
        filters=[
            BoundsGuard(lower=[-0.005], upper=[0.087], name="raw_bounds_guard"),
            StatefulDynamicsFilter(
                max_delta=[0.03],
                max_delta_change=[0.03],
                lower=[0.0],
                upper=[0.087],
            ),
            BoundsGuard(lower=[0.0], upper=[0.087], name="filtered_bounds_guard"),
        ],
    )

    assert scheduler.execute(chunk) == 1
    assert float(scheduler.history[-1][0]) == pytest.approx(0.0)


def test_max_delta_guard_uses_observed_current_action_for_first_step():
    scheduler = ActionScheduler(
        RejectingClient(),
        dry_run=True,
        filters=[MaxDeltaGuard(max_delta=np.full(2, 0.2))],
    )

    with pytest.raises(SafetyGuardError, match="abs_delta"):
        scheduler.execute(_chunk(np.array([0.8, 0.5]), np.array([0.5, 0.5])))


def test_max_delta_guard_rebases_new_chunk_to_observed_current_action():
    scheduler = ActionScheduler(
        RejectingClient(),
        dry_run=True,
        filters=[MaxDeltaGuard(max_delta=np.array([0.175, 0.199]))],
    )
    first = ActionChunk(
        actions=np.array([[0.80943125, 1.2188293]], dtype=np.float32),
        action_space="abs_qpos",
        hz=10.0,
        metadata={"current_action": np.array([0.7, 1.1], dtype=np.float32)},
    )
    second = ActionChunk(
        actions=np.array([[0.5915023, 0.9869707]], dtype=np.float32),
        action_space="abs_qpos",
        hz=10.0,
        metadata={"current_action": np.array([0.5987263, 0.9885941], dtype=np.float32)},
    )

    assert scheduler.execute(first) == 1
    assert scheduler.execute(second) == 1


def test_max_delta_guard_still_checks_adjacent_commands_inside_chunk():
    scheduler = ActionScheduler(
        RejectingClient(),
        dry_run=True,
        filters=[MaxDeltaGuard(max_delta=np.array([0.2], dtype=np.float32))],
    )
    chunk = ActionChunk(
        actions=np.array([[0.1], [0.4]], dtype=np.float32),
        action_space="abs_qpos",
        hz=10.0,
        metadata={"current_action": np.array([0.0], dtype=np.float32)},
    )

    with pytest.raises(SafetyGuardError, match="action index 1"):
        scheduler.execute(chunk)


def test_max_delta_guard_accepts_one_float32_ulp_at_configured_boundary():
    limit = np.float32(0.033)
    rounded_boundary = np.nextafter(limit, np.float32(np.inf))
    scheduler = ActionScheduler(
        RejectingClient(),
        dry_run=True,
        filters=[MaxDeltaGuard(max_delta=np.array([limit], dtype=np.float32))],
    )

    assert scheduler.execute(
        _chunk(np.array([rounded_boundary]), np.array([0.0]))
    ) == 1


def test_max_delta_guard_rejects_more_than_one_float32_ulp_over_boundary():
    limit = np.float32(0.033)
    two_ulps_over = np.nextafter(
        np.nextafter(limit, np.float32(np.inf)),
        np.float32(np.inf),
    )
    scheduler = ActionScheduler(
        RejectingClient(),
        dry_run=True,
        filters=[MaxDeltaGuard(max_delta=np.array([limit], dtype=np.float32))],
    )

    with pytest.raises(SafetyGuardError, match="abs_delta"):
        scheduler.execute(_chunk(np.array([two_ulps_over]), np.array([0.0])))


def test_max_delta_filter_limits_each_chunk_from_observed_current_action():
    scheduler = ActionScheduler(
        RejectingClient(),
        dry_run=True,
        filters=[MaxDeltaFilter(max_delta=np.array([0.1], dtype=np.float32))],
    )
    first = ActionChunk(
        actions=np.array([[0.05], [0.30]], dtype=np.float32),
        action_space="abs_qpos",
        hz=10.0,
        metadata={"current_action": np.array([0.0], dtype=np.float32)},
    )
    second = ActionChunk(
        actions=np.array([[0.35]], dtype=np.float32),
        action_space="abs_qpos",
        hz=10.0,
        metadata={"current_action": np.array([0.12], dtype=np.float32)},
    )

    assert scheduler.execute(first) == 2
    assert scheduler.execute(second) == 1
    np.testing.assert_allclose(
        np.asarray(scheduler.history).reshape(-1),
        np.array([0.05, 0.15, 0.22], dtype=np.float32),
    )


def test_max_delta_filter_requires_measured_qpos_when_configured():
    scheduler = ActionScheduler(
        RejectingClient(),
        dry_run=True,
        filters=[
            MaxDeltaFilter(
                max_delta=np.array([0.1], dtype=np.float32),
                require_previous=True,
            )
        ],
    )
    chunk = ActionChunk(
        actions=np.array([[0.3]], dtype=np.float32),
        action_space="abs_qpos",
        hz=10.0,
    )

    with pytest.raises(ValueError, match="requires current_action metadata"):
        scheduler.execute(chunk)


def test_max_delta_filter_reports_first_step_clipping_from_measured_qpos(capsys):
    scheduler = ActionScheduler(
        RejectingClient(),
        dry_run=True,
        filters=[
            MaxDeltaFilter(
                max_delta=np.array([0.1, 0.2], dtype=np.float32),
                require_previous=True,
                report_clipping=True,
            )
        ],
    )

    assert scheduler.execute(
        _chunk(np.array([0.8, 0.3]), np.array([0.5, 0.5]))
    ) == 1
    np.testing.assert_allclose(
        np.asarray(scheduler.history)[0],
        np.array([0.6, 0.3], dtype=np.float32),
    )
    output = capsys.readouterr().out
    assert "clipped action index 0" in output
    assert "dimensions=[0]" in output


def test_gripper_close_lead_advances_closing_but_not_opening():
    scheduler = ActionScheduler(
        RejectingClient(),
        dry_run=True,
        filters=[
            GripperCloseLeadFilter(
                lead_steps=2,
                gripper_indices=[1, 3],
                min_close_delta=0.01,
            )
        ],
    )
    chunk = ActionChunk(
        actions=np.array(
            [
                [1.0, 0.080, 2.0, 0.020],
                [1.1, 0.070, 2.1, 0.040],
                [1.2, 0.040, 2.2, 0.060],
                [1.3, 0.020, 2.3, 0.080],
            ],
            dtype=np.float32,
        ),
        action_space="abs_qpos",
        hz=10.0,
        metadata={"current_action": np.array([1.0, 0.080, 2.0, 0.020])},
    )

    assert scheduler.execute(chunk) == 4
    history = np.asarray(scheduler.history)
    np.testing.assert_allclose(history[:, 0], chunk.actions[:, 0])
    np.testing.assert_allclose(history[:, 2], chunk.actions[:, 2])
    np.testing.assert_allclose(history[:, 1], [0.040, 0.020, 0.020, 0.020])
    np.testing.assert_allclose(history[:, 3], chunk.actions[:, 3])


def test_gripper_close_height_gate_holds_open_and_requests_fresh_replan():
    scheduler = ActionScheduler(
        RejectingClient(),
        dry_run=True,
        filters=[
            GripperCloseHeightGateFilter(
                max_close_height_m=0.009,
                gripper_indices=[1],
                arm_starts=[0],
                arm_dof=1,
                height_fn=lambda joints: float(joints[0]),
            )
        ],
    )
    chunk = ActionChunk(
        actions=np.array(
            [[0.030, 0.020], [0.008, 0.020], [0.020, 0.020]],
            dtype=np.float32,
        ),
        action_space="abs_qpos",
        hz=10.0,
        metadata={"current_action": np.array([0.040, 0.080], dtype=np.float32)},
    )

    assert scheduler.execute(chunk) == 1
    np.testing.assert_allclose(
        np.asarray(scheduler.history),
        np.array([[0.030, 0.080]], dtype=np.float32),
    )


def test_gripper_close_height_gate_latches_a_low_close_until_explicit_open():
    scheduler = ActionScheduler(
        RejectingClient(),
        dry_run=True,
        filters=[
            GripperCloseHeightGateFilter(
                max_close_height_m=0.009,
                gripper_indices=[1],
                arm_starts=[0],
                arm_dof=1,
                height_fn=lambda joints: float(joints[0]),
            )
        ],
    )
    chunk = ActionChunk(
        actions=np.array(
            [[0.008, 0.020], [0.030, 0.020], [0.030, 0.080]],
            dtype=np.float32,
        ),
        action_space="abs_qpos",
        hz=10.0,
        metadata={"current_action": np.array([0.008, 0.080], dtype=np.float32)},
    )

    assert scheduler.execute(chunk) == 3
    np.testing.assert_allclose(
        np.asarray(scheduler.history)[:, 1],
        np.array([0.020, 0.020, 0.080], dtype=np.float32),
    )


def test_stateful_dynamics_filter_rebases_to_measured_position_and_velocity():
    scheduler = ActionScheduler(
        RejectingClient(),
        dry_run=True,
        filters=[
            StatefulDynamicsFilter(
                max_delta=np.array([0.10], dtype=np.float32),
                max_delta_change=np.array([0.04], dtype=np.float32),
            )
        ],
    )
    first = ActionChunk(
        actions=np.array([[0.30], [0.30]], dtype=np.float32),
        action_space="abs_qpos",
        hz=10.0,
        metadata={
            "current_action": np.array([0.0], dtype=np.float32),
            "current_velocity": np.array([0.0], dtype=np.float32),
        },
    )
    second = ActionChunk(
        actions=np.array([[-0.30], [-0.30]], dtype=np.float32),
        action_space="abs_qpos",
        hz=10.0,
        # Simulate tracking lag while the hardware is still moving forward at
        # 0.8 units/s. The first new target should decelerate from measured
        # velocity, not continue from the unattained 0.12 command target.
        metadata={
            "current_action": np.array([0.02], dtype=np.float32),
            "current_velocity": np.array([0.8], dtype=np.float32),
        },
    )

    assert scheduler.execute(first) == 2
    assert scheduler.execute(second) == 2
    np.testing.assert_allclose(
        np.asarray(scheduler.history).reshape(-1),
        np.array([0.04, 0.12, 0.06, 0.06], dtype=np.float32),
        atol=1e-6,
    )


def test_stateful_dynamics_filter_requires_measured_velocity_at_chunk_start():
    scheduler = ActionScheduler(
        RejectingClient(),
        dry_run=True,
        filters=[
            StatefulDynamicsFilter(
                max_delta=np.array([0.10], dtype=np.float32),
                max_delta_change=np.array([0.04], dtype=np.float32),
            )
        ],
    )
    chunk = ActionChunk(
        actions=np.array([[0.1]], dtype=np.float32),
        action_space="abs_qpos",
        hz=10.0,
        metadata={"current_action": np.array([0.0], dtype=np.float32)},
    )

    with pytest.raises(ValueError, match="current_action and current_velocity"):
        scheduler.execute(chunk)


def test_stateful_dynamics_filter_brakes_measured_velocity_before_soft_rate_limit():
    scheduler = ActionScheduler(
        RejectingClient(),
        dry_run=True,
        filters=[
            StatefulDynamicsFilter(
                max_delta=np.array([0.10], dtype=np.float32),
                max_delta_change=np.array([0.04], dtype=np.float32),
            )
        ],
    )
    chunk = ActionChunk(
        actions=np.array([[-1.0]], dtype=np.float32),
        action_space="abs_qpos",
        hz=10.0,
        metadata={
            "current_action": np.array([0.0], dtype=np.float32),
            "current_velocity": np.array([2.0], dtype=np.float32),
        },
    )

    assert scheduler.execute(chunk) == 1
    np.testing.assert_allclose(
        np.asarray(scheduler.history).reshape(-1),
        np.array([0.16], dtype=np.float32),
        atol=1e-6,
    )


def test_stateful_dynamics_filter_hard_bounds_override_soft_braking_envelope():
    scheduler = ActionScheduler(
        RejectingClient(),
        dry_run=True,
        filters=[
            StatefulDynamicsFilter(
                max_delta=np.array([0.10], dtype=np.float32),
                max_delta_change=np.array([0.005], dtype=np.float32),
                lower=np.array([-0.05], dtype=np.float32),
                upper=np.array([1.00], dtype=np.float32),
            ),
            BoundsGuard(
                lower=np.array([-0.05], dtype=np.float32),
                upper=np.array([1.00], dtype=np.float32),
            ),
        ],
    )
    chunk = ActionChunk(
        actions=np.array([[-0.04], [0.10]], dtype=np.float32),
        action_space="abs_qpos",
        hz=10.0,
        metadata={
            "current_action": np.array([-0.049], dtype=np.float32),
            "current_velocity": np.array([-0.20], dtype=np.float32),
        },
    )

    assert scheduler.execute(chunk) == 2
    history = np.asarray(scheduler.history).reshape(-1)
    assert history[0] == pytest.approx(-0.05)
    assert history[1] > history[0]
    assert np.all(history >= -0.05)


def test_stateful_dynamics_filter_reset_starts_from_new_observation():
    scheduler = ActionScheduler(
        RejectingClient(),
        dry_run=True,
        filters=[
            StatefulDynamicsFilter(
                max_delta=np.array([0.10], dtype=np.float32),
                max_delta_change=np.array([0.04], dtype=np.float32),
            )
        ],
    )

    assert scheduler.execute(_chunk(np.array([1.0]), np.array([0.0]))) == 1
    scheduler.reset()
    assert scheduler.execute(_chunk(np.array([1.0]), np.array([0.5]))) == 1
    np.testing.assert_allclose(
        np.asarray(scheduler.history).reshape(-1),
        np.array([0.04, 0.54], dtype=np.float32),
        atol=1e-6,
    )


def test_action_hz_deadline_is_preserved_across_chunks(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(scheduler_module.time, "monotonic", lambda: clock[0])
    client = TimedClient(clock)
    scheduler = ActionScheduler(client, dry_run=False, wait=True)
    first = ActionChunk(
        actions=np.array([[0.0], [0.1]], dtype=np.float32),
        action_space="abs_qpos",
        hz=10.0,
        metadata={"current_action": np.array([0.0], dtype=np.float32)},
    )
    second = ActionChunk(
        actions=np.array([[0.2]], dtype=np.float32),
        action_space="abs_qpos",
        hz=10.0,
        metadata={"current_action": np.array([0.1], dtype=np.float32)},
    )

    assert scheduler.execute(first) == 2
    clock[0] += 0.02  # Fast policy re-inference must not shorten the next period.
    assert scheduler.execute(second) == 1

    assert client.sent_at == pytest.approx([10.0, 10.1, 10.2])


def test_scheduler_idle_spins_robot_client(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(scheduler_module.time, "monotonic", lambda: clock[0])
    client = TimedClient(clock)
    scheduler = ActionScheduler(client, dry_run=True)

    scheduler.idle(0.04)

    assert clock[0] == pytest.approx(10.04)
