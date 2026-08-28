from __future__ import annotations

import json
import pickle
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf

from prometheus.policy.scheduler import ActionChunk
from prometheus.workflows.rollout_sync import (
    _check_startup_action_semantics,
    _ensure_startup_home,
    run_from_config,
)


def test_startup_action_semantics_accepts_training_like_left_first_chunk():
    current = np.zeros(14, dtype=np.float32)
    actions = np.zeros((8, 14), dtype=np.float32)
    actions[:, 1] = np.linspace(0.0, 0.18, 8)
    actions[:, 8] = np.linspace(0.0, 0.025, 8)
    chunk = ActionChunk(
        actions,
        action_space="abs_qpos",
        hz=10.0,
        metadata={"current_action": current},
    )

    result = _check_startup_action_semantics(
        chunk,
        {
            "enabled": True,
            "chunk_count": 1,
            "inactive_dimensions": [7, 8, 9, 10, 11, 12],
            "max_displacement": 0.04,
        },
        chunk_index=0,
    )

    assert result is not None
    assert result["rejected"] is False


def test_startup_action_semantics_rejects_right_first_chunk():
    current = np.zeros(14, dtype=np.float32)
    actions = np.zeros((8, 14), dtype=np.float32)
    actions[:, 9] = np.linspace(0.0, 0.10, 8)
    chunk = ActionChunk(
        actions,
        action_space="abs_qpos",
        hz=10.0,
        metadata={"current_action": current},
    )

    result = _check_startup_action_semantics(
        chunk,
        {
            "enabled": True,
            "chunk_count": 1,
            "inactive_dimensions": [7, 8, 9, 10, 11, 12],
            "max_displacement": 0.04,
        },
        chunk_index=0,
    )

    assert result is not None
    assert result["rejected"] is True


def test_startup_action_semantics_rejects_large_first_action_on_active_arm():
    current = np.zeros(14, dtype=np.float32)
    actions = np.zeros((8, 14), dtype=np.float32)
    actions[0, 9] = 0.20
    chunk = ActionChunk(
        actions,
        action_space="abs_qpos",
        hz=10.0,
        metadata={"current_action": current},
    )

    result = _check_startup_action_semantics(
        chunk,
        {
            "enabled": True,
            "chunk_count": 1,
            "inactive_dimensions": [0, 1, 2, 3, 4, 5],
            "max_displacement": 0.04,
            "first_action_max_delta": [0.05] * 14,
        },
        chunk_index=0,
    )

    assert result is not None
    assert result["rejected"] is True
    assert result["first_action_exceeded_dimensions"] == [9]


def test_startup_action_semantics_can_check_only_task_active_dimensions():
    current = np.zeros(14, dtype=np.float32)
    actions = np.zeros((8, 14), dtype=np.float32)
    actions[0, 0] = 0.20
    actions[0, 9] = 0.02
    chunk = ActionChunk(
        actions,
        action_space="abs_qpos",
        hz=10.0,
        metadata={"current_action": current},
    )

    result = _check_startup_action_semantics(
        chunk,
        {
            "enabled": True,
            "chunk_count": 1,
            "inactive_dimensions": [],
            "max_displacement": 0.04,
            "first_action_max_delta": [0.05] * 14,
            "first_action_dimensions": [7, 8, 9, 10, 11, 12, 13],
        },
        chunk_index=0,
    )

    assert result is not None
    assert result["rejected"] is False
    assert result["first_action_exceeded_dimensions"] == []


def test_rollout_sync_noop(tmp_path):
    cfg = OmegaConf.load("tests/configs/rollout_sync_noop.yaml")
    cfg.run.output_dir = str(tmp_path / "output")
    cfg.run.runtime_dir = str(tmp_path / "output" / "runtime")

    assert run_from_config(cfg) == 0

    manifest = json.loads((tmp_path / "output" / "workflow_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["chunks"] == 3
    assert manifest["actions_sent"] == 3
    assert manifest["task"]["id"] == "noop_task"
    assert manifest["robot_client"]["sent_actions"] == 3


def test_rollout_sync_records_action_stream_when_enabled(tmp_path):
    cfg = OmegaConf.load("tests/configs/rollout_sync_noop.yaml")
    cfg.run.output_dir = str(tmp_path / "output")
    cfg.run.runtime_dir = str(tmp_path / "output" / "runtime")
    cfg.data.recording = {
        "enabled": True,
        "output_dir": str(tmp_path / "output" / "episodes"),
        "episode_id": "0001",
        "queue_size": 16,
        "fps": 30.0,
    }

    assert run_from_config(cfg) == 0

    episode_dir = tmp_path / "output" / "episodes" / "0001"
    manifest = json.loads((episode_dir / "manifests" / "action.json").read_text())
    with (episode_dir / "action" / "action_chunk_dict.pkl").open("rb") as file:
        chunks = pickle.load(file)
    with (episode_dir / "action" / "executed_action_dict.pkl").open("rb") as file:
        actions = pickle.load(file)

    action_manifest = manifest["prometheus_action_recorder"]
    assert action_manifest["status"] == "saved"
    assert action_manifest["counts"]["action.policy_chunk"] == 3
    assert action_manifest["counts"]["action.executed"] == 3
    assert action_manifest["policy_action_chunks"] == 3
    assert action_manifest["executed_actions"] == 3
    assert len(chunks["chunks"]) == 3
    assert len(actions["actions"]) == 3
    assert actions["action_space"] == ["noop_action", "noop_action", "noop_action"]
    assert actions["hz"] == [20.0, 20.0, 20.0]


def test_rollout_sync_can_limit_each_chunk_to_one_action(tmp_path):
    cfg = OmegaConf.load("tests/configs/rollout_sync_noop.yaml")
    cfg.run.output_dir = str(tmp_path / "output")
    cfg.run.runtime_dir = str(tmp_path / "output" / "runtime")
    cfg.policy.horizon = 8
    cfg.workflow.action_steps_per_chunk = 1

    assert run_from_config(cfg) == 0

    manifest = json.loads((tmp_path / "output" / "workflow_manifest.json").read_text())
    assert manifest["chunks"] == 3
    assert manifest["actions_sent"] == 3
    assert manifest["scheduler"]["action_steps_per_chunk"] == 1


def test_rollout_sync_reports_post_chunk_observation_delay(tmp_path):
    cfg = OmegaConf.load("tests/configs/rollout_sync_noop.yaml")
    cfg.run.output_dir = str(tmp_path / "output")
    cfg.run.runtime_dir = str(tmp_path / "output" / "runtime")
    cfg.workflow.post_chunk_observation_delay_s = 0.001

    assert run_from_config(cfg) == 0

    manifest = json.loads((tmp_path / "output" / "workflow_manifest.json").read_text())
    assert manifest["scheduler"]["post_chunk_observation_delay_s"] == 0.001


def test_rollout_sync_resets_home_after_safety_guard_rejection(tmp_path):
    cfg = OmegaConf.load("tests/configs/rollout_sync_noop.yaml")
    cfg.run.output_dir = str(tmp_path / "output")
    cfg.run.runtime_dir = str(tmp_path / "output" / "runtime")
    cfg.scheduler.filters = [
        {
            "_target_": "prometheus.policy.scheduler.BoundsGuard",
            "lower": [1.0, 1.0],
            "upper": [2.0, 2.0],
        }
    ]
    cfg.workflow.reset_home_on_guard_rejection = True
    cfg.workflow.reset_home_timeout_s = 1.0

    assert run_from_config(cfg) == 1

    manifest = json.loads((tmp_path / "output" / "workflow_manifest.json").read_text())
    assert manifest["status"] == "error"
    assert manifest["actions_sent"] == 0
    assert manifest["guard_rejection_reset_attempted"] is True
    assert manifest["reset_home_succeeded"] is True
    assert manifest["robot_client"]["reset_count"] == 1


def test_rollout_sync_resets_home_after_non_guard_error(tmp_path):
    cfg = OmegaConf.load("tests/configs/rollout_sync_noop.yaml")
    cfg.run.output_dir = str(tmp_path / "output")
    cfg.run.runtime_dir = str(tmp_path / "output" / "runtime")
    cfg.scheduler.filters = [
        {
            "_target_": "prometheus.policy.scheduler.BoundsGuard",
            "lower": [1.0, 1.0],
            "upper": [2.0, 2.0],
        }
    ]
    cfg.workflow.reset_home_on_error = True
    cfg.workflow.reset_home_timeout_s = 1.0

    assert run_from_config(cfg) == 1

    manifest = json.loads((tmp_path / "output" / "workflow_manifest.json").read_text())
    assert manifest["status"] == "error"
    assert manifest["error_reset_attempted"] is True
    assert manifest["reset_home_succeeded"] is True
    assert manifest["robot_client"]["reset_count"] == 1


def test_startup_home_check_resets_and_confirms_new_measured_state():
    target = np.array(
        [0.0, 0.0, 0.0, -0.03, 0.0, 0.0, 0.082] * 2,
        dtype=np.float32,
    )

    class Data:
        qpos = target.copy()
        qpos[0] = -0.145
        stamp_ns = 1

        def latest_numpy(self, names):
            assert names == ["robot_state"]
            return {
                "robot_state": SimpleNamespace(
                    data={"position": self.qpos.copy()},
                    stamp_ns=self.stamp_ns,
                )
            }

    class Client:
        reset_count = 0

        def reset_home(self, *, timeout, wait):
            self.reset_count += 1
            data.qpos = target.copy()
            data.stamp_ns += 1
            return {"accepted": True}

        def spin_once(self, _timeout_s):
            return None

    data = Data()
    client = Client()
    attempted, before, after = _ensure_startup_home(
        client,
        data,
        state_stream="robot_state",
        target=target,
        tolerance=np.array([0.03] * 6 + [0.005] + [0.03] * 6 + [0.005]),
        timeout_s=1.0,
        settle_s=0.0,
    )

    assert attempted is True
    assert client.reset_count == 1
    assert before[0] == np.float32(-0.145)
    np.testing.assert_allclose(after, target)


def test_rollout_sync_resets_home_before_cleanup_on_keyboard_interrupt(tmp_path):
    cfg = OmegaConf.load("tests/configs/rollout_sync_noop.yaml")
    cfg.run.output_dir = str(tmp_path / "output")
    cfg.run.runtime_dir = str(tmp_path / "output" / "runtime")
    cfg.policy.interrupt_after = 1
    cfg.workflow.reset_home_on_interrupt = True
    cfg.workflow.reset_home_timeout_s = 1.0

    assert run_from_config(cfg) == 130

    manifest = json.loads((tmp_path / "output" / "workflow_manifest.json").read_text())
    assert manifest["status"] == "interrupted"
    assert manifest["stop_reason"] == "keyboard_interrupt"
    assert manifest["interrupt_reset_attempted"] is True
    assert manifest["reset_home_succeeded"] is True
    assert manifest["robot_client"]["reset_count"] == 1
    events = manifest["robot_client"]["events"]
    assert events.index("reset_home") < events.index("release_control")
    assert events.index("reset_home") < events.index("close")


def test_rollout_sync_dry_run_reports_zero_robot_actions(tmp_path):
    cfg = OmegaConf.load("tests/configs/rollout_sync_noop.yaml")
    cfg.run.output_dir = str(tmp_path / "output")
    cfg.run.runtime_dir = str(tmp_path / "output" / "runtime")
    cfg.scheduler.dry_run = True

    assert run_from_config(cfg) == 0

    manifest = json.loads((tmp_path / "output" / "workflow_manifest.json").read_text())
    assert manifest["actions_processed"] == 3
    assert manifest["actions_sent"] == 0
    assert manifest["robot_client"]["sent_actions"] == 0
