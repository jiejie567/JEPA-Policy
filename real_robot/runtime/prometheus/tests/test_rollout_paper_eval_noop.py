from __future__ import annotations

import json
from collections import deque

from omegaconf import OmegaConf

from prometheus.workflows.rollout_paper_eval import run_from_config


class ScriptedCommands:
    def __init__(self, commands: list[str], active_commands: list[str] | None = None):
        self.commands = deque(commands)
        self.active_commands = deque(active_commands or [])

    def get(self, _timeout_s: float = 0.0) -> str | None:
        return self.commands.popleft() if self.commands else None

    def poll(self) -> str | None:
        return self.active_commands.popleft() if self.active_commands else None

    def discard(self) -> None:
        pass


def test_paper_eval_reuses_policy_hardware_and_scheduler_across_episodes(tmp_path):
    cfg = OmegaConf.load("tests/configs/rollout_sync_noop.yaml")
    cfg.run.output_dir = str(tmp_path / "output")
    cfg.run.runtime_dir = str(tmp_path / "output" / "runtime")
    cfg.workflow.paper_eval = {"recovery_seconds": 0, "max_episodes": 2}
    cfg.workflow.reset_home_timeout_s = 1.0

    commands = ScriptedCommands(
        ["start", "start"],
        active_commands=[None, "success", None, None, "failure"],
    )
    assert run_from_config(cfg, commands=commands) == 0

    manifest = json.loads((tmp_path / "output" / "workflow_manifest.json").read_text())
    assert manifest["persistent_resources"] is True
    assert manifest["checkpoint_loads"] == 1
    assert manifest["episode_count"] == 2
    assert manifest["success_count"] == 1
    assert manifest["failure_count"] == 1
    assert manifest["success_rate"] == 0.5
    assert [item["outcome"] for item in manifest["episodes"]] == ["success", "failure"]
    assert manifest["policy"]["details"]["policy"]["infer_count"] == 2
    assert manifest["robot_client"]["sent_actions"] == 0
    assert manifest["robot_client"]["reset_count"] == 2
    assert (
        tmp_path / "output" / "paper_eval" / "episodes" / "success" / "0000.json"
    ).is_file()
    assert (
        tmp_path / "output" / "paper_eval" / "episodes" / "failure" / "0001.json"
    ).is_file()
    summary = json.loads(
        (tmp_path / "output" / "paper_eval" / "summary.json").read_text()
    )
    assert summary["evaluated_episodes"] == 2
    assert summary["success_rate"] == 0.5


def test_paper_eval_skip_is_not_counted_or_saved(tmp_path):
    cfg = OmegaConf.load("tests/configs/rollout_sync_noop.yaml")
    cfg.run.output_dir = str(tmp_path / "output")
    cfg.run.runtime_dir = str(tmp_path / "output" / "runtime")
    cfg.workflow.paper_eval = {"recovery_seconds": 0, "max_episodes": 1}
    cfg.workflow.reset_home_timeout_s = 1.0
    cfg.data.recording = {
        "enabled": True,
        "output_dir": str(tmp_path / "output" / "data"),
        "queue_size": 64,
        "fps": 30.0,
        "enable_plots": False,
        "default_event_value": 0,
    }

    commands = ScriptedCommands(
        ["start", "start"], active_commands=["skip", None, "success"]
    )
    assert run_from_config(cfg, commands=commands) == 0

    manifest = json.loads((tmp_path / "output" / "workflow_manifest.json").read_text())
    assert manifest["episode_count"] == 1
    assert manifest["success_count"] == 1
    assert manifest["failure_count"] == 0
    assert manifest["skipped_count"] == 1
    assert manifest["robot_client"]["reset_count"] == 2
    assert not (tmp_path / "output" / "paper_eval" / "episodes" / "skip").exists()
    assert not (tmp_path / "output" / "data" / "0000").exists()
    assert (tmp_path / "output" / "data" / "success" / "0000").is_dir()
    rows = (tmp_path / "output" / "paper_eval" / "episodes.tsv").read_text().splitlines()
    assert len(rows) == 2
    assert "\tsuccess\t" in rows[1]


def test_paper_eval_safety_guard_marks_failure_and_keeps_session_orderly(tmp_path):
    cfg = OmegaConf.load("tests/configs/rollout_sync_noop.yaml")
    cfg.run.output_dir = str(tmp_path / "output")
    cfg.run.runtime_dir = str(tmp_path / "output" / "runtime")
    cfg.workflow.paper_eval = {"recovery_seconds": 0, "max_episodes": 1}
    cfg.workflow.reset_home_timeout_s = 1.0
    cfg.scheduler.filters = [
        {
            "_target_": "prometheus.policy.scheduler.BoundsGuard",
            "name": "forced_test_guard",
            "lower": [1.0] * 2,
            "upper": [2.0] * 2,
        }
    ]

    assert run_from_config(cfg, commands=ScriptedCommands(["start"])) == 0

    manifest = json.loads((tmp_path / "output" / "workflow_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["episode_count"] == 1
    assert manifest["success_count"] == 0
    assert manifest["failure_count"] == 1
    assert manifest["checkpoint_loads"] == 1
    assert manifest["episodes"][0]["stop_reason"] == "safety_guard"
    assert manifest["robot_client"]["reset_count"] == 1
    assert (
        tmp_path / "output" / "paper_eval" / "episodes" / "failure" / "0000.json"
    ).is_file()
