import os

import pytest

from jepa_policy.logger import _wandb_run_id_from_env


@pytest.mark.parametrize("value", [None, "", "   "])
def test_empty_wandb_run_id_is_removed(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("WANDB_RUN_ID", raising=False)
    else:
        monkeypatch.setenv("WANDB_RUN_ID", value)

    assert _wandb_run_id_from_env() is None
    assert "WANDB_RUN_ID" not in os.environ


def test_nonempty_wandb_run_id_is_preserved(monkeypatch):
    monkeypatch.setenv("WANDB_RUN_ID", "  existing-run-id  ")

    assert _wandb_run_id_from_env() == "existing-run-id"
    assert os.environ["WANDB_RUN_ID"] == "existing-run-id"
