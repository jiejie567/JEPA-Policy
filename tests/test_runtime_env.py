from types import SimpleNamespace

import pytest

from mip import runtime_env


@pytest.mark.parametrize(
    ("task", "versions"),
    [
        (
            SimpleNamespace(env_type="ph", env_name="square"),
            {"mujoco": "3.3.6", "robosuite": "1.5.1"},
        ),
        (
            SimpleNamespace(env_type="libero", env_name="mug_mug"),
            {"mujoco": "3.10.0", "numpy": "1.26.0", "robosuite": "1.4.0"},
        ),
        (
            SimpleNamespace(env_type="mimicgen", env_name="CoffeePreparation_D1"),
            {
                "mimicgen": "1.0.0",
                "mujoco": "3.3.6",
                "numpy": "1.26.4",
                "robomimic": "0.3.0",
                "robosuite": "1.4.1",
            },
        ),
    ],
)
def test_validate_runtime_environment_accepts_matching_versions(
    monkeypatch, task, versions
):
    monkeypatch.setattr(runtime_env, "_installed_version", versions.get)

    runtime_env.validate_runtime_environment(task)


def test_validate_runtime_environment_rejects_crossed_environment(monkeypatch):
    task = SimpleNamespace(env_type="libero", env_name="mug_mug")
    versions = {"mujoco": "3.3.6", "numpy": "2.2.6", "robosuite": "1.5.1"}
    monkeypatch.setattr(runtime_env, "_installed_version", versions.get)

    with pytest.raises(RuntimeError, match="tools/run_libero_train.sh"):
        runtime_env.validate_runtime_environment(task)
