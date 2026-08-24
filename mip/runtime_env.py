"""Runtime checks that keep robomimic and LIBERO dependencies isolated."""

from __future__ import annotations

import os
from importlib import metadata

from mip.libero_utils import is_libero_task
from mip.mimicgen_utils import is_mimicgen_task


_EXPECTED_VERSIONS = {
    "robomimic": {
        "mujoco": "3.3.6",
        "robosuite": "1.5.1",
    },
    "libero": {
        "mujoco": "3.10.0",
        "numpy": "1.26.0",
        "robosuite": "1.4.0",
    },
    "mimicgen": {
        # The vendored code reports 1.0.1 / 0.3.1 via __version__, while
        # the installed distributions retain their upstream metadata versions.
        "mimicgen": "1.0.0",
        "mujoco": "3.3.6",
        "numpy": "1.26.4",
        "robomimic": "0.3.0",
        "robosuite": "1.4.1",
    },
}

_LAUNCHERS = {
    "robomimic": "tools/run_robomimic_train.sh",
    "libero": "tools/run_libero_train.sh",
    "mimicgen": "tools/run_mimicgen_train.sh",
}


def _installed_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def validate_runtime_environment(task_config) -> None:
    """Fail early when a task is launched from the other benchmark's venv."""
    if os.getenv("MIP_ALLOW_ENV_MISMATCH", "0") == "1":
        return

    if is_libero_task(task_config):
        benchmark = "libero"
    elif is_mimicgen_task(task_config):
        benchmark = "mimicgen"
    else:
        benchmark = "robomimic"
    expected = _EXPECTED_VERSIONS[benchmark]
    mismatches = []
    for package, expected_version in expected.items():
        installed_version = _installed_version(package)
        if installed_version != expected_version:
            actual = installed_version or "not installed"
            mismatches.append(f"{package}={actual} (expected {expected_version})")

    if not mismatches:
        return

    mismatch_text = ", ".join(mismatches)
    launcher = _LAUNCHERS[benchmark]
    raise RuntimeError(
        f"Wrong Python environment for {benchmark}: {mismatch_text}. "
        f"Launch this task with {launcher}. "
        "Set MIP_ALLOW_ENV_MISMATCH=1 only for an intentional dependency test."
    )
