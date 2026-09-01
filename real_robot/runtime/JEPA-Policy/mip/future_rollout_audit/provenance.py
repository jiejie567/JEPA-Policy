"""Reproducibility snapshots for pilot and formal selection evaluation."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import torch

from mip.future_rollout_audit.schemas import canonical_sha256


RUNTIME_SOURCE_GLOBS = (
    "mip/future_rollout_audit/*.py",
    "mip/agent.py",
    "mip/env_utils.py",
    "mip/envs/persistent_image_rollout.py",
    "mip/envs/robomimic/*.py",
    "mip/networks/chitfm.py",
    "mip/samplers.py",
)
RUNTIME_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "EGL_PLATFORM",
    "JEPA_POLICY_EGL_DEVICE_ID",
    "LD_LIBRARY_PATH",
    "LIBERO_CONFIG_PATH",
    "MUJOCO_GL",
    "PPU_HOME",
    "PPU_PATH",
    "PPU_SDK",
    "PPU_VERSION",
    "PYOPENGL_PLATFORM",
    "PYTHONPATH",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(repo_root: Path, *command: str) -> bytes:
    return subprocess.run(
        command,
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def runtime_source_manifest(repo_root: str | Path) -> dict:
    root = Path(repo_root).resolve()
    paths = {
        path.resolve()
        for pattern in RUNTIME_SOURCE_GLOBS
        for path in root.glob(pattern)
        if path.is_file()
    }
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(paths)
    ]
    return {
        "files": files,
        "runtime_source_bundle_sha256": canonical_sha256(files),
    }


def _untracked_manifest(repo_root: Path) -> list[dict]:
    raw = _run(repo_root, "git", "ls-files", "--others", "--exclude-standard", "-z")
    relative_paths = sorted(
        item.decode("utf-8") for item in raw.split(b"\0") if item
    )
    output = []
    for relative in relative_paths:
        path = repo_root / relative
        if path.is_file():
            output.append(
                {
                    "path": relative,
                    "sha256": file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return output


def _container_identity() -> dict:
    environment = {
        key: value
        for key, value in sorted(os.environ.items())
        if any(token in key.upper() for token in ("IMAGE", "CONTAINER", "DSW"))
        and "TOKEN" not in key.upper()
        and "PASSWORD" not in key.upper()
        and "SECRET" not in key.upper()
    }
    os_release = ""
    path = Path("/etc/os-release")
    if path.exists():
        os_release = path.read_text(encoding="utf-8", errors="replace")
    identity = {
        "environment": environment,
        "hostname": platform.node(),
        "kernel": platform.release(),
        "os_release": os_release,
    }
    identity["container_image_digest"] = canonical_sha256(identity)
    identity["container_image_digest_kind"] = "runtime_identity_fallback_sha256"
    for key, value in environment.items():
        if "DIGEST" in key.upper() and len(value) >= 32:
            identity["container_image_digest"] = value
            identity["container_image_digest_kind"] = f"environment:{key}"
            break
    return identity


def build_environment_provenance(repo_root: str | Path) -> dict:
    """Capture dirty-revision, dependency, source, and runtime identities."""

    root = Path(repo_root).resolve()
    head = _run(root, "git", "rev-parse", "HEAD").decode("ascii").strip()
    tracked_diff = _run(root, "git", "diff", "--binary", "--no-ext-diff")
    tracked_diff_sha256 = hashlib.sha256(tracked_diff).hexdigest()
    untracked = _untracked_manifest(root)
    source = runtime_source_manifest(root)
    pip_freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.decode("utf-8").splitlines()
    snapshot = {
        "format_version": 1,
        "record_kind": "future_rollout_environment_provenance",
        "repo_root": str(root),
        "head_commit": head,
        "worktree_dirty": bool(tracked_diff or untracked),
        "tracked_diff_sha256": tracked_diff_sha256,
        "untracked_file_manifest": untracked,
        "untracked_file_manifest_sha256": canonical_sha256(untracked),
        **source,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "pip_freeze": pip_freeze,
        "pip_freeze_sha256": canonical_sha256(pip_freeze),
        "container_identity": _container_identity(),
        "runtime_environment": {
            key: os.environ[key]
            for key in RUNTIME_ENVIRONMENT_KEYS
            if key in os.environ
        },
    }
    snapshot["environment_provenance_sha256"] = canonical_sha256(snapshot)
    return snapshot


def build_evaluator_provenance(
    repo_root: str | Path,
    config_path: str | Path,
    environment_snapshot: dict,
) -> dict:
    root = Path(repo_root).resolve()
    selection_runner = root / "mip/future_rollout_audit/selection_runner.py"
    selection_module = root / "mip/future_rollout_audit/selection.py"
    seeding_module = root / "mip/future_rollout_audit/seeding.py"
    code_payload = [
        file_sha256(selection_runner),
        file_sha256(selection_module),
        file_sha256(seeding_module),
    ]
    return {
        "evaluator_id": "selection-evaluator/v1",
        "code_sha256": canonical_sha256(code_payload),
        "config_sha256": file_sha256(config_path),
        "success_semantics_implementation_sha256": file_sha256(selection_module),
        "seed_generator_sha256": file_sha256(seeding_module),
        "runtime_source_bundle_sha256": environment_snapshot[
            "runtime_source_bundle_sha256"
        ],
        "environment_provenance_sha256": environment_snapshot[
            "environment_provenance_sha256"
        ],
    }
