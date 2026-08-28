"""Deterministic inventory and manifest construction for full-camera Future4."""

from __future__ import annotations

import csv
import hashlib
import io
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from mip.future_rollout_audit.schemas import validate_manifest


TASK_ALIASES = {"mimicgen_kitchen": "kitchen"}
FULL_CAMERA_KEYS = {
    "mug_mug": ("agentview_rgb", "eye_in_hand_rgb"),
    "moka_moka": ("agentview_rgb", "eye_in_hand_rgb"),
    "square": ("agentview_image", "robot0_eye_in_hand_image"),
    "tool_hang": ("robot0_eye_in_hand_image", "sideview_image"),
    "transport": (
        "robot0_eye_in_hand_image",
        "robot1_eye_in_hand_image",
        "shouldercamera0_image",
        "shouldercamera1_image",
    ),
    "coffee_preparation": ("agentview_image", "robot0_eye_in_hand_image"),
    "kitchen": ("agentview_image", "robot0_eye_in_hand_image"),
    "three_piece_assembly": ("agentview_image", "robot0_eye_in_hand_image"),
}
CHECKPOINT_PATTERN = re.compile(r"(?:model|checkpoint)[_-]?(\d+)\.pt$")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_task(config) -> str:
    raw = str(config.task.env_name)
    return TASK_ALIASES.get(raw, raw)


def _camera_keys(config) -> tuple[str, ...]:
    return tuple(
        sorted(
            key
            for key, value in config.task.shape_meta.obs.items()
            if value.get("type", "low_dim") == "rgb"
        )
    )


def _future_steps(config) -> list[int]:
    values = getattr(config.task, "future_state_steps_list", None)
    if values:
        return [int(value) for value in values]
    return [int(getattr(config.task, "future_state_steps", -1))]


def _is_full_camera_future4(config) -> bool:
    task = _canonical_task(config)
    if task not in FULL_CAMERA_KEYS:
        return False
    return all(
        (
            _camera_keys(config) == FULL_CAMERA_KEYS[task],
            bool(getattr(config.task, "future_state_enabled", False)),
            _future_steps(config) == [4],
            bool(getattr(config.optimization, "future_joint_mode", False)),
            getattr(config.optimization, "future_embed_loss_mode", None)
            == "mip_two_step",
            abs(
                float(
                    getattr(config.optimization, "future_state_loss_ratio", -1.0)
                )
                - 0.1
            )
            < 1e-12,
        )
    )


def _logged_step(
    path: Path,
    checkpoint: dict[str, Any],
    logged_step_override: int | None,
) -> int:
    training_state = checkpoint.get("training_state") or {}
    if "n_gradient_step" in training_state:
        return int(training_state["n_gradient_step"])
    if logged_step_override is not None:
        return int(logged_step_override)
    match = CHECKPOINT_PATTERN.search(path.name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Checkpoint does not expose its logged step: {path}")


def checkpoint_record(
    path: str | Path,
    *,
    content_hash: bool = True,
    logged_step_override: int | None = None,
) -> dict:
    checkpoint_path = Path(path).resolve()
    state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    required = {"encoder", "encoder_ema", "flow_map", "flow_map_ema"}
    missing = sorted(required.difference(state))
    if missing:
        raise ValueError(f"Checkpoint {checkpoint_path} missing keys: {missing}")
    logged_step = _logged_step(checkpoint_path, state, logged_step_override)
    del state
    stat = checkpoint_path.stat()
    fingerprint = (
        file_sha256(checkpoint_path)
        if content_hash
        else f"uncomputed:size={stat.st_size}:mtime_ns={stat.st_mtime_ns}"
    )
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_fingerprint": fingerprint,
        "checkpoint_fingerprint_kind": "sha256" if content_hash else "uncomputed",
        "checkpoint_logged_step": logged_step,
        "optimizer_updates_completed": logged_step + 1,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "historical_success_status": "legacy_or_unverified_success_semantics",
    }


def _alias_step_overrides(run_dir: Path) -> dict[str, int]:
    """Resolve legacy best/latest aliases without accepting their success semantics."""

    latest_path = run_dir / "models" / "model_latest.pt"
    if not latest_path.is_file():
        return {}
    latest = torch.load(
        latest_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    training_state = latest.get("training_state") or {}
    latest_step = training_state.get("n_gradient_step")
    overrides = {}
    if latest_step is not None:
        overrides["model_latest.pt"] = int(latest_step)
    best_metrics = training_state.get("best_metrics") or {}
    history = training_state.get("eval_history") or []
    success_keys = sorted(
        key for key in best_metrics if str(key).startswith("mean_success_")
    )
    if success_keys and history:
        key = success_keys[0]
        best_value = float(best_metrics[key])
        matching_steps = [
            int(item["step"])
            for item in history
            if key in item and abs(float(item[key]) - best_value) < 1e-12
        ]
        if matching_steps:
            # The logger replaces model_best only on strict improvement, so
            # the first occurrence of the final maximum owns the alias.
            overrides["model_best.pt"] = min(matching_steps)
    del latest
    return overrides


def discover_inventory(
    repo_root: str | Path,
    *,
    content_hash: bool = True,
) -> dict:
    """Discover full-camera Future4 candidates without selecting any of them."""

    root = Path(repo_root).resolve()
    search_roots = (
        root / "logs" / "ablation2" / "libero",
        root / "logs" / "ablation2" / "robomimic",
        root / "logs" / "mimicgen",
    )
    config_paths = sorted(
        config_path
        for search_root in search_roots
        for config_path in search_root.glob("*/resolved_config.yaml")
    )
    candidates = []
    rejected = []
    source_mtimes = []
    for config_path in config_paths:
        config = OmegaConf.load(config_path)
        if not _is_full_camera_future4(config):
            continue
        task = _canonical_task(config)
        run_dir = config_path.parent
        checkpoint_paths = sorted((run_dir / "models").glob("model_*.pt"))
        if not checkpoint_paths:
            rejected.append(
                {
                    "path": str(run_dir.resolve()),
                    "reason": "missing_checkpoint_candidates",
                }
            )
            continue
        config_bytes = config_path.read_bytes()
        config_sha = hashlib.sha256(config_bytes).hexdigest()
        source_mtimes.append(config_path.stat().st_mtime_ns)
        alias_steps = _alias_step_overrides(run_dir)
        for path in checkpoint_paths:
            try:
                record = checkpoint_record(
                    path,
                    content_hash=content_hash,
                    logged_step_override=alias_steps.get(path.name),
                )
            except ValueError as exc:
                rejected.append(
                    {"path": str(path.resolve()), "reason": str(exc)}
                )
                continue
            source_mtimes.append(record["mtime_ns"])
            candidates.append(
                {
                    "task": task,
                    "training_seed": int(config.optimization.seed),
                    "run_dir": str(run_dir.resolve()),
                    "config_path": str(config_path.resolve()),
                    "config_sha256": config_sha,
                    "camera_key_order": list(_camera_keys(config)),
                    "future4": True,
                    "ratio": "0.100000000000",
                    **record,
                }
            )
    candidates.sort(
        key=lambda item: (
            item["task"],
            item["training_seed"],
            item["optimizer_updates_completed"],
            item["checkpoint_path"],
        )
    )
    rejected.sort(key=lambda item: (item["reason"], item["path"]))
    maximum = max(source_mtimes, default=0)
    timestamp = datetime.fromtimestamp(maximum / 1e9, tz=UTC).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")
    return {
        "format_version": 1,
        "inventory_kind": "future_rollout_full_camera_candidates",
        "source_snapshot_max_mtime_utc": timestamp,
        "candidate_count": len(candidates),
        "rejected_count": len(rejected),
        "candidates": candidates,
        "rejected": rejected,
    }


def blocked_formal_manifest(source_snapshot_max_mtime_utc: str) -> dict:
    manifest = {
        "format_version": 1,
        "manifest_kind": "formal_selection",
        "status": "blocked_no_verified_selection",
        "consumable_by_formal_audit": False,
        "entry_count": 0,
        "selected_count": 0,
        "source_snapshot_max_mtime_utc": source_snapshot_max_mtime_utc,
        "entries": [],
    }
    validate_manifest(manifest)
    return manifest


def smoke_manifest(source_snapshot_max_mtime_utc: str, entries: list[dict]) -> dict:
    ordered = sorted(
        entries,
        key=lambda item: (
            item["task"], item["training_seed"], item["checkpoint_path"]
        ),
    )
    manifest = {
        "format_version": 1,
        "manifest_kind": "smoke_only",
        "status": "ready_for_smoke",
        "consumable_by_formal_audit": False,
        "entry_count": len(ordered),
        "selected_count": 0,
        "smoke_checkpoint_count": len(ordered),
        "source_snapshot_max_mtime_utc": source_snapshot_max_mtime_utc,
        "entries": ordered,
    }
    validate_manifest(manifest)
    return manifest


def inventory_csv_bytes(inventory: dict) -> bytes:
    fields = (
        "task",
        "training_seed",
        "checkpoint_path",
        "checkpoint_fingerprint",
        "checkpoint_logged_step",
        "optimizer_updates_completed",
        "size_bytes",
        "historical_success_status",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=fields,
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(inventory["candidates"])
    return stream.getvalue().encode("utf-8")
