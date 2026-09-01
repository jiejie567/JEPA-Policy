"""Closed result schemas and fail-closed manifest validation."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any


class SchemaValidationError(ValueError):
    """Raised when an audit document does not satisfy its closed schema."""


FORMAL_TASKS = (
    "coffee_preparation",
    "kitchen",
    "moka_moka",
    "mug_mug",
    "square",
    "three_piece_assembly",
    "tool_hang",
    "transport",
)
FORMAL_TRAINING_SEEDS = (41, 42, 43)


EVALUATOR_PROVENANCE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "evaluator_id",
        "code_sha256",
        "config_sha256",
        "success_semantics_implementation_sha256",
        "seed_generator_sha256",
        "runtime_source_bundle_sha256",
        "environment_provenance_sha256",
    ],
    "properties": {
        "evaluator_id": {"type": "string"},
        "code_sha256": {"type": "string"},
        "config_sha256": {"type": "string"},
        "success_semantics_implementation_sha256": {"type": "string"},
        "seed_generator_sha256": {"type": "string"},
        "runtime_source_bundle_sha256": {"type": "string"},
        "environment_provenance_sha256": {"type": "string"},
    },
}


SEED_STREAMS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "master_sha256",
        "env",
        "policy",
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    ],
    "properties": {
        "master_sha256": {"type": "string"},
        "env": {"type": "integer"},
        "policy": {"type": "integer"},
        "python": {"type": "integer"},
        "numpy": {"type": "integer"},
        "torch_cpu": {"type": "integer"},
        "torch_cuda": {"type": "integer"},
    },
}


SELECTION_EPISODE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "evidence_kind",
        "task",
        "training_seed",
        "selection_episode_id",
        "checkpoint_fingerprint",
        "evaluator_provenance_sha256",
        "selection_seed_manifest_sha256",
        "seed_streams",
        "success",
        "success_semantics_id",
        "termination_reason",
        "reward_sum",
        "policy_decision_count",
        "environment_steps",
        "sampler_mode",
        "episode_row_sha256",
    ],
    "properties": {
        "evidence_kind": {"type": "string"},
        "task": {"type": "string"},
        "training_seed": {"type": "integer"},
        "selection_episode_id": {"type": "integer"},
        "checkpoint_fingerprint": {"type": "string"},
        "evaluator_provenance_sha256": {"type": "string"},
        "selection_seed_manifest_sha256": {"type": "string"},
        "seed_streams": SEED_STREAMS_SCHEMA,
        "success": {"type": "boolean"},
        "success_semantics_id": {"type": "string"},
        "termination_reason": {"type": "string"},
        "reward_sum": {"type": "number"},
        "policy_decision_count": {"type": "integer"},
        "environment_steps": {"type": "integer"},
        "sampler_mode": {"type": "string"},
        "episode_row_sha256": {"type": "string"},
    },
}


SELECTION_EVIDENCE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "format_version",
        "record_kind",
        "status",
        "evidence_kind",
        "usable_for_formal_selection",
        "seed_domain",
        "task",
        "training_seed",
        "camera_id",
        "future4",
        "ratio",
        "checkpoint_path",
        "checkpoint_fingerprint",
        "checkpoint_logged_step",
        "optimizer_updates_completed",
        "success_semantics_id",
        "evaluator_provenance",
        "evaluator_provenance_sha256",
        "selection_seed_manifest_path",
        "selection_seed_manifest_sha256",
        "selection_equivalence",
        "sampler_mode",
        "requested_episode_ids",
        "success_count",
        "episode_count",
        "episodes",
    ],
    "properties": {
        "format_version": {"const": 1},
        "record_kind": {"const": "selection_episode_evidence"},
        "status": {"type": "string"},
        "evidence_kind": {"type": "string"},
        "usable_for_formal_selection": {"type": "boolean"},
        "seed_domain": {"type": "string"},
        "task": {"type": "string"},
        "training_seed": {"type": "integer"},
        "camera_id": {"type": "string"},
        "future4": {"type": "boolean"},
        "ratio": {"type": "string"},
        "checkpoint_path": {"type": "string"},
        "checkpoint_fingerprint": {"type": "string"},
        "checkpoint_logged_step": {"type": "integer"},
        "optimizer_updates_completed": {"type": "integer"},
        "success_semantics_id": {"type": "string"},
        "evaluator_provenance": EVALUATOR_PROVENANCE_SCHEMA,
        "evaluator_provenance_sha256": {"type": "string"},
        "selection_seed_manifest_path": {"type": "string"},
        "selection_seed_manifest_sha256": {"type": "string"},
        "selection_equivalence": {
            "type": "object",
            "additionalProperties": False,
            "required": ["use_optimized", "status", "cases_checked"],
            "properties": {
                "use_optimized": {"type": "boolean"},
                "status": {"type": "string"},
                "cases_checked": {"type": "integer"},
            },
        },
        "sampler_mode": {"type": "string"},
        "requested_episode_ids": {"type": "array"},
        "success_count": {"type": "integer"},
        "episode_count": {"type": "integer"},
        "episodes": {"type": "array", "items": SELECTION_EPISODE_SCHEMA},
    },
}


SELECTION_RESULT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "format_version",
        "evidence_kind",
        "usable_for_formal_selection",
        "task",
        "training_seed",
        "camera_id",
        "future4",
        "ratio",
        "checkpoint_path",
        "checkpoint_fingerprint",
        "checkpoint_logged_step",
        "optimizer_updates_completed",
        "success_semantics_id",
        "evaluator_provenance",
        "evaluations",
        "total_success_count",
        "total_episode_count",
        "verified_pooled_success",
        "selection_score",
        "selection_score_type",
    ],
    "properties": {
        "format_version": {"const": 2},
        "evidence_kind": {"const": "verified_selection"},
        "usable_for_formal_selection": {"const": True},
        "task": {"type": "string"},
        "training_seed": {"type": "integer"},
        "camera_id": {"type": "string"},
        "future4": {},
        "ratio": {"type": "string"},
        "checkpoint_path": {"type": "string"},
        "checkpoint_fingerprint": {"type": "string"},
        "checkpoint_logged_step": {"type": "integer"},
        "optimizer_updates_completed": {"type": "integer"},
        "success_semantics_id": {"type": "string"},
        "evaluator_provenance": EVALUATOR_PROVENANCE_SCHEMA,
        "evaluations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "evaluation_id",
                    "evidence_kind",
                    "usable_for_formal_selection",
                    "success_count",
                    "episode_count",
                    "selection_seed_manifest_sha256",
                    "evaluator_provenance_sha256",
                    "evidence_sha256",
                ],
                "properties": {
                    "evaluation_id": {"type": "string"},
                    "evidence_kind": {"const": "verified_selection"},
                    "usable_for_formal_selection": {"const": True},
                    "success_count": {"type": "integer"},
                    "episode_count": {"type": "integer"},
                    "selection_seed_manifest_sha256": {"type": "string"},
                    "evaluator_provenance_sha256": {"type": "string"},
                    "evidence_sha256": {"type": "string"},
                },
            },
        },
        "total_success_count": {"type": "integer"},
        "total_episode_count": {"type": "integer"},
        "verified_pooled_success": {"type": "number"},
        "selection_score": {"type": "number"},
        "selection_score_type": {"const": "verified_pooled_success"},
        "plateau_diagnostic": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "available",
                "episode_weighted_centered_success",
                "used_for_selection",
            ],
            "properties": {
                "available": {},
                "episode_weighted_centered_success": {},
                "used_for_selection": {"const": False},
            },
        },
    },
}


def _closed_object(value: Any, schema: dict, path: str) -> None:
    if not isinstance(value, dict):
        raise SchemaValidationError(f"{path} must be an object")
    required = set(schema.get("required", []))
    missing = sorted(required.difference(value))
    if missing:
        raise SchemaValidationError(f"{path} missing required fields: {missing}")
    allowed = set(schema.get("properties", {}))
    unknown = sorted(set(value).difference(allowed))
    if schema.get("additionalProperties") is False and unknown:
        raise SchemaValidationError(f"{path} has unknown fields: {unknown}")
    for key, child_schema in schema.get("properties", {}).items():
        if key not in value:
            continue
        child = value[key]
        if "const" in child_schema and child != child_schema["const"]:
            raise SchemaValidationError(
                f"{path}.{key} must equal {child_schema['const']!r}"
            )
        if child_schema.get("type") == "object":
            _closed_object(child, child_schema, f"{path}.{key}")
        elif child_schema.get("type") == "array":
            if not isinstance(child, list):
                raise SchemaValidationError(f"{path}.{key} must be an array")
            item_schema = child_schema.get("items")
            if item_schema:
                for index, item in enumerate(child):
                    _closed_object(item, item_schema, f"{path}.{key}[{index}]")
        elif child_schema.get("type") == "string" and not isinstance(child, str):
            raise SchemaValidationError(f"{path}.{key} must be a string")
        elif child_schema.get("type") == "boolean" and not isinstance(child, bool):
            raise SchemaValidationError(f"{path}.{key} must be a boolean")
        elif child_schema.get("type") == "integer" and (
            not isinstance(child, int) or isinstance(child, bool)
        ):
            raise SchemaValidationError(f"{path}.{key} must be an integer")
        elif child_schema.get("type") == "number" and (
            not isinstance(child, (int, float)) or isinstance(child, bool)
        ):
            raise SchemaValidationError(f"{path}.{key} must be a number")


def validate_selection_result(result: dict) -> None:
    """Validate a selection result and its count invariants."""

    _closed_object(result, SELECTION_RESULT_SCHEMA, "selection_result")
    total_success = 0
    total_episodes = 0
    for evaluation in result["evaluations"]:
        success_count = evaluation["success_count"]
        episode_count = evaluation["episode_count"]
        if not isinstance(success_count, int) or not isinstance(episode_count, int):
            raise SchemaValidationError("selection counts must be integers")
        if episode_count <= 0 or success_count < 0 or success_count > episode_count:
            raise SchemaValidationError("invalid selection success/episode counts")
        total_success += success_count
        total_episodes += episode_count
    if total_success != result["total_success_count"]:
        raise SchemaValidationError("total_success_count does not match evaluations")
    if total_episodes != result["total_episode_count"]:
        raise SchemaValidationError("total_episode_count does not match evaluations")
    pooled = total_success / total_episodes
    if result["verified_pooled_success"] != pooled:
        raise SchemaValidationError("verified_pooled_success is not count pooled")
    if result["selection_score"] != pooled:
        raise SchemaValidationError("selection_score is not verified pooled success")


def _validate_sha256(value: object, path: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise SchemaValidationError(f"{path} must be a SHA256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise SchemaValidationError(f"{path} must be a SHA256 hex digest") from exc


def selection_episode_resume_key(row: dict) -> tuple:
    """Return the complete pilot/formal-safe logical identity for one row."""

    return (
        row["checkpoint_fingerprint"],
        row["task"],
        int(row["training_seed"]),
        int(row["selection_episode_id"]),
        row["evaluator_provenance_sha256"],
        row["selection_seed_manifest_sha256"],
        row["evidence_kind"],
    )


def validate_selection_evidence(evidence: dict, seed_manifest: dict | None = None) -> None:
    """Validate raw per-episode evidence and pilot/formal isolation."""

    _closed_object(evidence, SELECTION_EVIDENCE_SCHEMA, "selection_evidence")
    kind = evidence["evidence_kind"]
    expected = {
        "pilot_only": (False, "selection-pilot/v1"),
        "verified_selection": (True, "selection-evaluation/v1"),
    }
    if kind not in expected:
        raise SchemaValidationError(f"unsupported evidence_kind: {kind!r}")
    usable, domain = expected[kind]
    if evidence["usable_for_formal_selection"] is not usable:
        raise SchemaValidationError("evidence usability does not match evidence_kind")
    if evidence["seed_domain"] != domain:
        raise SchemaValidationError("evidence seed domain does not match evidence_kind")
    if evidence["status"] not in {"in_progress", "complete"}:
        raise SchemaValidationError("unsupported selection evidence status")
    if evidence["sampler_mode"] not in {"selection_optimized", "selection_full"}:
        raise SchemaValidationError("unsupported selection sampler mode")
    if evidence["selection_equivalence"]["use_optimized"] != (
        evidence["sampler_mode"] == "selection_optimized"
    ):
        raise SchemaValidationError("sampler mode contradicts equivalence result")
    _validate_sha256(evidence["checkpoint_fingerprint"], "checkpoint_fingerprint")
    _validate_sha256(
        evidence["evaluator_provenance_sha256"],
        "evaluator_provenance_sha256",
    )
    _validate_sha256(
        evidence["selection_seed_manifest_sha256"],
        "selection_seed_manifest_sha256",
    )
    if canonical_sha256(evidence["evaluator_provenance"]) != evidence[
        "evaluator_provenance_sha256"
    ]:
        raise SchemaValidationError("evaluator provenance SHA256 mismatch")
    requested_ids = [int(item) for item in evidence["requested_episode_ids"]]
    if requested_ids != sorted(set(requested_ids)) or not requested_ids:
        raise SchemaValidationError("requested episode IDs must be sorted and unique")
    rows = evidence["episodes"]
    if evidence["episode_count"] != len(rows):
        raise SchemaValidationError("selection episode_count does not match rows")
    if evidence["success_count"] != sum(int(row["success"]) for row in rows):
        raise SchemaValidationError("selection success_count does not match rows")
    row_ids = [int(row["selection_episode_id"]) for row in rows]
    if row_ids != sorted(set(row_ids)) or not set(row_ids).issubset(requested_ids):
        raise SchemaValidationError("selection episode rows are duplicated or unexpected")
    if evidence["status"] == "complete" and row_ids != requested_ids:
        raise SchemaValidationError("complete selection evidence is missing episodes")
    for index, row in enumerate(rows):
        for field in (
            "evidence_kind",
            "task",
            "training_seed",
            "checkpoint_fingerprint",
            "evaluator_provenance_sha256",
            "selection_seed_manifest_sha256",
            "success_semantics_id",
            "sampler_mode",
        ):
            if row[field] != evidence[field]:
                raise SchemaValidationError(
                    f"selection episode {index} disagrees on {field}"
                )
        row_payload = {key: value for key, value in row.items() if key != "episode_row_sha256"}
        if canonical_sha256(row_payload) != row["episode_row_sha256"]:
            raise SchemaValidationError(f"selection episode {index} SHA256 mismatch")
    if seed_manifest is not None:
        from mip.future_rollout_audit.seeding import validate_selection_seed_manifest

        validate_selection_seed_manifest(seed_manifest)
        if seed_manifest["selection_seed_manifest_sha256"] != evidence[
            "selection_seed_manifest_sha256"
        ]:
            raise SchemaValidationError("selection evidence references another seed manifest")
        if seed_manifest["task"] != evidence["task"]:
            raise SchemaValidationError("selection evidence task differs from seed manifest")
        by_id = {
            int(item["selection_episode_id"]): item
            for item in seed_manifest["episodes"]
        }
        for row in rows:
            expected_streams = dict(by_id[int(row["selection_episode_id"])])
            del expected_streams["selection_episode_id"]
            if row["seed_streams"] != expected_streams:
                raise SchemaValidationError("selection episode streams differ from seed manifest")


FORMAL_ENTRY_FIELDS = {
    "task",
    "training_seed",
    "checkpoint_path",
    "checkpoint_fingerprint",
    "selection_score",
    "selection_score_type",
    "success_semantics_id",
    "evaluator_provenance",
    "selection_seed_manifest_sha256",
    "selection_evidence_sha256",
}
SMOKE_ENTRY_FIELDS = {
    "task",
    "training_seed",
    "checkpoint_path",
    "checkpoint_fingerprint",
    "entry_status",
}
COMMON_MANIFEST_FIELDS = {
    "format_version",
    "manifest_kind",
    "status",
    "consumable_by_formal_audit",
    "entry_count",
    "selected_count",
    "source_snapshot_max_mtime_utc",
    "entries",
}


def validate_manifest(manifest: dict) -> None:
    """Validate formal and smoke variants without accepting extra fields."""

    if not isinstance(manifest, dict):
        raise SchemaValidationError("manifest must be an object")
    kind = manifest.get("manifest_kind")
    allowed = set(COMMON_MANIFEST_FIELDS)
    required = set(COMMON_MANIFEST_FIELDS)
    if kind == "smoke_only":
        allowed.add("smoke_checkpoint_count")
        required.add("smoke_checkpoint_count")
        entry_fields = SMOKE_ENTRY_FIELDS
    elif kind == "formal_selection":
        entry_fields = FORMAL_ENTRY_FIELDS
    else:
        raise SchemaValidationError(f"unsupported manifest_kind: {kind!r}")
    missing = sorted(required.difference(manifest))
    unknown = sorted(set(manifest).difference(allowed))
    if missing or unknown:
        raise SchemaValidationError(
            f"manifest closed-schema violation missing={missing} unknown={unknown}"
        )
    if manifest["format_version"] != 1:
        raise SchemaValidationError("unsupported manifest format_version")
    entries = manifest["entries"]
    if not isinstance(entries, list) or manifest["entry_count"] != len(entries):
        raise SchemaValidationError("entry_count does not match entries")
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != entry_fields:
            raise SchemaValidationError(
                f"manifest entry {index} does not satisfy its closed schema"
            )
    if kind == "smoke_only":
        if manifest["selected_count"] != 0:
            raise SchemaValidationError("smoke selected_count must be zero")
        if manifest["smoke_checkpoint_count"] != len(entries):
            raise SchemaValidationError("smoke checkpoint count mismatch")
        if manifest["consumable_by_formal_audit"]:
            raise SchemaValidationError("smoke manifest cannot be formal-consumable")


def validate_formal_manifest_for_audit(manifest: dict) -> None:
    """Enforce the exact formal-audit consumption gate."""

    validate_manifest(manifest)
    required = {
        "manifest_kind": "formal_selection",
        "status": "ready",
        "consumable_by_formal_audit": True,
        "entry_count": 24,
        "selected_count": 24,
    }
    mismatches = {
        key: {"expected": expected, "actual": manifest.get(key)}
        for key, expected in required.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise SchemaValidationError(f"formal manifest is not consumable: {mismatches}")
    for entry in manifest["entries"]:
        if entry["selection_score_type"] != "verified_pooled_success":
            raise SchemaValidationError("formal entry lacks verified selection score")
        if not entry["success_semantics_id"] or not entry["evaluator_provenance"]:
            raise SchemaValidationError("formal entry lacks verified provenance")
    expected_pairs = {
        (task, seed) for task in FORMAL_TASKS for seed in FORMAL_TRAINING_SEEDS
    }
    actual_pairs = [
        (entry["task"], int(entry["training_seed"]))
        for entry in manifest["entries"]
    ]
    if len(set(actual_pairs)) != len(actual_pairs) or set(actual_pairs) != expected_pairs:
        raise SchemaValidationError(
            "formal manifest must contain exactly one verified checkpoint for "
            "each task x training_seed pair"
        )


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    """Hash canonical JSON without allowing self-referential fields."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def write_canonical_json(path: str | Path, value: object) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical_json_bytes(value))
    return destination


def write_canonical_json_atomic(path: str | Path, value: object) -> Path:
    """Atomically replace a canonical JSON artifact on the same filesystem."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return destination
