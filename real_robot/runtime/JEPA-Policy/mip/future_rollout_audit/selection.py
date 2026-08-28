"""Verified checkpoint selection without audit-metric leakage."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
from typing import Iterable

import numpy as np
import torch

from mip.future_rollout_audit.schemas import (
    canonical_sha256,
    selection_episode_resume_key,
    validate_selection_evidence,
    validate_selection_result,
)
from mip.future_rollout_audit.seeding import preserve_rng_state


SUCCESS_SEMANTICS_WHITELIST = {
    "mug_mug": "positive_sparse_reward_v1",
    "moka_moka": "positive_sparse_reward_v1",
    "square": "env_task_predicate_v1",
    "tool_hang": "env_task_predicate_v1",
    "transport": "env_task_predicate_v1",
    "coffee_preparation": "env_task_predicate_v1",
    "kitchen": "env_task_predicate_v1",
    "three_piece_assembly": "env_task_predicate_v1",
}
LEGACY_REJECTION = "legacy_or_unverified_success_semantics"
BOOTSTRAP_METHOD = "paired_percentile"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_SEED_DOMAIN = "selection-stopping-bootstrap/v1"


def expected_success_semantics(task: str) -> str:
    try:
        return SUCCESS_SEMANTICS_WHITELIST[task]
    except KeyError as exc:
        raise ValueError(f"No verified success semantics for task {task!r}") from exc


def pool_verified_selection(
    *,
    task: str,
    training_seed: int,
    camera_id: str,
    future4: bool,
    ratio: str,
    checkpoint_path: str,
    checkpoint_fingerprint: str,
    checkpoint_logged_step: int,
    optimizer_updates_completed: int,
    success_semantics_id: str,
    evaluator_provenance: dict,
    evaluations: Iterable[dict],
) -> dict:
    """Pool counts for one checkpoint and produce a closed result record."""

    expected = expected_success_semantics(task)
    if success_semantics_id != expected:
        raise ValueError(LEGACY_REJECTION)
    ordered = sorted(
        (deepcopy(item) for item in evaluations),
        key=lambda item: item["evaluation_id"],
    )
    total_success = sum(int(item["success_count"]) for item in ordered)
    total_episodes = sum(int(item["episode_count"]) for item in ordered)
    if total_episodes <= 0:
        raise ValueError("verified selection requires at least one episode")
    pooled = total_success / total_episodes
    for item in ordered:
        if item.get("evidence_kind") != "verified_selection" or item.get(
            "usable_for_formal_selection"
        ) is not True:
            raise ValueError("pilot evidence cannot be pooled for formal selection")
        if item.get("evaluator_provenance_sha256") != canonical_sha256(
            evaluator_provenance
        ):
            raise ValueError("evaluation provenance differs from pooled provenance")
    result = {
        "format_version": 2,
        "evidence_kind": "verified_selection",
        "usable_for_formal_selection": True,
        "task": task,
        "training_seed": int(training_seed),
        "camera_id": str(camera_id),
        "future4": bool(future4),
        "ratio": str(ratio),
        "checkpoint_path": checkpoint_path,
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "checkpoint_logged_step": int(checkpoint_logged_step),
        "optimizer_updates_completed": int(optimizer_updates_completed),
        "success_semantics_id": success_semantics_id,
        "evaluator_provenance": deepcopy(evaluator_provenance),
        "evaluations": ordered,
        "total_success_count": total_success,
        "total_episode_count": total_episodes,
        "verified_pooled_success": pooled,
        "selection_score": pooled,
        "selection_score_type": "verified_pooled_success",
    }
    validate_selection_result(result)
    return result


def merge_episode_rows(existing: Iterable[dict], incoming: Iterable[dict]) -> list[dict]:
    """Resume idempotently and fail closed on conflicting logical episodes."""

    merged: dict[tuple, dict] = {}
    for row in [*map(deepcopy, existing), *map(deepcopy, incoming)]:
        key = selection_episode_resume_key(row)
        previous = merged.get(key)
        if previous is not None and previous != row:
            raise ValueError(f"selection_resume_conflict: {key!r}")
        merged[key] = row
    return sorted(
        merged.values(),
        key=lambda row: (
            int(row["selection_episode_id"]),
            selection_episode_resume_key(row),
        ),
    )


def _paired_rows(left: dict, right: dict) -> tuple[np.ndarray, np.ndarray, list[int]]:
    validate_selection_evidence(left)
    validate_selection_evidence(right)
    for evidence in (left, right):
        if evidence["evidence_kind"] != "verified_selection" or not evidence[
            "usable_for_formal_selection"
        ]:
            raise ValueError("paired stopping diagnostics require verified evidence")
        if evidence["status"] != "complete":
            raise ValueError("paired stopping diagnostics require complete evidence")
    for field in (
        "task",
        "training_seed",
        "selection_seed_manifest_sha256",
        "success_semantics_id",
    ):
        if left[field] != right[field]:
            raise ValueError(f"paired evidence differs on {field}")
    left_by_id = {
        int(row["selection_episode_id"]): bool(row["success"])
        for row in left["episodes"]
    }
    right_by_id = {
        int(row["selection_episode_id"]): bool(row["success"])
        for row in right["episodes"]
    }
    if left_by_id.keys() != right_by_id.keys():
        raise ValueError("paired candidates must contain identical episode IDs")
    episode_ids = sorted(left_by_id)
    return (
        np.asarray([left_by_id[item] for item in episode_ids], dtype=np.float64),
        np.asarray([right_by_id[item] for item in episode_ids], dtype=np.float64),
        episode_ids,
    )


def paired_bootstrap_interval(
    left: dict,
    right: dict,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    confidence_level: float = BOOTSTRAP_CONFIDENCE_LEVEL,
) -> dict:
    """Bootstrap paired success differences by shared selection episode ID."""

    if resamples != BOOTSTRAP_RESAMPLES or confidence_level != BOOTSTRAP_CONFIDENCE_LEVEL:
        raise ValueError("paired bootstrap settings are preregistered and immutable")
    left_values, right_values, episode_ids = _paired_rows(left, right)
    differences = left_values - right_values
    seed_payload = (
        BOOTSTRAP_SEED_DOMAIN,
        left["task"],
        int(left["training_seed"]),
        len(episode_ids),
        left["checkpoint_fingerprint"],
        right["checkpoint_fingerprint"],
        left["selection_seed_manifest_sha256"],
    )
    digest = hashlib.sha256(
        b"\0".join(str(item).encode("utf-8") for item in seed_payload)
    ).digest()
    bootstrap_seed = int.from_bytes(digest[:8], "big", signed=False)
    generator = np.random.default_rng(bootstrap_seed)
    indices = generator.integers(
        0,
        len(episode_ids),
        size=(resamples, len(episode_ids)),
        endpoint=False,
    )
    estimates = differences[indices].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(estimates, [alpha, 1.0 - alpha], method="linear")
    return {
        "method": BOOTSTRAP_METHOD,
        "bootstrap_unit": "selection_episode_id",
        "resampling": "paired",
        "candidate_episode_rows_resampled_together": True,
        "resamples": resamples,
        "confidence_level": confidence_level,
        "seed_domain": BOOTSTRAP_SEED_DOMAIN,
        "bootstrap_seed": bootstrap_seed,
        "episode_count": len(episode_ids),
        "observed_success_difference": float(differences.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "contains_zero": bool(lower <= 0.0 <= upper),
        "used_for_ranking": False,
        "used_only_for_episode_extension": True,
    }


def group_extension_decision(evidences: Iterable[dict]) -> dict:
    """Apply the frozen 32 -> 64 -> 128 whole-group extension rule."""

    group = [deepcopy(item) for item in evidences]
    if len(group) < 2:
        raise ValueError("extension decisions require at least two candidates")
    for item in group:
        validate_selection_evidence(item)
        if item["evidence_kind"] != "verified_selection":
            raise ValueError("pilot evidence cannot drive formal episode extension")
    group_key = (group[0]["task"], int(group[0]["training_seed"]))
    episode_ids = group[0]["requested_episode_ids"]
    for item in group:
        if (item["task"], int(item["training_seed"])) != group_key:
            raise ValueError("extension candidates must share task and training seed")
        if item["status"] != "complete" or item["requested_episode_ids"] != episode_ids:
            raise ValueError("every candidate must complete the same episode prefix")
        if item["selection_seed_manifest_sha256"] != group[0][
            "selection_seed_manifest_sha256"
        ]:
            raise ValueError("extension candidates must share one seed manifest")
    count = len(episode_ids)
    if count not in {32, 64, 128}:
        raise ValueError("formal extension decisions are defined only at 32/64/128")
    ranked = sorted(
        group,
        key=lambda item: (
            -float(item["success_count"] / item["episode_count"]),
            -int(item["episode_count"]),
            int(item["optimizer_updates_completed"]),
            str(item["checkpoint_path"]),
        ),
    )
    diagnostic = paired_bootstrap_interval(ranked[0], ranked[1])
    success_count_difference = abs(
        int(ranked[0]["success_count"]) - int(ranked[1]["success_count"])
    )
    if count == 32:
        extend = success_count_difference <= 2 or diagnostic["contains_zero"]
        next_count = 64 if extend else None
    elif count == 64:
        extend = diagnostic["contains_zero"]
        next_count = 128 if extend else None
    else:
        extend = False
        next_count = None
    return {
        "format_version": 1,
        "task": group_key[0],
        "training_seed": group_key[1],
        "current_episode_count": count,
        "extend_entire_group": extend,
        "next_episode_count": next_count,
        "group_candidate_count": len(group),
        "group_checkpoint_fingerprints": sorted(
            item["checkpoint_fingerprint"] for item in group
        ),
        "top_checkpoint_fingerprint": ranked[0]["checkpoint_fingerprint"],
        "second_checkpoint_fingerprint": ranked[1]["checkpoint_fingerprint"],
        "top_two_success_count_difference": success_count_difference,
        "paired_bootstrap": diagnostic,
        "used_for_ranking": False,
    }


def rank_checkpoints(records: Iterable[dict]) -> list[dict]:
    """Apply the frozen formal ranking with no future-loss inputs."""

    checked = []
    for record in records:
        validate_selection_result(record)
        if "optimizer_updates_completed" not in record:
            raise ValueError("ranking record lacks optimizer_updates_completed")
        checked.append(record)
    return sorted(
        checked,
        key=lambda record: (
            -float(record["selection_score"]),
            -int(record["total_episode_count"]),
            int(record["optimizer_updates_completed"]),
            str(record["checkpoint_path"]),
        ),
    )


def add_centered_plateau_diagnostics(records: Iterable[dict]) -> list[dict]:
    """Attach episode-weighted three-point diagnostics without changing rank."""

    output = [deepcopy(record) for record in records]
    grouped: dict[tuple, list[dict]] = {}
    group_fields = ("task", "training_seed", "camera_id", "future4", "ratio")
    for record in output:
        key = tuple(record[field] for field in group_fields)
        grouped.setdefault(key, []).append(record)
    for group in grouped.values():
        group.sort(key=lambda record: int(record["optimizer_updates_completed"]))
        for index, record in enumerate(group):
            diagnostic = {
                "available": False,
                "episode_weighted_centered_success": None,
                "used_for_selection": False,
            }
            if 0 < index < len(group) - 1:
                window = group[index - 1 : index + 2]
                successes = sum(int(item["total_success_count"]) for item in window)
                episodes = sum(int(item["total_episode_count"]) for item in window)
                diagnostic.update(
                    {
                        "available": True,
                        "episode_weighted_centered_success": successes / episodes,
                    }
                )
            record["plateau_diagnostic"] = diagnostic
    return output


@dataclass(frozen=True)
class SelectionEquivalence:
    use_optimized: bool
    status: str
    cases_checked: int

    def to_dict(self) -> dict:
        return asdict(self)


def _rng_equal(left, right) -> bool:
    if not torch.equal(left[0], right[0]):
        return False
    if len(left[1]) != len(right[1]):
        return False
    return all(torch.equal(a, b) for a, b in zip(left[1], right[1], strict=True))


def verify_selection_action_equivalence(
    agent,
    fixtures: Iterable[tuple[torch.Tensor, object]],
    seeds: Iterable[int],
) -> SelectionEquivalence:
    """Prove optimized actions/RNG/module buffers equal the full sampler."""

    cases = 0
    modules = (agent.encoder_ema, agent.flow_map_ema)
    before_state = {
        f"{module_index}:{name}": value.detach().cpu().clone()
        for module_index, module in enumerate(modules)
        for name, value in module.state_dict().items()
    }
    before_training = tuple(module.training for module in modules)
    with preserve_rng_state():
        for fixture_index, (act_0, obs) in enumerate(fixtures):
            for seed in seeds:
                cases += 1
                torch.manual_seed(int(seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(seed))
                cpu_before = torch.random.get_rng_state()
                cuda_before = (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                )
                full = agent.sample_joint(
                    act_0,
                    obs,
                    sampler_invocation_id=f"preflight-full-{fixture_index}-{seed}",
                    mode="selection_full",
                )
                full_rng = (
                    torch.random.get_rng_state(),
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                )
                torch.random.set_rng_state(cpu_before)
                if torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(cuda_before)
                optimized = agent.sample_joint(
                    act_0,
                    obs,
                    sampler_invocation_id=f"preflight-optimized-{fixture_index}-{seed}",
                    mode="selection_optimized",
                )
                optimized_rng = (
                    torch.random.get_rng_state(),
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                )
                if not torch.equal(full.action, optimized.action) or not _rng_equal(
                    full_rng, optimized_rng
                ):
                    return SelectionEquivalence(
                        False, "selection_action_only_equivalence_failed", cases
                    )
    after_state = {
        f"{module_index}:{name}": value.detach().cpu()
        for module_index, module in enumerate(modules)
        for name, value in module.state_dict().items()
    }
    if (
        before_state.keys() != after_state.keys()
        or tuple(module.training for module in modules) != before_training
        or any(
            not torch.equal(before_state[key], after_state[key])
            for key in before_state
        )
    ):
        return SelectionEquivalence(
            False, "selection_action_only_equivalence_failed", cases
        )
    return SelectionEquivalence(True, "verified_bitwise_equal", cases)
