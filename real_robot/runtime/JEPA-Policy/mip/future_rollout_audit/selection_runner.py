"""Real-environment selection evaluator with no future-audit data path."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from omegaconf import OmegaConf
import torch

from mip.agent import TrainingAgent
from mip.envs.persistent_image_rollout import PersistentImageRolloutPool
from mip.future_rollout_audit.provenance import (
    build_evaluator_provenance,
    file_sha256,
)
from mip.future_rollout_audit.schemas import (
    canonical_sha256,
    validate_selection_evidence,
    write_canonical_json_atomic,
)
from mip.future_rollout_audit.seeding import (
    EpisodeSeeds,
    preserve_rng_state,
    seed_decision_process_streams,
    validate_selection_seed_manifest,
)
from mip.future_rollout_audit.selection import (
    expected_success_semantics,
    merge_episode_rows,
    verify_selection_action_equivalence,
)


EVIDENCE_CONFIGURATION = {
    "pilot_only": {
        "usable_for_formal_selection": False,
        "seed_domain": "selection-pilot/v1",
    },
    "verified_selection": {
        "usable_for_formal_selection": True,
        "seed_domain": "selection-evaluation/v1",
    },
}
TERMINATION_REASONS = {
    "success",
    "environment_terminated",
    "environment_truncated",
    "max_episode_steps",
}


def _canonical_task(env_name: str) -> str:
    return "kitchen" if env_name == "mimicgen_kitchen" else env_name


def _normalize_observation(raw: dict, dataset, device: str) -> dict[str, torch.Tensor]:
    output = {}
    for key, value in raw.items():
        array = np.asarray(value, dtype=np.float32)
        normalized = dataset.normalizer["obs"][key].normalize(array)
        output[key] = torch.as_tensor(normalized, device=device, dtype=torch.float32)
    return output


def _policy_action(config, dataset, normalized_action: torch.Tensor) -> np.ndarray:
    action = dataset.normalizer["action"].unnormalize(
        normalized_action.detach().cpu().numpy()
    )
    start = int(config.task.obs_steps) - 1
    end = start + int(config.task.act_steps)
    action = action[:, start:end, :]
    if bool(config.task.abs_action) and config.task.env_name in {
        "can",
        "lift",
        "square",
        "tool_hang",
        "transport",
    }:
        action = dataset.undo_transform_action(action)
    return action


def _success_from_predicate(info: dict) -> bool:
    if not isinstance(info, dict) or "success" not in info:
        raise RuntimeError(
            "env_task_predicate_v1 requires wrapper-provided info['success']"
        )
    return bool(np.asarray(info["success"], dtype=bool).reshape(-1).any())


def evaluate_step_success(
    *,
    task: str,
    env_type: str,
    reward: float,
    info: dict,
) -> tuple[bool, str]:
    """Apply the task whitelist's repaired success definition exactly."""

    semantics = expected_success_semantics(task)
    if semantics == "positive_sparse_reward_v1":
        if env_type != "libero":
            raise RuntimeError(
                "positive_sparse_reward_v1 is only valid for LIBERO tasks"
            )
        return bool(float(reward) > 0.0), semantics
    if semantics == "env_task_predicate_v1":
        if env_type == "libero":
            raise RuntimeError("LIBERO may not use env_task_predicate_v1")
        return _success_from_predicate(info), semantics
    raise RuntimeError(f"No evaluator implementation for {semantics!r}")


def _preflight_fixtures(dataset, config) -> list[tuple[torch.Tensor, dict]]:
    sample = dataset[0]
    if "obs" not in sample or not isinstance(sample["obs"], dict):
        raise RuntimeError("selection preflight requires a structured dataset observation")
    fixtures = []
    for batch_size in (1, 2, 8):
        obs = {
            key: value.unsqueeze(0)
            .repeat(batch_size, *([1] * value.ndim))
            .to(device=config.optimization.device, dtype=torch.float32)
            for key, value in sample["obs"].items()
        }
        act_0 = torch.zeros(
            (batch_size, config.task.horizon, config.task.act_dim),
            device=config.optimization.device,
            dtype=torch.float32,
        )
        fixtures.append((act_0, obs))
    return fixtures


def _load_components(
    config_path: str | Path,
    checkpoint_path: str | Path,
    *,
    worker_count: int = 8,
):
    from mip.datasets.robot_dataset import make_dataset

    config = OmegaConf.load(config_path)
    config.optimization.future_target_type = getattr(
        config.task,
        "future_target_type",
        config.optimization.future_target_type,
    )
    config.optimization.use_compile = False
    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    config.eval.parallel_rollout_workers = int(worker_count)
    config.task.num_envs = 1
    if config.task.obs_type == "image":
        config.task.obs_dim = config.network.emb_dim
    dataset = make_dataset(config.task)
    agent = TrainingAgent(config)
    agent.load(str(checkpoint_path), load_optimizer=False)
    agent.eval()
    pool = PersistentImageRolloutPool(config, lazy=True)
    return config, dataset, agent, pool


def _base_evidence(
    *,
    candidate: dict,
    config,
    evidence_kind: str,
    seed_manifest: dict,
    seed_manifest_path: str | Path,
    evaluator_provenance: dict,
    equivalence,
) -> dict:
    settings = EVIDENCE_CONFIGURATION[evidence_kind]
    sampler_mode = (
        "selection_optimized" if equivalence.use_optimized else "selection_full"
    )
    task = _canonical_task(str(config.task.env_name))
    return {
        "format_version": 1,
        "record_kind": "selection_episode_evidence",
        "status": "in_progress",
        "evidence_kind": evidence_kind,
        "usable_for_formal_selection": settings["usable_for_formal_selection"],
        "seed_domain": settings["seed_domain"],
        "task": task,
        "training_seed": int(config.optimization.seed),
        "camera_id": str(candidate.get("camera_id", "full")),
        "future4": bool(candidate["future4"]),
        "ratio": str(candidate["ratio"]),
        "checkpoint_path": str(Path(candidate["checkpoint_path"]).resolve()),
        "checkpoint_fingerprint": candidate["checkpoint_fingerprint"],
        "checkpoint_logged_step": int(candidate["checkpoint_logged_step"]),
        "optimizer_updates_completed": int(
            candidate["optimizer_updates_completed"]
        ),
        "success_semantics_id": expected_success_semantics(task),
        "evaluator_provenance": deepcopy(evaluator_provenance),
        "evaluator_provenance_sha256": canonical_sha256(evaluator_provenance),
        "selection_seed_manifest_path": str(Path(seed_manifest_path).resolve()),
        "selection_seed_manifest_sha256": seed_manifest[
            "selection_seed_manifest_sha256"
        ],
        "selection_equivalence": equivalence.to_dict(),
        "sampler_mode": sampler_mode,
        "requested_episode_ids": list(seed_manifest["episode_ids"]),
        "success_count": 0,
        "episode_count": 0,
        "episodes": [],
    }


def _row(
    evidence: dict,
    seed_record: dict,
    *,
    success: bool,
    termination_reason: str,
    reward_sum: float,
    policy_decision_count: int,
    environment_steps: int,
) -> dict:
    if termination_reason not in TERMINATION_REASONS:
        raise ValueError(f"Unsupported termination reason: {termination_reason!r}")
    seed_streams = dict(seed_record)
    episode_id = int(seed_streams.pop("selection_episode_id"))
    row = {
        "evidence_kind": evidence["evidence_kind"],
        "task": evidence["task"],
        "training_seed": evidence["training_seed"],
        "selection_episode_id": episode_id,
        "checkpoint_fingerprint": evidence["checkpoint_fingerprint"],
        "evaluator_provenance_sha256": evidence["evaluator_provenance_sha256"],
        "selection_seed_manifest_sha256": evidence[
            "selection_seed_manifest_sha256"
        ],
        "seed_streams": seed_streams,
        "success": bool(success),
        "success_semantics_id": evidence["success_semantics_id"],
        "termination_reason": termination_reason,
        "reward_sum": float(reward_sum),
        "policy_decision_count": int(policy_decision_count),
        "environment_steps": int(environment_steps),
        "sampler_mode": evidence["sampler_mode"],
    }
    row["episode_row_sha256"] = canonical_sha256(row)
    return row


def _load_resume(path: Path, expected: dict, seed_manifest: dict) -> dict:
    if not path.exists():
        return expected
    with path.open("r", encoding="utf-8") as handle:
        existing = json.load(handle)
    validate_selection_evidence(existing, seed_manifest)
    immutable = set(expected).difference(
        {"status", "success_count", "episode_count", "episodes"}
    )
    mismatches = sorted(key for key in immutable if existing[key] != expected[key])
    if mismatches:
        raise RuntimeError(f"selection_resume_header_conflict: {mismatches}")
    return existing


def _save_evidence(path: Path, evidence: dict, seed_manifest: dict) -> None:
    evidence["episodes"] = merge_episode_rows(evidence["episodes"], [])
    evidence["episode_count"] = len(evidence["episodes"])
    evidence["success_count"] = sum(
        int(item["success"]) for item in evidence["episodes"]
    )
    evidence["status"] = (
        "complete"
        if [item["selection_episode_id"] for item in evidence["episodes"]]
        == evidence["requested_episode_ids"]
        else "in_progress"
    )
    validate_selection_evidence(evidence, seed_manifest)
    write_canonical_json_atomic(path, evidence)


def run_selection_checkpoint(
    *,
    repo_root: str | Path,
    candidate: dict,
    seed_manifest: dict,
    seed_manifest_path: str | Path,
    environment_provenance: dict,
    output_path: str | Path,
    evidence_kind: str,
    worker_count: int = 8,
) -> dict:
    """Evaluate one checkpoint without importing or invoking audit capture/scoring."""

    if evidence_kind not in EVIDENCE_CONFIGURATION:
        raise ValueError(f"Unsupported evidence_kind: {evidence_kind!r}")
    validate_selection_seed_manifest(seed_manifest)
    settings = EVIDENCE_CONFIGURATION[evidence_kind]
    if seed_manifest["seed_domain"] != settings["seed_domain"]:
        raise ValueError("seed manifest domain does not match evidence kind")
    checkpoint_path = Path(candidate["checkpoint_path"]).resolve()
    if candidate["checkpoint_fingerprint_kind"] != "sha256":
        raise ValueError("selection requires a content SHA256 checkpoint fingerprint")
    if file_sha256(checkpoint_path) != candidate["checkpoint_fingerprint"]:
        raise RuntimeError("checkpoint fingerprint changed after inventory")
    config_path = checkpoint_path.parent.parent / "resolved_config.yaml"
    evaluator_provenance = build_evaluator_provenance(
        repo_root,
        config_path,
        environment_provenance,
    )
    config, dataset, agent, pool = _load_components(
        config_path,
        checkpoint_path,
        worker_count=worker_count,
    )
    task = _canonical_task(str(config.task.env_name))
    if task != candidate["task"] or task != seed_manifest["task"]:
        pool.close()
        raise RuntimeError("candidate, config and seed-manifest tasks disagree")
    if int(config.optimization.seed) != int(candidate["training_seed"]):
        pool.close()
        raise RuntimeError("checkpoint config training seed differs from inventory")
    equivalence = verify_selection_action_equivalence(
        agent,
        _preflight_fixtures(dataset, config),
        seeds=(3, 17),
    )
    expected = _base_evidence(
        candidate=candidate,
        config=config,
        evidence_kind=evidence_kind,
        seed_manifest=seed_manifest,
        seed_manifest_path=seed_manifest_path,
        evaluator_provenance=evaluator_provenance,
        equivalence=equivalence,
    )
    output = Path(output_path)
    evidence = _load_resume(output, expected, seed_manifest)
    existing_ids = {
        int(item["selection_episode_id"]) for item in evidence["episodes"]
    }
    missing_records = [
        item
        for item in seed_manifest["episodes"]
        if int(item["selection_episode_id"]) not in existing_ids
    ]
    try:
        for start in range(0, len(missing_records), pool.num_workers):
            batch = missing_records[start : start + pool.num_workers]
            worker_indices = list(range(len(batch)))
            reset_results = pool.reset(
                [int(item["env"]) for item in batch],
                worker_indices=worker_indices,
            )
            observations = [item[0] for item in reset_results]
            done = [False] * len(batch)
            succeeded = [False] * len(batch)
            rewards = [0.0] * len(batch)
            decisions = [0] * len(batch)
            environment_steps = [0] * len(batch)
            termination_reasons = [None] * len(batch)
            while not all(done):
                active = [index for index, value in enumerate(done) if not value]
                actions = []
                for local_index in active:
                    decision_id = decisions[local_index]
                    policy_obs = _normalize_observation(
                        observations[local_index], dataset, config.optimization.device
                    )
                    act_0 = torch.zeros(
                        (1, config.task.horizon, config.task.act_dim),
                        device=config.optimization.device,
                        dtype=torch.float32,
                    )
                    seed_streams = dict(batch[local_index])
                    seed_streams.pop("selection_episode_id")
                    with preserve_rng_state():
                        seed_decision_process_streams(
                            EpisodeSeeds(**seed_streams), decision_id
                        )
                        sampled = agent.sample_joint(
                            act_0,
                            policy_obs,
                            sampler_invocation_id=(
                                f"selection:{evidence_kind}:{task}:"
                                f"seed{config.optimization.seed}:episode"
                                f"{batch[local_index]['selection_episode_id']}:"
                                f"decision{decision_id}"
                            ),
                            mode=evidence["sampler_mode"],
                        )
                    if sampled.future_pred_1 is not None:
                        raise RuntimeError("selection sampler exposed a future prediction")
                    actions.append(_policy_action(config, dataset, sampled.action))
                step_results = pool.step_with_metadata(
                    actions,
                    worker_indices=active,
                )
                for position, local_index in enumerate(active):
                    observation, reward, terminated, truncated, info, metadata = (
                        step_results[position]
                    )
                    observations[local_index] = observation
                    reward_value = float(np.asarray(reward).reshape(-1)[0])
                    rewards[local_index] += reward_value
                    decisions[local_index] += 1
                    environment_steps[local_index] += int(
                        metadata["primitive_actions_completed"]
                    )
                    step_success, semantics = evaluate_step_success(
                        task=task,
                        env_type=str(config.task.env_type),
                        reward=reward_value,
                        info=info,
                    )
                    if semantics != evidence["success_semantics_id"]:
                        raise RuntimeError("runtime success semantics changed")
                    succeeded[local_index] |= step_success
                    terminated_value = bool(
                        np.asarray(terminated, dtype=bool).reshape(-1)[0]
                    )
                    truncated_value = bool(
                        np.asarray(truncated, dtype=bool).reshape(-1)[0]
                    )
                    reached_limit = environment_steps[local_index] >= int(
                        config.task.max_episode_steps
                    )
                    done[local_index] = bool(
                        step_success or terminated_value or truncated_value or reached_limit
                    )
                    if done[local_index]:
                        if step_success:
                            reason = "success"
                        elif reached_limit:
                            reason = "max_episode_steps"
                        elif truncated_value:
                            reason = "environment_truncated"
                        else:
                            reason = "environment_terminated"
                        termination_reasons[local_index] = reason
            new_rows = [
                _row(
                    evidence,
                    seed_record,
                    success=succeeded[index],
                    termination_reason=str(termination_reasons[index]),
                    reward_sum=rewards[index],
                    policy_decision_count=decisions[index],
                    environment_steps=environment_steps[index],
                )
                for index, seed_record in enumerate(batch)
            ]
            evidence["episodes"] = merge_episode_rows(
                evidence["episodes"], new_rows
            )
            _save_evidence(output, evidence, seed_manifest)
    finally:
        pool.close()
    _save_evidence(output, evidence, seed_manifest)
    return evidence
