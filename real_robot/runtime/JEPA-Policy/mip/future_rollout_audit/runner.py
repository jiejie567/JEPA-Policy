"""Future Rollout Audit decision processing and smoke runner."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from mip.agent import TrainingAgent
from mip.envs.persistent_image_rollout import PersistentImageRolloutPool
from mip.future_rollout_audit.contracts import (
    FutureSpaceContractError,
    build_future_space_contract,
    encode_audit_future_target,
    validate_future_space_contract,
)
from mip.future_rollout_audit.schemas import validate_formal_manifest_for_audit
from mip.future_rollout_audit.seeding import (
    DEFAULT_AUDIT_SEED_DOMAIN,
    audit_episode_seeds,
    preserve_rng_state,
    seed_decision_process_streams,
)


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor dtype, shape, and owning contiguous CPU bytes."""

    value = tensor.detach().to(device="cpu", copy=True).contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).removeprefix("torch.").encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(dim) for dim in value.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class PredictionRecord:
    decision_id: str
    sampler_invocation_id: str
    pass_0_call_count: int
    pass_1_call_count: int
    extra_sampler_call_count: int
    future_pred_shape: tuple[int, ...]
    future_pred_dtype: str
    future_pred_sha256: str
    future_pred: torch.Tensor


def prediction_record(sample_result, decision_id: str) -> PredictionRecord:
    """Validate and retain an independent copy of one sampler prediction."""

    prediction = sample_result.future_pred_1
    if prediction is None:
        raise RuntimeError("Audit sampler returned no future_pred_1")
    saved = prediction.detach().to(device="cpu", copy=True).clone().contiguous()
    return PredictionRecord(
        decision_id=str(decision_id),
        sampler_invocation_id=sample_result.trace.sampler_invocation_id,
        pass_0_call_count=int(sample_result.trace.pass_0_call_count),
        pass_1_call_count=int(sample_result.trace.pass_1_call_count),
        extra_sampler_call_count=int(
            sample_result.trace.extra_sampler_call_count
        ),
        future_pred_shape=tuple(saved.shape),
        future_pred_dtype=str(saved.dtype).removeprefix("torch."),
        future_pred_sha256=tensor_sha256(saved),
        future_pred=saved,
    )


def compute_future_metrics(
    agent,
    prediction: torch.Tensor,
    target: torch.Tensor,
    contract: dict,
) -> dict[str, float]:
    """Compute metrics only after exact contract validation."""

    validate_future_space_contract(contract, agent, prediction, target)
    pred64 = prediction.to(torch.float64)
    target64 = target.to(torch.float64)
    error = pred64 - target64
    mse = error.square().mean()
    target_scale = target64.flatten(1).square().mean(dim=1).sqrt().clamp_min(1e-6)
    scale = target_scale.view(-1, *([1] * (target64.ndim - 1)))
    nmse = (error / scale).square().mean()
    interval = 1.0 - float(agent.config.optimization.t_two_step)
    raw_loss = (error / scale / interval).square().mean()
    pred_flat = pred64.flatten(1)
    target_flat = target64.flatten(1)
    cosine = torch.nn.functional.cosine_similarity(
        pred_flat, target_flat, dim=1, eps=1e-12
    ).mean()
    return {
        "mse": float(mse),
        "nmse": float(nmse),
        "cosine": float(cosine),
        "raw_loss": float(raw_loss),
    }


def primary_episode_score(decisions: list[dict], required_decisions: int = 5) -> dict:
    """Enforce the five-valid-target primary eligibility gate."""

    prefix = decisions[:required_decisions]
    eligible = len(prefix) == required_decisions and all(
        item.get("target_valid", False) and "metrics" in item for item in prefix
    )
    if not eligible:
        return {
            "primary_eligible": False,
            "primary_score": None,
            "insufficient_target": True,
        }
    values = np.asarray([item["metrics"]["nmse"] for item in prefix], dtype=np.float64)
    return {
        "primary_eligible": True,
        "primary_score": float(np.median(values)),
        "insufficient_target": False,
    }


def binary_auroc(labels: list[bool], scores: list[float]) -> float | None:
    """Compute tie-aware AUROC, returning None without both outcome classes."""

    if len(labels) != len(scores):
        raise ValueError("AUROC labels and scores must have equal length")
    positives = sum(bool(label) for label in labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(np.asarray(scores, dtype=np.float64), kind="mergesort")
    sorted_scores = np.asarray(scores, dtype=np.float64)[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = sum(
        rank for rank, label in zip(ranks, labels, strict=True) if label
    )
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def require_formal_manifest(manifest: dict) -> None:
    validate_formal_manifest_for_audit(manifest)


def _normalize_observation(raw: dict, dataset, device: str) -> dict[str, torch.Tensor]:
    output = {}
    for key, value in raw.items():
        array = np.asarray(value, dtype=np.float32)
        normalized = dataset.normalizer["obs"][key].normalize(array)
        output[key] = torch.as_tensor(
            normalized, device=device, dtype=torch.float32
        )
    return output


def _success_from_info(info) -> bool:
    if not isinstance(info, dict) or "success" not in info:
        return False
    return bool(np.asarray(info["success"], dtype=bool).reshape(-1).any())


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


def _load_smoke_components(config_path: str | Path, checkpoint_path: str | Path):
    from mip.datasets.robot_dataset import make_dataset

    config = OmegaConf.load(config_path)
    config.optimization.future_target_type = getattr(
        config.task,
        "future_target_type",
        config.optimization.future_target_type,
    )
    config.optimization.use_compile = False
    config.eval.parallel_rollout_workers = 8
    config.task.num_envs = 1
    if config.task.obs_type == "image":
        config.task.obs_dim = config.network.emb_dim
    dataset = make_dataset(config.task)
    agent = TrainingAgent(config)
    agent.load(str(checkpoint_path), load_optimizer=False)
    agent.eval()
    pool = PersistentImageRolloutPool(config, lazy=True)
    return config, dataset, agent, pool


def run_smoke_checkpoint(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    episode_count: int = 16,
    audit_seed_domain: str = DEFAULT_AUDIT_SEED_DOMAIN,
) -> dict:
    """Run a smoke-only rollout; this function cannot produce selection records."""

    if episode_count not in {4, 8, 16}:
        raise ValueError("Audit smoke episode_count must be one of 4, 8, or 16")
    config, dataset, agent, pool = _load_smoke_components(
        config_path, checkpoint_path
    )
    task = str(config.task.env_name)
    if task == "mimicgen_kitchen":
        task = "kitchen"
    training_seed = int(config.optimization.seed)
    capture_keys = tuple(sorted(config.task.shape_meta.obs.keys()))
    episodes = []
    try:
        for first_episode in range(0, episode_count, pool.num_workers):
            batch_ids = list(
                range(first_episode, min(first_episode + pool.num_workers, episode_count))
            )
            seeds = [
                audit_episode_seeds(
                    task,
                    training_seed,
                    item,
                    seed_domain=audit_seed_domain,
                )
                for item in batch_ids
            ]
            worker_indices = list(range(len(batch_ids)))
            reset_results = pool.reset(
                [item.env for item in seeds],
                worker_indices=worker_indices,
            )
            observations = [item[0] for item in reset_results]
            done = [False] * len(batch_ids)
            rewards = [0.0] * len(batch_ids)
            success = [False] * len(batch_ids)
            decisions: list[list[dict]] = [[] for _ in batch_ids]
            while not all(done):
                active = [index for index, value in enumerate(done) if not value]
                env_actions = []
                predictions = []
                for local_index in active:
                    decision_index = len(decisions[local_index])
                    policy_obs = _normalize_observation(
                        observations[local_index],
                        dataset,
                        config.optimization.device,
                    )
                    act_0 = torch.zeros(
                        (1, config.task.horizon, config.task.act_dim),
                        device=config.optimization.device,
                    )
                    invocation_id = (
                        f"{task}:seed{training_seed}:episode{batch_ids[local_index]}:"
                        f"decision{decision_index}"
                    )
                    with preserve_rng_state():
                        seed_decision_process_streams(
                            seeds[local_index], decision_index
                        )
                        sampled = agent.sample_joint(
                            act_0,
                            policy_obs,
                            sampler_invocation_id=invocation_id,
                            mode="audit",
                        )
                    record = prediction_record(sampled, invocation_id)
                    predictions.append(record)
                    action = _policy_action(config, dataset, sampled.action)
                    env_actions.append(action)

                step_results = pool.step_with_capture(
                    env_actions,
                    capture_after_steps=4,
                    capture_keys=capture_keys,
                    worker_indices=active,
                )
                for position, local_index in enumerate(active):
                    observation, reward, terminated, truncated, info, capture = (
                        step_results[position]
                    )
                    observations[local_index] = observation
                    reward_value = float(np.asarray(reward).reshape(-1)[0])
                    rewards[local_index] += reward_value
                    is_libero = str(config.task.env_type) == "libero"
                    step_success = reward_value > 0 if is_libero else _success_from_info(info)
                    success[local_index] |= step_success
                    done[local_index] |= bool(
                        step_success
                        or np.asarray(terminated, dtype=bool).reshape(-1)[0]
                        or np.asarray(truncated, dtype=bool).reshape(-1)[0]
                    )
                    prediction = predictions[position]
                    decision = {
                        "decision_id": prediction.decision_id,
                        "sampler_invocation_id": prediction.sampler_invocation_id,
                        "pass_0_call_count": prediction.pass_0_call_count,
                        "pass_1_call_count": prediction.pass_1_call_count,
                        "extra_sampler_call_count": (
                            prediction.extra_sampler_call_count
                        ),
                        "future_pred_shape": list(prediction.future_pred_shape),
                        "future_pred_dtype": prediction.future_pred_dtype,
                        "future_pred_sha256": prediction.future_pred_sha256,
                        "target_valid": bool(capture["valid"]),
                        "capture_env_step": capture["capture_env_step"],
                    }
                    if capture["valid"]:
                        target_obs = _normalize_observation(
                            capture["observation"],
                            dataset,
                            config.optimization.device,
                        )
                        target = encode_audit_future_target(agent, target_obs)
                        contract = build_future_space_contract(
                            agent, prediction.future_pred, target
                        )
                        decision["future_space_contract"] = contract
                        decision["metrics"] = compute_future_metrics(
                            agent, prediction.future_pred, target, contract
                        )
                    decisions[local_index].append(decision)
                    if len(decisions[local_index]) * int(config.task.act_steps) >= int(
                        config.task.max_episode_steps
                    ):
                        done[local_index] = True
            for local_index, episode_id in enumerate(batch_ids):
                score = primary_episode_score(decisions[local_index])
                episodes.append(
                    {
                        "audit_episode_id": episode_id,
                        "master_seed_sha256": seeds[local_index].master_sha256,
                        "success": bool(success[local_index]),
                        "reward": float(rewards[local_index]),
                        "early_termination": len(decisions[local_index]) < 5,
                        "decisions": decisions[local_index],
                        **score,
                    }
                )
    except FutureSpaceContractError:
        raise
    finally:
        pool.close()
    eligible = [item for item in episodes if item["primary_eligible"]]
    # Higher prediction error is interpreted as evidence of rollout failure.
    primary_auroc_failure = binary_auroc(
        [not item["success"] for item in eligible],
        [item["primary_score"] for item in eligible],
    )
    return {
        "format_version": 2,
        "run_kind": "smoke_only",
        "seed_domain": audit_seed_domain,
        "task": task,
        "training_seed": training_seed,
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "episode_count": len(episodes),
        "success_count": sum(int(item["success"]) for item in episodes),
        "primary_eligible_count": len(eligible),
        "primary_auroc_failure": primary_auroc_failure,
        "decision_count": sum(len(item["decisions"]) for item in episodes),
        "valid_target_count": sum(
            int(decision["target_valid"])
            for item in episodes
            for decision in item["decisions"]
        ),
        "decision_invalid_target_rate": sum(
            int(not decision["target_valid"])
            for item in episodes
            for decision in item["decisions"]
        )
        / sum(len(item["decisions"]) for item in episodes),
        "episode_primary_ineligible_rate": sum(
            int(not item["primary_eligible"]) for item in episodes
        )
        / len(episodes),
        "episode_terminated_before_first5_valid_rate": sum(
            int(item["early_termination"]) for item in episodes
        )
        / len(episodes),
        "episodes": episodes,
    }


def smoke_report(raw_result: dict, *, raw_result_path: str, raw_result_sha256: str) -> dict:
    """Regenerate unambiguous eligibility rates from saved decision rows."""

    episodes = raw_result["episodes"]
    decisions = [decision for episode in episodes for decision in episode["decisions"]]
    if not episodes or not decisions:
        raise ValueError("smoke report requires non-empty episode and decision rows")
    eligible = sum(int(episode["primary_eligible"]) for episode in episodes)
    terminated_early = sum(
        int(
            episode.get(
                "early_termination",
                len(episode["decisions"]) < 5,
            )
        )
        for episode in episodes
    )
    valid_targets = sum(int(decision["target_valid"]) for decision in decisions)
    return {
        "format_version": 1,
        "report_kind": "future_rollout_smoke_report",
        "raw_result_path": str(Path(raw_result_path).resolve()),
        "raw_result_sha256": raw_result_sha256,
        "task": raw_result["task"],
        "training_seed": int(raw_result["training_seed"]),
        "episode_count": len(episodes),
        "success_count": int(raw_result["success_count"]),
        "decision_count": len(decisions),
        "valid_target_count": valid_targets,
        "invalid_target_count": len(decisions) - valid_targets,
        "decision_invalid_target_rate": (len(decisions) - valid_targets)
        / len(decisions),
        "episode_primary_eligible_count": eligible,
        "episode_primary_ineligible_rate": (len(episodes) - eligible)
        / len(episodes),
        "episode_terminated_before_first5_valid_rate": terminated_early
        / len(episodes),
        "primary_auroc_failure": raw_result["primary_auroc_failure"],
        "interpretation_gate": "pipeline_smoke_only_not_classification_evidence",
    }
