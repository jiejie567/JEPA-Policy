"""Fixed-pool retrieval and candidate-label permutation null."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from mip.collapse_audit.metrics import EPS


@dataclass(frozen=True)
class RetrievalResult:
    valid_queries: int
    invalid_queries: int
    invalid_sample_ids: tuple[str, ...]
    recall_at_1: float
    recall_at_5: float
    median_rank: float
    mean_negative_margin: float
    null_replicates: int
    null_recall_at_1_mean: float
    null_recall_at_5_mean: float
    null_recall_at_5_p99: float
    null_median_rank_mean: float
    null_margin_mean: float
    recall_at_5_minus_null: float
    margin_minus_null: float
    recall_at_5_permutation_p: float
    chance_recall_at_1: float
    chance_recall_at_5: float

    def to_dict(self) -> dict:
        return asdict(self)


def _episode_equal_mean(values: np.ndarray, episodes: np.ndarray) -> float:
    means = [values[episodes == episode].mean() for episode in np.unique(episodes)]
    return float(np.mean(means))


def _episode_equal_median(values: np.ndarray, episodes: np.ndarray) -> float:
    medians = [
        np.median(values[episodes == episode]) for episode in np.unique(episodes)
    ]
    return float(np.median(medians))


def _normalize_with_target_mean(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if predictions.shape != targets.shape or predictions.dim() != 2:
        raise ValueError(
            "Predictions and targets must have the same [N,D] shape, got "
            f"{tuple(predictions.shape)} and {tuple(targets.shape)}"
        )
    predictions = predictions.detach().to(device="cpu", dtype=torch.float64)
    targets = targets.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(predictions).all() or not torch.isfinite(targets).all():
        raise ValueError("Retrieval inputs contain non-finite values")
    target_mean = targets.mean(dim=0, keepdim=True)
    centered_predictions = predictions - target_mean
    centered_targets = targets - target_mean
    prediction_norms = centered_predictions.norm(dim=1)
    target_norms = centered_targets.norm(dim=1)
    valid_predictions = prediction_norms > eps
    valid_targets = target_norms > eps
    normalized_predictions = centered_predictions / prediction_norms.clamp_min(eps)[:, None]
    normalized_targets = centered_targets / target_norms.clamp_min(eps)[:, None]
    return (
        normalized_predictions,
        normalized_targets,
        valid_predictions,
        valid_targets,
    )


def evaluate_fixed_pool_retrieval(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    sample_manifest: dict,
    pool_manifest: dict,
    *,
    null_replicates: int = 1000,
    null_seed: int = 20260729,
    eps: float = EPS,
) -> RetrievalResult:
    """Evaluate retrieval using one checkpoint-wide target mean.

    Ties are broken lexicographically by candidate sample ID. Invalid centered
    vectors are recorded and excluded; they are never replaced with zeros.
    """

    samples = sample_manifest["samples"]
    if predictions.shape[0] != len(samples):
        raise ValueError(
            f"Tensor rows ({predictions.shape[0]}) do not match manifest "
            f"samples ({len(samples)})"
        )
    if null_replicates <= 0:
        raise ValueError("null_replicates must be positive")
    by_id = {record["sample_id"]: index for index, record in enumerate(samples)}
    episode_by_id = {
        record["sample_id"]: int(record["episode_index"]) for record in samples
    }
    normalized_predictions, normalized_targets, valid_predictions, valid_targets = (
        _normalize_with_target_mean(predictions, targets, eps=eps)
    )

    observed_ranks = []
    observed_margins = []
    episode_ids = []
    all_candidate_ranks = []
    all_candidate_margins = []
    invalid_ids = set()
    for pool in pool_manifest["pools"]:
        query_id = pool["query_sample_id"]
        query_index = by_id[query_id]
        candidate_ids = tuple(pool["candidate_sample_ids"])
        candidate_indices = torch.tensor(
            [by_id[candidate_id] for candidate_id in candidate_ids],
            dtype=torch.long,
        )
        candidate_valid = valid_targets[candidate_indices]
        if not bool(valid_predictions[query_index]) or not bool(candidate_valid.all()):
            if not bool(valid_predictions[query_index]):
                invalid_ids.add(query_id)
            invalid_ids.update(
                candidate_id
                for candidate_id, valid in zip(
                    candidate_ids, candidate_valid.tolist(), strict=True
                )
                if not valid
            )
            continue

        similarities = (
            normalized_targets[candidate_indices]
            @ normalized_predictions[query_index]
        ).numpy()
        order = sorted(
            range(len(candidate_ids)),
            key=lambda index: (-similarities[index], candidate_ids[index]),
        )
        ranks = np.empty(len(order), dtype=np.int64)
        for rank, candidate_position in enumerate(order, start=1):
            ranks[candidate_position] = rank
        total_similarity = similarities.sum()
        margins = similarities - (
            total_similarity - similarities
        ) / (len(similarities) - 1)
        positive_position = int(pool["positive_candidate_index"])
        observed_ranks.append(ranks[positive_position])
        observed_margins.append(margins[positive_position])
        episode_ids.append(episode_by_id[query_id])
        all_candidate_ranks.append(ranks)
        all_candidate_margins.append(margins)

    if not observed_ranks:
        raise ValueError("No valid retrieval queries remain")
    ranks = np.asarray(observed_ranks)
    margins = np.asarray(observed_margins)
    episodes = np.asarray(episode_ids)
    candidate_ranks = np.stack(all_candidate_ranks)
    candidate_margins = np.stack(all_candidate_margins)
    pool_size = candidate_ranks.shape[1]
    recall1 = _episode_equal_mean((ranks <= 1).astype(float), episodes)
    recall5 = _episode_equal_mean((ranks <= min(5, pool_size)).astype(float), episodes)
    median_rank = _episode_equal_median(ranks, episodes)
    margin = _episode_equal_mean(margins, episodes)

    rng = np.random.default_rng(null_seed)
    null_r1 = np.empty(null_replicates)
    null_r5 = np.empty(null_replicates)
    null_rank = np.empty(null_replicates)
    null_margin = np.empty(null_replicates)
    row_indices = np.arange(candidate_ranks.shape[0])
    for replicate in range(null_replicates):
        pseudo_positions = rng.integers(0, pool_size, size=len(row_indices))
        pseudo_ranks = candidate_ranks[row_indices, pseudo_positions]
        pseudo_margins = candidate_margins[row_indices, pseudo_positions]
        null_r1[replicate] = _episode_equal_mean(
            (pseudo_ranks <= 1).astype(float), episodes
        )
        null_r5[replicate] = _episode_equal_mean(
            (pseudo_ranks <= min(5, pool_size)).astype(float), episodes
        )
        null_rank[replicate] = _episode_equal_median(pseudo_ranks, episodes)
        null_margin[replicate] = _episode_equal_mean(pseudo_margins, episodes)

    null_r5_mean = float(null_r5.mean())
    null_margin_mean = float(null_margin.mean())
    permutation_p = float(
        (1 + np.count_nonzero(null_r5 >= recall5)) / (null_replicates + 1)
    )
    return RetrievalResult(
        valid_queries=len(ranks),
        invalid_queries=len(pool_manifest["pools"]) - len(ranks),
        invalid_sample_ids=tuple(sorted(invalid_ids)),
        recall_at_1=recall1,
        recall_at_5=recall5,
        median_rank=median_rank,
        mean_negative_margin=margin,
        null_replicates=null_replicates,
        null_recall_at_1_mean=float(null_r1.mean()),
        null_recall_at_5_mean=null_r5_mean,
        null_recall_at_5_p99=float(np.quantile(null_r5, 0.99)),
        null_median_rank_mean=float(null_rank.mean()),
        null_margin_mean=null_margin_mean,
        recall_at_5_minus_null=recall5 - null_r5_mean,
        margin_minus_null=margin - null_margin_mean,
        recall_at_5_permutation_p=permutation_p,
        chance_recall_at_1=1.0 / pool_size,
        chance_recall_at_5=min(5, pool_size) / pool_size,
    )
