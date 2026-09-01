"""Deterministic sample and retrieval manifests for the collapse audit."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from mip.dataset_utils import resolve_future_sample_positions


MASTER_SAMPLE_COUNT = 4096
RETRIEVAL_POOL_SIZE = 256
DEFAULT_MANIFEST_SEED = 20260729


@dataclass(frozen=True)
class AuditSample:
    task: str
    sample_id: str
    dataset_sample_index: int
    episode_index: int
    current_frame_indices: tuple[int, ...]
    current_anchor_frame_index: int
    logical_future_horizon: int
    logical_future_position: int
    resolved_future_frame_index: int
    future_was_padded_or_clamped: bool
    future_progress: float
    future_progress_decile: int


@dataclass(frozen=True)
class RetrievalPool:
    query_sample_id: str
    positive_sample_id: str
    candidate_sample_ids: tuple[str, ...]
    positive_candidate_index: int
    future_progress_decile: int
    pool_seed: int


def _stable_seed(*parts: Any) -> int:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _episode_metadata(episode_ends: np.ndarray, source_index: int):
    episode_index = int(np.searchsorted(episode_ends, source_index, side="right"))
    episode_start = 0 if episode_index == 0 else int(episode_ends[episode_index - 1])
    episode_end = int(episode_ends[episode_index])
    denominator = max(episode_end - episode_start - 1, 1)
    progress = (source_index - episode_start) / denominator
    progress = float(min(max(progress, 0.0), 1.0))
    decile = min(int(progress * 10.0), 9)
    return episode_index, progress, decile


def build_sample_manifest(
    dataset,
    task: str,
    *,
    master_count: int = MASTER_SAMPLE_COUNT,
    seed: int = DEFAULT_MANIFEST_SEED,
) -> dict[str, Any]:
    """Build a nested master sample set from the actual training sampler."""

    sampler = dataset.sampler
    if len(sampler) < master_count:
        raise ValueError(
            f"Task {task} has {len(sampler)} samples, fewer than {master_count}"
        )
    n_obs_steps = int(dataset.n_obs_steps)
    future_steps = tuple(int(step) for step in dataset.future_steps)
    if future_steps != (4,):
        raise ValueError(f"Frozen audit requires Future4, got {future_steps}")
    future_position = resolve_future_sample_positions(
        n_obs_steps, future_steps, sampler.sequence_length
    )[0]

    task_seed = _stable_seed(seed, task, "master")
    rng = np.random.default_rng(task_seed)
    selected = rng.permutation(len(sampler))[:master_count]
    episode_ends = np.asarray(sampler.replay_buffer.episode_ends[:], dtype=np.int64)
    records = []
    for order, dataset_index_value in enumerate(selected):
        dataset_index = int(dataset_index_value)
        current = tuple(
            sampler.resolve_sample_position(dataset_index, position).source_index
            for position in range(n_obs_steps)
        )
        resolved_future = sampler.resolve_sample_position(
            dataset_index, future_position.resolved_position
        )
        episode_index, progress, decile = _episode_metadata(
            episode_ends, resolved_future.source_index
        )
        records.append(
            AuditSample(
                task=task,
                sample_id=f"{task}:{order:04d}",
                dataset_sample_index=dataset_index,
                episode_index=episode_index,
                current_frame_indices=current,
                current_anchor_frame_index=current[-1],
                logical_future_horizon=future_position.logical_horizon,
                logical_future_position=future_position.logical_position,
                resolved_future_frame_index=resolved_future.source_index,
                future_was_padded_or_clamped=(
                    future_position.was_clamped or resolved_future.was_padded
                ),
                future_progress=progress,
                future_progress_decile=decile,
            )
        )

    record_dicts = [asdict(record) for record in records]
    digest = hashlib.sha256(
        json.dumps(record_dicts, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    nested_prefixes = [
        count for count in (512, 1024, 2048, 4096) if count <= master_count
    ]
    if not nested_prefixes or nested_prefixes[-1] != master_count:
        nested_prefixes.append(master_count)
    return {
        "schema_version": 1,
        "task": task,
        "seed": seed,
        "task_seed": task_seed,
        "master_count": master_count,
        "nested_prefixes": nested_prefixes,
        "sha256": digest,
        "samples": record_dicts,
    }


def build_retrieval_pool_manifest(
    sample_manifest: dict[str, Any],
    *,
    pool_size: int = RETRIEVAL_POOL_SIZE,
    seed: int = DEFAULT_MANIFEST_SEED,
) -> dict[str, Any]:
    """Freeze one task/progress-matched candidate pool per query."""

    samples = [AuditSample(**record) for record in sample_manifest["samples"]]
    by_decile: dict[int, list[AuditSample]] = {}
    for sample in samples:
        by_decile.setdefault(sample.future_progress_decile, []).append(sample)

    pools = []
    for query in samples:
        eligible_by_frame = {}
        for candidate in by_decile[query.future_progress_decile]:
            if candidate.episode_index == query.episode_index:
                continue
            eligible_by_frame.setdefault(
                candidate.resolved_future_frame_index, candidate
            )
        eligible = list(eligible_by_frame.values())
        if len(eligible) < pool_size - 1:
            raise ValueError(
                f"Query {query.sample_id} has only {len(eligible)} distinct "
                f"cross-episode negatives for pool_size={pool_size}"
            )
        pool_seed = _stable_seed(seed, query.sample_id, "retrieval")
        rng = np.random.default_rng(pool_seed)
        chosen_indices = rng.choice(
            len(eligible), size=pool_size - 1, replace=False
        )
        candidates = [query] + [eligible[int(index)] for index in chosen_indices]
        rng.shuffle(candidates)
        candidate_ids = tuple(candidate.sample_id for candidate in candidates)
        positive_index = candidate_ids.index(query.sample_id)
        pools.append(
            RetrievalPool(
                query_sample_id=query.sample_id,
                positive_sample_id=query.sample_id,
                candidate_sample_ids=candidate_ids,
                positive_candidate_index=positive_index,
                future_progress_decile=query.future_progress_decile,
                pool_seed=pool_seed,
            )
        )

    pool_dicts = [asdict(pool) for pool in pools]
    digest = hashlib.sha256(
        json.dumps(pool_dicts, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": 1,
        "task": sample_manifest["task"],
        "seed": seed,
        "pool_size": pool_size,
        "sha256": digest,
        "pools": pool_dicts,
    }
