"""Episode bootstrap for checkpoint-level spectral uncertainty.

Episode first/second sufficient statistics are kept in memory while one
matrix is processed.  Each replicate rebuilds the paired learned/control
spectrum from the same resampled episodes; controls never receive an
independent bootstrap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch

from mip.collapse_audit.metrics import EPS, effective_rank_from_eigenvalues


LOGGER = logging.getLogger("collapse_audit.bootstrap")
TE32_PRACTICAL_BOUNDARY = 0.05
TE32_BOUNDARY_ADJACENCY = 0.01
PRIORITY_RULE = (
    "stability artifact, numerical degeneracy, TE32 <= 0.05 anomaly, "
    "or |TE32 - 0.05| <= 0.01 boundary adjacency"
)


def _atomic_json_save(value: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(destination)


def _entropy_rank_batch(eigenvalues: torch.Tensor) -> torch.Tensor:
    singular_values = torch.sqrt(eigenvalues.clamp_min(0.0))
    totals = singular_values.sum(dim=1, keepdim=True)
    probabilities = singular_values / totals.clamp_min(EPS)
    entropy = -(probabilities * probabilities.clamp_min(EPS).log()).sum(dim=1)
    ranks = entropy.exp()
    return torch.where(totals[:, 0] <= EPS, torch.zeros_like(ranks), ranks)


def _rank_k_control_batch(
    descending_eigenvalues: torch.Tensor, k: int
) -> torch.Tensor:
    retained = descending_eigenvalues[:, :k]
    return _entropy_rank_batch(retained)


def episode_sufficient_statistics(
    embedding: torch.Tensor,
    episode_ids: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    matrix = embedding.detach().to(device="cpu", dtype=torch.float64)
    if matrix.dim() != 2 or matrix.shape[0] != len(episode_ids):
        raise ValueError("embedding and episode IDs must align as [N,D]")
    if not torch.isfinite(matrix).all():
        raise ValueError("embedding contains non-finite values")
    unique = np.unique(episode_ids)
    counts = torch.empty(len(unique), dtype=torch.float64)
    sums = torch.empty((len(unique), matrix.shape[1]), dtype=torch.float64)
    second_moments = torch.empty(
        (len(unique), matrix.shape[1], matrix.shape[1]), dtype=torch.float64
    )
    for index, episode in enumerate(unique):
        rows = matrix[torch.from_numpy(np.flatnonzero(episode_ids == episode))]
        counts[index] = rows.shape[0]
        sums[index] = rows.sum(dim=0)
        second_moments[index] = rows.T @ rows
    return counts, sums, second_moments, unique


def episode_bootstrap_spectral(
    embedding: torch.Tensor,
    episode_ids: np.ndarray,
    *,
    replicates: int,
    seed: int,
    batch_size: int = 4,
) -> dict:
    """Return paired episode-bootstrap draws and percentile summaries."""

    counts, sums, second_moments, unique = episode_sufficient_statistics(
        embedding, episode_ids
    )
    episode_count = len(unique)
    if episode_count < 2:
        raise ValueError("episode bootstrap requires at least two episodes")
    dimension = embedding.shape[1]
    rng = np.random.default_rng(seed)
    weights_np = rng.multinomial(
        episode_count,
        np.full(episode_count, 1.0 / episode_count),
        size=replicates,
    )
    output_names = (
        "relative_centered_energy",
        "centered_effective_rank",
        "normalized_effective_rank",
        "ev1",
        "te8",
        "te32",
        "delta_r8",
        "delta_r32",
    )
    draws = {name: np.empty(replicates, dtype=np.float64) for name in output_names}

    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        weights = torch.from_numpy(weights_np[start:stop]).to(torch.float64)
        total_counts = weights @ counts
        total_sums = weights @ sums
        total_second = torch.einsum("be,edh->bdh", weights, second_moments)
        centered_gram = total_second - torch.einsum(
            "bd,bh->bdh", total_sums, total_sums
        ) / total_counts[:, None, None]
        centered_gram = (centered_gram + centered_gram.transpose(1, 2)) * 0.5
        eigenvalues = torch.linalg.eigvalsh(centered_gram).clamp_min(0.0)
        descending = eigenvalues.flip(1)
        energy = descending.sum(dim=1)
        raw_energy = total_second.diagonal(dim1=1, dim2=2).sum(dim=1)
        centered_rank = _entropy_rank_batch(eigenvalues)
        maximum_rank = torch.minimum(
            (total_counts - 1).to(torch.int64),
            torch.full_like(total_counts, dimension, dtype=torch.int64),
        ).to(torch.float64)
        ev1 = descending[:, :1].sum(dim=1) / energy.clamp_min(EPS)
        cev8 = descending[:, :8].sum(dim=1) / energy.clamp_min(EPS)
        cev32 = descending[:, :32].sum(dim=1) / energy.clamp_min(EPS)
        degenerate = energy <= EPS
        ev1 = torch.where(degenerate, torch.zeros_like(ev1), ev1)
        te8 = torch.where(degenerate, torch.zeros_like(cev8), 1.0 - cev8)
        te32 = torch.where(degenerate, torch.zeros_like(cev32), 1.0 - cev32)
        centered_energy = energy / total_counts
        relative_energy = centered_energy / (raw_energy / total_counts + EPS)
        rank8 = _rank_k_control_batch(descending, 8)
        rank32 = _rank_k_control_batch(descending, 32)
        values = {
            "relative_centered_energy": relative_energy,
            "centered_effective_rank": centered_rank,
            "normalized_effective_rank": centered_rank / maximum_rank,
            "ev1": ev1,
            "te8": te8,
            "te32": te32,
            "delta_r8": centered_rank - rank8,
            "delta_r32": centered_rank - rank32,
        }
        for name, value in values.items():
            draws[name][start:stop] = value.numpy()

    summaries = {}
    for name, values in draws.items():
        summaries[name] = {
            "mean": float(values.mean()),
            "one_sided_95_lcb": float(np.quantile(values, 0.05)),
            "one_sided_95_ucb": float(np.quantile(values, 0.95)),
            "two_sided_95_ci": [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ],
        }
    return {
        "episode_count": episode_count,
        "replicates": replicates,
        "seed": seed,
        "controls_recomputed_within_each_paired_replicate": True,
        "summaries": summaries,
    }


def _stable_seed(identity: str, base_seed: int) -> int:
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:4], "little")) % (2**32)


def _is_priority_matrix(
    *,
    artifact_path: str,
    stability_artifacts: set[str],
    te32: float,
    degenerate: bool,
) -> bool:
    """Promote stability cases, all anomalies, and boundary-adjacent matrices."""

    low_rank_anomaly = te32 <= TE32_PRACTICAL_BOUNDARY
    boundary_adjacent = (
        abs(te32 - TE32_PRACTICAL_BOUNDARY) <= TE32_BOUNDARY_ADJACENCY
    )
    return (
        artifact_path in stability_artifacts
        or low_rank_anomaly
        or boundary_adjacent
        or degenerate
    )


def bootstrap_all(
    *,
    point_metrics_dir: Path,
    manifests_dir: Path,
    output_dir: Path,
    sample_count: int,
    screen_replicates: int,
    priority_replicates: int,
    stability_path: Path,
    base_seed: int,
    batch_size: int,
    task_filter: str | None,
    resume: bool,
) -> None:
    stability_artifacts: set[str] = set()
    if stability_path.is_file():
        with stability_path.open("r", encoding="utf-8") as handle:
            stability = json.load(handle)
        stability_artifacts = {
            record["artifact_path"] for record in stability["results"]
        }
    for point_path in sorted(point_metrics_dir.glob("*.json")):
        with point_path.open("r", encoding="utf-8") as handle:
            point = json.load(handle)
        task = point["task"]
        if task_filter is not None and task != task_filter:
            continue
        task_output = output_dir / f"{task}.json"
        if resume and task_output.is_file():
            try:
                with task_output.open("r", encoding="utf-8") as handle:
                    existing = json.load(handle)
                if (
                    existing.get("schema_version") == 2
                    and existing.get("priority_rule") == PRIORITY_RULE
                    and existing.get("sample_count") == sample_count
                    and existing.get("screen_replicates") == screen_replicates
                    and existing.get("priority_replicates") == priority_replicates
                    and len(existing.get("records", [])) == 12
                ):
                    LOGGER.info("Resume skip complete bootstrap task=%s", task)
                    continue
            except Exception:
                pass
        with (manifests_dir / f"sample_manifest_{task}.json").open(
            "r", encoding="utf-8"
        ) as handle:
            manifest = json.load(handle)
        episode_ids = np.asarray(
            [int(sample["episode_index"]) for sample in manifest["samples"][:sample_count]],
            dtype=np.int64,
        )
        output_records = []
        for record in point["records"]:
            artifact = torch.load(
                record["artifact_path"],
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
            result_record = {
                "task": task,
                "seed": record["seed"],
                "variant": record["variant"],
                "checkpoint_kind": record["checkpoint_kind"],
                "artifact_path": record["artifact_path"],
                "representations": {},
            }
            for name, point_representation in record["representations"].items():
                te32 = float(point_representation["metrics"]["te32"])
                priority = _is_priority_matrix(
                    artifact_path=record["artifact_path"],
                    stability_artifacts=stability_artifacts,
                    te32=te32,
                    degenerate=bool(point_representation["metrics"]["degenerate"]),
                )
                replicates = priority_replicates if priority else screen_replicates
                identity = (
                    f"{task}:{record['variant']}:{record['seed']}:"
                    f"{record['checkpoint_kind']}:{name}"
                )
                LOGGER.info(
                    "Bootstrap %s reps=%d shape=%s",
                    identity,
                    replicates,
                    tuple(artifact[name][:sample_count].shape),
                )
                result_record["representations"][name] = episode_bootstrap_spectral(
                    artifact[name][:sample_count],
                    episode_ids,
                    replicates=replicates,
                    seed=_stable_seed(identity, base_seed),
                    batch_size=batch_size,
                )
                result_record["representations"][name]["priority_promoted"] = priority
            output_records.append(result_record)
        _atomic_json_save(
            {
                "schema_version": 2,
                "task": task,
                "sample_count": sample_count,
                "screen_replicates": screen_replicates,
                "priority_replicates": priority_replicates,
                "te32_practical_boundary": TE32_PRACTICAL_BOUNDARY,
                "te32_boundary_adjacency": TE32_BOUNDARY_ADJACENCY,
                "priority_rule": PRIORITY_RULE,
                "records": output_records,
            },
            task_output,
        )
        LOGGER.info("Saved bootstrap task=%s", task)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--point-metrics-dir", type=Path, default=Path("outputs/collapse_audit/point_metrics"))
    parser.add_argument("--manifests-dir", type=Path, default=Path("outputs/collapse_audit/manifests"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/collapse_audit/bootstrap"))
    parser.add_argument("--sample-count", type=int, default=2048)
    parser.add_argument("--screen-replicates", type=int, default=200)
    parser.add_argument("--priority-replicates", type=int, default=2000)
    parser.add_argument("--stability", type=Path, default=Path("outputs/collapse_audit/sample_stability.json"))
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--task")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    bootstrap_all(
        point_metrics_dir=args.point_metrics_dir,
        manifests_dir=args.manifests_dir,
        output_dir=args.output_dir,
        sample_count=args.sample_count,
        screen_replicates=args.screen_replicates,
        priority_replicates=args.priority_replicates,
        stability_path=args.stability,
        base_seed=args.seed,
        batch_size=args.batch_size,
        task_filter=args.task,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
