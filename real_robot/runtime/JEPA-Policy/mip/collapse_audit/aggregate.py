"""Frozen run/task-level aggregation for the collapse audit.

The seven tasks are fixed rather than resampled.  Within each task, the three
training seeds are resampled and averaged, after which tasks receive equal
weight.  Best and latest are always reported separately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TASKS = (
    "mug_mug",
    "moka_moka",
    "tool_hang",
    "square",
    "coffee_preparation",
    "kitchen",
    "three_piece_assembly",
)
SEEDS = (41, 42, 43)
ENDPOINTS = ("best", "latest")
TE32_BOUNDARY = 0.05
DEGENERACY_BOUNDARY = 1e-12


def _atomic_json_save(value: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(destination)


def _load_records(metrics_dir: Path) -> list[dict]:
    records: list[dict] = []
    for task in TASKS:
        path = metrics_dir / f"{task}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing point metrics: {path}")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        records.extend(payload["records"])
    expected = len(TASKS) * len(SEEDS) * len(ENDPOINTS) * 2
    if len(records) != expected:
        raise RuntimeError(f"Expected {expected} checkpoint records, got {len(records)}")
    return records


def _index(records: list[dict]) -> dict[tuple[str, int, str, str], dict]:
    result = {}
    for record in records:
        key = (
            record["task"],
            int(record["seed"]),
            record["variant"],
            record["checkpoint_kind"],
        )
        if key in result:
            raise RuntimeError(f"Duplicate record: {key}")
        result[key] = record
    return result


def _value(record: dict, representation: str, metric: str) -> float:
    if metric.startswith("retrieval."):
        return float(record["retrieval"][metric.split(".", 1)[1]])
    if metric == "delta_r32":
        return float(
            record["representations"][representation]["controls"]
            ["rank_controls"]["rank32"]
            ["learned_minus_control_effective_rank"]
        )
    if metric == "relative_centered_energy_minus_tau":
        return float(
            record["representations"][representation]["metrics"]
            ["relative_centered_energy"]
            - DEGENERACY_BOUNDARY
        )
    return float(record["representations"][representation]["metrics"][metric])


def fixed_task_seed_bootstrap(
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict:
    """Bootstrap [task, seed] values while retaining every fixed task."""

    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(TASKS), len(SEEDS)):
        raise ValueError(f"Expected shape {(len(TASKS), len(SEEDS))}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Cannot aggregate non-finite values")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(SEEDS), size=(replicates, len(TASKS), len(SEEDS)))
    sampled = np.take_along_axis(
        np.broadcast_to(values, (replicates,) + values.shape),
        draws,
        axis=2,
    )
    statistics = sampled.mean(axis=2).mean(axis=1)
    return {
        "estimate": float(values.mean(axis=1).mean()),
        "one_sided_95_lcb": float(np.quantile(statistics, 0.05)),
        "one_sided_95_ucb": float(np.quantile(statistics, 0.95)),
        "two_sided_95_ci": [
            float(np.quantile(statistics, 0.025)),
            float(np.quantile(statistics, 0.975)),
        ],
        "bootstrap_replicates": int(replicates),
        "tasks_fixed": True,
        "seeds_resampled_within_task": True,
        "tasks_equally_weighted": True,
    }


def _matrix(
    indexed: dict,
    *,
    variant: str,
    endpoint: str,
    representation: str,
    metric: str,
) -> np.ndarray:
    return np.asarray(
        [
            [
                _value(indexed[(task, seed, variant, endpoint)], representation, metric)
                for seed in SEEDS
            ]
            for task in TASKS
        ],
        dtype=np.float64,
    )


def _summarize_endpoint(
    indexed: dict,
    *,
    variant: str,
    endpoint: str,
    representation: str,
    metrics: tuple[str, ...],
    replicates: int,
    seed: int,
) -> dict:
    result = {}
    for offset, metric in enumerate(metrics):
        values = _matrix(
            indexed,
            variant=variant,
            endpoint=endpoint,
            representation=representation,
            metric=metric,
        )
        summary = fixed_task_seed_bootstrap(
            values, replicates=replicates, seed=seed + offset
        )
        summary["task_seed_values"] = {
            task: {str(s): float(values[ti, si]) for si, s in enumerate(SEEDS)}
            for ti, task in enumerate(TASKS)
        }
        result[metric] = summary
    return result


def _encoder_exceptions(indexed: dict, variant: str) -> list[dict]:
    exceptions = []
    for endpoint in ENDPOINTS:
        for task in TASKS:
            values = [
                _value(indexed[(task, seed, variant, endpoint)], "z_t", "te32")
                for seed in SEEDS
            ]
            failing = [seed for seed, value in zip(SEEDS, values, strict=True) if value <= TE32_BOUNDARY]
            diversity = [
                _value(
                    indexed[(task, seed, variant, endpoint)],
                    "z_t",
                    "relative_centered_energy",
                )
                for seed in SEEDS
            ]
            degenerate = [
                seed
                for seed, value in zip(SEEDS, diversity, strict=True)
                if value <= DEGENERACY_BOUNDARY
            ]
            if len(failing) >= 2 or len(degenerate) >= 2:
                exceptions.append(
                    {
                        "task": task,
                        "endpoint": endpoint,
                        "reason": "encoder_low_rank_or_complete_collapse_exception",
                        "failing_seeds": failing,
                        "values": values,
                        "relative_centered_energy": diversity,
                        "degenerate_seeds": degenerate,
                    }
                )
    return exceptions


def _predictor_exceptions(indexed: dict) -> tuple[list[dict], list[dict]]:
    predictor = []
    target = []
    for endpoint in ENDPOINTS:
        for task in TASKS:
            pred_te32 = [
                _value(indexed[(task, seed, "future4_ratio010", endpoint)], "pred0", "te32")
                for seed in SEEDS
            ]
            retrieval = [
                _value(
                    indexed[(task, seed, "future4_ratio010", endpoint)],
                    "pred0",
                    "retrieval.recall_at_5_minus_null",
                )
                for seed in SEEDS
            ]
            target_te32 = [
                _value(indexed[(task, seed, "future4_ratio010", endpoint)], "future_target", "te32")
                for seed in SEEDS
            ]
            pred_diversity = [
                _value(
                    indexed[(task, seed, "future4_ratio010", endpoint)],
                    "pred0",
                    "relative_centered_energy",
                )
                for seed in SEEDS
            ]
            target_diversity = [
                _value(
                    indexed[(task, seed, "future4_ratio010", endpoint)],
                    "future_target",
                    "relative_centered_energy",
                )
                for seed in SEEDS
            ]
            bad_rank = [s for s, v in zip(SEEDS, pred_te32, strict=True) if v <= TE32_BOUNDARY]
            bad_retrieval = [s for s, v in zip(SEEDS, retrieval, strict=True) if v <= 0.0]
            bad_diversity = [
                s
                for s, v in zip(SEEDS, pred_diversity, strict=True)
                if v <= DEGENERACY_BOUNDARY
            ]
            if (
                len(bad_rank) >= 2
                or len(bad_retrieval) >= 2
                or len(bad_diversity) >= 2
            ):
                predictor.append(
                    {
                        "task": task,
                        "endpoint": endpoint,
                        "reason": "predictor_rank_or_retrieval_exception",
                        "pred0_te32": pred_te32,
                        "retrieval_recall5_minus_null": retrieval,
                        "rank_failing_seeds": bad_rank,
                        "retrieval_failing_seeds": bad_retrieval,
                        "pred0_relative_centered_energy": pred_diversity,
                        "degenerate_seeds": bad_diversity,
                    }
                )
            bad_target = [s for s, v in zip(SEEDS, target_te32, strict=True) if v <= TE32_BOUNDARY]
            bad_target_diversity = [
                s
                for s, v in zip(SEEDS, target_diversity, strict=True)
                if v <= DEGENERACY_BOUNDARY
            ]
            if len(bad_target) >= 2 or len(bad_target_diversity) >= 2:
                target.append(
                    {
                        "task": task,
                        "endpoint": endpoint,
                        "reason": "at_least_2_of_3_target_te32_at_or_below_0.05",
                        "values": target_te32,
                        "failing_seeds": bad_target,
                        "relative_centered_energy": target_diversity,
                        "degenerate_seeds": bad_target_diversity,
                    }
                )
    return predictor, target


def aggregate(metrics_dir: Path, *, replicates: int, seed: int) -> dict:
    indexed = _index(_load_records(metrics_dir))
    encoder_metrics = (
        "relative_centered_energy",
        "relative_centered_energy_minus_tau",
        "centered_effective_rank",
        "normalized_effective_rank",
        "te8",
        "te32",
        "delta_r32",
    )
    predictor_metrics = encoder_metrics + (
        "retrieval.recall_at_5_minus_null",
        "retrieval.margin_minus_null",
    )
    summaries = {"encoder": {}, "future": {}}
    counter = 0
    for variant in ("action_only", "future4_ratio010"):
        summaries["encoder"][variant] = {}
        for endpoint in ENDPOINTS:
            summaries["encoder"][variant][endpoint] = _summarize_endpoint(
                indexed,
                variant=variant,
                endpoint=endpoint,
                representation="z_t",
                metrics=encoder_metrics,
                replicates=replicates,
                seed=seed + 100 * counter,
            )
            counter += 1
    for representation in ("future_target", "pred0"):
        summaries["future"][representation] = {}
        for endpoint in ENDPOINTS:
            metrics = predictor_metrics if representation == "pred0" else encoder_metrics
            summaries["future"][representation][endpoint] = _summarize_endpoint(
                indexed,
                variant="future4_ratio010",
                endpoint=endpoint,
                representation=representation,
                metrics=metrics,
                replicates=replicates,
                seed=seed + 100 * counter,
            )
            counter += 1

    # Secondary matched comparison.  The difference is formed before the
    # shared seed indices are bootstrapped, preserving each (task, seed) pair.
    paired = {}
    for endpoint in ENDPOINTS:
        future = _matrix(indexed, variant="future4_ratio010", endpoint=endpoint, representation="z_t", metric="normalized_effective_rank")
        baseline = _matrix(indexed, variant="action_only", endpoint=endpoint, representation="z_t", metric="normalized_effective_rank")
        paired[endpoint] = fixed_task_seed_bootstrap(
            future - baseline,
            replicates=replicates,
            seed=seed + 100 * counter,
        )
        counter += 1

    latest_minus_best = {"encoder": {}, "future": {}}
    for variant in ("action_only", "future4_ratio010"):
        latest_minus_best["encoder"][variant] = {}
        for metric in ("relative_centered_energy", "normalized_effective_rank", "te32"):
            latest = _matrix(
                indexed,
                variant=variant,
                endpoint="latest",
                representation="z_t",
                metric=metric,
            )
            best = _matrix(
                indexed,
                variant=variant,
                endpoint="best",
                representation="z_t",
                metric=metric,
            )
            latest_minus_best["encoder"][variant][metric] = fixed_task_seed_bootstrap(
                latest - best,
                replicates=replicates,
                seed=seed + 100 * counter,
            )
            counter += 1
    for representation in ("future_target", "pred0"):
        latest_minus_best["future"][representation] = {}
        metrics = ["relative_centered_energy", "normalized_effective_rank", "te32"]
        if representation == "pred0":
            metrics.append("retrieval.recall_at_5_minus_null")
        for metric in metrics:
            latest = _matrix(
                indexed,
                variant="future4_ratio010",
                endpoint="latest",
                representation=representation,
                metric=metric,
            )
            best = _matrix(
                indexed,
                variant="future4_ratio010",
                endpoint="best",
                representation=representation,
                metric=metric,
            )
            latest_minus_best["future"][representation][metric] = fixed_task_seed_bootstrap(
                latest - best,
                replicates=replicates,
                seed=seed + 100 * counter,
            )
            counter += 1

    predictor_exceptions, target_exceptions = _predictor_exceptions(indexed)
    action_exceptions = _encoder_exceptions(indexed, "action_only")
    future_encoder_exceptions = _encoder_exceptions(indexed, "future4_ratio010")

    def both_endpoints_pass(group: dict, required: tuple[tuple[str, float], ...]) -> bool:
        return all(
            group[endpoint][metric]["one_sided_95_lcb"] > boundary
            for endpoint in ENDPOINTS
            for metric, boundary in required
        )

    claims = {
        "encoder_action_only_global": both_endpoints_pass(
            summaries["encoder"]["action_only"],
            (("relative_centered_energy_minus_tau", 0.0), ("te32", TE32_BOUNDARY)),
        ),
        "encoder_future4_global": both_endpoints_pass(
            summaries["encoder"]["future4_ratio010"],
            (("relative_centered_energy_minus_tau", 0.0), ("te32", TE32_BOUNDARY)),
        ),
        "predictor_future4_global": both_endpoints_pass(
            summaries["future"]["pred0"],
            (
                ("relative_centered_energy_minus_tau", 0.0),
                ("te32", TE32_BOUNDARY),
                ("retrieval.recall_at_5_minus_null", 0.0),
            ),
        ),
    }
    claims["encoder_action_only_without_task_exception"] = (
        claims["encoder_action_only_global"] and not action_exceptions
    )
    claims["encoder_future4_without_task_exception"] = (
        claims["encoder_future4_global"] and not future_encoder_exceptions
    )
    claims["predictor_future4_without_task_exception"] = (
        claims["predictor_future4_global"] and not predictor_exceptions
    )
    return {
        "schema_version": 1,
        "scope": {
            "tasks": list(TASKS),
            "seeds": list(SEEDS),
            "endpoints": list(ENDPOINTS),
            "best_latest_averaged_for_primary_claim": False,
            "task_resampling_in_primary_bootstrap": False,
            "te32_practical_boundary": TE32_BOUNDARY,
            "numerical_degeneracy_boundary": DEGENERACY_BOUNDARY,
        },
        "summaries": summaries,
        "secondary_paired_future_minus_baseline_normalized_rank": paired,
        "secondary_paired_latest_minus_best": latest_minus_best,
        "claim_checks": claims,
        "exceptions": {
            "encoder_action_only": action_exceptions,
            "encoder_future4_ratio010": future_encoder_exceptions,
            "predictor": predictor_exceptions,
            "future_target": target_exceptions,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("outputs/collapse_audit/point_metrics"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/collapse_audit/global_summary.json"),
    )
    parser.add_argument("--replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    result = aggregate(args.metrics_dir, replicates=args.replicates, seed=args.seed)
    _atomic_json_save(result, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
