"""Select representative checkpoints and audit spectral sample-count stability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mip.collapse_audit.metrics import EPS, compute_spectral_metrics, constant_control


SAMPLE_COUNTS = (512, 1024, 2048, 4096)


def _atomic_json_save(value: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(destination)


def _load_records(metrics_dir: Path) -> list[dict]:
    records = []
    for path in sorted(metrics_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            records.extend(json.load(handle)["records"])
    if len(records) != 84:
        raise RuntimeError(f"Expected 84 scored checkpoints, got {len(records)}")
    return records


def select_checkpoints(records: list[dict], count: int = 8) -> list[dict]:
    """Deterministic coverage selection, including spectral and retrieval extremes."""

    selected: list[dict] = []
    identities: set[tuple] = set()

    def add(record: dict) -> None:
        identity = (
            record["task"], record["seed"], record["variant"], record["checkpoint_kind"]
        )
        if identity not in identities:
            identities.add(identity)
            selected.append(record)

    rank_sorted = sorted(
        records,
        key=lambda r: r["representations"]["z_t"]["metrics"]["normalized_effective_rank"],
    )
    add(rank_sorted[0])
    add(rank_sorted[-1])
    future = [r for r in records if r["variant"] == "future4_ratio010"]
    retrieval_sorted = sorted(future, key=lambda r: r["retrieval"]["recall_at_5_minus_null"])
    add(retrieval_sorted[0])
    add(retrieval_sorted[-1])

    # Ensure both endpoint/variant strata and maximize task coverage.
    for variant in ("action_only", "future4_ratio010"):
        for endpoint in ("best", "latest"):
            candidates = [r for r in records if r["variant"] == variant and r["checkpoint_kind"] == endpoint]
            candidates.sort(
                key=lambda r: (
                    r["task"] in {x["task"] for x in selected},
                    r["task"],
                    r["seed"],
                )
            )
            add(candidates[0])
    for record in sorted(
        records,
        key=lambda r: (
            r["task"] in {x["task"] for x in selected},
            r["task"], r["variant"], r["checkpoint_kind"], r["seed"],
        ),
    ):
        if len(selected) >= count:
            break
        add(record)
    return selected[:count]


def run_stability(metrics_dir: Path) -> dict:
    import torch

    selected = select_checkpoints(_load_records(metrics_dir))
    results = []
    switch_to_4096 = False
    constant_control_residuals = []
    for record in selected:
        artifact = torch.load(
            record["artifact_path"], map_location="cpu", weights_only=False, mmap=True
        )
        representation_results = {}
        for name in ("z_t", "future_target", "pred0"):
            if name not in artifact:
                continue
            metrics_by_count = {}
            for sample_count in SAMPLE_COUNTS:
                metrics = compute_spectral_metrics(artifact[name][:sample_count])
                metrics_by_count[str(sample_count)] = {
                    "centered_effective_rank": metrics.centered_effective_rank,
                    "rankme_raw": metrics.rankme_raw,
                    "ev1": metrics.ev1,
                }
            constant_control_residuals.append(
                compute_spectral_metrics(
                    constant_control(artifact[name][:2048])
                ).relative_centered_energy
            )
            rank2048 = metrics_by_count["2048"]["centered_effective_rank"]
            rank4096 = metrics_by_count["4096"]["centered_effective_rank"]
            absolute_change = abs(rank4096 - rank2048)
            relative_change = absolute_change / max(abs(rank4096), 1e-12)
            ev1_change = abs(
                metrics_by_count["4096"]["ev1"]
                - metrics_by_count["2048"]["ev1"]
            )
            failed = (
                relative_change > 0.05
                or absolute_change > 2.0
                or ev1_change > 0.02
            )
            switch_to_4096 = switch_to_4096 or failed
            representation_results[name] = {
                "metrics_by_sample_count": metrics_by_count,
                "rank_2048_to_4096_absolute_change": absolute_change,
                "rank_2048_to_4096_relative_change": relative_change,
                "ev1_2048_to_4096_absolute_change": ev1_change,
                "stability_failed": failed,
            }
        results.append(
            {
                "task": record["task"],
                "seed": record["seed"],
                "variant": record["variant"],
                "checkpoint_kind": record["checkpoint_kind"],
                "artifact_path": record["artifact_path"],
                "selection_reason": "coverage plus rank/retrieval extremes",
                "representations": representation_results,
            }
        )
    maximum_constant_residual = max(constant_control_residuals, default=0.0)
    return {
        "schema_version": 1,
        "sample_counts": list(SAMPLE_COUNTS),
        "selected_checkpoint_count": len(selected),
        "switch_rule": (
            "use 4096 if any |r4096-r2048|/r4096 > 0.05, "
            "absolute rank difference > 2, or absolute EV1 difference > 0.02"
        ),
        "switch_all_formal_spectral_metrics_to_4096": switch_to_4096,
        "numerical_degeneracy_calibration": {
            "maximum_float64_constant_control_residual": maximum_constant_residual,
            "tau_deg": max(EPS, 100.0 * maximum_constant_residual),
            "formula": "max(1e-12, 100 * maximum float64 constant-control residual)",
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics-dir", type=Path, default=Path("outputs/collapse_audit/point_metrics")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/collapse_audit/sample_stability.json")
    )
    args = parser.parse_args()
    result = run_stability(args.metrics_dir)
    _atomic_json_save(result, args.output)
    print(json.dumps({
        "output": str(args.output),
        "switch_to_4096": result["switch_all_formal_spectral_metrics_to_4096"],
    }, indent=2))


if __name__ == "__main__":
    main()
