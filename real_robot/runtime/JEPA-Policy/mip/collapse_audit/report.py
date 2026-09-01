"""Create auditable CSV and Markdown summaries from frozen point results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from mip.collapse_audit.aggregate import ENDPOINTS, SEEDS, TASKS


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _records(metrics_dir: Path) -> list[dict]:
    records = []
    for task in TASKS:
        records.extend(_load_json(metrics_dir / f"{task}.json")["records"])
    return records


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_rows(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        for name, representation in record["representations"].items():
            metrics = representation["metrics"]
            row = {
                "task": record["task"],
                "seed": record["seed"],
                "variant": record["variant"],
                "checkpoint": record["checkpoint_kind"],
                "representation": name,
                "n_samples": metrics["n_samples"],
                "dimension": metrics["dimension"],
                "relative_centered_energy": metrics["relative_centered_energy"],
                "centered_effective_rank": metrics["centered_effective_rank"],
                "normalized_effective_rank": metrics["normalized_effective_rank"],
                "rankme_raw": metrics["rankme_raw"],
                "ev1": metrics["ev1"],
                "te8": metrics["te8"],
                "te32": metrics["te32"],
                "delta_r32": representation["controls"]["rank_controls"]["rank32"]["learned_minus_control_effective_rank"],
                "degenerate": metrics["degenerate"],
                "retrieval_recall1": "",
                "retrieval_recall5": "",
                "retrieval_median_rank": "",
                "retrieval_recall5_minus_null": "",
                "retrieval_margin_minus_null": "",
            }
            if name == "pred0" and "retrieval" in record:
                retrieval = record["retrieval"]
                row.update(
                    {
                        "retrieval_recall1": retrieval["recall_at_1"],
                        "retrieval_recall5": retrieval["recall_at_5"],
                        "retrieval_median_rank": retrieval["median_rank"],
                        "retrieval_recall5_minus_null": retrieval["recall_at_5_minus_null"],
                        "retrieval_margin_minus_null": retrieval["margin_minus_null"],
                    }
                )
            rows.append(row)
    return rows


def task_rows(checkpoints: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in checkpoints:
        key = (
            row["task"],
            row["variant"],
            row["checkpoint"],
            row["representation"],
        )
        grouped[key].append(row)
    rows = []
    for key, group in sorted(grouped.items()):
        if sorted(int(row["seed"]) for row in group) != list(SEEDS):
            raise RuntimeError(f"Task group does not contain seeds 41/42/43: {key}")
        row = {
            "task": key[0],
            "variant": key[1],
            "checkpoint": key[2],
            "representation": key[3],
            "seed_count": len(group),
        }
        for metric in (
            "relative_centered_energy",
            "centered_effective_rank",
            "normalized_effective_rank",
            "ev1",
            "te8",
            "te32",
            "delta_r32",
        ):
            values = np.asarray([float(item[metric]) for item in group])
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_min"] = float(values.min())
        retrieval_values = [
            float(item["retrieval_recall5_minus_null"])
            for item in group
            if item["retrieval_recall5_minus_null"] != ""
        ]
        row["retrieval_recall5_minus_null_mean"] = (
            float(np.mean(retrieval_values)) if retrieval_values else ""
        )
        row["te32_exception_2_of_3"] = (
            sum(float(item["te32"]) <= 0.05 for item in group) >= 2
        )
        row["retrieval_exception_2_of_3"] = (
            bool(retrieval_values)
            and sum(value <= 0.0 for value in retrieval_values) >= 2
        )
        rows.append(row)
    return rows


def _exception_lines(summary: dict) -> list[str]:
    lines = []
    for category, exceptions in summary["exceptions"].items():
        if not exceptions:
            lines.append(f"- {category}: none")
            continue
        for exception in exceptions:
            lines.append(
                f"- {category}: {exception['task']} / {exception['endpoint']} "
                f"({exception['reason']})"
            )
    return lines


def markdown_report(
    summary: dict,
    stability: dict,
    task_summary_rows: list[dict],
) -> str:
    lines = [
        "# JEPA-Policy collapse audit summary",
        "",
        "This report follows the frozen seven-task, three-seed, Future4 "
        "ratio=0.1 specification. Best and latest are not averaged.",
        "",
        "## Gates",
        "",
        "| Gate | Result |",
        "|---|---:|",
    ]
    for name, passed in summary["claim_checks"].items():
        lines.append(f"| `{name}` | {'PASS' if passed else 'FAIL'} |")
    calibration = stability["numerical_degeneracy_calibration"]
    lines.extend(
        [
            "",
            f"Spectral sample count selected: **{4096 if stability['switch_all_formal_spectral_metrics_to_4096'] else 2048}**.",
            f"Numerical degeneracy boundary: `{calibration['tau_deg']:.3e}`.",
            "",
            "## Mandatory task-level exceptions",
            "",
            *_exception_lines(summary),
            "",
            "## Task-level means",
            "",
            "| Task | Variant | Endpoint | Representation | TE32 mean | r_norm mean | Retrieval ΔR@5 |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in task_summary_rows:
        retrieval = row["retrieval_recall5_minus_null_mean"]
        retrieval_text = "—" if retrieval == "" else f"{retrieval:.4f}"
        lines.append(
            f"| {row['task']} | {row['variant']} | {row['checkpoint']} | "
            f"{row['representation']} | {row['te32_mean']:.4f} | "
            f"{row['normalized_effective_rank_mean']:.4f} | {retrieval_text} |"
        )
    any_exception = any(summary["exceptions"].values())
    lines.extend(["", "## Wording guard", ""])
    if any_exception:
        lines.append(
            "At least one preregistered task-level exception is present. Do not "
            "write an unqualified ‘across all seven tasks’ no-collapse claim; "
            "report the global average together with the named exceptions."
        )
    else:
        lines.append(
            "No task-level exception is present. The frozen all-seven-task wording "
            "is permitted only for claim gates marked PASS above."
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", type=Path, default=Path("outputs/collapse_audit/point_metrics"))
    parser.add_argument("--global-summary", type=Path, default=Path("outputs/collapse_audit/global_summary.json"))
    parser.add_argument("--stability", type=Path, default=Path("outputs/collapse_audit/sample_stability.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/collapse_audit/report"))
    args = parser.parse_args()
    records = _records(args.metrics_dir)
    checkpoints = checkpoint_rows(records)
    tasks = task_rows(checkpoints)
    _write_csv(args.output_dir / "checkpoint_metrics.csv", checkpoints)
    _write_csv(args.output_dir / "task_metrics.csv", tasks)
    report = markdown_report(
        _load_json(args.global_summary), _load_json(args.stability), tasks
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "summary.md"
    temporary = destination.with_suffix(".md.tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(destination)
    print(f"Saved report under {args.output_dir}")


if __name__ == "__main__":
    main()
