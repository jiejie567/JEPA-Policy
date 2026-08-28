"""Paper-facing diagnostic plots for the frozen collapse audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mip.collapse_audit.aggregate import ENDPOINTS, SEEDS, TASKS


LABELS = {
    "mug_mug": "MugMug",
    "moka_moka": "MokaMoka",
    "tool_hang": "ToolHang",
    "square": "Square",
    "coffee_preparation": "CoffeePrep",
    "kitchen": "Kitchen",
    "three_piece_assembly": "ThreePiece",
}


def _records(metrics_dir: Path) -> list[dict]:
    records = []
    for task in TASKS:
        with (metrics_dir / f"{task}.json").open("r", encoding="utf-8") as handle:
            records.extend(json.load(handle)["records"])
    return records


def _save(fig, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_encoder_te32(records: list[dict], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    variants = (("action_only", "Action-only", "#4C78A8"), ("future4_ratio010", "Future4", "#F58518"))
    x = np.arange(len(TASKS))
    for axis, endpoint in zip(axes, ENDPOINTS, strict=True):
        for variant_index, (variant, label, color) in enumerate(variants):
            offset = (-0.14, 0.14)[variant_index]
            means = []
            for task_index, task in enumerate(TASKS):
                values = [
                    record["representations"]["z_t"]["metrics"]["te32"]
                    for record in records
                    if record["task"] == task
                    and record["variant"] == variant
                    and record["checkpoint_kind"] == endpoint
                ]
                if len(values) != len(SEEDS):
                    raise RuntimeError("encoder plot requires three seeds per group")
                means.append(np.mean(values))
                axis.scatter(
                    np.full(len(values), task_index + offset),
                    values,
                    color=color,
                    s=20,
                    alpha=0.65,
                    edgecolors="none",
                )
            axis.plot(x + offset, means, color=color, marker="o", label=label)
        axis.axhline(0.05, color="black", linestyle="--", linewidth=1, label="TE32 boundary")
        axis.set_title(endpoint.capitalize())
        axis.set_xticks(x, [LABELS[task] for task in TASKS], rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Encoder tail energy beyond top 32 (TE32)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Encoder dimensionality audit (dots: seeds; lines: task means)")
    _save(fig, output_dir, "encoder_te32_best_latest")


def plot_predictor(records: list[dict], output_dir: Path) -> None:
    future = [record for record in records if record["variant"] == "future4_ratio010"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
    x = np.arange(len(TASKS))
    for column, endpoint in enumerate(ENDPOINTS):
        for task_index, task in enumerate(TASKS):
            group = [
                record
                for record in future
                if record["task"] == task and record["checkpoint_kind"] == endpoint
            ]
            te32 = [record["representations"]["pred0"]["metrics"]["te32"] for record in group]
            retrieval = [record["retrieval"]["recall_at_5_minus_null"] for record in group]
            axes[0, column].scatter(np.full(3, task_index), te32, color="#54A24B", s=22, alpha=0.7)
            axes[0, column].plot(task_index, np.mean(te32), marker="D", color="black", markersize=4)
            axes[1, column].scatter(np.full(3, task_index), retrieval, color="#E45756", s=22, alpha=0.7)
            axes[1, column].plot(task_index, np.mean(retrieval), marker="D", color="black", markersize=4)
        axes[0, column].axhline(0.05, color="black", linestyle="--", linewidth=1)
        axes[1, column].axhline(0.0, color="black", linestyle="--", linewidth=1)
        axes[0, column].set_title(endpoint.capitalize())
        axes[1, column].set_xticks(x, [LABELS[task] for task in TASKS], rotation=35, ha="right")
        for row in range(2):
            axes[row, column].grid(axis="y", alpha=0.2)
    axes[0, 0].set_ylabel("pred0 TE32")
    axes[1, 0].set_ylabel("Recall@5 observed − null")
    fig.suptitle("Future predictor geometry and sample-specific retrieval")
    _save(fig, output_dir, "predictor_te32_retrieval")


def plot_representative_spectra(records: list[dict], output_dir: Path) -> None:
    candidates = []
    for record in records:
        for name, representation in record["representations"].items():
            candidates.append((representation["metrics"]["te32"], record, name, representation))
    candidates.sort(key=lambda value: value[0])
    chosen = [candidates[0], candidates[len(candidates) // 2], candidates[-1]]
    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    for te32, record, name, representation in chosen:
        cumulative = representation["spectrum"]["cumulative_explained_energy"]
        components = np.arange(1, len(cumulative) + 1)
        label = (
            f"{LABELS[record['task']]} {record['variant']} "
            f"{record['checkpoint_kind']} {name} (TE32={te32:.3f})"
        )
        axis.plot(components, cumulative, label=label)
    axis.axvline(32, color="black", linestyle="--", linewidth=1)
    axis.axhline(0.95, color="gray", linestyle=":", linewidth=1)
    axis.set_xscale("log")
    axis.set_xlim(1, max(len(item[3]["spectrum"]["cumulative_explained_energy"]) for item in chosen))
    axis.set_ylim(0, 1.01)
    axis.set_xlabel("Number of centered singular directions")
    axis.set_ylabel("Cumulative explained energy")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=7)
    axis.set_title("Representative centered spectra")
    _save(fig, output_dir, "representative_centered_spectra")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", type=Path, default=Path("outputs/collapse_audit/point_metrics"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/collapse_audit/report/figures"))
    args = parser.parse_args()
    records = _records(args.metrics_dir)
    plot_encoder_te32(records, args.output_dir)
    plot_predictor(records, args.output_dir)
    plot_representative_spectra(records, args.output_dir)
    print(f"Saved figures under {args.output_dir}")


if __name__ == "__main__":
    main()
