"""Command-line entry points for current-stage audit infrastructure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _command_inventory(args) -> None:
    from mip.future_rollout_audit.inventory import (
        discover_inventory,
        inventory_csv_bytes,
    )
    from mip.future_rollout_audit.schemas import write_canonical_json

    inventory = discover_inventory(
        args.repo_root,
        content_hash=not args.skip_content_hash,
    )
    output = Path(args.output)
    write_canonical_json(output, inventory)
    output.with_suffix(".csv").write_bytes(inventory_csv_bytes(inventory))


def _smoke_entry(candidate: dict) -> dict:
    if candidate["checkpoint_fingerprint_kind"] != "sha256":
        raise ValueError("Smoke manifest requires content-hashed checkpoints")
    return {
        "task": candidate["task"],
        "training_seed": candidate["training_seed"],
        "checkpoint_path": candidate["checkpoint_path"],
        "checkpoint_fingerprint": candidate["checkpoint_fingerprint"],
        "entry_status": "smoke_only_unselected",
    }


def _command_manifests(args) -> None:
    from mip.future_rollout_audit.inventory import (
        blocked_formal_manifest,
        smoke_manifest,
    )
    from mip.future_rollout_audit.schemas import write_canonical_json

    inventory = _load_json(args.inventory)
    by_key = {
        (
            item["task"],
            item["training_seed"],
            Path(item["checkpoint_path"]).name,
        ): item
        for item in inventory["candidates"]
    }
    requested = (
        ("moka_moka", 42, "model_latest.pt"),
        ("square", 42, "model_latest.pt"),
    )
    missing = [key for key in requested if key not in by_key]
    if missing:
        raise ValueError(f"Inventory lacks required smoke checkpoints: {missing}")
    output_dir = Path(args.output_dir)
    write_canonical_json(
        output_dir / "formal_manifest.json",
        blocked_formal_manifest(inventory["source_snapshot_max_mtime_utc"]),
    )
    write_canonical_json(
        output_dir / "smoke_manifest.json",
        smoke_manifest(
            inventory["source_snapshot_max_mtime_utc"],
            [_smoke_entry(by_key[key]) for key in requested],
        ),
    )


def _command_smoke(args) -> None:
    from mip.future_rollout_audit.runner import run_smoke_checkpoint
    from mip.future_rollout_audit.schemas import (
        validate_manifest,
        write_canonical_json,
    )

    manifest = _load_json(args.manifest)
    validate_manifest(manifest)
    if manifest["manifest_kind"] != "smoke_only":
        raise ValueError("Smoke runner only accepts manifest_kind=smoke_only")
    output_dir = Path(args.output_dir)
    selected_entries = [
        entry
        for entry in manifest["entries"]
        if (args.task is None or entry["task"] == args.task)
        and (
            args.training_seed is None
            or entry["training_seed"] == args.training_seed
        )
    ]
    if not selected_entries:
        raise ValueError("Smoke manifest has no entry matching the requested filter")
    results = []
    for entry in selected_entries:
        checkpoint = Path(entry["checkpoint_path"])
        config = checkpoint.parent.parent / "resolved_config.yaml"
        result = run_smoke_checkpoint(
            config_path=config,
            checkpoint_path=checkpoint,
            episode_count=args.episode_count,
            audit_seed_domain=args.seed_domain,
        )
        destination = output_dir / f"{entry['task']}_seed{entry['training_seed']}.json"
        write_canonical_json(destination, result)
        results.append(
            {
                "task": entry["task"],
                "training_seed": entry["training_seed"],
                "result_path": str(destination.resolve()),
                "episode_count": result["episode_count"],
                "primary_eligible_count": result["primary_eligible_count"],
            }
        )
    summary_name = (
        "smoke_summary.json"
        if args.task is None
        else f"smoke_summary_{args.task}_seed{args.training_seed}.json"
    )
    write_canonical_json(
        output_dir / summary_name,
        {
            "format_version": 1,
            "run_kind": "smoke_only",
            "results": results,
        },
    )


def _command_formal(args) -> None:
    from mip.future_rollout_audit.runner import require_formal_manifest

    manifest = _load_json(args.manifest)
    require_formal_manifest(manifest)
    raise RuntimeError("formal_audit_not_enabled_in_current_stage")


def _command_smoke_report(args) -> None:
    from mip.future_rollout_audit.provenance import file_sha256
    from mip.future_rollout_audit.runner import smoke_report
    from mip.future_rollout_audit.schemas import write_canonical_json

    source = Path(args.input)
    result = smoke_report(
        _load_json(source),
        raw_result_path=str(source.resolve()),
        raw_result_sha256=file_sha256(source),
    )
    write_canonical_json(args.output, result)


def _command_freeze_provenance(args) -> None:
    from mip.future_rollout_audit.provenance import build_environment_provenance
    from mip.future_rollout_audit.schemas import write_canonical_json_atomic

    write_canonical_json_atomic(
        args.output,
        build_environment_provenance(args.repo_root),
    )


def _command_selection_seed_manifest(args) -> None:
    from mip.future_rollout_audit.schemas import write_canonical_json_atomic
    from mip.future_rollout_audit.seeding import build_selection_seed_manifest

    manifest = build_selection_seed_manifest(
        args.task,
        range(args.episode_count),
        seed_domain=args.seed_domain,
    )
    write_canonical_json_atomic(args.output, manifest)


def _command_selection(args) -> None:
    from mip.future_rollout_audit.selection_runner import run_selection_checkpoint

    inventory = _load_json(args.inventory)
    requested = str(Path(args.checkpoint_path).resolve())
    matches = [
        item
        for item in inventory["candidates"]
        if str(Path(item["checkpoint_path"]).resolve()) == requested
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one inventory candidate for {requested!r}, found {len(matches)}"
        )
    result = run_selection_checkpoint(
        repo_root=args.repo_root,
        candidate=matches[0],
        seed_manifest=_load_json(args.seed_manifest),
        seed_manifest_path=args.seed_manifest,
        environment_provenance=_load_json(args.environment_provenance),
        output_path=args.output,
        evidence_kind=args.evidence_kind,
        worker_count=args.worker_count,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "evidence_kind": result["evidence_kind"],
                "usable_for_formal_selection": result[
                    "usable_for_formal_selection"
                ],
                "episode_count": result["episode_count"],
                "success_count": result["success_count"],
                "output": str(Path(args.output).resolve()),
            },
            sort_keys=True,
        )
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("--repo-root", default=".")
    inventory.add_argument("--output", required=True)
    inventory.add_argument("--skip-content-hash", action="store_true")
    inventory.set_defaults(function=_command_inventory)

    manifests = subparsers.add_parser("manifests")
    manifests.add_argument("--inventory", required=True)
    manifests.add_argument("--output-dir", required=True)
    manifests.set_defaults(function=_command_manifests)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--manifest", required=True)
    smoke.add_argument("--output-dir", required=True)
    smoke.add_argument("--task")
    smoke.add_argument("--training-seed", type=int)
    smoke.add_argument("--episode-count", type=int, default=16)
    smoke.add_argument(
        "--seed-domain",
        default="future-rollout-audit/audit-master/v1",
    )
    smoke.set_defaults(function=_command_smoke)

    smoke_report_parser = subparsers.add_parser("smoke-report")
    smoke_report_parser.add_argument("--input", required=True)
    smoke_report_parser.add_argument("--output", required=True)
    smoke_report_parser.set_defaults(function=_command_smoke_report)

    provenance = subparsers.add_parser("freeze-provenance")
    provenance.add_argument("--repo-root", default=".")
    provenance.add_argument("--output", required=True)
    provenance.set_defaults(function=_command_freeze_provenance)

    seed_manifest = subparsers.add_parser("selection-seed-manifest")
    seed_manifest.add_argument("--task", required=True)
    seed_manifest.add_argument("--episode-count", type=int, required=True)
    seed_manifest.add_argument(
        "--seed-domain",
        required=True,
        choices=("selection-pilot/v1", "selection-evaluation/v1"),
    )
    seed_manifest.add_argument("--output", required=True)
    seed_manifest.set_defaults(function=_command_selection_seed_manifest)

    formal = subparsers.add_parser("formal")
    formal.add_argument("--manifest", required=True)
    formal.set_defaults(function=_command_formal)

    selection = subparsers.add_parser("selection-evaluation")
    selection.add_argument("--repo-root", default=".")
    selection.add_argument("--inventory", required=True)
    selection.add_argument("--checkpoint-path", required=True)
    selection.add_argument("--seed-manifest", required=True)
    selection.add_argument("--environment-provenance", required=True)
    selection.add_argument("--output", required=True)
    selection.add_argument("--worker-count", type=int, default=8)
    selection.add_argument(
        "--evidence-kind",
        required=True,
        choices=("pilot_only", "verified_selection"),
    )
    selection.set_defaults(function=_command_selection)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
