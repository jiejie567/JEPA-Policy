"""Build and validate the frozen 21-pair collapse-audit inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


EXPECTED_TASKS = (
    "mug_mug",
    "moka_moka",
    "tool_hang",
    "square",
    "coffee_preparation",
    "kitchen",
    "three_piece_assembly",
)
EXPECTED_SEEDS = (41, 42, 43)
CHECKPOINT_NAMES = ("model_best.pt", "model_latest.pt")
TASK_ALIASES = {"mimicgen_kitchen": "kitchen"}


class InventoryError(RuntimeError):
    """Raised when the frozen inventory hard gate fails."""


@dataclass(frozen=True)
class CheckpointRecord:
    """One checkpoint selected for the collapse audit."""

    kind: str
    path: str
    size_bytes: int
    required_keys_verified: bool


@dataclass(frozen=True)
class RunRecord:
    """Resolved metadata for one formal run."""

    task: str
    seed: int
    variant: str
    benchmark: str
    run_dir: str
    config_path: str
    configuration_id: str
    compatibility_signature: str
    compatibility_payload: dict[str, Any]
    checkpoints: tuple[CheckpointRecord, ...]


@dataclass(frozen=True)
class PairRecord:
    """A matched Future4/action-only pair."""

    task: str
    seed: int
    baseline_run_dir: str
    future_run_dir: str
    compatible: bool
    differences: dict[str, dict[str, Any]]


def _plain_config(path: Path) -> dict[str, Any]:
    config = OmegaConf.load(path)
    return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]


def _canonical_task(config: dict[str, Any]) -> str:
    """Map historical config names onto the frozen paper task names."""

    raw_name = str(_get(config, "task.env_name", ""))
    return TASK_ALIASES.get(raw_name, raw_name)


def _get(config: dict[str, Any], dotted: str, default: Any = None) -> Any:
    value: Any = config
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _dataset_identity(config: dict[str, Any]) -> dict[str, Any]:
    raw_path = _get(config, "task.dataset_path")
    if not raw_path:
        return {
            "path": None,
            "dataset_repo": _get(config, "task.dataset_repo"),
            "dataset_filename": _get(config, "task.dataset_filename"),
        }

    path = Path(str(raw_path)).expanduser().resolve()
    identity: dict[str, Any] = {"path": str(path)}
    if path.exists():
        stat = path.stat()
        identity.update(
            {
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    else:
        identity["missing"] = True
    return identity


def _camera_keys(config: dict[str, Any]) -> list[str]:
    obs_meta = _get(config, "task.shape_meta.obs", {})
    if not isinstance(obs_meta, dict):
        return []
    return [
        key
        for key, value in obs_meta.items()
        if isinstance(value, dict) and value.get("type") == "rgb"
    ]


def _compatibility_payload(config: dict[str, Any]) -> dict[str, Any]:
    """Return fields that must match in a baseline/Future4 pair."""

    network_fields = (
        "network_type",
        "emb_dim",
        "rgb_model_name",
        "rgb_model_weights",
        "imagenet_norm",
        "share_rgb_model",
        "use_seq",
        "keep_horizon_dims",
        "num_encoder_layers",
        "encoder_dropout",
    )
    task_fields = (
        "env_type",
        "obs_type",
        "obs_steps",
        "resize_shape",
        "crop_shape",
        "crop_ratio",
        "random_crop",
        "crop_mode",
        "eval_crop_mode",
        "temporal_consistent_crop",
        "use_group_norm",
        "use_seq",
        "shape_meta",
        "abs_action",
    )
    return {
        "network": {
            field: _get(config, f"network.{field}") for field in network_fields
        },
        "task": {field: _get(config, f"task.{field}") for field in task_fields},
        "camera_keys": _camera_keys(config),
        "dataset": _dataset_identity(config),
    }


def _stable_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _future_steps(config: dict[str, Any]) -> list[int]:
    steps = _get(config, "task.future_state_steps_list", [])
    if steps:
        return [int(step) for step in steps]
    return [int(_get(config, "task.future_state_steps", 1))]


def _classify_variant(config: dict[str, Any]) -> str | None:
    enabled = bool(_get(config, "task.future_state_enabled", False))
    use_loss = bool(_get(config, "optimization.use_future_embed_loss", False))
    token_count = int(_get(config, "network.n_future_tokens", 0))
    if not enabled and not use_loss and token_count == 0:
        return "action_only"

    is_frozen_future = all(
        (
            enabled,
            use_loss,
            token_count == 1,
            _future_steps(config) == [4],
            _get(config, "optimization.future_embed_loss_mode") == "mip_two_step",
            bool(_get(config, "optimization.future_joint_mode", False)),
            _get(config, "optimization.future_state_loss_mode") == "ratio",
            abs(
                float(_get(config, "optimization.future_state_loss_ratio", -1.0))
                - 0.1
            )
            < 1e-12,
        )
    )
    return "future4_ratio010" if is_frozen_future else None


def _configuration_id(config: dict[str, Any], variant: str) -> str:
    task = str(_get(config, "task.env_name"))
    if variant == "action_only":
        horizon = "none"
        ratio = "none"
    else:
        horizon = "4"
        ratio = "0.1"
    mode = _get(config, "optimization.future_embed_loss_mode", "none")
    head = "joint" if _get(config, "optimization.future_joint_mode", False) else "none"
    return f"{task}:{variant}:h{horizon}:r{ratio}:{mode}:{head}"


def _checkpoint_records(
    run_dir: Path, verify_checkpoint_keys: bool
) -> tuple[CheckpointRecord, ...]:
    records = []
    for filename in CHECKPOINT_NAMES:
        path = run_dir / "models" / filename
        if not path.is_file() or path.stat().st_size <= 0:
            raise InventoryError(f"Missing or empty checkpoint: {path}")
        verified = False
        if verify_checkpoint_keys:
            import torch

            # The formal checkpoints are 1.3--1.4 GiB each. We only inspect
            # top-level keys here, so mmap their storages instead of reading
            # every tensor into RAM.
            state = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
            required = {"encoder", "encoder_ema", "flow_map", "flow_map_ema"}
            missing = sorted(required.difference(state))
            del state
            if missing:
                raise InventoryError(
                    f"Checkpoint {path} is missing required keys: {missing}"
                )
            verified = True
        records.append(
            CheckpointRecord(
                kind=filename.removeprefix("model_").removesuffix(".pt"),
                path=str(path.resolve()),
                size_bytes=path.stat().st_size,
                required_keys_verified=verified,
            )
        )
    return tuple(records)


def _differences(left: Any, right: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(left, dict) and isinstance(right, dict):
        result = {}
        for key in sorted(set(left) | set(right)):
            child = f"{prefix}.{key}" if prefix else key
            result.update(_differences(left.get(key), right.get(key), child))
        return result
    if left == right:
        return {}
    return {prefix: {"baseline": left, "future": right}}


def build_inventory(
    repo_root: str | Path,
    *,
    verify_checkpoint_keys: bool = True,
) -> dict[str, Any]:
    """Discover and validate the frozen 21-pair inventory."""

    repo_root = Path(repo_root).resolve()
    search_roots = (
        repo_root / "logs" / "ablation2" / "libero",
        repo_root / "logs" / "ablation2" / "robomimic",
        repo_root / "logs" / "mimicgen",
    )
    selected: dict[tuple[str, int, str], RunRecord] = {}

    for search_root in search_roots:
        if not search_root.is_dir():
            raise InventoryError(f"Missing formal-run root: {search_root}")
        for config_path in sorted(search_root.glob("*/resolved_config.yaml")):
            config = _plain_config(config_path)
            task = _canonical_task(config)
            seed = int(_get(config, "optimization.seed", -1))
            if task not in EXPECTED_TASKS or seed not in EXPECTED_SEEDS:
                continue
            variant = _classify_variant(config)
            if variant is None:
                continue
            key = (task, seed, variant)
            if key in selected:
                raise InventoryError(
                    f"Duplicate formal run for {key}: "
                    f"{selected[key].run_dir} and {config_path.parent}"
                )
            payload = _compatibility_payload(config)
            selected[key] = RunRecord(
                task=task,
                seed=seed,
                variant=variant,
                benchmark=str(_get(config, "task.env_type", "unknown")),
                run_dir=str(config_path.parent.resolve()),
                config_path=str(config_path.resolve()),
                configuration_id=_configuration_id(config, variant),
                compatibility_signature=_stable_hash(payload),
                compatibility_payload=payload,
                checkpoints=_checkpoint_records(
                    config_path.parent, verify_checkpoint_keys
                ),
            )

    expected_keys = {
        (task, seed, variant)
        for task in EXPECTED_TASKS
        for seed in EXPECTED_SEEDS
        for variant in ("action_only", "future4_ratio010")
    }
    missing = sorted(expected_keys.difference(selected))
    unexpected = sorted(set(selected).difference(expected_keys))
    if missing or unexpected:
        raise InventoryError(
            f"Frozen inventory mismatch: missing={missing}, unexpected={unexpected}"
        )

    pairs = []
    for task in EXPECTED_TASKS:
        for seed in EXPECTED_SEEDS:
            baseline = selected[(task, seed, "action_only")]
            future = selected[(task, seed, "future4_ratio010")]
            differences = _differences(
                baseline.compatibility_payload, future.compatibility_payload
            )
            pairs.append(
                PairRecord(
                    task=task,
                    seed=seed,
                    baseline_run_dir=baseline.run_dir,
                    future_run_dir=future.run_dir,
                    compatible=not differences,
                    differences=differences,
                )
            )

    incompatible = [pair for pair in pairs if not pair.compatible]
    if incompatible:
        summary = {
            f"{pair.task}:seed{pair.seed}": pair.differences
            for pair in incompatible
        }
        raise InventoryError(
            "Matched-pair compatibility hard gate failed:\n"
            + json.dumps(summary, indent=2, sort_keys=True)
        )

    runs = [selected[key] for key in sorted(selected)]
    checkpoint_count = sum(len(run.checkpoints) for run in runs)
    if len(runs) != 42 or len(pairs) != 21 or checkpoint_count != 84:
        raise InventoryError(
            "Frozen count assertion failed: "
            f"runs={len(runs)}, pairs={len(pairs)}, checkpoints={checkpoint_count}"
        )

    return {
        "schema_version": 1,
        "repo_root": str(repo_root),
        "checkpoint_keys_verified": verify_checkpoint_keys,
        "counts": {
            "runs": len(runs),
            "future_runs": sum(run.variant == "future4_ratio010" for run in runs),
            "baseline_runs": sum(run.variant == "action_only" for run in runs),
            "pairs": len(pairs),
            "checkpoints": checkpoint_count,
            "representation_matrices": 168,
        },
        "runs": [asdict(run) for run in runs],
        "pairs": [asdict(pair) for pair in pairs],
    }


def _configure_logging(log_path: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path)
    parser.add_argument(
        "--skip-checkpoint-load",
        action="store_true",
        help="Validate checkpoint presence and size without loading state dicts.",
    )
    args = parser.parse_args()
    _configure_logging(args.log)

    logging.info("Building frozen collapse-audit inventory")
    inventory = build_inventory(
        args.repo_root,
        verify_checkpoint_keys=not args.skip_checkpoint_load,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logging.info("Inventory hard gate passed: %s", inventory["counts"])
    logging.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
