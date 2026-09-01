"""Extract and score the frozen formal collapse-audit checkpoint inventory."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from mip.agent import TrainingAgent
from mip.collapse_audit.metrics import (
    compute_centered_spectrum,
    compute_control_summary,
    compute_spectral_metrics,
)
from mip.collapse_audit.retrieval import evaluate_fixed_pool_retrieval
from mip.collapse_audit.taps import extract_audit_tensors
from mip.datasets.robot_dataset import make_dataset
from mip.torch_utils import set_seed


LOGGER = logging.getLogger("collapse_audit.formal")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_torch_save(value: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(destination)


def _atomic_json_save(value: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(destination)


def _to_device(value, device: str):
    if isinstance(value, dict):
        return {key: _to_device(child, device) for key, child in value.items()}
    result = value.to(device, non_blocking=True)
    if result.dtype == torch.uint8:
        result = result.float().div_(255.0).mul_(2.0).sub_(1.0)
    return result


def _prepare_batch(batch: dict, config) -> tuple[dict, torch.Tensor, dict]:
    obs = {
        key: value[:, : config.task.obs_steps]
        for key, value in batch["obs"].items()
    }
    act = batch["action"][:, : config.task.horizon]
    return obs, act, batch["future_obs"]


def _materialize_audit_batches(
    dataset,
    sample_manifest: dict,
    *,
    batch_size: int,
) -> list[dict]:
    records = sample_manifest["samples"]
    batches = []
    started = time.perf_counter()
    for start in range(0, len(records), batch_size):
        selected = records[start : start + batch_size]
        batches.append(
            default_collate(
                [dataset[int(record["dataset_sample_index"])] for record in selected]
            )
        )
        if len(batches) % 10 == 0 or start + batch_size >= len(records):
            LOGGER.info(
                "Materialized %d/%d audit samples in %.1fs",
                min(start + batch_size, len(records)),
                len(records),
                time.perf_counter() - started,
            )
    return batches


def _load_formal_ema_weights(
    agent: TrainingAgent,
    checkpoint_path: Path,
    *,
    future_enabled: bool,
) -> None:
    state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    agent.encoder_ema.load_state_dict(state["encoder_ema"])
    if future_enabled:
        agent.flow_map_ema.load_state_dict(state["flow_map_ema"])
    del state
    agent.eval()


def _artifact_is_complete(path: Path, variant: str, sample_count: int) -> bool:
    if not path.is_file():
        return False
    try:
        artifact = torch.load(
            path, map_location="cpu", weights_only=False, mmap=True
        )
        expected = {"z_t"}
        if variant == "future4_ratio010":
            expected.update(("future_target", "pred0"))
        return all(
            key in artifact and artifact[key].shape[0] == sample_count
            for key in expected
        )
    except Exception:
        return False


@torch.no_grad()
def _extract_checkpoint(
    run: dict,
    checkpoint: dict,
    cached_batches: list[dict],
    sample_manifest: dict,
    output_path: Path,
) -> None:
    variant = run["variant"]
    config = OmegaConf.load(run["config_path"])
    config.optimization.future_target_type = getattr(
        config.task, "future_target_type", config.optimization.future_target_type
    )
    config.task.obs_dim = config.network.emb_dim
    set_seed(int(config.optimization.seed))
    agent = TrainingAgent(config)
    _load_formal_ema_weights(
        agent,
        Path(checkpoint["path"]),
        future_enabled=variant == "future4_ratio010",
    )

    representations: dict[str, list[torch.Tensor]] = {"z_t": []}
    if variant == "future4_ratio010":
        representations.update({"future_target": [], "pred0": []})
    started = time.perf_counter()
    for batch_index, cpu_batch in enumerate(cached_batches):
        obs_cpu, act_cpu, future_cpu = _prepare_batch(cpu_batch, config)
        obs = _to_device(obs_cpu, config.optimization.device)
        if variant == "future4_ratio010":
            act = _to_device(act_cpu, config.optimization.device)
            future_obs = _to_device(future_cpu, config.optimization.device)
            tapped = extract_audit_tensors(
                agent,
                act,
                obs,
                future_obs,
                use_ema=True,
            )
            representations["z_t"].append(tapped.z_t.float().cpu())
            representations["future_target"].append(
                tapped.future_target.float().cpu()
            )
            representations["pred0"].append(tapped.pred0.float().cpu())
        else:
            observation_embedding = agent.encoder_ema(obs, None)
            if observation_embedding.dim() != 3:
                raise RuntimeError(
                    "baseline z_t expects [N,T,D], got "
                    f"{tuple(observation_embedding.shape)}"
                )
            representations["z_t"].append(
                observation_embedding[:, -1, :].float().cpu()
            )
        if (batch_index + 1) % 10 == 0 or batch_index + 1 == len(cached_batches):
            LOGGER.info(
                "Extracting %s seed=%s %s: batch %d/%d (%.1fs)",
                variant,
                run["seed"],
                checkpoint["kind"],
                batch_index + 1,
                len(cached_batches),
                time.perf_counter() - started,
            )

    tensors = {
        name: torch.cat(chunks, dim=0)
        for name, chunks in representations.items()
    }
    sample_count = int(sample_manifest["master_count"])
    if any(value.shape[0] != sample_count for value in tensors.values()):
        raise RuntimeError("extracted tensor row count does not match manifest")
    artifact = {
        "schema_version": 1,
        "task": run["task"],
        "seed": int(run["seed"]),
        "variant": variant,
        "checkpoint_kind": checkpoint["kind"],
        "checkpoint_path": checkpoint["path"],
        "checkpoint_size_bytes": int(checkpoint["size_bytes"]),
        "config_path": run["config_path"],
        "sample_manifest_sha256": sample_manifest["sha256"],
        "sample_count": sample_count,
        "weights": "ema",
        "tap_definition": {
            "z_t": "observation encoder output after temporal fusion, last observation token, before output heads",
            "future_target": "EMA target at the training pre-reduction future-loss tap",
            "pred0": "EMA zero-input prediction at the training pre-reduction future-loss tap",
            "token_pooling": "none (n_future_tokens=1)",
        },
        **tensors,
    }
    _atomic_torch_save(artifact, output_path)
    LOGGER.info("Saved %s", output_path)
    del agent, artifact, tensors, representations
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_task(
    *,
    task: str,
    inventory_path: Path,
    manifests_dir: Path,
    output_dir: Path,
    batch_size: int,
) -> None:
    inventory = _load_json(inventory_path)
    sample_manifest_path = manifests_dir / f"sample_manifest_{task}.json"
    sample_manifest = _load_json(sample_manifest_path)
    task_runs = sorted(
        (run for run in inventory["runs"] if run["task"] == task),
        key=lambda run: (run["seed"], run["variant"]),
    )
    if len(task_runs) != 6:
        raise RuntimeError(f"Expected six runs for {task}, got {len(task_runs)}")
    source_run = next(
        run
        for run in task_runs
        if run["variant"] == "future4_ratio010" and int(run["seed"]) == 41
    )
    source_config = OmegaConf.load(source_run["config_path"])
    LOGGER.info("Loading dataset once for task=%s", task)
    dataset = make_dataset(source_config.task)
    batches = _materialize_audit_batches(
        dataset, sample_manifest, batch_size=batch_size
    )
    del dataset
    gc.collect()

    for run in task_runs:
        for checkpoint in run["checkpoints"]:
            output_path = (
                output_dir
                / "embeddings"
                / task
                / (
                    f"{run['variant']}_seed{run['seed']}_"
                    f"{checkpoint['kind']}.pt"
                )
            )
            if _artifact_is_complete(
                output_path,
                run["variant"],
                int(sample_manifest["master_count"]),
            ):
                LOGGER.info("Resume skip complete artifact %s", output_path)
                continue
            _extract_checkpoint(
                run, checkpoint, batches, sample_manifest, output_path
            )


def score_task(
    *,
    task: str,
    inventory_path: Path,
    manifests_dir: Path,
    output_dir: Path,
    spectral_sample_count: int,
    null_replicates: int,
) -> None:
    inventory = _load_json(inventory_path)
    sample_manifest = _load_json(
        manifests_dir / f"sample_manifest_{task}.json"
    )
    pool_manifest = _load_json(
        manifests_dir / f"retrieval_pool_manifest_{task}.json"
    )
    records = []
    for run in inventory["runs"]:
        if run["task"] != task:
            continue
        for checkpoint in run["checkpoints"]:
            artifact_path = (
                output_dir
                / "embeddings"
                / task
                / (
                    f"{run['variant']}_seed{run['seed']}_"
                    f"{checkpoint['kind']}.pt"
                )
            )
            artifact = torch.load(
                artifact_path, map_location="cpu", weights_only=False, mmap=True
            )
            record = {
                "task": task,
                "seed": int(run["seed"]),
                "variant": run["variant"],
                "checkpoint_kind": checkpoint["kind"],
                "artifact_path": str(artifact_path.resolve()),
                "spectral_sample_count": spectral_sample_count,
                "representations": {},
            }
            for name in ("z_t", "future_target", "pred0"):
                if name not in artifact:
                    continue
                embedding = artifact[name][:spectral_sample_count]
                record["representations"][name] = {
                    "metrics": compute_spectral_metrics(embedding).to_dict(),
                    "controls": compute_control_summary(embedding),
                    "spectrum": compute_centered_spectrum(embedding),
                }
            if run["variant"] == "future4_ratio010":
                record["retrieval"] = evaluate_fixed_pool_retrieval(
                    artifact["pred0"],
                    artifact["future_target"],
                    sample_manifest,
                    pool_manifest,
                    null_replicates=null_replicates,
                    null_seed=20260729,
                ).to_dict()
                padded_by_id = {
                    sample["sample_id"]: bool(
                        sample["future_was_padded_or_clamped"]
                    )
                    for sample in sample_manifest["samples"]
                }
                padded_count = sum(padded_by_id.values())
                robustness = {
                    "audit_sample_padded_or_clamped_count": padded_count,
                    "audit_sample_padded_or_clamped_fraction": (
                        padded_count / len(padded_by_id)
                    ),
                    "query_strata": {},
                }
                for stratum, keep_padded in (
                    ("unpadded_only", False), ("padded_or_clamped_only", True)
                ):
                    stratum_pools = [
                        pool
                        for pool in pool_manifest["pools"]
                        if padded_by_id[pool["query_sample_id"]] is keep_padded
                    ]
                    if not stratum_pools:
                        robustness["query_strata"][stratum] = {
                            "query_count": 0,
                            "retrieval": None,
                        }
                        continue
                    filtered_manifest = {
                        **pool_manifest,
                        "pools": stratum_pools,
                    }
                    robustness["query_strata"][stratum] = {
                        "query_count": len(stratum_pools),
                        "retrieval": evaluate_fixed_pool_retrieval(
                            artifact["pred0"],
                            artifact["future_target"],
                            sample_manifest,
                            filtered_manifest,
                            null_replicates=null_replicates,
                            null_seed=20260729,
                        ).to_dict(),
                    }
                record["retrieval_robustness"] = robustness
            records.append(record)
            LOGGER.info(
                "Scored task=%s variant=%s seed=%s checkpoint=%s",
                task,
                run["variant"],
                run["seed"],
                checkpoint["kind"],
            )
    _atomic_json_save(
        {
            "schema_version": 1,
            "task": task,
            "spectral_sample_count": spectral_sample_count,
            "retrieval_sample_count": int(sample_manifest["master_count"]),
            "retrieval_null_replicates": null_replicates,
            "records": records,
        },
        output_dir / "point_metrics" / f"{task}.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("extract", "score", "all"))
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--inventory", type=Path, default=Path("outputs/collapse_audit/inventory.json")
    )
    parser.add_argument(
        "--manifests-dir",
        type=Path,
        default=Path("outputs/collapse_audit/manifests"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/collapse_audit")
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--spectral-sample-count", type=int, default=2048)
    parser.add_argument("--null-replicates", type=int, default=1000)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.mode in ("extract", "all"):
        extract_task(
            task=args.task,
            inventory_path=args.inventory,
            manifests_dir=args.manifests_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )
    if args.mode in ("score", "all"):
        score_task(
            task=args.task,
            inventory_path=args.inventory,
            manifests_dir=args.manifests_dir,
            output_dir=args.output_dir,
            spectral_sample_count=args.spectral_sample_count,
            null_replicates=args.null_replicates,
        )


if __name__ == "__main__":
    main()
