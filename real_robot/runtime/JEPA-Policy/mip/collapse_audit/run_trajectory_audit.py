"""Audit the four preregistered 50k online-weight training trajectories."""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from mip.agent import TrainingAgent
from mip.collapse_audit.bootstrap import episode_bootstrap_spectral
from mip.collapse_audit.fixed_noise import (
    build_fixed_noise_artifact,
    save_fixed_noise_artifact,
    validate_fixed_noise_artifact,
)
from mip.collapse_audit.metrics import (
    compute_control_summary,
    compute_spectral_metrics,
)
from mip.collapse_audit.retrieval import evaluate_fixed_pool_retrieval
from mip.collapse_audit.run_formal_audit import (
    _atomic_json_save,
    _atomic_torch_save,
    _load_json,
    _materialize_audit_batches,
    _prepare_batch,
    _to_device,
)
from mip.collapse_audit.taps import extract_audit_tensors
from mip.datasets.robot_dataset import make_dataset
from mip.torch_utils import set_seed


LOGGER = logging.getLogger("collapse_audit.trajectory")
SNAPSHOT_STEPS = (0, 1000, 5000, 10000, 25000, 50000)
RUNS = (
    (
        "moka_moka",
        42,
        Path(
            "logs/collapse-audit/libero/"
            "moka_moka_image_ratio010_seed42_trajectory50k"
        ),
    ),
    (
        "moka_moka",
        43,
        Path(
            "logs/collapse-audit/libero/"
            "moka_moka_image_ratio010_seed43_trajectory50k"
        ),
    ),
    (
        "tool_hang",
        42,
        Path(
            "logs/collapse-audit/robomimic/"
            "tool_hang_ph_image_ratio010_seed42_trajectory50k"
        ),
    ),
    (
        "tool_hang",
        43,
        Path(
            "logs/collapse-audit/robomimic/"
            "tool_hang_ph_image_ratio010_seed43_trajectory50k"
        ),
    ),
)


def _snapshot_path(run_dir: Path, step: int) -> Path:
    return run_dir / "models" / f"model_collapse_step{step:06d}.pt"


def validate_trajectory_inputs() -> None:
    missing = []
    for _task, _seed, run_dir in RUNS:
        if not (run_dir / "resolved_config.yaml").is_file():
            missing.append(str(run_dir / "resolved_config.yaml"))
        for step in SNAPSHOT_STEPS:
            if not _snapshot_path(run_dir, step).is_file():
                missing.append(str(_snapshot_path(run_dir, step)))
    if missing:
        preview = "\n".join(missing[:12])
        raise FileNotFoundError(
            f"Trajectory audit is not ready; missing {len(missing)} inputs:\n{preview}"
        )


def _load_online_weights(agent: TrainingAgent, checkpoint_path: Path) -> None:
    state = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    agent.encoder.load_state_dict(state["encoder"])
    agent.flow_map.load_state_dict(state["flow_map"])
    del state
    agent.eval()


def _load_or_create_fixed_noise(
    path: Path,
    *,
    manifest_sha256: str,
    action_shape: tuple[int, ...],
    future_shape: tuple[int, ...],
) -> dict:
    if path.is_file():
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        validate_fixed_noise_artifact(
            artifact,
            sample_manifest_sha256=manifest_sha256,
            action_shape=action_shape,
            future_shape=future_shape,
        )
        return artifact
    artifact = build_fixed_noise_artifact(
        sample_manifest_sha256=manifest_sha256,
        action_shape=action_shape,
        future_shape=future_shape,
        seed=20260729,
    )
    save_fixed_noise_artifact(path, artifact)
    return artifact


def _training_log_at_steps(run_dir: Path) -> dict[str, dict]:
    path = run_dir / "metrics.jsonl"
    if not path.is_file():
        return {}
    entries = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "loss_future_raw" in entry:
                entries.append(entry)
    result = {}
    for step in SNAPSHOT_STEPS:
        if not entries:
            break
        nearest = min(entries, key=lambda entry: abs((int(entry["step"]) + 1) - step))
        result[str(step)] = {
            "logged_step": int(nearest["step"]) + 1,
            "loss_future_raw": float(nearest["loss_future_raw"]),
            "auxiliary_only": True,
        }
    return result


@torch.no_grad()
def extract_run(
    *,
    task: str,
    seed: int,
    run_dir: Path,
    cached_batches: list[dict],
    sample_manifest: dict,
    pool_manifest: dict,
    output_dir: Path,
    spectral_sample_count: int,
    null_replicates: int,
) -> None:
    config = OmegaConf.load(run_dir / "resolved_config.yaml")
    config.optimization.future_target_type = getattr(
        config.task, "future_target_type", config.optimization.future_target_type
    )
    config.task.obs_dim = config.network.emb_dim
    set_seed(int(config.optimization.seed))
    agent = TrainingAgent(config)
    first_obs_cpu, first_act_cpu, first_future_cpu = _prepare_batch(
        cached_batches[0], config
    )
    first_obs = _to_device(first_obs_cpu, config.optimization.device)
    first_act = _to_device(first_act_cpu, config.optimization.device)
    first_future = _to_device(first_future_cpu, config.optimization.device)
    preview = extract_audit_tensors(
        agent, first_act, first_obs, first_future, use_ema=False
    )
    sample_count = int(sample_manifest["master_count"])
    action_shape = (sample_count,) + tuple(first_act.shape[1:])
    future_shape = (sample_count,) + tuple(preview.future_target.shape[1:])
    noise_path = output_dir / "fixed_noise" / f"{task}.pt"
    fixed_noise = _load_or_create_fixed_noise(
        noise_path,
        manifest_sha256=sample_manifest["sha256"],
        action_shape=action_shape,
        future_shape=future_shape,
    )
    training_log = _training_log_at_steps(run_dir)

    run_records = []
    episode_ids = np.asarray(
        [
            int(sample["episode_index"])
            for sample in sample_manifest["samples"][:spectral_sample_count]
        ]
    )
    for step in SNAPSHOT_STEPS:
        checkpoint_path = _snapshot_path(run_dir, step)
        artifact_path = (
            output_dir
            / "trajectory"
            / "embeddings"
            / f"{task}_seed{seed}_step{step:06d}.pt"
        )
        if artifact_path.is_file():
            artifact = torch.load(
                artifact_path, map_location="cpu", weights_only=False, mmap=True
            )
        else:
            _load_online_weights(agent, checkpoint_path)
            chunks = {"z_t": [], "future_target": [], "pred0": []}
            losses = {"pred0_raw": 0.0, "pred1_raw": 0.0, "total_raw": 0.0}
            seen = 0
            for cpu_batch in cached_batches:
                obs_cpu, act_cpu, future_cpu = _prepare_batch(cpu_batch, config)
                batch_count = act_cpu.shape[0]
                obs = _to_device(obs_cpu, config.optimization.device)
                act = _to_device(act_cpu, config.optimization.device)
                future_obs = _to_device(future_cpu, config.optimization.device)
                action_noise = fixed_noise["action_noise"][seen : seen + batch_count].to(
                    config.optimization.device
                )
                future_noise = fixed_noise["future_noise"][seen : seen + batch_count].to(
                    config.optimization.device
                )
                tapped = extract_audit_tensors(
                    agent,
                    act,
                    obs,
                    future_obs,
                    use_ema=False,
                    action_noise=action_noise,
                    future_noise=future_noise,
                )
                chunks["z_t"].append(tapped.z_t.float().cpu())
                chunks["future_target"].append(tapped.future_target.float().cpu())
                chunks["pred0"].append(tapped.pred0.float().cpu())
                losses["pred0_raw"] += float(tapped.future_loss_pred0_raw) * batch_count
                losses["pred1_raw"] += float(tapped.future_loss_pred1_raw) * batch_count
                losses["total_raw"] += float(tapped.future_loss_total_raw) * batch_count
                seen += batch_count
            if seen != sample_count:
                raise RuntimeError(f"Expected {sample_count} samples, extracted {seen}")
            artifact = {
                "schema_version": 1,
                "task": task,
                "seed": seed,
                "step": step,
                "weights": "online",
                "checkpoint_path": str(checkpoint_path.resolve()),
                "sample_manifest_sha256": sample_manifest["sha256"],
                "fixed_noise_path": str(noise_path.resolve()),
                "fixed_noise_hashes": {
                    "action": fixed_noise["action_noise_sha256"],
                    "future": fixed_noise["future_noise_sha256"],
                },
                "fixed_audit_future_loss": {
                    name: value / sample_count for name, value in losses.items()
                },
                **{name: torch.cat(value, dim=0) for name, value in chunks.items()},
            }
            _atomic_torch_save(artifact, artifact_path)

        representations = {}
        for name in ("z_t", "future_target", "pred0"):
            embedding = artifact[name][:spectral_sample_count]
            representations[name] = {
                "metrics": compute_spectral_metrics(embedding).to_dict(),
                "controls": compute_control_summary(embedding),
                "bootstrap": episode_bootstrap_spectral(
                    embedding,
                    episode_ids,
                    replicates=1000 if name in ("z_t", "pred0") else 200,
                    seed=20260729 + step + seed,
                    batch_size=8,
                ),
            }
        retrieval = evaluate_fixed_pool_retrieval(
            artifact["pred0"],
            artifact["future_target"],
            sample_manifest,
            pool_manifest,
            null_replicates=null_replicates,
            null_seed=20260729,
        ).to_dict()
        run_records.append(
            {
                "task": task,
                "seed": seed,
                "step": step,
                "artifact_path": str(artifact_path.resolve()),
                "representations": representations,
                "fixed_audit_future_loss": artifact["fixed_audit_future_loss"],
                "training_log_auxiliary": training_log.get(str(step)),
                "retrieval": retrieval,
            }
        )
        LOGGER.info("Completed trajectory audit task=%s seed=%d step=%d", task, seed, step)
    _atomic_json_save(
        {
            "schema_version": 1,
            "task": task,
            "seed": seed,
            "weights": "online",
            "spectral_sample_count": spectral_sample_count,
            "records": run_records,
        },
        output_dir / "trajectory" / "point_metrics" / f"{task}_seed{seed}.json",
    )
    del agent
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("validate", "run"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/collapse_audit"))
    parser.add_argument("--manifests-dir", type=Path, default=Path("outputs/collapse_audit/manifests"))
    parser.add_argument("--spectral-sample-count", type=int, default=4096)
    parser.add_argument("--null-replicates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    validate_trajectory_inputs()
    if args.mode == "validate":
        print("All four trajectory runs and 24 snapshots are ready")
        return
    for task in ("moka_moka", "tool_hang"):
        matching_runs = [run for run in RUNS if run[0] == task]
        config = OmegaConf.load(matching_runs[0][2] / "resolved_config.yaml")
        sample_manifest = _load_json(args.manifests_dir / f"sample_manifest_{task}.json")
        pool_manifest = _load_json(args.manifests_dir / f"retrieval_pool_manifest_{task}.json")
        dataset = make_dataset(config.task)
        batches = _materialize_audit_batches(
            dataset, sample_manifest, batch_size=args.batch_size
        )
        del dataset
        for _, seed, run_dir in matching_runs:
            extract_run(
                task=task,
                seed=seed,
                run_dir=run_dir,
                cached_batches=batches,
                sample_manifest=sample_manifest,
                pool_manifest=pool_manifest,
                output_dir=args.output_dir,
                spectral_sample_count=args.spectral_sample_count,
                null_replicates=args.null_replicates,
            )


if __name__ == "__main__":
    main()
