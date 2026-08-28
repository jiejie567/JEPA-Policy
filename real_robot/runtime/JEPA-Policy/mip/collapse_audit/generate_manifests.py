"""Generate and verify the seven frozen collapse-audit manifests."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
from omegaconf import OmegaConf

from mip.collapse_audit.manifests import (
    DEFAULT_MANIFEST_SEED,
    MASTER_SAMPLE_COUNT,
    RETRIEVAL_POOL_SIZE,
    build_retrieval_pool_manifest,
    build_sample_manifest,
)
from mip.dataset_utils import ReplayBuffer, SequenceSampler


def _configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _metadata_dataset(task_config):
    """Build the exact training index sampler without decoding image arrays."""

    dataset_path = Path(str(task_config.dataset_path)).expanduser().resolve()
    with h5py.File(dataset_path, "r") as handle:
        demos = handle["data"]
        total_demos = len(demos)
        validation_count = int(
            total_demos * float(task_config.val_dataset_percentage)
        )
        train_count = total_demos - validation_count
        lengths = [len(demos[f"demo_{index}"]["actions"]) for index in range(train_count)]
    episode_ends = np.cumsum(lengths, dtype=np.int64)
    replay = ReplayBuffer(
        {"data": {}, "meta": {"episode_ends": episode_ends}}
    )
    sampler = SequenceSampler(
        replay,
        sequence_length=int(task_config.horizon),
        pad_before=int(task_config.obs_steps) - 1,
        pad_after=int(task_config.act_steps) - 1,
        keys=(),
    )
    future_steps_list = list(task_config.future_state_steps_list or [])
    future_steps = (
        future_steps_list
        if future_steps_list
        else [int(task_config.future_state_steps)]
    )
    return SimpleNamespace(
        sampler=sampler,
        n_obs_steps=int(task_config.obs_steps),
        future_steps=future_steps,
        dataset_path=dataset_path,
        episode_lengths=lengths,
    )


def _verify_metadata_resolver(dataset, manifest: dict) -> None:
    """Check every resolved source index against its episode boundary."""

    episode_ends = np.asarray(dataset.sampler.replay_buffer.episode_ends)
    for record in manifest["samples"]:
        episode = int(record["episode_index"])
        start = 0 if episode == 0 else int(episode_ends[episode - 1])
        end = int(episode_ends[episode])
        anchor = int(record["current_anchor_frame_index"])
        expected_future = min(anchor + 4, end - 1)
        if not start <= anchor < end:
            raise RuntimeError(f"Current anchor crossed episode boundary: {record}")
        if record["resolved_future_frame_index"] != expected_future:
            raise RuntimeError(
                f"Future4 source mismatch for {record['sample_id']}: "
                f"expected={expected_future}, "
                f"observed={record['resolved_future_frame_index']}"
            )


def generate_all(
    inventory_path: Path,
    output_dir: Path,
    *,
    master_count: int = MASTER_SAMPLE_COUNT,
    pool_size: int = RETRIEVAL_POOL_SIZE,
    seed: int = DEFAULT_MANIFEST_SEED,
) -> None:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    future_runs = {}
    for run in inventory["runs"]:
        if run["variant"] != "future4_ratio010" or int(run["seed"]) != 41:
            continue
        future_runs[run["task"]] = run
    if len(future_runs) != 7:
        raise RuntimeError(f"Expected seven Future4 seed-41 configs, got {future_runs}")

    for task, run in sorted(future_runs.items()):
        logging.info("Loading training dataset for task=%s", task)
        config = OmegaConf.load(run["config_path"])
        dataset = _metadata_dataset(config.task)
        logging.info(
            "Building sample manifest task=%s dataset_samples=%d episodes=%d",
            task,
            len(dataset.sampler),
            len(dataset.sampler.replay_buffer.episode_ends),
        )
        sample_manifest = build_sample_manifest(
            dataset, task, master_count=master_count, seed=seed
        )
        _verify_metadata_resolver(dataset, sample_manifest)
        sample_manifest["source_config_path"] = str(Path(run["config_path"]).resolve())
        sample_manifest["source_dataset_path"] = str(dataset.dataset_path)
        sample_manifest["source_dataset_size_bytes"] = dataset.dataset_path.stat().st_size
        sample_path = output_dir / f"sample_manifest_{task}.json"
        _atomic_json(sample_path, sample_manifest)
        logging.info(
            "Verified and wrote %s sha256=%s padded_rate=%.6f",
            sample_path,
            sample_manifest["sha256"],
            np.mean(
                [
                    record["future_was_padded_or_clamped"]
                    for record in sample_manifest["samples"]
                ]
            ),
        )

        retrieval_manifest = build_retrieval_pool_manifest(
            sample_manifest, pool_size=pool_size, seed=seed
        )
        retrieval_path = output_dir / f"retrieval_pool_manifest_{task}.json"
        _atomic_json(retrieval_path, retrieval_manifest)
        logging.info(
            "Wrote %s sha256=%s",
            retrieval_path,
            retrieval_manifest["sha256"],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--master-count", type=int, default=MASTER_SAMPLE_COUNT)
    parser.add_argument("--pool-size", type=int, default=RETRIEVAL_POOL_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_MANIFEST_SEED)
    args = parser.parse_args()
    _configure_logging(args.log)
    generate_all(
        args.inventory,
        args.output_dir,
        master_count=args.master_count,
        pool_size=args.pool_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
