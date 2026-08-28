"""Schedule-independent, domain-separated episode seeds."""

from __future__ import annotations

import hashlib
import json
import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Iterator

import numpy as np
import torch


SEED_MODULUS = 2**31 - 1
SEED_ALGORITHM_VERSION = "future-rollout-audit-seeding/v2"
SELECTION_SEED_DOMAINS = (
    "selection-pilot/v1",
    "selection-evaluation/v1",
)
DEFAULT_AUDIT_SEED_DOMAIN = "future-rollout-audit/audit-master/v1"
STREAM_DOMAINS = (
    "env/v1",
    "policy/v1",
    "python/v1",
    "numpy/v1",
    "torch-cpu/v1",
    "torch-cuda/v1",
)


def _sha256_fields(domain: str, *fields: object) -> bytes:
    encoded = [str(field).encode("utf-8") for field in fields]
    payload = b"\0".join([domain.encode("ascii"), *encoded])
    return hashlib.sha256(payload).digest()


def _seed_from_digest(digest: bytes) -> int:
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value % (SEED_MODULUS - 1) + 1


@dataclass(frozen=True)
class EpisodeSeeds:
    """All deterministic streams for one logical episode."""

    master_sha256: str
    env: int
    policy: int
    python: int
    numpy: int
    torch_cpu: int
    torch_cuda: int

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


def _derive_streams(master_digest: bytes) -> EpisodeSeeds:
    values = {
        domain.split("/", 1)[0].replace("-", "_"): _seed_from_digest(
            hashlib.sha256(
                b"future-rollout-audit/stream/v1\0"
                + domain.encode("ascii")
                + b"\0"
                + master_digest
            ).digest()
        )
        for domain in STREAM_DOMAINS
    }
    return EpisodeSeeds(
        master_sha256=master_digest.hex(),
        env=values["env"],
        policy=values["policy"],
        python=values["python"],
        numpy=values["numpy"],
        torch_cpu=values["torch_cpu"],
        torch_cuda=values["torch_cuda"],
    )


def selection_episode_seeds(
    task: str,
    selection_episode_id: int,
    *,
    seed_domain: str = "selection-evaluation/v1",
) -> EpisodeSeeds:
    """Derive streams shared by every checkpoint evaluated on an episode."""

    if seed_domain not in SELECTION_SEED_DOMAINS:
        raise ValueError(f"Unsupported selection seed domain: {seed_domain!r}")
    master = _sha256_fields(
        f"future-rollout-audit/{seed_domain}/master",
        task,
        int(selection_episode_id),
    )
    return _derive_streams(master)


def audit_episode_seeds(
    task: str,
    training_seed: int,
    audit_episode_id: int,
    *,
    seed_domain: str = DEFAULT_AUDIT_SEED_DOMAIN,
) -> EpisodeSeeds:
    """Derive audit streams independent of workers and scheduling."""

    master = _sha256_fields(
        seed_domain,
        task,
        int(training_seed),
        int(audit_episode_id),
    )
    return _derive_streams(master)


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_selection_seed_manifest(
    task: str,
    episode_ids: Iterator[int] | list[int] | tuple[int, ...],
    *,
    seed_domain: str,
) -> dict:
    """Freeze every domain-separated stream used by a selection episode set."""

    if seed_domain not in SELECTION_SEED_DOMAINS:
        raise ValueError(f"Unsupported selection seed domain: {seed_domain!r}")
    normalized_ids = [int(item) for item in episode_ids]
    if not normalized_ids or normalized_ids != sorted(set(normalized_ids)):
        raise ValueError("selection episode IDs must be non-empty, sorted and unique")
    manifest = {
        "format_version": 1,
        "seed_algorithm_version": SEED_ALGORITHM_VERSION,
        "seed_domain": seed_domain,
        "task": str(task),
        "episode_ids": normalized_ids,
        "episodes": [
            {
                "selection_episode_id": episode_id,
                **selection_episode_seeds(
                    task,
                    episode_id,
                    seed_domain=seed_domain,
                ).to_dict(),
            }
            for episode_id in normalized_ids
        ],
    }
    manifest["selection_seed_manifest_sha256"] = _canonical_hash(manifest)
    return manifest


def validate_selection_seed_manifest(manifest: dict) -> None:
    """Fail closed on altered domains, IDs, streams, or manifest identity."""

    required = {
        "format_version",
        "seed_algorithm_version",
        "seed_domain",
        "task",
        "episode_ids",
        "episodes",
        "selection_seed_manifest_sha256",
    }
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ValueError("selection seed manifest violates its closed schema")
    if manifest["format_version"] != 1:
        raise ValueError("unsupported selection seed manifest format")
    if manifest["seed_algorithm_version"] != SEED_ALGORITHM_VERSION:
        raise ValueError("selection seed algorithm version mismatch")
    expected = build_selection_seed_manifest(
        str(manifest["task"]),
        [int(item) for item in manifest["episode_ids"]],
        seed_domain=str(manifest["seed_domain"]),
    )
    if manifest != expected:
        raise ValueError("selection seed manifest content or SHA256 mismatch")


def seed_process_streams(seeds: EpisodeSeeds) -> None:
    """Seed policy-side global generators from their dedicated streams."""

    random.seed(seeds.python)
    np.random.seed(seeds.numpy)
    torch.manual_seed(seeds.torch_cpu)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seeds.torch_cuda)


def seed_decision_process_streams(seeds: EpisodeSeeds, decision_id: int) -> None:
    """Seed every decision-local framework stream independently."""

    random.seed(decision_stream_seed(seeds, "python/v1", decision_id))
    np.random.seed(decision_stream_seed(seeds, "numpy/v1", decision_id))
    torch.manual_seed(decision_stream_seed(seeds, "torch-cpu/v1", decision_id))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            decision_stream_seed(seeds, "torch-cuda/v1", decision_id)
        )


def decision_stream_seed(
    seeds: EpisodeSeeds, stream: str, decision_id: int
) -> int:
    """Derive a decision-local stream without worker or batch identity."""

    if stream not in STREAM_DOMAINS:
        raise ValueError(f"Unknown random stream: {stream!r}")
    master = bytes.fromhex(seeds.master_sha256)
    digest = hashlib.sha256(
        b"future-rollout-audit/decision-stream/v1\0"
        + stream.encode("ascii")
        + b"\0"
        + str(int(decision_id)).encode("ascii")
        + b"\0"
        + master
    ).digest()
    return _seed_from_digest(digest)


@contextmanager
def preserve_rng_state() -> Iterator[None]:
    """Restore Python, NumPy, CPU Torch and CUDA Torch RNG state on exit."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
