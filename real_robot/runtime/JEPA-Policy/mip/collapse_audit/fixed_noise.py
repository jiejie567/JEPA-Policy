"""Persist deterministic fixed noise tensors for trajectory audit losses."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import torch


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor metadata and contiguous CPU bytes."""
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def build_fixed_noise_artifact(
    *,
    sample_manifest_sha256: str,
    action_shape: Sequence[int],
    future_shape: Sequence[int],
    seed: int = 20260729,
) -> dict:
    """Build device-independent float32 noises tied to one sample manifest."""
    action_shape = tuple(int(value) for value in action_shape)
    future_shape = tuple(int(value) for value in future_shape)
    if not sample_manifest_sha256:
        raise ValueError("sample_manifest_sha256 must be non-empty")
    if not action_shape or not future_shape:
        raise ValueError("noise shapes must be non-empty")
    if action_shape[0] != future_shape[0]:
        raise ValueError("action and future noises must have the same sample count")
    if any(value <= 0 for value in (*action_shape, *future_shape)):
        raise ValueError("noise dimensions must be positive")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    # Preserve the exact random-call order used by the joint training loss.
    future_noise = torch.randn(
        future_shape, generator=generator, dtype=torch.float32
    )
    action_noise = torch.randn(
        action_shape, generator=generator, dtype=torch.float32
    )
    return {
        "format_version": 1,
        "sample_manifest_sha256": sample_manifest_sha256,
        "seed": int(seed),
        "generator_state": generator.get_state(),
        "random_call_order": ["future_noise", "action_noise"],
        "future_noise": future_noise,
        "action_noise": action_noise,
        "future_noise_sha256": tensor_sha256(future_noise),
        "action_noise_sha256": tensor_sha256(action_noise),
    }


def validate_fixed_noise_artifact(
    artifact: dict,
    *,
    sample_manifest_sha256: str,
    action_shape: Sequence[int],
    future_shape: Sequence[int],
) -> None:
    """Fail closed when a stored artifact is not the requested fixed noise."""
    if artifact.get("format_version") != 1:
        raise ValueError("unsupported fixed-noise artifact format")
    if artifact.get("sample_manifest_sha256") != sample_manifest_sha256:
        raise ValueError("fixed-noise artifact belongs to another sample manifest")
    action_noise = artifact.get("action_noise")
    future_noise = artifact.get("future_noise")
    if not torch.is_tensor(action_noise) or not torch.is_tensor(future_noise):
        raise ValueError("fixed-noise artifact is missing tensors")
    if tuple(action_noise.shape) != tuple(action_shape):
        raise ValueError("stored action-noise shape mismatch")
    if tuple(future_noise.shape) != tuple(future_shape):
        raise ValueError("stored future-noise shape mismatch")
    if tensor_sha256(action_noise) != artifact.get("action_noise_sha256"):
        raise ValueError("stored action-noise hash mismatch")
    if tensor_sha256(future_noise) != artifact.get("future_noise_sha256"):
        raise ValueError("stored future-noise hash mismatch")


def save_fixed_noise_artifact(path: str | Path, artifact: dict) -> Path:
    """Atomically save an already validated fixed-noise artifact."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(artifact, temporary)
    temporary.replace(destination)
    return destination
