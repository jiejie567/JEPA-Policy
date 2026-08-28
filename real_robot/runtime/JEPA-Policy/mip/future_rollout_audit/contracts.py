"""Prediction-target representation-space contract."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from typing import Iterator

import torch

from mip.future_rollout_audit.seeding import preserve_rng_state


class FutureSpaceContractError(RuntimeError):
    """Raised before metrics when prediction and target spaces do not match."""

    code = "future_space_contract_mismatch"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _torch_dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


@contextmanager
def _temporary_eval(*modules) -> Iterator[None]:
    states = [module.training for module in modules]
    try:
        for module in modules:
            module.eval()
        yield
    finally:
        for module, state in zip(modules, states, strict=True):
            module.train(state)


def encode_audit_future_target(agent, future_obs) -> torch.Tensor:
    """Run the training target constructor with frozen EMA evaluation weights."""

    encoder = agent.encoder_ema
    with preserve_rng_state(), _temporary_eval(encoder), torch.inference_mode():
        target = agent._encode_future_target(future_obs, encoder=encoder)
        return target.detach().to(device="cpu", copy=True).clone().contiguous()


def _projector_name(encoder) -> str:
    if hasattr(encoder, "encode_rgb_features"):
        return "none"
    projector = getattr(encoder, "future_projector", None)
    if projector is None:
        return "encoder_forward"
    return f"{type(projector).__module__}.{type(projector).__qualname__}"


def _token_reduction(prediction: torch.Tensor, target: torch.Tensor) -> str:
    if prediction.ndim == 2 and target.ndim == 2:
        return "single_future_token_squeeze;comparison=elementwise_no_pooling"
    if prediction.ndim == 3 and target.ndim == 3:
        return "none;comparison=per_token_elementwise"
    raise FutureSpaceContractError(
        "future_space_contract_mismatch: unsupported prediction/target ranks "
        f"{prediction.ndim}/{target.ndim}"
    )


def build_future_space_contract(agent, prediction: torch.Tensor, target: torch.Tensor) -> dict:
    """Build and validate the exact runtime representation contract."""

    if tuple(prediction.shape) != tuple(target.shape):
        raise FutureSpaceContractError(
            "future_space_contract_mismatch: shape differs: "
            f"prediction={tuple(prediction.shape)} target={tuple(target.shape)}"
        )
    if prediction.dtype != target.dtype:
        raise FutureSpaceContractError(
            "future_space_contract_mismatch: dtype differs: "
            f"prediction={prediction.dtype} target={target.dtype}"
        )
    if prediction.ndim not in (2, 3) or prediction.shape[-1] <= 0:
        raise FutureSpaceContractError(
            "future_space_contract_mismatch: expected [B,D] or [B,T,D] tensors"
        )

    encoder = agent.encoder_ema
    camera_keys = list(getattr(encoder, "rgb_keys", []))
    if camera_keys != sorted(camera_keys):
        raise FutureSpaceContractError(
            "future_space_contract_mismatch: encoder camera keys are not sorted"
        )
    target_source = (
        "encoder_ema/training_future_target_construction:encode_rgb_features"
        if hasattr(encoder, "encode_rgb_features")
        else "encoder_ema/training_future_target_construction:encoder_forward"
    )
    payload = {
        "contract_version": 1,
        "prediction_source": "flow_map_ema.net.future_head/pass_1",
        "target_encoder_source": target_source,
        "projector_name": _projector_name(encoder),
        "camera_key_order": camera_keys,
        "token_shape": list(prediction.shape[1:]),
        "token_reduction": _token_reduction(prediction, target),
        "normalization": {
            "prediction_embedding": "none",
            "target_embedding": "none",
            "raw_loss": "target_rms_then_interval",
        },
        "dtype": _torch_dtype_name(prediction.dtype),
        "embedding_dim": int(prediction.shape[-1]),
    }
    payload["contract_sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return payload


def validate_future_space_contract(
    contract: dict,
    agent,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> None:
    """Fail closed unless a stored contract equals current runtime truth."""

    if not isinstance(contract, dict):
        raise FutureSpaceContractError(
            "future_space_contract_mismatch: contract must be an object"
        )
    actual = build_future_space_contract(agent, prediction, target)
    if contract != actual:
        raise FutureSpaceContractError(
            "future_space_contract_mismatch: stored contract does not match runtime"
        )
