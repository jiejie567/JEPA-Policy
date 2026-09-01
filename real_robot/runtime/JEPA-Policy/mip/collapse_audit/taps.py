"""Frozen representation tap points for the collapse audit."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class AuditTensors:
    """Representations recomputed at the training pre-loss locations."""

    z_t: torch.Tensor
    observation_embedding: torch.Tensor
    future_target: torch.Tensor
    pred0: torch.Tensor
    pred1: torch.Tensor | None
    future_loss_pred0_raw: torch.Tensor
    future_loss_pred1_raw: torch.Tensor | None
    future_loss_total_raw: torch.Tensor | None
    weights: str


def _copy_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _copy_tree(child) for key, child in value.items()}
    if torch.is_tensor(value):
        return value.clone()
    return value


@contextmanager
def _temporary_eval(*modules):
    states = [module.training for module in modules]
    try:
        for module in modules:
            module.eval()
        yield
    finally:
        for module, was_training in zip(modules, states, strict=True):
            module.train(was_training)


def _align_single_future_token(
    target: torch.Tensor,
    pred0: torch.Tensor,
    pred1: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if pred0.dim() == 2 and target.dim() == 3:
        if target.shape[1] != 1:
            raise RuntimeError(
                "One prediction token cannot audit multiple future targets"
            )
        target = target[:, 0, :]
    elif pred0.dim() == 3 and target.dim() == 2:
        if pred0.shape[1] != 1:
            raise RuntimeError(
                "Multiple prediction tokens cannot audit one future target"
            )
        pred0 = pred0[:, 0, :]
        if pred1 is not None:
            pred1 = pred1[:, 0, :]
    if pred0.shape != target.shape:
        raise RuntimeError(
            "Future prediction/target shape mismatch at audit tap: "
            f"pred0={tuple(pred0.shape)}, target={tuple(target.shape)}"
        )
    if pred1 is not None and pred1.shape != target.shape:
        raise RuntimeError(
            "Second future prediction/target shape mismatch at audit tap: "
            f"pred1={tuple(pred1.shape)}, target={tuple(target.shape)}"
        )
    return target, pred0, pred1


@torch.no_grad()
def extract_audit_tensors(
    agent,
    act: torch.Tensor,
    obs: torch.Tensor | dict[str, torch.Tensor],
    future_obs: torch.Tensor | dict[str, torch.Tensor],
    *,
    use_ema: bool,
    action_noise: torch.Tensor | None = None,
    future_noise: torch.Tensor | None = None,
) -> AuditTensors:
    """Recompute the frozen pre-loss tensors with online or EMA weights.

    ``pred0`` is deterministic and always returned. ``pred1`` and its loss are
    returned only when both fixed noise tensors are supplied. No pooling is
    applied: the current representation is the last temporal observation token.
    """

    if not getattr(agent.config.optimization, "future_joint_mode", False):
        raise RuntimeError("The frozen Future4 audit requires future_joint_mode=true")
    if agent.config.optimization.future_embed_loss_mode != "mip_two_step":
        raise RuntimeError("The frozen Future4 audit requires mip_two_step mode")
    network_config = getattr(agent.config, "network", None)
    token_count = getattr(
        network_config,
        "n_future_tokens",
        getattr(agent.flow_map.net, "n_future_tokens", 0),
    )
    if token_count != 1:
        raise RuntimeError("The frozen Future4 audit requires n_future_tokens=1")
    if (action_noise is None) != (future_noise is None):
        raise ValueError("action_noise and future_noise must be supplied together")

    if use_ema:
        encoder = agent.encoder_ema
        flow_map = agent.flow_map_ema
        weights = "ema"
    else:
        encoder = agent.encoder
        flow_map = agent.flow_map
        weights = "online"

    obs = _copy_tree(obs)
    future_obs = _copy_tree(future_obs)
    with _temporary_eval(encoder, flow_map):
        future_target = None
        if (
            getattr(encoder, "temporal_consistent_crop", False)
            and hasattr(encoder, "encode_obs_and_future_rgb")
        ):
            observation_embedding, future_target, _ = (
                encoder.encode_obs_and_future_rgb(obs, future_obs, mask=None)
            )
        else:
            observation_embedding = encoder(obs, None)
        if future_target is None:
            future_target = agent._encode_future_target(
                future_obs, encoder=encoder
            )
        future_target = future_target.detach()

        if observation_embedding.dim() != 3:
            raise RuntimeError(
                "Frozen z_t tap expects [N,T,D] observation embeddings, got "
                f"{tuple(observation_embedding.shape)}"
            )
        z_t = observation_embedding[:, -1, :]

        batch_size = act.shape[0]
        s = torch.zeros(batch_size, device=act.device, dtype=act.dtype)
        t_value = float(agent.config.optimization.t_two_step)
        t = torch.full_like(s, t_value)
        act0 = torch.zeros_like(act)
        future0 = torch.zeros_like(future_target)
        _act_pred0, pred0 = flow_map.net.joint_forward(
            act0, s, t, observation_embedding, future0
        )
        if pred0 is None:
            raise RuntimeError("Future head returned no pred0 at the audit tap")

        pred1 = None
        if action_noise is not None and future_noise is not None:
            if action_noise.shape != act.shape:
                raise ValueError(
                    f"action_noise shape {tuple(action_noise.shape)} != "
                    f"act shape {tuple(act.shape)}"
                )
            if future_noise.shape != future_target.shape:
                raise ValueError(
                    f"future_noise shape {tuple(future_noise.shape)} != "
                    f"target shape {tuple(future_target.shape)}"
                )
            act_t = act + (1.0 - t_value) * action_noise
            future_t = future_target + (1.0 - t_value) * future_noise
            _act_pred1, pred1 = flow_map.net.joint_forward(
                act_t,
                t,
                torch.ones_like(t),
                observation_embedding,
                future_t,
            )
            if pred1 is None:
                raise RuntimeError("Future head returned no pred1 at the audit tap")

        future_target, pred0, pred1 = _align_single_future_token(
            future_target, pred0, pred1
        )
        loss0 = agent._normalized_future_loss(
            pred0, future_target, interval=t_value
        ).detach()
        loss1 = None
        total = None
        if pred1 is not None:
            loss1 = agent._normalized_future_loss(
                pred1, future_target, interval=1.0 - t_value
            ).detach()
            total = loss0 + loss1

    return AuditTensors(
        z_t=z_t.detach(),
        observation_embedding=observation_embedding.detach(),
        future_target=future_target.detach(),
        pred0=pred0.detach(),
        pred1=None if pred1 is None else pred1.detach(),
        future_loss_pred0_raw=loss0,
        future_loss_pred1_raw=loss1,
        future_loss_total_raw=total,
        weights=weights,
    )
