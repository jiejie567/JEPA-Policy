"""Hard-gate the frozen audit taps on one real formal checkpoint and batch."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from mip.agent import TrainingAgent
from mip.collapse_audit.taps import extract_audit_tensors
from mip.datasets.robot_dataset import make_dataset
from mip.torch_utils import set_seed


def _to_device(value, device: str):
    if isinstance(value, dict):
        return {key: _to_device(child, device) for key, child in value.items()}
    result = value.to(device, non_blocking=True)
    if result.dtype == torch.uint8:
        result = result.float().div_(255.0).mul_(2.0).sub_(1.0)
    return result


def _assert_close(left: torch.Tensor, right: torch.Tensor, name: str) -> None:
    try:
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
    except AssertionError as exc:
        raise AssertionError(f"real tap mismatch for {name}: {exc}") from exc


def run_gate(
    config_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    *,
    batch_size: int,
) -> dict:
    config = OmegaConf.load(config_path)
    config.optimization.future_target_type = config.task.future_target_type
    config.task.obs_dim = config.network.emb_dim
    set_seed(int(config.optimization.seed))

    dataset = make_dataset(config.task)
    batch = default_collate([dataset[index] for index in range(batch_size)])
    device = str(config.optimization.device)
    obs = _to_device(
        {
            key: value[:, : config.task.obs_steps]
            for key, value in batch["obs"].items()
        },
        device,
    )
    future_obs = _to_device(batch["future_obs"], device)
    act = _to_device(batch["action"][:, : config.task.horizon], device)
    delta_t = torch.full(
        (act.shape[0],),
        float(config.log.validation_delta_t),
        device=device,
    )

    agent = TrainingAgent(config)
    agent.load(str(checkpoint_path), load_optimizer=False)
    agent.eval()
    with torch.no_grad():
        target_shape = agent._encode_future_target(future_obs).shape
    generator = torch.Generator(device="cpu").manual_seed(20260729)
    future_noise = torch.randn(target_shape, generator=generator).to(device)
    action_noise = torch.randn(act.shape, generator=generator).to(device)

    with torch.no_grad():
        _loss, training_info = agent._compute_joint_mip_two_step_loss(
            act,
            obs,
            delta_t,
            future_obs,
            action_noise=action_noise,
            future_noise=future_noise,
        )
    online = extract_audit_tensors(
        agent,
        act,
        obs,
        future_obs,
        use_ema=False,
        action_noise=action_noise,
        future_noise=future_noise,
    )
    comparisons = {
        "observation_embedding": (
            online.observation_embedding,
            training_info["_joint_obs_emb"],
        ),
        "z_t": (online.z_t, training_info["_joint_obs_emb"][:, -1]),
        "future_target": (
            online.future_target,
            training_info["_joint_future_target"],
        ),
        "pred0": (online.pred0, training_info["_joint_future_pred_0"]),
        "pred1": (online.pred1, training_info["_joint_future_pred_1"]),
        "future_loss_pred0_raw": (
            online.future_loss_pred0_raw,
            training_info["loss_future_first_raw"],
        ),
        "future_loss_pred1_raw": (
            online.future_loss_pred1_raw,
            training_info["loss_future_second_raw"],
        ),
        "future_loss_total_raw": (
            online.future_loss_total_raw,
            training_info["loss_future_total"],
        ),
    }
    for name, (left, right) in comparisons.items():
        _assert_close(left, right, name)

    ema = extract_audit_tensors(
        agent,
        act,
        obs,
        future_obs,
        use_ema=True,
        action_noise=action_noise,
        future_noise=future_noise,
    )
    ema_repeat = extract_audit_tensors(
        agent,
        act,
        obs,
        future_obs,
        use_ema=True,
        action_noise=action_noise,
        future_noise=future_noise,
    )
    for name in ("z_t", "future_target", "pred0", "pred1"):
        value = getattr(ema, name)
        _assert_close(value, getattr(ema_repeat, name), f"EMA deterministic {name}")
        if not torch.isfinite(value).all():
            raise RuntimeError(f"non-finite EMA tap: {name}")

    if online.z_t.shape != (batch_size, 384):
        raise RuntimeError(f"unexpected z_t shape: {tuple(online.z_t.shape)}")
    for name in ("future_target", "pred0", "pred1"):
        shape = tuple(getattr(online, name).shape)
        if shape != (batch_size, 1024):
            raise RuntimeError(f"unexpected {name} shape: {shape}")

    result = {
        "status": "passed",
        "config_path": str(config_path.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "batch_size": batch_size,
        "online_matches_training_pre_loss": True,
        "ema_repeat_deterministic": True,
        "weights": {"formal_audit": ema.weights, "training_gate": online.weights},
        "shapes": {
            "z_t": list(online.z_t.shape),
            "future_target": list(online.future_target.shape),
            "pred0": list(online.pred0.shape),
            "pred1": list(online.pred1.shape),
        },
        "losses": {
            "pred0_raw": float(online.future_loss_pred0_raw.item()),
            "pred1_raw": float(online.future_loss_pred1_raw.item()),
            "total_raw": float(online.future_loss_total_raw.item()),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    result = run_gate(
        args.config,
        args.checkpoint,
        args.output,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
