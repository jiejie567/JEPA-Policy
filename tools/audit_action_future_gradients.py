"""Read-only diagnostics for action/future gradient interaction.

This script loads an existing checkpoint, never calls optimizer.step(), and never
saves model state. Virtual SGD updates are applied under no_grad and immediately
reverted before processing the next case.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mip.agent import TrainingAgent
from mip.datasets.robot_dataset import make_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--virtual-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--virtual-lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=24680)
    return parser.parse_args()


def clone_tree(value):
    if isinstance(value, dict):
        return {key: clone_tree(item) for key, item in value.items()}
    return value.clone()


def to_device(value, device):
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value.to(device, non_blocking=True)


def permute_tree(value, permutation):
    if isinstance(value, dict):
        return {key: permute_tree(item, permutation) for key, item in value.items()}
    return value[permutation]


def prepare_batch(raw_batch, config, device):
    obs_batch = raw_batch["obs"]
    obs = {
        key: value[:, : config.task.obs_steps].to(device, non_blocking=True)
        for key, value in obs_batch.items()
    }
    act = raw_batch["action"][:, : config.task.horizon].to(
        device, non_blocking=True
    )
    future_obs = to_device(raw_batch["future_obs"], device)
    delta_t = torch.ones(act.shape[0], device=device)
    return {"obs": obs, "act": act, "future_obs": future_obs, "delta_t": delta_t}


@contextmanager
def isolated_rng(seed, device):
    devices = [device.index or torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        yield


def compute_losses(agent, batch, seed):
    # MultiImageObsEncoder flattens values by assigning back into the observation
    # dict, so every diagnostic forward must receive its own container/tensors.
    obs = clone_tree(batch["obs"])
    future_obs = clone_tree(batch["future_obs"])
    with isolated_rng(seed, batch["act"].device):
        action_loss, info = agent.loss_fn(
            agent.config.optimization,
            agent.flow_map,
            agent.encoder,
            agent.interpolant,
            batch["act"],
            obs,
            batch["delta_t"],
            return_obs_emb=True,
        )
        obs_emb = info.pop("_obs_emb_for_reuse")
        direct_loss, _ = agent._compute_direct_future_embed_loss(
            batch["act"], obs, batch["delta_t"], future_obs, obs_emb
        )
        two_step_loss, two_info = agent._compute_mip_two_step_future_embed_loss(
            batch["act"], obs, batch["delta_t"], future_obs, obs_emb
        )
        selected = two_step_loss if agent.config.optimization.future_embed_loss_mode == "mip_two_step" else direct_loss
        weight, weighted, _ = agent._future_loss_weight_and_info(selected, action_loss)
    return {
        "action": action_loss,
        "direct": direct_loss,
        "denoise_direct": two_info["loss_future_direct_raw"],
        "denoise_second": two_info["loss_future_denoise_raw"],
        "two_step": two_step_loss,
        "weight": weight,
        "weighted": weighted,
    }


def parameter_groups(agent):
    net = agent.flow_map.net
    encoder = [(f"encoder.{name}", param) for name, param in agent.encoder.named_parameters() if param.requires_grad]
    action_head = [(f"flow_map.net.head.{name}", param) for name, param in net.head.named_parameters() if param.requires_grad]
    future_head = [(f"flow_map.net.future_head.{name}", param) for name, param in net.future_head.named_parameters() if param.requires_grad]
    future_conditioning = []
    for module_name in ("future_input_emb",):
        module = getattr(net, module_name, None)
        if module is not None:
            future_conditioning.extend(
                (f"flow_map.net.{module_name}.{name}", param)
                for name, param in module.named_parameters()
                if param.requires_grad
            )
    if getattr(net, "future_tokens", None) is not None and net.future_tokens.requires_grad:
        future_conditioning.append(("flow_map.net.future_tokens", net.future_tokens))

    excluded = {id(param) for _, param in action_head + future_head + future_conditioning}
    shared_trunk = [
        (f"flow_map.net.{name}", param)
        for name, param in net.named_parameters()
        if param.requires_grad
        and id(param) not in excluded
        and not name.startswith(("scalar_head.", "input_processor.", "final_processor."))
    ]
    all_trainable = encoder + [
        (f"flow_map.net.{name}", param)
        for name, param in net.named_parameters()
        if param.requires_grad
    ]
    return {
        "encoder": encoder,
        "shared_trunk": shared_trunk,
        "action_head": action_head,
        "future_head": future_head,
        "future_conditioning": future_conditioning,
        "all_trainable": all_trainable,
    }


def grads(loss, named_params, retain_graph=True):
    params = [param for _, param in named_params]
    values = torch.autograd.grad(
        loss, params, retain_graph=retain_graph, allow_unused=True
    )
    return values


def norm(grads_):
    terms = [grad.float().square().sum() for grad in grads_ if grad is not None]
    if not terms:
        return 0.0
    return float(torch.stack(terms).sum().sqrt().detach().cpu())


def alignment(action_grads, future_grads):
    action_sq = None
    future_sq = None
    dot = None
    for ga, gf in zip(action_grads, future_grads, strict=True):
        if ga is not None:
            term = ga.float().square().sum()
            action_sq = term if action_sq is None else action_sq + term
        if gf is not None:
            term = gf.float().square().sum()
            future_sq = term if future_sq is None else future_sq + term
        if ga is not None and gf is not None:
            term = (ga.float() * gf.float()).sum()
            dot = term if dot is None else dot + term
    action_norm = math.sqrt(float(action_sq.detach().cpu())) if action_sq is not None else 0.0
    future_norm = math.sqrt(float(future_sq.detach().cpu())) if future_sq is not None else 0.0
    cosine = 0.0
    if dot is not None and action_norm > 0 and future_norm > 0:
        cosine = float(dot.detach().cpu()) / (action_norm * future_norm)
    return action_norm, future_norm, cosine


def summarize(values):
    values = [
        float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
        for value in values
    ]
    if not values:
        return {}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
        "fraction_lt_zero": statistics.fmean(value < 0 for value in values),
        "fraction_gt_zero": statistics.fmean(value > 0 for value in values),
    }


def evaluate_action(agent, batch, seed):
    obs = clone_tree(batch["obs"])
    with isolated_rng(seed, batch["act"].device):
        loss, _ = agent.loss_fn(
            agent.config.optimization,
            agent.flow_map,
            agent.encoder,
            agent.interpolant,
            batch["act"],
            obs,
            batch["delta_t"],
        )
    return float(loss.detach().cpu())


def apply_virtual_update(named_params, action_grads, future_grads, lr, include_future):
    with torch.no_grad():
        for (_, param), ga, gf in zip(named_params, action_grads, future_grads, strict=True):
            update = None
            if ga is not None:
                update = ga
            if include_future and gf is not None:
                update = gf if update is None else update + gf
            if update is not None:
                param.add_(update, alpha=-lr)


def revert_virtual_update(named_params, action_grads, future_grads, lr, include_future):
    with torch.no_grad():
        for (_, param), ga, gf in zip(named_params, action_grads, future_grads, strict=True):
            update = None
            if ga is not None:
                update = ga
            if include_future and gf is not None:
                update = gf if update is None else update + gf
            if update is not None:
                param.add_(update, alpha=lr)


def main():
    args = parse_args()
    device = torch.device(args.device)
    config = OmegaConf.load(args.config)
    config.optimization.device = str(device)
    config.optimization.use_compile = False
    config.optimization.dataloader_num_workers = 0
    config.optimization.future_target_type = config.task.future_target_type
    config.task.obs_dim = config.network.emb_dim

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset = make_dataset(config.task)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    agent = TrainingAgent(config)
    agent.load(args.checkpoint, load_optimizer=False)
    agent.eval()
    groups = parameter_groups(agent)

    batches = []
    iterator = iter(loader)
    needed = max(args.batches, 2 * args.virtual_batches)
    for _ in range(needed):
        try:
            raw = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            raw = next(iterator)
        batches.append(prepare_batch(raw, config, device))

    measurements = defaultdict(list)
    for index, batch in enumerate(batches[: args.batches]):
        seed = args.seed + index
        base = compute_losses(agent, batch, seed)
        permutation = torch.arange(batch["act"].shape[0] - 1, -1, -1, device=device)
        obs_shuffled = clone_tree(batch)
        obs_shuffled["obs"] = permute_tree(batch["obs"], permutation)
        action_shuffled = clone_tree(batch)
        action_shuffled["act"] = batch["act"][permutation]
        obs_losses = compute_losses(agent, obs_shuffled, seed)
        action_losses = compute_losses(agent, action_shuffled, seed)

        for key in ("direct", "denoise_direct", "denoise_second", "two_step"):
            measurements[f"shuffle_obs/{key}_base"].append(base[key])
            measurements[f"shuffle_obs/{key}_delta"].append(obs_losses[key] - base[key])
            measurements[f"shuffle_action/{key}_delta"].append(action_losses[key] - base[key])

        for group_name in ("encoder", "shared_trunk"):
            ga = grads(base["action"], groups[group_name])
            gf = grads(base["weighted"], groups[group_name])
            an, fn, cosine = alignment(ga, gf)
            measurements[f"gradient/{group_name}_action_norm"].append(an)
            measurements[f"gradient/{group_name}_future_norm"].append(fn)
            measurements[f"gradient/{group_name}_cosine"].append(cosine)
            measurements[f"gradient/{group_name}_norm_ratio"].append(fn / max(an, 1e-12))

        future_on_action_head = grads(base["weighted"], groups["action_head"])
        action_on_future_head = grads(base["action"], groups["future_head"])
        action_on_future_conditioning = grads(base["action"], groups["future_conditioning"])
        measurements["cross/action_head_future_grad_norm"].append(norm(future_on_action_head))
        measurements["cross/future_head_action_grad_norm"].append(norm(action_on_future_head))
        measurements["cross/future_conditioning_action_grad_norm"].append(norm(action_on_future_conditioning))

        del base, obs_losses, action_losses

    virtual = defaultdict(list)
    all_params = groups["all_trainable"]
    for index in range(args.virtual_batches):
        train_batch = batches[2 * index]
        valid_batch = batches[2 * index + 1]
        seed = args.seed + 10000 + index
        train_losses = compute_losses(agent, train_batch, seed)
        ga = grads(train_losses["action"], all_params)
        gf = grads(train_losses["weighted"], all_params)

        apply_virtual_update(all_params, ga, gf, args.virtual_lr, include_future=False)
        action_train = evaluate_action(agent, train_batch, seed + 100000)
        action_valid = evaluate_action(agent, valid_batch, seed + 200000)
        revert_virtual_update(all_params, ga, gf, args.virtual_lr, include_future=False)

        apply_virtual_update(all_params, ga, gf, args.virtual_lr, include_future=True)
        joint_train = evaluate_action(agent, train_batch, seed + 100000)
        joint_valid = evaluate_action(agent, valid_batch, seed + 200000)
        revert_virtual_update(all_params, ga, gf, args.virtual_lr, include_future=True)

        virtual["interference_train"].append(joint_train - action_train)
        virtual["interference_valid"].append(joint_valid - action_valid)
        virtual["action_only_train"].append(action_train)
        virtual["joint_train"].append(joint_train)
        virtual["action_only_valid"].append(action_valid)
        virtual["joint_valid"].append(joint_valid)

    result = {
        "metadata": {
            "config": str(Path(args.config).resolve()),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "batches": args.batches,
            "virtual_batches": args.virtual_batches,
            "batch_size": args.batch_size,
            "virtual_lr": args.virtual_lr,
            "device": str(device),
            "model_mode": "eval (controlled diagnostic; no dropout/random crop/BN updates)",
            "virtual_update_scope": "all trainable online encoder and flow_map.net parameters; plain SGD formula without optimizer state or weight decay",
            "parameter_counts": {
                key: sum(param.numel() for _, param in value)
                for key, value in groups.items()
            },
        },
        "statistics": {key: summarize(value) for key, value in measurements.items()},
        "virtual_one_step": {key: summarize(value) for key, value in virtual.items()},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
