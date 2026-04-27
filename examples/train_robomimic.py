"""Training pipeline for robomimic dataset.

Author: Chaoyi Pan
Date: 2025-10-03
"""

import json
import os
import time
from pathlib import Path

# Set MuJoCo rendering backend before importing any robomimic/mujoco modules
# Try OSMesa for headless rendering (software rendering, more compatible but slower)
os.environ["MUJOCO_GL"] = "osmesa"

import hydra
import loguru
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

# Some environments in this workspace have a cuDNN installation that is
# incompatible with the currently working CUDA driver / wheel combo. Allow an
# explicit opt-out so training can still proceed with native CUDA kernels.
if os.getenv("MIP_DISABLE_CUDNN", "0") == "1":
    torch.backends.cudnn.enabled = False

torch.set_float32_matmul_precision("high")

from mip.agent import TrainingAgent
from mip.config import Config
from mip.dataset_utils import loop_dataloader
from mip.datasets.robot_dataset import make_dataset
from mip.envs.robot_env import make_vec_env
from mip.libero_utils import is_libero_task
from mip.logger import Logger, compute_average_metrics, update_best_metrics
from mip.samplers import get_default_step_list
from mip.scheduler import WarmupAnnealingScheduler
from mip.torch_utils import set_seed


def _format_suffix_value(value):
    text = str(value)
    return text.replace(".", "p").replace("/", "_")


def get_checkpoint_base_name(config: Config) -> str:
    base_name = (
        f"{config.task.env_name}_{config.task.env_type}_{config.task.obs_type}_"
        f"{config.optimization.loss_type}_{config.network.network_type}_"
        f"{config.network.emb_dim}_seed{config.optimization.seed}"
    )

    if getattr(config.task, "future_state_enabled", False):
        future_steps_list = getattr(config.task, "future_state_steps_list", [])
        if future_steps_list:
            future_tag = "future" + "-".join(str(step) for step in future_steps_list)
        else:
            future_tag = f"future{config.task.future_state_steps}"
        base_name += (
            f"_{future_tag}"
            f"_{config.task.future_target_type}"
            f"_ftok{config.network.n_future_tokens}"
        )
        if getattr(config.optimization, "use_future_embed_loss", False):
            future_weight = getattr(config.optimization, "future_embed_loss_weight", 0.0)
            future_mode = getattr(config.optimization, "future_embed_loss_mode", "direct")
            base_name += f"_fembed{future_mode}_{_format_suffix_value(future_weight)}"
        else:
            future_weight = getattr(config.optimization, "future_state_loss_weight", 0.0)
            base_name += f"_fw{_format_suffix_value(future_weight)}"

    if getattr(config.optimization, "freeze_encoder", False):
        base_name += "_freezeenc"

    if getattr(config.optimization, "use_sigreg", False):
        sigreg_weight = getattr(config.optimization, "sigreg_weight", 0.0)
        base_name += f"_sigreg{_format_suffix_value(sigreg_weight)}"

    return base_name


def build_training_state(n_gradient_step: int, best_metrics: dict, eval_history: list):
    return {
        "n_gradient_step": n_gradient_step,
        "best_metrics": best_metrics,
        "eval_history": eval_history,
    }


def infer_resume_state_from_metrics(log_dir: str | Path):
    metrics_path = Path(log_dir) / "metrics.jsonl"
    if not metrics_path.exists():
        return None

    last_train_step = None
    best_metrics = {}
    eval_history = []

    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "loss" in entry:
                last_train_step = entry.get("step", last_train_step)

            if any(k.startswith("mean_success_") for k in entry):
                eval_history.append(entry)
                best_metrics = update_best_metrics(best_metrics, entry)

    if last_train_step is None:
        return None

    return build_training_state(
        n_gradient_step=last_train_step,
        best_metrics=best_metrics,
        eval_history=eval_history,
    )


def train(config: Config, envs, dataset, agent, logger, resume_state=None):
    """Standalone training function.

    Args:
        config: Configuration for training
        envs: Environment
        dataset: Training dataset
        agent: Agent to train
        logger: Logger for metrics
        resume_state: Optional dict with training state to resume from
    """
    # dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.optimization.batch_size,
        num_workers=config.optimization.dataloader_num_workers,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process after each epoch
        persistent_workers=(
            config.optimization.dataloader_persistent_workers
            and config.optimization.dataloader_num_workers > 0
        ),
    )
    loop_loader = loop_dataloader(dataloader)

    # lr scheduler
    lr_scheduler = CosineAnnealingLR(
        agent.optimizer, T_max=config.optimization.gradient_steps
    )

    # warmup scheduler (mainly for flow map learning)
    warmup_scheduler = WarmupAnnealingScheduler(
        max_steps=config.optimization.gradient_steps,
        warmup_ratio=config.optimization.warmup_ratio,
        rampup_ratio=config.optimization.rampup_ratio,
        min_value=config.optimization.min_value,
        max_value=config.optimization.max_value,
    )

    # Resume from checkpoint if available
    start_step = 0
    best_metrics = {}
    eval_history = []
    if resume_state is not None:
        start_step = resume_state.get("n_gradient_step", 0) + 1
        best_metrics = resume_state.get("best_metrics", {})
        eval_history = resume_state.get("eval_history", [])
        loguru.logger.info(f"Resuming training from step {start_step}")
        loguru.logger.info(f"Restored best metrics: {best_metrics}")
        if start_step > 0:
            lr_scheduler.step(start_step)

    info_list = []
    start_time = time.time()
    for n_gradient_step in range(start_step, config.optimization.gradient_steps):
        # get batch from dataloader
        batch = next(loop_loader)

        # preprocess data
        if config.task.obs_type == "image":
            obs_batch = batch["obs"]
            obs = {}
            for k in obs_batch:
                obs[k] = obs_batch[k][:, : config.task.obs_steps, :].to(
                    config.optimization.device
                )
        elif config.task.obs_type == "state":
            obs = batch["obs"]["state"].to(config.optimization.device)
            obs = obs[:, : config.task.obs_steps, :]  # (B, obs_horizon, obs_dim)
        act = batch["action"].to(config.optimization.device)
        act = act[:, : config.task.horizon, :]  # (B, horizon, act_dim)

        # update diffusion
        delta_t_scalar = warmup_scheduler(n_gradient_step)
        batch_size = act.shape[0]
        delta_t = torch.full(
            (batch_size,), delta_t_scalar, device=config.optimization.device
        )
        # info = agent.update(act, obs, delta_t)
        future_obs = None
        if "future_obs" in batch:
            future_obs = {}

            for k in batch["future_obs"]:
                future_obs[k] = batch["future_obs"][k].to(config.optimization.device)

        info = agent.update(act, obs, delta_t, future_obs=future_obs)

        lr_scheduler.step()
        info_list.append(info)

        # log metrics
        if ((n_gradient_step + 1) % config.log.log_freq) == 0:
            metrics = {
                "step": n_gradient_step,
                "total_time": time.time() - start_time,
                "lr": lr_scheduler.get_last_lr()[0],
                "delta_t": delta_t_scalar,
            }
            for key in info:
                try:
                    metrics[key] = np.nanmean([info[key] for info in info_list])
                except Exception:
                    metrics[key] = np.nan
            logger.log(metrics, category="train")
            info_list = []

        if ((n_gradient_step + 1) % config.log.save_freq) == 0:
            loguru.logger.info("Save model...")
            training_state = build_training_state(
                n_gradient_step=n_gradient_step,
                best_metrics=best_metrics,
                eval_history=eval_history,
            )
            logger.save_agent(
                agent=agent,
                identifier="latest",
                training_state=training_state,
            )

        if ((n_gradient_step + 1) % config.log.eval_freq) == 0:
            loguru.logger.info("Evaluate model...")
            agent.eval()
            metrics = {"step": n_gradient_step}
            num_steps_list = get_default_step_list(config.optimization.loss_type)
            for num_steps in num_steps_list:
                metrics.update(eval(config, envs, dataset, agent, logger, num_steps))

            # Update best metrics and average metrics
            old_best_metrics = best_metrics.copy()
            best_metrics = update_best_metrics(best_metrics, metrics)
            eval_history.append(metrics.copy())
            avg_metrics = compute_average_metrics(eval_history)

            # Check if this is a new best model based on success rate
            # Use the first num_steps in the list as the primary metric
            primary_metric_key = f"mean_success_{num_steps_list[0]}"
            if primary_metric_key in metrics:
                is_new_best = (
                    primary_metric_key not in old_best_metrics
                    or metrics[primary_metric_key]
                    > old_best_metrics[primary_metric_key]
                )
                if is_new_best:
                    success_rate = metrics[primary_metric_key]
                    loguru.logger.info(
                        f"New best model! {primary_metric_key} = {success_rate:.4f}"
                    )
                    # Save to local models directory
                    logger.save_agent(agent=agent, identifier="best")

                    # Save to global checkpoints directory with success rate comparison
                    # Include training state for resuming
                    checkpoint_base_name = get_checkpoint_base_name(config)
                    training_state = build_training_state(
                        n_gradient_step=n_gradient_step,
                        best_metrics=best_metrics,
                        eval_history=eval_history,
                    )
                    logger.save_global_checkpoint(
                        agent, checkpoint_base_name, success_rate, training_state=training_state
                    )

            # Add best and average metrics to current metrics for logging
            for key, value in best_metrics.items():
                metrics[f"best_{key}"] = value
            for key, value in avg_metrics.items():
                metrics[key] = value

            # Print best and average metrics
            loguru.logger.info("Best metrics so far:")
            for key, value in best_metrics.items():
                loguru.logger.info(f"  {key}: {value:.4f}")
            if avg_metrics:
                loguru.logger.info("Average metrics (last 5 evals):")
                for key, value in avg_metrics.items():
                    loguru.logger.info(f"  {key}: {value:.4f}")

            logger.log(metrics, category="eval")
            agent.train()


def eval(config: Config, envs, dataset, agent, logger, num_steps=1):
    """Standalone inference function to evaluate a trained agent and optionally save a video.

    Args:
        config: Configuration object containing evaluation parameters
        envs: Environment
        dataset: Dataset
        agent: Trained agent
        logger: Logger for metrics
        num_steps: Number of steps for sampling

    Returns:
        dict: Metrics including mean step, reward, and success rate
    """
    # ---------------- Start Rollout ----------------
    episode_rewards = []
    episode_steps = []
    episode_success = []
    episode_kit_success = []

    for i in range(config.log.eval_episodes // config.task.num_envs):
        ep_reward = [0.0] * config.task.num_envs
        obs, _ = envs.reset()
        t = 0

        # initialize video stream
        if config.log.save_video:
            logger.video_init(envs.envs[0], enable=True, video_id=str(i))  # save videos

        while t < config.task.max_episode_steps:
            if config.task.obs_type == "state":
                obs = obs.astype(np.float32)  # (num_envs, obs_steps, obs_dim)
                # normalize obs
                obs = dataset.normalizer["obs"]["state"].normalize(obs)
                obs = torch.tensor(
                    obs, device=config.optimization.device, dtype=torch.float32
                )  # (num_envs, obs_steps, obs_dim)
                obs = {"state": obs}
            else:  # image-based observation
                obs_raw = obs
                obs = {}
                for k in obs_raw:
                    obs[k] = obs_raw[k].astype(
                        np.float32
                    )  # (num_envs, obs_steps, obs_dim)
                    obs[k] = dataset.normalizer["obs"][k].normalize(obs[k])
                    obs[k] = torch.tensor(
                        obs[k], device=config.optimization.device, dtype=torch.float32
                    )  # (num_envs, obs_steps, obs_dim)

            act_0 = torch.randn(
                (config.task.num_envs, config.task.horizon, config.task.act_dim),
                device=config.optimization.device,
            )
            # run sampling (num_envs, horizon, action_dim)
            act_normed = agent.sample(
                act_0=act_0,
                obs=obs,
                num_steps=num_steps,
                use_ema=True,
            )

            # unnormalize prediction
            act_normed = (
                act_normed.detach().to("cpu").numpy()
            )  # (num_envs, horizon, action_dim)
            act = dataset.normalizer["action"].unnormalize(act_normed)

            # get action by slicing from start to end
            start = config.task.obs_steps - 1
            end = start + config.task.act_steps
            act = act[:, start:end, :]

            if config.task.abs_action and not is_libero_task(config.task) and config.task.env_name in [
                "can",
                "lift",
                "square",
                "tool_hang",
                "transport",
            ]:
                act = dataset.undo_transform_action(act)
            obs, reward, terminated, truncated, info = envs.step(act)
            terminated | truncated
            ep_reward += reward
            t += config.task.act_steps
        success = [1.0 if s > 0 else 0.0 for s in ep_reward]

        # evaluate kitchen
        kit_success = []
        if "kitchen" in config.task.env_name:
            task_completion_counts = [
                len(info[i]["completed_tasks"][0]) for i in range(config.task.num_envs)
            ]
            for num in task_completion_counts:
                sublist = [1 if i < num else 0 for i in range(7)]
                kit_success.append(sublist)
            # Use p4 success rate as the main success metric for kitchen environments
            success = [1 if num >= 4 else 0 for num in task_completion_counts]

        episode_rewards.append(ep_reward)
        episode_steps.append(t)
        episode_success.append(success)
        episode_kit_success.append(kit_success)
    loguru.logger.info(
        f"Nstep: {num_steps} Mean step: {np.nanmean(episode_steps)} Mean reward: {np.nanmean(episode_rewards)} Mean success: {np.nanmean(episode_success)}"
    )

    metrics = {
        f"mean_step_{num_steps}": np.nanmean(episode_steps),
        f"mean_reward_{num_steps}": np.nanmean(episode_rewards),
        f"mean_success_{num_steps}": np.nanmean(episode_success),
    }

    if "kitchen" in config.task.env_name:
        mean_kit_success = np.mean(np.array(episode_kit_success), axis=(0, 1))
        kit_metrics = {}
        for i in range(7):
            kit_metrics[f"p{i + 1}_NFE{num_steps}"] = mean_kit_success[i]
        metrics.update(kit_metrics)
        loguru.logger.info(f"Kit metrics: {kit_metrics}")

    return metrics


@hydra.main(version_base=None, config_path="configs/", config_name="main")
def main(config):
    """Main pipeline function that calls the appropriate standalone function based on mode."""
    config.optimization.future_target_type = getattr(
        config.task, "future_target_type", config.optimization.future_target_type
    )
    if getattr(config.optimization, "disable_cudnn", False):
        torch.backends.cudnn.enabled = False
    # general config setup
    set_seed(config.optimization.seed)
    logger = Logger(config)
    loguru.logger.info("Finished setting up logger")

    # env setup
    envs = make_vec_env(config.task, seed=config.optimization.seed)
    obs, info = envs.reset()
    if config.task.obs_type == "state":
        config.task.obs_dim = obs.shape[-1]
    else:
        # For image observations, set obs_dim to embedding dimension
        # This is used by the network but not actually used when encoder_type is "image"
        config.task.obs_dim = config.network.emb_dim
    loguru.logger.info("Finished setting up env")

    # dataset setup
    dataset = make_dataset(config.task)
    loguru.logger.info("Finished setting up dataset")

    agent = TrainingAgent(config)
    resume_state = None

    if config.optimization.model_path and config.optimization.model_path != "None":
        loguru.logger.info(f"Loading model from {config.optimization.model_path}")
        resume_state = agent.load(config.optimization.model_path, load_optimizer=True)
        if resume_state is None:
            resume_state = infer_resume_state_from_metrics(config.log.log_dir)
            if resume_state is not None:
                loguru.logger.info(
                    f"Inferred resume state from metrics at step {resume_state['n_gradient_step']}"
                )
    elif config.mode == "train" and config.optimization.auto_resume:
        # Automatically look for checkpoint to resume from
        checkpoint_base_name = get_checkpoint_base_name(config)
        checkpoint_path = logger.find_latest_checkpoint(checkpoint_base_name)
        if checkpoint_path:
            loguru.logger.info(f"Found checkpoint to resume from: {checkpoint_path}")
            loguru.logger.info("Loading checkpoint with optimizer state...")
            resume_state = agent.load(str(checkpoint_path), load_optimizer=True)
        else:
            loguru.logger.info("No checkpoint found, starting training from scratch")
    elif config.mode == "train" and not config.optimization.auto_resume:
        loguru.logger.info("Auto-resume disabled, starting training from scratch")

    if config.mode == "train":
        train(config, envs, dataset, agent, logger, resume_state=resume_state)
    elif config.mode == "eval":
        if not config.optimization.model_path:
            raise ValueError("Empty model for inference")
        agent.eval()

        num_steps_list = get_default_step_list(config.optimization.loss_type)
        for num_steps in num_steps_list:
            metrics = {"step": num_steps}
            metrics.update(eval(config, envs, dataset, agent, logger, num_steps))
            logger.log(metrics, category="eval")

        # print result in easy to read format
        for key, val in metrics.items():
            if "mean_success" in key:
                loguru.logger.info(f"{key} - {val}")
    else:
        raise ValueError("Illegal mode")


if __name__ == "__main__":
    main()
