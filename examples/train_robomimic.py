"""Training pipeline for robomimic dataset.

Author: Chaoyi Pan
Date: 2025-10-03
"""

import json
import os
import time
from functools import wraps
from pathlib import Path

# Set MuJoCo rendering backend before importing any robomimic/mujoco modules.
# Default to EGL for headless GPU rendering, while allowing callers to override.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg_cache")

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

from jepa_policy.agent import TrainingAgent
from jepa_policy.config import Config
from jepa_policy.dataset_utils import loop_dataloader
from jepa_policy.datasets.robot_dataset import make_dataset
from jepa_policy.envs.robot_env import make_vec_env
from jepa_policy.envs.persistent_image_rollout import PersistentImageRolloutPool
from jepa_policy.eval_rng import get_episode_seeds, get_rollout_seed, isolated_torch_rng
from jepa_policy.libero_utils import is_libero_task
from jepa_policy.logger import Logger, compute_average_metrics, update_best_metrics
from jepa_policy.photometric_augmentation import augment_robot_rgb_batch
from jepa_policy.runtime_env import validate_runtime_environment
from jepa_policy.samplers import get_default_step_list
from jepa_policy.scheduler import WarmupAnnealingScheduler
from jepa_policy.torch_utils import set_seed


def isolate_rollout_rng(function):
    """Run rollout evaluation without advancing the training Torch RNG."""

    @wraps(function)
    def wrapped(config, *args, **kwargs):
        with isolated_torch_rng(
            get_rollout_seed(config), config.optimization.device
        ):
            return function(config, *args, **kwargs)

    return wrapped


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
            f"_ftok{getattr(config.network, 'n_future_tokens', 0)}"
        )
        if getattr(config.optimization, "use_future_embed_loss", False):
            future_mode = getattr(config.optimization, "future_state_loss_mode", "fixed")
            if future_mode == "ratio":
                future_ratio = getattr(config.optimization, "future_state_loss_ratio", 0.0)
                base_name += f"_fratio{_format_suffix_value(future_ratio)}"
            else:
                future_weight = getattr(config.optimization, "future_embed_loss_weight", 0.0)
                embed_mode = getattr(config.optimization, "future_embed_loss_mode", "direct")
                base_name += f"_fembed{embed_mode}_{_format_suffix_value(future_weight)}"
        else:
            future_mode = getattr(config.optimization, "future_state_loss_mode", "fixed")
            if future_mode == "ratio":
                future_ratio = getattr(config.optimization, "future_state_loss_ratio", 0.0)
                base_name += f"_fratio{_format_suffix_value(future_ratio)}"
            else:
                future_weight = getattr(config.optimization, "future_state_loss_weight", 0.0)
                base_name += f"_fw{_format_suffix_value(future_weight)}"

    if getattr(config.optimization, "freeze_encoder", False):
        base_name += "_freezeenc"

    return base_name


def build_training_state(n_gradient_step: int, best_metrics: dict, eval_history: list):
    return {
        "n_gradient_step": n_gradient_step,
        "best_metrics": best_metrics,
        "eval_history": eval_history,
    }


def resolve_training_end_step(config) -> int:
    """Return optimizer steps to execute without changing schedule horizons."""
    gradient_steps = int(config.optimization.gradient_steps)
    stop_after_steps = getattr(config.optimization, "stop_after_steps", None)
    if stop_after_steps is None:
        return gradient_steps
    stop_after_steps = int(stop_after_steps)
    if not 0 < stop_after_steps <= gradient_steps:
        raise ValueError(
            "optimization.stop_after_steps must be in "
            f"[1, {gradient_steps}], got {stop_after_steps}"
        )
    return stop_after_steps


def validate_snapshot_steps(config, training_end_step: int) -> tuple[int, ...]:
    """Validate and normalize exact completed-step snapshot positions."""
    values = tuple(int(step) for step in getattr(config.log, "snapshot_steps", []))
    if len(values) != len(set(values)) or tuple(sorted(values)) != values:
        raise ValueError("log.snapshot_steps must be unique and increasing")
    if any(step < 0 or step > training_end_step for step in values):
        raise ValueError(
            "log.snapshot_steps must be between 0 and the effective training "
            f"end ({training_end_step}), got {values}"
        )
    return values


def save_trajectory_snapshot(
    logger,
    agent,
    completed_steps: int,
    best_metrics: dict,
    eval_history: list,
) -> None:
    """Save a model-only checkpoint for offline trajectory auditing."""
    training_state = build_training_state(
        n_gradient_step=completed_steps - 1,
        best_metrics=best_metrics,
        eval_history=eval_history,
    )
    logger.save_agent(
        agent=agent,
        identifier=f"collapse_step{completed_steps:06d}",
        training_state=training_state,
        include_optimizer=False,
    )


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


def _fixed_batch_to_cpu(value, batch_size):
    if isinstance(value, dict):
        return {
            key: _fixed_batch_to_cpu(item, batch_size)
            for key, item in value.items()
        }
    return value[:batch_size].detach().cpu().clone()


def _fixed_batch_to_device(value, device):
    if isinstance(value, dict):
        return {
            key: _fixed_batch_to_device(item, device)
            for key, item in value.items()
        }
    return value.to(device, non_blocking=True)


def _observation_to_device(value, device):
    """Move an observation while preserving compact cached RGB transfers."""
    value = value.to(device, non_blocking=True)
    if value.dtype == torch.uint8:
        # RoboCasa's array cache stores the already-resized RGB frames as
        # uint8. Keep them compact through DataLoader IPC, pinned memory and
        # H2D, then reproduce ImageNormalizer's [0, 1] -> [-1, 1] mapping.
        value = value.to(dtype=torch.float32)
        value.div_(255.0).mul_(2.0).sub_(1.0)
    return value


def train(
    config: Config, envs, dataset, agent, logger, resume_state=None, parallel_pool=None
):
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

    training_end_step = resolve_training_end_step(config)
    snapshot_steps = validate_snapshot_steps(config, training_end_step)
    loguru.logger.info(
        "Training step plan: "
        f"scheduler_horizon={config.optimization.gradient_steps}, "
        f"stop_after_steps={training_end_step}, "
        f"snapshot_steps={list(snapshot_steps)}"
    )
    if start_step == 0 and 0 in snapshot_steps:
        save_trajectory_snapshot(
            logger, agent, 0, best_metrics, eval_history
        )

    info_list = []
    fixed_validation_batch = None
    start_time = time.time()
    for n_gradient_step in range(start_step, training_end_step):
        # get batch from dataloader
        data_wait_started = time.perf_counter()
        batch = next(loop_loader)
        data_wait_seconds = time.perf_counter() - data_wait_started
        preprocess_started = time.perf_counter()

        # preprocess data
        if config.task.obs_type == "image":
            obs_batch = batch["obs"]
            obs = {}
            for k in obs_batch:
                obs[k] = _observation_to_device(
                    obs_batch[k][:, : config.task.obs_steps, :],
                    config.optimization.device,
                )
        elif config.task.obs_type == "state":
            obs = batch["obs"]["state"].to(config.optimization.device, non_blocking=True)
            obs = obs[:, : config.task.obs_steps, :]  # (B, obs_horizon, obs_dim)
        act = batch["action"].to(config.optimization.device, non_blocking=True)
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
                future_obs[k] = _observation_to_device(
                    batch["future_obs"][k],
                    config.optimization.device,
                )

        if (
            fixed_validation_batch is None
            and future_obs is not None
            and config.log.validation_freq > 0
        ):
            validation_batch_size = min(
                config.log.validation_batch_size, act.shape[0]
            )
            fixed_validation_batch = {
                "obs": _fixed_batch_to_cpu(obs, validation_batch_size),
                "act": _fixed_batch_to_cpu(act, validation_batch_size),
                "future_obs": _fixed_batch_to_cpu(
                    future_obs, validation_batch_size
                ),
            }
            loguru.logger.info(
                "Captured isolated fixed validation batch: "
                f"batch_size={validation_batch_size}, "
                f"seed={config.log.validation_seed}"
            )

        if config.task.obs_type == "image":
            obs, future_obs = augment_robot_rgb_batch(
                obs, future_obs, config.task
            )

        gradient_diagnostic_freq = config.log.gradient_diagnostic_freq
        compute_gradient_diagnostics = (
            gradient_diagnostic_freq > 0
            and (n_gradient_step + 1) % gradient_diagnostic_freq == 0
        )

        preprocess_seconds = time.perf_counter() - preprocess_started
        update_started = time.perf_counter()
        info = agent.update(
            act,
            obs,
            delta_t,
            future_obs=future_obs,
            sync_metrics=False,
            compute_gradient_diagnostics=compute_gradient_diagnostics,
        )
        info["data_wait_seconds"] = data_wait_seconds
        info["preprocess_seconds"] = preprocess_seconds
        info["update_seconds"] = time.perf_counter() - update_started

        lr_scheduler.step()
        info_list.append(info)

        completed_steps = n_gradient_step + 1
        if completed_steps in snapshot_steps:
            loguru.logger.info(
                f"Save collapse trajectory snapshot at step {completed_steps}..."
            )
            save_trajectory_snapshot(
                logger,
                agent,
                completed_steps,
                best_metrics,
                eval_history,
            )

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
                    values = [
                        step_info[key] for step_info in info_list if key in step_info
                    ]
                    if values and torch.is_tensor(values[0]):
                        metrics[key] = (
                            torch.stack(values).float().mean().item()
                        )
                    else:
                        metrics[key] = np.nanmean(values)
                except Exception:
                    metrics[key] = np.nan
            logger.log(metrics, category="train")
            info_list = []

        if (
            fixed_validation_batch is not None
            and config.log.validation_freq > 0
            and (n_gradient_step + 1) % config.log.validation_freq == 0
        ):
            validation_obs = _fixed_batch_to_device(
                fixed_validation_batch["obs"], config.optimization.device
            )
            validation_act = _fixed_batch_to_device(
                fixed_validation_batch["act"], config.optimization.device
            )
            validation_future_obs = _fixed_batch_to_device(
                fixed_validation_batch["future_obs"],
                config.optimization.device,
            )
            validation_delta_t = torch.full(
                (validation_act.shape[0],),
                config.log.validation_delta_t,
                device=config.optimization.device,
            )
            validation_metrics = agent.validation_metrics(
                validation_act,
                validation_obs,
                validation_delta_t,
                validation_future_obs,
                seed=config.log.validation_seed,
            )
            validation_metrics["step"] = n_gradient_step
            logger.log(validation_metrics, category="val")

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

        if (
            config.log.eval_freq > 0
            and (envs is not None or parallel_pool is not None)
            and ((n_gradient_step + 1) % config.log.eval_freq) == 0
        ):
            loguru.logger.info("Evaluate model...")
            agent.eval()
            metrics = {"step": n_gradient_step}
            num_steps_list = get_default_step_list(config.optimization.loss_type)
            for num_steps in num_steps_list:
                if parallel_pool is not None:
                    try:
                        eval_metrics = parallel_image_eval(
                            config, parallel_pool, dataset, agent, num_steps
                        )
                    except Exception as exc:
                        loguru.logger.exception(
                            f"Persistent parallel eval failed ({exc}); "
                            "closing workers and falling back to serial eval"
                        )
                        parallel_pool.close()
                        parallel_pool = None
                        if envs is None:
                            envs = make_vec_env(
                                config.task, seed=config.optimization.seed
                            )
                        eval_metrics = eval(
                            config, envs, dataset, agent, logger, num_steps
                        )
                else:
                    eval_metrics = eval(
                        config, envs, dataset, agent, logger, num_steps
                    )
                metrics.update(eval_metrics)

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
                    training_state = build_training_state(
                        n_gradient_step=n_gradient_step,
                        best_metrics=best_metrics,
                        eval_history=eval_history,
                    )
                    # Keep the run-local best checkpoint resumable too.
                    logger.save_agent(
                        agent=agent,
                        identifier="best",
                        training_state=training_state,
                    )

                    # Save to global checkpoints directory with success rate comparison
                    checkpoint_base_name = get_checkpoint_base_name(config)
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


def _extract_success_info(info, num_envs):
    """Return per-env success flags from vector env info["success"]."""
    if not isinstance(info, dict) or "success" not in info:
        return np.zeros(num_envs, dtype=bool)

    def reduce_success(value):
        """Reduce nested/object success payloads without NumPy coercion errors."""
        if isinstance(value, dict):
            return any(reduce_success(item) for item in value.values())
        if isinstance(value, np.ndarray):
            if value.shape == ():
                return reduce_success(value.item())
            return any(reduce_success(item) for item in value.reshape(-1))
        if isinstance(value, (list, tuple)):
            return any(reduce_success(item) for item in value)
        return bool(value)

    raw_success = info["success"]
    if isinstance(raw_success, np.ndarray) and raw_success.shape == ():
        return np.full(
            num_envs, reduce_success(raw_success.item()), dtype=bool
        )
    if not isinstance(raw_success, (np.ndarray, list, tuple)):
        return np.full(
            num_envs, reduce_success(raw_success), dtype=bool
        )
    if isinstance(raw_success, (np.ndarray, list, tuple)):
        if len(raw_success) == num_envs:
            return np.asarray(
                [reduce_success(item) for item in raw_success], dtype=bool
            )
        if len(raw_success) == 1:
            return np.full(
                num_envs, reduce_success(raw_success[0]), dtype=bool
            )

    if num_envs == 1:
        return np.asarray([reduce_success(raw_success)], dtype=bool)

    loguru.logger.warning(
        "Unexpected info['success'] payload for "
        f"num_envs={num_envs}; treating success as False."
    )
    return np.zeros(num_envs, dtype=bool)


@isolate_rollout_rng
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
    env_type = getattr(config.task, "env_type", None)
    is_adroit_task = env_type == "adroit"
    is_terminal_success_task = env_type == "adroit" or (
        config.task.env_name in {"can", "lift", "square", "tool_hang", "transport"}
    )

    rollout_seed = get_rollout_seed(config)
    num_envs = int(config.task.num_envs)
    for i in range(config.log.eval_episodes // num_envs):
        episode_ids = list(range(i * num_envs, (i + 1) * num_envs))
        episode_seeds = get_episode_seeds(rollout_seed, episode_ids)
        ep_reward = np.zeros(config.task.num_envs, dtype=np.float32)
        ep_success_info = np.zeros(config.task.num_envs, dtype=bool)
        ep_done = np.zeros(config.task.num_envs, dtype=bool)
        try:
            obs, _ = envs.reset(seed=episode_seeds)
        except TypeError:
            obs, _ = envs.reset(seed=episode_seeds[0])
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

            # MIP ignores the supplied action and starts from zero internally.
            # A zero placeholder avoids consuming rollout RNG for no effect;
            # stochastic samplers create their own noise inside the isolated
            # rollout RNG context.
            act_0 = torch.zeros(
                (config.task.num_envs, config.task.horizon, config.task.act_dim),
                device=config.optimization.device,
            )
            # run sampling (num_envs, horizon, action_dim)
            joint_rollout = bool(
                getattr(config.optimization, "future_joint_mode", False)
            )
            sample_output = agent.sample(
                act_0=act_0,
                obs=obs,
                num_steps=num_steps,
                use_ema=True,
                return_future=joint_rollout,
            )
            if joint_rollout:
                act_normed, future_pred_1 = sample_output
            else:
                act_normed = sample_output

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
            if is_terminal_success_task:
                step_success = _extract_success_info(info, config.task.num_envs)
                ep_success_info |= step_success
                ep_done |= (
                    step_success
                    | np.asarray(terminated, dtype=bool)
                    | np.asarray(truncated, dtype=bool)
                )
            ep_reward += reward
            t += config.task.act_steps
            if is_terminal_success_task and np.all(ep_done):
                break

        reward_positive_success = [1.0 if s > 0 else 0.0 for s in ep_reward]
        success_info = ep_success_info.astype(np.float32).tolist()
        success = (
            success_info if is_terminal_success_task else reward_positive_success
        )

        # evaluate kitchen
        kit_success = []
        if config.task.env_name == "kitchen":
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
        f"mean_step_{num_steps}": float(np.nanmean(episode_steps)),
        f"mean_reward_{num_steps}": float(np.nanmean(episode_rewards)),
        f"mean_success_{num_steps}": float(np.nanmean(episode_success)),
    }
    if is_adroit_task:
        loguru.logger.info(
            f"Nstep: {num_steps} Adroit info success: {metrics[f'mean_success_{num_steps}']}"
        )

    if config.task.env_name == "kitchen":
        mean_kit_success = np.mean(np.array(episode_kit_success), axis=(0, 1))
        kit_metrics = {}
        for i in range(7):
            kit_metrics[f"p{i + 1}_NFE{num_steps}"] = mean_kit_success[i]
        metrics.update(kit_metrics)
        loguru.logger.info(f"Kit metrics: {kit_metrics}")

    return metrics



@isolate_rollout_rng
def parallel_image_eval(config, pool, dataset, agent, num_steps=1):
    """Evaluate image policies with env-only workers and batched GPU inference."""
    workers = pool.num_workers
    success_from_reward = is_libero_task(config.task)
    if config.task.obs_type != "image":
        raise ValueError("Persistent parallel rollout only supports image observations")
    if config.log.save_video or config.task.save_video:
        raise ValueError("Persistent parallel rollout does not support video")
    if config.log.eval_episodes % workers != 0:
        raise ValueError("log.eval_episodes must be divisible by parallel worker count")
    if getattr(config.task, "env_type", None) == "adroit":
        raise ValueError("Persistent parallel rollout does not support Adroit")

    started = time.perf_counter()
    episode_rewards = []
    episode_steps = []
    episode_success = []
    episode_records = []
    reward_success_disagreements = 0

    for first_episode in range(0, config.log.eval_episodes, workers):
        batch_started = time.perf_counter()
        episode_ids = list(range(first_episode, first_episode + workers))
        seeds = get_episode_seeds(get_rollout_seed(config), episode_ids)
        reset_results = pool.reset(seeds)
        obs_parts = [result[0] for result in reset_results]
        ep_reward = np.zeros(workers, dtype=np.float32)
        ep_success = np.zeros(workers, dtype=bool)
        ep_done = np.zeros(workers, dtype=bool)
        ep_lengths = np.zeros(workers, dtype=np.int32)
        t = 0

        while t < config.task.max_episode_steps and not np.all(ep_done):
            active_indices = np.flatnonzero(~ep_done).tolist()
            obs_raw = {
                key: np.concatenate(
                    [obs_parts[index][key] for index in active_indices],
                    axis=0,
                )
                for key in obs_parts[active_indices[0]]
            }
            obs = {}
            for key, value in obs_raw.items():
                value = value.astype(np.float32)
                value = dataset.normalizer["obs"][key].normalize(value)
                obs[key] = torch.as_tensor(
                    value,
                    device=config.optimization.device,
                    dtype=torch.float32,
                )

            # See serial eval above: this is a shape/device placeholder for
            # MIP and must not consume the training process's RNG.
            act_0 = torch.zeros(
                (
                    len(active_indices),
                    config.task.horizon,
                    config.task.act_dim,
                ),
                device=config.optimization.device,
            )
            joint_rollout = bool(
                getattr(config.optimization, "future_joint_mode", False)
            )
            sample_output = agent.sample(
                act_0=act_0,
                obs=obs,
                num_steps=num_steps,
                use_ema=True,
                return_future=joint_rollout,
            )
            if joint_rollout:
                act_normed, future_pred_1 = sample_output
            else:
                act_normed = sample_output
            act = dataset.normalizer["action"].unnormalize(
                act_normed.detach().cpu().numpy()
            )
            start = config.task.obs_steps - 1
            end = start + config.task.act_steps
            act = act[:, start:end, :]
            if config.task.abs_action and config.task.env_name in [
                "can", "lift", "square", "tool_hang", "transport"
            ]:
                act = dataset.undo_transform_action(act)

            step_results = pool.step(
                [act[i : i + 1] for i in range(len(active_indices))],
                worker_indices=active_indices,
            )
            for worker_index, result in zip(
                active_indices, step_results, strict=True
            ):
                observation, reward, terminated, truncated, info = result
                obs_parts[worker_index] = observation
                reward_value = float(
                    np.asarray(reward, dtype=np.float32).reshape(-1)[0]
                )
                ep_reward[worker_index] += reward_value
                # LIBERO signals task completion with its sparse positive reward
                # and does not populate info["success"]. Keep this consistent
                # with serial eval, which also uses reward > 0 for LIBERO.
                success = (
                    reward_value > 0.0
                    if success_from_reward
                    else bool(_extract_success_info(info, 1)[0])
                )
                ep_success[worker_index] |= success
                ep_done[worker_index] |= bool(
                    success
                    or np.asarray(terminated, dtype=bool).reshape(-1)[0]
                    or np.asarray(truncated, dtype=bool).reshape(-1)[0]
                )
                ep_lengths[worker_index] += int(config.task.act_steps)
            t += config.task.act_steps
            rollout_iterations = t // config.task.act_steps
            if rollout_iterations % 10 == 0 or np.all(ep_done):
                worker_alive = [
                    process.is_alive()
                    for process in getattr(pool, "processes", [])
                ]
                loguru.logger.info(
                    "Parallel rollout heartbeat: "
                    f"episodes={episode_ids} sim_step={t}/"
                    f"{config.task.max_episode_steps} "
                    f"active={int(np.count_nonzero(~ep_done))}/{workers} "
                    f"worker_alive={worker_alive} "
                    f"batch_seconds={time.perf_counter() - batch_started:.1f} "
                    f"total_seconds={time.perf_counter() - started:.1f}"
                )

        successes = ep_success.astype(np.float32).tolist()
        reward_success = ep_reward > 0
        reward_success_disagreements += int(
            np.count_nonzero(reward_success != ep_success)
        )
        episode_rewards.extend(ep_reward.tolist())
        episode_steps.extend(ep_lengths.tolist())
        episode_success.extend(successes)
        episode_records.extend(
            {
                "episode_id": episode_id,
                "seed": seed,
                "return": float(reward),
                "success": float(success),
                "length": int(length),
            }
            for episode_id, seed, reward, success, length in zip(
                episode_ids,
                seeds,
                ep_reward,
                successes,
                ep_lengths,
                strict=True,
            )
        )
        loguru.logger.info(
            "Parallel rollout batch complete: "
            f"episodes={episode_ids} lengths={ep_lengths.tolist()} "
            f"successes={successes} "
            f"batch_seconds={time.perf_counter() - batch_started:.1f}"
        )

    elapsed = time.perf_counter() - started
    loguru.logger.info(f"Parallel episode results: {episode_records}")
    loguru.logger.info(
        f"Parallel Nstep: {num_steps} workers: {workers} "
        f"Mean step: {np.nanmean(episode_steps)} "
        f"Mean reward: {np.nanmean(episode_rewards)} "
        f"Mean success: {np.nanmean(episode_success)} "
        f"reward/success disagreements: {reward_success_disagreements} "
        f"rollout_seconds: {elapsed:.3f}"
    )
    return {
        f"mean_step_{num_steps}": float(np.nanmean(episode_steps)),
        f"mean_reward_{num_steps}": float(np.nanmean(episode_rewards)),
        f"mean_success_{num_steps}": float(np.nanmean(episode_success)),
        f"reward_success_disagreements_{num_steps}": int(
            reward_success_disagreements
        ),
        f"parallel_rollout_seconds_{num_steps}": float(elapsed),
    }

@hydra.main(version_base=None, config_path="configs/", config_name="main")
def main(config):
    """Main pipeline function that calls the appropriate standalone function based on mode."""
    validate_runtime_environment(config.task)
    config.optimization.future_target_type = getattr(
        config.task, "future_target_type", config.optimization.future_target_type
    )
    if getattr(config.optimization, "disable_cudnn", False):
        torch.backends.cudnn.enabled = False
    # general config setup
    set_seed(config.optimization.seed)
    logger = Logger(config)
    loguru.logger.info("Finished setting up logger")

    # Preserve the original serial setup by default. In experimental parallel
    # image mode, avoid keeping an unused second set of EGL environments alive;
    # construct the serial env lazily only if fallback is required.
    parallel_requested = bool(getattr(config.eval, "parallel_rollout", False))
    envs = None
    if parallel_requested and config.task.obs_type == "image":
        config.task.obs_dim = config.network.emb_dim
        loguru.logger.info("Deferring serial env creation for parallel image eval")
    else:
        envs = make_vec_env(config.task, seed=get_rollout_seed(config))
        obs, info = envs.reset()
        if config.task.obs_type == "state":
            config.task.obs_dim = obs.shape[-1]
        else:
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
        # Prefer the run-local periodic checkpoint. This remains available even
        # when a spot node is reclaimed before an evaluation has produced a
        # success-tagged global checkpoint.
        local_checkpoint = (
            Path(config.log.log_dir) / "models" / "model_latest.pt"
        )
        checkpoint_path = (
            local_checkpoint if local_checkpoint.is_file() else None
        )
        if checkpoint_path is None:
            checkpoint_base_name = get_checkpoint_base_name(config)
            checkpoint_path = logger.find_latest_checkpoint(
                checkpoint_base_name
            )
        if checkpoint_path:
            loguru.logger.info(f"Found checkpoint to resume from: {checkpoint_path}")
            loguru.logger.info("Loading checkpoint with optimizer state...")
            resume_state = agent.load(str(checkpoint_path), load_optimizer=True)
        else:
            loguru.logger.info("No checkpoint found, starting training from scratch")
    elif config.mode == "train" and not config.optimization.auto_resume:
        loguru.logger.info("Auto-resume disabled, starting training from scratch")

    parallel_pool = None
    if bool(getattr(config.eval, "parallel_rollout", False)):
        try:
            lazy_rollout = config.mode == "train"
            parallel_pool = PersistentImageRolloutPool(
                config, lazy=lazy_rollout
            )
            if lazy_rollout:
                loguru.logger.info(
                    "Persistent parallel image rollout configured lazily: "
                    f"workers={parallel_pool.num_workers}; workers will start "
                    "at the first evaluation"
                )
            else:
                loguru.logger.info(
                    f"Persistent parallel image rollout ready: "
                    f"workers={parallel_pool.num_workers}, "
                    f"startup_seconds={parallel_pool.startup_seconds:.3f}"
                )
        except Exception as exc:
            loguru.logger.exception(
                f"Could not start persistent parallel rollout ({exc}); "
                "using original serial eval"
            )
            parallel_pool = None

    if parallel_pool is None and envs is None:
        envs = make_vec_env(config.task, seed=get_rollout_seed(config))
        loguru.logger.info("Created serial env after parallel startup fallback")

    try:
        if config.mode == "train":
            train(
                config, envs, dataset, agent, logger, resume_state=resume_state,
                parallel_pool=parallel_pool,
            )
        elif config.mode == "eval":
            if not config.optimization.model_path:
                raise ValueError("Empty model for inference")
            agent.eval()

            num_steps_list = get_default_step_list(config.optimization.loss_type)
            for num_steps in num_steps_list:
                metrics = {"step": num_steps}
                if parallel_pool is not None:
                    try:
                        eval_metrics = parallel_image_eval(
                            config, parallel_pool, dataset, agent, num_steps
                        )
                    except Exception as exc:
                        loguru.logger.exception(
                            f"Persistent parallel eval failed ({exc}); "
                            "falling back to serial eval"
                        )
                        parallel_pool.close()
                        parallel_pool = None
                        if envs is None:
                            envs = make_vec_env(
                                config.task, seed=get_rollout_seed(config)
                            )
                        eval_metrics = eval(
                            config, envs, dataset, agent, logger, num_steps
                        )
                else:
                    eval_metrics = eval(
                        config, envs, dataset, agent, logger, num_steps
                    )
                metrics.update(eval_metrics)
                logger.log(metrics, category="eval")

            for key, val in metrics.items():
                if "mean_success" in key:
                    loguru.logger.info(f"{key} - {val}")
        else:
            raise ValueError("Illegal mode")
    finally:
        if parallel_pool is not None:
            parallel_pool.close()
        if envs is not None:
            envs.close()
        # Explicitly finalize offline W&B history before the launcher syncs it.
        # Passing None avoids an extra final checkpoint; periodic checkpoints
        # remain controlled by log.save_freq.
        logger.finish(None)


if __name__ == "__main__":
    main()
