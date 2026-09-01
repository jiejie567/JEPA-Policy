if __name__ == "__main__":
    import sys
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import copy
import os
import pathlib
import random
import time

import hydra
import numpy as np
import torch
import tqdm
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.policy.diffusion_transformer_aligned_image_policy import (
    AlignedDiffusionTransformerHybridImagePolicy,
)
from diffusion_policy.workspace.base_workspace import BaseWorkspace


OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainDiffusionTransformerH10AlignedWorkspace(BaseWorkspace):
    """Step-exact formal workspace for the horizon-10 transformer baseline."""

    include_keys = ["global_step", "optimizer_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: AlignedDiffusionTransformerHybridImagePolicy = (
            hydra.utils.instantiate(cfg.policy)
        )
        self.ema_model: AlignedDiffusionTransformerHybridImagePolicy | None = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)
        self.global_step = 0
        self.optimizer_step = 0
        self.epoch = 0

    @staticmethod
    def _due(step: int, interval) -> bool:
        return interval is not None and step > 0 and step % int(interval) == 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        if cfg.training.resume:
            latest_ckpt_path = self.get_checkpoint_path()
            resume_path = latest_ckpt_path
            if not resume_path.is_file():
                step_checkpoints = sorted(
                    latest_ckpt_path.parent.glob("step=*.ckpt")
                )
                if step_checkpoints:
                    resume_path = step_checkpoints[-1]
            if resume_path.is_file():
                print(f"Resuming from checkpoint {resume_path}")
                self.load_checkpoint(path=resume_path)

        dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        if len(train_dataloader) == 0:
            raise RuntimeError("training dataloader is empty")

        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)

        max_updates = cfg.training.max_global_steps
        if max_updates is None:
            max_updates = (
                len(train_dataloader)
                * cfg.training.num_epochs
                // cfg.training.gradient_accumulate_every
            )
        max_updates = int(max_updates)
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=max_updates,
            last_epoch=self.optimizer_step - 1,
        )

        ema: EMAModel | None = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        env_runner: BaseImageRunner = hydra.utils.instantiate(
            cfg.task.env_runner, output_dir=self.output_dir
        )
        assert isinstance(env_runner, BaseImageRunner)

        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging,
        )
        wandb.config.update({"output_dir": self.output_dir})
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )

        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        self.optimizer.zero_grad(set_to_none=True)

        if cfg.training.debug:
            max_updates = 2
            cfg.training.rollout_every_steps = None
            cfg.training.val_every_steps = None
            cfg.training.sample_every_steps = 1
            cfg.training.checkpoint_every_steps = 1

        train_iterator = iter(train_dataloader)
        batch_idx = 0
        train_sampling_batch = None
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        progress = tqdm.tqdm(
            total=max_updates,
            initial=self.optimizer_step,
            desc="Training updates",
            mininterval=cfg.training.tqdm_interval_sec,
        )

        with JsonLogger(log_path) as json_logger:
            while self.optimizer_step < max_updates:
                data_wait_started = time.perf_counter()
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    self.epoch += 1
                    batch_idx = 0
                    train_iterator = iter(train_dataloader)
                    batch = next(train_iterator)
                data_wait_seconds = time.perf_counter() - data_wait_started

                preprocess_started = time.perf_counter()
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                if train_sampling_batch is None:
                    train_sampling_batch = dict_apply(batch, lambda x: x[:16].detach())
                preprocess_seconds = time.perf_counter() - preprocess_started

                update_started = time.perf_counter()
                raw_loss = self.model.compute_loss(batch)
                loss = raw_loss / cfg.training.gradient_accumulate_every
                loss.backward()

                micro_step = self.global_step + 1
                optimizer_due = (
                    micro_step % cfg.training.gradient_accumulate_every == 0
                )
                if optimizer_due:
                    if cfg.training.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), cfg.training.grad_clip_norm
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    lr_scheduler.step()
                    self.optimizer_step += 1
                    if ema is not None:
                        ema.step(self.model)
                    progress.update(1)
                self.global_step = micro_step
                batch_idx += 1
                update_seconds = time.perf_counter() - update_started

                step_log = {
                    "train_loss": raw_loss.item(),
                    "global_step": self.global_step,
                    "optimizer_step": self.optimizer_step,
                    "epoch": self.epoch,
                    "lr": lr_scheduler.get_last_lr()[0],
                    "data_wait_seconds": data_wait_seconds,
                    "preprocess_seconds": preprocess_seconds,
                    "update_seconds": update_seconds,
                }
                progress.set_postfix(loss=step_log["train_loss"], refresh=False)

                if not optimizer_due:
                    wandb_run.log(step_log, step=self.optimizer_step)
                    json_logger.log(step_log)
                    continue

                rollout_due = self._due(
                    self.optimizer_step, cfg.training.rollout_every_steps
                )
                val_due = self._due(
                    self.optimizer_step, cfg.training.val_every_steps
                )
                sample_due = self._due(
                    self.optimizer_step, cfg.training.sample_every_steps
                )
                checkpoint_due = self._due(
                    self.optimizer_step, cfg.training.checkpoint_every_steps
                )
                eval_due = rollout_due or val_due or sample_due or checkpoint_due

                if eval_due:
                    policy = self.ema_model if self.ema_model is not None else self.model
                    policy.eval()
                    if rollout_due:
                        step_log.update(env_runner.run(policy))
                    if val_due and len(val_dataloader) > 0:
                        with torch.no_grad():
                            val_losses = []
                            for val_idx, val_batch in enumerate(val_dataloader):
                                val_batch = dict_apply(
                                    val_batch,
                                    lambda x: x.to(device, non_blocking=True),
                                )
                                val_losses.append(self.model.compute_loss(val_batch))
                                if (
                                    cfg.training.max_val_steps is not None
                                    and val_idx >= cfg.training.max_val_steps - 1
                                ):
                                    break
                            if val_losses:
                                step_log["val_loss"] = torch.stack(val_losses).mean().item()
                    if sample_due:
                        with torch.no_grad():
                            sample_batch = dict_apply(
                                train_sampling_batch,
                                lambda x: x.to(device, non_blocking=True),
                            )
                            result = policy.predict_action(sample_batch["obs"])
                            step_log["train_action_mse_error"] = (
                                torch.nn.functional.mse_loss(
                                    result["action_pred"], sample_batch["action"]
                                ).item()
                            )
                    policy.train()

                if checkpoint_due:
                    if bool(cfg.checkpoint.get("save_step_ckpt", False)):
                        step_ckpt_path = pathlib.Path(self.output_dir).joinpath(
                            "checkpoints", f"step={self.optimizer_step:06d}.ckpt"
                        )
                        if step_ckpt_path.exists():
                            raise FileExistsError(
                                f"refusing to overwrite immutable checkpoint: {step_ckpt_path}"
                            )
                        self.save_checkpoint(path=step_ckpt_path, use_thread=False)
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()
                    metric_dict = {
                        key.replace("/", "_"): value
                        for key, value in step_log.items()
                    }
                    if cfg.checkpoint.topk.monitor_key in metric_dict:
                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                        if topk_ckpt_path is not None:
                            self.save_checkpoint(path=topk_ckpt_path)

                wandb_run.log(step_log, step=self.optimizer_step)
                json_logger.log(step_log)

        progress.close()


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = TrainDiffusionTransformerH10AlignedWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
