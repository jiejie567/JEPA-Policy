"""Torch training agent for behavior cloning."""

from copy import deepcopy

import loguru
import torch
import torch.nn as nn

from mip.config import Config
from mip.flow_map import FlowMap
from mip.interpolant import Interpolant
from mip.losses import get_loss_fn, get_norm
from mip.network_utils import get_encoder, get_network
from mip.samplers import get_sampler
from mip.sigreg import SIGReg
from mip.torch_utils import report_parameters


class TrainingAgent:
    """Training agent for behavior cloning with flow matching."""

    def __init__(
        self,
        config: Config,
    ):
        """Initialize the training agent.

        Args:
            flow_map: The flow map model
            encoder: The observation encoder
            config: Full configuration object
        """
        self.config = config
        self.loss_fn = get_loss_fn(config.optimization.loss_type)
        self.sampler = get_sampler(config.optimization.loss_type)
        self.interpolant = Interpolant(config.optimization.interp_type)
        net = get_network(config.network, config.task)
        report_parameters(net, model_name="Action Network")
        self.flow_map = FlowMap(net).to(config.optimization.device)
        self.encoder = get_encoder(config.network, config.task).to(
            config.optimization.device
        )
        report_parameters(self.encoder, model_name="Encoder Network")
        self.encoder_ema = deepcopy(self.encoder).requires_grad_(False)
        self.flow_map_ema = deepcopy(self.flow_map).requires_grad_(False)
        self._future_nonfinite_logged = False

        self._maybe_load_encoder_checkpoint()
        self._maybe_freeze_encoder()
        self.sigreg = None
        if getattr(config.optimization, "use_sigreg", False):
            self.sigreg = SIGReg(
                knots=getattr(config.optimization, "sigreg_knots", 17),
                num_proj=getattr(config.optimization, "sigreg_num_proj", 1024),
            ).to(config.optimization.device)
            loguru.logger.info(
                "SIGReg enabled "
                f"(weight={config.optimization.sigreg_weight}, "
                f"knots={config.optimization.sigreg_knots}, "
                f"num_proj={config.optimization.sigreg_num_proj})"
            )

        params = list(self.flow_map.parameters())
        if not config.optimization.freeze_encoder:
            params = list(self.encoder.parameters()) + params
        self.optimizer = torch.optim.AdamW(
            params,
            lr=config.optimization.lr,
            weight_decay=config.optimization.weight_decay,
        )

        # Compile training and sampling functions for faster execution
        self.use_compile = config.optimization.use_compile
        if self.use_compile:
            loguru.logger.info("Compiling forward+backward with torch.compile")
            # Compile the training step (forward + backward)
            self._compute_loss_and_grads = torch.compile(
                self._compute_loss_and_grads_impl
            )
            loguru.logger.info("Compiling sampler with torch.compile")
            # Compile the sampler (used during evaluation)
            # Note: Some networks with dynamic shapes may trigger compilation warnings
            self._compiled_sampler = torch.compile(self.sampler)
        else:
            self._compute_loss_and_grads = self._compute_loss_and_grads_impl
            self._compiled_sampler = self.sampler

    def _maybe_load_encoder_checkpoint(self):
        checkpoint_path = getattr(
            self.config.optimization, "encoder_checkpoint_path", None
        )
        if not checkpoint_path or checkpoint_path == "None":
            return

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.config.optimization.device,
            weights_only=False,
        )
        use_ema = getattr(
            self.config.optimization, "encoder_checkpoint_use_ema", True
        )
        encoder_key = "encoder_ema" if use_ema and "encoder_ema" in checkpoint else "encoder"
        if encoder_key not in checkpoint:
            raise KeyError(
                f"Checkpoint {checkpoint_path} does not contain `{encoder_key}`"
            )

        self.encoder.load_state_dict(checkpoint[encoder_key], strict=True)
        self.encoder_ema.load_state_dict(checkpoint[encoder_key], strict=True)
        loguru.logger.info(
            f"Loaded encoder weights from {checkpoint_path} using key `{encoder_key}`"
        )

    def _maybe_freeze_encoder(self):
        if not getattr(self.config.optimization, "freeze_encoder", False):
            return
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.encoder_ema.requires_grad_(False)
        self.encoder_ema.eval()
        loguru.logger.info("Encoder frozen for training")

    def _encode_future_target(self, future_obs: dict[str, torch.Tensor]) -> torch.Tensor:
        if hasattr(self.encoder, "encode_raw_dino"):
            return self.encoder.encode_raw_dino(future_obs)

        encoder_input = future_obs
        reduce_sequence_output = False
        if (
            isinstance(future_obs, dict)
            and "state" in future_obs
            and future_obs["state"].dim() == 2
            and hasattr(self.encoder, "To")
        ):
            encoder_input = {
                "state": future_obs["state"].unsqueeze(1).expand(
                    -1, self.encoder.To, -1
                )
            }
            reduce_sequence_output = True
        if getattr(self.encoder, "use_seq", False):
            encoder_input = {}
            for k, v in future_obs.items():
                # Single future frame comes in as:
                #   rgb:    (B, C, H, W)
                #   lowdim: (B, D)
                # Multi-future frames already come in as:
                #   rgb:    (B, T, C, H, W)
                #   lowdim: (B, T, D)
                # The image encoder expects the latter sequence-aware format.
                if v.dim() in (2, 4):
                    encoder_input[k] = v.unsqueeze(1)
                else:
                    encoder_input[k] = v

        future_embed_target = self.encoder(encoder_input, None)
        if reduce_sequence_output and future_embed_target.dim() == 3:
            future_embed_target = future_embed_target.mean(dim=1)
        if future_embed_target.dim() == 3 and future_embed_target.shape[1] == 1:
            future_embed_target = future_embed_target[:, 0, :]
        return future_embed_target

    def _copy_obs(self, obs):
        if isinstance(obs, dict):
            return {k: v.clone() for k, v in obs.items()}
        return obs.clone() if torch.is_tensor(obs) else obs

    def _compute_direct_future_embed_loss(
        self,
        act: torch.Tensor,
        obs: torch.Tensor | dict,
        delta_t: torch.Tensor,
        future_obs: torch.Tensor | dict,
    ):
        obs_emb = self.encoder(obs, None)
        future_embed_target = self._encode_future_target(future_obs).detach()
        s = torch.zeros_like(delta_t, device=delta_t.device)
        t = torch.ones_like(delta_t, device=delta_t.device)
        _, _, future_embed_pred = self.flow_map.net(act, s, t, obs_emb)
        if future_embed_pred is None:
            raise RuntimeError(
                "Direct future embedding loss requires a network configured with future tokens"
            )

        if future_embed_pred.dim() == 2 and future_embed_target.dim() == 3:
            if future_embed_target.shape[1] != 1:
                raise RuntimeError(
                    "Future target has multiple steps but network only outputs one future token"
                )
            future_embed_target = future_embed_target[:, 0, :]
        elif future_embed_pred.dim() == 3 and future_embed_target.dim() == 2:
            if future_embed_pred.shape[1] != 1:
                raise RuntimeError(
                    "Network outputs multiple future tokens but future target has one step"
                )
            future_embed_pred = future_embed_pred[:, 0, :]

        if future_embed_pred.shape != future_embed_target.shape:
            raise RuntimeError(
                "Future prediction/target shape mismatch: "
                f"pred={tuple(future_embed_pred.shape)} "
                f"target={tuple(future_embed_target.shape)}"
            )

        future_loss = torch.mean((future_embed_pred - future_embed_target) ** 2)
        info = {
            "loss_future": future_loss.detach(),
            "loss_future_raw": future_loss.detach(),
        }
        return future_loss, info

    def _compute_mip_two_step_future_embed_loss(
        self,
        act: torch.Tensor,
        obs: torch.Tensor | dict,
        delta_t: torch.Tensor,
        future_obs: torch.Tensor | dict,
    ):
        obs_emb = self.encoder(obs, None)
        s = torch.zeros_like(delta_t, device=delta_t.device)
        t = (
            torch.zeros_like(delta_t, device=delta_t.device)
            + self.config.optimization.t_two_step
        )
        future_embed_target = self._encode_future_target(future_obs).detach()
        future_embed_0 = torch.zeros_like(future_embed_target)
        future_noise = torch.randn_like(future_embed_target)
        future_embed_t = (
            future_embed_target
            + (1 - self.config.optimization.t_two_step) * future_noise
        )
        act_0 = torch.zeros_like(act)
        act_noise = torch.randn_like(act)
        act_t = act + (1 - self.config.optimization.t_two_step) * act_noise

        _, _, future_embed_pred_0 = self.flow_map.net(
            act_0, s, t, obs_emb, future_input=future_embed_0
        )
        _, _, future_embed_pred_1 = self.flow_map.net(
            act_t,
            t,
            torch.ones_like(t),
            obs_emb,
            future_input=future_embed_t,
        )
        if future_embed_pred_0 is None or future_embed_pred_1 is None:
            raise RuntimeError(
                "MIP-style future embedding loss requires a network configured with future tokens"
            )

        if future_embed_pred_0.dim() == 2 and future_embed_target.dim() == 3:
            if future_embed_target.shape[1] != 1:
                raise RuntimeError(
                    "Future target has multiple steps but network only outputs one future token"
                )
            future_embed_target = future_embed_target[:, 0, :]
        elif future_embed_pred_0.dim() == 3 and future_embed_target.dim() == 2:
            if future_embed_pred_0.shape[1] != 1:
                raise RuntimeError(
                    "Network outputs multiple future tokens but future target has one step"
                )
            future_embed_pred_0 = future_embed_pred_0[:, 0, :]
            future_embed_pred_1 = future_embed_pred_1[:, 0, :]

        if future_embed_pred_0.shape != future_embed_target.shape:
            raise RuntimeError(
                "Future prediction/target shape mismatch: "
                f"pred={tuple(future_embed_pred_0.shape)} "
                f"target={tuple(future_embed_target.shape)}"
            )

        future_loss_0 = torch.mean(
            ((future_embed_pred_0 - future_embed_target) / self.config.optimization.t_two_step)
            ** 2
        )
        future_loss_1 = torch.mean(
            ((future_embed_pred_1 - future_embed_target) / (1 - self.config.optimization.t_two_step))
            ** 2
        )
        future_loss = future_loss_0 + future_loss_1
        info = {
            "loss_future": future_loss.detach(),
            "loss_future_raw": future_loss.detach(),
        }
        return future_loss, info

    def _compute_joint_mip_two_step_loss(
        self,
        act: torch.Tensor,
        obs: torch.Tensor | dict,
        delta_t: torch.Tensor,
        future_obs: torch.Tensor | dict,
    ):
        obs_emb = self.encoder(obs, None)
        s = torch.zeros_like(delta_t, device=delta_t.device)
        t = torch.zeros_like(delta_t, device=delta_t.device) + self.config.optimization.t_two_step
        future_embed_target = self._encode_future_target(future_obs).detach()
        future_embed_0 = torch.zeros_like(future_embed_target)
        future_noise = torch.randn_like(future_embed_target)
        future_embed_t = future_embed_target + (1 - self.config.optimization.t_two_step) * future_noise
        act_0 = torch.zeros_like(act)
        act_noise = torch.randn_like(act)
        act_t = act + (1 - self.config.optimization.t_two_step) * act_noise

        act_pred_0, _, future_embed_pred_0 = self.flow_map.net(
            act_0, s, t, obs_emb, future_input=future_embed_0
        )
        act_pred_1, _, future_embed_pred_1 = self.flow_map.net(
            act_t,
            t,
            torch.ones_like(t),
            obs_emb,
            future_input=future_embed_t,
        )
        if future_embed_pred_0 is None or future_embed_pred_1 is None:
            raise RuntimeError(
                "Joint MIP two-step loss requires a network configured with future tokens"
            )

        if future_embed_pred_0.dim() == 2 and future_embed_target.dim() == 3:
            if future_embed_target.shape[1] != 1:
                raise RuntimeError(
                    "Future target has multiple steps but network only outputs one future token"
                )
            future_embed_target = future_embed_target[:, 0, :]
        elif future_embed_pred_0.dim() == 3 and future_embed_target.dim() == 2:
            if future_embed_pred_0.shape[1] != 1:
                raise RuntimeError(
                    "Network outputs multiple future tokens but future target has one step"
                )
            future_embed_pred_0 = future_embed_pred_0[:, 0, :]
            future_embed_pred_1 = future_embed_pred_1[:, 0, :]

        if future_embed_pred_0.shape != future_embed_target.shape:
            raise RuntimeError(
                "Future prediction/target shape mismatch: "
                f"pred={tuple(future_embed_pred_0.shape)} "
                f"target={tuple(future_embed_target.shape)}"
            )

        action_loss_0 = (
            get_norm(act_pred_0 - act, self.config.optimization.norm_type)
            / self.config.optimization.t_two_step
        ) ** 2
        action_loss_1 = (
            get_norm(act_pred_1 - act, self.config.optimization.norm_type)
            / (1 - self.config.optimization.t_two_step)
        ) ** 2
        action_loss = torch.mean(action_loss_0 + action_loss_1)

        future_loss_0 = torch.mean(
            (
                (future_embed_pred_0 - future_embed_target)
                / self.config.optimization.t_two_step
            )
            ** 2
        )
        future_loss_1 = torch.mean(
            (
                (future_embed_pred_1 - future_embed_target)
                / (1 - self.config.optimization.t_two_step)
            )
            ** 2
        )
        future_loss = future_loss_0 + future_loss_1
        loss = (
            self.config.optimization.loss_scale * action_loss
            + self.config.optimization.future_embed_loss_weight * future_loss
        )
        info = {
            "loss_action": action_loss.detach(),
            "loss_action_raw": action_loss.detach(),
            "loss_future": future_loss.detach(),
            "loss_future_raw": future_loss.detach(),
        }
        return loss, info

    def _tensor_stats(self, name: str, tensor: torch.Tensor) -> str:
        finite = torch.isfinite(tensor)
        num_bad = (~finite).sum().item()
        if finite.any():
            finite_vals = tensor[finite]
            return (
                f"{name}: shape={tuple(tensor.shape)} bad={num_bad} "
                f"min={finite_vals.min().item():.4e} "
                f"max={finite_vals.max().item():.4e} "
                f"mean={finite_vals.mean().item():.4e}"
            )
        return f"{name}: shape={tuple(tensor.shape)} bad={num_bad} all_nonfinite"

    def _log_future_nonfinite_once(
        self,
        future_embed_target: torch.Tensor,
        future_embed_pred: torch.Tensor,
        obs_emb: torch.Tensor,
        act: torch.Tensor,
        future_obs: dict[str, torch.Tensor] | torch.Tensor | None,
    ):
        if self._future_nonfinite_logged:
            return
        self._future_nonfinite_logged = True

        lines = [
            "Future embedding branch produced non-finite values.",
            self._tensor_stats("future_embed_target", future_embed_target),
            self._tensor_stats("future_embed_pred", future_embed_pred),
            self._tensor_stats("obs_emb", obs_emb),
            self._tensor_stats("act", act),
        ]

        if isinstance(future_obs, dict):
            for key, value in future_obs.items():
                lines.append(self._tensor_stats(f"future_obs[{key}]", value))
        elif torch.is_tensor(future_obs):
            lines.append(self._tensor_stats("future_obs", future_obs))

        loguru.logger.warning("\n".join(lines))

    def _compute_loss_and_grads_impl(
        self, act: torch.Tensor, obs: torch.Tensor, delta_t: torch.Tensor
    ):
        """Compute loss and gradients (this function will be compiled).

        Args:
            act: Action tensor of shape (batch_size, Ta, act_dim)
            obs: Observation tensor of shape (batch_size, To, obs_dim)
            delta_t: Time step differences of shape (batch_size,)

        Returns:
            Tuple of (loss, info_dict)
        """
        loss, info = self.loss_fn(
            self.config.optimization,
            self.flow_map,
            self.encoder,
            self.interpolant,
            act,
            obs,
            delta_t,
        )
        loss.backward()
        return loss, info

    def update(
        self,
        act: torch.Tensor,
        obs: torch.Tensor,
        delta_t: torch.Tensor,
        future_obs: torch.Tensor | dict | None = None,
    ):
        """Update the model parameters with a training batch."""
        self.optimizer.zero_grad(set_to_none=True)
        obs_for_sigreg = self._copy_obs(obs)
        obs_for_future = self._copy_obs(obs)

        use_direct_future_embed_loss = (
            future_obs is not None
            and getattr(self.config.optimization, "use_future_embed_loss", False)
            and getattr(self.config.task, "future_state_enabled", False)
        )

        future_mode = getattr(
            self.config.optimization, "future_embed_loss_mode", "direct"
        )
        use_joint_future_mip_loss = (
            use_direct_future_embed_loss
            and self.config.optimization.loss_type == "mip"
            and future_mode == "mip_two_step"
        )

        if use_joint_future_mip_loss:
            loss, info = self._compute_joint_mip_two_step_loss(
                act,
                obs_for_future,
                delta_t,
                future_obs,
            )
        elif (
            future_obs is not None
            and self.config.optimization.loss_type == "mip"
            and getattr(self.config.task, "future_state_enabled", False)
            and not use_direct_future_embed_loss
        ):
            loss, info = self.loss_fn(
                self.config.optimization,
                self.flow_map,
                self.encoder,
                self.interpolant,
                act,
                obs,
                delta_t,
                future_obs=future_obs,
            )
        else:
            loss, info = self.loss_fn(
                self.config.optimization,
                self.flow_map,
                self.encoder,
                self.interpolant,
                act,
                obs,
                delta_t,
            )

        if use_direct_future_embed_loss and not use_joint_future_mip_loss:
            if future_mode == "direct":
                future_loss, future_info = self._compute_direct_future_embed_loss(
                    act,
                    obs_for_future,
                    delta_t,
                    future_obs,
                )
            else:
                raise ValueError(
                    f"Unsupported future_embed_loss_mode: {future_mode}"
                )
            loss = loss + self.config.optimization.future_embed_loss_weight * future_loss
            info.update(future_info)

        if not torch.isfinite(loss):
            raise RuntimeError("Training loss became non-finite during training")

        sigreg_loss = None
        if self.sigreg is not None and self.config.optimization.sigreg_weight > 0:
            # LeWorldModel-style SIGReg regularizes the encoder latent space itself,
            # not the future head output. We therefore apply it directly to the
            # current observation embeddings produced by the encoder.
            obs_emb = self.encoder(obs_for_sigreg, None)
            if obs_emb.dim() == 2:
                sigreg_input = obs_emb.unsqueeze(0)
            elif obs_emb.dim() == 3:
                sigreg_input = obs_emb.transpose(0, 1)
            else:
                raise ValueError(
                    f"Unsupported encoder embedding shape for SIGReg: {tuple(obs_emb.shape)}"
                )
            sigreg_loss = self.sigreg(sigreg_input)
            if not torch.isfinite(sigreg_loss):
                raise RuntimeError("SIGReg loss became non-finite during training")
            loss = loss + self.config.optimization.sigreg_weight * sigreg_loss

        loss.backward()

        params = list(self.encoder.parameters()) + list(self.flow_map.parameters())
        grad_norm = (
            nn.utils.clip_grad_norm_(params, self.config.optimization.grad_clip_norm)
            if self.config.optimization.grad_clip_norm
            else torch.zeros(1)
        )

        self.optimizer.step()

        if self.config.optimization.ema_rate < 1:
            self.ema_update()

        out = {"loss": loss.item(), "grad_norm": grad_norm.item()}
        for key, value in info.items():
            if isinstance(value, torch.Tensor):
                out[key] = value.item()
            else:
                out[key] = value
        if sigreg_loss is not None:
            out["loss_sigreg_encoder"] = sigreg_loss.item()
        return out


    def ema_update(self):
        """Update exponential moving average parameters."""
        params = list(self.encoder.parameters()) + list(self.flow_map.parameters())
        ema_params = list(self.encoder_ema.parameters()) + list(
            self.flow_map_ema.parameters()
        )
        with torch.no_grad():
            for p, p_ema in zip(params, ema_params, strict=False):
                p_ema.data.mul_(self.config.optimization.ema_rate).add_(
                    p.data, alpha=1.0 - self.config.optimization.ema_rate
                )

    def sample(
        self,
        act_0: torch.Tensor,
        obs: torch.Tensor,
        num_steps: int = -1,
        use_ema: bool = True,
    ):
        """Sample actions from the learned policy.

        Args:
            act_0: Initial action tensor of shape (batch_size, Ta, act_dim)
            obs: Observation tensor of shape (batch_size, To, obs_dim)
            num_steps: Number of sampling steps (default: use config value)
            use_ema: Whether to use EMA parameters for sampling

        Returns:
            Sampled action tensor of shape (batch_size, Ta, act_dim)
        """
        # manually set num_steps if needed
        if num_steps >= 1:
            config = deepcopy(self.config.optimization)
            config.num_steps = int(num_steps)
        else:
            config = self.config.optimization
        # choose model
        if self.config.optimization.ema_rate < 1:
            if use_ema:
                flow_map = self.flow_map_ema
                encoder = self.encoder_ema
            else:
                flow_map = self.flow_map
                encoder = self.encoder
        else:
            flow_map = self.flow_map
            encoder = self.encoder
        with torch.no_grad():
            act = self._compiled_sampler(config, flow_map, encoder, act_0, obs)
        return act

    def save(self, path: str, training_state: dict = None):
        """Save agent models to path.

        Args:
            path: Path to save checkpoint
            training_state: Optional dict with training state (n_gradient_step, best_metrics, eval_history)
        """
        # save flow map, encoder, encoder_ema, flow_map_ema, optimizer
        checkpoint = {
            "flow_map": self.flow_map.state_dict(),
            "encoder": self.encoder.state_dict(),
            "encoder_ema": self.encoder_ema.state_dict(),
            "flow_map_ema": self.flow_map_ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        # Add training state if provided
        if training_state is not None:
            checkpoint["training_state"] = training_state

        torch.save(checkpoint, path)

    def load(self, path: str, load_optimizer: bool = False):
        """Load agent models from path.

        Args:
            path: Path to load checkpoint from
            load_optimizer: Whether to load optimizer state

        Returns:
            training_state dict if available, None otherwise
        """
        # load flow map, encoder, encoder_ema, flow_map_ema
        state_dict = torch.load(
            path,
            map_location=self.config.optimization.device,
            weights_only=False,
        )
        self.flow_map.load_state_dict(state_dict["flow_map"])
        self.encoder.load_state_dict(state_dict["encoder"])
        self.encoder_ema.load_state_dict(state_dict["encoder_ema"])
        self.flow_map_ema.load_state_dict(state_dict["flow_map_ema"])

        # Load optimizer state if requested and available
        if load_optimizer and "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])
            loguru.logger.info("Loaded optimizer state")

        # Return training state if available
        training_state = state_dict.get("training_state", None)
        if training_state:
            loguru.logger.info(f"Loaded training state from step {training_state.get('n_gradient_step', 'unknown')}")

        return training_state

    def eval(self):
        """Set all models to evaluation mode."""
        self.flow_map.eval()
        self.encoder.eval()
        self.flow_map_ema.eval()
        self.encoder_ema.eval()

    def train(self):
        """Set all models to training mode."""
        self.flow_map.train()
        if getattr(self.config.optimization, "freeze_encoder", False):
            self.encoder.eval()
        else:
            self.encoder.train()
        self.flow_map_ema.train()
        self.encoder_ema.eval()
