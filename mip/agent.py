"""Torch training agent for behavior cloning."""

from copy import deepcopy

import loguru
import torch
import torch.nn as nn
from mip.config import Config
from mip.flow_map import FlowMap
from mip.interpolant import Interpolant
from mip.losses import get_loss_fn
from mip.network_utils import get_encoder, get_network
from mip.samplers import JointSamplerMode, JointSamplerResult, get_sampler, joint_mip_sampler
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
        loguru.logger.info(
            "Image encoder config: "
            f"rgb_model_name={getattr(config.network, 'rgb_model_name', None)}, "
            f"rgb_model_weights={getattr(config.network, 'rgb_model_weights', None)}, "
            f"imagenet_norm={getattr(config.network, 'imagenet_norm', False)}, "
            f"freeze_encoder={getattr(config.optimization, 'freeze_encoder', False)}, "
            f"use_group_norm={getattr(config.task, 'use_group_norm', False)}"
        )
        self.encoder = get_encoder(config.network, config.task).to(
            config.optimization.device
        )
        report_parameters(self.encoder, model_name="Encoder Network")
        loguru.logger.info(
            "Encoder obs feature reuse enabled for action/future losses "
            "(one crop and one encoder forward per training step)"
        )
        future_out_dim = None
        if (
            getattr(config.optimization, "use_future_embed_loss", False)
            and getattr(config.network, "n_future_tokens", 0) > 0
            and hasattr(self.encoder, "rgb_feature_dim")
        ):
            future_out_dim = self.encoder.rgb_feature_dim()
            loguru.logger.info(
                f"Using RGB-only future target dim: {future_out_dim}"
            )
        net = get_network(config.network, config.task, future_out_dim=future_out_dim)
        report_parameters(net, model_name="Action Network")
        self.flow_map = FlowMap(net).to(config.optimization.device)
        self.encoder_ema = deepcopy(self.encoder).requires_grad_(False)
        self.flow_map_ema = deepcopy(self.flow_map).requires_grad_(False)
        self._foreach_ema_enabled = hasattr(torch, "_foreach_lerp_")
        self._foreach_ema_warned = False
        if self._foreach_ema_enabled:
            try:
                ema_params = list(self.encoder_ema.parameters()) + list(
                    self.flow_map_ema.parameters()
                )
                params = list(self.encoder.parameters()) + list(
                    self.flow_map.parameters()
                )
                with torch.no_grad():
                    torch._foreach_lerp_(ema_params, params, 0.0)
            except (RuntimeError, TypeError, NotImplementedError):
                self._foreach_ema_enabled = False
        loguru.logger.info(
            "EMA update backend: "
            + ("torch._foreach_lerp_" if self._foreach_ema_enabled else "parameter loop fallback")
        )
        self._future_nonfinite_logged = False

        self._maybe_load_encoder_checkpoint()
        self._maybe_freeze_encoder()
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

    def _encode_future_target(
        self,
        future_obs: dict[str, torch.Tensor],
        *,
        encoder: nn.Module | None = None,
    ) -> torch.Tensor:
        encoder = self.encoder if encoder is None else encoder
        if hasattr(encoder, "encode_rgb_features"):
            return encoder.encode_rgb_features(future_obs)

        encoder_input = future_obs
        reduce_sequence_output = False
        if (
            isinstance(future_obs, dict)
            and "state" in future_obs
            and future_obs["state"].dim() == 2
            and hasattr(encoder, "To")
        ):
            encoder_input = {
                "state": future_obs["state"].unsqueeze(1).expand(
                    -1, encoder.To, -1
                )
            }
            reduce_sequence_output = True
        if getattr(encoder, "use_seq", False):
            encoder_input = {}
            for k, v in future_obs.items():
                # Single future frame comes in as:
                #   rgb:    (B, C, H, W)
                #   lowdim: (B, D)
                #   point_cloud: (B, N, C)
                # Multi-future frames already come in as:
                #   rgb:    (B, T, C, H, W)
                #   lowdim: (B, T, D)
                #   point_cloud: (B, T, N, C)
                # The image encoder expects the latter sequence-aware format.
                if k == "point_cloud":
                    if v.dim() == 3:
                        encoder_input[k] = v.unsqueeze(1)
                    else:
                        encoder_input[k] = v
                elif v.dim() in (2, 4):
                    encoder_input[k] = v.unsqueeze(1)
                else:
                    encoder_input[k] = v
            if (
                isinstance(future_obs, dict)
                and "prev_action" not in encoder_input
                and getattr(encoder, "prev_action_dim", 0) > 0
            ):
                agent_pos = encoder_input.get("agent_pos")
                if agent_pos is None:
                    raise RuntimeError(
                        "Future target encoder requires prev_action but future_obs "
                        "does not contain agent_pos to infer its batch/time shape"
                    )
                encoder_input["prev_action"] = torch.zeros(
                    (*agent_pos.shape[:-1], encoder.prev_action_dim),
                    dtype=agent_pos.dtype,
                    device=agent_pos.device,
                )

        future_embed_target = encoder(encoder_input, None)
        if reduce_sequence_output and future_embed_target.dim() == 3:
            future_embed_target = future_embed_target.mean(dim=1)
        if future_embed_target.dim() == 3 and future_embed_target.shape[1] == 1:
            future_embed_target = future_embed_target[:, 0, :]
        return future_embed_target

    def _copy_obs(self, obs):
        if isinstance(obs, dict):
            return {k: v.clone() for k, v in obs.items()}
        return obs.clone() if torch.is_tensor(obs) else obs

    def _future_target_scale(self, future_embed_target: torch.Tensor) -> torch.Tensor:
        scale = future_embed_target.detach().flatten(1).pow(2).mean(dim=1).sqrt()
        scale = scale.clamp_min(1e-6)
        return scale.view(-1, *([1] * (future_embed_target.dim() - 1)))

    def _normalized_future_loss(
        self,
        future_embed_pred: torch.Tensor,
        future_embed_target: torch.Tensor,
        interval: float,
    ) -> torch.Tensor:
        target_scale = self._future_target_scale(future_embed_target)
        normalized_error = (
            (future_embed_pred - future_embed_target)
            / target_scale
            / interval
        )
        return (normalized_error ** 2).mean()

    def _future_loss_weight_and_info(
        self, future_loss: torch.Tensor, action_term: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        future_loss_mode = getattr(
            self.config.optimization, "future_state_loss_mode", "fixed"
        )
        if future_loss_mode == "fixed":
            future_weight = torch.as_tensor(
                self.config.optimization.future_embed_loss_weight,
                device=future_loss.device,
                dtype=future_loss.dtype,
            )
        elif future_loss_mode == "ratio":
            future_weight = (
                self.config.optimization.future_state_loss_ratio
                * action_term.detach()
                / (future_loss.detach() + 1e-8)
            )
            future_weight = torch.clamp(
                future_weight,
                min=self.config.optimization.future_state_loss_weight_min,
                max=self.config.optimization.future_state_loss_weight_max,
            )
        else:
            raise ValueError(
                "future_state_loss_mode must be 'fixed' or 'ratio', "
                f"got {future_loss_mode!r}."
            )

        weighted_future_loss = future_weight * future_loss
        future_ratio = weighted_future_loss.detach() / (action_term.detach() + 1e-8)
        return future_weight, weighted_future_loss, {
            "loss_future": weighted_future_loss.detach(),
            "loss_future_weighted": weighted_future_loss.detach(),
            "loss_future_raw": future_loss.detach(),
            "loss_future_weight": future_weight.detach(),
            "loss_future_ratio": future_ratio,
        }

    def _compute_direct_future_embed_loss(
        self,
        act: torch.Tensor,
        obs: torch.Tensor | dict,
        delta_t: torch.Tensor,
        future_obs: torch.Tensor | dict,
        obs_emb: torch.Tensor | None = None,
    ):
        if obs_emb is None:
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

        future_loss = self._normalized_future_loss(
            future_embed_pred,
            future_embed_target,
            interval=1.0,
        )
        if not torch.isfinite(future_loss):
            self._log_future_nonfinite_once(
                future_embed_target,
                future_embed_pred,
                obs_emb,
                act,
                future_obs,
            )
        info = {
            "loss_future_raw": future_loss.detach(),
            "_obs_emb_used_for_future": obs_emb,
        }
        return future_loss, info

    def _compute_mip_two_step_future_embed_loss(
        self,
        act: torch.Tensor,
        obs: torch.Tensor | dict,
        delta_t: torch.Tensor,
        future_obs: torch.Tensor | dict,
        obs_emb: torch.Tensor | None = None,
    ):
        if obs_emb is None:
            obs_emb = self.encoder(obs, None)
        s = torch.zeros_like(delta_t, device=delta_t.device)
        t = (
            torch.zeros_like(delta_t, device=delta_t.device)
            + self.config.optimization.future_t_two_step
        )
        future_embed_target = self._encode_future_target(future_obs).detach()
        future_embed_0 = torch.zeros_like(future_embed_target)
        future_noise = torch.randn_like(future_embed_target)
        future_embed_t = (
            future_embed_target
            + (1 - self.config.optimization.future_t_two_step) * future_noise
        )
        act_0 = torch.zeros_like(act)
        act_noise = torch.randn_like(act)
        act_t = act + (1 - self.config.optimization.future_t_two_step) * act_noise

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

        future_loss_0 = self._normalized_future_loss(
            future_embed_pred_0,
            future_embed_target,
            interval=self.config.optimization.future_t_two_step,
        )
        future_loss_1 = self._normalized_future_loss(
            future_embed_pred_1,
            future_embed_target,
            interval=1 - self.config.optimization.future_t_two_step,
        )
        future_loss = future_loss_0 + future_loss_1
        weighted_future_loss = (
            self.config.optimization.future_embed_loss_weight * future_loss
        )
        info = {
            "loss_future": weighted_future_loss.detach(),
            "loss_future_raw": future_loss.detach(),
            "loss_future_direct_raw": future_loss_0.detach(),
            "loss_future_denoise_raw": future_loss_1.detach(),
            "_obs_emb_used_for_future": obs_emb,
        }
        return future_loss, info

    def _compute_joint_mip_two_step_loss(
        self,
        act: torch.Tensor,
        obs: torch.Tensor | dict,
        delta_t: torch.Tensor,
        future_obs: torch.Tensor | dict,
        obs_emb: torch.Tensor | None = None,
        action_noise: torch.Tensor | None = None,
        future_noise: torch.Tensor | None = None,
    ):
        if not getattr(self.config.optimization, "future_joint_mode", False):
            raise RuntimeError("Joint MIP loss requires future_joint_mode=true")
        if self.config.optimization.future_embed_loss_mode != "mip_two_step":
            raise RuntimeError(
                "future_joint_mode currently requires future_embed_loss_mode='mip_two_step'"
            )
        if (
            self.config.optimization.future_t_two_step
            != self.config.optimization.t_two_step
        ):
            raise ValueError(
                "A joint transformer pass has one shared time condition; "
                "future_t_two_step must equal t_two_step"
            )
        future_embed_target = None
        if (
            obs_emb is None
            and getattr(self.encoder, "temporal_consistent_crop", False)
            and hasattr(self.encoder, "encode_obs_and_future_rgb")
        ):
            # Crop current and future frames together so the RGB-only target
            # cannot independently resample a spatial window. This changes only
            # image augmentation; the MIP stage inputs below remain unchanged.
            obs_emb, future_embed_target, _ = self.encoder.encode_obs_and_future_rgb(
                obs, future_obs, mask=None
            )
        elif obs_emb is None:
            obs_emb = self.encoder(obs, None)
        s = torch.zeros_like(delta_t, device=delta_t.device)
        t = torch.zeros_like(delta_t, device=delta_t.device) + self.config.optimization.t_two_step
        if future_embed_target is None:
            future_embed_target = self._encode_future_target(future_obs)
        future_embed_target = future_embed_target.detach()
        future_embed_0 = torch.zeros_like(future_embed_target)
        if (action_noise is None) != (future_noise is None):
            raise ValueError(
                "action_noise and future_noise must be supplied together"
            )
        if future_noise is None:
            future_noise = torch.randn_like(future_embed_target)
        else:
            future_noise = future_noise.to(
                device=future_embed_target.device,
                dtype=future_embed_target.dtype,
            )
            if future_noise.shape != future_embed_target.shape:
                raise ValueError(
                    "future_noise shape mismatch: "
                    f"{tuple(future_noise.shape)} != "
                    f"{tuple(future_embed_target.shape)}"
                )
        future_embed_t = future_embed_target + (1 - self.config.optimization.t_two_step) * future_noise
        act_0 = torch.zeros_like(act)
        if action_noise is None:
            action_noise = torch.randn_like(act)
        else:
            action_noise = action_noise.to(device=act.device, dtype=act.dtype)
            if action_noise.shape != act.shape:
                raise ValueError(
                    "action_noise shape mismatch: "
                    f"{tuple(action_noise.shape)} != {tuple(act.shape)}"
                )
        act_t = act + (1 - self.config.optimization.t_two_step) * action_noise

        head_only_stopgrad = bool(
            getattr(
                self.config.optimization,
                "future_head_only_stopgrad",
                False,
            )
        )
        joint_forward_kwargs = (
            {"stopgrad_future_trunk": True} if head_only_stopgrad else {}
        )
        act_pred_0, future_embed_pred_0 = self.flow_map.net.joint_forward(
            act_0,
            s,
            t,
            obs_emb,
            future_embed_0,
            **joint_forward_kwargs,
        )
        act_pred_1, future_embed_pred_1 = self.flow_map.net.joint_forward(
            act_t,
            t,
            torch.ones_like(t),
            obs_emb,
            future_embed_t,
            **joint_forward_kwargs,
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

        action_loss_0 = ((act_pred_0 - act) ** 2).mean(dim=-1) / self.config.optimization.t_two_step ** 2
        action_loss_1 = ((act_pred_1 - act) ** 2).mean(dim=-1) / (1 - self.config.optimization.t_two_step) ** 2
        action_loss_first = torch.mean(action_loss_0)
        action_loss_second = torch.mean(action_loss_1)
        action_loss = action_loss_first + action_loss_second

        future_loss_0 = self._normalized_future_loss(
            future_embed_pred_0,
            future_embed_target,
            interval=self.config.optimization.t_two_step,
        )
        future_loss_1 = self._normalized_future_loss(
            future_embed_pred_1,
            future_embed_target,
            interval=1 - self.config.optimization.t_two_step,
        )
        future_loss = future_loss_0 + future_loss_1
        action_term = self.config.optimization.loss_scale * action_loss
        future_weight, weighted_future_loss, future_info = (
            self._future_loss_weight_and_info(future_loss, action_term)
        )
        loss = action_term + weighted_future_loss
        info = {
            "loss_action": action_loss.detach(),
            "loss_action_raw": action_loss.detach(),
            "loss_action_first_raw": action_loss_first.detach(),
            "loss_action_second_raw": action_loss_second.detach(),
            "loss_action_total": action_loss.detach(),
            "loss_future_first_raw": future_loss_0.detach(),
            "loss_future_second_raw": future_loss_1.detach(),
            "loss_future_total": future_loss.detach(),
            "weighted_future_loss": weighted_future_loss.detach(),
            "loss_total": loss.detach(),
            "future_head_only_stopgrad": torch.as_tensor(
                float(head_only_stopgrad), device=loss.device
            ),
            "_joint_action_term": action_term,
            "_joint_weighted_future_loss": weighted_future_loss,
            "_joint_future_input_0": future_embed_0,
            "_joint_future_pred_0": future_embed_pred_0,
            "_joint_future_pred_1": future_embed_pred_1,
            "_joint_future_target": future_embed_target,
            "_joint_obs_emb": obs_emb,
        }
        info.update(future_info)
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

    def _parameter_group_gradient_diagnostics(
        self,
        action_term: torch.Tensor,
        weighted_future_loss: torch.Tensor,
        parameter_groups: dict[str, list[torch.nn.Parameter]],
    ) -> dict[str, torch.Tensor]:
        """Measure gradient alignment for disjoint groups without writing ``.grad``."""
        parameters = []
        group_indices = {}
        parameter_indices = {}
        for group_name, group_parameters in parameter_groups.items():
            indices = []
            for parameter in group_parameters:
                if not parameter.requires_grad:
                    continue
                parameter_id = id(parameter)
                if parameter_id not in parameter_indices:
                    parameter_indices[parameter_id] = len(parameters)
                    parameters.append(parameter)
                indices.append(parameter_indices[parameter_id])
            group_indices[group_name] = indices

        if not parameters:
            return {}

        action_grads = torch.autograd.grad(
            action_term,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        future_grads = torch.autograd.grad(
            weighted_future_loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )

        device = action_term.device
        diagnostics = {}
        for group_name, indices in group_indices.items():
            if not indices:
                continue
            action_sq = torch.zeros((), device=device)
            future_sq = torch.zeros((), device=device)
            dot = torch.zeros((), device=device)
            for index in indices:
                action_grad = action_grads[index]
                future_grad = future_grads[index]
                if action_grad is not None:
                    action_sq = action_sq + action_grad.float().square().sum()
                if future_grad is not None:
                    future_sq = future_sq + future_grad.float().square().sum()
                if action_grad is not None and future_grad is not None:
                    dot = dot + (action_grad.float() * future_grad.float()).sum()

            action_norm = action_sq.sqrt()
            future_norm = future_sq.sqrt()
            norm_product = action_norm * future_norm
            cosine = torch.where(
                norm_product > 0,
                dot / norm_product.clamp_min(1e-12),
                torch.zeros_like(dot),
            )
            norm_ratio = future_norm / action_norm.clamp_min(1e-12)
            diagnostics.update(
                {
                    f"grad_{group_name}_action_norm": action_norm.detach(),
                    f"grad_{group_name}_future_norm": future_norm.detach(),
                    f"grad_{group_name}_cosine": cosine.detach(),
                    f"grad_{group_name}_norm_ratio": norm_ratio.detach(),
                }
            )
        return diagnostics

    def _joint_gradient_diagnostics(
        self,
        action_term: torch.Tensor,
        weighted_future_loss: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Measure joint-task gradients without changing the training backward pass."""
        net = self.flow_map.net
        diagnostics = {}

        # Keep the two large groups separate so temporary diagnostic gradients do
        # not increase peak memory by materializing encoder and decoder gradients
        # at the same time.
        if hasattr(net, "gradient_diagnostic_parameter_groups"):
            trunk_groups = net.gradient_diagnostic_parameter_groups()
        else:
            # Preserve the legacy JEPA-Policy diagnostic and metric names.
            trunk_groups = {
                "shared": [
                    parameter
                    for parameter in net.decoder.parameters()
                    if parameter.requires_grad
                ]
            }
        diagnostics.update(
            self._parameter_group_gradient_diagnostics(
                action_term,
                weighted_future_loss,
                trunk_groups,
            )
        )
        diagnostics.update(
            self._parameter_group_gradient_diagnostics(
                action_term,
                weighted_future_loss,
                {
                    "encoder": [
                        parameter
                        for parameter in self.encoder.parameters()
                        if parameter.requires_grad
                    ]
                },
            )
        )

        future_input_emb = getattr(net, "future_input_emb", None)
        future_type_emb = getattr(net, "future_type_emb", None)
        action_head = getattr(net, "head", None)
        if action_head is None:
            action_head = getattr(net, "action_head", None)
        future_head = getattr(net, "future_head", None)
        small_groups = {
            "future_input": (
                list(future_input_emb.parameters())
                if future_input_emb is not None
                else []
            ),
            "future_type": (
                [future_type_emb]
                if future_type_emb is not None and future_type_emb.requires_grad
                else []
            ),
            "action_head": (
                list(action_head.parameters()) if action_head is not None else []
            ),
            "future_head": (
                list(future_head.parameters()) if future_head is not None else []
            ),
        }
        small_group_diagnostics = self._parameter_group_gradient_diagnostics(
            action_term,
            weighted_future_loss,
            small_groups,
        )
        # The action and future heads are deliberately loss-exclusive. Their
        # cross-loss norms verify that separation; cosine and future/action ratio
        # are undefined or uninformative when one side is exactly zero.
        for group_name in ("action_head", "future_head"):
            small_group_diagnostics.pop(f"grad_{group_name}_cosine", None)
            small_group_diagnostics.pop(f"grad_{group_name}_norm_ratio", None)
        diagnostics.update(small_group_diagnostics)
        return diagnostics

    @staticmethod
    def _future_prediction_diagnostics(
        future_pred_first: torch.Tensor,
        future_pred_second: torch.Tensor,
        future_target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return detached scale and alignment metrics for both future passes."""

        def flatten_float(tensor):
            return tensor.detach().float().flatten(1)

        target = flatten_float(future_target)
        pred_first = flatten_float(future_pred_first)
        pred_second = flatten_float(future_pred_second)

        def mean_rms(tensor):
            return tensor.square().mean(dim=1).sqrt().mean()

        def mean_cosine(prediction):
            numerator = (prediction * target).sum(dim=1)
            denominator = (
                prediction.square().sum(dim=1).sqrt()
                * target.square().sum(dim=1).sqrt()
            )
            cosine = torch.where(
                denominator > 0,
                numerator / denominator.clamp_min(1e-12),
                torch.zeros_like(numerator),
            )
            return cosine.mean()

        return {
            "future_target_rms": mean_rms(target),
            "future_pred_first_rms": mean_rms(pred_first),
            "future_pred_second_rms": mean_rms(pred_second),
            "future_pred_target_cosine_first": mean_cosine(pred_first),
            "future_pred_target_cosine_second": mean_cosine(pred_second),
        }

    @torch.no_grad()
    def validation_metrics(
        self,
        act: torch.Tensor,
        obs: torch.Tensor | dict,
        delta_t: torch.Tensor,
        future_obs: torch.Tensor | dict,
        seed: int,
        action_noise: torch.Tensor | None = None,
        future_noise: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Evaluate a fixed diagnostic batch with isolated Torch RNG state."""
        if not getattr(self.config.optimization, "use_future_embed_loss", False):
            return {}

        cuda_devices = []
        if act.is_cuda:
            cuda_devices = [act.device.index or torch.cuda.current_device()]

        encoder_was_training = self.encoder.training
        flow_map_was_training = self.flow_map.training
        try:
            self.encoder.eval()
            self.flow_map.eval()
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(seed)
                if act.is_cuda:
                    torch.cuda.manual_seed_all(seed)

                if getattr(
                    self.config.optimization, "future_joint_mode", False
                ):
                    total_loss, combined_info = (
                        self._compute_joint_mip_two_step_loss(
                            act,
                            obs,
                            delta_t,
                            future_obs,
                            action_noise=action_noise,
                            future_noise=future_noise,
                        )
                    )
                    combined_info = {
                        key: value
                        for key, value in combined_info.items()
                        if not key.startswith("_")
                    }
                    combined_info["loss_total"] = total_loss.detach()
                    return {
                        key: float(value.detach().float().item())
                        for key, value in combined_info.items()
                        if torch.is_tensor(value)
                    }

                action_term, info = self.loss_fn(
                    self.config.optimization,
                    self.flow_map,
                    self.encoder,
                    self.interpolant,
                    act,
                    obs,
                    delta_t,
                    return_obs_emb=True,
                )
                cached_obs_emb = info.pop("_obs_emb_for_reuse", None)
                future_mode = getattr(
                    self.config.optimization, "future_embed_loss_mode", "direct"
                )
                if future_mode == "direct":
                    future_loss, detail_info = self._compute_direct_future_embed_loss(
                        act,
                        obs,
                        delta_t,
                        future_obs,
                        obs_emb=cached_obs_emb,
                    )
                elif future_mode == "mip_two_step":
                    future_loss, detail_info = (
                        self._compute_mip_two_step_future_embed_loss(
                            act,
                            obs,
                            delta_t,
                            future_obs,
                            obs_emb=cached_obs_emb,
                        )
                    )
                else:
                    raise ValueError(
                        f"Unsupported future_embed_loss_mode: {future_mode}"
                    )
                detail_info.pop("_obs_emb_used_for_future", None)
                _, weighted_future_loss, weight_info = (
                    self._future_loss_weight_and_info(future_loss, action_term)
                )
                total_loss = action_term + weighted_future_loss

                combined_info = dict(info)
                combined_info.update(detail_info)
                combined_info.update(weight_info)
                combined_info["loss_total"] = total_loss.detach()
                return {
                    key: float(value.detach().float().item())
                    for key, value in combined_info.items()
                    if not key.startswith("_") and torch.is_tensor(value)
                }
        finally:
            self.encoder.train(encoder_was_training)
            self.flow_map.train(flow_map_was_training)

    def update(
        self,
        act: torch.Tensor,
        obs: torch.Tensor,
        delta_t: torch.Tensor,
        future_obs: torch.Tensor | dict | None = None,
        sync_metrics: bool = True,
        compute_gradient_diagnostics: bool = False,
    ):
        """Update the model parameters with a training batch."""
        self.optimizer.zero_grad(set_to_none=True)
        obs_for_future = self._copy_obs(obs)
        cached_obs_emb = None

        joint_mode = bool(
            getattr(self.config.optimization, "future_joint_mode", False)
        )
        joint_future_predictions = None
        if joint_mode:
            if future_obs is None:
                raise ValueError("future_joint_mode requires future_obs")
            if not getattr(self.config.optimization, "use_future_embed_loss", False):
                raise ValueError(
                    "future_joint_mode requires use_future_embed_loss=true"
                )
            if not getattr(self.config.task, "future_state_enabled", False):
                raise ValueError("future_joint_mode requires future_state_enabled=true")

            loss, info = self._compute_joint_mip_two_step_loss(
                act,
                obs_for_future,
                delta_t,
                future_obs,
            )
            action_term_for_diagnostics = info.pop("_joint_action_term")
            weighted_future_loss_for_diagnostics = info.pop(
                "_joint_weighted_future_loss"
            )
            # These graph tensors are exposed by the joint loss for unit-level
            # dependency diagnostics, but are not training log metrics.
            info.pop("_joint_future_input_0", None)
            if compute_gradient_diagnostics:
                joint_future_predictions = (
                    info.pop("_joint_future_pred_0"),
                    info.pop("_joint_future_pred_1"),
                    info.pop("_joint_future_target"),
                )
            else:
                info.pop("_joint_future_pred_0", None)
                info.pop("_joint_future_pred_1", None)
                info.pop("_joint_future_target", None)
            cached_obs_emb = info.pop("_joint_obs_emb", None)

        use_direct_future_embed_loss = (
            not joint_mode
            and
            future_obs is not None
            and getattr(self.config.optimization, "use_future_embed_loss", False)
            and getattr(self.config.task, "future_state_enabled", False)
        )

        future_mode = getattr(
            self.config.optimization, "future_embed_loss_mode", "direct"
        )
        reuse_obs_emb = (
            use_direct_future_embed_loss
            and self.config.optimization.loss_type == "mip"
        )
        loss_kwargs = {"return_obs_emb": True} if reuse_obs_emb else {}
        if joint_mode:
            pass
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
                **loss_kwargs,
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
                **loss_kwargs,
            )

        if not joint_mode:
            cached_obs_emb = info.pop("_obs_emb_for_reuse", None)
        if not joint_mode:
            action_term_for_diagnostics = loss
            weighted_future_loss_for_diagnostics = None

        if use_direct_future_embed_loss:
            if future_mode == "direct":
                future_loss, future_detail_info = self._compute_direct_future_embed_loss(
                    act,
                    obs_for_future,
                    delta_t,
                    future_obs,
                    obs_emb=cached_obs_emb,
                )
            elif future_mode == "mip_two_step":
                future_loss, future_detail_info = (
                    self._compute_mip_two_step_future_embed_loss(
                        act,
                        obs_for_future,
                        delta_t,
                        future_obs,
                        obs_emb=cached_obs_emb,
                    )
                )
            else:
                raise ValueError(
                    f"Unsupported future_embed_loss_mode: {future_mode}"
                )
            future_obs_emb = future_detail_info.pop("_obs_emb_used_for_future", None)
            if cached_obs_emb is None or future_obs_emb is not cached_obs_emb:
                raise RuntimeError(
                    "Action and future losses must share the exact same "
                    "encoder(obs) tensor within each training step"
                )
            future_weight, weighted_future_loss, future_weight_info = (
                self._future_loss_weight_and_info(future_loss, loss)
            )
            weighted_future_loss_for_diagnostics = weighted_future_loss
            loss = loss + weighted_future_loss
            info.update(future_detail_info)
            info.update(future_weight_info)

        if not torch.isfinite(loss):
            raise RuntimeError("Training loss became non-finite during training")

        info["loss_total"] = loss.detach()

        if (
            compute_gradient_diagnostics
            and weighted_future_loss_for_diagnostics is not None
        ):
            info.update(
                self._joint_gradient_diagnostics(
                    action_term_for_diagnostics,
                    weighted_future_loss_for_diagnostics,
                )
            )
            if joint_future_predictions is not None:
                info.update(
                    self._future_prediction_diagnostics(
                        *joint_future_predictions,
                    )
                )

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

        def metric_value(value):
            if isinstance(value, torch.Tensor):
                value = value.detach()
                return value.item() if sync_metrics else value
            return value

        out = {
            "loss": metric_value(loss),
            "grad_norm": metric_value(grad_norm),
        }
        for key, value in info.items():
            out[key] = metric_value(value)
        return out


    def ema_update(self):
        """Update EMA with foreach when supported, otherwise use the safe loop."""
        params = list(self.encoder.parameters()) + list(self.flow_map.parameters())
        ema_params = list(self.encoder_ema.parameters()) + list(
            self.flow_map_ema.parameters()
        )
        rate = self.config.optimization.ema_rate
        with torch.no_grad():
            if self._foreach_ema_enabled:
                try:
                    torch._foreach_lerp_(ema_params, params, 1.0 - rate)
                    return
                except (RuntimeError, TypeError, NotImplementedError) as exc:
                    self._foreach_ema_enabled = False
                    if not self._foreach_ema_warned:
                        loguru.logger.warning(
                            f"foreach EMA unavailable at runtime ({exc}); "
                            "falling back to parameter loop"
                        )
                        self._foreach_ema_warned = True
            for p, p_ema in zip(params, ema_params, strict=False):
                p_ema.mul_(rate).add_(p, alpha=1.0 - rate)

    def sample(
        self,
        act_0: torch.Tensor,
        obs: torch.Tensor,
        num_steps: int = -1,
        use_ema: bool = True,
        return_future: bool = False,
    ):
        """Sample actions from the learned policy.

        Args:
            act_0: Initial action tensor of shape (batch_size, Ta, act_dim)
            obs: Observation tensor of shape (batch_size, To, obs_dim)
            num_steps: Number of sampling steps (default: use config value)
            use_ema: Whether to use EMA parameters for sampling
            return_future: Return ``(action_pred_1, future_pred_1)`` for the
                joint MIP rollout path. The environment should still execute
                only ``action_pred_1``.

        Returns:
            Sampled action tensor, or ``(action_pred_1, future_pred_1)`` when
            ``return_future=True``.
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
            if return_future:
                if config.loss_type != "mip":
                    raise ValueError(
                        "return_future=True is only supported by the MIP sampler"
                    )
                return self.sampler(
                    config,
                    flow_map,
                    encoder,
                    act_0,
                    obs,
                    return_future=True,
                )
            return self._compiled_sampler(config, flow_map, encoder, act_0, obs)

    def sample_joint(
        self,
        act_0: torch.Tensor,
        obs: torch.Tensor | dict[str, torch.Tensor],
        *,
        sampler_invocation_id: str,
        mode: JointSamplerMode,
        use_ema: bool = True,
    ) -> JointSamplerResult:
        """Run one traceable Future4 sampler invocation.

        This method is deliberately separate from ``sample`` so existing
        training and evaluation callers retain their return types.  Audit mode
        returns an owning CPU snapshot of pass-1 future prediction; selection
        modes never expose or serialize a future output.
        """

        config = self.config.optimization
        if config.loss_type != "mip":
            raise ValueError("sample_joint is only supported by the MIP sampler")
        if self.config.optimization.ema_rate < 1 and use_ema:
            flow_map = self.flow_map_ema
            encoder = self.encoder_ema
        else:
            flow_map = self.flow_map
            encoder = self.encoder
        with torch.inference_mode():
            return joint_mip_sampler(
                config,
                flow_map,
                encoder,
                act_0,
                obs,
                sampler_invocation_id=sampler_invocation_id,
                mode=mode,
            )

    def save(
        self,
        path: str,
        training_state: dict = None,
        *,
        include_optimizer: bool = True,
    ):
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
        }
        if include_optimizer:
            checkpoint["optimizer"] = self.optimizer.state_dict()

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
        # Pre-joint checkpoints stored one fixed future-content parameter under
        # ``future_tokens``. It is only a token-type identifier in the joint
        # architecture, so migrate the value explicitly for load compatibility.
        for container_key in ("flow_map", "flow_map_ema"):
            container = state_dict.get(container_key, {})
            old_key = "net.future_tokens"
            new_key = "net.future_type_emb"
            if old_key in container and new_key not in container:
                container[new_key] = container.pop(old_key)
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
