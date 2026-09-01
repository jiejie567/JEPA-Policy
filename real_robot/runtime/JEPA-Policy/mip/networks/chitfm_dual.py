"""Isolated dual-branch ChiTransformer variants.

This module intentionally does not subclass or modify the legacy
``ChiTransformer``.  The two architectures here share only the public network
protocol used by the training agent.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mip.embeddings import SUPPORTED_TIMESTEP_EMBEDDING
from mip.networks.base import BaseNetwork


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.MultiheadAttention):
        for name in ("in_proj_weight", "q_proj_weight", "k_proj_weight", "v_proj_weight"):
            weight = getattr(module, name, None)
            if weight is not None:
                nn.init.normal_(weight, mean=0.0, std=0.02)
        for name in ("in_proj_bias", "bias_k", "bias_v"):
            bias = getattr(module, name, None)
            if bias is not None:
                nn.init.zeros_(bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class _DecoderBranchLayer(nn.Module):
    """Pre-norm self-attention, fixed-memory attention, and FFN block."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        ffn_dim: int,
        dropout: float,
        *,
        memory_dim: int,
    ) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.self_dropout = nn.Dropout(dropout)

        self.memory_norm = nn.LayerNorm(d_model)
        self.memory_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
            kdim=memory_dim,
            vdim=memory_dim,
        )
        self.memory_dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_out = nn.Linear(ffn_dim, d_model)
        self.ffn_residual_dropout = nn.Dropout(dropout)

    def self_step(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.self_norm(x)
        update, _ = self.self_attn(normed, normed, normed, need_weights=False)
        return x + self.self_dropout(update)

    def memory_ffn_step(
        self, x: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        query = self.memory_norm(x)
        update, _ = self.memory_attn(query, memory, memory, need_weights=False)
        x = x + self.memory_dropout(update)
        ffn_input = self.ffn_norm(x)
        update = self.ffn_out(self.ffn_dropout(F.gelu(self.ffn_in(ffn_input))))
        return x + self.ffn_residual_dropout(update)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        return self.memory_ffn_step(self.self_step(x), memory)


class _BidirectionalCrossLayer(nn.Module):
    """One synchronous, bidirectional cross-attention stage."""

    def __init__(
        self,
        action_dim: int,
        future_dim: int,
        action_heads: int,
        future_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.action_query_norm = nn.LayerNorm(action_dim)
        self.future_for_action_norm = nn.LayerNorm(future_dim)
        self.action_reads_future = nn.MultiheadAttention(
            action_dim,
            action_heads,
            dropout=dropout,
            batch_first=True,
            kdim=future_dim,
            vdim=future_dim,
        )
        self.action_dropout = nn.Dropout(dropout)

        self.future_query_norm = nn.LayerNorm(future_dim)
        self.action_for_future_norm = nn.LayerNorm(action_dim)
        self.future_reads_action = nn.MultiheadAttention(
            future_dim,
            future_heads,
            dropout=dropout,
            batch_first=True,
            kdim=action_dim,
            vdim=action_dim,
        )
        self.future_dropout = nn.Dropout(dropout)

    def forward(
        self, action: torch.Tensor, future: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Both directions consume the same pre-cross snapshots.  Do not update
        # either stream until both attention results have been computed.
        action_snapshot = action
        future_snapshot = future
        future_context = self.future_for_action_norm(future_snapshot)
        action_context = self.action_for_future_norm(action_snapshot)
        action_update, _ = self.action_reads_future(
            self.action_query_norm(action_snapshot),
            future_context,
            future_context,
            need_weights=False,
        )
        future_update, _ = self.future_reads_action(
            self.future_query_norm(future_snapshot),
            action_context,
            action_context,
            need_weights=False,
        )
        return (
            action_snapshot + self.action_dropout(action_update),
            future_snapshot + self.future_dropout(future_update),
        )


class _DualChiTransformerBase(BaseNetwork):
    architecture_name = "dual_base"

    def __init__(
        self,
        act_dim: int,
        obs_dim: int,
        Ta: int,
        To: int,
        d_model: int = 384,
        nhead: int = 6,
        num_layers: int = 8,
        p_drop_emb: float = 0.0,
        p_drop_attn: float = 0.1,
        n_cond_layers: int = 0,
        timestep_emb_type: str = "positional",
        timestep_emb_params: dict | None = None,
        n_future_tokens: int = 1,
        future_out_dim: int | None = None,
        future_d_model: int = 192,
        future_nhead: int = 6,
        future_num_layers: int = 8,
        future_ffn_dim: int = 768,
        use_causal_mask: bool = False,
        use_memory_mask: bool = False,
    ) -> None:
        super().__init__(act_dim, Ta, obs_dim, To, d_model, num_layers)
        if n_future_tokens <= 0:
            raise ValueError("Dual transformers require n_future_tokens > 0")
        if num_layers != future_num_layers:
            raise ValueError(
                "Action and future branches must have equal depth for controlled comparisons: "
                f"{num_layers} != {future_num_layers}"
            )
        if d_model % nhead != 0 or future_d_model % future_nhead != 0:
            raise ValueError("Each branch hidden dimension must be divisible by its head count")
        if use_causal_mask or use_memory_mask:
            raise ValueError(
                "Dual v1 supports only the current JEPA-Policy unmasked configuration"
            )

        self.Ta = Ta
        self.To = To
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.d_model = d_model
        self.n_future_tokens = n_future_tokens
        self.future_out_dim = obs_dim if future_out_dim is None else future_out_dim
        self.future_d_model = future_d_model
        self.num_layers = num_layers
        self.future_num_layers = future_num_layers
        self.action_nhead = nhead
        self.future_nhead = future_nhead
        self.attention_dropout = p_drop_attn

        # Completely separate action/future token stems and positional state.
        self.action_input_emb = nn.Linear(act_dim, d_model)
        self.action_pos_emb = nn.Parameter(torch.zeros(1, Ta, d_model))
        self.future_input_emb = nn.Linear(self.future_out_dim, future_d_model)
        self.future_type_emb = nn.Parameter(
            torch.zeros(1, n_future_tokens, future_d_model)
        )
        self.future_pos_emb = nn.Parameter(
            torch.zeros(1, n_future_tokens, future_d_model)
        )
        self.input_dropout = nn.Dropout(p_drop_emb)

        # One shared condition encoder, evaluated once per model forward.
        self.cond_obs_emb = nn.Linear(obs_dim, d_model)
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, 1 + To, d_model))
        if n_cond_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=4 * d_model,
                dropout=p_drop_attn,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.condition_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=n_cond_layers
            )
        else:
            self.condition_encoder = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.Mish(),
                nn.Linear(4 * d_model, d_model),
            )
        self.future_memory_adapter = nn.Linear(d_model, future_d_model)

        timestep_emb_params = timestep_emb_params or {}
        self.map_s = SUPPORTED_TIMESTEP_EMBEDDING[timestep_emb_type](
            d_model // 2, **timestep_emb_params
        )
        self.map_t = SUPPORTED_TIMESTEP_EMBEDDING[timestep_emb_type](
            d_model // 2, **timestep_emb_params
        )

        self.action_layers = nn.ModuleList(
            _DecoderBranchLayer(
                d_model,
                nhead,
                4 * d_model,
                p_drop_attn,
                memory_dim=d_model,
            )
            for _ in range(num_layers)
        )
        self.future_layers = nn.ModuleList(
            _DecoderBranchLayer(
                future_d_model,
                future_nhead,
                future_ffn_dim,
                p_drop_attn,
                memory_dim=future_d_model,
            )
            for _ in range(future_num_layers)
        )

        self.action_final_norm = nn.LayerNorm(d_model)
        self.future_final_norm = nn.LayerNorm(future_d_model)
        self.action_head = nn.Linear(d_model, act_dim)
        self.future_head = nn.Linear(future_d_model, self.future_out_dim)

        self.input_processor = nn.Linear(act_dim, d_model // 4)
        self.final_processor = nn.Linear(d_model, d_model // 4)
        self.scalar_head = nn.Linear(d_model // 4 + d_model // 4 + d_model, 1)

        self.apply(_init_weights)
        nn.init.normal_(self.action_pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.future_pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.future_type_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.cond_pos_emb, mean=0.0, std=0.02)
        nn.init.zeros_(self.scalar_head.weight)
        nn.init.zeros_(self.scalar_head.bias)

    def _validate_future_input(
        self, future_input: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        if future_input.dim() == 2:
            future_input = future_input.unsqueeze(1)
        expected = (batch_size, self.n_future_tokens, self.future_out_dim)
        if tuple(future_input.shape) != expected:
            raise ValueError(
                f"future_input must have shape {expected}, got {tuple(future_input.shape)}"
            )
        return future_input

    def _encode_condition(
        self,
        condition: torch.Tensor | None,
        s: torch.Tensor,
        t: torch.Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if condition is None:
            condition = torch.zeros(
                batch_size, self.To, self.obs_dim, device=device, dtype=dtype
            )
        if not torch.is_tensor(s):
            s = torch.tensor(s, device=device, dtype=dtype)
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=device, dtype=dtype)
        s = s.to(device=device).reshape(-1)
        t = t.to(device=device).reshape(-1)
        if s.numel() == 1:
            s = s.expand(batch_size)
        if t.numel() == 1:
            t = t.expand(batch_size)
        if s.numel() != batch_size or t.numel() != batch_size:
            raise ValueError("s and t must be scalar or have one value per batch item")

        time_emb = torch.cat([self.map_s(s), self.map_t(t)], dim=-1).unsqueeze(1)
        cond_tokens = torch.cat([time_emb, self.cond_obs_emb(condition)], dim=1)
        cond_tokens = self.input_dropout(
            cond_tokens + self.cond_pos_emb[:, : cond_tokens.shape[1]]
        )
        action_memory = self.condition_encoder(cond_tokens)
        # Computed once, then reused unchanged by every future layer.
        future_memory = self.future_memory_adapter(action_memory)
        return action_memory, future_memory

    def _prepare_action(self, action_input: torch.Tensor) -> torch.Tensor:
        return self.input_dropout(
            self.action_input_emb(action_input) + self.action_pos_emb
        )

    def _prepare_future(self, future_input: torch.Tensor) -> torch.Tensor:
        return self.input_dropout(
            self.future_input_emb(future_input)
            + self.future_type_emb
            + self.future_pos_emb
        )

    def _run_branches(
        self,
        action: torch.Tensor,
        future: torch.Tensor,
        action_memory: torch.Tensor,
        future_memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _scalar_prediction(
        self,
        action_input: torch.Tensor,
        action_hidden: torch.Tensor,
        action_memory: torch.Tensor,
    ) -> torch.Tensor:
        pieces = (
            self.input_processor(action_input.mean(dim=1)),
            self.final_processor(action_hidden.mean(dim=1)),
            action_memory.mean(dim=1),
        )
        return self.scalar_head(torch.cat(pieces, dim=-1))

    def _forward_all(
        self,
        action_input: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor | None,
        future_input: torch.Tensor,
        *,
        project_future: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        batch_size = action_input.shape[0]
        future_input = self._validate_future_input(future_input, batch_size)
        action_memory, future_memory = self._encode_condition(
            condition,
            s,
            t,
            batch_size,
            action_input.device,
            action_input.dtype,
        )
        action_hidden, future_hidden = self._run_branches(
            self._prepare_action(action_input),
            self._prepare_future(future_input),
            action_memory,
            future_memory,
        )
        action_pred = self.action_head(self.action_final_norm(action_hidden))
        scalar_pred = self._scalar_prediction(
            action_input, action_hidden, action_memory
        )
        future_pred = None
        if project_future:
            future_pred = self.future_head(self.future_final_norm(future_hidden))
            if self.n_future_tokens == 1:
                future_pred = future_pred[:, 0]
        return action_pred, scalar_pred, future_pred

    def forward(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor | None = None,
        future_input: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if future_input is None:
            future_input = torch.zeros(
                x.shape[0],
                self.n_future_tokens,
                self.future_out_dim,
                device=x.device,
                dtype=x.dtype,
            )
        action, scalar, future = self._forward_all(
            x, s, t, condition, future_input, project_future=True
        )
        assert future is not None
        return action, scalar, future

    def joint_forward(
        self,
        action_input: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        future_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if future_input is None:
            raise ValueError("joint_forward requires explicit future_input")
        action, _, future = self._forward_all(
            action_input,
            s,
            t,
            condition,
            future_input,
            project_future=True,
        )
        assert future is not None
        return action, future

    def forward_with_aux(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def gradient_diagnostic_parameter_groups(
        self,
    ) -> dict[str, list[nn.Parameter]]:
        """Expose dual-only groups without emulating the legacy shared decoder."""
        condition_modules = (
            self.cond_obs_emb,
            self.condition_encoder,
            self.future_memory_adapter,
            self.map_s,
            self.map_t,
        )
        condition_parameters = [
            parameter
            for module in condition_modules
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        condition_parameters.append(self.cond_pos_emb)
        groups = {
            "condition": condition_parameters,
            "action_trunk": [
                parameter
                for parameter in self.action_layers.parameters()
                if parameter.requires_grad
            ],
            "future_trunk": [
                parameter
                for parameter in self.future_layers.parameters()
                if parameter.requires_grad
            ],
        }
        cross_layers = getattr(self, "cross_layers", None)
        if cross_layers is not None:
            groups["cross_trunk"] = [
                parameter
                for parameter in cross_layers.parameters()
                if parameter.requires_grad
            ]
        return groups


class ChiDualIndependentTransformer(_DualChiTransformerBase):
    """Two parameter-independent branches sharing only fixed observation memory."""

    architecture_name = "chitransformer_dual_independent_v1"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.register_buffer(
            "_dual_independent_architecture_v1", torch.tensor(1), persistent=True
        )

    def _run_branches(
        self,
        action: torch.Tensor,
        future: torch.Tensor,
        action_memory: torch.Tensor,
        future_memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.action_layers:
            action = layer(action, action_memory)
        for layer in self.future_layers:
            future = layer(future, future_memory)
        return action, future

    def joint_action_forward(
        self,
        action_input: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        future_input: torch.Tensor,
    ) -> torch.Tensor:
        # Validate the public contract, but deliberately do not embed or execute
        # the independent future branch.
        self._validate_future_input(future_input, action_input.shape[0])
        action_memory, _ = self._encode_condition(
            condition,
            s,
            t,
            action_input.shape[0],
            action_input.device,
            action_input.dtype,
        )
        action = self._prepare_action(action_input)
        for layer in self.action_layers:
            action = layer(action, action_memory)
        return self.action_head(self.action_final_norm(action))


class ChiDualCrossTransformer(_DualChiTransformerBase):
    """Two branches with synchronous bidirectional cross-attention at every layer."""

    architecture_name = "chitransformer_dual_cross_v1"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cross_layers = nn.ModuleList(
            _BidirectionalCrossLayer(
                self.d_model,
                self.future_d_model,
                self.action_nhead,
                self.future_nhead,
                self.attention_dropout,
            )
            for _ in range(self.num_layers)
        )
        self.cross_layers.apply(_init_weights)
        self.register_buffer(
            "_dual_cross_architecture_v1", torch.tensor(1), persistent=True
        )

    def _run_branches(
        self,
        action: torch.Tensor,
        future: torch.Tensor,
        action_memory: torch.Tensor,
        future_memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for action_layer, future_layer, cross_layer in zip(
            self.action_layers, self.future_layers, self.cross_layers
        ):
            action = action_layer.self_step(action)
            future = future_layer.self_step(future)
            action, future = cross_layer(action, future)
            action = action_layer.memory_ffn_step(action, action_memory)
            future = future_layer.memory_ffn_step(future, future_memory)
        return action, future

    def joint_action_forward(
        self,
        action_input: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        future_input: torch.Tensor,
    ) -> torch.Tensor:
        action, _, _ = self._forward_all(
            action_input,
            s,
            t,
            condition,
            future_input,
            project_future=False,
        )
        return action
