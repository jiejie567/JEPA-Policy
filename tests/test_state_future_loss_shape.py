from types import SimpleNamespace

import torch
import torch.nn as nn

from jepa_policy.flow_map import FlowMap
from jepa_policy.losses import mip_loss
from jepa_policy.networks.chitfm import ChiTransformer


def _optimization_config(**overrides):
    config = SimpleNamespace(
        t_two_step=0.5,
        loss_scale=1.0,
        norm_type="l2",
        future_target_type="state",
        future_state_loss_weight=0.25,
        future_state_loss_mode="fixed",
        future_state_loss_ratio=0.1,
        future_state_loss_weight_min=0.01,
        future_state_loss_weight_max=1.0,
        future_embed_loss_weight=0.25,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


class MLPEncoder(nn.Module):
    """Minimal stand-in for the MLPEncoder used by state-observation tasks."""

    def __init__(self, obs_dim=53, emb_dim=64, To=2):
        super().__init__()
        self.To = To
        self.proj = nn.Linear(obs_dim, emb_dim)

    def forward(self, obs, _mask=None):
        return self.proj(obs)


def _make_flow_map(emb_dim=64, act_dim=10, horizon=10):
    torch.manual_seed(0)
    net = ChiTransformer(
        act_dim=act_dim,
        Ta=horizon,
        obs_dim=emb_dim,
        To=2,
        d_model=emb_dim,
        nhead=4,
        num_layers=2,
        n_future_tokens=1,
        future_out_dim=emb_dim,
    )
    return FlowMap(net)


def test_state_future_loss_single_step_shape_matches():
    """Raw future state (B, obs_dim) must be encoded to emb_dim before the
    future loss so it matches the network's future head output dimension."""
    emb_dim = 64
    config = _optimization_config(future_state_loss_mode="fixed")
    flow_map = _make_flow_map(emb_dim=emb_dim)
    encoder = MLPEncoder(obs_dim=53, emb_dim=emb_dim, To=2)

    act = torch.randn(4, 10, 10)
    obs = torch.randn(4, 2, 53)
    delta_t = torch.full((4,), 0.5)
    future_obs = {"state": torch.randn(4, 53)}

    loss, info = mip_loss(
        config,
        flow_map,
        encoder,
        interp=None,
        act=act,
        obs=obs,
        delta_t=delta_t,
        future_obs=future_obs,
    )
    assert torch.isfinite(loss)
    assert "loss_future_raw" in info
    assert torch.isfinite(info["loss_future_raw"])


def test_state_future_loss_multi_step_shape_matches():
    """Multiple future state frames (B, T, obs_dim) are encoded and reduced
    across the sequence dimension before the future loss."""
    emb_dim = 64
    config = _optimization_config(future_state_loss_mode="fixed")
    flow_map = _make_flow_map(emb_dim=emb_dim)
    encoder = MLPEncoder(obs_dim=53, emb_dim=emb_dim, To=2)

    act = torch.randn(4, 10, 10)
    obs = torch.randn(4, 2, 53)
    delta_t = torch.full((4,), 0.5)
    future_obs = {"state": torch.randn(4, 3, 53)}

    loss, info = mip_loss(
        config,
        flow_map,
        encoder,
        interp=None,
        act=act,
        obs=obs,
        delta_t=delta_t,
        future_obs=future_obs,
    )
    assert torch.isfinite(loss)
    assert torch.isfinite(info["loss_future_raw"])
