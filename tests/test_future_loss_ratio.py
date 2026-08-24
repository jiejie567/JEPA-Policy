from types import SimpleNamespace

import torch

from mip.agent import TrainingAgent
from mip.losses import mip_loss


class DummyEncoder:
    def __call__(self, obs, _):
        return obs


class DummyFlowMap:
    def get_velocity(self, _t, act, _obs_emb):
        return act

    def net(self, act, _s, _t, obs_emb):
        future_pred = torch.zeros(
            obs_emb.shape[0],
            1,
            device=act.device,
            dtype=act.dtype,
        )
        return act, None, future_pred


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


def test_fixed_future_loss_matches_weighted_original_behavior():
    config = _optimization_config(future_state_loss_mode="fixed")
    act = torch.ones(4, 1)
    obs = torch.ones(4, 1)
    delta_t = torch.full((4,), 0.1)
    future_obs = torch.ones(4, 1, 1)

    torch.manual_seed(0)
    loss, info = mip_loss(
        config,
        DummyFlowMap(),
        DummyEncoder(),
        interp=None,
        act=act,
        obs=obs,
        delta_t=delta_t,
        future_obs=future_obs,
    )

    expected_future = config.future_state_loss_weight * info["loss_future_raw"]
    assert torch.allclose(info["loss_future"], expected_future)
    assert torch.allclose(loss, info["loss_action_raw"] + expected_future)


def test_ratio_future_weight_uses_action_over_future_and_clamps():
    owner = SimpleNamespace(
        config=SimpleNamespace(
            optimization=_optimization_config(future_state_loss_mode="ratio")
        )
    )
    future_loss = torch.tensor(2.0)
    action_loss = torch.tensor(10.0)

    future_weight, weighted_future_loss, info = TrainingAgent._future_loss_weight_and_info(
        owner,
        future_loss,
        action_loss,
    )

    assert torch.allclose(future_weight, torch.tensor(0.5))
    assert torch.allclose(weighted_future_loss, torch.tensor(1.0))
    assert torch.allclose(info["loss_future_ratio"], torch.tensor(0.1))

    owner.config.optimization.future_state_loss_ratio = 10.0
    future_weight, _, _ = TrainingAgent._future_loss_weight_and_info(
        owner,
        future_loss,
        action_loss,
    )
    assert torch.allclose(future_weight, torch.tensor(1.0))

    owner.config.optimization.future_state_loss_ratio = 0.0001
    future_weight, _, _ = TrainingAgent._future_loss_weight_and_info(
        owner,
        future_loss,
        action_loss,
    )
    assert torch.allclose(future_weight, torch.tensor(0.01))


def test_action_only_baseline_ignores_future_loss_mode():
    act = torch.ones(4, 1)
    obs = torch.ones(4, 1)
    delta_t = torch.full((4,), 0.1)

    torch.manual_seed(0)
    fixed_loss, fixed_info = mip_loss(
        _optimization_config(future_state_loss_mode="fixed"),
        DummyFlowMap(),
        DummyEncoder(),
        interp=None,
        act=act,
        obs=obs,
        delta_t=delta_t,
        future_obs=None,
    )

    torch.manual_seed(0)
    ratio_loss, ratio_info = mip_loss(
        _optimization_config(future_state_loss_mode="ratio"),
        DummyFlowMap(),
        DummyEncoder(),
        interp=None,
        act=act,
        obs=obs,
        delta_t=delta_t,
        future_obs=None,
    )

    assert "loss_future" not in fixed_info
    assert "loss_future" not in ratio_info
    assert torch.allclose(fixed_loss, ratio_loss)
