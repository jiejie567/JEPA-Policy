from copy import deepcopy
from types import SimpleNamespace

import torch
import torch.nn as nn

from jepa_policy.agent import TrainingAgent
from jepa_policy.flow_map import FlowMap
from jepa_policy.networks.chitfm import ChiTransformer
from jepa_policy.samplers import mip_sampler


class TinyEncoder(nn.Module):
    use_seq = False

    def __init__(self, input_dim=3, output_dim=8):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, obs, _mask=None):
        return self.proj(obs)


def make_agent(future_weight=0.01, future_head_only_stopgrad=False):
    torch.manual_seed(7)
    agent = TrainingAgent.__new__(TrainingAgent)
    optimization = SimpleNamespace(
        future_joint_mode=True,
        future_head_only_stopgrad=future_head_only_stopgrad,
        loss_type="mip",
        future_embed_loss_mode="mip_two_step",
        use_future_embed_loss=True,
        t_two_step=0.9,
        future_t_two_step=0.9,
        loss_scale=1.0,
        future_state_loss_mode="fixed",
        future_embed_loss_weight=future_weight,
        future_state_loss_ratio=0.05,
        future_state_loss_weight_min=1e-4,
        future_state_loss_weight_max=1.0,
        freeze_encoder=False,
        grad_clip_norm=10.0,
        ema_rate=1.0,
        device="cpu",
    )
    agent.config = SimpleNamespace(
        optimization=optimization,
        task=SimpleNamespace(future_state_enabled=True),
    )
    agent.encoder = TinyEncoder()
    net = ChiTransformer(
        act_dim=2,
        Ta=4,
        obs_dim=8,
        To=2,
        d_model=8,
        nhead=2,
        num_layers=1,
        p_drop_emb=0.0,
        p_drop_attn=0.0,
        n_cond_layers=0,
        n_future_tokens=1,
        future_out_dim=8,
        use_causal_mask=False,
        use_memory_mask=False,
    )
    agent.flow_map = FlowMap(net)
    agent.sampler = mip_sampler
    agent._compiled_sampler = mip_sampler
    agent.encoder_ema = deepcopy(agent.encoder).requires_grad_(False)
    agent.flow_map_ema = deepcopy(agent.flow_map).requires_grad_(False)
    agent.optimizer = torch.optim.AdamW(
        list(agent.encoder.parameters()) + list(agent.flow_map.parameters()),
        lr=1e-4,
    )
    agent._future_nonfinite_logged = False
    return agent


def make_batch(batch_size=4):
    torch.manual_seed(11)
    act = torch.randn(batch_size, 4, 2)
    obs = torch.randn(batch_size, 2, 3)
    future_obs = torch.randn(batch_size, 3)
    delta_t = torch.ones(batch_size)
    return act, obs, future_obs, delta_t


def grad_norm(loss, parameters):
    grads = torch.autograd.grad(
        loss, list(parameters), retain_graph=True, allow_unused=True
    )
    terms = [grad.square().sum() for grad in grads if grad is not None]
    if not terms:
        return 0.0
    return torch.stack(terms).sum().sqrt().item()


def test_joint_shapes_two_transformer_forwards_and_loss_metrics():
    agent = make_agent()
    act, obs, future_obs, delta_t = make_batch()
    calls = 0
    predictions = []
    original = agent.flow_map.net.joint_forward

    def record_predictions(*args):
        output = original(*args)
        predictions.append(output)
        return output

    agent.flow_map.net.joint_forward = record_predictions

    def count_forward(_module, _inputs, _output):
        nonlocal calls
        calls += 1

    handle = agent.flow_map.net.decoder.register_forward_hook(count_forward)
    loss, info = agent._compute_joint_mip_two_step_loss(
        act, obs, delta_t, future_obs
    )
    handle.remove()

    assert calls == 2
    future_target = agent._encode_future_target(future_obs)
    assert predictions[0][0].shape == act.shape
    assert predictions[1][0].shape == act.shape
    assert predictions[0][1].shape == future_target.shape
    assert predictions[1][1].shape == future_target.shape
    assert info["_joint_future_input_0"].shape == future_target.shape
    assert info["_joint_future_input_0"].norm().item() == 0.0
    assert loss.ndim == 0 and torch.isfinite(loss)
    for key in (
        "loss_action_first_raw",
        "loss_action_second_raw",
        "loss_action_total",
        "loss_future_first_raw",
        "loss_future_second_raw",
        "loss_future_total",
        "weighted_future_loss",
        "loss_total",
    ):
        assert key in info
        assert torch.isfinite(info[key])


def test_joint_gradient_paths_and_head_separation():
    agent = make_agent()
    act, obs, future_obs, delta_t = make_batch()
    _loss, info = agent._compute_joint_mip_two_step_loss(
        act, obs, delta_t, future_obs
    )
    action_loss = info["_joint_action_term"]
    future_loss = info["_joint_weighted_future_loss"]
    net = agent.flow_map.net

    assert grad_norm(action_loss, net.future_input_emb.parameters()) > 0
    assert grad_norm(future_loss, net.input_emb.parameters()) > 0
    assert grad_norm(action_loss, net.head.parameters()) > 0
    assert grad_norm(future_loss, net.future_head.parameters()) > 0
    assert grad_norm(action_loss, net.future_head.parameters()) == 0
    assert grad_norm(future_loss, net.head.parameters()) == 0
    assert grad_norm(action_loss, net.decoder.parameters()) > 0
    assert grad_norm(future_loss, net.decoder.parameters()) > 0
    assert grad_norm(action_loss, agent.encoder.parameters()) > 0
    assert grad_norm(future_loss, agent.encoder.parameters()) > 0


def test_head_only_stopgrad_preserves_forward_values_and_blocks_future_trunk_gradients():
    full_agent = make_agent(future_head_only_stopgrad=False)
    stopgrad_agent = make_agent(future_head_only_stopgrad=True)
    act, obs, future_obs, delta_t = make_batch()

    target = full_agent._encode_future_target(future_obs)
    generator = torch.Generator().manual_seed(20260812)
    action_noise = torch.randn(act.shape, generator=generator)
    future_noise = torch.randn(target.shape, generator=generator)

    full_loss, full_info = full_agent._compute_joint_mip_two_step_loss(
        act,
        obs,
        delta_t,
        future_obs,
        action_noise=action_noise,
        future_noise=future_noise,
    )
    stopgrad_loss, stopgrad_info = (
        stopgrad_agent._compute_joint_mip_two_step_loss(
            act,
            obs,
            delta_t,
            future_obs,
            action_noise=action_noise,
            future_noise=future_noise,
        )
    )

    # Detaching changes only backward connectivity, never forward values.
    torch.testing.assert_close(full_loss, stopgrad_loss, rtol=0, atol=0)
    for key in (
        "_joint_action_term",
        "_joint_weighted_future_loss",
        "_joint_future_pred_0",
        "_joint_future_pred_1",
    ):
        torch.testing.assert_close(
            full_info[key], stopgrad_info[key], rtol=0, atol=0
        )

    action_loss = stopgrad_info["_joint_action_term"]
    future_loss = stopgrad_info["_joint_weighted_future_loss"]
    net = stopgrad_agent.flow_map.net
    assert grad_norm(action_loss, net.decoder.parameters()) > 0
    assert grad_norm(action_loss, stopgrad_agent.encoder.parameters()) > 0
    assert grad_norm(future_loss, net.decoder.parameters()) == 0
    assert grad_norm(future_loss, net.input_emb.parameters()) == 0
    assert grad_norm(future_loss, net.future_input_emb.parameters()) == 0
    assert grad_norm(future_loss, stopgrad_agent.encoder.parameters()) == 0
    assert grad_norm(future_loss, net.future_head.parameters()) > 0
    assert stopgrad_info["future_head_only_stopgrad"].item() == 1.0


def test_head_only_stopgrad_update_keeps_future_head_trainable():
    agent = make_agent(future_head_only_stopgrad=True)
    act, obs, future_obs, delta_t = make_batch()
    before = [parameter.detach().clone() for parameter in agent.flow_map.net.future_head.parameters()]
    metrics = agent.update(act, obs, delta_t, future_obs=future_obs)
    after = list(agent.flow_map.net.future_head.parameters())

    assert any(not torch.equal(old, new) for old, new in zip(before, after, strict=True))
    assert metrics["future_head_only_stopgrad"] == 1.0


def test_zero_future_is_projected_and_prediction_depends_on_observation():
    agent = make_agent()
    act, obs, future_obs, delta_t = make_batch()
    _loss, info = agent._compute_joint_mip_two_step_loss(
        act, obs, delta_t, future_obs
    )
    net = agent.flow_map.net
    zero_future = info["_joint_future_input_0"]
    projected = net.future_input_emb(zero_future.unsqueeze(1))
    future_token_before_transformer = (
        projected
        + net.future_type_emb
        + net.pos_emb[:, net.Ta : net.Ta + net.n_future_tokens]
    )
    future_pred = info["_joint_future_pred_0"]
    obs_emb = info["_joint_obs_emb"]

    assert zero_future.norm().item() == 0.0
    # Linear projection of an exact zero input is zero with the current
    # zero-initialized bias; slot type/position still identify the future slot.
    assert projected.norm().item() == 0.0
    assert future_token_before_transformer.norm().item() > 0.0
    assert future_pred.var(dim=0).mean().item() > 0.0
    assert grad_norm(future_pred.square().mean(), [obs_emb]) > 0.0

    print(
        "zero-future diagnostics:",
        {
            "future_input_0_norm": zero_future.norm().item(),
            "future_projection_norm": projected.norm().item(),
            "future_pred_0_sample_variance": future_pred.var(dim=0).mean().item(),
            "future_pred_0_obs_grad_norm": grad_norm(
                future_pred.square().mean(), [obs_emb]
            ),
        },
    )

    shuffled = obs.flip(0)
    s = torch.zeros(act.shape[0])
    t = torch.full_like(s, 0.9)
    with torch.no_grad():
        shuffled_emb = agent.encoder(shuffled)
        _, shuffled_pred = net.joint_forward(
            torch.zeros_like(act), s, t, shuffled_emb, zero_future
        )
    assert not torch.allclose(future_pred.detach(), shuffled_pred)


def test_update_uses_exactly_two_joint_transformer_forwards():
    agent = make_agent()
    act, obs, future_obs, delta_t = make_batch()
    calls = 0

    def count_forward(_module, _inputs, _output):
        nonlocal calls
        calls += 1

    handle = agent.flow_map.net.decoder.register_forward_hook(count_forward)
    metrics = agent.update(act, obs, delta_t, future_obs=future_obs)
    handle.remove()
    assert calls == 2
    assert torch.isfinite(torch.tensor(metrics["loss_total"]))


def test_mip_rollout_ignores_external_action_noise():
    agent = make_agent()
    _act, obs, _future_obs, _delta_t = make_batch()
    agent.eval()

    zero_input = torch.zeros(4, 4, 2)
    random_input = torch.randn_like(zero_input)
    with torch.no_grad():
        from_zero = agent.sample(zero_input, obs, num_steps=1, use_ema=True)
        from_random = agent.sample(random_input, obs, num_steps=1, use_ema=True)

    assert torch.equal(from_zero, from_random)


def test_gradient_diagnostics_are_observation_only():
    control_agent = make_agent()
    diagnostic_agent = make_agent()
    act, obs, future_obs, delta_t = make_batch()

    torch.manual_seed(123)
    control_metrics = control_agent.update(
        act,
        obs,
        delta_t,
        future_obs=future_obs,
        compute_gradient_diagnostics=False,
    )
    control_rng_state = torch.get_rng_state().clone()

    torch.manual_seed(123)
    diagnostic_metrics = diagnostic_agent.update(
        act,
        obs,
        delta_t,
        future_obs=future_obs,
        compute_gradient_diagnostics=True,
    )
    diagnostic_rng_state = torch.get_rng_state().clone()

    # Diagnostic autograd traversals must not consume RNG or alter the normal
    # loss.backward()/optimizer update.
    assert torch.equal(control_rng_state, diagnostic_rng_state)
    assert control_metrics["loss"] == diagnostic_metrics["loss"]
    assert control_metrics["grad_norm"] == diagnostic_metrics["grad_norm"]
    control_parameters = list(control_agent.encoder.named_parameters()) + list(
        control_agent.flow_map.named_parameters()
    )
    diagnostic_parameters = list(
        diagnostic_agent.encoder.named_parameters()
    ) + list(diagnostic_agent.flow_map.named_parameters())
    for (control_name, control_parameter), (
        diagnostic_name,
        diagnostic_parameter,
    ) in zip(
        control_parameters,
        diagnostic_parameters,
        strict=True,
    ):
        assert control_name == diagnostic_name
        assert torch.equal(control_parameter, diagnostic_parameter)

    expected_metrics = (
        "grad_shared_action_norm",
        "grad_shared_future_norm",
        "grad_shared_cosine",
        "grad_shared_norm_ratio",
        "grad_encoder_action_norm",
        "grad_encoder_future_norm",
        "grad_encoder_cosine",
        "grad_encoder_norm_ratio",
        "grad_future_input_action_norm",
        "grad_future_input_future_norm",
        "grad_future_input_cosine",
        "grad_future_input_norm_ratio",
        "grad_future_type_action_norm",
        "grad_future_type_future_norm",
        "grad_future_type_cosine",
        "grad_future_type_norm_ratio",
        "grad_action_head_action_norm",
        "grad_action_head_future_norm",
        "grad_future_head_action_norm",
        "grad_future_head_future_norm",
        "future_target_rms",
        "future_pred_first_rms",
        "future_pred_second_rms",
        "future_pred_target_cosine_first",
        "future_pred_target_cosine_second",
    )
    for key in expected_metrics:
        assert key in diagnostic_metrics
        assert torch.isfinite(torch.tensor(diagnostic_metrics[key]))

    assert diagnostic_metrics["grad_action_head_future_norm"] == 0.0
    assert diagnostic_metrics["grad_future_head_action_norm"] == 0.0
    assert -1.0 <= diagnostic_metrics["future_pred_target_cosine_first"] <= 1.0
    assert -1.0 <= diagnostic_metrics["future_pred_target_cosine_second"] <= 1.0


def test_eval_second_pass_uses_first_predictions():
    agent = make_agent()
    act, obs, _future_obs, _delta_t = make_batch()
    calls = []
    original = agent.flow_map.net.joint_forward

    def record(action_input, s, t, condition, future_input):
        result = original(action_input, s, t, condition, future_input)
        calls.append((action_input.detach().clone(), future_input.detach().clone(), result))
        return result

    agent.flow_map.net.joint_forward = record
    sampled = mip_sampler(
        agent.config.optimization,
        agent.flow_map,
        agent.encoder,
        torch.randn_like(act),
        obs,
    )

    assert len(calls) == 2
    assert calls[0][0].norm().item() == 0.0
    assert calls[0][1].norm().item() == 0.0
    assert torch.allclose(calls[1][0], calls[0][2][0])
    assert torch.allclose(calls[1][1], calls[0][2][1])
    assert torch.allclose(sampled, calls[1][2][0])


def test_eval_can_return_second_step_future_prediction():
    agent = make_agent()
    act, obs, _future_obs, _delta_t = make_batch()
    action_pred_1, future_pred_1 = agent.sample(
        torch.randn_like(act),
        obs,
        use_ema=False,
        return_future=True,
    )

    assert action_pred_1.shape == act.shape
    assert future_pred_1.shape == (act.shape[0], agent.flow_map.net.future_out_dim)
    assert torch.isfinite(action_pred_1).all()
    assert torch.isfinite(future_pred_1).all()


def test_no_fixed_future_content_parameter_remains():
    agent = make_agent()
    names = dict(agent.flow_map.net.named_parameters())
    assert "future_tokens" not in names
    assert "future_type_emb" in names


def test_ten_step_joint_smoke_checkpoint_reload_and_eval(tmp_path):
    agent = make_agent()
    act, obs, future_obs, delta_t = make_batch()
    required_metrics = (
        "loss_action_first_raw",
        "loss_action_second_raw",
        "loss_action_total",
        "loss_future_first_raw",
        "loss_future_second_raw",
        "loss_future_total",
        "weighted_future_loss",
        "loss_total",
    )

    step_metrics = []
    for _ in range(10):
        metrics = agent.update(act, obs, delta_t, future_obs=future_obs)
        assert all(torch.isfinite(torch.tensor(metrics[key])) for key in required_metrics)
        step_metrics.append({key: metrics[key] for key in required_metrics})

    checkpoint = tmp_path / "joint_smoke.pt"
    training_state = {"n_gradient_step": 9, "best_metrics": {}, "eval_history": []}
    agent.save(checkpoint, training_state=training_state)
    assert checkpoint.is_file()

    reloaded = make_agent()
    restored = reloaded.load(checkpoint, load_optimizer=True)
    assert restored["n_gradient_step"] == 9
    reloaded.eval()
    sampled = mip_sampler(
        reloaded.config.optimization,
        reloaded.flow_map,
        reloaded.encoder,
        torch.empty_like(act),
        obs,
    )
    assert sampled.shape == act.shape
    assert torch.isfinite(sampled).all()

    last = step_metrics[-1]
    print("10-step joint smoke final metrics:", last)
