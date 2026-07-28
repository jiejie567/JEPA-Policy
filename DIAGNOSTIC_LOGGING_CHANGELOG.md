# Diagnostic Logging Changelog

## 2026-07-20: rollout RNG isolation

Rollout evaluation now uses the independent `eval.rollout_seed` (default
`12345`) for environment episode seeds and restores Python, NumPy, CPU Torch,
and CUDA Torch RNG states on return. Evaluation therefore no longer advances
the training RNG or changes its future stochastic trajectory.

MIP sampling always replaces its input action with zeros internally. Serial,
persistent-parallel, and standalone image rollout paths now pass a zero action
placeholder instead of allocating an unused `torch.randn` tensor. This does not
change MIP policy outputs. Non-MIP stochastic samplers retain their internal
random sampling, contained inside the isolated rollout RNG context.

## 2026-07-13: future-objective diagnostics

This change adds observation-only diagnostics. It does not change the model
architecture, loss formula, optimizer, EMA update, checkpoint schema, or rollout
policy.

### Training metrics

Every `log.gradient_diagnostic_freq` steps (default: 1000), W&B receives:

- `train/loss_action_raw`
- `train/loss_future_direct_raw`
- `train/loss_future_denoise_raw`
- `train/loss_future_weighted` (`train/loss_future` is retained as an alias)
- `train/loss_total`
- `train/grad_shared_action_norm`
- `train/grad_shared_future_norm`
- `train/grad_shared_cosine`
- `train/grad_shared_norm_ratio`
- `train/grad_encoder_{action_norm,future_norm,cosine,norm_ratio}`
- `train/grad_future_input_{action_norm,future_norm,cosine,norm_ratio}`
- `train/grad_future_type_{action_norm,future_norm,cosine,norm_ratio}`
- `train/grad_action_head_{action_norm,future_norm}`
- `train/grad_future_head_{action_norm,future_norm}`
- `train/future_target_rms`
- `train/future_pred_{first,second}_rms`
- `train/future_pred_target_cosine_{first,second}`

Gradient diagnostics use `torch.autograd.grad` on the online encoder, shared
decoder, future input/type parameters, and action/future heads. The large encoder
and decoder groups are evaluated separately to limit peak diagnostic memory.
These calls do not write parameter `.grad` buffers; the normal
`loss_total.backward()` remains the only source of optimizer gradients. Future
prediction scale/cosine values are computed from detached tensors and likewise do
not affect the training graph.

### Fixed diagnostic validation batch

The first training batch contributes a CPU-resident fixed subset (default: 16
samples). Every `log.validation_freq` steps (default: 10000), it is evaluated with:

- fixed observation, action, and future observation tensors;
- `encoder.eval()` and `flow_map.eval()`;
- `torch.no_grad()`;
- isolated Torch CPU/CUDA RNG seeded by `log.validation_seed`;
- fixed `log.validation_delta_t`.

Metrics are uploaded under `val/*`. Model modes and Torch RNG state are restored
afterward. No backward pass, optimizer step, or EMA update occurs.

This is a fixed diagnostic batch from the training dataset, not a held-out dataset.
Its purpose is to separate model drift from batch, crop, and diffusion-noise
variation.

### Configuration

```yaml
log:
  gradient_diagnostic_freq: 1000
  validation_freq: 10000
  validation_batch_size: 16
  validation_seed: 12345
  validation_delta_t: 1.0
```

Set either frequency to `0` to disable that diagnostic. Existing experiment
configuration and checkpoints remain compatible because all fields have defaults
and no state-dict keys were added or removed.
