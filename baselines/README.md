# Paper baselines

This directory contains the reproducibility material for the two external
comparison methods in the main paper table. It deliberately does not vendor
complete third-party repositories.

## MIP (action-only)

MIP is an exact in-repository control: JEPA Policy extends the released
[Minimum Flow Policies](https://github.com/simchowitzlabpublic/much-ado-about-noising)
implementation, and disabling the future token and future loss recovers the
action-only method while preserving the architecture, optimizer, data loader,
and evaluation code.

Run it with the same task overrides used for JEPA Policy:

```bash
uv run examples/train_robomimic.py \
  -cn exps/mip_action_only.yaml \
  task=square_ph_image \
  optimization.seed=41 \
  optimization.gradient_steps=300000
```

The paper uses seeds `41`, `42`, and `43`. The eight task names are listed in
the root README.

## Diffusion Policy

The aligned Diffusion Policy recipe, task configurations, and the small source
overlay required by the paper protocol are in [`diffusion_policy`](diffusion_policy/README.md).
The upstream repository is pinned to an exact commit so the comparison can be
reconstructed without copying unrelated upstream code.

## Reporting protocol

All main-table runs use 300,000 optimizer steps, batch size 256, learning rate
`1e-4`, EMA rate `0.995`, seeds `41/42/43`, and evaluation every 10,000 steps.
Each evaluation contains 40 rollouts beginning at evaluation seed 12345. The
reported result is the best of the 30 evaluations, averaged over the three
training seeds.

