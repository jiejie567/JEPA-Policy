# Aligned Diffusion Policy baseline

The paper compares against a capacity-matched Diffusion Transformer trained
with the same task data and optimization/evaluation budget as JEPA Policy. It
uses a 10-step action horizon, observes two frames, executes eight actions, and
runs the trained 100-step DDPM sampler at evaluation time.

## Upstream version

- Repository: <https://github.com/real-stanford/diffusion_policy>
- Commit: `27cbbf34cc32fa3720f36bc3b9d1319bfc67af89`

Clone and pin the upstream repository:

```bash
git clone https://github.com/real-stanford/diffusion_policy.git
cd diffusion_policy
git checkout 27cbbf34cc32fa3720f36bc3b9d1319bfc67af89
```

Copy the published overlay and aligned configurations into that checkout:

```bash
cp -R /path/to/JEPA-Policy/baselines/diffusion_policy/overlay/. .
cp /path/to/JEPA-Policy/baselines/diffusion_policy/config/train_diffusion_transformer_h10_aligned_workspace.yaml \
  diffusion_policy/config/
cp /path/to/JEPA-Policy/baselines/diffusion_policy/config/task/*.yaml \
  diffusion_policy/config/task/
```

Then install Diffusion Policy using its upstream instructions. LIBERO and
MimicGen require their benchmark-specific robosuite environments, as described
in the JEPA Policy README.

## Data roots

The task files contain no machine-specific paths. Set the roots that apply to
the selected task:

```bash
export ROBOMIMIC_DATA_ROOT=/path/to/robomimic
export LIBERO_ROOT=/path/to/LIBERO
export LIBERO_DATA_ROOT=/path/to/libero_10
export MIMICGEN_DATA_ROOT=/path/to/mimicgen/core
```

Expected robomimic paths below `ROBOMIMIC_DATA_ROOT` are
`square/ph/image.hdf5`, `tool_hang/ph/image.hdf5`, and
`transport/ph/image_4cam.hdf5`.

## Training

The command below reproduces one aligned run:

```bash
python train.py \
  --config-name=train_diffusion_transformer_h10_aligned_workspace \
  task=square_image_abs \
  training.seed=41 \
  task.dataset._target_=diffusion_policy.dataset.jepa_aligned_replay_image_dataset.JepaAlignedReplayImageDataset \
  task.dataset.val_ratio=0.0 \
  logging.mode=disabled
```

Use seeds `41`, `42`, and `43`. The task configuration matrix is:

| Paper task | Hydra task | Action variants |
| --- | --- | --- |
| MokaMoka | `moka_moka_image` | delta |
| MugMug | `mug_mug_image` | delta |
| Square | `square_image_abs` | delta, absolute |
| Tool Hang | `tool_hang_image_abs` | delta, absolute |
| Transport (4cam) | `transport_image_abs` | delta, absolute |
| Coffee Preparation D1 | `coffee_preparation_d1_image` | delta |
| Kitchen D1 | `kitchen_d1_image` | delta |
| Three Piece Assembly D1 | `three_piece_assembly_d1_image` | delta |

For a robomimic delta-controller run, append
`+task.env_runner.controller_input_type=delta`; omit that override for the
absolute-controller run. As stated in the paper, the reported robomimic score
is the better of the two action parameterizations for each task. This rule was
fixed because the absolute-action variant failed to learn three robomimic
tasks. The LIBERO comparison retains Diffusion Policy's 10-step horizon; JEPA
Policy uses 16 steps on those two tasks, which is the disclosed horizon
mismatch.

## What the overlay changes

The overlay contains only the comparison-specific implementation:

- exact 300,000-optimizer-step scheduling and 10,000-step evaluation cadence;
- the capacity-matched eight-layer Diffusion Transformer preset;
- JEPA-aligned per-dimension normalization;
- temporally consistent image cropping and deterministic evaluation cropping;
- evaluation runner compatibility used by the eight benchmark configurations.

It does not contain cluster launchers, datasets, checkpoints, results, or a
copy of the upstream repository.
