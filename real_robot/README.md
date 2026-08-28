# JEPA Policy Real-Robot Inference

This directory contains the ARX5/X5 real-robot inference stack used to compare
JEPA Policy, MIP, and Diffusion Policy under one observation, action, safety,
recording, and operator-evaluation pipeline.

The release intentionally excludes checkpoints, datasets, robot recordings,
virtual environments, compiled SDK binaries, internal server paths, and
short-horizon A1/A2/A4/A8 commissioning artifacts.

The deployment environment uses Python 3.10 and is intentionally isolated from
the repository's Python 3.12 training environment. Run the commands in this
document from `real_robot/`.

## Evaluation contract

- Robot: dual-arm ARX5/X5, left CAN `can3`, right CAN `can1` by default.
- Observations: base, left-wrist, and right-wrist RGB plus 14-D joint state.
- Image input: direct `640x480 -> 128x128` `cv2.INTER_AREA` resize.
- Control: 14-D absolute joint position (`abs_qpos`) at 10 Hz.
- JEPA/MIP: two observations and eight executable actions per prediction.
- DP16: 16 denoising steps, one-step latency compensation, A8 execution.
- DP100: 100 denoising steps, three-step latency compensation, A6 execution.
- Gripper calibration: a `-0.005 m` closing bias, matching the reported runs.
- All methods share the same bounds guards and task-derived dynamics filters.

## Layout

```text
runtime/prometheus/          hardware, sessions, scheduling, workflows
runtime/JEPA-Policy/         MIP baseline and future-supervised JEPA model
runtime/diffusion_policy/    Diffusion Policy experiment snapshot
tools/arx4_jepa_eval/        dry-run and formal real-robot launchers
checkpoints/                 external checkpoint layout (weights not included)
stats/                       external JEPA/MIP normalization statistics
vendor/arx5-sdk/             pinned ARX5 SDK checkout (not vendored here)
```

## Hardware and software

The reported setup used:

- one dual-arm ARX5/X5 system with an accessible physical emergency stop;
- three Intel RealSense RGB cameras: base, left wrist, and right wrist;
- a workstation running Ubuntu, ROS 2 Humble, Python 3.10.12, and CUDA 12.1;
- PyTorch 2.4.1 and FFmpeg with `hevc_nvenc` and `libx264`; and
- the pinned ARX5 SDK revision documented below.

Different robot revisions, joint ordering, gripper calibration, cameras, or
network/CAN layouts require local validation. The supplied constants describe
the reported apparatus; they are not generic robot defaults.

## Installation

Clone the repository and enter the deployment directory:

```bash
git clone https://github.com/jiejie567/JEPA-Policy.git
cd JEPA-Policy/real_robot
```

Create the reported Python environment and install the three runtime packages:

```bash
conda env create -f environment.yml
conda activate prometheus-jepa-policy
python -m pip install -e runtime/prometheus
python -m pip install -e runtime/JEPA-Policy --no-deps
python -m pip install -e runtime/diffusion_policy --no-deps

git clone https://github.com/MrSecant/arx5-sdk.git vendor/arx5-sdk
git -C vendor/arx5-sdk checkout ce0d1e76a9237de30908ab259dd9f3e4621056cf
```

Build the ARX5 Python extension for CPython 3.10 following the SDK instructions.
The launcher expects `vendor/arx5-sdk/python/arx5_interface*.so`,
`vendor/arx5-sdk/lib/x86_64/`, and `vendor/arx5-sdk/models/X5.urdf`.

Copy `.env.example` to a shell-local file, fill in the three camera serials,
and export the variables before starting a launcher. Do not commit that file.

## Checkpoints and statistics

Place externally distributed artifacts in the following layout:

```text
checkpoints/jepa/<task>/model_step<step>.pt
checkpoints/mip/<task>/model_step<step>.pt
checkpoints/diffusion_policy/<task>/step=<step>.ckpt
stats/<task>/stats.json
```

Supported tasks are `cabinet`, `cup_stack`, `cup_upright`, `pen_insert`, and
`plate_grape`. JEPA and MIP require the matching `stats.json`; Diffusion Policy
restores its normalizer from the checkpoint.

[`checkpoints/ARTIFACT_MANIFEST.json`](checkpoints/ARTIFACT_MANIFEST.json)
enumerates the exact 75-model evaluation matrix: three methods, five tasks, and
steps 60k, 80k, 100k, 140k, and 180k. Released weights should be accompanied by
SHA-256 checksums and hosted outside the Git repository, for example in a GitHub
Release or a dedicated model host.

The expected statistics hashes are recorded in
[`stats/MANIFEST.sha256`](stats/MANIFEST.sha256). Verify them after copying the
five `stats.json` files:

```bash
(cd stats && sha256sum -c MANIFEST.sha256)
```

## Dry-run and evaluation

Start with offline inference, which does not open CAN, ROS, or RealSense:

```bash
METHOD=jepa TASK=pen_insert STEP=100000 \
  bash tools/arx4_jepa_eval/run_dry_rollout.sh
```

Then validate live observations without sending actions:

```bash
METHOD=mip TASK=pen_insert STEP=100000 ACTION_STEPS=1 \
  bash tools/arx4_jepa_eval/run_live_dry_rollout.sh
```

The paper evaluation loop keeps the model and hardware session alive and asks
the operator to label each episode with `S` (success), `F` (failure), or `A`
(skip). `R` discards the active episode and returns home.

```bash
bash tools/arx4_jepa_eval/run_paper_eval_loop.sh \
  --method jepa --task cup_upright --step 100000 --episodes 20

bash tools/arx4_jepa_eval/run_paper_eval_loop.sh \
  --method mip --task cup_stack --step 100000 --episodes 20

bash tools/arx4_jepa_eval/run_paper_eval_loop.sh \
  --method dp --task pen_insert --step 100000 --episodes 20
```

`dp16` selects the 16-step Diffusion Policy ablation. `dp` selects the
training-matched 100-step schedule.

Each formal session writes a workflow manifest, per-episode operator labels,
an aggregate summary, camera recordings, timestamps, and runtime timing logs
under `PROMETHEUS_RUN_ROOT`. Those generated artifacts are ignored by Git.

## Safety

This code can command physical hardware. Keep a trained operator at the
physical emergency stop whenever the robot is enabled. Before sending actions,
verify joint order, units, camera mapping, checkpoint/task pairing, home pose,
workspace clearance, bounds, CAN interfaces, and gripper direction on your own
system. Run the offline dry-run and then the observation-only live dry-run
first. The included limits match the reported ARX5/X5 setup and are not
universal safety guarantees.

Stop immediately if the observed joint state, camera ordering, timing, or first
command differs from the validated configuration. Do not run unattended.

## Upstream code

The JEPA/MIP snapshot is derived from *Much Ado About Noising* and the
Diffusion Policy snapshot is derived from the official Diffusion Policy
repository. Their original MIT licenses are preserved in their runtime
directories. See `NOTICE` for attribution and release provenance.
