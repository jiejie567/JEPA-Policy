#!/usr/bin/env bash
set -euo pipefail

# Run one third of the 18-run Transport four-camera Ablation-2 matrix.
# The full matrix is:
#   seeds 41/42/43 x
#   {baseline, future4 ratio 0.05/0.1/0.2, future2 ratio 0.1,
#    future6 ratio 0.1}.
# Variants are crossed with seeds across three nodes so each node owns six
# runs. This fits the PPU worker's supported 8-device allocation.

NODE_INDEX="${1:-}"
[[ "$NODE_INDEX" == "1" || "$NODE_INDEX" == "2" || "$NODE_INDEX" == "3" ]] || {
  echo "Usage: $0 {1|2|3}" >&2
  exit 2
}

REPO="/mnt/data_nas/ykj_jepa_policy/code/JEPA-Policy"
TRAIN="$REPO/tools/run_robomimic_train.sh"
PYTHON="/mnt/data_nas/ykj_jepa_policy/venvs/jepa_ppu/bin/python"
DATASET="/mnt/data_nas/ykj_jepa_policy/datasets/robomimic/transport/ph/image_4cam.hdf5"
DATASET_SIZE=3319270588
DATASET_SHA256="240315111ebed7a68d2a0a1896639c36eaa823b591feeee1eeeb5fb8f9eac8bd"
TASK="transport_ph_image_4cam"
ENTITY="jepa-policy"
PROJECT="ablation2"
LOG_ROOT="$REPO/logs/$PROJECT/robomimic"
RUN_TAG="${RUN_TAG:-}"
tag_component="${RUN_TAG:+_$RUN_TAG}"
CACHE_ROOT="/tmp/jepa_policy_transport4cam_node${NODE_INDEX}${tag_component}"
START_GAP="${START_GAP:-180}"
DRY_RUN="${DRY_RUN:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-8}"
EVAL_WORKERS="${EVAL_WORKERS:-8}"
EVAL_EPISODES="${EVAL_EPISODES:-40}"

EGL_RUNTIME_ROOT="/mnt/data_nas/ykj_jepa_policy/venvs/egl_noble_x86_64"
EGL_LIBRARY_DIR="$EGL_RUNTIME_ROOT/usr/lib/x86_64-linux-gnu"
EGL_DRI_DIR="$EGL_LIBRARY_DIR/dri"
EGL_VENDOR_JSON="$EGL_RUNTIME_ROOT/usr/share/glvnd/egl_vendor.d/50_mesa.json"
EGL_ENV=(
  "MUJOCO_GL=egl"
  "PYOPENGL_PLATFORM=egl"
  "EGL_PLATFORM=surfaceless"
  "LD_LIBRARY_PATH=$EGL_LIBRARY_DIR:${LD_LIBRARY_PATH:-}"
  "__EGL_VENDOR_LIBRARY_FILENAMES=$EGL_VENDOR_JSON"
  "LIBGL_DRIVERS_PATH=$EGL_DRI_DIR"
  "LIBGL_ALWAYS_SOFTWARE=1"
  "MESA_LOADER_DRIVER_OVERRIDE=llvmpipe"
  "LP_NUM_THREADS=1"
)

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

qualified_run_name() {
  local base_name="$1"
  if [[ -n "$RUN_TAG" ]]; then
    printf '%s_%s\n' "$base_name" "$RUN_TAG"
  else
    printf '%s\n' "$base_name"
  fi
}

[[ -x "$TRAIN" ]] || fail "missing training wrapper: $TRAIN"
[[ -x "$PYTHON" ]] || fail "missing robomimic Python: $PYTHON"
[[ "$RUN_TAG" =~ ^[A-Za-z0-9._-]*$ ]] ||
  fail "RUN_TAG contains unsupported characters"
[[ "$START_GAP" =~ ^[0-9]+$ ]] || fail "START_GAP must be non-negative"
[[ "$DRY_RUN" =~ ^[01]$ && "$PREFLIGHT_ONLY" =~ ^[01]$ ]] ||
  fail "DRY_RUN and PREFLIGHT_ONLY must be 0 or 1"
[[ "$DATALOADER_WORKERS" =~ ^[0-9]+$ ]] ||
  fail "DATALOADER_WORKERS must be non-negative"
[[ "$EVAL_WORKERS" =~ ^[1-9][0-9]*$ ]] ||
  fail "EVAL_WORKERS must be positive"
[[ "$EVAL_EPISODES" =~ ^[1-9][0-9]*$ ]] ||
  fail "EVAL_EPISODES must be positive"
(( EVAL_EPISODES % EVAL_WORKERS == 0 )) ||
  fail "EVAL_EPISODES must be divisible by EVAL_WORKERS"
[[ -f "$DATASET" ]] || fail "four-camera dataset is missing: $DATASET"
[[ "$(stat -c %s "$DATASET")" == "$DATASET_SIZE" ]] ||
  fail "four-camera dataset size does not match the validated artifact"
cd "$REPO" || fail "cannot enter repository: $REPO"

ALL_SEEDS=(41 42 43)
ALL_VARIANTS=(
  baseline
  ratio005
  ratio010
  ratio020
  future2_ratio010
  future6_ratio010
)
ALL_RATIOS=(none 0.05 0.1 0.2 0.1 0.1)
ALL_FUTURE_STEPS=(0 4 4 4 2 6)

SEEDS=()
VARIANTS=()
RATIOS=()
FUTURE_STEPS=()
RUN_NAMES=()

for seed_index in "${!ALL_SEEDS[@]}"; do
  seed="${ALL_SEEDS[$seed_index]}"
  for variant_index in "${!ALL_VARIANTS[@]}"; do
    assigned_node="$(( (seed_index + variant_index) % 3 + 1 ))"
    [[ "$assigned_node" == "$NODE_INDEX" ]] || continue
    variant="${ALL_VARIANTS[$variant_index]}"
    base_name="${TASK}_${variant}_seed${seed}"
    SEEDS+=("$seed")
    VARIANTS+=("$variant")
    RATIOS+=("${ALL_RATIOS[$variant_index]}")
    FUTURE_STEPS+=("${ALL_FUTURE_STEPS[$variant_index]}")
    RUN_NAMES+=("$(qualified_run_name "$base_name")")
  done
done

(( ${#RUN_NAMES[@]} == 6 )) || fail "each node must contain exactly six runs"
declare -A SEEN_RUNS=()
for run_name in "${RUN_NAMES[@]}"; do
  [[ -z "${SEEN_RUNS[$run_name]:-}" ]] || fail "duplicate run name: $run_name"
  SEEN_RUNS[$run_name]=1
done

echo "Transport 4cam Ablation-2 node $NODE_INDEX: 6 runs, entity=$ENTITY, project=$PROJECT"
for i in "${!RUN_NAMES[@]}"; do
  printf 'PLAN gpu=%d seed=%s variant=%s ratio=%s future_steps=%s run=%s\n' \
    "$i" "${SEEDS[$i]}" "${VARIANTS[$i]}" "${RATIOS[$i]}" \
    "${FUTURE_STEPS[$i]}" "${RUN_NAMES[$i]}"
done

# Independently verify every trajectory and all four camera arrays. This is
# metadata-only except for small sampled frames and is safe to run on NAS.
DATASET_PATH="$DATASET" "$PYTHON" - <<'PY' || exit 2
import os

import h5py
import numpy as np

camera_keys = (
    "shouldercamera0_image",
    "shouldercamera1_image",
    "robot0_eye_in_hand_image",
    "robot1_eye_in_hand_image",
)
lowdim_shapes = {
    "robot0_eef_pos": (3,),
    "robot0_eef_quat": (4,),
    "robot0_gripper_qpos": (2,),
    "robot1_eef_pos": (3,),
    "robot1_eef_quat": (4,),
    "robot1_gripper_qpos": (2,),
}
with h5py.File(os.environ["DATASET_PATH"], "r") as dataset:
    demos = sorted(
        dataset["data"], key=lambda name: int(name.rsplit("_", 1)[-1])
    )
    assert len(demos) == 200, len(demos)
    total = 0
    for demo_name in demos:
        demo = dataset[f"data/{demo_name}"]
        assert "next_obs" not in demo, demo_name
        samples = len(demo["actions"])
        assert demo["actions"].shape == (samples, 14), (
            demo_name,
            demo["actions"].shape,
        )
        total += samples
        for key in camera_keys:
            images = demo[f"obs/{key}"]
            assert images.shape == (samples, 84, 84, 3), (
                demo_name,
                key,
                images.shape,
            )
            assert images.dtype == np.uint8, (demo_name, key, images.dtype)
        for key, shape in lowdim_shapes.items():
            values = demo[f"obs/{key}"]
            assert values.shape == (samples, *shape), (
                demo_name,
                key,
                values.shape,
            )
    assert total == 93752, total
    assert int(dataset["data"].attrs["total"]) == total
    first_demo = dataset[f"data/{demos[0]}"]
    for key in camera_keys:
        frame = first_demo[f"obs/{key}"][0]
        assert int(frame.max()) > int(frame.min()), key
print(
    "DATASET_OK demos=200 samples=93752 cameras=4 resolution=84x84 "
    "raw_action_dim=14 policy_action_dim=20"
)
PY

mkdir -p "$LOG_ROOT" "$CACHE_ROOT/configs"

COMMON_ARGS=(
  "task=$TASK"
  "~task.dataset_repo"
  "+task.dataset_path=$DATASET"
  "network=chitransformer"
  "network.emb_dim=384"
  "network.use_causal_mask=false"
  "network.use_memory_mask=false"
  "network.rgb_model_name=resnet18"
  "network.rgb_model_weights=null"
  "network.imagenet_norm=false"
  "task.crop_shape=null"
  "task.crop_ratio=0.9"
  "task.random_crop=true"
  "task.crop_mode=temporal_consistent"
  "task.eval_crop_mode=center"
  "task.temporal_consistent_crop=true"
  "optimization.loss_type=mip"
  "optimization.t_two_step=0.9"
  "optimization.freeze_encoder=false"
  "optimization.model_path=null"
  "optimization.batch_size=256"
  "optimization.gradient_steps=300000"
  "optimization.dataloader_num_workers=$DATALOADER_WORKERS"
  "optimization.dataloader_persistent_workers=true"
  "optimization.use_compile=false"
  "optimization.auto_resume=false"
  "eval.parallel_rollout=true"
  "eval.parallel_rollout_workers=$EVAL_WORKERS"
  "eval.persistent_workers=true"
  "eval.rollout_seed=12345"
  "eval.worker_timeout_seconds=1800"
  "log.wandb_mode=online"
  "log.entity=$ENTITY"
  "log.project=$PROJECT"
  "log.group=${TASK}_joint_crop90"
  "log.log_freq=1000"
  "log.gradient_diagnostic_freq=1000"
  "log.validation_freq=10000"
  "log.validation_batch_size=16"
  "log.validation_seed=12345"
  "log.validation_delta_t=1.0"
  "log.eval_freq=10000"
  "log.eval_episodes=$EVAL_EPISODES"
  "log.save_video=false"
  "log.save_freq=5000"
)

RUN_ARGS_FILES=()
for i in "${!RUN_NAMES[@]}"; do
  seed="${SEEDS[$i]}"
  variant="${VARIANTS[$i]}"
  ratio="${RATIOS[$i]}"
  future_steps="${FUTURE_STEPS[$i]}"
  run_name="${RUN_NAMES[$i]}"
  run_dir="$LOG_ROOT/$run_name"
  args=(
    "${COMMON_ARGS[@]}"
    "optimization.seed=$seed"
    "log.exp_name=$run_name"
    "log.log_dir=$run_dir"
  )
  if [[ "$variant" == "baseline" ]]; then
    args+=(
      "network.n_future_tokens=0"
      "++task.future_state_enabled=false"
      "optimization.use_future_embed_loss=false"
      "optimization.future_joint_mode=false"
      "optimization.future_embed_loss_weight=0.0"
    )
  else
    args+=(
      "network.n_future_tokens=1"
      "++task.future_state_enabled=true"
      "++task.future_target_type=embedding"
      "++task.future_state_steps=$future_steps"
      "++task.future_state_steps_list=[$future_steps]"
      "optimization.use_future_embed_loss=true"
      "optimization.future_embed_loss_mode=mip_two_step"
      "optimization.future_joint_mode=true"
      "optimization.future_t_two_step=0.9"
      "optimization.future_state_loss_mode=ratio"
      "optimization.future_state_loss_ratio=$ratio"
      "optimization.future_state_loss_weight_min=0.000001"
      "optimization.future_state_loss_weight_max=0.1"
    )
  fi

  config_path="$CACHE_ROOT/configs/$run_name.yaml"
  EXPECTED_RUN="$run_name" EXPECTED_SEED="$seed" \
  EXPECTED_VARIANT="$variant" EXPECTED_RATIO="$ratio" \
  EXPECTED_FUTURE_STEPS="$future_steps" EXPECTED_DATASET="$DATASET" \
  EXPECTED_DATALOADER_WORKERS="$DATALOADER_WORKERS" \
  EXPECTED_EVAL_WORKERS="$EVAL_WORKERS" \
  EXPECTED_EVAL_EPISODES="$EVAL_EPISODES" \
  "$PYTHON" - "${args[@]}" >"$config_path" <<'PY'
import os
from pathlib import Path
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

with initialize_config_dir(
    version_base=None,
    config_dir="/mnt/data_nas/ykj_jepa_policy/code/JEPA-Policy/examples/configs",
):
    cfg = compose(config_name="main", overrides=sys.argv[1:])
c = OmegaConf.to_container(cfg, resolve=True)
t, n, o, e, log = (
    c["task"],
    c["network"],
    c["optimization"],
    c["eval"],
    c["log"],
)
expected_variant = os.environ["EXPECTED_VARIANT"]
expected_ratio = os.environ["EXPECTED_RATIO"]
expected_future_steps = int(os.environ["EXPECTED_FUTURE_STEPS"])
camera_keys = {
    "shouldercamera0_image",
    "shouldercamera1_image",
    "robot0_eye_in_hand_image",
    "robot1_eye_in_hand_image",
}

assert t["env_name"] == "transport" and t["env_type"] == "ph"
assert t["obs_type"] == "image" and t["abs_action"] is True
assert t["dataset_path"] == os.environ["EXPECTED_DATASET"]
assert t.get("dataset_repo") is None
assert Path(t["dataset_path"]).name == Path(t["dataset_filename"]).name
assert t["act_dim"] == 20 and t["horizon"] == 10
assert camera_keys.issubset(t["shape_meta"]["obs"])
for key in camera_keys:
    assert t["shape_meta"]["obs"][key]["shape"] == [3, 84, 84]
assert t["crop_ratio"] == 0.9
assert t["crop_mode"] == "temporal_consistent"
assert t["eval_crop_mode"] == "center"
assert t["temporal_consistent_crop"] is True

assert n["network_type"] == "chitransformer" and n["emb_dim"] == 384
assert o["loss_type"] == "mip" and o["batch_size"] == 256
assert o["gradient_steps"] == 300000
assert o["seed"] == int(os.environ["EXPECTED_SEED"])
assert o["dataloader_num_workers"] == int(
    os.environ["EXPECTED_DATALOADER_WORKERS"]
)
assert o["dataloader_persistent_workers"] is True
assert e["parallel_rollout"] is True
assert e["parallel_rollout_workers"] == int(
    os.environ["EXPECTED_EVAL_WORKERS"]
)
assert e["rollout_seed"] == 12345
assert log["wandb_mode"] == "online"
assert log["entity"] == "jepa-policy" and log["project"] == "ablation2"
assert log["group"] == "transport_ph_image_4cam_joint_crop90"
assert log["exp_name"] == os.environ["EXPECTED_RUN"]
assert log["eval_episodes"] == int(os.environ["EXPECTED_EVAL_EPISODES"])

if expected_variant == "baseline":
    assert t["future_state_enabled"] is False
    assert n["n_future_tokens"] == 0
    assert o["use_future_embed_loss"] is False
    assert o["future_joint_mode"] is False
else:
    assert t["future_state_enabled"] is True
    assert t["future_target_type"] == "embedding"
    assert t["future_state_steps"] == expected_future_steps
    assert t["future_state_steps_list"] == [expected_future_steps]
    assert n["n_future_tokens"] == 1
    assert o["use_future_embed_loss"] is True
    assert o["future_embed_loss_mode"] == "mip_two_step"
    assert o["future_joint_mode"] is True
    assert o["future_state_loss_mode"] == "ratio"
    assert o["future_state_loss_ratio"] == float(expected_ratio)

print(OmegaConf.to_yaml(cfg, resolve=True), end="")
PY
  args_file="$CACHE_ROOT/configs/$run_name.args"
  printf '%s\n' "${args[@]}" >"$args_file"
  RUN_ARGS_FILES+=("$args_file")
  echo "CONFIG_OK $run_name"
done

if (( DRY_RUN != 0 )); then
  echo "DRY_RUN_OK node=$NODE_INDEX runs=${#RUN_NAMES[@]}"
  exit 0
fi

[[ -n "${WANDB_API_KEY:-}" ]] || fail "WANDB_API_KEY is not injected"
echo "$DATASET_SHA256  $DATASET" | sha256sum -c - ||
  fail "four-camera dataset checksum failed"

hardware="$("$PYTHON" - <<'PY'
import torch

assert torch.__version__.startswith("2.6.0"), torch.__version__
assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 8, torch.cuda.device_count()
probe = torch.ones(4, device="cuda:0")
assert probe.sum().item() == 4
print(f"torch={torch.__version__} devices={torch.cuda.device_count()}")
PY
)" || fail "PPU PyTorch preflight failed"
echo "HARDWARE_OK $hardware"

cpu_count="$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)"
memory_gib="$(awk '/^MemTotal:/ {print int($2/1024/1024)}' /proc/meminfo)"
(( cpu_count >= 80 )) || fail "need 80 CPUs, found $cpu_count"
(( memory_gib >= 700 )) ||
  fail "need about 800 GB RAM, found ${memory_gib}GiB"

"$PYTHON" - <<'PY' || fail "robomimic package/version preflight failed"
from importlib import metadata
import sys

expected = {
    "torch": "2.6.0",
    "numpy": "2.2.6",
    "h5py": "3.16.0",
    "mujoco": "3.3.6",
    "robosuite": "1.5.1",
    "robomimic": "0.4.0",
    "wandb": "0.28.0",
    "hydra-core": "1.3.2",
}
prefix = "/mnt/data_nas/ykj_jepa_policy/venvs/jepa_ppu"
assert sys.prefix == prefix, (sys.prefix, prefix)
for package, version in expected.items():
    distribution = metadata.distribution(package)
    assert distribution.version == version, (
        package,
        distribution.version,
        version,
    )
    assert str(distribution.locate_file("")).startswith(prefix), package
print("VERSIONS_OK")
PY

WANDB_ENTITY="$ENTITY" WANDB_PROJECT="$PROJECT" "$PYTHON" - <<'PY' || fail "W&B authentication/entity/project preflight failed"
import os
import wandb

api = wandb.Api(api_key=os.environ["WANDB_API_KEY"], timeout=30)
viewer = api.viewer
projects = {project.name for project in api.projects(entity=os.environ["WANDB_ENTITY"])}
assert os.environ["WANDB_PROJECT"] in projects
username = getattr(viewer, "username", None) or getattr(viewer, "name", "unknown")
print(
    f"WANDB_OK user={username} target="
    f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}"
)
PY

for required_file in \
  "$EGL_LIBRARY_DIR/libEGL.so.0" \
  "$EGL_LIBRARY_DIR/libEGL.so.1" \
  "$EGL_LIBRARY_DIR/libEGL_mesa.so.0" \
  "$EGL_DRI_DIR/swrast_dri.so" \
  "$EGL_VENDOR_JSON"; do
  [[ -e "$required_file" ]] || fail "missing EGL runtime file: $required_file"
done

env \
  -u MUJOCO_EGL_DEVICE_ID \
  CUDA_VISIBLE_DEVICES=7 \
  JEPA_POLICY_EGL_DEVICE_ID=0 \
  TRANSPORT_DATASET="$DATASET" \
  "${EGL_ENV[@]}" \
  "$PYTHON" - <<'PY' || fail "Transport four-camera EGL reset failed"
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from mip.envs.robot_env import make_vec_env

with initialize_config_dir(
    version_base=None,
    config_dir=str(Path("examples/configs").resolve()),
):
    cfg = compose(
        config_name="main",
        overrides=[
            "task=transport_ph_image_4cam",
            "task.num_envs=1",
            "~task.dataset_repo",
            f"+task.dataset_path={os.environ['TRANSPORT_DATASET']}",
        ],
    )
assert cfg.task.dataset_path == os.environ["TRANSPORT_DATASET"]
assert cfg.task.get("dataset_repo") is None
env = make_vec_env(cfg.task, seed=12345)
obs, _ = env.reset()
for key in (
    "shouldercamera0_image",
    "shouldercamera1_image",
    "robot0_eye_in_hand_image",
    "robot1_eye_in_hand_image",
):
    assert obs[key].shape[-3:] == (3, 84, 84), (key, obs[key].shape)
env.close()
print("TRANSPORT_4CAM_ENV_OK physical_gpu=7 local_cuda=0 local_egl=0")
PY

if (( PREFLIGHT_ONLY != 0 )); then
  echo "PREFLIGHT_ONLY_OK node=$NODE_INDEX"
  exit 0
fi

for run_name in "${RUN_NAMES[@]}"; do
  run_dir="$LOG_ROOT/$run_name"
  [[ ! -e "$run_dir" ]] ||
    fail "fresh run directory already exists; choose a new RUN_TAG: $run_dir"
  [[ ! -e "$LOG_ROOT/$run_name.launcher.log" ]] ||
    fail "fresh launcher log already exists; choose a new RUN_TAG: $run_name"
done

manifest="$LOG_ROOT/manifest_transport_4cam_node${NODE_INDEX}${tag_component}.tsv"
status_file="$LOG_ROOT/status_transport_4cam_node${NODE_INDEX}${tag_component}.tsv"
printf 'pid\tgpu\trun_name\tlog\n' >"$manifest"
printf 'pid\tgpu\trun_name\texit_status\n' >"$status_file"

PIDS=()
terminate_children() {
  local pid
  trap - INT TERM
  for pid in "${PIDS[@]}"; do
    kill -TERM -- "-$pid" 2>/dev/null || true
  done
}
trap terminate_children INT TERM

for i in "${!RUN_NAMES[@]}"; do
  run_name="${RUN_NAMES[$i]}"
  run_dir="$LOG_ROOT/$run_name"
  run_cache="$CACHE_ROOT/$run_name"
  launcher_log="$LOG_ROOT/$run_name.launcher.log"
  mkdir -p \
    "$run_dir" \
    "$run_cache/numba" \
    "$run_cache/matplotlib" \
    "$run_cache/xdg"
  mapfile -t args <"${RUN_ARGS_FILES[$i]}"
  cp "$CACHE_ROOT/configs/$run_name.yaml" "$run_dir/resolved_config.yaml"
  printf '%q ' "$TRAIN" "${args[@]}" >"$run_dir/command.sh"
  printf '\n' >>"$run_dir/command.sh"

  setsid env \
    -u MUJOCO_EGL_DEVICE_ID \
    CUDA_VISIBLE_DEVICES="$i" \
    JEPA_POLICY_EGL_DEVICE_ID=0 \
    "${EGL_ENV[@]}" \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    HF_HUB_OFFLINE=1 \
    TORCH_HOME="/mnt/data_nas/ykj_jepa_policy/checkpoints/torchvision" \
    WANDB_MODE=online \
    WANDB_ENTITY="$ENTITY" \
    WANDB_PROJECT="$PROJECT" \
    WANDB_NAME="$run_name" \
    WANDB_RUN_GROUP="${TASK}_joint_crop90" \
    WANDB_RUN_ID="$run_name" \
    WANDB_DIR="$REPO/wandb" \
    NUMBA_CACHE_DIR="$run_cache/numba" \
    MPLCONFIGDIR="$run_cache/matplotlib" \
    XDG_CACHE_HOME="$run_cache/xdg" \
    "$TRAIN" "${args[@]}" >"$launcher_log" 2>&1 &
  pid=$!
  PIDS+=("$pid")
  printf '%s\t%s\t%s\t%s\n' "$pid" "$i" "$run_name" "$launcher_log" |
    tee -a "$manifest"
  if (( i + 1 < ${#RUN_NAMES[@]} )); then sleep "$START_GAP"; fi
done

failed=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then status=0; else status=$?; failed=1; fi
  printf '%s\t%s\t%s\t%s\n' \
    "${PIDS[$i]}" "$i" "${RUN_NAMES[$i]}" "$status" |
    tee -a "$status_file"
done

(( failed == 0 )) || fail "one or more Transport four-camera runs failed"
echo "ALL_RUNS_COMPLETED node=$NODE_INDEX"
