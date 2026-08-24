#!/usr/bin/env bash

# Launch 14 independent single-GPU JEPA-Policy ablation runs on one 16-GPU node.
# GPUs 14 and 15 are intentionally unused.
#
# This script does not install packages or modify source code. When executed, it
# creates normal run outputs under logs/ and wandb/. It expects the two existing
# virtual environments and all five datasets to be available.
# Use START_RUN_INDEX=1 and RUN_SUFFIX=<tag> to retry GPUs 1-13 while
# preserving the original GPU 0 run, logs, and W&B run names.
# On PPU nodes, LIBERO automatically overlays only the PPU torch stack from
# jepa_ppu while retaining LIBERO's benchmark-specific dependencies.

set -u -o pipefail

REPO="/mnt/data_nas/ykj_jepa_policy/code/JEPA-Policy"
ROBOMIMIC_PY="${ROBOMIMIC_PY:-/mnt/data_nas/ykj_jepa_policy/venvs/jepa_ppu/bin/python}"
LIBERO_PY="${LIBERO_PY:-/mnt/data_nas/ykj_jepa_policy/venvs/libero/bin/python}"
LIBERO_ROOT="$REPO/third_party/LIBERO"

TOOL_HANG_DATASET="/mnt/data_nas/ykj_jepa_policy/datasets/robomimic/tool_hang/ph/image.hdf5"
TRANSPORT_DATASET="/mnt/data_nas/ykj_jepa_policy/datasets/robomimic/transport/ph/image.hdf5"
SQUARE_DATASET="/mnt/data_nas/ykj_jepa_policy/datasets/robomimic/square/ph/image.hdf5"
MOKA_MOKA_DATASET="/mnt/data_nas/ykj_jepa_policy/datasets/libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5"
MUG_MUG_DATASET="/mnt/data_nas/ykj_jepa_policy/datasets/libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo.hdf5"

START_GAP="${START_GAP:-60}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-8}"
EVAL_WORKERS="${EVAL_WORKERS:-10}"
EVAL_EPISODES="${EVAL_EPISODES:-40}"
START_RUN_INDEX="${START_RUN_INDEX:-0}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
LIBERO_PPU_TORCH_OVERLAY="${LIBERO_PPU_TORCH_OVERLAY:-auto}"

LOG_ROOT="$REPO/logs"
WANDB_ROOT="$REPO/wandb"
RUN_FILE_SUFFIX="${RUN_SUFFIX:+_$RUN_SUFFIX}"
MANIFEST="$LOG_ROOT/run_14_ablation_single_node${RUN_FILE_SUFFIX}.tsv"
STATUS_FILE="$LOG_ROOT/run_14_ablation_single_node_status${RUN_FILE_SUFFIX}.tsv"
EGL_RUNTIME_ROOT="/mnt/data_nas/ykj_jepa_policy/venvs/egl_noble_x86_64"
EGL_LIBRARY_DIR="$EGL_RUNTIME_ROOT/usr/lib/x86_64-linux-gnu"
EGL_DRI_DIR="$EGL_LIBRARY_DIR/dri"
EGL_VENDOR_JSON="$EGL_RUNTIME_ROOT/usr/share/glvnd/egl_vendor.d/50_mesa.json"
CACHE_ROOT="/tmp/jepa_policy_ablation_14"
PPU_TORCH_SITE="/mnt/data_nas/ykj_jepa_policy/venvs/jepa_ppu/lib/python3.12/site-packages"
LIBERO_TORCH_OVERLAY_DIR="$CACHE_ROOT/libero_ppu_torch_overlay"
ALL_RUNS=14

RUN_NAMES=(
  tool_hang_ph_image_ratio005
  tool_hang_ph_image_ratio020
  transport_ph_image_ratio005
  transport_ph_image_ratio020
  square_ph_image_baseline
  square_ph_image_ratio005
  square_ph_image_ratio020
  moka_moka_image_ratio005
  moka_moka_image_ratio010
  moka_moka_image_ratio020
  mug_mug_image_baseline
  mug_mug_image_ratio005
  mug_mug_image_ratio010
  mug_mug_image_ratio020
)

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

qualified_run_name() {
  local base_name="$1"
  if [[ -n "$RUN_SUFFIX" ]]; then
    printf '%s_%s\n' "$base_name" "$RUN_SUFFIX"
  else
    printf '%s\n' "$base_name"
  fi
}

[[ "$START_GAP" =~ ^[0-9]+$ ]] || fail "START_GAP must be a non-negative integer"
[[ "$DATALOADER_WORKERS" =~ ^[0-9]+$ ]] || fail "DATALOADER_WORKERS must be a non-negative integer"
[[ "$EVAL_WORKERS" =~ ^[1-9][0-9]*$ ]] || fail "EVAL_WORKERS must be a positive integer"
[[ "$EVAL_EPISODES" =~ ^[1-9][0-9]*$ ]] || fail "EVAL_EPISODES must be a positive integer"
[[ "$START_RUN_INDEX" =~ ^([0-9]|1[0-3])$ ]] || \
  fail "START_RUN_INDEX must be between 0 and 13"
[[ "$RUN_SUFFIX" =~ ^[A-Za-z0-9._-]*$ ]] || \
  fail "RUN_SUFFIX may contain only letters, digits, dots, underscores, and hyphens"
[[ "$LIBERO_PPU_TORCH_OVERLAY" =~ ^(auto|on|off)$ ]] || \
  fail "LIBERO_PPU_TORCH_OVERLAY must be auto, on, or off"
(( EVAL_EPISODES % EVAL_WORKERS == 0 )) || \
  fail "EVAL_EPISODES ($EVAL_EPISODES) must be divisible by EVAL_WORKERS ($EVAL_WORKERS)"
TOTAL_RUNS=$((ALL_RUNS - START_RUN_INDEX))
[[ -n "${WANDB_API_KEY:-}" ]] || \
  fail "WANDB_API_KEY is unavailable; inject it through a DLC Secret"

[[ -x "$ROBOMIMIC_PY" ]] || fail "robomimic Python not found: $ROBOMIMIC_PY"
[[ -x "$LIBERO_PY" ]] || fail "LIBERO Python not found: $LIBERO_PY"
[[ -d "$LIBERO_ROOT" ]] || fail "LIBERO root not found: $LIBERO_ROOT"
[[ -f "$LIBERO_ROOT/.libero_config/config.yaml" ]] || \
  fail "LIBERO config not found: $LIBERO_ROOT/.libero_config/config.yaml"

command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is unavailable"
command -v setsid >/dev/null 2>&1 || fail "setsid is unavailable"
GPU_LIST="$(nvidia-smi -L 2>/dev/null)" || fail "nvidia-smi could not enumerate GPUs"
GPU_COUNT=0
while IFS= read -r gpu_line; do
  [[ -n "$gpu_line" ]] && GPU_COUNT=$((GPU_COUNT + 1))
done <<<"$GPU_LIST"
(( GPU_COUNT >= 14 )) || fail "at least 14 visible GPUs are required; found $GPU_COUNT"

USE_LIBERO_PPU_TORCH=0
case "$LIBERO_PPU_TORCH_OVERLAY" in
  on)
    USE_LIBERO_PPU_TORCH=1
    ;;
  auto)
    [[ "$GPU_LIST" == *PPU* ]] && USE_LIBERO_PPU_TORCH=1
    ;;
esac

LIBERO_PYTHONPATH="$REPO:$LIBERO_ROOT"
LIBERO_TORCH_VERSION="2.9.0+cu128"
LIBERO_TORCHVISION_VERSION="0.24.0+cu128"
LIBERO_TORCHAUDIO_VERSION="2.9.0+cu128"
LIBERO_TRITON_VERSION="3.5.0"
LIBERO_TORCH_ROOT="$(dirname "$(dirname "$LIBERO_PY")")"

link_overlay_entry() {
  local entry="$1"
  local source_path="$PPU_TORCH_SITE/$entry"
  local target_path="$LIBERO_TORCH_OVERLAY_DIR/$entry"

  [[ -e "$source_path" ]] || fail "PPU torch overlay source is missing: $source_path"
  if [[ -L "$target_path" ]]; then
    [[ "$(readlink "$target_path")" == "$source_path" ]] || \
      fail "unexpected existing PPU torch overlay link: $target_path"
  elif [[ -e "$target_path" ]]; then
    fail "unexpected existing PPU torch overlay entry: $target_path"
  else
    ln -s "$source_path" "$target_path" || \
      fail "could not create PPU torch overlay link: $target_path"
  fi
}

if (( USE_LIBERO_PPU_TORCH != 0 )); then
  mkdir -p "$LIBERO_TORCH_OVERLAY_DIR"
  for overlay_entry in \
    torch torch-2.6.0.dist-info \
    torchvision torchvision-0.21.0.dist-info \
    torchaudio torchaudio-2.6.0.dist-info \
    functorch torchgen torio triton triton-3.2.0.dist-info; do
    link_overlay_entry "$overlay_entry"
  done
  LIBERO_PYTHONPATH="$LIBERO_TORCH_OVERLAY_DIR:$LIBERO_PYTHONPATH"
  LIBERO_TORCH_VERSION="2.6.0"
  LIBERO_TORCHVISION_VERSION="0.21.0"
  LIBERO_TORCHAUDIO_VERSION="2.6.0"
  LIBERO_TRITON_VERSION="3.2.0"
  LIBERO_TORCH_ROOT="$PPU_TORCH_SITE"
  echo "LIBERO PPU torch overlay enabled: $LIBERO_TORCH_OVERLAY_DIR"
fi

if ! env \
  -u VIRTUAL_ENV \
  -u CONDA_PREFIX \
  -u CONDA_DEFAULT_ENV \
  -u PYTHONHOME \
  -u PYTHONPATH \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="$REPO" \
  "$ROBOMIMIC_PY" - <<'PY'
from importlib import metadata
import sys

expected = {
    "torch": "2.6.0",
    "numpy": "2.2.6",
    "mujoco": "3.3.6",
    "robosuite": "1.5.1",
    "robomimic": "0.4.0",
    "wandb": "0.28.0",
}
expected_prefix = "/mnt/data_nas/ykj_jepa_policy/venvs/jepa_ppu"
assert sys.prefix == expected_prefix, (sys.prefix, expected_prefix)
for package, version in expected.items():
    distribution = metadata.distribution(package)
    assert distribution.version == version, (package, distribution.version, version)
    assert str(distribution.locate_file("")).startswith(expected_prefix), (
        package,
        distribution.locate_file(""),
    )
try:
    metadata.version("libero")
except metadata.PackageNotFoundError:
    pass
else:
    raise AssertionError("LIBERO distribution leaked into the robomimic environment")
PY
then
  fail "robomimic Python environment isolation/version preflight failed"
fi

if ! env \
  -u VIRTUAL_ENV \
  -u CONDA_PREFIX \
  -u CONDA_DEFAULT_ENV \
  -u PYTHONHOME \
  -u PYTHONPATH \
  PYTHONNOUSERSITE=1 \
  EXPECTED_LIBERO_TORCH_VERSION="$LIBERO_TORCH_VERSION" \
  EXPECTED_LIBERO_TORCHVISION_VERSION="$LIBERO_TORCHVISION_VERSION" \
  EXPECTED_LIBERO_TORCHAUDIO_VERSION="$LIBERO_TORCHAUDIO_VERSION" \
  EXPECTED_LIBERO_TRITON_VERSION="$LIBERO_TRITON_VERSION" \
  EXPECTED_LIBERO_TORCH_ROOT="$LIBERO_TORCH_ROOT" \
  PYTHONPATH="$LIBERO_PYTHONPATH" \
  "$LIBERO_PY" - <<'PY'
from importlib import metadata
from importlib import import_module
import os
from pathlib import Path
import sys

expected = {
    "torch": os.environ["EXPECTED_LIBERO_TORCH_VERSION"],
    "torchvision": os.environ["EXPECTED_LIBERO_TORCHVISION_VERSION"],
    "torchaudio": os.environ["EXPECTED_LIBERO_TORCHAUDIO_VERSION"],
    "triton": os.environ["EXPECTED_LIBERO_TRITON_VERSION"],
    "numpy": "1.26.0",
    "mujoco": "3.10.0",
    "robosuite": "1.4.0",
    "robomimic": "0.4.0",
    "libero": "0.1.0",
    "bddl": "1.0.1",
    "gym": "0.26.2",
    "wandb": "0.28.0",
}
expected_prefix = "/mnt/data_nas/ykj_jepa_policy/venvs/libero"
expected_libero_root = "/mnt/data_nas/ykj_jepa_policy/code/JEPA-Policy/third_party/LIBERO"
expected_torch_root = Path(os.environ["EXPECTED_LIBERO_TORCH_ROOT"]).resolve()
assert sys.prefix == expected_prefix, (sys.prefix, expected_prefix)
for package, version in expected.items():
    distribution = metadata.distribution(package)
    assert distribution.version == version, (package, distribution.version, version)
    location = str(distribution.locate_file(""))
    if package in {"torch", "torchvision", "torchaudio", "triton"}:
        module_path = Path(import_module(package).__file__).resolve()
        assert module_path.is_relative_to(expected_torch_root), (
            package,
            module_path,
            expected_torch_root,
        )
    elif package == "libero":
        assert location.startswith(expected_libero_root), (package, location)
    else:
        assert location.startswith(expected_prefix), (package, location)
PY
then
  fail "LIBERO Python environment isolation/version preflight failed"
fi

cuda_preflight() {
  local label="$1"
  local gpu="$2"
  local python_bin="$3"
  local pythonpath="$4"

  if ! env \
    -u VIRTUAL_ENV \
    -u CONDA_PREFIX \
    -u CONDA_DEFAULT_ENV \
    -u PYTHONHOME \
    -u PYTHONPATH \
    -u MUJOCO_EGL_DEVICE_ID \
    CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONNOUSERSITE=1 \
    PYTHONPATH="$pythonpath" \
    "$python_bin" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        f"CUDA is unavailable: torch={torch.__version__}, "
        f"cuda_build={torch.version.cuda}, count={torch.cuda.device_count()}"
    )
torch.cuda.init()
probe = torch.zeros(1, device="cuda")
torch.cuda.synchronize()
print(
    f"CUDA preflight passed: torch={torch.__version__}, "
    f"device={probe.device}, name={torch.cuda.get_device_name(0)}"
)
PY
  then
    fail "$label CUDA preflight failed on physical GPU $gpu"
  fi
}

if (( START_RUN_INDEX <= 6 )); then
  cuda_preflight robomimic "$START_RUN_INDEX" "$ROBOMIMIC_PY" "$REPO"
fi
libero_preflight_gpu=7
(( START_RUN_INDEX > libero_preflight_gpu )) && libero_preflight_gpu="$START_RUN_INDEX"
cuda_preflight LIBERO "$libero_preflight_gpu" "$LIBERO_PY" "$LIBERO_PYTHONPATH"

if ! env \
  -u VIRTUAL_ENV \
  -u CONDA_PREFIX \
  -u CONDA_DEFAULT_ENV \
  -u PYTHONHOME \
  -u PYTHONPATH \
  PYTHONNOUSERSITE=1 \
  WANDB_MODE=online \
  WANDB_ENTITY=jepa-policy \
  WANDB_PROJECT=ablation \
  PYTHONPATH="$LIBERO_PYTHONPATH" \
  "$LIBERO_PY" - <<'PY'
import os
import wandb

api = wandb.Api(api_key=os.environ["WANDB_API_KEY"], timeout=30)
viewer = api.viewer
project_names = {project.name for project in api.projects(entity="jepa-policy")}
if "ablation" not in project_names:
    raise RuntimeError(
        "W&B project jepa-policy/ablation is unavailable to this account; "
        "create it or grant access before launching"
    )
username = getattr(viewer, "username", None) or getattr(viewer, "name", "unknown")
print(f"W&B preflight passed: user={username}, target=jepa-policy/ablation")
PY
then
  fail "W&B authentication/network/entity/project preflight failed"
fi

for dataset in \
  "$TOOL_HANG_DATASET" \
  "$TRANSPORT_DATASET" \
  "$SQUARE_DATASET" \
  "$MOKA_MOKA_DATASET" \
  "$MUG_MUG_DATASET"; do
  [[ -f "$dataset" ]] || fail "dataset not found: $dataset"
done

[[ "$(uname -m)" == "x86_64" ]] || \
  fail "the bundled EGL runtime requires x86_64; found $(uname -m)"
for required_file in \
  "$EGL_LIBRARY_DIR/libEGL.so.0" \
  "$EGL_LIBRARY_DIR/libEGL.so.1" \
  "$EGL_LIBRARY_DIR/libEGL_mesa.so.0" \
  "$EGL_LIBRARY_DIR/libGL.so.1" \
  "$EGL_LIBRARY_DIR/libGLX.so.0" \
  "$EGL_LIBRARY_DIR/libOpenGL.so.0" \
  "$EGL_DRI_DIR/swrast_dri.so" \
  "$EGL_VENDOR_JSON"; do
  [[ -e "$required_file" ]] || fail "bundled EGL runtime file not found: $required_file"
done

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

EGL_PREFLIGHT_CACHE="$CACHE_ROOT/egl_preflight"
mkdir -p "$EGL_PREFLIGHT_CACHE"
for render_python in "$ROBOMIMIC_PY" "$LIBERO_PY"; do
  if ! env \
    -u VIRTUAL_ENV \
    -u CONDA_PREFIX \
    -u CONDA_DEFAULT_ENV \
    -u PYTHONHOME \
    -u PYTHONPATH \
    PYTHONNOUSERSITE=1 \
    XDG_CACHE_HOME="$EGL_PREFLIGHT_CACHE" \
    MUJOCO_EGL_DEVICE_ID=0 \
    "${EGL_ENV[@]}" \
    "$render_python" - <<'PY'
import mujoco

model = mujoco.MjModel.from_xml_string(
    """
    <mujoco>
      <visual>
        <global offwidth="32" offheight="32"/>
        <quality shadowsize="0"/>
      </visual>
      <worldbody>
        <light pos="0 0 3"/>
        <geom type="plane" size="3 3 .1" rgba=".3 .3 .3 1"/>
        <geom type="sphere" pos="0 0 .5" size=".5" rgba="1 0 0 1"/>
      </worldbody>
    </mujoco>
    """
)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
camera = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(camera)
camera.type = mujoco.mjtCamera.mjCAMERA_FREE
camera.lookat[:] = [0, 0, 0.3]
camera.distance = 3.0
camera.azimuth = 90.0
camera.elevation = -25.0
renderer = mujoco.Renderer(model, height=32, width=32)
try:
    renderer.update_scene(data, camera=camera)
    image = renderer.render()
    assert image.shape == (32, 32, 3), image.shape
    assert int(image.max()) > int(image.min()), (image.min(), image.max())
finally:
    renderer.close()
PY
  then
    fail "MuJoCo EGL render preflight failed: $render_python"
  fi
done

# Reproduce the important multi-GPU case before W&B creates any run: the
# training process sees a nonzero physical CUDA card, while Mesa still renders
# through its sole EGL device (index 0). Both installed robosuite versions
# otherwise mistake CUDA_VISIBLE_DEVICES for the Mesa EGL index.
EGL_ROUTE_PREFLIGHT_GPU=3
EGL_ROUTE_PYTHONS=("$ROBOMIMIC_PY" "$LIBERO_PY")
EGL_ROUTE_PYTHONPATHS=("$REPO" "$LIBERO_PYTHONPATH")
for route_index in 0 1; do
  render_python="${EGL_ROUTE_PYTHONS[$route_index]}"
  render_pythonpath="${EGL_ROUTE_PYTHONPATHS[$route_index]}"
  if ! env \
    -u VIRTUAL_ENV \
    -u CONDA_PREFIX \
    -u CONDA_DEFAULT_ENV \
    -u PYTHONHOME \
    -u PYTHONPATH \
    -u MUJOCO_EGL_DEVICE_ID \
    CUDA_VISIBLE_DEVICES="$EGL_ROUTE_PREFLIGHT_GPU" \
    JEPA_POLICY_EGL_DEVICE_ID=0 \
    PYTHONNOUSERSITE=1 \
    XDG_CACHE_HOME="$EGL_PREFLIGHT_CACHE" \
    PYTHONPATH="$render_pythonpath" \
    "${EGL_ENV[@]}" \
    "$render_python" - <<'PY'
import os

from mip.envs.egl_device import install_robosuite_egl_device_override

render_device = install_robosuite_egl_device_override()
assert render_device == 0, render_device

from robosuite.renderers.context.egl_context import EGLGLContext

context = EGLGLContext(
    max_width=32,
    max_height=32,
    device_id=int(os.environ["CUDA_VISIBLE_DEVICES"]),
)
try:
    context.make_current()
finally:
    context.free()
print(
    "robosuite EGL route passed: "
    f"cuda_visible={os.environ['CUDA_VISIBLE_DEVICES']}, "
    f"render_device={render_device}"
)
PY
  then
    fail "robosuite EGL route preflight failed: $render_python"
  fi
done
echo "EGL preflight passed: cuda=physical, render=mesa-llvmpipe:0, runtime=$EGL_RUNTIME_ROOT"

mkdir -p "$LOG_ROOT" "$WANDB_ROOT" "$CACHE_ROOT"

declare -A SEEN_RUN_NAMES=()
for (( run_index=START_RUN_INDEX; run_index<ALL_RUNS; run_index++ )); do
  base_run_name="${RUN_NAMES[$run_index]}"
  run_name="$(qualified_run_name "$base_run_name")"
  [[ -z "${SEEN_RUN_NAMES[$run_name]+x}" ]] || fail "duplicate run name: $run_name"
  SEEN_RUN_NAMES[$run_name]=1
  [[ ! -e "$LOG_ROOT/$run_name" ]] || fail "log directory already exists: $LOG_ROOT/$run_name"
  [[ ! -e "$LOG_ROOT/$run_name.log" ]] || fail "log file already exists: $LOG_ROOT/$run_name.log"
done

printf 'pid\tgpu\trun_name\tlog_path\n' >"$MANIFEST"
printf 'pid\tgpu\trun_name\texit_status\n' >"$STATUS_FILE"

PIDS=()
GPUS=()
LAUNCHED_RUN_NAMES=()
LOG_PATHS=()
LAUNCHED=0

launch_run() {
  local gpu="$1"
  (( gpu >= START_RUN_INDEX )) || return 0
  local run_name
  run_name="$(qualified_run_name "$2")"
  local group="$3"
  local benchmark="$4"
  local python_bin="$5"
  shift 5

  local log_path="$LOG_ROOT/$run_name.log"
  local run_cache="$CACHE_ROOT/$run_name"
  local run_pythonpath="$REPO"
  local benchmark_env=()
  local command_args=("$@")
  local saw_exp_name=0
  local saw_log_dir=0
  local arg_index
  for arg_index in "${!command_args[@]}"; do
    case "${command_args[$arg_index]}" in
      log.exp_name=*)
        command_args[$arg_index]="log.exp_name=$run_name"
        saw_exp_name=1
        ;;
      log.log_dir=*)
        command_args[$arg_index]="log.log_dir=$LOG_ROOT/$run_name"
        saw_log_dir=1
        ;;
    esac
  done
  (( saw_exp_name != 0 )) || fail "missing log.exp_name override for $run_name"
  (( saw_log_dir != 0 )) || fail "missing log.log_dir override for $run_name"

  if [[ "$benchmark" == "libero" ]]; then
    run_pythonpath="$LIBERO_PYTHONPATH"
    benchmark_env+=("LIBERO_CONFIG_PATH=$LIBERO_ROOT/.libero_config")
  fi

  mkdir -p "$run_cache/numba" "$run_cache/matplotlib" "$run_cache/xdg"

  setsid env \
    -u VIRTUAL_ENV \
    -u CONDA_PREFIX \
    -u CONDA_DEFAULT_ENV \
    -u PYTHONHOME \
    -u PYTHONPATH \
    -u MUJOCO_EGL_DEVICE_ID \
    CUDA_VISIBLE_DEVICES="$gpu" \
    JEPA_POLICY_EGL_DEVICE_ID=0 \
    "${EGL_ENV[@]}" \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    HF_HUB_OFFLINE=1 \
    TORCH_HOME="/mnt/data_nas/ykj_jepa_policy/checkpoints/torchvision" \
    WANDB_MODE=online \
    WANDB_ENTITY=jepa-policy \
    WANDB_PROJECT=ablation \
    WANDB_NAME="$run_name" \
    WANDB_RUN_GROUP="$group" \
    WANDB_DIR="$WANDB_ROOT" \
    "${benchmark_env[@]}" \
    PYTHONPATH="$run_pythonpath" \
    NUMBA_CACHE_DIR="$run_cache/numba" \
    MPLCONFIGDIR="$run_cache/matplotlib" \
    XDG_CACHE_HOME="$run_cache/xdg" \
    "$python_bin" "$REPO/examples/train_robomimic.py" "${command_args[@]}" \
    >"$log_path" 2>&1 &

  local pid=$!
  PIDS+=("$pid")
  GPUS+=("$gpu")
  LAUNCHED_RUN_NAMES+=("$run_name")
  LOG_PATHS+=("$log_path")
  printf '%s\t%s\t%s\t%s\n' "$pid" "$gpu" "$run_name" "$log_path" | tee -a "$MANIFEST"

  LAUNCHED=$((LAUNCHED + 1))
  if (( LAUNCHED < TOTAL_RUNS && START_GAP > 0 )); then
    sleep "$START_GAP"
  fi
}

cd "$REPO" || fail "cannot enter repository: $REPO"

# 1. Tool Hang, target future/action loss ratio 0.05, GPU 0
launch_run 0 tool_hang_ph_image_ratio005 tool_hang_ph_image_joint robomimic "$ROBOMIMIC_PY" \
  task=tool_hang_ph_image \
  '~task.dataset_repo' \
  +task.dataset_path="$TOOL_HANG_DATASET" \
  network=chitransformer \
  network.emb_dim=384 \
  network.n_future_tokens=1 \
  network.use_causal_mask=false \
  network.use_memory_mask=false \
  network.rgb_model_name=resnet18 \
  network.rgb_model_weights=null \
  network.imagenet_norm=false \
  task.crop_shape=null \
  task.crop_ratio=0.9 \
  task.random_crop=true \
  '++task.future_state_enabled=true' \
  '++task.future_target_type=embedding' \
  '++task.future_state_steps=4' \
  '++task.future_state_steps_list=[4]' \
  optimization.loss_type=mip \
  optimization.t_two_step=0.9 \
  optimization.use_future_embed_loss=true \
  optimization.future_embed_loss_mode=mip_two_step \
  optimization.future_joint_mode=true \
  optimization.future_t_two_step=0.9 \
  optimization.future_state_loss_mode=ratio \
  optimization.future_state_loss_ratio=0.05 \
  optimization.future_state_loss_weight_min=0.000001 \
  optimization.future_state_loss_weight_max=0.1 \
  optimization.freeze_encoder=false \
  optimization.model_path=null \
  optimization.seed=42 \
  optimization.batch_size=256 \
  optimization.gradient_steps=300000 \
  optimization.dataloader_num_workers="$DATALOADER_WORKERS" \
  optimization.use_compile=false \
  optimization.auto_resume=false \
  eval.parallel_rollout=true \
  eval.parallel_rollout_workers="$EVAL_WORKERS" \
  eval.persistent_workers=true \
  eval.worker_timeout_seconds=1800 \
  log.wandb_mode=online \
  log.entity=jepa-policy \
  log.project=ablation \
  log.group=tool_hang_ph_image_joint \
  log.exp_name=tool_hang_ph_image_ratio005 \
  log.log_dir="$LOG_ROOT/tool_hang_ph_image_ratio005" \
  log.log_freq=1000 \
  log.gradient_diagnostic_freq=1000 \
  log.validation_freq=10000 \
  log.validation_batch_size=16 \
  log.validation_seed=12345 \
  log.validation_delta_t=1.0 \
  log.eval_freq=10000 \
  log.eval_episodes="$EVAL_EPISODES" \
  log.save_video=false \
  log.save_freq=10000

# 2. Tool Hang, target future/action loss ratio 0.2, GPU 1
launch_run 1 tool_hang_ph_image_ratio020 tool_hang_ph_image_joint robomimic "$ROBOMIMIC_PY" \
  task=tool_hang_ph_image \
  '~task.dataset_repo' \
  +task.dataset_path="$TOOL_HANG_DATASET" \
  network=chitransformer \
  network.emb_dim=384 \
  network.n_future_tokens=1 \
  network.use_causal_mask=false \
  network.use_memory_mask=false \
  network.rgb_model_name=resnet18 \
  network.rgb_model_weights=null \
  network.imagenet_norm=false \
  task.crop_shape=null \
  task.crop_ratio=0.9 \
  task.random_crop=true \
  '++task.future_state_enabled=true' \
  '++task.future_target_type=embedding' \
  '++task.future_state_steps=4' \
  '++task.future_state_steps_list=[4]' \
  optimization.loss_type=mip \
  optimization.t_two_step=0.9 \
  optimization.use_future_embed_loss=true \
  optimization.future_embed_loss_mode=mip_two_step \
  optimization.future_joint_mode=true \
  optimization.future_t_two_step=0.9 \
  optimization.future_state_loss_mode=ratio \
  optimization.future_state_loss_ratio=0.2 \
  optimization.future_state_loss_weight_min=0.000001 \
  optimization.future_state_loss_weight_max=0.1 \
  optimization.freeze_encoder=false \
  optimization.model_path=null \
  optimization.seed=42 \
  optimization.batch_size=256 \
  optimization.gradient_steps=300000 \
  optimization.dataloader_num_workers="$DATALOADER_WORKERS" \
  optimization.use_compile=false \
  optimization.auto_resume=false \
  eval.parallel_rollout=true \
  eval.parallel_rollout_workers="$EVAL_WORKERS" \
  eval.persistent_workers=true \
  eval.worker_timeout_seconds=1800 \
  log.wandb_mode=online \
  log.entity=jepa-policy \
  log.project=ablation \
  log.group=tool_hang_ph_image_joint \
  log.exp_name=tool_hang_ph_image_ratio020 \
  log.log_dir="$LOG_ROOT/tool_hang_ph_image_ratio020" \
  log.log_freq=1000 \
  log.gradient_diagnostic_freq=1000 \
  log.validation_freq=10000 \
  log.validation_batch_size=16 \
  log.validation_seed=12345 \
  log.validation_delta_t=1.0 \
  log.eval_freq=10000 \
  log.eval_episodes="$EVAL_EPISODES" \
  log.save_video=false \
  log.save_freq=10000

# 3. Transport PH, target future/action loss ratio 0.05, GPU 2
launch_run 2 transport_ph_image_ratio005 transport_ph_image_joint robomimic "$ROBOMIMIC_PY" \
  task=transport_ph_image \
  '~task.dataset_repo' \
  +task.dataset_path="$TRANSPORT_DATASET" \
  network=chitransformer \
  network.emb_dim=384 \
  network.n_future_tokens=1 \
  network.use_causal_mask=false \
  network.use_memory_mask=false \
  network.rgb_model_name=resnet18 \
  network.rgb_model_weights=null \
  network.imagenet_norm=false \
  task.crop_shape=null \
  task.crop_ratio=0.9 \
  task.random_crop=true \
  '++task.future_state_enabled=true' \
  '++task.future_target_type=embedding' \
  '++task.future_state_steps=4' \
  '++task.future_state_steps_list=[4]' \
  optimization.loss_type=mip \
  optimization.t_two_step=0.9 \
  optimization.use_future_embed_loss=true \
  optimization.future_embed_loss_mode=mip_two_step \
  optimization.future_joint_mode=true \
  optimization.future_t_two_step=0.9 \
  optimization.future_state_loss_mode=ratio \
  optimization.future_state_loss_ratio=0.05 \
  optimization.future_state_loss_weight_min=0.000001 \
  optimization.future_state_loss_weight_max=0.1 \
  optimization.freeze_encoder=false \
  optimization.model_path=null \
  optimization.seed=42 \
  optimization.batch_size=256 \
  optimization.gradient_steps=300000 \
  optimization.dataloader_num_workers="$DATALOADER_WORKERS" \
  optimization.use_compile=false \
  optimization.auto_resume=false \
  eval.parallel_rollout=true \
  eval.parallel_rollout_workers="$EVAL_WORKERS" \
  eval.persistent_workers=true \
  eval.worker_timeout_seconds=1800 \
  log.wandb_mode=online \
  log.entity=jepa-policy \
  log.project=ablation \
  log.group=transport_ph_image_joint \
  log.exp_name=transport_ph_image_ratio005 \
  log.log_dir="$LOG_ROOT/transport_ph_image_ratio005" \
  log.log_freq=1000 \
  log.gradient_diagnostic_freq=1000 \
  log.validation_freq=10000 \
  log.validation_batch_size=16 \
  log.validation_seed=12345 \
  log.validation_delta_t=1.0 \
  log.eval_freq=10000 \
  log.eval_episodes="$EVAL_EPISODES" \
  log.save_video=false \
  log.save_freq=10000

# 4. Transport PH, target future/action loss ratio 0.2, GPU 3
launch_run 3 transport_ph_image_ratio020 transport_ph_image_joint robomimic "$ROBOMIMIC_PY" \
  task=transport_ph_image \
  '~task.dataset_repo' \
  +task.dataset_path="$TRANSPORT_DATASET" \
  network=chitransformer \
  network.emb_dim=384 \
  network.n_future_tokens=1 \
  network.use_causal_mask=false \
  network.use_memory_mask=false \
  network.rgb_model_name=resnet18 \
  network.rgb_model_weights=null \
  network.imagenet_norm=false \
  task.crop_shape=null \
  task.crop_ratio=0.9 \
  task.random_crop=true \
  '++task.future_state_enabled=true' \
  '++task.future_target_type=embedding' \
  '++task.future_state_steps=4' \
  '++task.future_state_steps_list=[4]' \
  optimization.loss_type=mip \
  optimization.t_two_step=0.9 \
  optimization.use_future_embed_loss=true \
  optimization.future_embed_loss_mode=mip_two_step \
  optimization.future_joint_mode=true \
  optimization.future_t_two_step=0.9 \
  optimization.future_state_loss_mode=ratio \
  optimization.future_state_loss_ratio=0.2 \
  optimization.future_state_loss_weight_min=0.000001 \
  optimization.future_state_loss_weight_max=0.1 \
  optimization.freeze_encoder=false \
  optimization.model_path=null \
  optimization.seed=42 \
  optimization.batch_size=256 \
  optimization.gradient_steps=300000 \
  optimization.dataloader_num_workers="$DATALOADER_WORKERS" \
  optimization.use_compile=false \
  optimization.auto_resume=false \
  eval.parallel_rollout=true \
  eval.parallel_rollout_workers="$EVAL_WORKERS" \
  eval.persistent_workers=true \
  eval.worker_timeout_seconds=1800 \
  log.wandb_mode=online \
  log.entity=jepa-policy \
  log.project=ablation \
  log.group=transport_ph_image_joint \
  log.exp_name=transport_ph_image_ratio020 \
  log.log_dir="$LOG_ROOT/transport_ph_image_ratio020" \
  log.log_freq=1000 \
  log.gradient_diagnostic_freq=1000 \
  log.validation_freq=10000 \
  log.validation_batch_size=16 \
  log.validation_seed=12345 \
  log.validation_delta_t=1.0 \
  log.eval_freq=10000 \
  log.eval_episodes="$EVAL_EPISODES" \
  log.save_video=false \
  log.save_freq=10000

# 5. Square action-only baseline, GPU 4
launch_run 4 square_ph_image_baseline square_ph_image_joint robomimic "$ROBOMIMIC_PY" \
  task=square_ph_image \
  '~task.dataset_repo' \
  +task.dataset_path="$SQUARE_DATASET" \
  network=chitransformer \
  network.emb_dim=384 \
  network.n_future_tokens=0 \
  network.use_causal_mask=false \
  network.use_memory_mask=false \
  network.rgb_model_name=resnet18 \
  network.rgb_model_weights=null \
  network.imagenet_norm=false \
  task.crop_shape=null \
  task.crop_ratio=0.9 \
  task.random_crop=true \
  '++task.future_state_enabled=false' \
  optimization.loss_type=mip \
  optimization.t_two_step=0.9 \
  optimization.use_future_embed_loss=false \
  optimization.future_joint_mode=false \
  optimization.future_embed_loss_weight=0.0 \
  optimization.freeze_encoder=false \
  optimization.model_path=null \
  optimization.seed=42 \
  optimization.batch_size=256 \
  optimization.gradient_steps=300000 \
  optimization.dataloader_num_workers="$DATALOADER_WORKERS" \
  optimization.use_compile=false \
  optimization.auto_resume=false \
  eval.parallel_rollout=true \
  eval.parallel_rollout_workers="$EVAL_WORKERS" \
  eval.persistent_workers=true \
  eval.worker_timeout_seconds=1800 \
  log.wandb_mode=online \
  log.entity=jepa-policy \
  log.project=ablation \
  log.group=square_ph_image_joint \
  log.exp_name=square_ph_image_baseline \
  log.log_dir="$LOG_ROOT/square_ph_image_baseline" \
  log.log_freq=1000 \
  log.gradient_diagnostic_freq=1000 \
  log.validation_freq=10000 \
  log.validation_batch_size=16 \
  log.validation_seed=12345 \
  log.validation_delta_t=1.0 \
  log.eval_freq=10000 \
  log.eval_episodes="$EVAL_EPISODES" \
  log.save_video=false \
  log.save_freq=10000

# 6. Square, target future/action loss ratio 0.05, GPU 5
launch_run 5 square_ph_image_ratio005 square_ph_image_joint robomimic "$ROBOMIMIC_PY" \
  task=square_ph_image \
  '~task.dataset_repo' \
  +task.dataset_path="$SQUARE_DATASET" \
  network=chitransformer \
  network.emb_dim=384 \
  network.n_future_tokens=1 \
  network.use_causal_mask=false \
  network.use_memory_mask=false \
  network.rgb_model_name=resnet18 \
  network.rgb_model_weights=null \
  network.imagenet_norm=false \
  task.crop_shape=null \
  task.crop_ratio=0.9 \
  task.random_crop=true \
  '++task.future_state_enabled=true' \
  '++task.future_target_type=embedding' \
  '++task.future_state_steps=4' \
  '++task.future_state_steps_list=[4]' \
  optimization.loss_type=mip \
  optimization.t_two_step=0.9 \
  optimization.use_future_embed_loss=true \
  optimization.future_embed_loss_mode=mip_two_step \
  optimization.future_joint_mode=true \
  optimization.future_t_two_step=0.9 \
  optimization.future_state_loss_mode=ratio \
  optimization.future_state_loss_ratio=0.05 \
  optimization.future_state_loss_weight_min=0.000001 \
  optimization.future_state_loss_weight_max=0.1 \
  optimization.freeze_encoder=false \
  optimization.model_path=null \
  optimization.seed=42 \
  optimization.batch_size=256 \
  optimization.gradient_steps=300000 \
  optimization.dataloader_num_workers="$DATALOADER_WORKERS" \
  optimization.use_compile=false \
  optimization.auto_resume=false \
  eval.parallel_rollout=true \
  eval.parallel_rollout_workers="$EVAL_WORKERS" \
  eval.persistent_workers=true \
  eval.worker_timeout_seconds=1800 \
  log.wandb_mode=online \
  log.entity=jepa-policy \
  log.project=ablation \
  log.group=square_ph_image_joint \
  log.exp_name=square_ph_image_ratio005 \
  log.log_dir="$LOG_ROOT/square_ph_image_ratio005" \
  log.log_freq=1000 \
  log.gradient_diagnostic_freq=1000 \
  log.validation_freq=10000 \
  log.validation_batch_size=16 \
  log.validation_seed=12345 \
  log.validation_delta_t=1.0 \
  log.eval_freq=10000 \
  log.eval_episodes="$EVAL_EPISODES" \
  log.save_video=false \
  log.save_freq=10000

# 7. Square, target future/action loss ratio 0.2, GPU 6
launch_run 6 square_ph_image_ratio020 square_ph_image_joint robomimic "$ROBOMIMIC_PY" \
  task=square_ph_image \
  '~task.dataset_repo' \
  +task.dataset_path="$SQUARE_DATASET" \
  network=chitransformer \
  network.emb_dim=384 \
  network.n_future_tokens=1 \
  network.use_causal_mask=false \
  network.use_memory_mask=false \
  network.rgb_model_name=resnet18 \
  network.rgb_model_weights=null \
  network.imagenet_norm=false \
  task.crop_shape=null \
  task.crop_ratio=0.9 \
  task.random_crop=true \
  '++task.future_state_enabled=true' \
  '++task.future_target_type=embedding' \
  '++task.future_state_steps=4' \
  '++task.future_state_steps_list=[4]' \
  optimization.loss_type=mip \
  optimization.t_two_step=0.9 \
  optimization.use_future_embed_loss=true \
  optimization.future_embed_loss_mode=mip_two_step \
  optimization.future_joint_mode=true \
  optimization.future_t_two_step=0.9 \
  optimization.future_state_loss_mode=ratio \
  optimization.future_state_loss_ratio=0.2 \
  optimization.future_state_loss_weight_min=0.000001 \
  optimization.future_state_loss_weight_max=0.1 \
  optimization.freeze_encoder=false \
  optimization.model_path=null \
  optimization.seed=42 \
  optimization.batch_size=256 \
  optimization.gradient_steps=300000 \
  optimization.dataloader_num_workers="$DATALOADER_WORKERS" \
  optimization.use_compile=false \
  optimization.auto_resume=false \
  eval.parallel_rollout=true \
  eval.parallel_rollout_workers="$EVAL_WORKERS" \
  eval.persistent_workers=true \
  eval.worker_timeout_seconds=1800 \
  log.wandb_mode=online \
  log.entity=jepa-policy \
  log.project=ablation \
  log.group=square_ph_image_joint \
  log.exp_name=square_ph_image_ratio020 \
  log.log_dir="$LOG_ROOT/square_ph_image_ratio020" \
  log.log_freq=1000 \
  log.gradient_diagnostic_freq=1000 \
  log.validation_freq=10000 \
  log.validation_batch_size=16 \
  log.validation_seed=12345 \
  log.validation_delta_t=1.0 \
  log.eval_freq=10000 \
  log.eval_episodes="$EVAL_EPISODES" \
  log.save_video=false \
  log.save_freq=10000

# 8. Moka Moka, target future/action loss ratio 0.05, GPU 7
launch_run 7 moka_moka_image_ratio005 moka_moka_image_joint libero "$LIBERO_PY" \
  task=moka_moka_image \
  task.dataset_repo=null \
  task.dataset_path="$MOKA_MOKA_DATASET" \
  task.libero_root="$LIBERO_ROOT" \
  network=chitransformer \
  network.emb_dim=384 \
  network.n_future_tokens=1 \
  network.use_causal_mask=false \
  network.use_memory_mask=false \
  network.rgb_model_name=resnet18 \
  network.rgb_model_weights=null \
  network.imagenet_norm=false \
  task.crop_shape=null \
  task.crop_ratio=0.9 \
  task.random_crop=true \
  '++task.future_state_enabled=true' \
  '++task.future_target_type=embedding' \
  '++task.future_state_steps=4' \
  '++task.future_state_steps_list=[4]' \
  optimization.loss_type=mip \
  optimization.t_two_step=0.9 \
  optimization.use_future_embed_loss=true \
  optimization.future_embed_loss_mode=mip_two_step \
  optimization.future_joint_mode=true \
  optimization.future_t_two_step=0.9 \
  optimization.future_state_loss_mode=ratio \
  optimization.future_state_loss_ratio=0.05 \
  optimization.future_state_loss_weight_min=0.000001 \
  optimization.future_state_loss_weight_max=0.1 \
  optimization.freeze_encoder=false \
  optimization.model_path=null \
  optimization.seed=42 \
  optimization.batch_size=256 \
  optimization.gradient_steps=300000 \
  optimization.dataloader_num_workers="$DATALOADER_WORKERS" \
  optimization.use_compile=false \
  optimization.auto_resume=false \
  eval.parallel_rollout=true \
  eval.parallel_rollout_workers="$EVAL_WORKERS" \
  eval.persistent_workers=true \
  eval.worker_timeout_seconds=1800 \
  log.wandb_mode=online \
  log.entity=jepa-policy \
  log.project=ablation \
  log.group=moka_moka_image_joint \
  log.exp_name=moka_moka_image_ratio005 \
  log.log_dir="$LOG_ROOT/moka_moka_image_ratio005" \
  log.log_freq=1000 \
  log.gradient_diagnostic_freq=1000 \
  log.validation_freq=10000 \
  log.validation_batch_size=16 \
  log.validation_seed=12345 \
  log.validation_delta_t=1.0 \
  log.eval_freq=10000 \
  log.eval_episodes="$EVAL_EPISODES" \
  log.save_video=false \
  log.save_freq=10000

# 9. Moka Moka, target future/action loss ratio 0.1, GPU 8
launch_run 8 moka_moka_image_ratio010 moka_moka_image_joint libero "$LIBERO_PY" \
  task=moka_moka_image \
  task.dataset_repo=null \
  task.dataset_path="$MOKA_MOKA_DATASET" \
  task.libero_root="$LIBERO_ROOT" \
  network=chitransformer \
  network.emb_dim=384 \
  network.n_future_tokens=1 \
  network.use_causal_mask=false \
  network.use_memory_mask=false \
  network.rgb_model_name=resnet18 \
  network.rgb_model_weights=null \
  network.imagenet_norm=false \
  task.crop_shape=null \
  task.crop_ratio=0.9 \
  task.random_crop=true \
  '++task.future_state_enabled=true' \
  '++task.future_target_type=embedding' \
  '++task.future_state_steps=4' \
  '++task.future_state_steps_list=[4]' \
  optimization.loss_type=mip \
  optimization.t_two_step=0.9 \
  optimization.use_future_embed_loss=true \
  optimization.future_embed_loss_mode=mip_two_step \
  optimization.future_joint_mode=true \
  optimization.future_t_two_step=0.9 \
  optimization.future_state_loss_mode=ratio \
  optimization.future_state_loss_ratio=0.1 \
  optimization.future_state_loss_weight_min=0.000001 \
  optimization.future_state_loss_weight_max=0.1 \
  optimization.freeze_encoder=false \
  optimization.model_path=null \
  optimization.seed=42 \
  optimization.batch_size=256 \
  optimization.gradient_steps=300000 \
  optimization.dataloader_num_workers="$DATALOADER_WORKERS" \
  optimization.use_compile=false \
  optimization.auto_resume=false \
  eval.parallel_rollout=true \
  eval.parallel_rollout_workers="$EVAL_WORKERS" \
  eval.persistent_workers=true \
  eval.worker_timeout_seconds=1800 \
  log.wandb_mode=online \
  log.entity=jepa-policy \
  log.project=ablation \
  log.group=moka_moka_image_joint \
  log.exp_name=moka_moka_image_ratio010 \
  log.log_dir="$LOG_ROOT/moka_moka_image_ratio010" \
  log.log_freq=1000 \
  log.gradient_diagnostic_freq=1000 \
  log.validation_freq=10000 \
  log.validation_batch_size=16 \
  log.validation_seed=12345 \
  log.validation_delta_t=1.0 \
  log.eval_freq=10000 \
  log.eval_episodes="$EVAL_EPISODES" \
  log.save_video=false \
  log.save_freq=10000

# 10. Moka Moka, target future/action loss ratio 0.2, GPU 9
launch_run 9 moka_moka_image_ratio020 moka_moka_image_joint libero "$LIBERO_PY" \
  task=moka_moka_image \
  task.dataset_repo=null \
  task.dataset_path="$MOKA_MOKA_DATASET" \
  task.libero_root="$LIBERO_ROOT" \
  network=chitransformer \
  network.emb_dim=384 \
  network.n_future_tokens=1 \
  network.use_causal_mask=false \
  network.use_memory_mask=false \
  network.rgb_model_name=resnet18 \
  network.rgb_model_weights=null \
  network.imagenet_norm=false \
  task.crop_shape=null \
  task.crop_ratio=0.9 \
  task.random_crop=true \
  '++task.future_state_enabled=true' \
  '++task.future_target_type=embedding' \
  '++task.future_state_steps=4' \
  '++task.future_state_steps_list=[4]' \
  optimization.loss_type=mip \
  optimization.t_two_step=0.9 \
  optimization.use_future_embed_loss=true \
  optimization.future_embed_loss_mode=mip_two_step \
  optimization.future_joint_mode=true \
  optimization.future_t_two_step=0.9 \
  optimization.future_state_loss_mode=ratio \
  optimization.future_state_loss_ratio=0.2 \
  optimization.future_state_loss_weight_min=0.000001 \
  optimization.future_state_loss_weight_max=0.1 \
  optimization.freeze_encoder=false \
  optimization.model_path=null \
  optimization.seed=42 \
  optimization.batch_size=256 \
  optimization.gradient_steps=300000 \
  optimization.dataloader_num_workers="$DATALOADER_WORKERS" \
  optimization.use_compile=false \
  optimization.auto_resume=false \
  eval.parallel_rollout=true \
  eval.parallel_rollout_workers="$EVAL_WORKERS" \
  eval.persistent_workers=true \
  eval.worker_timeout_seconds=1800 \
  log.wandb_mode=online \
  log.entity=jepa-policy \
  log.project=ablation \
  log.group=moka_moka_image_joint \
  log.exp_name=moka_moka_image_ratio020 \
  log.log_dir="$LOG_ROOT/moka_moka_image_ratio020" \
  log.log_freq=1000 \
  log.gradient_diagnostic_freq=1000 \
  log.validation_freq=10000 \
  log.validation_batch_size=16 \
  log.validation_seed=12345 \
  log.validation_delta_t=1.0 \
  log.eval_freq=10000 \
  log.eval_episodes="$EVAL_EPISODES" \
  log.save_video=false \
  log.save_freq=10000

# 11. Mug Mug action-only baseline, GPU 10
launch_run 10 mug_mug_image_baseline mug_mug_image_joint libero "$LIBERO_PY" \
  task=mug_mug_image \
  task.dataset_repo=null \
  task.dataset_path="$MUG_MUG_DATASET" \
  task.libero_root="$LIBERO_ROOT" \
  network=chitransformer \
  network.emb_dim=384 \
  network.n_future_tokens=0 \
  network.use_causal_mask=false \
  network.use_memory_mask=false \
  network.rgb_model_name=resnet18 \
  network.rgb_model_weights=null \
  network.imagenet_norm=false \
  task.crop_shape=null \
  task.crop_ratio=0.9 \
  task.random_crop=true \
  '++task.future_state_enabled=false' \
  optimization.loss_type=mip \
  optimization.t_two_step=0.9 \
  optimization.use_future_embed_loss=false \
  optimization.future_joint_mode=false \
  optimization.future_embed_loss_weight=0.0 \
  optimization.freeze_encoder=false \
  optimization.model_path=null \
  optimization.seed=42 \
  optimization.batch_size=256 \
  optimization.gradient_steps=300000 \
  optimization.dataloader_num_workers="$DATALOADER_WORKERS" \
  optimization.use_compile=false \
  optimization.auto_resume=false \
  eval.parallel_rollout=true \
  eval.parallel_rollout_workers="$EVAL_WORKERS" \
  eval.persistent_workers=true \
  eval.worker_timeout_seconds=1800 \
  log.wandb_mode=online \
  log.entity=jepa-policy \
  log.project=ablation \
  log.group=mug_mug_image_joint \
  log.exp_name=mug_mug_image_baseline \
  log.log_dir="$LOG_ROOT/mug_mug_image_baseline" \
  log.log_freq=1000 \
  log.gradient_diagnostic_freq=1000 \
  log.validation_freq=10000 \
  log.validation_batch_size=16 \
  log.validation_seed=12345 \
  log.validation_delta_t=1.0 \
  log.eval_freq=10000 \
  log.eval_episodes="$EVAL_EPISODES" \
  log.save_video=false \
  log.save_freq=10000

# 12. Mug Mug, target future/action loss ratio 0.05, GPU 11
launch_run 11 mug_mug_image_ratio005 mug_mug_image_joint libero "$LIBERO_PY" \
  task=mug_mug_image \
  task.dataset_repo=null \
  task.dataset_path="$MUG_MUG_DATASET" \
  task.libero_root="$LIBERO_ROOT" \
  network=chitransformer \
  network.emb_dim=384 \
  network.n_future_tokens=1 \
  network.use_causal_mask=false \
  network.use_memory_mask=false \
  network.rgb_model_name=resnet18 \
  network.rgb_model_weights=null \
  network.imagenet_norm=false \
  task.crop_shape=null \
  task.crop_ratio=0.9 \
  task.random_crop=true \
  '++task.future_state_enabled=true' \
  '++task.future_target_type=embedding' \
  '++task.future_state_steps=4' \
  '++task.future_state_steps_list=[4]' \
  optimization.loss_type=mip \
  optimization.t_two_step=0.9 \
  optimization.use_future_embed_loss=true \
  optimization.future_embed_loss_mode=mip_two_step \
  optimization.future_joint_mode=true \
  optimization.future_t_two_step=0.9 \
  optimization.future_state_loss_mode=ratio \
  optimization.future_state_loss_ratio=0.05 \
  optimization.future_state_loss_weight_min=0.000001 \
  optimization.future_state_loss_weight_max=0.1 \
  optimization.freeze_encoder=false \
  optimization.model_path=null \
  optimization.seed=42 \
  optimization.batch_size=256 \
  optimization.gradient_steps=300000 \
  optimization.dataloader_num_workers="$DATALOADER_WORKERS" \
  optimization.use_compile=false \
  optimization.auto_resume=false \
  eval.parallel_rollout=true \
  eval.parallel_rollout_workers="$EVAL_WORKERS" \
  eval.persistent_workers=true \
  eval.worker_timeout_seconds=1800 \
  log.wandb_mode=online \
  log.entity=jepa-policy \
  log.project=ablation \
  log.group=mug_mug_image_joint \
  log.exp_name=mug_mug_image_ratio005 \
  log.log_dir="$LOG_ROOT/mug_mug_image_ratio005" \
  log.log_freq=1000 \
  log.gradient_diagnostic_freq=1000 \
  log.validation_freq=10000 \
  log.validation_batch_size=16 \
  log.validation_seed=12345 \
  log.validation_delta_t=1.0 \
  log.eval_freq=10000 \
  log.eval_episodes="$EVAL_EPISODES" \
  log.save_video=false \
  log.save_freq=10000

# 13. Mug Mug, target future/action loss ratio 0.1, GPU 12
launch_run 12 mug_mug_image_ratio010 mug_mug_image_joint libero "$LIBERO_PY" \
  task=mug_mug_image \
  task.dataset_repo=null \
  task.dataset_path="$MUG_MUG_DATASET" \
  task.libero_root="$LIBERO_ROOT" \
  network=chitransformer \
  network.emb_dim=384 \
  network.n_future_tokens=1 \
  network.use_causal_mask=false \
  network.use_memory_mask=false \
  network.rgb_model_name=resnet18 \
  network.rgb_model_weights=null \
  network.imagenet_norm=false \
  task.crop_shape=null \
  task.crop_ratio=0.9 \
  task.random_crop=true \
  '++task.future_state_enabled=true' \
  '++task.future_target_type=embedding' \
  '++task.future_state_steps=4' \
  '++task.future_state_steps_list=[4]' \
  optimization.loss_type=mip \
  optimization.t_two_step=0.9 \
  optimization.use_future_embed_loss=true \
  optimization.future_embed_loss_mode=mip_two_step \
  optimization.future_joint_mode=true \
  optimization.future_t_two_step=0.9 \
  optimization.future_state_loss_mode=ratio \
  optimization.future_state_loss_ratio=0.1 \
  optimization.future_state_loss_weight_min=0.000001 \
  optimization.future_state_loss_weight_max=0.1 \
  optimization.freeze_encoder=false \
  optimization.model_path=null \
  optimization.seed=42 \
  optimization.batch_size=256 \
  optimization.gradient_steps=300000 \
  optimization.dataloader_num_workers="$DATALOADER_WORKERS" \
  optimization.use_compile=false \
  optimization.auto_resume=false \
  eval.parallel_rollout=true \
  eval.parallel_rollout_workers="$EVAL_WORKERS" \
  eval.persistent_workers=true \
  eval.worker_timeout_seconds=1800 \
  log.wandb_mode=online \
  log.entity=jepa-policy \
  log.project=ablation \
  log.group=mug_mug_image_joint \
  log.exp_name=mug_mug_image_ratio010 \
  log.log_dir="$LOG_ROOT/mug_mug_image_ratio010" \
  log.log_freq=1000 \
  log.gradient_diagnostic_freq=1000 \
  log.validation_freq=10000 \
  log.validation_batch_size=16 \
  log.validation_seed=12345 \
  log.validation_delta_t=1.0 \
  log.eval_freq=10000 \
  log.eval_episodes="$EVAL_EPISODES" \
  log.save_video=false \
  log.save_freq=10000

# 14. Mug Mug, target future/action loss ratio 0.2, GPU 13
launch_run 13 mug_mug_image_ratio020 mug_mug_image_joint libero "$LIBERO_PY" \
  task=mug_mug_image \
  task.dataset_repo=null \
  task.dataset_path="$MUG_MUG_DATASET" \
  task.libero_root="$LIBERO_ROOT" \
  network=chitransformer \
  network.emb_dim=384 \
  network.n_future_tokens=1 \
  network.use_causal_mask=false \
  network.use_memory_mask=false \
  network.rgb_model_name=resnet18 \
  network.rgb_model_weights=null \
  network.imagenet_norm=false \
  task.crop_shape=null \
  task.crop_ratio=0.9 \
  task.random_crop=true \
  '++task.future_state_enabled=true' \
  '++task.future_target_type=embedding' \
  '++task.future_state_steps=4' \
  '++task.future_state_steps_list=[4]' \
  optimization.loss_type=mip \
  optimization.t_two_step=0.9 \
  optimization.use_future_embed_loss=true \
  optimization.future_embed_loss_mode=mip_two_step \
  optimization.future_joint_mode=true \
  optimization.future_t_two_step=0.9 \
  optimization.future_state_loss_mode=ratio \
  optimization.future_state_loss_ratio=0.2 \
  optimization.future_state_loss_weight_min=0.000001 \
  optimization.future_state_loss_weight_max=0.1 \
  optimization.freeze_encoder=false \
  optimization.model_path=null \
  optimization.seed=42 \
  optimization.batch_size=256 \
  optimization.gradient_steps=300000 \
  optimization.dataloader_num_workers="$DATALOADER_WORKERS" \
  optimization.use_compile=false \
  optimization.auto_resume=false \
  eval.parallel_rollout=true \
  eval.parallel_rollout_workers="$EVAL_WORKERS" \
  eval.persistent_workers=true \
  eval.worker_timeout_seconds=1800 \
  log.wandb_mode=online \
  log.entity=jepa-policy \
  log.project=ablation \
  log.group=mug_mug_image_joint \
  log.exp_name=mug_mug_image_ratio020 \
  log.log_dir="$LOG_ROOT/mug_mug_image_ratio020" \
  log.log_freq=1000 \
  log.gradient_diagnostic_freq=1000 \
  log.validation_freq=10000 \
  log.validation_batch_size=16 \
  log.validation_seed=12345 \
  log.validation_delta_t=1.0 \
  log.eval_freq=10000 \
  log.eval_episodes="$EVAL_EPISODES" \
  log.save_video=false \
  log.save_freq=10000

ANY_FAILED=0
declare -A PID_TO_INDEX=()
for index in "${!PIDS[@]}"; do
  PID_TO_INDEX["${PIDS[$index]}"]="$index"
done

remaining_pids=("${PIDS[@]}")
while (( ${#remaining_pids[@]} > 0 )); do
  completed_pid=""
  if wait -n -p completed_pid "${remaining_pids[@]}"; then
    status=0
  else
    status=$?
    ANY_FAILED=1
  fi
  [[ -n "$completed_pid" && -n "${PID_TO_INDEX[$completed_pid]+x}" ]] || \
    fail "wait -n returned an unknown process: ${completed_pid:-<none>}"

  index="${PID_TO_INDEX[$completed_pid]}"
  gpu="${GPUS[$index]}"
  run_name="${LAUNCHED_RUN_NAMES[$index]}"
  pid="$completed_pid"
  printf '%s\t%s\t%s\t%s\n' "$pid" "$gpu" "$run_name" "$status" | tee -a "$STATUS_FILE"

  next_remaining_pids=()
  for pending_pid in "${remaining_pids[@]}"; do
    [[ "$pending_pid" == "$completed_pid" ]] || next_remaining_pids+=("$pending_pid")
  done
  remaining_pids=("${next_remaining_pids[@]}")
done

if (( ANY_FAILED != 0 )); then
  echo "One or more experiments failed. See $STATUS_FILE and the per-run logs." >&2
  exit 1
fi

echo "All $TOTAL_RUNS selected experiments completed successfully."
exit 0
