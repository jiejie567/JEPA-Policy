#!/usr/bin/env bash

# Shared implementation for the two ablation2 node launchers. This file is
# sourced by the benchmark-specific scripts in the repository root.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "ERROR: source this file from an ablation2 launcher" >&2
  exit 2
fi

set -u -o pipefail

: "${BENCHMARK:?BENCHMARK must be set before sourcing the common launcher}"
: "${EXPECTED_IMAGE_SIZE:?EXPECTED_IMAGE_SIZE must be set}"
(( ${#TASKS[@]} > 0 )) || {
  echo "ERROR: TASKS must not be empty" >&2
  exit 2
}

REPO="/mnt/data_nas/ykj_jepa_policy/code/JEPA-Policy"
TRAIN_PY="$REPO/examples/train_robomimic.py"
LIBERO_ROOT="$REPO/third_party/LIBERO"
ROBOMIMIC_PY="${ROBOMIMIC_PY:-/mnt/data_nas/ykj_jepa_policy/venvs/jepa_ppu/bin/python}"
LIBERO_PY="${LIBERO_PY:-/mnt/data_nas/ykj_jepa_policy/venvs/libero/bin/python}"

# Intentionally fixed for this experiment batch so inherited shell variables
# cannot send runs back to the previous `ablation` project by accident.
WANDB_ENTITY="jepa-policy"
WANDB_PROJECT="ablation2"
START_GAP="${START_GAP:-60}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-8}"
EVAL_WORKERS="${EVAL_WORKERS:-10}"
EVAL_EPISODES="${EVAL_EPISODES:-40}"
SEED="${SEED:-42}"
GPU_OFFSET="${GPU_OFFSET:-0}"
FUTURE_HORIZON_ABLATION="${FUTURE_HORIZON_ABLATION:-0}"
START_RUN_INDEX="${START_RUN_INDEX:-0}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
DRY_RUN="${DRY_RUN:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
MIN_GPUS="${MIN_GPUS:-16}"
MIN_CPUS="${MIN_CPUS:-160}"
# A nominal 1600 GB node normally reports roughly 1490 GiB after GB/GiB
# conversion and firmware reservations.
MIN_MEMORY_GIB="${MIN_MEMORY_GIB:-1400}"
LIBERO_PPU_TORCH_OVERLAY="${LIBERO_PPU_TORCH_OVERLAY:-auto}"
TRAIN_OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
TRAIN_MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
TRAIN_OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
TRAIN_NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

LOG_ROOT="$REPO/logs/$WANDB_PROJECT/$BENCHMARK"
WANDB_ROOT="$REPO/wandb"
CACHE_ROOT="/tmp/jepa_policy_${WANDB_PROJECT}_${BENCHMARK}_seed${SEED}"
RUN_FILE_SUFFIX="_seed${SEED}${RUN_SUFFIX:+_$RUN_SUFFIX}"
MANIFEST="$LOG_ROOT/manifest${RUN_FILE_SUFFIX}.tsv"
STATUS_FILE="$LOG_ROOT/status${RUN_FILE_SUFFIX}.tsv"

EGL_RUNTIME_ROOT="/mnt/data_nas/ykj_jepa_policy/venvs/egl_noble_x86_64"
EGL_LIBRARY_DIR="$EGL_RUNTIME_ROOT/usr/lib/x86_64-linux-gnu"
EGL_DRI_DIR="$EGL_LIBRARY_DIR/dri"
EGL_VENDOR_JSON="$EGL_RUNTIME_ROOT/usr/share/glvnd/egl_vendor.d/50_mesa.json"
PPU_TORCH_SITE="/mnt/data_nas/ykj_jepa_policy/venvs/jepa_ppu/lib/python3.12/site-packages"
LIBERO_TORCH_OVERLAY_DIR="$CACHE_ROOT/libero_ppu_torch_overlay"

if [[ "$FUTURE_HORIZON_ABLATION" == "1" ]]; then
  VARIANTS=(future2_ratio010 future6_ratio010)
  RATIOS=(0.1 0.1)
  FUTURE_STEPS=(2 6)
else
  VARIANTS=(baseline ratio005 ratio010 ratio020)
  RATIOS=(none 0.05 0.1 0.2)
  FUTURE_STEPS=(0 4 4 4)
fi
RUN_TASKS=()
RUN_VARIANTS=()
RUN_RATIOS=()
RUN_FUTURE_STEPS=()
RUN_NAMES=()

for task in "${TASKS[@]}"; do
  for variant_index in "${!VARIANTS[@]}"; do
    variant="${VARIANTS[$variant_index]}"
    RUN_TASKS+=("$task")
    RUN_VARIANTS+=("$variant")
    RUN_RATIOS+=("${RATIOS[$variant_index]}")
    RUN_FUTURE_STEPS+=("${FUTURE_STEPS[$variant_index]}")
    RUN_NAMES+=("${task}_${variant}_seed${SEED}")
  done
done
ALL_RUNS="${#RUN_NAMES[@]}"

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

is_boolean_flag() {
  [[ "$1" == "0" || "$1" == "1" ]]
}

[[ "$BENCHMARK" == "robomimic" || "$BENCHMARK" == "libero" ]] || \
  fail "BENCHMARK must be robomimic or libero"
[[ "$SEED" =~ ^[0-9]+$ ]] || fail "SEED must be a non-negative integer"
[[ "$GPU_OFFSET" =~ ^[0-9]+$ ]] || fail "GPU_OFFSET must be a non-negative integer"
is_boolean_flag "$FUTURE_HORIZON_ABLATION" || \
  fail "FUTURE_HORIZON_ABLATION must be 0 or 1"
[[ "$START_GAP" =~ ^[0-9]+$ ]] || fail "START_GAP must be a non-negative integer"
[[ "$DATALOADER_WORKERS" =~ ^[0-9]+$ ]] || \
  fail "DATALOADER_WORKERS must be a non-negative integer"
[[ "$EVAL_WORKERS" =~ ^[1-9][0-9]*$ ]] || \
  fail "EVAL_WORKERS must be a positive integer"
[[ "$EVAL_EPISODES" =~ ^[1-9][0-9]*$ ]] || \
  fail "EVAL_EPISODES must be a positive integer"
[[ "$START_RUN_INDEX" =~ ^[0-9]+$ ]] || \
  fail "START_RUN_INDEX must be a non-negative integer"
(( START_RUN_INDEX < ALL_RUNS )) || \
  fail "START_RUN_INDEX must be between 0 and $((ALL_RUNS - 1))"
[[ "$RUN_SUFFIX" =~ ^[A-Za-z0-9._-]*$ ]] || \
  fail "RUN_SUFFIX may contain only letters, digits, dots, underscores, and hyphens"
is_boolean_flag "$DRY_RUN" || fail "DRY_RUN must be 0 or 1"
is_boolean_flag "$PREFLIGHT_ONLY" || fail "PREFLIGHT_ONLY must be 0 or 1"
[[ "$MIN_GPUS" =~ ^[1-9][0-9]*$ ]] || fail "MIN_GPUS must be positive"
[[ "$MIN_CPUS" =~ ^[1-9][0-9]*$ ]] || fail "MIN_CPUS must be positive"
[[ "$MIN_MEMORY_GIB" =~ ^[1-9][0-9]*$ ]] || fail "MIN_MEMORY_GIB must be positive"
[[ "$LIBERO_PPU_TORCH_OVERLAY" =~ ^(auto|on|off)$ ]] || \
  fail "LIBERO_PPU_TORCH_OVERLAY must be auto, on, or off"
[[ "$TRAIN_OMP_NUM_THREADS" =~ ^[1-9][0-9]*$ ]] || fail "OMP_NUM_THREADS must be positive"
[[ "$TRAIN_MKL_NUM_THREADS" =~ ^[1-9][0-9]*$ ]] || fail "MKL_NUM_THREADS must be positive"
[[ "$TRAIN_OPENBLAS_NUM_THREADS" =~ ^[1-9][0-9]*$ ]] || \
  fail "OPENBLAS_NUM_THREADS must be positive"
[[ "$TRAIN_NUMEXPR_NUM_THREADS" =~ ^[1-9][0-9]*$ ]] || \
  fail "NUMEXPR_NUM_THREADS must be positive"
(( EVAL_EPISODES % EVAL_WORKERS == 0 )) || \
  fail "EVAL_EPISODES ($EVAL_EPISODES) must be divisible by EVAL_WORKERS ($EVAL_WORKERS)"

[[ -d "$REPO" ]] || fail "repository not found: $REPO"
[[ -f "$TRAIN_PY" ]] || fail "training entrypoint not found: $TRAIN_PY"

echo "Run matrix: benchmark=$BENCHMARK runs=$ALL_RUNS seed=$SEED entity=$WANDB_ENTITY project=$WANDB_PROJECT"
for run_index in "${!RUN_NAMES[@]}"; do
  printf 'PLAN gpu=%s task=%s variant=%s ratio=%s future_steps=%s run=%s\n' \
    "$((GPU_OFFSET + run_index))" \
    "${RUN_TASKS[$run_index]}" \
    "${RUN_VARIANTS[$run_index]}" \
    "${RUN_RATIOS[$run_index]}" \
    "${RUN_FUTURE_STEPS[$run_index]}" \
    "$(qualified_run_name "${RUN_NAMES[$run_index]}")"
done

GPU_LIST=""
GPU_COUNT=0
if (( DRY_RUN == 0 )); then
  [[ -n "${WANDB_API_KEY:-}" ]] || \
    fail "WANDB_API_KEY is unavailable; inject it through the node secret"
  command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is unavailable"
  command -v setsid >/dev/null 2>&1 || fail "setsid is unavailable"
  GPU_LIST="$(nvidia-smi -L 2>/dev/null)" || fail "nvidia-smi could not enumerate GPUs"
  while IFS= read -r gpu_line; do
    [[ -n "$gpu_line" ]] && GPU_COUNT=$((GPU_COUNT + 1))
  done <<<"$GPU_LIST"
  (( GPU_COUNT >= MIN_GPUS )) || \
    fail "at least $MIN_GPUS visible GPUs are required; found $GPU_COUNT"

  # GNU nproc honors OMP_NUM_THREADS/OMP_THREAD_LIMIT. Those variables are
  # intentionally set to 1 for each training process, but must not be mistaken
  # for the DLC container's allocated CPU count during hardware preflight.
  CPU_COUNT="$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)"
  (( CPU_COUNT >= MIN_CPUS )) || \
    fail "at least $MIN_CPUS CPUs are required; found $CPU_COUNT"
  MEMORY_KIB="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
  MEMORY_GIB=$((MEMORY_KIB / 1024 / 1024))
  (( MEMORY_GIB >= MIN_MEMORY_GIB )) || \
    fail "at least $MIN_MEMORY_GIB GiB RAM is required; found $MEMORY_GIB GiB"
  echo "Hardware preflight: gpus=$GPU_COUNT cpus=$CPU_COUNT memory=${MEMORY_GIB}GiB"
fi

USE_LIBERO_PPU_TORCH=0
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
      fail "unexpected existing overlay link: $target_path"
  elif [[ -e "$target_path" ]]; then
    fail "unexpected existing overlay entry: $target_path"
  else
    ln -s "$source_path" "$target_path" || fail "could not create overlay link: $target_path"
  fi
}

if [[ "$BENCHMARK" == "libero" ]]; then
  [[ -x "$LIBERO_PY" ]] || fail "LIBERO Python not found: $LIBERO_PY"
  [[ -d "$LIBERO_ROOT" ]] || fail "LIBERO root not found: $LIBERO_ROOT"
  [[ -f "$LIBERO_ROOT/.libero_config/config.yaml" ]] || \
    fail "LIBERO config not found: $LIBERO_ROOT/.libero_config/config.yaml"

  case "$LIBERO_PPU_TORCH_OVERLAY" in
    on) USE_LIBERO_PPU_TORCH=1 ;;
    auto)
      if (( DRY_RUN == 0 )) && [[ "$GPU_LIST" == *PPU* ]]; then
        USE_LIBERO_PPU_TORCH=1
      fi
      ;;
  esac

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
  PYTHON_BIN="$LIBERO_PY"
  RUN_PYTHONPATH="$LIBERO_PYTHONPATH"
else
  [[ -x "$ROBOMIMIC_PY" ]] || fail "robomimic Python not found: $ROBOMIMIC_PY"
  PYTHON_BIN="$ROBOMIMIC_PY"
  RUN_PYTHONPATH="$REPO"
fi

clean_python() {
  env \
    -u VIRTUAL_ENV \
    -u CONDA_PREFIX \
    -u CONDA_DEFAULT_ENV \
    -u PYTHONHOME \
    -u PYTHONPATH \
    PYTHONNOUSERSITE=1 \
    PYTHONPATH="$RUN_PYTHONPATH" \
    "$PYTHON_BIN" "$@"
}

if [[ "$BENCHMARK" == "robomimic" ]]; then
  if ! clean_python - <<'PY'
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
expected_prefix = "/mnt/data_nas/ykj_jepa_policy/venvs/jepa_ppu"
assert sys.prefix == expected_prefix, (sys.prefix, expected_prefix)
for package, version in expected.items():
    distribution = metadata.distribution(package)
    assert distribution.version == version, (package, distribution.version, version)
    assert str(distribution.locate_file("")).startswith(expected_prefix), package
try:
    metadata.version("libero")
except metadata.PackageNotFoundError:
    pass
else:
    raise AssertionError("LIBERO leaked into the robomimic environment")
PY
  then
    fail "robomimic environment isolation/version preflight failed"
  fi
else
  if ! env \
    -u VIRTUAL_ENV \
    -u CONDA_PREFIX \
    -u CONDA_DEFAULT_ENV \
    -u PYTHONHOME \
    -u PYTHONPATH \
    EXPECTED_LIBERO_TORCH_VERSION="$LIBERO_TORCH_VERSION" \
    EXPECTED_LIBERO_TORCHVISION_VERSION="$LIBERO_TORCHVISION_VERSION" \
    EXPECTED_LIBERO_TORCHAUDIO_VERSION="$LIBERO_TORCHAUDIO_VERSION" \
    EXPECTED_LIBERO_TRITON_VERSION="$LIBERO_TRITON_VERSION" \
    EXPECTED_LIBERO_TORCH_ROOT="$LIBERO_TORCH_ROOT" \
    PYTHONNOUSERSITE=1 \
    PYTHONPATH="$RUN_PYTHONPATH" \
    "$PYTHON_BIN" - <<'PY'
from importlib import import_module, metadata
import os
from pathlib import Path
import sys

expected = {
    "torch": os.environ["EXPECTED_LIBERO_TORCH_VERSION"],
    "torchvision": os.environ["EXPECTED_LIBERO_TORCHVISION_VERSION"],
    "torchaudio": os.environ["EXPECTED_LIBERO_TORCHAUDIO_VERSION"],
    "triton": os.environ["EXPECTED_LIBERO_TRITON_VERSION"],
    "numpy": "1.26.0",
    "h5py": "3.16.0",
    "mujoco": "3.10.0",
    "robosuite": "1.4.0",
    "robomimic": "0.4.0",
    "libero": "0.1.0",
    "bddl": "1.0.1",
    "wandb": "0.28.0",
    "hydra-core": "1.3.4",
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
    fail "LIBERO environment isolation/version preflight failed"
  fi
fi
echo "Environment preflight passed: benchmark=$BENCHMARK python=$PYTHON_BIN"

DATASET_SPEC=""
for task in "${TASKS[@]}"; do
  dataset_path="${DATASET_PATHS[$task]:-}"
  image_keys="${IMAGE_KEYS[$task]:-}"
  [[ -n "$dataset_path" ]] || fail "dataset path is missing for task $task"
  [[ -n "$image_keys" ]] || fail "image key list is missing for task $task"
  [[ -f "$dataset_path" ]] || fail "dataset not found: $dataset_path"
  DATASET_SPEC+="${task}|${dataset_path}|${image_keys}"$'\n'
done

if ! env \
  -u VIRTUAL_ENV \
  -u CONDA_PREFIX \
  -u CONDA_DEFAULT_ENV \
  -u PYTHONHOME \
  -u PYTHONPATH \
  DATASET_SPEC="$DATASET_SPEC" \
  EXPECTED_IMAGE_SIZE="$EXPECTED_IMAGE_SIZE" \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="$RUN_PYTHONPATH" \
  "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

import h5py

expected_size = int(os.environ["EXPECTED_IMAGE_SIZE"])
for spec in os.environ["DATASET_SPEC"].splitlines():
    task, filename, keys_csv = spec.split("|", 2)
    keys = keys_csv.split(",")
    with h5py.File(filename, "r") as dataset:
        demos = sorted(dataset["data"].keys())
        assert demos, (task, "no demos")
        for demo_name in demos:
            demo = dataset["data"][demo_name]
            assert "actions" in demo and len(demo["actions"]) > 0, (task, demo_name)
            for key in keys:
                images = demo["obs"][key]
                assert images.shape[-3:] == (expected_size, expected_size, 3), (
                    task,
                    demo_name,
                    key,
                    images.shape,
                )
        first = dataset["data"][demos[0]]
        print(
            f"Dataset preflight: task={task} demos={len(demos)} "
            f"frames={len(first['actions'])} image={expected_size}x{expected_size} "
            f"file={Path(filename).name}"
        )
PY
then
  fail "dataset structure/image-size preflight failed"
fi

cuda_preflight() {
  local gpu="$1"
  if ! env \
    -u VIRTUAL_ENV \
    -u CONDA_PREFIX \
    -u CONDA_DEFAULT_ENV \
    -u PYTHONHOME \
    -u PYTHONPATH \
    -u MUJOCO_EGL_DEVICE_ID \
    CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONNOUSERSITE=1 \
    PYTHONPATH="$RUN_PYTHONPATH" \
    "$PYTHON_BIN" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        f"CUDA unavailable: torch={torch.__version__}, "
        f"cuda_build={torch.version.cuda}, count={torch.cuda.device_count()}"
    )
torch.cuda.init()
probe = torch.zeros(1, device="cuda")
torch.cuda.synchronize()
print(
    f"CUDA preflight: torch={torch.__version__} "
    f"device={probe.device} name={torch.cuda.get_device_name(0)}"
)
PY
  then
    fail "CUDA preflight failed on physical GPU $gpu"
  fi
}

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

if (( DRY_RUN == 0 )); then
  cuda_preflight 0
  cuda_preflight "$((GPU_OFFSET + ALL_RUNS - 1))"

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

  EGL_PREFLIGHT_CACHE="$CACHE_ROOT/egl_preflight"
  mkdir -p "$EGL_PREFLIGHT_CACHE"
  if ! env \
    -u VIRTUAL_ENV \
    -u CONDA_PREFIX \
    -u CONDA_DEFAULT_ENV \
    -u PYTHONHOME \
    -u PYTHONPATH \
    PYTHONNOUSERSITE=1 \
    XDG_CACHE_HOME="$EGL_PREFLIGHT_CACHE" \
    MUJOCO_EGL_DEVICE_ID=0 \
    PYTHONPATH="$RUN_PYTHONPATH" \
    "${EGL_ENV[@]}" \
    "$PYTHON_BIN" - <<'PY'
import mujoco

model = mujoco.MjModel.from_xml_string(
    "<mujoco><visual><global offwidth='32' offheight='32'/></visual>"
    "<worldbody><light pos='0 0 3'/><geom type='sphere' size='.5'/></worldbody></mujoco>"
)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
renderer = mujoco.Renderer(model, height=32, width=32)
try:
    renderer.update_scene(data)
    image = renderer.render()
    assert image.shape == (32, 32, 3), image.shape
finally:
    renderer.close()
print("EGL render preflight passed")
PY
  then
    fail "MuJoCo EGL render preflight failed"
  fi

  # Reproduce the nonzero-physical-GPU routing used by most runs. Mesa exposes
  # one EGL device, so robosuite must route rendering to EGL device 0 instead
  # of treating CUDA_VISIBLE_DEVICES as an EGL index.
  if ! env \
    -u VIRTUAL_ENV \
    -u CONDA_PREFIX \
    -u CONDA_DEFAULT_ENV \
    -u PYTHONHOME \
    -u PYTHONPATH \
    -u MUJOCO_EGL_DEVICE_ID \
    CUDA_VISIBLE_DEVICES="$((GPU_OFFSET + ALL_RUNS - 1))" \
    JEPA_POLICY_EGL_DEVICE_ID=0 \
    PYTHONNOUSERSITE=1 \
    XDG_CACHE_HOME="$EGL_PREFLIGHT_CACHE" \
    PYTHONPATH="$RUN_PYTHONPATH" \
    "${EGL_ENV[@]}" \
    "$PYTHON_BIN" - <<'PY'
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
    "robosuite EGL route preflight: "
    f"cuda_visible={os.environ['CUDA_VISIBLE_DEVICES']} render_device={render_device}"
)
PY
  then
    fail "robosuite nonzero-GPU EGL routing preflight failed"
  fi

  if ! env \
    -u VIRTUAL_ENV \
    -u CONDA_PREFIX \
    -u CONDA_DEFAULT_ENV \
    -u PYTHONHOME \
    -u PYTHONPATH \
    WANDB_MODE=online \
    WANDB_ENTITY="$WANDB_ENTITY" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    PYTHONNOUSERSITE=1 \
    PYTHONPATH="$RUN_PYTHONPATH" \
    "$PYTHON_BIN" - <<'PY'
import os

import wandb

api = wandb.Api(api_key=os.environ["WANDB_API_KEY"], timeout=30)
viewer = api.viewer
projects = {project.name for project in api.projects(entity=os.environ["WANDB_ENTITY"])}
username = getattr(viewer, "username", None) or getattr(viewer, "name", "unknown")
state = "exists" if os.environ["WANDB_PROJECT"] in projects else "will be created by first run"
print(
    f"W&B preflight: user={username} target={os.environ['WANDB_ENTITY']}/"
    f"{os.environ['WANDB_PROJECT']} ({state})"
)
PY
  then
    fail "W&B authentication/network/entity preflight failed"
  fi
fi

COMMAND_ARGS=()
compose_run_args() {
  local task="$1"
  local variant="$2"
  local ratio="$3"
  local future_steps="$4"
  local run_name="$5"
  local dataset_path="${DATASET_PATHS[$task]}"
  local group="${task}_joint_crop90"

  COMMAND_ARGS=(
    "task=$task"
  )
  if [[ "$BENCHMARK" == "robomimic" ]]; then
    COMMAND_ARGS+=(
      "~task.dataset_repo"
      "+task.dataset_path=$dataset_path"
    )
  else
    COMMAND_ARGS+=(
      "task.dataset_repo=null"
      "task.dataset_path=$dataset_path"
      "task.libero_root=$LIBERO_ROOT"
    )
  fi

  COMMAND_ARGS+=(
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
    "optimization.seed=$SEED"
    "optimization.batch_size=256"
    "optimization.gradient_steps=300000"
    "optimization.dataloader_num_workers=$DATALOADER_WORKERS"
    "optimization.use_compile=false"
    "optimization.auto_resume=false"
    "eval.parallel_rollout=true"
    "eval.parallel_rollout_workers=$EVAL_WORKERS"
    "eval.persistent_workers=true"
    "eval.worker_timeout_seconds=1800"
    "log.wandb_mode=online"
    "log.entity=$WANDB_ENTITY"
    "log.project=$WANDB_PROJECT"
    "log.group=$group"
    "log.exp_name=$run_name"
    "log.log_dir=$LOG_ROOT/$run_name"
    "log.log_freq=1000"
    "log.gradient_diagnostic_freq=1000"
    "log.validation_freq=10000"
    "log.validation_batch_size=16"
    "log.validation_seed=12345"
    "log.validation_delta_t=1.0"
    "log.eval_freq=10000"
    "log.eval_episodes=$EVAL_EPISODES"
    "log.save_video=false"
    "log.save_freq=10000"
  )

  if [[ "$variant" == "baseline" ]]; then
    COMMAND_ARGS+=(
      "network.n_future_tokens=0"
      "++task.future_state_enabled=false"
      "optimization.use_future_embed_loss=false"
      "optimization.future_joint_mode=false"
      "optimization.future_embed_loss_weight=0.0"
    )
  else
    COMMAND_ARGS+=(
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
}

mkdir -p "$LOG_ROOT" "$WANDB_ROOT" "$CACHE_ROOT/configs"
for run_index in "${!RUN_NAMES[@]}"; do
  task="${RUN_TASKS[$run_index]}"
  variant="${RUN_VARIANTS[$run_index]}"
  ratio="${RUN_RATIOS[$run_index]}"
  future_steps="${RUN_FUTURE_STEPS[$run_index]}"
  run_name="$(qualified_run_name "${RUN_NAMES[$run_index]}")"
  compose_run_args "$task" "$variant" "$ratio" "$future_steps" "$run_name"
  config_path="$CACHE_ROOT/configs/$run_name.yaml"
  if ! env \
    -u VIRTUAL_ENV \
    -u CONDA_PREFIX \
    -u CONDA_DEFAULT_ENV \
    -u PYTHONHOME \
    -u PYTHONPATH \
    CONFIG_PATH="$config_path" \
    EXPECTED_TASK="$task" \
    EXPECTED_VARIANT="$variant" \
    EXPECTED_RATIO="$ratio" \
    EXPECTED_FUTURE_STEPS="$future_steps" \
    EXPECTED_RUN_NAME="$run_name" \
    EXPECTED_DATASET_PATH="${DATASET_PATHS[$task]}" \
    EXPECTED_IMAGE_SIZE="$EXPECTED_IMAGE_SIZE" \
    EXPECTED_ENTITY="$WANDB_ENTITY" \
    EXPECTED_PROJECT="$WANDB_PROJECT" \
    EXPECTED_SEED="$SEED" \
    PYTHONNOUSERSITE=1 \
    PYTHONPATH="$RUN_PYTHONPATH" \
    "$PYTHON_BIN" - "${COMMAND_ARGS[@]}" >"$config_path" <<'PY'
import os
from pathlib import Path
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

config_dir = Path("/mnt/data_nas/ykj_jepa_policy/code/JEPA-Policy/examples/configs")
with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
    composed = compose(config_name="main", overrides=sys.argv[1:])
config = OmegaConf.to_container(composed, resolve=True)

task = config["task"]
network = config["network"]
optimization = config["optimization"]
log = config["log"]
expected_variant = os.environ["EXPECTED_VARIANT"]
expected_ratio = os.environ["EXPECTED_RATIO"]
expected_future_steps = int(os.environ["EXPECTED_FUTURE_STEPS"])
expected_size = int(os.environ["EXPECTED_IMAGE_SIZE"])

assert task["env_name"] in os.environ["EXPECTED_TASK"], (task["env_name"], os.environ["EXPECTED_TASK"])
assert task["dataset_path"] == os.environ["EXPECTED_DATASET_PATH"]
assert task.get("dataset_repo") is None
assert Path(task["dataset_path"]).name == Path(task["dataset_filename"]).name
assert task["crop_shape"] is None
assert task["crop_ratio"] == 0.9
assert task["crop_mode"] == "temporal_consistent"
assert task["eval_crop_mode"] == "center"
assert task["temporal_consistent_crop"] is True
margin = max(1, round(expected_size * (1.0 - task["crop_ratio"]) / 2.0))
resolved = (expected_size - 2 * margin, expected_size - 2 * margin)
assert resolved == ((76, 76) if expected_size == 84 else (116, 116)), resolved
assert log["entity"] == os.environ["EXPECTED_ENTITY"]
assert log["project"] == os.environ["EXPECTED_PROJECT"]
assert log["exp_name"] == os.environ["EXPECTED_RUN_NAME"]
assert optimization["seed"] == int(os.environ["EXPECTED_SEED"])
assert optimization["gradient_steps"] == 300000

if task.get("env_type") == "libero":
    asset_root = Path(task["libero_root"]) / "libero" / "libero"
    benchmark = task["libero_benchmark_name"]
    task_name = task["libero_task_name"]
    bddl_path = asset_root / "bddl_files" / benchmark / f"{task_name}.bddl"
    init_path = asset_root / "init_files" / benchmark / f"{task_name}.pruned_init"
    assert bddl_path.is_file(), bddl_path
    assert init_path.is_file(), init_path

if expected_variant == "baseline":
    assert task["future_state_enabled"] is False
    assert network["n_future_tokens"] == 0
    assert optimization["use_future_embed_loss"] is False
    assert optimization["future_joint_mode"] is False
else:
    assert task["future_state_enabled"] is True
    assert task["future_state_steps"] == expected_future_steps
    assert task["future_state_steps_list"] == [expected_future_steps]
    assert network["n_future_tokens"] == 1
    assert optimization["use_future_embed_loss"] is True
    assert optimization["future_joint_mode"] is True
    assert optimization["future_state_loss_mode"] == "ratio"
    assert optimization["future_state_loss_ratio"] == float(expected_ratio)

print(
    f"CONFIG_OK {os.environ['EXPECTED_RUN_NAME']} crop={resolved} "
    f"future_ratio={expected_ratio} project={log['project']}",
    file=sys.stderr,
)
print(OmegaConf.to_yaml(composed, resolve=True), end="")
PY
  then
    fail "Hydra composition/resolved config validation failed for $run_name"
  fi
done

echo "Validated $ALL_RUNS configs for $BENCHMARK."
if (( DRY_RUN != 0 )); then
  echo "Dry run complete; no CUDA, W&B network call, or training process was started."
  exit 0
fi
if (( PREFLIGHT_ONLY != 0 )); then
  echo "Full node preflight complete; no training process was started."
  exit 0
fi

for (( run_index=START_RUN_INDEX; run_index<ALL_RUNS; run_index++ )); do
  run_name="$(qualified_run_name "${RUN_NAMES[$run_index]}")"
  [[ ! -e "$LOG_ROOT/$run_name" ]] || \
    fail "log directory already exists: $LOG_ROOT/$run_name"
  [[ ! -e "$LOG_ROOT/$run_name.launcher.log" ]] || \
    fail "launcher log already exists: $LOG_ROOT/$run_name.launcher.log"
done

printf 'pid\tgpu\trun_name\tlog_path\n' >"$MANIFEST"
printf 'pid\tgpu\trun_name\texit_status\n' >"$STATUS_FILE"

PIDS=()
GPUS=()
LAUNCHED_RUN_NAMES=()
LAUNCHER_LOGS=()
LAUNCHED=0
TOTAL_SELECTED=$((ALL_RUNS - START_RUN_INDEX))

terminate_children() {
  local pid
  trap - INT TERM
  echo "Termination requested; forwarding SIGTERM to all training process groups." >&2
  for pid in "${PIDS[@]}"; do
    kill -TERM -- "-$pid" 2>/dev/null || true
  done
}
trap terminate_children INT TERM

launch_run() {
  local gpu="$1"
  local task="$2"
  local variant="$3"
  local ratio="$4"
  local future_steps="$5"
  local base_name="$6"
  local run_name
  run_name="$(qualified_run_name "$base_name")"
  local run_dir="$LOG_ROOT/$run_name"
  local launcher_log="$LOG_ROOT/$run_name.launcher.log"
  local run_cache="$CACHE_ROOT/$run_name"
  local benchmark_env=()

  compose_run_args "$task" "$variant" "$ratio" "$future_steps" "$run_name"
  mkdir -p "$run_dir" "$run_cache/numba" "$run_cache/matplotlib" "$run_cache/xdg"
  {
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu"
    printf '%q ' "$PYTHON_BIN" "$TRAIN_PY" "${COMMAND_ARGS[@]}"
    printf '\n'
  } >"$run_dir/command.sh"
  cp "$CACHE_ROOT/configs/$run_name.yaml" "$run_dir/resolved_config.yaml"

  if [[ "$BENCHMARK" == "libero" ]]; then
    benchmark_env+=("LIBERO_CONFIG_PATH=$LIBERO_ROOT/.libero_config")
  fi

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
    OMP_NUM_THREADS="$TRAIN_OMP_NUM_THREADS" \
    MKL_NUM_THREADS="$TRAIN_MKL_NUM_THREADS" \
    OPENBLAS_NUM_THREADS="$TRAIN_OPENBLAS_NUM_THREADS" \
    NUMEXPR_NUM_THREADS="$TRAIN_NUMEXPR_NUM_THREADS" \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    HF_HUB_OFFLINE=1 \
    TORCH_HOME="/mnt/data_nas/ykj_jepa_policy/checkpoints/torchvision" \
    WANDB_MODE=online \
    WANDB_ENTITY="$WANDB_ENTITY" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_NAME="$run_name" \
    WANDB_RUN_GROUP="${task}_joint_crop90" \
    WANDB_DIR="$WANDB_ROOT" \
    "${benchmark_env[@]}" \
    PYTHONPATH="$RUN_PYTHONPATH" \
    NUMBA_CACHE_DIR="$run_cache/numba" \
    MPLCONFIGDIR="$run_cache/matplotlib" \
    XDG_CACHE_HOME="$run_cache/xdg" \
    "$PYTHON_BIN" "$TRAIN_PY" "${COMMAND_ARGS[@]}" \
    >"$launcher_log" 2>&1 &

  local pid=$!
  PIDS+=("$pid")
  GPUS+=("$gpu")
  LAUNCHED_RUN_NAMES+=("$run_name")
  LAUNCHER_LOGS+=("$launcher_log")
  printf '%s\t%s\t%s\t%s\n' "$pid" "$gpu" "$run_name" "$launcher_log" | tee -a "$MANIFEST"

  LAUNCHED=$((LAUNCHED + 1))
  if (( LAUNCHED < TOTAL_SELECTED && START_GAP > 0 )); then
    sleep "$START_GAP"
  fi
}

cd "$REPO" || fail "cannot enter repository: $REPO"
for (( run_index=START_RUN_INDEX; run_index<ALL_RUNS; run_index++ )); do
  launch_run \
    "$((GPU_OFFSET + run_index))" \
    "${RUN_TASKS[$run_index]}" \
    "${RUN_VARIANTS[$run_index]}" \
    "${RUN_RATIOS[$run_index]}" \
    "${RUN_FUTURE_STEPS[$run_index]}" \
    "${RUN_NAMES[$run_index]}"
done

ANY_FAILED=0
for index in "${!PIDS[@]}"; do
  pid="${PIDS[$index]}"
  if wait "$pid"; then
    status=0
  else
    status=$?
    ANY_FAILED=1
  fi
  printf '%s\t%s\t%s\t%s\n' \
    "$pid" "${GPUS[$index]}" "${LAUNCHED_RUN_NAMES[$index]}" "$status" | tee -a "$STATUS_FILE"
done

if (( ANY_FAILED != 0 )); then
  echo "One or more experiments failed. See $STATUS_FILE and per-run logs." >&2
  exit 1
fi

echo "All $TOTAL_SELECTED selected $BENCHMARK experiments completed successfully."
