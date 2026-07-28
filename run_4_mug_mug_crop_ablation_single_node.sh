#!/usr/bin/env bash

# Launch the four MugMug crop-ablation runs as independent single-GPU jobs on
# one DLC node. The current trainer has no DDP path, so GPUs 0, 4, 8, and 12
# each host one run; the other twelve GPUs are intentionally unused. This
# preserves the single-GPU optimization protocol used by run_14_ablation.
#
# Expected DLC allocation: 16 GPUs, 184 CPUs, and 1600 GB RAM. Resource
# allocation itself is configured when the DLC job is created, not by this
# script. Use DRY_RUN=1 to compose/validate configs and print commands without
# checking accelerators, creating logs, contacting W&B, or starting training.

set -u -o pipefail

REPO="/mnt/data_nas/ykj_jepa_policy/code/JEPA-Policy"
LIBERO_PY="${LIBERO_PY:-/mnt/data_nas/ykj_jepa_policy/venvs/libero/bin/python}"
LIBERO_ROOT="$REPO/third_party/LIBERO"
MUG_MUG_DATASET="/mnt/data_nas/ykj_jepa_policy/datasets/libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo.hdf5"

DRY_RUN="${DRY_RUN:-0}"
START_GAP="${START_GAP:-60}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-8}"
EVAL_WORKERS="${EVAL_WORKERS:-20}"
EVAL_EPISODES="${EVAL_EPISODES:-40}"
REQUIRED_GPU_COUNT="${REQUIRED_GPU_COUNT:-16}"
REQUIRED_CPU_COUNT="${REQUIRED_CPU_COUNT:-184}"
REQUIRED_MEMORY_GB="${REQUIRED_MEMORY_GB:-1600}"
LIBERO_PPU_TORCH_OVERLAY="${LIBERO_PPU_TORCH_OVERLAY:-auto}"

LOG_ROOT="$REPO/logs"
WANDB_ROOT="$REPO/wandb"
RUN_FILE_SUFFIX="${RUN_SUFFIX:+_$RUN_SUFFIX}"
MANIFEST="$LOG_ROOT/run_4_mug_mug_crop_ablation${RUN_FILE_SUFFIX}.tsv"
STATUS_FILE="$LOG_ROOT/run_4_mug_mug_crop_ablation_status${RUN_FILE_SUFFIX}.tsv"
CACHE_ROOT="/tmp/jepa_policy_mug_mug_crop_4"

EGL_RUNTIME_ROOT="/mnt/data_nas/ykj_jepa_policy/venvs/egl_noble_x86_64"
EGL_LIBRARY_DIR="$EGL_RUNTIME_ROOT/usr/lib/x86_64-linux-gnu"
EGL_DRI_DIR="$EGL_LIBRARY_DIR/dri"
EGL_VENDOR_JSON="$EGL_RUNTIME_ROOT/usr/share/glvnd/egl_vendor.d/50_mesa.json"

PPU_TORCH_SITE="/mnt/data_nas/ykj_jepa_policy/venvs/jepa_ppu/lib/python3.12/site-packages"
LIBERO_TORCH_OVERLAY_DIR="$CACHE_ROOT/libero_ppu_torch_overlay"
LIBERO_PYTHONPATH="$REPO:$LIBERO_ROOT"
EXPECTED_LIBERO_TORCH_VERSION="2.9.0+cu128"
EXPECTED_LIBERO_TORCHVISION_VERSION="0.24.0+cu128"
EXPECTED_LIBERO_TORCHAUDIO_VERSION="2.9.0+cu128"
EXPECTED_LIBERO_TRITON_VERSION="3.5.0"
EXPECTED_LIBERO_TORCH_ROOT="$(dirname "$(dirname "$LIBERO_PY")")"

CONFIG_NAMES=(
  exps/mug_mug_baseline_nocrop_seed42
  exps/mug_mug_baseline_crop116_temporal_seed42
  exps/mug_mug_future4_ratio010_nocrop_seed42
  exps/mug_mug_future4_ratio010_crop116_temporal_seed42
)
BASE_RUN_NAMES=(
  mug_mug_baseline_nocrop_seed42
  mug_mug_baseline_crop116_temporal_seed42
  mug_mug_future4_ratio010_nocrop_seed42
  mug_mug_future4_ratio010_crop116_temporal_seed42
)
GPU_IDS=(0 4 8 12)

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

[[ "$DRY_RUN" =~ ^(0|1)$ ]] || fail "DRY_RUN must be 0 or 1"
[[ "$START_GAP" =~ ^[0-9]+$ ]] || fail "START_GAP must be a non-negative integer"
[[ "$DATALOADER_WORKERS" =~ ^[0-9]+$ ]] || \
  fail "DATALOADER_WORKERS must be a non-negative integer"
[[ "$EVAL_WORKERS" =~ ^[1-9][0-9]*$ ]] || \
  fail "EVAL_WORKERS must be a positive integer"
[[ "$EVAL_EPISODES" =~ ^[1-9][0-9]*$ ]] || \
  fail "EVAL_EPISODES must be a positive integer"
(( EVAL_EPISODES % EVAL_WORKERS == 0 )) || \
  fail "EVAL_EPISODES ($EVAL_EPISODES) must be divisible by EVAL_WORKERS ($EVAL_WORKERS)"
[[ "$RUN_SUFFIX" =~ ^[A-Za-z0-9._-]*$ ]] || \
  fail "RUN_SUFFIX may contain only letters, digits, dots, underscores, and hyphens"
[[ "$REQUIRED_GPU_COUNT" =~ ^[1-9][0-9]*$ ]] || \
  fail "REQUIRED_GPU_COUNT must be a positive integer"
[[ "$REQUIRED_CPU_COUNT" =~ ^[1-9][0-9]*$ ]] || \
  fail "REQUIRED_CPU_COUNT must be a positive integer"
[[ "$REQUIRED_MEMORY_GB" =~ ^[1-9][0-9]*$ ]] || \
  fail "REQUIRED_MEMORY_GB must be a positive integer"
[[ "$LIBERO_PPU_TORCH_OVERLAY" =~ ^(auto|on|off)$ ]] || \
  fail "LIBERO_PPU_TORCH_OVERLAY must be auto, on, or off"
[[ -x "$LIBERO_PY" ]] || fail "LIBERO Python not found: $LIBERO_PY"

validate_configs() {
  DATALOADER_WORKERS="$DATALOADER_WORKERS" \
  EVAL_WORKERS="$EVAL_WORKERS" \
  EVAL_EPISODES="$EVAL_EPISODES" \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$LIBERO_PYTHONPATH" \
  "$LIBERO_PY" - "${CONFIG_NAMES[@]}" <<'PY'
import os
from pathlib import Path
import sys

from hydra import compose, initialize_config_dir
from mip.encoders import resolve_crop_shape


config_names = sys.argv[1:]
config_dir = str((Path.cwd() / "examples" / "configs").resolve())
data_workers = int(os.environ["DATALOADER_WORKERS"])
eval_workers = int(os.environ["EVAL_WORKERS"])
eval_episodes = int(os.environ["EVAL_EPISODES"])
overrides = [
    f"optimization.dataloader_num_workers={data_workers}",
    f"eval.parallel_rollout_workers={eval_workers}",
    f"log.eval_episodes={eval_episodes}",
]
with initialize_config_dir(version_base=None, config_dir=config_dir):
    configs = [compose(config_name=name, overrides=overrides) for name in config_names]

for index, (name, config) in enumerate(zip(config_names, configs, strict=True)):
    is_future = index >= 2
    is_crop = index % 2 == 1
    expected_crop = [116, 116] if is_crop else None
    resolved_crop = None
    if is_crop:
        resolved_crop = list(
            resolve_crop_shape(
                tuple(config.task.shape_meta.obs.agentview_rgb.shape),
                crop_shape=config.task.crop_shape,
                crop_ratio=config.task.crop_ratio,
                key="agentview_rgb",
            )
        )
    checks = {
        "seed": config.optimization.seed == 42,
        "batch_size": config.optimization.batch_size == 256,
        "gradient_steps": config.optimization.gradient_steps == 300000,
        "dataloader_workers": config.optimization.dataloader_num_workers == data_workers,
        "mip_time": config.optimization.t_two_step == 0.9,
        "future_mip_time": config.optimization.future_t_two_step == 0.9,
        "future_ratio": config.optimization.future_state_loss_ratio == 0.1,
        "obs_steps": config.task.obs_steps == 2,
        "horizon": config.task.horizon == 16,
        "act_steps": config.task.act_steps == 8,
        "future_steps": config.task.future_state_steps == 4,
        "crop_shape_auto": config.task.crop_shape is None,
        "crop_ratio": config.task.crop_ratio == (0.9 if is_crop else None),
        "resolved_crop": resolved_crop == expected_crop,
        "random_crop": config.task.random_crop is is_crop,
        "crop_mode": config.task.crop_mode == (
            "temporal_consistent" if is_crop else "none"
        ),
        "eval_crop_mode": config.task.eval_crop_mode == "center",
        "temporal_crop": config.task.temporal_consistent_crop is is_crop,
        "future_enabled": config.task.future_state_enabled is is_future,
        "future_joint": config.optimization.future_joint_mode is is_future,
        "future_loss": config.optimization.use_future_embed_loss is is_future,
        "future_tokens": config.network.n_future_tokens == int(is_future),
        "eval_freq": config.log.eval_freq == 10000,
        "eval_episodes": config.log.eval_episodes == eval_episodes,
        "rollout_workers": config.eval.parallel_rollout_workers == eval_workers,
        "rollout_seed": config.eval.rollout_seed == 12345,
        "wandb_entity": config.log.entity == "jepa-policy",
        "wandb_project": config.log.project == "jepa-policy",
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"{name}: invalid effective config fields: {failed}")
    print(
        f"CONFIG_OK name={name} seed=42 batch=256 steps=300000 "
        f"future={is_future} ratio={0.1 if is_future else 0.0} "
        f"crop={expected_crop} temporal={is_crop} eval=center-if-crop "
        f"eval_episodes={eval_episodes} rollout_workers={eval_workers} "
        "rollout_seed=12345"
    )
PY
}

cd "$REPO" || fail "cannot enter repository: $REPO"
validate_configs || fail "Hydra config validation failed"

if [[ "$DRY_RUN" == "1" ]]; then
  for index in "${!CONFIG_NAMES[@]}"; do
    run_name="$(qualified_run_name "${BASE_RUN_NAMES[$index]}")"
    printf 'DRY_RUN gpu=%s run=%s command=' "${GPU_IDS[$index]}" "$run_name"
    printf '%q ' \
      env \
      CUDA_VISIBLE_DEVICES="${GPU_IDS[$index]}" \
      JEPA_POLICY_EGL_DEVICE_ID=0 \
      WANDB_ENTITY=jepa-policy \
      WANDB_PROJECT=jepa-policy \
      WANDB_RUN_GROUP=mug_mug_crop_2x2 \
      "$LIBERO_PY" \
      "$REPO/examples/train_robomimic.py" \
      "--config-name=${CONFIG_NAMES[$index]}" \
      "optimization.dataloader_num_workers=$DATALOADER_WORKERS" \
      "eval.parallel_rollout_workers=$EVAL_WORKERS" \
      "log.eval_episodes=$EVAL_EPISODES" \
      "log.exp_name=$run_name" \
      "log.log_dir=$LOG_ROOT/$run_name"
    printf '\n'
  done
  echo "DRY_RUN complete; no training process was started."
  exit 0
fi

[[ -n "${WANDB_API_KEY:-}" ]] || \
  fail "WANDB_API_KEY is unavailable; inject it through a DLC Secret"
[[ -d "$LIBERO_ROOT" ]] || fail "LIBERO root not found: $LIBERO_ROOT"
[[ -f "$LIBERO_ROOT/.libero_config/config.yaml" ]] || \
  fail "LIBERO config not found: $LIBERO_ROOT/.libero_config/config.yaml"
[[ -f "$MUG_MUG_DATASET" ]] || fail "dataset not found: $MUG_MUG_DATASET"
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is unavailable"
command -v setsid >/dev/null 2>&1 || fail "setsid is unavailable"

GPU_LIST="$(nvidia-smi -L 2>/dev/null)" || fail "nvidia-smi could not enumerate GPUs"
GPU_COUNT=0
while IFS= read -r gpu_line; do
  [[ -n "$gpu_line" ]] && GPU_COUNT=$((GPU_COUNT + 1))
done <<<"$GPU_LIST"
(( GPU_COUNT >= REQUIRED_GPU_COUNT )) || \
  fail "expected at least $REQUIRED_GPU_COUNT GPUs; found $GPU_COUNT"

# GNU nproc honors OMP_NUM_THREADS / OMP_THREAD_LIMIT. The launch command sets
# OMP_NUM_THREADS=1 intentionally for each worker, but that must not make the
# node-level resource preflight believe the DLC has only one CPU.
CPU_COUNT="$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)"
(( CPU_COUNT >= REQUIRED_CPU_COUNT )) || \
  fail "expected at least $REQUIRED_CPU_COUNT CPUs; found $CPU_COUNT"
HOST_MEMORY_BYTES="$(awk '/MemTotal:/ {printf "%.0f", $2 * 1024}' /proc/meminfo)"
MEMORY_BYTES="$HOST_MEMORY_BYTES"
if [[ -r /sys/fs/cgroup/memory.max ]]; then
  read -r CGROUP_MEMORY_BYTES </sys/fs/cgroup/memory.max
  if [[ "$CGROUP_MEMORY_BYTES" =~ ^[0-9]+$ ]] && \
    (( CGROUP_MEMORY_BYTES < MEMORY_BYTES )); then
    MEMORY_BYTES="$CGROUP_MEMORY_BYTES"
  fi
elif [[ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]]; then
  read -r CGROUP_MEMORY_BYTES </sys/fs/cgroup/memory/memory.limit_in_bytes
  if [[ "$CGROUP_MEMORY_BYTES" =~ ^[0-9]+$ ]] && \
    (( CGROUP_MEMORY_BYTES < MEMORY_BYTES )); then
    MEMORY_BYTES="$CGROUP_MEMORY_BYTES"
  fi
fi
MEMORY_GB=$((MEMORY_BYTES / 1000000000))
(( MEMORY_GB >= REQUIRED_MEMORY_GB )) || \
  fail "expected at least $REQUIRED_MEMORY_GB GB RAM; found approximately $MEMORY_GB GB"
echo "DLC resources: gpus=$GPU_COUNT cpus=$CPU_COUNT memory_gb~=$MEMORY_GB"
echo "Training GPU assignment: 0,4,8,12 (one independent run per GPU)"

USE_LIBERO_PPU_TORCH=0
case "$LIBERO_PPU_TORCH_OVERLAY" in
  on)
    USE_LIBERO_PPU_TORCH=1
    ;;
  auto)
    [[ "$GPU_LIST" == *PPU* ]] && USE_LIBERO_PPU_TORCH=1
    ;;
esac

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

mkdir -p "$CACHE_ROOT"
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
  EXPECTED_LIBERO_TORCH_VERSION="2.6.0"
  EXPECTED_LIBERO_TORCHVISION_VERSION="0.21.0"
  EXPECTED_LIBERO_TORCHAUDIO_VERSION="2.6.0"
  EXPECTED_LIBERO_TRITON_VERSION="3.2.0"
  EXPECTED_LIBERO_TORCH_ROOT="$PPU_TORCH_SITE"
  echo "LIBERO PPU torch overlay enabled: $LIBERO_TORCH_OVERLAY_DIR"
fi

if ! env \
  -u VIRTUAL_ENV \
  -u CONDA_PREFIX \
  -u CONDA_DEFAULT_ENV \
  -u PYTHONHOME \
  -u PYTHONPATH \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 \
  EXPECTED_LIBERO_TORCH_VERSION="$EXPECTED_LIBERO_TORCH_VERSION" \
  EXPECTED_LIBERO_TORCHVISION_VERSION="$EXPECTED_LIBERO_TORCHVISION_VERSION" \
  EXPECTED_LIBERO_TORCHAUDIO_VERSION="$EXPECTED_LIBERO_TORCHAUDIO_VERSION" \
  EXPECTED_LIBERO_TRITON_VERSION="$EXPECTED_LIBERO_TRITON_VERSION" \
  EXPECTED_LIBERO_TORCH_ROOT="$EXPECTED_LIBERO_TORCH_ROOT" \
  EXPECTED_LIBERO_PREFIX="$(dirname "$(dirname "$LIBERO_PY")")" \
  EXPECTED_LIBERO_ROOT="$LIBERO_ROOT" \
  PYTHONPATH="$LIBERO_PYTHONPATH" \
  "$LIBERO_PY" - <<'PY'
from importlib import import_module, metadata
import os
from pathlib import Path
import sys


expected_prefix = Path(os.environ["EXPECTED_LIBERO_PREFIX"]).resolve()
expected_libero_root = Path(os.environ["EXPECTED_LIBERO_ROOT"]).resolve()
expected_torch_root = Path(os.environ["EXPECTED_LIBERO_TORCH_ROOT"]).resolve()
assert Path(sys.prefix).resolve() == expected_prefix, (sys.prefix, expected_prefix)

expected_versions = {
    "torch": os.environ["EXPECTED_LIBERO_TORCH_VERSION"],
    "torchvision": os.environ["EXPECTED_LIBERO_TORCHVISION_VERSION"],
    "torchaudio": os.environ["EXPECTED_LIBERO_TORCHAUDIO_VERSION"],
    "triton": os.environ["EXPECTED_LIBERO_TRITON_VERSION"],
    "numpy": "1.26.0",
    "mujoco": "3.10.0",
    "robosuite": "1.4.0",
    "robomimic": "0.4.0",
    "libero": "0.1.0",
    "wandb": "0.28.0",
}
for package, expected_version in expected_versions.items():
    distribution = metadata.distribution(package)
    assert distribution.version == expected_version, (
        package,
        distribution.version,
        expected_version,
    )
    distribution_root = Path(distribution.locate_file("")).resolve()
    if package in {"torch", "torchvision", "torchaudio", "triton"}:
        module_root = Path(import_module(package).__file__).resolve()
        assert module_root.is_relative_to(expected_torch_root), (
            package,
            module_root,
            expected_torch_root,
        )
    elif package == "libero":
        assert distribution_root.is_relative_to(expected_libero_root), (
            package,
            distribution_root,
            expected_libero_root,
        )
    else:
        assert distribution_root.is_relative_to(expected_prefix), (
            package,
            distribution_root,
            expected_prefix,
        )

print(
    "LIBERO environment isolation passed: "
    f"python={sys.executable}, prefix={sys.prefix}, torch_root={expected_torch_root}"
)
PY
then
  fail "LIBERO Python environment isolation/version preflight failed"
fi

for required_file in \
  "$EGL_LIBRARY_DIR/libEGL.so.0" \
  "$EGL_LIBRARY_DIR/libEGL.so.1" \
  "$EGL_LIBRARY_DIR/libEGL_mesa.so.0" \
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

for gpu in "${GPU_IDS[@]}"; do
  if ! env \
    -u VIRTUAL_ENV \
    -u CONDA_PREFIX \
    -u CONDA_DEFAULT_ENV \
    -u PYTHONHOME \
    -u PYTHONPATH \
    CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    PYTHONPATH="$LIBERO_PYTHONPATH" \
    "$LIBERO_PY" -c \
      'import torch; assert torch.cuda.is_available(); torch.zeros(1, device="cuda"); torch.cuda.synchronize()'; then
    fail "CUDA preflight failed on physical GPU $gpu"
  fi
done

mkdir -p "$LOG_ROOT" "$WANDB_ROOT"
for base_run_name in "${BASE_RUN_NAMES[@]}"; do
  run_name="$(qualified_run_name "$base_run_name")"
  [[ ! -e "$LOG_ROOT/$run_name" ]] || fail "log directory already exists: $LOG_ROOT/$run_name"
  [[ ! -e "$LOG_ROOT/$run_name.log" ]] || fail "log file already exists: $LOG_ROOT/$run_name.log"
done

printf 'pid\tgpu\trun_name\tlog_path\n' >"$MANIFEST"
printf 'pid\tgpu\trun_name\texit_status\n' >"$STATUS_FILE"

PIDS=()
RUN_NAMES=()
LOG_PATHS=()

terminate_runs() {
  echo "Stopping launched MugMug runs..." >&2
  for pid in "${PIDS[@]}"; do
    kill -TERM -- "-$pid" 2>/dev/null || true
  done
}
trap terminate_runs INT TERM

for index in "${!CONFIG_NAMES[@]}"; do
  gpu="${GPU_IDS[$index]}"
  run_name="$(qualified_run_name "${BASE_RUN_NAMES[$index]}")"
  log_path="$LOG_ROOT/$run_name.log"
  run_cache="$CACHE_ROOT/$run_name"
  mkdir -p \
    "$run_cache/numba" \
    "$run_cache/matplotlib" \
    "$run_cache/xdg" \
    "$run_cache/torch_extensions" \
    "$run_cache/triton"

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
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    HF_HUB_OFFLINE=1 \
    TORCH_HOME="/mnt/data_nas/ykj_jepa_policy/checkpoints/torchvision" \
    WANDB_MODE=online \
    WANDB_ENTITY=jepa-policy \
    WANDB_PROJECT=jepa-policy \
    WANDB_NAME="$run_name" \
    WANDB_RUN_GROUP=mug_mug_crop_2x2 \
    WANDB_DIR="$WANDB_ROOT" \
    LIBERO_CONFIG_PATH="$LIBERO_ROOT/.libero_config" \
    PYTHONPATH="$LIBERO_PYTHONPATH" \
    NUMBA_CACHE_DIR="$run_cache/numba" \
    MPLCONFIGDIR="$run_cache/matplotlib" \
    XDG_CACHE_HOME="$run_cache/xdg" \
    TORCH_EXTENSIONS_DIR="$run_cache/torch_extensions" \
    TRITON_CACHE_DIR="$run_cache/triton" \
    "$LIBERO_PY" "$REPO/examples/train_robomimic.py" \
    "--config-name=${CONFIG_NAMES[$index]}" \
    "optimization.dataloader_num_workers=$DATALOADER_WORKERS" \
    "eval.parallel_rollout_workers=$EVAL_WORKERS" \
    "log.eval_episodes=$EVAL_EPISODES" \
    "log.exp_name=$run_name" \
    "log.log_dir=$LOG_ROOT/$run_name" \
    >"$log_path" 2>&1 &

  pid=$!
  PIDS+=("$pid")
  RUN_NAMES+=("$run_name")
  LOG_PATHS+=("$log_path")
  printf '%s\t%s\t%s\t%s\n' "$pid" "$gpu" "$run_name" "$log_path" | tee -a "$MANIFEST"

  if (( index + 1 < ${#CONFIG_NAMES[@]} && START_GAP > 0 )); then
    sleep "$START_GAP"
  fi
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
    "$pid" "${GPU_IDS[$index]}" "${RUN_NAMES[$index]}" "$status" | tee -a "$STATUS_FILE"
done

trap - INT TERM
if (( ANY_FAILED != 0 )); then
  echo "One or more experiments failed. See $STATUS_FILE and per-run logs." >&2
  exit 1
fi

echo "All four MugMug crop-ablation runs completed successfully."
