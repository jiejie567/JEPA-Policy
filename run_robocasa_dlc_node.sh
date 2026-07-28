#!/usr/bin/env bash
set -euo pipefail

NODE_INDEX="${1:-}"
[[ "$NODE_INDEX" == "1" || "$NODE_INDEX" == "2" || "$NODE_INDEX" == "3" ]] || {
  echo "Usage: $0 {1|2|3}" >&2
  exit 2
}

REPO="/mnt/data_nas/ykj_jepa_policy/code/JEPA-Policy"
TRAIN="$REPO/tools/run_robocasa_train.sh"
PYTHON="$REPO/tools/run_robocasa_python.sh"
ARRAY_CACHE_TOOL="$REPO/tools/prepare_robocasa_array_cache.py"
ROLLOUT_BENCH_TOOL="$REPO/tools/bench_robocasa_rollout_pool.py"
LOG_ROOT="$REPO/logs/robocasa"
START_GAP="${START_GAP:-180}"
RUN_TAG="${RUN_TAG:-}"
TAG_COMPONENT="${RUN_TAG:+_$RUN_TAG}"
CACHE_ROOT="${CACHE_ROOT:-/dev/shm/jepa_policy_robocasa_node${NODE_INDEX}${TAG_COMPONENT}}"
DRY_RUN="${DRY_RUN:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
# Keep the original 300k-step optimization protocol. A shorter comparison can
# still be requested explicitly at launch time, e.g. GRADIENT_STEPS=50000.
GRADIENT_STEPS="${GRADIENT_STEPS:-300000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-12}"
ARRAY_CACHE_WORKERS="${ARRAY_CACHE_WORKERS:-${FAST_CACHE_WORKERS:-32}}"
LOG_FREQ="${LOG_FREQ:-100}"
EVAL_FREQ="${EVAL_FREQ:-10000}"
VALIDATION_FREQ="${VALIDATION_FREQ:-10000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
ROBOCASA_WANDB_MODE="${ROBOCASA_WANDB_MODE:-online}"
SYNC_WANDB="${SYNC_WANDB:-0}"
EVAL_WORKERS="${EVAL_WORKERS:-10}"
EVAL_EPISODES=40
AUTO_RESUME="${AUTO_RESUME:-1}"
RESUME_EXISTING="${RESUME_EXISTING:-1}"

TASKS=()
VARIANTS=()
SEEDS=()

# Assign the complete 3 tasks x 3 seeds x 2 variants grid round-robin across
# three nodes. Each node gets exactly six runs and three of each variant.
all_tasks=(
  steam_in_microwave_robocasa_image
  store_leftovers_in_bowl_robocasa_image
  load_dishwasher_robocasa_image
)
run_index=0
for task in "${all_tasks[@]}"; do
  for seed in 41 42 43; do
    for variant in baseline future4_ratio010; do
      assigned_node=$((run_index % 3 + 1))
      if (( assigned_node == NODE_INDEX )); then
        TASKS+=("$task"); VARIANTS+=("$variant"); SEEDS+=("$seed")
      fi
      run_index=$((run_index + 1))
    done
  done
done

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

source_dataset_for() {
  case "$1" in
    steam_in_microwave_robocasa_image)
      echo "$REPO/datasets/robocasa/v1.0/target/composite/SteamInMicrowave/20250814/lerobot"
      ;;
    store_leftovers_in_bowl_robocasa_image)
      echo "$REPO/datasets/robocasa/v1.0/target/composite/StoreLeftoversInBowl/20250813/lerobot"
      ;;
    load_dishwasher_robocasa_image)
      echo "$REPO/datasets/robocasa/v1.0/target/composite/LoadDishwasher/20250811/lerobot"
      ;;
    *) fail "unknown RoboCasa task: $1" ;;
  esac
}

local_dataset_for() {
  echo "$CACHE_ROOT/datasets/$1/lerobot"
}

[[ -x "$TRAIN" && -x "$PYTHON" && -f "$ARRAY_CACHE_TOOL" &&
  -f "$ROLLOUT_BENCH_TOOL" ]] ||
  fail "missing RoboCasa launcher, array-cache tool, or rollout benchmark"
[[ "$DRY_RUN" =~ ^[01]$ && "$PREFLIGHT_ONLY" =~ ^[01]$ &&
  "$AUTO_RESUME" =~ ^[01]$ && "$RESUME_EXISTING" =~ ^[01]$ ]] ||
  fail "DRY_RUN, PREFLIGHT_ONLY, AUTO_RESUME, and RESUME_EXISTING must be 0 or 1"
[[ "$START_GAP" =~ ^[0-9]+$ ]] || fail "START_GAP must be non-negative"
[[ -z "$RUN_TAG" || "$RUN_TAG" =~ ^[A-Za-z0-9._-]+$ ]] ||
  fail "RUN_TAG contains unsupported characters"
[[ "$GRADIENT_STEPS" =~ ^[1-9][0-9]*$ ]] ||
  fail "GRADIENT_STEPS must be positive"
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] ||
  fail "BATCH_SIZE must be positive"
[[ "$DATALOADER_WORKERS" =~ ^[1-9][0-9]*$ ]] ||
  fail "DATALOADER_WORKERS must be positive"
[[ "$ARRAY_CACHE_WORKERS" =~ ^[1-9][0-9]*$ ]] ||
  fail "ARRAY_CACHE_WORKERS must be positive"
[[ "$LOG_FREQ" =~ ^[1-9][0-9]*$ && "$EVAL_FREQ" =~ ^[1-9][0-9]*$ &&
  "$VALIDATION_FREQ" =~ ^[1-9][0-9]*$ && "$SAVE_FREQ" =~ ^[1-9][0-9]*$ ]] ||
  fail "LOG_FREQ, EVAL_FREQ, VALIDATION_FREQ, and SAVE_FREQ must be positive"
[[ "$SYNC_WANDB" =~ ^[01]$ ]] || fail "SYNC_WANDB must be 0 or 1"
[[ "$ROBOCASA_WANDB_MODE" == "online" ||
  "$ROBOCASA_WANDB_MODE" == "offline" ]] ||
  fail "ROBOCASA_WANDB_MODE must be online or offline"
(( SYNC_WANDB == 0 || ROBOCASA_WANDB_MODE == "offline" )) ||
  fail "SYNC_WANDB is only valid with ROBOCASA_WANDB_MODE=offline"
[[ "$EVAL_WORKERS" =~ ^[1-9][0-9]*$ ]] ||
  fail "EVAL_WORKERS must be positive"
(( ${#TASKS[@]} == 6 )) || fail "each node must contain exactly 6 runs"
(( EVAL_EPISODES % EVAL_WORKERS == 0 )) || fail "eval episodes must divide workers"

mkdir -p "$LOG_ROOT" "$CACHE_ROOT/configs"
export ROBOCASA_USE_PPU_TORCH=1

RUN_NAMES=()
RUN_ARGS_FILES=()
echo "RoboCasa node $NODE_INDEX: 6 runs, batch_size=$BATCH_SIZE, rollout_workers=$EVAL_WORKERS, eval_freq=$EVAL_FREQ, auto_resume=$AUTO_RESUME, wandb_mode=$ROBOCASA_WANDB_MODE"
for i in "${!TASKS[@]}"; do
  task="${TASKS[$i]}"
  variant="${VARIANTS[$i]}"
  seed="${SEEDS[$i]}"
  run_name="${task}_${variant}_seed${seed}"
  [[ -z "$RUN_TAG" ]] || run_name="${run_name}_${RUN_TAG}"
  config_name="exps/robocasa_mip_${variant}"
  dataset_path="$(local_dataset_for "$task")"
  args=(
    "--config-name" "$config_name"
    "task=$task"
    "task.dataset_path=$dataset_path"
    "optimization.seed=$seed"
    "optimization.gradient_steps=$GRADIENT_STEPS"
    "optimization.batch_size=$BATCH_SIZE"
    "optimization.dataloader_num_workers=$DATALOADER_WORKERS"
    "optimization.dataloader_persistent_workers=true"
    "optimization.device=cuda"
    "optimization.auto_resume=$([[ "$AUTO_RESUME" == "1" ]] && echo true || echo false)"
    "eval.parallel_rollout=true"
    "eval.parallel_rollout_workers=$EVAL_WORKERS"
    "eval.persistent_workers=true"
    "eval.rollout_seed=12345"
    "eval.worker_timeout_seconds=3600"
    "log.wandb_mode=$ROBOCASA_WANDB_MODE"
    "log.entity=jepa-policy"
    "log.project=robocasa"
    "log.group=${task}_formal"
    "log.exp_name=$run_name"
    "log.log_dir=$LOG_ROOT/$run_name"
    "log.log_freq=$LOG_FREQ"
    "log.eval_freq=$EVAL_FREQ"
    "log.save_freq=$SAVE_FREQ"
    "log.validation_freq=$VALIDATION_FREQ"
    "log.eval_episodes=$EVAL_EPISODES"
    "log.save_video=false"
  )
  if [[ "$variant" == "future4_ratio010" ]]; then
    args+=(
      "task.future_state_enabled=true"
      "task.future_state_steps=4"
      "task.future_state_steps_list=[4]"
      "optimization.use_future_embed_loss=true"
      "optimization.future_embed_loss_mode=mip_two_step"
      "optimization.future_joint_mode=true"
      "optimization.future_state_loss_mode=ratio"
      "optimization.future_state_loss_ratio=0.1"
      "network.n_future_tokens=1"
    )
  fi

  EXPECTED_TASK="$task" EXPECTED_VARIANT="$variant" EXPECTED_SEED="$seed" \
  EXPECTED_DATASET="$dataset_path" EXPECTED_GRADIENT_STEPS="$GRADIENT_STEPS" \
  EXPECTED_BATCH_SIZE="$BATCH_SIZE" EXPECTED_EVAL_WORKERS="$EVAL_WORKERS" \
  EXPECTED_DATALOADER_WORKERS="$DATALOADER_WORKERS" \
  EXPECTED_WANDB_MODE="$ROBOCASA_WANDB_MODE" \
  "$PYTHON" - "${args[@]}" >"$CACHE_ROOT/configs/$run_name.yaml" <<'PY'
import os, sys
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
with initialize_config_dir(
    version_base=None,
    config_dir="/mnt/data_nas/ykj_jepa_policy/code/JEPA-Policy/examples/configs",
):
    cfg = compose(config_name=sys.argv[2], overrides=sys.argv[3:])
c = OmegaConf.to_container(cfg, resolve=True)
t, n, o, e, l = c["task"], c["network"], c["optimization"], c["eval"], c["log"]
assert t["env_type"] == "robocasa"
assert o["seed"] == int(os.environ["EXPECTED_SEED"])
assert t["dataset_path"] == os.environ["EXPECTED_DATASET"]
assert o["gradient_steps"] == int(os.environ["EXPECTED_GRADIENT_STEPS"])
assert o["batch_size"] == int(os.environ["EXPECTED_BATCH_SIZE"])
assert o["dataloader_num_workers"] == int(os.environ["EXPECTED_DATALOADER_WORKERS"])
assert e["parallel_rollout"]
assert e["parallel_rollout_workers"] == int(os.environ["EXPECTED_EVAL_WORKERS"])
assert l["eval_episodes"] == 40 and l["project"] == "robocasa"
assert l["wandb_mode"] == os.environ["EXPECTED_WANDB_MODE"]
if os.environ["EXPECTED_VARIANT"] == "baseline":
    assert not t["future_state_enabled"] and n["n_future_tokens"] == 0
    assert not o["use_future_embed_loss"] and not o["future_joint_mode"]
else:
    assert t["future_state_enabled"] and t["future_state_steps_list"] == [4]
    assert n["n_future_tokens"] == 1 and o["use_future_embed_loss"]
    assert o["future_embed_loss_mode"] == "mip_two_step"
    assert o["future_joint_mode"] and o["future_state_loss_mode"] == "ratio"
    assert o["future_state_loss_ratio"] == 0.1
print(OmegaConf.to_yaml(cfg, resolve=True), end="")
PY
  args_file="$CACHE_ROOT/configs/$run_name.args"
  printf '%s\n' "${args[@]}" >"$args_file"
  RUN_NAMES+=("$run_name")
  RUN_ARGS_FILES+=("$args_file")
  printf 'PLAN gpu=%d task=%s variant=%s seed=%s\n' \
    "$i" "$task" "$variant" "$seed"
done

if (( DRY_RUN != 0 )); then
  echo "DRY_RUN_OK node=$NODE_INDEX"
  exit 0
fi

[[ -n "${WANDB_API_KEY:-}" ]] || fail "WANDB_API_KEY is not injected"
hardware="$("$PYTHON" - <<'PY'
import torch
assert torch.__version__.startswith("2.6.0"), torch.__version__
assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 6, torch.cuda.device_count()
print(f"torch={torch.__version__} devices={torch.cuda.device_count()}")
PY
)" || fail "PPU Torch preflight failed"
echo "HARDWARE_OK $hardware"

[[ -d /dev/shm && -w /dev/shm ]] ||
  fail "/dev/shm is unavailable for the shared RoboCasa array cache"
available_shm_kib="$(df --output=avail -k /dev/shm | tail -n 1 | tr -d ' ')"
required_shm_kib=$((220 * 1024 * 1024))
(( available_shm_kib >= required_shm_kib )) ||
  fail "RoboCasa array cache needs at least 220 GiB free in /dev/shm"

echo "===== Build node-shared uint8 array dataset caches ====="
for task in "${all_tasks[@]}"; do
  source_dataset="$(source_dataset_for "$task")"
  local_dataset="$(local_dataset_for "$task")"
  ROBOCASA_USE_PPU_TORCH=0 "$PYTHON" "$ARRAY_CACHE_TOOL" \
    --source "$source_dataset" \
    --destination "$local_dataset" \
    --workers "$ARRAY_CACHE_WORKERS" \
    --height 128 \
    --width 128 \
    --force
done

ROBOCASA_BENCH_DATASET="$(local_dataset_for steam_in_microwave_robocasa_image)" \
"$PYTHON" - <<'PY' || fail "array-cache dataset benchmark failed"
import os
import statistics
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from mip.datasets.robot_dataset import make_dataset

with initialize_config_dir(
    version_base=None,
    config_dir=str(Path("examples/configs").resolve()),
):
    cfg = compose(
        config_name="exps/robocasa_mip_baseline",
        overrides=[
            "task=steam_in_microwave_robocasa_image",
            f"task.dataset_path={os.environ['ROBOCASA_BENCH_DATASET']}",
        ],
    )
dataset = make_dataset(cfg.task)
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    num_workers=12,
    persistent_workers=True,
    pin_memory=True,
)
iterator = iter(loader)
waits = []
for _ in range(41):
    started = time.perf_counter()
    next(iterator)
    waits.append(time.perf_counter() - started)
steady = waits[5:]
mean_seconds = statistics.fmean(steady)
print(
    f"ARRAY_CACHE_BENCHMARK batch_size=256 workers=12 "
    f"batches={len(steady)} mean_wait_seconds={mean_seconds:.6f} "
    f"median_wait_seconds={statistics.median(steady):.6f} "
    f"max_wait_seconds={max(steady):.6f}"
)
assert mean_seconds < 0.30, mean_seconds
PY

# Test the exact visibility rule used by a nonzero training slot. Physical GPU
# 5 must become the process-local cuda:0; the training code never receives a
# physical device index.
CUDA_VISIBLE_DEVICES=5 "$PYTHON" - <<'PY' || fail "GPU remapping preflight failed"
import os, torch
assert os.environ["CUDA_VISIBLE_DEVICES"] == "5"
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
x = torch.ones(4, device="cuda:0")
assert x.sum().item() == 4
print("GPU_MAPPING_OK physical=5 local=0", torch.cuda.get_device_name(0))
PY

env -u MUJOCO_EGL_DEVICE_ID \
  CUDA_VISIBLE_DEVICES=5 \
  JEPA_POLICY_EGL_DEVICE_ID=0 \
  TASK_NAME=steam_in_microwave_robocasa_image \
  "$PYTHON" - <<'PY' || fail "nonzero-GPU RoboCasa EGL reset failed"
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
        overrides=[f"task={os.environ['TASK_NAME']}", "task.num_envs=1"],
    )
env = make_vec_env(cfg.task, seed=12345)
obs, _ = env.reset()
assert obs["agentview_left_image"].shape[-3:] == (3, 128, 128)
env.close()
print("NONZERO_GPU_EGL_OK physical=5 local_cuda=0 local_egl=0")
PY

env -u MUJOCO_EGL_DEVICE_ID \
  CUDA_VISIBLE_DEVICES=5 \
  JEPA_POLICY_EGL_DEVICE_ID=0 \
  ROBOCASA_USE_PPU_TORCH=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  "$PYTHON" "$ROLLOUT_BENCH_TOOL" \
    --task steam_in_microwave_robocasa_image \
    --workers "$EVAL_WORKERS" \
    --cycles 1 ||
  fail "${EVAL_WORKERS}-worker RoboCasa rollout pool preflight failed"

"$PYTHON" "$REPO/tools/check_robocasa_setup.py" || fail "RoboCasa setup check failed"
if (( PREFLIGHT_ONLY != 0 )); then
  echo "PREFLIGHT_ONLY_OK node=$NODE_INDEX"
  exit 0
fi

for run_name in "${RUN_NAMES[@]}"; do
  (( RESUME_EXISTING != 0 )) || [[ ! -e "$LOG_ROOT/$run_name" ]] ||
    fail "log directory already exists: $LOG_ROOT/$run_name"
  if (( AUTO_RESUME != 0 )); then
    [[ -f "$LOG_ROOT/$run_name/models/model_latest.pt" ]] ||
      fail "resume checkpoint is missing: $LOG_ROOT/$run_name/models/model_latest.pt"
  fi
done

manifest="$LOG_ROOT/manifest_node${NODE_INDEX}${TAG_COMPONENT}.tsv"
status_file="$LOG_ROOT/status_node${NODE_INDEX}${TAG_COMPONENT}.tsv"
printf 'pid\tphysical_gpu\tlocal_cuda\trun_name\tlog\n' >"$manifest"
printf 'pid\tphysical_gpu\trun_name\texit_status\n' >"$status_file"
PIDS=()

for i in "${!RUN_NAMES[@]}"; do
  run_name="${RUN_NAMES[$i]}"
  run_dir="$LOG_ROOT/$run_name"
  run_cache="$CACHE_ROOT/$run_name"
  launcher_log="$LOG_ROOT/$run_name.launcher.log"
  mkdir -p "$run_dir" "$run_cache"/{numba,matplotlib,xdg,huggingface}
  mapfile -t args <"${RUN_ARGS_FILES[$i]}"
  cp "$CACHE_ROOT/configs/$run_name.yaml" "$run_dir/resolved_config.yaml"
  printf '%q ' "$TRAIN" "${args[@]}" >"$run_dir/command.sh"
  printf '\n' >>"$run_dir/command.sh"

  wandb_run_id=""
  if (( RESUME_EXISTING != 0 )); then
    latest_wandb_run="$(find "$run_dir/wandb" -maxdepth 1 -type d -name 'run-*' \
      -printf '%T@ %f\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2- || true)"
    if [[ -n "$latest_wandb_run" ]]; then
      wandb_run_id="${latest_wandb_run#run-}"
      wandb_run_id="${wandb_run_id#*-}"
    fi
  fi

  setsid env \
    -u MUJOCO_EGL_DEVICE_ID \
    CUDA_VISIBLE_DEVICES="$i" \
    JEPA_POLICY_EGL_DEVICE_ID=0 \
    ROBOCASA_USE_PPU_TORCH=1 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 \
    WANDB_MODE="$ROBOCASA_WANDB_MODE" \
    WANDB_ENTITY=jepa-policy WANDB_PROJECT=robocasa \
    WANDB_NAME="$run_name" WANDB_RUN_GROUP="${TASKS[$i]}_formal" \
    WANDB_RUN_ID="$wandb_run_id" \
    WANDB_DIR="$REPO/wandb" \
    ROBOCASA_CACHE_ROOT="$run_cache" \
    "$TRAIN" "${args[@]}" >>"$launcher_log" 2>&1 &
  pid=$!
  PIDS+=("$pid")
  printf '%s\t%s\t0\t%s\t%s\n' \
    "$pid" "$i" "$run_name" "$launcher_log" | tee -a "$manifest"
  if (( i + 1 < ${#RUN_NAMES[@]} )); then sleep "$START_GAP"; fi
done

failed=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then status=0; else status=$?; failed=1; fi
  printf '%s\t%s\t%s\t%s\n' \
    "${PIDS[$i]}" "$i" "${RUN_NAMES[$i]}" "$status" | tee -a "$status_file"
done

if (( SYNC_WANDB != 0 )); then
  echo "===== Serial W&B sync ====="
  for run_name in "${RUN_NAMES[@]}"; do
    run_dir="$LOG_ROOT/$run_name"
    if [[ ! -d "$run_dir/wandb" ]]; then
      echo "WARNING: no offline W&B directory found for $run_name" >&2
      continue
    fi
    while IFS= read -r offline_run; do
      synced=0
      for attempt in 1 2 3 4 5; do
        if WANDB_MODE=online "$PYTHON" -m wandb sync "$offline_run"; then
          synced=1
          break
        fi
        echo "W&B sync retry run=$run_name attempt=$attempt" >&2
        sleep $((attempt * 10))
      done
      if (( synced == 0 )); then
        echo "WARNING: W&B sync failed; offline run retained: $offline_run" >&2
      fi
    done < <(find "$run_dir/wandb" -maxdepth 1 -type d \
      -name 'offline-run-*' -print | sort)
  done
fi

(( failed == 0 )) || fail "one or more training runs failed"
echo "ALL_RUNS_COMPLETED node=$NODE_INDEX"
