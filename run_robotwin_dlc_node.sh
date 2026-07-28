#!/usr/bin/env bash
set -euo pipefail

NODE_INDEX="${1:-}"
[[ "$NODE_INDEX" == "1" || "$NODE_INDEX" == "2" ]] || {
  echo "Usage: $0 {1|2}" >&2
  exit 2
}

REPO="/mnt/data_nas/ykj_jepa_policy/code/JEPA-Policy"
SOURCE_ROBOTWIN_ROOT="$REPO/third_party/robotwin"
TRAIN="$REPO/tools/run_robotwin_train.sh"
PYTHON="$REPO/tools/run_robotwin_python.sh"
CHECK_TOOL="$REPO/tools/check_robotwin_setup.py"
ROLLOUT_BENCH_TOOL="$REPO/tools/bench_robotwin_rollout_pool.py"
SOURCE_DATASET_ROOT="$REPO/datasets/robotwin/cache"
LOG_ROOT="$REPO/logs/robotwin"
CACHE_ROOT="/dev/shm/jepa_policy_robotwin_node${NODE_INDEX}"
RUNTIME_ROBOTWIN_ROOT="$CACHE_ROOT/runtime/robotwin"
RUNTIME_VENV_ROOT="$CACHE_ROOT/runtime/venv"
RUNTIME_SYSTEM_LIB_ROOT="$CACHE_ROOT/runtime/system_libs"
LOCAL_DATASET_ROOT="$CACHE_ROOT/datasets"
RUNTIME_VENV_ARCHIVE="$REPO/artifacts/robotwin/robotwin_runtime_py312.tar.gz"
RUNTIME_VENV_SHA256="${ROBOTWIN_RUNTIME_VENV_SHA256:-7615b3c42d32af7ed092490539d6a89bc21912674bc3d9ad3e9cc5caf9e72039}"
RUNTIME_SYSTEM_LIB_ARCHIVE="$REPO/artifacts/robotwin/robotwin_system_libs_ubuntu24_x86_64.tar.gz"
RUNTIME_SYSTEM_LIB_SHA256="${ROBOTWIN_SYSTEM_LIB_SHA256:-1c368772abfa4d73431a17e8a4c2da1a91f248019f4212cf32b631e7a6cf0e6f}"

START_GAP="${START_GAP:-180}"
RUN_TAG="${RUN_TAG:-}"
DRY_RUN="${DRY_RUN:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
GRADIENT_STEPS="${GRADIENT_STEPS:-300000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-12}"
EVAL_WORKERS="${EVAL_WORKERS:-10}"
EVAL_EPISODES="${EVAL_EPISODES:-40}"
TRAIN_GPU_COUNT="${TRAIN_GPU_COUNT:-8}"
RUNS_PER_GPU="${RUNS_PER_GPU:-1}"
LOG_FREQ="${LOG_FREQ:-100}"
EVAL_FREQ="${EVAL_FREQ:-10000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
ROBOTWIN_WANDB_MODE="${ROBOTWIN_WANDB_MODE:-online}"

PINNED_COMMIT="c3ddfa8b97d5519efa828b075999bd0006778e5e"
TASKS=()
VARIANTS=()
SEEDS=()
ALL_TASKS=(
  stack_bowls_three_robotwin_image
  handover_block_robotwin_image
  put_object_cabinet_robotwin_image
)

# Every task-seed pair is split across nodes, so its baseline and Future run
# cannot accidentally share a process or GPU. Future load is balanced 4/5.
pair_index=0
for task in "${ALL_TASKS[@]}"; do
  for seed in 41 42 43; do
    if (( pair_index % 2 == 0 )); then
      node1_variant=baseline
      node2_variant=future4_ratio010
    else
      node1_variant=future4_ratio010
      node2_variant=baseline
    fi
    if [[ "$NODE_INDEX" == "1" ]]; then
      TASKS+=("$task"); VARIANTS+=("$node1_variant"); SEEDS+=("$seed")
    else
      TASKS+=("$task"); VARIANTS+=("$node2_variant"); SEEDS+=("$seed")
    fi
    pair_index=$((pair_index + 1))
  done
done

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

dataset_name_for() {
  case "$1" in
    stack_bowls_three_robotwin_image) echo "stack_bowls_three" ;;
    handover_block_robotwin_image) echo "handover_block" ;;
    put_object_cabinet_robotwin_image) echo "put_object_cabinet" ;;
    *) fail "unknown RoboTwin task: $1" ;;
  esac
}

[[ -x "$TRAIN" && -x "$PYTHON" && -x "$CHECK_TOOL" &&
  -x "$ROLLOUT_BENCH_TOOL" ]] || fail "missing RoboTwin executable"
[[ "$(git -C "$SOURCE_ROBOTWIN_ROOT" rev-parse HEAD)" == "$PINNED_COMMIT" ]] ||
  fail "RoboTwin source is not at the pinned commit"
git -C "$SOURCE_ROBOTWIN_ROOT" diff --check ||
  fail "RoboTwin policy-rollout patch has whitespace errors"
[[ "$DRY_RUN" =~ ^[01]$ && "$PREFLIGHT_ONLY" =~ ^[01]$ ]] ||
  fail "DRY_RUN and PREFLIGHT_ONLY must be 0 or 1"
[[ "$START_GAP" =~ ^[0-9]+$ ]] || fail "START_GAP must be non-negative"
for value in \
  "$GRADIENT_STEPS" "$BATCH_SIZE" "$DATALOADER_WORKERS" "$EVAL_WORKERS" \
  "$EVAL_EPISODES" "$TRAIN_GPU_COUNT" "$RUNS_PER_GPU" "$LOG_FREQ" \
  "$EVAL_FREQ" "$SAVE_FREQ"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail "numeric settings must be positive"
done
(( RUNS_PER_GPU <= 2 )) ||
  fail "RUNS_PER_GPU above 2 is not covered by the formal memory preflight"
(( EVAL_EPISODES % EVAL_WORKERS == 0 )) ||
  fail "EVAL_EPISODES must be divisible by EVAL_WORKERS"
[[ "$ROBOTWIN_WANDB_MODE" == "online" ||
  "$ROBOTWIN_WANDB_MODE" == "offline" ]] ||
  fail "ROBOTWIN_WANDB_MODE must be online or offline"
[[ "$RUN_TAG" =~ ^[A-Za-z0-9._-]*$ ]] ||
  fail "RUN_TAG contains unsupported characters"
(( ${#TASKS[@]} == 9 )) || fail "each node must own exactly nine runs"

mkdir -p "$LOG_ROOT" "$CACHE_ROOT/configs"
if (( DRY_RUN == 0 )); then
  [[ -f "$RUNTIME_VENV_ARCHIVE" ]] ||
    fail "missing packaged RoboTwin runtime: $RUNTIME_VENV_ARCHIVE"
  [[ -f "$RUNTIME_SYSTEM_LIB_ARCHIVE" ]] ||
    fail "missing packaged RoboTwin system libraries: $RUNTIME_SYSTEM_LIB_ARCHIVE"
  [[ "$RUNTIME_VENV_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
    fail "invalid packaged RoboTwin runtime SHA-256"
  [[ "$RUNTIME_SYSTEM_LIB_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
    fail "invalid packaged system-library SHA-256"
  actual_runtime_sha256="$(
    sha256sum "$RUNTIME_VENV_ARCHIVE" | awk '{print $1}'
  )"
  [[ "$actual_runtime_sha256" == "$RUNTIME_VENV_SHA256" ]] ||
    fail "packaged RoboTwin runtime SHA-256 mismatch"
  actual_system_lib_sha256="$(
    sha256sum "$RUNTIME_SYSTEM_LIB_ARCHIVE" | awk '{print $1}'
  )"
  [[ "$actual_system_lib_sha256" == "$RUNTIME_SYSTEM_LIB_SHA256" ]] ||
    fail "packaged RoboTwin system-library SHA-256 mismatch"
  mkdir -p "$CACHE_ROOT/runtime"
  tar -xzf "$RUNTIME_VENV_ARCHIVE" -C "$CACHE_ROOT/runtime"
  tar -xzf "$RUNTIME_SYSTEM_LIB_ARCHIVE" -C "$CACHE_ROOT/runtime"
  RUNTIME_DEPENDENCY_SITE="$RUNTIME_VENV_ROOT/lib/python3.12/site-packages"
  [[ -d "$RUNTIME_DEPENDENCY_SITE/sapien" ]] ||
    fail "packaged RoboTwin runtime did not contain SAPIEN"
  [[ -f "$RUNTIME_SYSTEM_LIB_ROOT/libX11.so.6" &&
    -f "$RUNTIME_SYSTEM_LIB_ROOT/libXext.so.6" &&
    -f "$RUNTIME_SYSTEM_LIB_ROOT/libEGL.so.1" &&
    -f "$RUNTIME_SYSTEM_LIB_ROOT/libGL.so.1" &&
    -f "$RUNTIME_SYSTEM_LIB_ROOT/libGLX.so.0" &&
    -f "$RUNTIME_SYSTEM_LIB_ROOT/libGLdispatch.so.0" &&
    -f "$RUNTIME_SYSTEM_LIB_ROOT/libglib-2.0.so.0" &&
    -f "$RUNTIME_SYSTEM_LIB_ROOT/libgthread-2.0.so.0" &&
    -f "$RUNTIME_SYSTEM_LIB_ROOT/libpcre2-8.so.0" &&
    -f "$RUNTIME_SYSTEM_LIB_ROOT/libvulkan.so.1" ]] ||
    fail "packaged RoboTwin system libraries are incomplete"
  export ROBOTWIN_PYTHON_BIN="$(command -v python3)"
  export ROBOTWIN_DEPENDENCY_SITE="$RUNTIME_DEPENDENCY_SITE"
  export ROBOTWIN_SYSTEM_LIB_DIR="$RUNTIME_SYSTEM_LIB_ROOT"
  export ROBOTWIN_USE_PPU_TORCH=0
else
  export ROBOTWIN_USE_PPU_TORCH="${ROBOTWIN_USE_PPU_TORCH:-1}"
fi

RUN_NAMES=()
RUN_ARGS_FILES=()
declare -A SEEN_RUNS=()
MAX_CONCURRENT_RUNS=$((TRAIN_GPU_COUNT * RUNS_PER_GPU))
echo "RoboTwin node $NODE_INDEX: 9 runs on $TRAIN_GPU_COUNT GPUs, runs_per_gpu=$RUNS_PER_GPU, max_concurrent=$MAX_CONCURRENT_RUNS, batch=$BATCH_SIZE, workers=$DATALOADER_WORKERS, rollout_workers=$EVAL_WORKERS"
for i in "${!TASKS[@]}"; do
  task="${TASKS[$i]}"
  variant="${VARIANTS[$i]}"
  seed="${SEEDS[$i]}"
  dataset_name="$(dataset_name_for "$task")"
  dataset_path="$LOCAL_DATASET_ROOT/$dataset_name"
  run_name="${dataset_name}_robotwin_image_${variant}_seed${seed}"
  [[ -z "$RUN_TAG" ]] || run_name="${run_name}_${RUN_TAG}"
  [[ -z "${SEEN_RUNS[$run_name]:-}" ]] || fail "duplicate run: $run_name"
  SEEN_RUNS[$run_name]=1
  args=(
    "--config-name" "exps/robotwin_mip_${variant}"
    "task=$task"
    "task.robotwin_root=$RUNTIME_ROBOTWIN_ROOT"
    "task.dataset_path=$dataset_path"
    "optimization.seed=$seed"
    "optimization.gradient_steps=$GRADIENT_STEPS"
    "optimization.batch_size=$BATCH_SIZE"
    "optimization.dataloader_num_workers=$DATALOADER_WORKERS"
    "optimization.dataloader_persistent_workers=true"
    "optimization.device=cuda"
    "optimization.auto_resume=true"
    "eval.parallel_rollout=true"
    "eval.parallel_rollout_workers=$EVAL_WORKERS"
    "eval.persistent_workers=true"
    "eval.rollout_seed=12345"
    "eval.worker_timeout_seconds=3600"
    "log.wandb_mode=$ROBOTWIN_WANDB_MODE"
    "log.entity=jepa-policy"
    "log.project=robotwin"
    "log.group=${dataset_name}_formal"
    "log.exp_name=$run_name"
    "log.log_dir=$LOG_ROOT/$run_name"
    "log.log_freq=$LOG_FREQ"
    "log.gradient_diagnostic_freq=1000"
    "log.validation_freq=$EVAL_FREQ"
    "log.eval_freq=$EVAL_FREQ"
    "log.eval_episodes=$EVAL_EPISODES"
    "log.save_freq=$SAVE_FREQ"
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
  EXPECTED_DATASET="$dataset_path" EXPECTED_ROOT="$RUNTIME_ROBOTWIN_ROOT" \
  EXPECTED_STEPS="$GRADIENT_STEPS" EXPECTED_BATCH="$BATCH_SIZE" \
  EXPECTED_DATA_WORKERS="$DATALOADER_WORKERS" \
  EXPECTED_EVAL_WORKERS="$EVAL_WORKERS" EXPECTED_EVAL_EPISODES="$EVAL_EPISODES" \
  EXPECTED_WANDB_MODE="$ROBOTWIN_WANDB_MODE" \
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
assert t["env_type"] == "robotwin"
assert t["robotwin_root"] == os.environ["EXPECTED_ROOT"]
assert t["dataset_path"] == os.environ["EXPECTED_DATASET"]
assert o["seed"] == int(os.environ["EXPECTED_SEED"])
assert o["gradient_steps"] == int(os.environ["EXPECTED_STEPS"])
assert o["batch_size"] == int(os.environ["EXPECTED_BATCH"])
assert o["dataloader_num_workers"] == int(os.environ["EXPECTED_DATA_WORKERS"])
assert e["parallel_rollout"]
assert e["parallel_rollout_workers"] == int(os.environ["EXPECTED_EVAL_WORKERS"])
assert l["eval_episodes"] == int(os.environ["EXPECTED_EVAL_EPISODES"])
assert l["entity"] == "jepa-policy" and l["project"] == "robotwin"
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
  wave=$((i / MAX_CONCURRENT_RUNS))
  slot=$((i % MAX_CONCURRENT_RUNS))
  physical_gpu=$((slot % TRAIN_GPU_COUNT))
  printf 'PLAN wave=%d gpu=%d task=%s variant=%s seed=%s run=%s\n' \
    "$wave" "$physical_gpu" "$task" "$variant" "$seed" "$run_name"
done

if (( DRY_RUN != 0 )); then
  echo "DRY_RUN_OK node=$NODE_INDEX"
  exit 0
fi

[[ -n "${WANDB_API_KEY:-}" ]] || fail "WANDB_API_KEY is not injected"
hardware="$(EXPECTED_GPU_COUNT="$TRAIN_GPU_COUNT" "$PYTHON" - <<'PY'
import os
import torch
assert torch.__version__.startswith("2.6.0"), torch.__version__
assert torch.cuda.is_available()
assert torch.cuda.device_count() >= int(os.environ["EXPECTED_GPU_COUNT"])
x = torch.ones(4, device="cuda:0")
assert x.sum().item() == 4
print(f"torch={torch.__version__} devices={torch.cuda.device_count()}")
PY
)" || fail "GPU Torch preflight failed"
echo "HARDWARE_OK $hardware"

[[ -d /dev/shm && -w /dev/shm ]] ||
  fail "/dev/shm is unavailable"
available_shm_kib="$(df --output=avail -k /dev/shm | tail -n 1 | tr -d ' ')"
required_shm_kib=$((40 * 1024 * 1024))
(( available_shm_kib >= required_shm_kib )) ||
  fail "RoboTwin runtime needs at least 40 GiB free in /dev/shm"

echo "===== Build node-local source, asset, and dataset snapshot ====="
mkdir -p "$RUNTIME_ROBOTWIN_ROOT" \
  "$RUNTIME_ROBOTWIN_ROOT/assets/embodiments" \
  "$RUNTIME_ROBOTWIN_ROOT/assets/objects" \
  "$LOCAL_DATASET_ROOT"
tar -C "$SOURCE_ROBOTWIN_ROOT" --exclude='./assets' -cf - . |
  tar -C "$RUNTIME_ROBOTWIN_ROOT" -xf -
cp -a "$SOURCE_ROBOTWIN_ROOT/assets/embodiments/aloha-agilex" \
  "$RUNTIME_ROBOTWIN_ROOT/assets/embodiments/"
OBJECTS=(
  002_bowl 036_cabinet 047_mouse 048_stapler 057_toycar
  073_rubikscube 075_bread 077_phone 081_playingcards
  107_soap 112_tea-box 113_coffee-box
)
for object_name in "${OBJECTS[@]}"; do
  [[ -d "$SOURCE_ROBOTWIN_ROOT/assets/objects/$object_name" ]] ||
    fail "missing source object asset: $object_name"
  cp -a "$SOURCE_ROBOTWIN_ROOT/assets/objects/$object_name" \
    "$RUNTIME_ROBOTWIN_ROOT/assets/objects/"
done
for task in "${ALL_TASKS[@]}"; do
  dataset_name="$(dataset_name_for "$task")"
  [[ -f "$SOURCE_DATASET_ROOT/$dataset_name/.jepa_robotwin_cache.json" ]] ||
    fail "missing source array cache: $dataset_name"
  cp -a "$SOURCE_DATASET_ROOT/$dataset_name" "$LOCAL_DATASET_ROOT/"
done

export ROBOTWIN_SOURCE_ROOT="$RUNTIME_ROBOTWIN_ROOT"
export ROBOTWIN_DATASET_ROOT="$LOCAL_DATASET_ROOT"
"$PYTHON" "$CHECK_TOOL" --require-gpu --env-smoke ||
  fail "RoboTwin source/data/environment preflight failed"

ROBOTWIN_BENCH_DATASET="$LOCAL_DATASET_ROOT/handover_block" \
"$PYTHON" - <<'PY' || fail "RoboTwin array-cache benchmark failed"
import os, statistics, time
from pathlib import Path
import torch
from hydra import compose, initialize_config_dir
from mip.datasets.robot_dataset import make_dataset
with initialize_config_dir(
    version_base=None,
    config_dir=str(Path("examples/configs").resolve()),
):
    cfg = compose(
        config_name="exps/robotwin_mip_baseline",
        overrides=[
            "task=handover_block_robotwin_image",
            f"task.dataset_path={os.environ['ROBOTWIN_BENCH_DATASET']}",
        ],
    )
dataset = make_dataset(cfg.task)
loader = torch.utils.data.DataLoader(
    dataset, batch_size=256, shuffle=True, num_workers=12,
    persistent_workers=True, pin_memory=True,
)
iterator = iter(loader)
waits = []
for _ in range(31):
    started = time.perf_counter()
    next(iterator)
    waits.append(time.perf_counter() - started)
steady = waits[5:]
mean_seconds = statistics.fmean(steady)
print(
    f"ROBOTWIN_ARRAY_CACHE_BENCHMARK batch_size=256 workers=12 "
    f"batches={len(steady)} mean_wait_seconds={mean_seconds:.6f} "
    f"median_wait_seconds={statistics.median(steady):.6f} "
    f"max_wait_seconds={max(steady):.6f}"
)
iterator._shutdown_workers()
assert mean_seconds < 0.50, mean_seconds
PY

PROBE_GPU=$((TRAIN_GPU_COUNT - 1))
CUDA_VISIBLE_DEVICES="$PROBE_GPU" EXPECTED_PROBE_GPU="$PROBE_GPU" \
"$PYTHON" - <<'PY' || fail "nonzero GPU mapping failed"
import os, torch
assert os.environ["CUDA_VISIBLE_DEVICES"] == os.environ["EXPECTED_PROBE_GPU"]
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
x = torch.ones(4, device="cuda:0")
assert x.sum().item() == 4
print(
    "GPU_MAPPING_OK",
    f"physical={os.environ['EXPECTED_PROBE_GPU']}",
    "local=0",
    torch.cuda.get_device_name(0),
)
PY

CUDA_VISIBLE_DEVICES="$PROBE_GPU" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
"$PYTHON" "$ROLLOUT_BENCH_TOOL" \
  --task handover_block_robotwin_image \
  --workers "$EVAL_WORKERS" --cycles 1 ||
  fail "RoboTwin rollout-pool preflight failed"

echo "===== Two-step baseline and Future optimizer smoke tests ====="
SMOKE_PIDS=()
SMOKE_VARIANTS=()
for variant in baseline future4_ratio010; do
  smoke_root="$CACHE_ROOT/smoke_$variant"
  smoke_args=(
    "--config-name" "exps/robotwin_mip_${variant}"
    "task=handover_block_robotwin_image"
    "task.robotwin_root=$RUNTIME_ROBOTWIN_ROOT"
    "task.dataset_path=$LOCAL_DATASET_ROOT/handover_block"
    "optimization.seed=41"
    "optimization.gradient_steps=2"
    "optimization.batch_size=$BATCH_SIZE"
    "optimization.dataloader_num_workers=2"
    "optimization.dataloader_persistent_workers=true"
    "optimization.device=cuda"
    "optimization.auto_resume=false"
    "eval.parallel_rollout=true"
    "eval.parallel_rollout_workers=1"
    "log.wandb_mode=disabled"
    "log.log_dir=$smoke_root"
    "log.log_freq=1"
    "log.gradient_diagnostic_freq=0"
    "log.validation_freq=1000"
    "log.eval_freq=1000"
    "log.save_freq=1000"
  )
  CUDA_VISIBLE_DEVICES="$PROBE_GPU" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  "$TRAIN" "${smoke_args[@]}" >"$smoke_root.launcher.log" 2>&1 &
  SMOKE_PIDS+=("$!")
  SMOKE_VARIANTS+=("$variant")
  if (( RUNS_PER_GPU == 1 )); then
    if ! wait "${SMOKE_PIDS[-1]}"; then
      tail -n 100 "$smoke_root.launcher.log" >&2 || true
      fail "$variant optimizer smoke test failed"
    fi
  fi
done
if (( RUNS_PER_GPU > 1 )); then
  smoke_failed=0
  for i in "${!SMOKE_PIDS[@]}"; do
    if ! wait "${SMOKE_PIDS[$i]}"; then
      smoke_failed=1
      tail -n 100 \
        "$CACHE_ROOT/smoke_${SMOKE_VARIANTS[$i]}.launcher.log" >&2 || true
    fi
  done
  (( smoke_failed == 0 )) ||
    fail "concurrent baseline/Future optimizer smoke test failed"
fi
echo "OPTIMIZER_SMOKE_OK batch_size=$BATCH_SIZE concurrent=$RUNS_PER_GPU"

WANDB_NAME_VALUE="robotwin_preflight_node${NODE_INDEX}_${RUN_TAG:-formal}" \
"$PYTHON" - <<'PY' || fail "W&B online write/read preflight failed"
import os, time, wandb
name = os.environ["WANDB_NAME_VALUE"]
run = wandb.init(
    entity="jepa-policy",
    project="robotwin",
    group="infra_preflight",
    job_type="preflight",
    name=name,
    mode="online",
    settings=wandb.Settings(init_timeout=180),
)
run.log({"preflight/online_write": 1, "preflight/time": time.time()})
path, run_id, url = run.path, run.id, run.url
run.finish()
api = wandb.Api(timeout=60)
last_error = None
for attempt in range(6):
    try:
        remote = api.run(path)
        if remote.id == run_id:
            print(f"WANDB_ONLINE_OK path={path} url={url}")
            break
    except Exception as exc:
        last_error = exc
    time.sleep(5)
else:
    raise RuntimeError(
        f"W&B run was not readable after upload: {path}; {last_error}"
    )
PY

if (( PREFLIGHT_ONLY != 0 )); then
  echo "PREFLIGHT_ONLY_OK node=$NODE_INDEX"
  exit 0
fi

for run_name in "${RUN_NAMES[@]}"; do
  if [[ -f "$LOG_ROOT/$run_name/models/model_latest.pt" ]]; then
    echo "RESUME_CHECKPOINT_FOUND run=$run_name"
  elif [[ -e "$LOG_ROOT/$run_name" ]]; then
    # A spot reclaim before the first periodic save leaves useful launcher and
    # W&B state but no model. Restart this same uniquely tagged run from step
    # zero instead of blocking every run on the node.
    echo "RESTART_WITHOUT_CHECKPOINT run=$run_name"
  fi
done

manifest="$LOG_ROOT/manifest_node${NODE_INDEX}_${RUN_TAG:-formal}.tsv"
status_file="$LOG_ROOT/status_node${NODE_INDEX}_${RUN_TAG:-formal}.tsv"
printf 'pid\twave\tphysical_gpu\tlocal_cuda\trun_name\tlog\n' >"$manifest"
printf 'pid\twave\tphysical_gpu\trun_name\texit_status\n' >"$status_file"
failed=0
for ((wave_start = 0; wave_start < ${#RUN_NAMES[@]}; wave_start += MAX_CONCURRENT_RUNS)); do
  wave=$((wave_start / MAX_CONCURRENT_RUNS))
  wave_end=$((wave_start + MAX_CONCURRENT_RUNS))
  (( wave_end <= ${#RUN_NAMES[@]} )) || wave_end="${#RUN_NAMES[@]}"
  echo "===== Launch wave $wave: runs $wave_start through $((wave_end - 1)) ====="
  WAVE_PIDS=()
  WAVE_INDICES=()
  WAVE_GPUS=()

  for ((i = wave_start; i < wave_end; i++)); do
    run_name="${RUN_NAMES[$i]}"
    run_dir="$LOG_ROOT/$run_name"
    run_cache="$CACHE_ROOT/$run_name"
    launcher_log="$LOG_ROOT/$run_name.launcher.log"
    wandb_run_id="$(
      printf '%s' "$run_name" | sha256sum | cut -c1-32
    )"
    slot=$((i - wave_start))
    physical_gpu=$((slot % TRAIN_GPU_COUNT))
    mkdir -p "$run_dir" "$run_cache"/{numba,matplotlib,xdg,huggingface}
    mapfile -t args <"${RUN_ARGS_FILES[$i]}"
    cp "$CACHE_ROOT/configs/$run_name.yaml" "$run_dir/resolved_config.yaml"
    printf '%q ' "$TRAIN" "${args[@]}" >"$run_dir/command.sh"
    printf '\n' >>"$run_dir/command.sh"

    setsid env \
      CUDA_VISIBLE_DEVICES="$physical_gpu" \
      ROBOTWIN_SOURCE_ROOT="$RUNTIME_ROBOTWIN_ROOT" \
      ROBOTWIN_DATASET_ROOT="$LOCAL_DATASET_ROOT" \
      ROBOTWIN_CACHE_ROOT="$run_cache" \
      ROBOTWIN_PYTHON_BIN="$ROBOTWIN_PYTHON_BIN" \
      ROBOTWIN_DEPENDENCY_SITE="$RUNTIME_DEPENDENCY_SITE" \
      ROBOTWIN_SYSTEM_LIB_DIR="$RUNTIME_SYSTEM_LIB_ROOT" \
      ROBOTWIN_USE_PPU_TORCH=0 \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 \
      WANDB_MODE="$ROBOTWIN_WANDB_MODE" \
      WANDB_ENTITY=jepa-policy WANDB_PROJECT=robotwin \
      WANDB_NAME="$run_name" \
      WANDB_RUN_ID="$wandb_run_id" \
      WANDB_RUN_GROUP="$(dataset_name_for "${TASKS[$i]}")_formal" \
      WANDB_DIR="$REPO/wandb" \
      "$TRAIN" "${args[@]}" >"$launcher_log" 2>&1 &
    pid=$!
    WAVE_PIDS+=("$pid")
    WAVE_INDICES+=("$i")
    WAVE_GPUS+=("$physical_gpu")
    printf '%s\t%s\t%s\t0\t%s\t%s\n' \
      "$pid" "$wave" "$physical_gpu" "$run_name" "$launcher_log" |
      tee -a "$manifest"
    if (( i + 1 < wave_end )); then sleep "$START_GAP"; fi
  done

  for j in "${!WAVE_PIDS[@]}"; do
    if wait "${WAVE_PIDS[$j]}"; then
      status=0
    else
      status=$?
      failed=1
    fi
    run_index="${WAVE_INDICES[$j]}"
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "${WAVE_PIDS[$j]}" "$wave" "${WAVE_GPUS[$j]}" \
      "${RUN_NAMES[$run_index]}" "$status" | tee -a "$status_file"
  done
done

(( failed == 0 )) || fail "one or more RoboTwin runs failed"
echo "ALL_RUNS_COMPLETED node=$NODE_INDEX"
