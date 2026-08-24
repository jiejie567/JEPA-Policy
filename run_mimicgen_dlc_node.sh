#!/usr/bin/env bash
set -euo pipefail

NODE_INDEX="${1:-}"
[[ "$NODE_INDEX" == "1" || "$NODE_INDEX" == "2" ]] || {
  echo "Usage: $0 {1|2}" >&2
  exit 2
}

REPO="/mnt/data_nas/ykj_jepa_policy/code/JEPA-Policy"
TRAIN="$REPO/tools/run_mimicgen_train.sh"
PYTHON="$REPO/tools/run_mimicgen_python.sh"
PROJECT="mimicgen"
ENTITY="jepa-policy"
LOG_ROOT="$REPO/logs/$PROJECT"
CACHE_ROOT="/tmp/jepa_policy_mimicgen_node${NODE_INDEX}"
START_GAP="${START_GAP:-180}"
DRY_RUN="${DRY_RUN:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
EVAL_WORKERS=20
EVAL_EPISODES=40

TASKS=()
VARIANTS=()
SEEDS=()

add_run() {
  TASKS+=("$1")
  VARIANTS+=("$2")
  SEEDS+=("$3")
}

if [[ "$NODE_INDEX" == "1" ]]; then
  for seed in 41 42 43; do
    add_run coffee_preparation_d1_image baseline "$seed"
    add_run coffee_preparation_d1_image future4_ratio010 "$seed"
  done
  add_run three_piece_assembly_d1_image baseline 41
  add_run three_piece_assembly_d1_image future4_ratio010 41
  add_run three_piece_assembly_d1_image baseline 42
else
  for seed in 41 42 43; do
    add_run kitchen_d1_image baseline "$seed"
    add_run kitchen_d1_image future4_ratio010 "$seed"
  done
  add_run three_piece_assembly_d1_image future4_ratio010 42
  add_run three_piece_assembly_d1_image baseline 43
  add_run three_piece_assembly_d1_image future4_ratio010 43
fi

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ -x "$TRAIN" ]] || fail "missing training wrapper: $TRAIN"
[[ -x "$PYTHON" ]] || fail "missing Python wrapper: $PYTHON"
[[ "$DRY_RUN" =~ ^[01]$ ]] || fail "DRY_RUN must be 0 or 1"
[[ "$PREFLIGHT_ONLY" =~ ^[01]$ ]] || fail "PREFLIGHT_ONLY must be 0 or 1"
[[ "$START_GAP" =~ ^[0-9]+$ ]] || fail "START_GAP must be non-negative"
(( EVAL_EPISODES % EVAL_WORKERS == 0 )) || fail "eval episodes must divide workers"
(( ${#TASKS[@]} == 9 )) || fail "each node must contain exactly 9 runs"

declare -A DATASETS=(
  [three_piece_assembly_d1_image]="$REPO/datasets/mimicgen/core/three_piece_assembly_d1/three_piece_assembly_d1.hdf5"
  [coffee_preparation_d1_image]="$REPO/datasets/mimicgen/core/coffee_preparation_d1/coffee_preparation_d1.hdf5"
  [kitchen_d1_image]="$REPO/datasets/mimicgen/core/kitchen_d1/kitchen_d1.hdf5"
)
declare -A SIZES=(
  [three_piece_assembly_d1_image]=3237234348
  [coffee_preparation_d1_image]=6923699908
  [kitchen_d1_image]=7069704890
)
declare -A SHAS=(
  [three_piece_assembly_d1_image]=7f5cad32fdf492b210c181b84b4856eaff1a90573ffbc5eeae871cbe8e01e586
  [coffee_preparation_d1_image]=0e9e1eac8d969c05a5fff90358f702b6530ea80bb7a9977497e3a3740b88cf55
  [kitchen_d1_image]=e43e339f85283aca458a2455acfe013a0642d72b117412d3218844ffd7d82dfb
)

unique_tasks=()
declare -A SEEN_TASK=()
for task in "${TASKS[@]}"; do
  if [[ -z "${SEEN_TASK[$task]:-}" ]]; then
    unique_tasks+=("$task")
    SEEN_TASK[$task]=1
  fi
done

echo "MimicGen node $NODE_INDEX: ${#TASKS[@]} runs, rollout_workers=$EVAL_WORKERS"
for i in "${!TASKS[@]}"; do
  printf 'PLAN gpu=%d task=%s variant=%s seed=%s\n' \
    "$i" "${TASKS[$i]}" "${VARIANTS[$i]}" "${SEEDS[$i]}"
done

for task in "${unique_tasks[@]}"; do
  file="${DATASETS[$task]}"
  [[ -f "$file" ]] || fail "dataset missing: $file"
  [[ "$(stat -c %s "$file")" == "${SIZES[$task]}" ]] || fail "wrong dataset size: $file"
  if (( DRY_RUN == 0 )); then
    echo "${SHAS[$task]}  $file" | sha256sum -c - || fail "checksum failed: $file"
  fi
done

mkdir -p "$LOG_ROOT" "$CACHE_ROOT/configs"

common_args=(
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
  "optimization.dataloader_num_workers=8"
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

RUN_NAMES=()
RUN_ARGS_FILES=()
for i in "${!TASKS[@]}"; do
  task="${TASKS[$i]}"
  variant="${VARIANTS[$i]}"
  seed="${SEEDS[$i]}"
  run_name="${task}_${variant}_seed${seed}"
  run_dir="$LOG_ROOT/$run_name"
  config_path="$CACHE_ROOT/configs/$run_name.yaml"
  args=("task=$task" "${common_args[@]}"
    "optimization.seed=$seed"
    "log.group=${task}_formal"
    "log.exp_name=$run_name"
    "log.log_dir=$run_dir")
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
      "++task.future_state_steps=4"
      "++task.future_state_steps_list=[4]"
      "optimization.use_future_embed_loss=true"
      "optimization.future_embed_loss_mode=mip_two_step"
      "optimization.future_joint_mode=true"
      "optimization.future_t_two_step=0.9"
      "optimization.future_state_loss_mode=ratio"
      "optimization.future_state_loss_ratio=0.1"
      "optimization.future_state_loss_weight_min=0.000001"
      "optimization.future_state_loss_weight_max=0.1"
    )
  fi

  EXPECTED_TASK="$task" EXPECTED_VARIANT="$variant" EXPECTED_SEED="$seed" \
  EXPECTED_RUN="$run_name" EXPECTED_DATASET="${DATASETS[$task]}" \
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
t, n, o, e, l = c["task"], c["network"], c["optimization"], c["eval"], c["log"]
assert t["dataset_path"] == os.environ["EXPECTED_DATASET"]
assert Path(t["dataset_path"]).is_file()
assert t["env_type"] == "mimicgen" and t["abs_action"] is False
assert n["network_type"] == "chitransformer" and n["emb_dim"] == 384
assert o["loss_type"] == "mip" and o["batch_size"] == 256
assert o["gradient_steps"] == 300000 and o["seed"] == int(os.environ["EXPECTED_SEED"])
assert e["parallel_rollout"] is True and e["parallel_rollout_workers"] == 20
assert l["eval_episodes"] == 40
assert l["entity"] == "jepa-policy" and l["project"] == "mimicgen"
assert l["exp_name"] == os.environ["EXPECTED_RUN"]
if os.environ["EXPECTED_VARIANT"] == "baseline":
    assert t["future_state_enabled"] is False
    assert n["n_future_tokens"] == 0
    assert o["use_future_embed_loss"] is False and o["future_joint_mode"] is False
else:
    assert t["future_state_enabled"] is True
    assert t["future_state_steps"] == 4 and t["future_state_steps_list"] == [4]
    assert n["n_future_tokens"] == 1
    assert o["future_joint_mode"] is True
    assert o["future_embed_loss_mode"] == "mip_two_step"
    assert o["future_state_loss_mode"] == "ratio"
    assert o["future_state_loss_ratio"] == 0.1
print(OmegaConf.to_yaml(cfg, resolve=True), end="")
PY
  RUN_NAMES+=("$run_name")
  args_file="$CACHE_ROOT/configs/$run_name.args"
  printf '%s\n' "${args[@]}" >"$args_file"
  RUN_ARGS_FILES+=("$args_file")
  echo "CONFIG_OK $run_name"
done

if (( DRY_RUN != 0 )); then
  echo "DRY_RUN_OK node=$NODE_INDEX"
  exit 0
fi

[[ -n "${WANDB_API_KEY:-}" ]] || fail "WANDB_API_KEY is not injected"
hardware="$("$PYTHON" - <<'PY'
import os, torch
assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 16, torch.cuda.device_count()
x = torch.ones(4, device="cuda")
assert x.sum().item() == 4
print(f"torch={torch.__version__} devices={torch.cuda.device_count()}")
PY
)" || fail "PPU PyTorch preflight failed"
echo "HARDWARE_OK $hardware"

cpu_count="$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)"
memory_gib="$(awk '/^MemTotal:/ {print int($2/1024/1024)}' /proc/meminfo)"
(( cpu_count >= 160 )) || fail "need 160 CPUs, found $cpu_count"
(( memory_gib >= 1400 )) || fail "need about 1600 GB RAM, found ${memory_gib}GiB"

"$PYTHON" - <<'PY' || fail "MimicGen package/version preflight failed"
import mimicgen, robomimic, robosuite, numpy, mujoco
versions = {
    "mimicgen": mimicgen.__version__,
    "robomimic": robomimic.__version__,
    "robosuite": robosuite.__version__,
    "numpy": numpy.__version__,
    "mujoco": mujoco.__version__,
}
print("VERSIONS_FOUND", " ".join(f"{name}={version}" for name, version in versions.items()))
assert mimicgen.__version__ == "1.0.1"
assert robomimic.__version__ == "0.3.1"
assert robosuite.__version__ == "1.4.1"
assert numpy.__version__ == "1.26.4"
assert mujoco.__version__ == "3.3.6"
print("VERSIONS_OK", " ".join(f"{name}={version}" for name, version in versions.items()))
PY

if ! WANDB_ENTITY="$ENTITY" WANDB_PROJECT="$PROJECT" "$PYTHON" - <<'PY'
import os, wandb
api = wandb.Api(api_key=os.environ["WANDB_API_KEY"], timeout=30)
projects = {p.name for p in api.projects(entity=os.environ["WANDB_ENTITY"])}
state = "exists" if os.environ["WANDB_PROJECT"] in projects else "will-be-created"
print("WANDB_OK", os.environ["WANDB_ENTITY"], os.environ["WANDB_PROJECT"], state)
PY
then
  fail "W&B authentication/entity preflight failed"
fi

for task in "${unique_tasks[@]}"; do
  # Reproduce the routing used by a nonzero physical training GPU. Once CUDA
  # visibility is narrowed to one card, CUDA is locally device 0 and Mesa
  # exposes only EGL device 0; never pass the physical CUDA id to MuJoCo.
  if ! env \
    -u MUJOCO_EGL_DEVICE_ID \
    CUDA_VISIBLE_DEVICES=15 \
    JEPA_POLICY_EGL_DEVICE_ID=0 \
    TASK_NAME="$task" \
    "$PYTHON" - <<'PY'
import os
from pathlib import Path
from hydra import compose, initialize_config_dir
from mip.envs.robot_env import make_vec_env
from mip.runtime_env import validate_runtime_environment
with initialize_config_dir(version_base=None, config_dir=str(Path("examples/configs").resolve())):
    cfg = compose(config_name="main", overrides=[f"task={os.environ['TASK_NAME']}", "task.num_envs=1"])
validate_runtime_environment(cfg.task)
env = make_vec_env(cfg.task, seed=12345)
obs, _ = env.reset()
assert obs["agentview_image"].shape[-3:] == (3, 84, 84)
assert obs["robot0_eye_in_hand_image"].shape[-3:] == (3, 84, 84)
env.close()
print("ENV_OK", os.environ["TASK_NAME"])
PY
  then
    fail "nonzero-GPU environment reset failed: $task"
  fi
done

if (( PREFLIGHT_ONLY != 0 )); then
  echo "PREFLIGHT_ONLY_OK node=$NODE_INDEX"
  exit 0
fi

for run_name in "${RUN_NAMES[@]}"; do
  [[ ! -e "$LOG_ROOT/$run_name" ]] || fail "log directory already exists: $LOG_ROOT/$run_name"
done

manifest="$LOG_ROOT/manifest_node${NODE_INDEX}.tsv"
status_file="$LOG_ROOT/status_node${NODE_INDEX}.tsv"
printf 'pid\tgpu\trun_name\tlog\n' >"$manifest"
printf 'pid\tgpu\trun_name\texit_status\n' >"$status_file"
PIDS=()

for i in "${!RUN_NAMES[@]}"; do
  run_name="${RUN_NAMES[$i]}"
  run_dir="$LOG_ROOT/$run_name"
  run_cache="$CACHE_ROOT/$run_name"
  launcher_log="$LOG_ROOT/$run_name.launcher.log"
  mkdir -p "$run_dir" "$run_cache/numba" "$run_cache/matplotlib" "$run_cache/xdg"
  mapfile -t args <"${RUN_ARGS_FILES[$i]}"
  cp "$CACHE_ROOT/configs/$run_name.yaml" "$run_dir/resolved_config.yaml"
  printf '%q ' "$TRAIN" "${args[@]}" >"$run_dir/command.sh"
  printf '\n' >>"$run_dir/command.sh"

  setsid env \
    -u MUJOCO_EGL_DEVICE_ID \
    CUDA_VISIBLE_DEVICES="$i" \
    JEPA_POLICY_EGL_DEVICE_ID=0 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 LP_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 \
    WANDB_MODE=online WANDB_ENTITY="$ENTITY" WANDB_PROJECT="$PROJECT" \
    WANDB_NAME="$run_name" WANDB_RUN_GROUP="${TASKS[$i]}_formal" \
    WANDB_DIR="$REPO/wandb" \
    NUMBA_CACHE_DIR="$run_cache/numba" \
    MPLCONFIGDIR="$run_cache/matplotlib" XDG_CACHE_HOME="$run_cache/xdg" \
    "$TRAIN" "${args[@]}" >"$launcher_log" 2>&1 &
  pid=$!
  PIDS+=("$pid")
  printf '%s\t%s\t%s\t%s\n' "$pid" "$i" "$run_name" "$launcher_log" | tee -a "$manifest"
  if (( i + 1 < ${#RUN_NAMES[@]} )); then sleep "$START_GAP"; fi
done

failed=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then status=0; else status=$?; failed=1; fi
  printf '%s\t%s\t%s\t%s\n' "${PIDS[$i]}" "$i" "${RUN_NAMES[$i]}" "$status" |
    tee -a "$status_file"
done
(( failed == 0 )) || fail "one or more training runs failed"
echo "ALL_RUNS_COMPLETED node=$NODE_INDEX"
