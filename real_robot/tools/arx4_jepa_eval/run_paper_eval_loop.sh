#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNNER="${SCRIPT_DIR}/run_real_rollout.sh"

METHOD="${METHOD:-jepa}"
TASK="${TASK:-cup_upright}"
STEP_EXPLICIT=false
if [[ -n "${STEP+x}" ]]; then
  STEP_EXPLICIT=true
fi
STEP="${STEP:-100000}"
RECOVERY_SECONDS="${RECOVERY_SECONDS:-30}"
MAX_EPISODES="${MAX_EPISODES:-0}"
DEVICE="${DEVICE:-cuda}"
DENOISE_STEPS="${DENOISE_STEPS:-100}"
DENOISE_STEPS_EXPLICIT=false
SESSION_NAME="${SESSION_NAME:-}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
ASSET_ROOT="${ASSET_ROOT:-${PROJECT_ROOT}}"
INFERENCE_ROOT="${INFERENCE_ROOT:-${PROJECT_ROOT}/runs}"
CHECK_ONLY=false
LIST_ONLY=false

usage() {
  cat <<'EOF'
Usage:
  bash tools/arx4_jepa_eval/run_paper_eval_loop.sh [options]

Options:
  --method NAME       jepa, mip, dp16 (16-step DDPM/L1/A8), or dp (100-step DDPM/L3/A6)
  --task NAME         cabinet, cup_stack, cup_upright, pen_insert, plate_grape
  --step STEP         060000, 080000, 100000, 140000, or 180000
                      aliases such as 60k and 100k are accepted; omitted when
                      --checkpoint has a standard step-bearing filename
  --checkpoint PATH   load this checkpoint instead of the standard asset path;
                      infer step from model_stepNNNNNN.pt or step=NNNNNN.ckpt
  --recovery-seconds N
                      scene-recovery countdown after reset (default: 30)
  --episodes N        stop after N episodes; 0 means run until Ctrl+C (default: 0)
  --device DEVICE     inference device (default: cuda)
  --denoise-steps N   DP DDPM inference steps, 1-100 (default: 100)
  --session NAME      persistent run-directory label
  --list              list locally complete checkpoints and exit
  --check             validate configuration/assets without opening hardware
  -h, --help          show this help

Controls:
  Space               start while idle; finish scene recovery early
  S / F               mark the active episode success / failure and save it
  A                   skip the active episode and delete its per-episode data
  R                   emergency discard/reset; when idle, reset Home
  Enter               ignored
  H                   show controls
  Ctrl+C              reset Home, close the persistent session, and exit

Checkpoint, CUDA, cameras, robot connection, and recording workers are created
once for the whole session. Per-episode policy/state/action data is still split.
EOF
}

die() {
  printf '[paper_eval] error: %s\n' "$*" >&2
  exit 2
}

normalize_method() {
  case "${1,,}" in
    jepa|jepapolicy|jepa_policy) printf 'jepa\n' ;;
    mip) printf 'mip\n' ;;
    dp|dp16|diffusion|diffusion_policy|diffusion_policy16|diffusion_policy_16) printf 'diffusion_policy\n' ;;
    *) return 1 ;;
  esac
}

normalize_step() {
  case "${1,,}" in
    60000|060000|60k) printf '060000\n' ;;
    80000|080000|80k) printf '080000\n' ;;
    100000|100k) printf '100000\n' ;;
    140000|140k) printf '140000\n' ;;
    180000|180k) printf '180000\n' ;;
    *) return 1 ;;
  esac
}

while (( $# > 0 )); do
  case "$1" in
    --method) (( $# >= 2 )) || die "--method requires a value"; METHOD="$2"; shift 2 ;;
    --task) (( $# >= 2 )) || die "--task requires a value"; TASK="$2"; shift 2 ;;
    --step) (( $# >= 2 )) || die "--step requires a value"; STEP="$2"; STEP_EXPLICIT=true; shift 2 ;;
    --checkpoint) (( $# >= 2 )) || die "--checkpoint requires a value"; CHECKPOINT_PATH="$2"; shift 2 ;;
    --chunks) die "--chunks was removed: paper evaluation now ends only on S/F/A" ;;
    --recovery-seconds) (( $# >= 2 )) || die "--recovery-seconds requires a value"; RECOVERY_SECONDS="$2"; shift 2 ;;
    --episodes) (( $# >= 2 )) || die "--episodes requires a value"; MAX_EPISODES="$2"; shift 2 ;;
    --device) (( $# >= 2 )) || die "--device requires a value"; DEVICE="$2"; shift 2 ;;
    --denoise-steps) (( $# >= 2 )) || die "--denoise-steps requires a value"; DENOISE_STEPS="$2"; DENOISE_STEPS_EXPLICIT=true; shift 2 ;;
    --session) (( $# >= 2 )) || die "--session requires a value"; SESSION_NAME="$2"; shift 2 ;;
    --list) LIST_ONLY=true; shift ;;
    --check) CHECK_ONLY=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

if [[ "${LIST_ONLY}" == "true" ]]; then
  find "${ASSET_ROOT}/checkpoints" -type f \
    \( -name 'model_step*.pt' -o -name 'step=*.ckpt' \) \
    ! -name '*.part' -printf '%P\n' 2>/dev/null | sort
  exit 0
fi

REQUESTED_METHOD="${METHOD,,}"
if [[ "${REQUESTED_METHOD}" =~ ^(dp16|diffusion_policy16|diffusion_policy_16)$ ]]; then
  if [[ "${DENOISE_STEPS_EXPLICIT}" == "true" && "${DENOISE_STEPS}" != "16" ]]; then
    die "--method dp16 fixes --denoise-steps at 16"
  fi
  DENOISE_STEPS=16
fi
METHOD="$(normalize_method "${METHOD}")" || die "unsupported method: ${METHOD}"
INFERRED_STEP=""
if [[ -n "${CHECKPOINT_PATH}" ]]; then
  checkpoint_name="${CHECKPOINT_PATH##*/}"
  if [[ "${checkpoint_name}" =~ ^model_step([0-9]{6})\.pt$ ]]; then
    INFERRED_STEP="${BASH_REMATCH[1]}"
  elif [[ "${checkpoint_name}" =~ ^step=([0-9]{6})\.ckpt$ ]]; then
    INFERRED_STEP="${BASH_REMATCH[1]}"
  elif [[ "${STEP_EXPLICIT}" != "true" ]]; then
    die "cannot infer step from custom checkpoint filename: ${checkpoint_name}; pass --step"
  fi
fi
if [[ -n "${INFERRED_STEP}" && "${STEP_EXPLICIT}" != "true" ]]; then
  STEP="${INFERRED_STEP}"
fi
STEP="$(normalize_step "${STEP}")" || die "unsupported checkpoint step: ${STEP}"
if [[ -n "${INFERRED_STEP}" && "${STEP}" != "${INFERRED_STEP}" ]]; then
  die "--step ${STEP} does not match custom checkpoint step ${INFERRED_STEP}"
fi
[[ "${TASK}" =~ ^(cabinet|cup_stack|cup_upright|pen_insert|plate_grape)$ ]] \
  || die "unsupported task: ${TASK}"
[[ "${RECOVERY_SECONDS}" =~ ^[0-9]+$ ]] || die "--recovery-seconds must be a non-negative integer"
[[ "${MAX_EPISODES}" =~ ^[0-9]+$ ]] || die "--episodes must be a non-negative integer"
[[ -n "${DEVICE}" ]] || die "--device must not be empty"
[[ "${DENOISE_STEPS}" =~ ^[1-9][0-9]*$ ]] \
  || die "--denoise-steps must be an integer from 1 to 100"
(( DENOISE_STEPS <= 100 )) || die "--denoise-steps must not exceed the 100 training steps"
if [[ "${METHOD}" != "diffusion_policy" && "${DENOISE_STEPS_EXPLICIT}" == "true" ]]; then
  die "--denoise-steps only applies to diffusion_policy"
fi
if [[ "${METHOD}" == "diffusion_policy" ]]; then
  METHOD_PROFILE="dp${DENOISE_STEPS}"
  if (( DENOISE_STEPS <= 16 )); then
    LATENCY_COMPENSATION_STEPS=1
  else
    LATENCY_COMPENSATION_STEPS=3
  fi
  EXECUTION_STEPS=$((9 - LATENCY_COMPENSATION_STEPS))
  if (( EXECUTION_STEPS > 8 )); then
    EXECUTION_STEPS=8
  fi
else
  METHOD_PROFILE="${METHOD}"
  LATENCY_COMPENSATION_STEPS=0
  EXECUTION_STEPS=8
fi

case "${METHOD}" in
  jepa) CHECKPOINT="${ASSET_ROOT}/checkpoints/jepa/${TASK}/model_step${STEP}.pt" ;;
  mip) CHECKPOINT="${ASSET_ROOT}/checkpoints/mip/${TASK}/model_step${STEP}.pt" ;;
  diffusion_policy) CHECKPOINT="${ASSET_ROOT}/checkpoints/diffusion_policy/${TASK}/step=${STEP}.ckpt" ;;
esac
CUSTOM_CHECKPOINT=false
CHECKPOINT_VARIANT=""
variant_marker="$(dirname "${CHECKPOINT}")/.checkpoint_variant"
if [[ -s "${variant_marker}" ]]; then
  IFS= read -r CHECKPOINT_VARIANT < "${variant_marker}"
  [[ "${CHECKPOINT_VARIANT}" =~ ^[A-Za-z0-9._-]+$ ]] \
    || die "invalid checkpoint variant marker: ${variant_marker}"
fi
if [[ -n "${CHECKPOINT_PATH}" ]]; then
  [[ "${CHECKPOINT_PATH}" == /* && "${CHECKPOINT_PATH}" != / ]] \
    || die "--checkpoint must be a safe absolute path"
  CHECKPOINT="${CHECKPOINT_PATH}"
  CUSTOM_CHECKPOINT=true
  checkpoint_parent="$(dirname "${CHECKPOINT}")"
  checkpoint_parent_name="$(basename "${checkpoint_parent}")"
  checkpoint_name="${CHECKPOINT##*/}"
  case "${METHOD}" in
    jepa)
      [[ "${checkpoint_name}" =~ ^model_step[0-9]{6}\.pt$ \
        && "${checkpoint_parent_name}" =~ ^(jepa|jepa_policy)$ ]] \
        || die "JEPA requires jepa[_policy]/model_stepNNNNNN.pt, got: ${CHECKPOINT}"
      ;;
    mip)
      [[ "${checkpoint_name}" =~ ^model_step[0-9]{6}\.pt$ \
        && "${checkpoint_parent_name}" == "mip" ]] \
        || die "MIP requires mip/model_stepNNNNNN.pt, got: ${CHECKPOINT}"
      ;;
    diffusion_policy)
      [[ "${checkpoint_name}" =~ ^step=[0-9]{6}\.ckpt$ \
        && "${checkpoint_parent_name}" =~ ^(dp|diffusion_policy)$ ]] \
        || die "DP requires dp/step=NNNNNN.ckpt, got: ${CHECKPOINT}"
      ;;
  esac
  CHECKPOINT_VARIANT="$(basename "$(dirname "${checkpoint_parent}")")"
  CHECKPOINT_VARIANT="${CHECKPOINT_VARIANT//[^A-Za-z0-9._-]/_}"
fi
[[ -f "${CHECKPOINT}" && -s "${CHECKPOINT}" ]] \
  || die "checkpoint is missing or incomplete: ${CHECKPOINT}"
if [[ "${CUSTOM_CHECKPOINT}" == "true" ]] && command -v unzip >/dev/null 2>&1; then
  unzip -tq "${CHECKPOINT}" >/dev/null 2>&1 \
    || die "custom checkpoint archive is truncated or invalid: ${CHECKPOINT}"
fi
if [[ "${METHOD}" != "diffusion_policy" ]]; then
  [[ -s "${ASSET_ROOT}/stats/${TASK}/stats.json" ]] \
    || die "matching task stats are missing: ${ASSET_ROOT}/stats/${TASK}/stats.json"
fi
[[ -x "${RUNNER}" ]] || die "rollout launcher is not executable: ${RUNNER}"

printf '%s\n' \
  '[paper_eval] configuration valid' \
  "method=${METHOD}" \
  "method_profile=${METHOD_PROFILE}" \
  "task=${TASK}" \
  "checkpoint_step=${STEP}" \
  "checkpoint=${CHECKPOINT}" \
  "checkpoint_variant=${CHECKPOINT_VARIANT:-standard}" \
  "denoise_steps=$([[ "${METHOD}" == "diffusion_policy" ]] && printf '%s' "${DENOISE_STEPS}" || printf 'n/a')" \
  "latency_compensation_steps=$([[ "${METHOD}" == "diffusion_policy" ]] && printf '%s' "${LATENCY_COMPENSATION_STEPS}" || printf 'n/a')" \
  "episode_end=manual S/F/A decision (no chunk timeout), A${EXECUTION_STEPS} at 10Hz" \
  "recovery_seconds=${RECOVERY_SECONDS}" \
  "max_episodes=${MAX_EPISODES} (0 means Ctrl+C)" \
  'persistent_resources=true recording=true reset_after_every_episode=true'

if [[ "${CHECK_ONLY}" == "true" ]]; then
  printf '[paper_eval] check-only: hardware was not opened\n'
  exit 0
fi

[[ -t 0 && -t 1 ]] || die "paper evaluation must run in an interactive terminal"
if [[ -z "${SESSION_NAME}" ]]; then
  SESSION_NAME="$(date '+%Y%m%d_%H%M%S')_${METHOD}_${TASK}_step${STEP}"
  if [[ "${METHOD}" == "diffusion_policy" ]]; then
    SESSION_NAME="${SESSION_NAME}_denoise${DENOISE_STEPS}_latency${LATENCY_COMPENSATION_STEPS}"
  fi
  if [[ -n "${CHECKPOINT_VARIANT}" ]]; then
    SESSION_NAME="${SESSION_NAME}_${CHECKPOINT_VARIANT}"
  fi
fi
[[ "${SESSION_NAME}" =~ ^[A-Za-z0-9._-]+$ ]] \
  || die "--session may contain only letters, numbers, dot, underscore, and dash"
RUN_DIR="${INFERENCE_ROOT}/${TASK}/${SESSION_NAME}"
[[ ! -e "${RUN_DIR}" ]] || die "session already exists: ${RUN_DIR}"

printf '%s\n' \
  '[paper_eval] persistent mode: checkpoint/CUDA/cameras/robot load once' \
  "session_run_dir=${RUN_DIR}" \
  'original_camera_video=session-wide continuous recording' \
  'policy_state_action_data=classified under data/success and data/failure'

exec env \
  ASSET_ROOT="${ASSET_ROOT}" \
  CHECKPOINT_PATH="${CHECKPOINT}" \
  INFERENCE_ROOT="${INFERENCE_ROOT}" \
  RUN_ID="${SESSION_NAME}" \
  METHOD="${METHOD}" \
  TASK="${TASK}" \
  STEP="${STEP}" \
  ACTION_STEPS="${EXECUTION_STEPS}" \
  RUN_CHUNKS=2 \
  DEVICE="${DEVICE}" \
  DP_INFERENCE_STEPS="${DENOISE_STEPS}" \
  DP_LATENCY_COMPENSATION_STEPS="${LATENCY_COMPENSATION_STEPS}" \
  FORMAL_RUN=true \
  RECORDING_ENABLED=true \
  RESET_HOME_ON_CLOSE=false \
  RESET_HOME_ON_SUCCESS=true \
  RESET_HOME_ON_INTERRUPT=true \
  REQUIRE_CONFIRMATION=false \
  WORKFLOW_MODULE=prometheus.workflows.rollout_paper_eval \
  PAPER_RECOVERY_SECONDS="${RECOVERY_SECONDS}" \
  PAPER_MAX_EPISODES="${MAX_EPISODES}" \
  bash "${RUNNER}"
