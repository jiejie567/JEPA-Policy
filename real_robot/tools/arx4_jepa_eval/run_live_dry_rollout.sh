#!/usr/bin/env bash
set -euo pipefail

# Live-observation dry-run: opens the X5/CAN/cameras but never sends abs_qpos.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/safety_profiles.sh"
source "${SCRIPT_DIR}/runtime_env.sh"
ASSET_ROOT="${ASSET_ROOT:-${PROJECT_ROOT}}"
SOURCE_ROOT="${SOURCE_ROOT:-${ASSET_ROOT}/runtime/prometheus}"
JEPA_ROOT="${JEPA_ROOT:-${ASSET_ROOT}/runtime/JEPA-Policy}"
DIFFUSION_ROOT="${DIFFUSION_ROOT:-${ASSET_ROOT}/runtime/diffusion_policy}"
ARX5_SDK_ROOT="${ARX5_SDK_ROOT:-${ASSET_ROOT}/vendor/arx5-sdk}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"

METHOD="${METHOD:-jepa}"
TASK="${TASK:-cabinet}"
STEP="${STEP:-100000}"
DEVICE="${DEVICE:-cuda}"
RUN_CHUNKS="${RUN_CHUNKS:-1}"
ACTION_STEPS="${ACTION_STEPS:-1}"
DP_INFERENCE_STEPS="${DP_INFERENCE_STEPS:-100}"
DP_LATENCY_COMPENSATION_STEPS="${DP_LATENCY_COMPENSATION_STEPS:-}"
RECORDING_ENABLED="${RECORDING_ENABLED:-false}"
RESULT_ROOT="${RESULT_ROOT:-${ASSET_ROOT}/validation/${TASK}}"
RGB_VALIDATION_REVISION="rgb128_direct_warmup60_dynamics_home_v6"
DP_STABILITY_REVISION="dp_rgb128_direct_seed41_bounded_dynamics_latency_aligned_home_v10"

die() {
  printf '[arx4_jepa_eval] error: %s\n' "$*" >&2
  exit 2
}

[[ "${METHOD}" =~ ^(jepa|mip|diffusion_policy)$ ]] || die "METHOD must be jepa, mip, or diffusion_policy"
[[ "${TASK}" =~ ^(cabinet|cup_stack|cup_upright|pen_insert|plate_grape)$ ]] || die "unsupported TASK: ${TASK}"
[[ "${STEP}" =~ ^(060000|080000|100000|140000|180000)$ ]] || die "unsupported STEP: ${STEP}"
if [[ "${METHOD}" == "diffusion_policy" ]]; then
  [[ "${RUN_CHUNKS}" =~ ^(1|2)$ ]] \
    || die "Diffusion Policy live dry-run permits RUN_CHUNKS=1 or 2"
else
  [[ "${RUN_CHUNKS}" == "1" ]] || die "JEPA/MIP live dry-run is fixed to RUN_CHUNKS=1"
fi
[[ "${ACTION_STEPS}" =~ ^[1-8]$ ]] || die "ACTION_STEPS must be an integer from 1 to 8"
[[ "${DP_INFERENCE_STEPS}" =~ ^[1-9][0-9]*$ ]] \
  || die "DP_INFERENCE_STEPS must be an integer from 1 to 100"
(( DP_INFERENCE_STEPS <= 100 )) || die "DP_INFERENCE_STEPS must not exceed 100"
if [[ -z "${DP_LATENCY_COMPENSATION_STEPS}" ]]; then
  if (( DP_INFERENCE_STEPS <= 16 )); then
    DP_LATENCY_COMPENSATION_STEPS=1
  else
    DP_LATENCY_COMPENSATION_STEPS=3
  fi
fi
[[ "${DP_LATENCY_COMPENSATION_STEPS}" =~ ^[0-8]$ ]] \
  || die "DP_LATENCY_COMPENSATION_STEPS must be an integer from 0 to 8"
DP_EXECUTABLE_ACTIONS=$((9 - DP_LATENCY_COMPENSATION_STEPS))
if (( DP_EXECUTABLE_ACTIONS > 8 )); then
  DP_EXECUTABLE_ACTIONS=8
fi
if [[ "${METHOD}" == "diffusion_policy" ]] && (( ACTION_STEPS > DP_EXECUTABLE_ACTIONS )); then
  die "ACTION_STEPS=${ACTION_STEPS} exceeds compensated DP horizon ${DP_EXECUTABLE_ACTIONS}"
fi
[[ "${RECORDING_ENABLED}" =~ ^(true|false)$ ]] || die "RECORDING_ENABLED must be true or false"
[[ -x "${PYTHON_BIN}" ]] || die "isolated Python is not executable: ${PYTHON_BIN}"
[[ -d "${SOURCE_ROOT}" ]] || die "isolated source not found: ${SOURCE_ROOT}"
[[ -d "${ARX5_SDK_ROOT}/python" ]] || die "isolated ARX5 SDK not found: ${ARX5_SDK_ROOT}"
[[ -e /dev/arxcan1 && -e /dev/arxcan3 ]] || die "ARX5 CAN links /dev/arxcan1 and /dev/arxcan3 are required"
arx4_acquire_hardware_lock \
  || die "can1/can3 are reserved by another ARX5 launcher; close or cancel the other terminal first"

if pgrep -af 'prometheus\.system\.collection_manager|prometheus\.workflows\.rollout_sync' >/dev/null; then
  pgrep -af 'prometheus\.system\.collection_manager|prometheus\.workflows\.rollout_sync' >&2 || true
  die "another robot owner or rollout process is running"
fi

case "${METHOD}" in
  jepa)
    CONFIG_NAME="rollout_jepa_safe_sync"
    CHECKPOINT="${ASSET_ROOT}/checkpoints/jepa/${TASK}/model_step${STEP}.pt"
    ;;
  mip)
    CONFIG_NAME="rollout_mip_safe_sync"
    CHECKPOINT="${ASSET_ROOT}/checkpoints/mip/${TASK}/model_step${STEP}.pt"
    ;;
  diffusion_policy)
    CONFIG_NAME="rollout_diffusion_policy_safe_sync"
    CHECKPOINT="${ASSET_ROOT}/checkpoints/diffusion_policy/${TASK}/step=${STEP}.ckpt"
    ;;
esac
[[ -s "${CHECKPOINT}" ]] || die "checkpoint missing: ${CHECKPOINT}"
CHECKPOINT_OVERRIDE="${CHECKPOINT//=/\\=}"

MAX_DELTA="$(arx4_max_delta_for_task "${TASK}")" || die "no safety profile for TASK=${TASK}"
STABLE_DELTA="$(arx4_stable_delta_for_profile "${METHOD}" "${TASK}" "${STEP}")" \
  || die "no velocity profile for METHOD=${METHOD} TASK=${TASK} STEP=${STEP}"
STABLE_DELTA_CHANGE="$(arx4_stable_delta_change_for_profile "${METHOD}" "${TASK}" "${STEP}")" \
  || die "no acceleration profile for METHOD=${METHOD} TASK=${TASK} STEP=${STEP}"
STARTUP_ACTION_GUARD=false
STARTUP_INACTIVE_DIMENSIONS='[7,8,9,10,11,12]'
STARTUP_FIRST_ACTION_DIMENSIONS='[]'
STARTUP_FIRST_ACTION_MAX_DELTA="${MAX_DELTA}"
HOLD_DIMENSIONS_ENABLED=false
HOLD_DIMENSIONS='[0]'
case "${TASK}" in
  pen_insert)
    # Match real rollout semantics: rely on physical bounds plus stateful p95
    # clipping instead of rejecting the unfiltered first policy action.
    STARTUP_ACTION_GUARD=false
    STARTUP_INACTIVE_DIMENSIONS='[]'
    STARTUP_FIRST_ACTION_DIMENSIONS='[]'
    ;;
  cup_stack|cup_upright)
    STARTUP_ACTION_GUARD=false
    STARTUP_INACTIVE_DIMENSIONS='[]'
    STARTUP_FIRST_ACTION_DIMENSIONS='[7,8,9,10,11,12,13]'
    # Match real rollout semantics: the startup guard uses the hard task
    # envelope, while StatefulDynamicsFilter applies the tighter p95 clipping.
    STARTUP_FIRST_ACTION_MAX_DELTA="${MAX_DELTA}"
    HOLD_DIMENSIONS_ENABLED=true
    HOLD_DIMENSIONS='[0,1,2,3,4,5,6]'
    ;;
esac
if [[ "${METHOD}" != "diffusion_policy" ]]; then
  [[ -d "${JEPA_ROOT}/mip" ]] || die "JEPA-Policy source missing: ${JEPA_ROOT}"
  [[ -s "${ASSET_ROOT}/stats/${TASK}/stats.json" ]] || die "task stats missing"
else
  [[ -d "${DIFFUSION_ROOT}/diffusion_policy" ]] || die "Diffusion Policy source missing: ${DIFFUSION_ROOT}"
fi
FFMPEG_REPORT="disabled"
if [[ "${RECORDING_ENABLED}" == "true" ]]; then
  arx4_prepare_recording_env || die "recording preflight failed"
  FFMPEG_REPORT="${ARX4_FFMPEG_REPORT}"
fi

arx4_prepare_runtime_env "${SOURCE_ROOT}" "${ARX5_SDK_ROOT}"
cd "${SOURCE_ROOT}"
export ROS_LOCALHOST_ONLY="${PROMETHEUS_ROS_LOCALHOST_ONLY:-1}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-LOCALHOST}"
export PROMETHEUS_POLICY_DIAGNOSTICS=1

COMMON_OVERRIDES=(
  "robot=arx5_prometheusv2"
  "user.task_id=${TASK}"
  "user.checkpoint=${CHECKPOINT_OVERRIDE}"
  "user.steps=${RUN_CHUNKS}"
  "user.action_steps_per_chunk=${ACTION_STEPS}"
  "user.max_delta=${MAX_DELTA}"
  "user.startup_action_guard_enabled=${STARTUP_ACTION_GUARD}"
  "user.startup_inactive_dimensions=${STARTUP_INACTIVE_DIMENSIONS}"
  "user.startup_first_action_dimensions=${STARTUP_FIRST_ACTION_DIMENSIONS}"
  "user.startup_first_action_max_delta=${STARTUP_FIRST_ACTION_MAX_DELTA}"
  "user.hold_dimensions_enabled=${HOLD_DIMENSIONS_ENABLED}"
  "user.hold_dimensions=${HOLD_DIMENSIONS}"
  "user.dry_run=true"
  "data.recording.enabled=${RECORDING_ENABLED}"
  "policy.inference.device=${DEVICE}"
)
if [[ "${METHOD}" == "diffusion_policy" ]]; then
  GRIPPER_CLOSE_LEAD_STEPS=0
  GRIPPER_CLOSE_HEIGHT_GATE=false
  if [[ "${TASK}" == "pen_insert" ]]; then
    GRIPPER_CLOSE_LEAD_STEPS=7
    GRIPPER_CLOSE_HEIGHT_GATE=true
  fi
  METHOD_OVERRIDES=(
    "policy.inference.project_root=${DIFFUSION_ROOT}"
    "policy.inference.num_inference_steps=${DP_INFERENCE_STEPS}"
    "policy.inference.latency_compensation_steps=${DP_LATENCY_COMPENSATION_STEPS}"
    "user.stable_delta=${STABLE_DELTA}"
    "user.stable_delta_change=${STABLE_DELTA_CHANGE}"
    "user.gripper_close_lead_steps=${GRIPPER_CLOSE_LEAD_STEPS}"
    "user.gripper_close_height_gate_enabled=${GRIPPER_CLOSE_HEIGHT_GATE}"
  )
  VALIDATION_REVISION="${DP_STABILITY_REVISION}"
else
  METHOD_OVERRIDES=(
    "user.model_task=${TASK}_arx_r5_image"
    "user.stats_path=${ASSET_ROOT}/stats/${TASK}/stats.json"
    "policy.inference.project_root=${JEPA_ROOT}"
    "user.stable_delta=${STABLE_DELTA}"
    "user.stable_delta_change=${STABLE_DELTA_CHANGE}"
  )
  VALIDATION_REVISION="${RGB_VALIDATION_REVISION}"
fi

mkdir -p -- "${RESULT_ROOT}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${RESULT_ROOT}/${METHOD}_step${STEP}_live_dry_run_${STAMP}.log"
PASS_MARKER="${RESULT_ROOT}/${METHOD}_step${STEP}_a${ACTION_STEPS}_live_dry_run.passed"
if [[ -e "${PASS_MARKER}" ]]; then
  mv -- "${PASS_MARKER}" "${PASS_MARKER}.stale_${STAMP}"
fi
printf '%s\n' \
  '[arx4_jepa_eval] LIVE INPUT / DRY ACTION MODE' \
  "method=${METHOD} task=${TASK} step=${STEP}" \
  "safety_profile=${METHOD}/${TASK}/step${STEP}:cache_10hz_base5050_v2_task_shared" \
  'X5 can3/can1 and three RealSense cameras will be opened.' \
  "ActionScheduler is dry-run; ${RUN_CHUNKS} chunk(s) x ${ACTION_STEPS} action(s) are checked but not sent." \
  "dp_denoise_steps=${DP_INFERENCE_STEPS} dp_latency_compensation_steps=${DP_LATENCY_COMPENSATION_STEPS}" \
  "recording_enabled=${RECORDING_ENABLED} ffmpeg=${FFMPEG_REPORT}" \
  'Safety overlay disables reset_to_home on cleanup and returns to teach mode.' \
  "log=${LOG_PATH}"

set +e
"${PYTHON_BIN}" -m prometheus.workflows.rollout_sync \
  --config-name "${CONFIG_NAME}" \
  "${COMMON_OVERRIDES[@]}" \
  "${METHOD_OVERRIDES[@]}" 2>&1 | tee "${LOG_PATH}"
status="${PIPESTATUS[0]}"
set -e
if [[ "${status}" -ne 0 ]]; then
  die "live dry-run failed with status ${status}; inspect ${LOG_PATH}"
fi
printf 'method=%s\ntask=%s\nstep=%s\naction_steps=%s\nrun_chunks=%s\nvalidation_revision=%s\nlog=%s\n' \
  "${METHOD}" "${TASK}" "${STEP}" "${ACTION_STEPS}" "${RUN_CHUNKS}" \
  "${VALIDATION_REVISION}" "${LOG_PATH}" \
  >"${PASS_MARKER}"
printf '[arx4_jepa_eval] live dry-run passed: %s\n' "${LOG_PATH}"
printf '[arx4_jepa_eval] validation marker: %s\n' "${PASS_MARKER}"
