#!/usr/bin/env bash
set -euo pipefail

# Interactive staged/formal launcher. It is gated by matching live dry-run markers.
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
TASK="${TASK:-pen_insert}"
STEP="${STEP:-100000}"
ACTION_STEPS="${ACTION_STEPS:-1}"
RUN_CHUNKS="${RUN_CHUNKS:-1}"
DEVICE="${DEVICE:-cuda}"
DP_INFERENCE_STEPS="${DP_INFERENCE_STEPS:-100}"
DP_LATENCY_COMPENSATION_STEPS="${DP_LATENCY_COMPENSATION_STEPS:-}"
RESET_HOME_ON_CLOSE="${RESET_HOME_ON_CLOSE:-false}"
RESET_HOME_ON_SUCCESS="${RESET_HOME_ON_SUCCESS:-false}"
RESET_HOME_ON_INTERRUPT="${RESET_HOME_ON_INTERRUPT:-true}"
FORMAL_RUN="${FORMAL_RUN:-false}"
RECORDING_ENABLED="${RECORDING_ENABLED:-false}"
REQUIRE_CONFIRMATION="${REQUIRE_CONFIRMATION:-true}"
WORKFLOW_MODULE="${WORKFLOW_MODULE:-prometheus.workflows.rollout_sync}"
PAPER_RECOVERY_SECONDS="${PAPER_RECOVERY_SECONDS:-30}"
PAPER_MAX_EPISODES="${PAPER_MAX_EPISODES:-0}"
INFERENCE_ROOT="${INFERENCE_ROOT:-${PROJECT_ROOT}/runs}"
RUN_ID="${RUN_ID:-}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"

die() {
  printf '[arx4_jepa_eval] error: %s\n' "$*" >&2
  exit 2
}

[[ -t 0 && -t 1 ]] || die "real rollout must be started from an interactive terminal"
[[ "${METHOD}" =~ ^(jepa|mip|diffusion_policy)$ ]] || die "METHOD must be jepa, mip, or diffusion_policy"
[[ "${TASK}" =~ ^(cabinet|cup_stack|cup_upright|pen_insert|plate_grape)$ ]] || die "unsupported TASK: ${TASK}"
[[ "${STEP}" =~ ^(060000|080000|100000|140000|180000)$ ]] || die "unsupported STEP: ${STEP}"
[[ "${ACTION_STEPS}" =~ ^[1-8]$ ]] || die "ACTION_STEPS must be an integer from 1 to 8"
[[ "${DP_INFERENCE_STEPS}" =~ ^[1-9][0-9]*$ ]] \
  || die "DP_INFERENCE_STEPS must be an integer from 1 to 100"
(( DP_INFERENCE_STEPS <= 100 )) \
  || die "DP_INFERENCE_STEPS must not exceed the 100 training steps"
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
(( DP_EXECUTABLE_ACTIONS >= 1 )) \
  || die "DP latency compensation exhausts the H=10 checkpoint horizon"
[[ "${RUN_CHUNKS}" =~ ^[1-9][0-9]*$ ]] || die "RUN_CHUNKS must be a positive integer"
[[ "${RESET_HOME_ON_CLOSE}" =~ ^(true|false)$ ]] || die "RESET_HOME_ON_CLOSE must be true or false"
[[ "${RESET_HOME_ON_SUCCESS}" =~ ^(true|false)$ ]] || die "RESET_HOME_ON_SUCCESS must be true or false"
[[ "${RESET_HOME_ON_INTERRUPT}" =~ ^(true|false)$ ]] || die "RESET_HOME_ON_INTERRUPT must be true or false"
[[ "${FORMAL_RUN}" =~ ^(true|false)$ ]] || die "FORMAL_RUN must be true or false"
[[ "${RECORDING_ENABLED}" =~ ^(true|false)$ ]] || die "RECORDING_ENABLED must be true or false"
[[ "${REQUIRE_CONFIRMATION}" =~ ^(true|false)$ ]] || die "REQUIRE_CONFIRMATION must be true or false"
[[ "${WORKFLOW_MODULE}" =~ ^prometheus\.workflows\.(rollout_sync|rollout_paper_eval)$ ]] \
  || die "unsupported WORKFLOW_MODULE: ${WORKFLOW_MODULE}"
[[ "${PAPER_RECOVERY_SECONDS}" =~ ^[0-9]+$ ]] || die "PAPER_RECOVERY_SECONDS must be non-negative"
[[ "${PAPER_MAX_EPISODES}" =~ ^[0-9]+$ ]] || die "PAPER_MAX_EPISODES must be non-negative"
[[ -z "${RUN_ID}" || "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || die "RUN_ID contains unsupported characters"
[[ "${RESET_HOME_ON_CLOSE}" != "true" || "${RESET_HOME_ON_SUCCESS}" != "true" ]] \
  || die "choose either RESET_HOME_ON_CLOSE or RESET_HOME_ON_SUCCESS, not both"
[[ "${RESET_HOME_ON_CLOSE}" != "true" || "${RESET_HOME_ON_INTERRUPT}" != "true" ]] \
  || die "choose workflow-owned RESET_HOME_ON_INTERRUPT or driver-owned RESET_HOME_ON_CLOSE, not both"
if [[ "${FORMAL_RUN}" == "true" ]]; then
  FORMAL_ACTION_STEPS=8
  if [[ "${METHOD}" == "diffusion_policy" ]]; then
    FORMAL_ACTION_STEPS="${DP_EXECUTABLE_ACTIONS}"
  fi
  [[ "${ACTION_STEPS}" == "${FORMAL_ACTION_STEPS}" ]] \
    || die "formal ${METHOD} rollout requires latency-aligned ACTION_STEPS=${FORMAL_ACTION_STEPS}"
  (( RUN_CHUNKS >= 2 )) || die "formal RUN_CHUNKS must be at least 2"
  if [[ "${WORKFLOW_MODULE}" != "prometheus.workflows.rollout_paper_eval" ]]; then
    if [[ "${METHOD}/${TASK}" == "diffusion_policy/pen_insert" ]]; then
      (( RUN_CHUNKS * ACTION_STEPS <= 256 )) \
        || die "formal diffusion_policy pen_insert rollout must not exceed 256 total actions"
    else
      (( RUN_CHUNKS * ACTION_STEPS <= 224 )) \
        || die "formal rollout must not exceed 224 total actions"
    fi
  fi
  [[ "${RESET_HOME_ON_CLOSE}" == "false" ]] || die "formal rollout requires workflow-owned ordered reset"
  [[ "${RESET_HOME_ON_SUCCESS}" == "true" ]] || die "formal rollout requires RESET_HOME_ON_SUCCESS=true"
  [[ "${RESET_HOME_ON_INTERRUPT}" == "true" ]] || die "formal rollout requires RESET_HOME_ON_INTERRUPT=true"
  [[ "${RECORDING_ENABLED}" == "true" ]] || die "formal rollout requires RECORDING_ENABLED=true"
else
  [[ "${RUN_CHUNKS}" == "1" ]] || die "staged real rollout is fixed to RUN_CHUNKS=1"
fi
[[ -x "${PYTHON_BIN}" ]] || die "isolated Python is not executable: ${PYTHON_BIN}"
[[ -d "${SOURCE_ROOT}" ]] || die "isolated source not found: ${SOURCE_ROOT}"
[[ -d "${ARX5_SDK_ROOT}/python" ]] || die "isolated ARX5 SDK not found: ${ARX5_SDK_ROOT}"
FFMPEG_REPORT="disabled"
if [[ "${RECORDING_ENABLED}" == "true" ]]; then
  arx4_prepare_recording_env || die "recording preflight failed"
  FFMPEG_REPORT="${ARX4_FFMPEG_REPORT}"
fi
[[ -e /dev/arxcan1 && -e /dev/arxcan3 ]] || die "ARX5 CAN links are missing"
arx4_acquire_hardware_lock \
  || die "can1/can3 are reserved by another ARX5 launcher; close or cancel the other terminal first"

if pgrep -af 'prometheus\.system\.collection_manager|prometheus\.workflows\.(rollout_sync|rollout_paper_eval)' >/dev/null; then
  pgrep -af 'prometheus\.system\.collection_manager|prometheus\.workflows\.(rollout_sync|rollout_paper_eval)' >&2 || true
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
if [[ -n "${CHECKPOINT_PATH}" ]]; then
  [[ "${CHECKPOINT_PATH}" == /* && "${CHECKPOINT_PATH}" != / ]] \
    || die "CHECKPOINT_PATH must be a safe absolute path"
  CHECKPOINT="${CHECKPOINT_PATH}"
fi
[[ -f "${CHECKPOINT}" && -s "${CHECKPOINT}" ]] || die "checkpoint missing: ${CHECKPOINT}"
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
    # Do not reject the raw first action before filtering. The scheduler first
    # checks absolute physical bounds, then rebases from measured qpos/qvel and
    # clips both arms to the pen_insert p95 velocity/acceleration envelope.
    STARTUP_ACTION_GUARD=false
    STARTUP_INACTIVE_DIMENSIONS='[]'
    STARTUP_FIRST_ACTION_DIMENSIONS='[]'
    ;;
  cup_stack|cup_upright)
    # Do not reject an unfiltered first prediction. The stateful dynamics
    # filter rebases it from measured qpos/qvel and clips it before publication.
    STARTUP_ACTION_GUARD=false
    STARTUP_INACTIVE_DIMENSIONS='[]'
    STARTUP_FIRST_ACTION_DIMENSIONS='[7,8,9,10,11,12,13]'
    # Reject only a first action outside the task's hard demonstrated
    # envelope. StatefulDynamicsFilter still clips every command to the
    # tighter training-p95 velocity/acceleration profile from measured qpos.
    STARTUP_FIRST_ACTION_MAX_DELTA="${MAX_DELTA}"
    HOLD_DIMENSIONS_ENABLED=true
    HOLD_DIMENSIONS='[0,1,2,3,4,5,6]'
    ;;
esac

arx4_prepare_runtime_env "${SOURCE_ROOT}" "${ARX5_SDK_ROOT}"
cd "${SOURCE_ROOT}"
if ! RUNTIME_REPORT="$("${PYTHON_BIN}" - <<'PY'
import sys
import numpy
import torch
import typing_extensions
import prometheus.policy.jepa_policy
import rclpy.action
import rclpy.executors

print(
    f"python={sys.version_info.major}.{sys.version_info.minor} "
    f"numpy={numpy.__version__} numpy_path={numpy.__file__} "
    f"typing_extensions_path={typing_extensions.__file__} "
    f"torch={torch.__version__} ros_native=ok"
)
PY
)"; then
  die "isolated Python/JEPA dependency preflight failed before robot startup"
fi

if [[ "${METHOD}" == "diffusion_policy" ]]; then
  [[ -d "${DIFFUSION_ROOT}/diffusion_policy" ]] || die "Diffusion Policy source missing"
  GRIPPER_CLOSE_LEAD_STEPS=0
  GRIPPER_CLOSE_HEIGHT_GATE=false
  if [[ "${TASK}" == "pen_insert" ]]; then
    GRIPPER_CLOSE_LEAD_STEPS=7
    GRIPPER_CLOSE_HEIGHT_GATE=true
  fi
  RESET_ON_GUARD_REJECTION=false
  if [[ "${FORMAL_RUN}" == "true" ]]; then
    RESET_ON_GUARD_REJECTION=true
  fi
  METHOD_OVERRIDES=(
    "policy.inference.project_root=${DIFFUSION_ROOT}"
    "policy.inference.num_inference_steps=${DP_INFERENCE_STEPS}"
    "policy.inference.latency_compensation_steps=${DP_LATENCY_COMPENSATION_STEPS}"
    "user.stable_delta=${STABLE_DELTA}"
    "user.stable_delta_change=${STABLE_DELTA_CHANGE}"
    "user.gripper_close_lead_steps=${GRIPPER_CLOSE_LEAD_STEPS}"
    "user.gripper_close_height_gate_enabled=${GRIPPER_CLOSE_HEIGHT_GATE}"
    "workflow.reset_home_on_guard_rejection=${RESET_ON_GUARD_REJECTION}"
  )
  STABILITY_REPORT="${STABLE_DELTA}"
  STABILITY_ACCEL_REPORT="${STABLE_DELTA_CHANGE}"
  GRIPPER_LEAD_REPORT="${GRIPPER_CLOSE_LEAD_STEPS} steps at 10Hz, closing only"
  GRIPPER_HEIGHT_GATE_REPORT="enabled=${GRIPPER_CLOSE_HEIGHT_GATE} max_eef_z=0.013m replan_on_block=true"
else
  [[ -d "${JEPA_ROOT}/mip" ]] || die "JEPA-Policy source missing"
  [[ -s "${ASSET_ROOT}/stats/${TASK}/stats.json" ]] || die "matching task stats missing"
  METHOD_OVERRIDES=(
    "user.model_task=${TASK}_arx_r5_image"
    "user.stats_path=${ASSET_ROOT}/stats/${TASK}/stats.json"
    "policy.inference.project_root=${JEPA_ROOT}"
    "user.stable_delta=${STABLE_DELTA}"
    "user.stable_delta_change=${STABLE_DELTA_CHANGE}"
  )
  STABILITY_REPORT="${STABLE_DELTA}"
  STABILITY_ACCEL_REPORT="${STABLE_DELTA_CHANGE}"
  GRIPPER_LEAD_REPORT="disabled"
  GRIPPER_HEIGHT_GATE_REPORT="disabled"
fi

TOTAL_ACTIONS=$((RUN_CHUNKS * ACTION_STEPS))
if [[ "${FORMAL_RUN}" == "true" ]]; then
  MODE_LABEL="FORMAL ROBOT MODE"
  if [[ "${WORKFLOW_MODULE}" == "prometheus.workflows.rollout_paper_eval" ]]; then
    RUN_DESCRIPTION="manual S/F/A evaluation with ${ACTION_STEPS} abs_qpos actions per chunk at 10Hz"
  else
    RUN_DESCRIPTION="up to ${RUN_CHUNKS} chunks x ${ACTION_STEPS} actions = ${TOTAL_ACTIONS} abs_qpos actions at 10Hz"
  fi
  EXPECTED_CONFIRMATION="RUN FORMAL ${METHOD} ${TASK} C${RUN_CHUNKS}"
else
  MODE_LABEL="REAL ROBOT MODE"
  RUN_DESCRIPTION="${ACTION_STEPS} abs_qpos action(s) at 10Hz, then stop"
  EXPECTED_CONFIRMATION="RUN ${METHOD} ${TASK} A${ACTION_STEPS}"
fi

printf '%s\n' \
  "[arx4_jepa_eval] ${MODE_LABEL}" \
  'robot=ARX5/X5 can3+can1' \
  "method=${METHOD} task=${TASK} checkpoint_step=${STEP}" \
  "safety_profile=${METHOD}/${TASK}/step${STEP}:cache_10hz_base5050_v2_task_shared" \
  "This run will execute ${RUN_DESCRIPTION}." \
  'Bounds and task-specific max-delta protections remain enabled.' \
  "max_delta_filter=${MAX_DELTA}" \
  "reset_home_on_close=${RESET_HOME_ON_CLOSE}" \
  "reset_home_on_success=${RESET_HOME_ON_SUCCESS}" \
  "reset_home_on_interrupt=${RESET_HOME_ON_INTERRUPT}" \
  'ensure_home_before_rollout=true (auto-reset only when outside tolerance)' \
  'reset_home_on_error=true' \
  "recording_enabled=${RECORDING_ENABLED}" \
  "dp_denoise_steps=$([[ "${METHOD}" == "diffusion_policy" ]] && printf '%s' "${DP_INFERENCE_STEPS}" || printf 'n/a')" \
  "dp_latency_compensation_steps=$([[ "${METHOD}" == "diffusion_policy" ]] && printf '%s' "${DP_LATENCY_COMPENSATION_STEPS}" || printf 'n/a')" \
  "dp_executable_action_horizon=$([[ "${METHOD}" == "diffusion_policy" ]] && printf '%s' "${DP_EXECUTABLE_ACTIONS}" || printf 'n/a')" \
  "stability_rate_limit=${STABILITY_REPORT}" \
  "stability_delta_change_limit=${STABILITY_ACCEL_REPORT}" \
  "hold_dimensions_enabled=${HOLD_DIMENSIONS_ENABLED} dimensions=${HOLD_DIMENSIONS}" \
  "gripper_close_lead=${GRIPPER_LEAD_REPORT}" \
  "gripper_close_height_gate=${GRIPPER_HEIGHT_GATE_REPORT}" \
  'validation_gate=disabled' \
  "reset_home_on_guard_rejection=${RESET_ON_GUARD_REJECTION:-false}" \
  "ffmpeg=${FFMPEG_REPORT}" \
  "runtime=${RUNTIME_REPORT}" \
  'Clear the workspace and keep the emergency stop ready.'
if [[ "${REQUIRE_CONFIRMATION}" == "true" ]]; then
  printf 'Type %s to continue: ' "${EXPECTED_CONFIRMATION}"
  read -r confirmation
  [[ "${confirmation}" == "${EXPECTED_CONFIRMATION}" ]] \
    || die "confirmation did not match; rollout cancelled"
else
  printf '[arx4_jepa_eval] confirmation=dedicated_launcher command=%s\n' "${EXPECTED_CONFIRMATION}"
fi

export ROS_LOCALHOST_ONLY="${PROMETHEUS_ROS_LOCALHOST_ONLY:-1}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-LOCALHOST}"
export PROMETHEUS_POLICY_DIAGNOSTICS=1

EXTRA_WORKFLOW_OVERRIDES=("user.inference_root=${INFERENCE_ROOT}")
if [[ -n "${RUN_ID}" ]]; then
  EXTRA_WORKFLOW_OVERRIDES+=("run.id=${RUN_ID}")
fi
if [[ "${WORKFLOW_MODULE}" == "prometheus.workflows.rollout_paper_eval" ]]; then
  EXTRA_WORKFLOW_OVERRIDES+=(
    "workflow.paper_eval.recovery_seconds=${PAPER_RECOVERY_SECONDS}"
    "workflow.paper_eval.max_episodes=${PAPER_MAX_EPISODES}"
  )
fi

exec "${PYTHON_BIN}" -m "${WORKFLOW_MODULE}" \
  --config-name "${CONFIG_NAME}" \
  "robot=arx5_prometheusv2" \
  "user.task_id=${TASK}" \
  "user.checkpoint=${CHECKPOINT_OVERRIDE}" \
  "user.steps=${RUN_CHUNKS}" \
  "user.action_steps_per_chunk=${ACTION_STEPS}" \
  "user.max_delta=${MAX_DELTA}" \
  "user.startup_action_guard_enabled=${STARTUP_ACTION_GUARD}" \
  "user.startup_inactive_dimensions=${STARTUP_INACTIVE_DIMENSIONS}" \
  "user.startup_first_action_dimensions=${STARTUP_FIRST_ACTION_DIMENSIONS}" \
  "user.startup_first_action_max_delta=${STARTUP_FIRST_ACTION_MAX_DELTA}" \
  "user.hold_dimensions_enabled=${HOLD_DIMENSIONS_ENABLED}" \
  "user.hold_dimensions=${HOLD_DIMENSIONS}" \
  "user.dry_run=false" \
  "robot.robot.reset_home_on_close=${RESET_HOME_ON_CLOSE}" \
  "workflow.ensure_home_before_rollout=true" \
  "workflow.reset_home_on_success=${RESET_HOME_ON_SUCCESS}" \
  "workflow.reset_home_on_interrupt=${RESET_HOME_ON_INTERRUPT}" \
  "workflow.reset_home_on_error=true" \
  "data.recording.enabled=${RECORDING_ENABLED}" \
  "policy.inference.device=${DEVICE}" \
  "${METHOD_OVERRIDES[@]}" \
  "${EXTRA_WORKFLOW_OVERRIDES[@]}"
