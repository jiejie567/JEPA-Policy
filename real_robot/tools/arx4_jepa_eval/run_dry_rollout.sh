#!/usr/bin/env bash
set -euo pipefail

# Isolated model dry-run. It does not open CAN, ROS, or RealSense devices.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
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
DP_INFERENCE_STEPS="${DP_INFERENCE_STEPS:-100}"
DP_LATENCY_COMPENSATION_STEPS="${DP_LATENCY_COMPENSATION_STEPS:-}"
RESULT_ROOT="${RESULT_ROOT:-${ASSET_ROOT}/validation/${TASK}}"

die() {
  printf '[arx4_jepa_eval] error: %s\n' "$*" >&2
  exit 2
}

[[ -d "${SOURCE_ROOT}" ]] || die "isolated source not extracted: ${SOURCE_ROOT}"
[[ -d "${JEPA_ROOT}" ]] || die "JEPA-Policy source not found: ${JEPA_ROOT}"
[[ -d "${DIFFUSION_ROOT}" ]] || die "Diffusion Policy source not found: ${DIFFUSION_ROOT}"
[[ -d "${ARX5_SDK_ROOT}/python" ]] || die "isolated ARX5 SDK not found: ${ARX5_SDK_ROOT}"
[[ -x "${PYTHON_BIN}" ]] || die "isolated Python is not executable: ${PYTHON_BIN}"
[[ "${METHOD}" =~ ^(jepa|mip|diffusion_policy)$ ]] || die "METHOD must be jepa, mip, or diffusion_policy"
[[ "${TASK}" =~ ^(cabinet|cup_stack|cup_upright|pen_insert|plate_grape)$ ]] || die "unsupported TASK: ${TASK}"
[[ "${STEP}" =~ ^(060000|080000|100000|140000|180000)$ ]] || die "unsupported STEP: ${STEP}"
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
[[ -s "${ASSET_ROOT}/stats/${TASK}/stats.json" ]] || die "stats missing"

case "${METHOD}" in
  jepa|mip)
    CHECKPOINT="${ASSET_ROOT}/checkpoints/${METHOD}/${TASK}/model_step${STEP}.pt"
    ;;
  diffusion_policy)
    CHECKPOINT="${ASSET_ROOT}/checkpoints/diffusion_policy/${TASK}/step=${STEP}.ckpt"
    ;;
esac
[[ -s "${CHECKPOINT}" ]] || die "checkpoint missing: ${CHECKPOINT}"

arx4_prepare_runtime_env "${SOURCE_ROOT}" "${ARX5_SDK_ROOT}"
cd "${SOURCE_ROOT}"
export ROS_LOCALHOST_ONLY="${PROMETHEUS_ROS_LOCALHOST_ONLY:-1}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-LOCALHOST}"

missing_imports="$(${PYTHON_BIN} - <<'PY'
import importlib

required = (
    "hydra",
    "omegaconf",
    "torch",
    "arx5_interface",
)
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {type(exc).__name__}: {exc}")
print("\n".join(missing))
PY
)"
if [[ -n "${missing_imports}" ]]; then
  die "isolated runtime imports failed:\n${missing_imports}"
fi

mkdir -p -- "${RESULT_ROOT}"
RESULT="${RESULT_ROOT}/${METHOD}_step${STEP}_offline_dry_run.json"
printf '[arx4_jepa_eval] offline dry-run method=%s task=%s step=%s\n' \
  "${METHOD}" "${TASK}" "${STEP}"
printf '[arx4_jepa_eval] no CAN, ROS, camera, or robot command will be opened\n'

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/offline_policy_dry_run.py" \
  --method "${METHOD}" \
  --task "${TASK}" \
  --step "${STEP}" \
  --device "${DEVICE}" \
  --denoise-steps "${DP_INFERENCE_STEPS}" \
  --latency-compensation-steps "${DP_LATENCY_COMPENSATION_STEPS}" \
  --asset-root "${ASSET_ROOT}" \
  --source-root "${SOURCE_ROOT}" \
  --jepa-root "${JEPA_ROOT}" \
  --diffusion-root "${DIFFUSION_ROOT}" \
  --result "${RESULT}"
