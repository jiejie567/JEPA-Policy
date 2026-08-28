#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/runtime_env.sh"
ASSET_ROOT="${ASSET_ROOT:-${PROJECT_ROOT}}"
SOURCE_ROOT="${SOURCE_ROOT:-${ASSET_ROOT}/runtime/prometheus}"
ARX5_SDK_ROOT="${ARX5_SDK_ROOT:-${ASSET_ROOT}/vendor/arx5-sdk}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
REQUIRE_CONFIRMATION="${REQUIRE_CONFIRMATION:-true}"

die() {
  printf '[arx4_jepa_eval] error: %s\n' "$*" >&2
  exit 2
}

[[ -t 0 && -t 1 ]] || die "reset must be started from an interactive terminal"
[[ "${REQUIRE_CONFIRMATION}" =~ ^(true|false)$ ]] \
  || die "REQUIRE_CONFIRMATION must be true or false"
[[ -x "${PYTHON_BIN}" ]] || die "isolated Python is not executable: ${PYTHON_BIN}"
[[ -d "${SOURCE_ROOT}" ]] || die "isolated source not found: ${SOURCE_ROOT}"
[[ -d "${ARX5_SDK_ROOT}/python" ]] || die "isolated ARX5 SDK not found: ${ARX5_SDK_ROOT}"
[[ -e /dev/arxcan1 && -e /dev/arxcan3 ]] || die "ARX5 CAN links are missing"
arx4_acquire_hardware_lock \
  || die "can1/can3 are reserved by another ARX5 launcher; close or cancel the other terminal first"

if pgrep -af 'prometheus\.system\.collection_manager|prometheus\.workflows\.rollout_sync' >/dev/null; then
  pgrep -af 'prometheus\.system\.collection_manager|prometheus\.workflows\.rollout_sync' >&2 || true
  die "another robot owner or rollout process is running"
fi

printf '%s\n' \
  '[arx4_jepa_eval] RESET HOME MODE' \
  'Both X5 arms will move to home, then enter teach/damping.' \
  'Confirm that neither arm is holding an object and the return path is clear.'
if [[ "${REQUIRE_CONFIRMATION}" == "true" ]]; then
  printf 'Type RESET HOME to continue: '
  read -r confirmation
  [[ "${confirmation}" == "RESET HOME" ]] || die "confirmation did not match; reset cancelled"
else
  printf '[arx4_jepa_eval] confirmation=disabled_by_explicit_command\n'
fi

arx4_prepare_runtime_env "${SOURCE_ROOT}" "${ARX5_SDK_ROOT}"
cd "${SOURCE_ROOT}"
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/reset_home.py"
