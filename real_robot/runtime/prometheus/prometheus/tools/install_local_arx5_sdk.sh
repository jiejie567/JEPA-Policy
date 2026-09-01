#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SDK_ROOT="${ARX5_SDK_ROOT:-${PROJECT_ROOT}/third_party/arx5-sdk}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
BUILD_DIR="${ARX5_BUILD_DIR:-${SDK_ROOT}/build}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"

ENV_PREFIX="$("${PYTHON_BIN}" -c 'import sys; print(sys.prefix)')"
SITE_PACKAGES="$("${PYTHON_BIN}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
SOEM_LIBRARY="${ENV_PREFIX}/lib/libsoem.so"
PTH_PATH="${SITE_PACKAGES}/prometheus_local_arx5_sdk.pth"

if [[ ! -f "${SDK_ROOT}/CMakeLists.txt" ]]; then
    echo "ARX5 SDK source not found: ${SDK_ROOT}" >&2
    exit 1
fi

if [[ ! -f "${SOEM_LIBRARY}" ]]; then
    echo "SOEM library not found: ${SOEM_LIBRARY}" >&2
    echo "Install it in the active environment with: conda install -c conda-forge soem=1.4.0" >&2
    exit 1
fi

if ! nm -D "${SOEM_LIBRARY}" | grep -qE '[[:space:]]EcatError$'; then
    echo "The SOEM library in ${ENV_PREFIX} is not compatible with ARX5 SDK." >&2
    echo "Install SOEM 1.4.0 before rebuilding." >&2
    exit 1
fi

echo "[ARX5] Python: ${PYTHON_BIN}"
echo "[ARX5] Environment: ${ENV_PREFIX}"
echo "[ARX5] Source: ${SDK_ROOT}"
echo "[ARX5] Build: ${BUILD_DIR}"

cmake \
    -S "${SDK_ROOT}" \
    -B "${BUILD_DIR}" \
    -DCMAKE_INSTALL_PREFIX="${ENV_PREFIX}" \
    -DPython3_EXECUTABLE="${PYTHON_BIN}"

cmake --build "${BUILD_DIR}" --parallel "${BUILD_JOBS}"

ARX5_EXTENSION="$(find "${SDK_ROOT}/python" -maxdepth 1 -type f \
    -name 'arx5_interface*.so' -print -quit)"
if [[ -z "${ARX5_EXTENSION}" ]]; then
    echo "ARX5 Python extension was not produced under ${SDK_ROOT}/python" >&2
    exit 1
fi

printf '%s\n' \
    "import sys; p = '${SDK_ROOT}/python'; sys.path.insert(0, p) if p not in sys.path else None" \
    > "${PTH_PATH}"

"${PYTHON_BIN}" -c \
    'import arx5_interface; print("arx5_interface:", arx5_interface.__file__)'

echo "[ARX5] Local SDK installation complete."
