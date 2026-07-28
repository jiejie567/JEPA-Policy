#!/usr/bin/env bash
set -euo pipefail

VENV="/mnt/data_nas/ykj_jepa_policy/venvs/jepa_ppu"
REPO="/mnt/data_nas/ykj_jepa_policy/code/JEPA-Policy"
CONFIG="${1:?请在命令末尾传入 Hydra config.yaml 的完整路径}"

export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export EGL_PLATFORM=surfaceless

export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"

echo "=================================================="
echo "=== Runtime environment ==="
echo "=================================================="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "MUJOCO_GL=${MUJOCO_GL}"
echo "PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM}"
echo "MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID}"
echo "EGL_PLATFORM=${EGL_PLATFORM}"
echo

echo "=================================================="
echo "=== Accelerator check ==="
echo "=================================================="
nvidia-smi || true
echo

echo "=================================================="
echo "=== Install EGL / Mesa runtime ==="
echo "=================================================="
export DEBIAN_FRONTEND=noninteractive

apt-get update

apt-get install -y --no-install-recommends \
  libegl1 \
  libegl-mesa0 \
  libgl1 \
  libglx0 \
  libglx-mesa0 \
  libopengl0 \
  libgl1-mesa-dri \
  libgbm1 \
  libosmesa6

ldconfig

echo
echo "=================================================="
echo "=== Build EGL compatibility link ==="
echo "=================================================="
EGL_COMPAT_DIR="/tmp/egl_compat"
mkdir -p "$EGL_COMPAT_DIR"

EGL_SO="$(
  ldconfig -p | awk '$1 == "libEGL.so.1" {print $NF; exit}'
)"

[ -n "$EGL_SO" ] || {
  echo "ERROR: libEGL.so.1 is unavailable after apt installation"
  exit 3
}

ln -sf "$EGL_SO" "$EGL_COMPAT_DIR/libEGL.so.0"
export LD_LIBRARY_PATH="$EGL_COMPAT_DIR:${LD_LIBRARY_PATH:-}"

echo "EGL compatibility target: $EGL_SO"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

echo
echo "=================================================="
echo "=== Python EGL / MuJoCo preflight ==="
echo "=================================================="
"$VENV/bin/python" - <<'PY'
import ctypes
import os
import sys
import traceback

for lib in ("libEGL.so.0", "libEGL.so.1", "libGLX.so.0", "libOpenGL.so.0"):
    try:
        ctypes.CDLL(lib)
        print(f"OK: {lib}")
    except OSError as exc:
        print(f"FAILED: {lib} -> {exc}")
        sys.exit(20)

try:
    from OpenGL import EGL
    print("OK: from OpenGL import EGL")
    import mujoco
    print("OK: mujoco import =", mujoco.__version__)
except Exception:
    traceback.print_exc()
    sys.exit(21)
PY

echo
echo "=================================================="
echo "=== Start env-only benchmark ==="
echo "=================================================="

cd "$REPO"

"$VENV/bin/python" "$REPO/tools/bench_square_env.py" \
  --config "$CONFIG" \
  --warmup-steps 5 \
  --benchmark-steps 100 \
  --env-counts 1 4
