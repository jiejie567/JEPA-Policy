#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data_nas/ykj_jepa_policy"
REPO="$ROOT/code/JEPA-Policy"
PYTHON_BIN="$ROOT/venvs/mimicgen/bin/python"
PPU_SITE="$ROOT/venvs/jepa_ppu/lib/python3.12/site-packages"
SYSTEM_SITE="/usr/local/lib/python3.12/site-packages"
MIMICGEN_ROOT="$REPO/third_party/mimicgen"
ROBOSUITE_ROOT="$REPO/third_party/mimicgen_robosuite"
ROBOMIMIC_ROOT="$REPO/third_party/mimicgen_robomimic"
TASK_ZOO_ROOT="$REPO/third_party/mimicgen_task_zoo"
SITECUSTOMIZE_ROOT="$REPO/tools/mimicgen_sitecustomize"
OVERLAY_ROOT="${MIMICGEN_TORCH_OVERLAY_DIR:-/tmp/jepa_mimicgen_torch_overlay}"
EGL_ROOT="$ROOT/venvs/egl_noble_x86_64"
EGL_LIB="$EGL_ROOT/usr/lib/x86_64-linux-gnu"
EGL_DRI="$EGL_LIB/dri"
EGL_JSON="$EGL_ROOT/usr/share/glvnd/egl_vendor.d/50_mesa.json"

[[ -x "$PYTHON_BIN" ]] || { echo "Missing MimicGen Python: $PYTHON_BIN" >&2; exit 1; }
[[ -d "$PPU_SITE/torch" ]] || { echo "Missing verified PPU torch: $PPU_SITE/torch" >&2; exit 1; }

mkdir -p "$OVERLAY_ROOT"
for entry in \
  torch torch-2.6.0.dist-info \
  torchvision torchvision-0.21.0.dist-info \
  torchaudio torchaudio-2.6.0.dist-info \
  functorch torchgen torio triton triton-3.2.0.dist-info \
  sympy mpmath; do
  source_path="$PPU_SITE/$entry"
  target_path="$OVERLAY_ROOT/$entry"
  [[ -e "$source_path" ]] || { echo "Missing overlay source: $source_path" >&2; exit 1; }
  if [[ -L "$target_path" ]]; then
    [[ "$(readlink "$target_path")" == "$source_path" ]] || {
      echo "Unexpected overlay link: $target_path" >&2
      exit 1
    }
  elif [[ -e "$target_path" ]]; then
    echo "Unexpected overlay entry: $target_path" >&2
    exit 1
  else
    ln -s "$source_path" "$target_path"
  fi
done

# W&B is intentionally reused from the verified JEPA environment. Its
# sentry-sdk dependency belongs to the immutable DLC base image rather than the
# JEPA venv, so expose only that dependency through the same narrow overlay.
for entry in sentry_sdk sentry_sdk-*.dist-info; do
  for source_path in "$SYSTEM_SITE"/$entry; do
    [[ -e "$source_path" ]] || continue
    target_path="$OVERLAY_ROOT/$(basename "$source_path")"
    if [[ -L "$target_path" ]]; then
      [[ "$(readlink "$target_path")" == "$source_path" ]] || {
        echo "Unexpected overlay link: $target_path" >&2
        exit 1
      }
    elif [[ -e "$target_path" ]]; then
      echo "Unexpected overlay entry: $target_path" >&2
      exit 1
    else
      ln -s "$source_path" "$target_path"
    fi
  done
done
[[ -d "$OVERLAY_ROOT/sentry_sdk" ]] || {
  echo "Missing sentry-sdk required by the verified W&B client" >&2
  exit 1
}

export PYTHONNOUSERSITE=1
export PYTHONPATH="$SITECUSTOMIZE_ROOT:$OVERLAY_ROOT:$MIMICGEN_ROOT:$ROBOSUITE_ROOT:$ROBOMIMIC_ROOT:$TASK_ZOO_ROOT:$REPO"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export EGL_PLATFORM="${EGL_PLATFORM:-surfaceless}"
export LD_LIBRARY_PATH="$EGL_LIB:${LD_LIBRARY_PATH:-}"
export __EGL_VENDOR_LIBRARY_FILENAMES="$EGL_JSON"
export LIBGL_DRIVERS_PATH="$EGL_DRI"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export MESA_LOADER_DRIVER_OVERRIDE="${MESA_LOADER_DRIVER_OVERRIDE:-llvmpipe}"

exec "$PYTHON_BIN" "$@"
