#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data_nas/ykj_jepa_policy"
REPO="$ROOT/code/JEPA-Policy"
PYTHON_BIN="$ROOT/venvs/robocasa/bin/python"
ROBOCASA_ROOT="$REPO/third_party/robocasa_v1_0_1"
ROBOSUITE_ROOT="$REPO/third_party/robocasa_robosuite"
CACHE_ROOT="${ROBOCASA_CACHE_ROOT:-$ROOT/cache/robocasa}"
PPU_SITE="$ROOT/venvs/jepa_ppu/lib/python3.12/site-packages"
PPU_OVERLAY="${ROBOCASA_TORCH_OVERLAY_DIR:-/tmp/jepa_robocasa_torch_overlay}"
EGL_ROOT="$ROOT/venvs/egl_noble_x86_64"
EGL_LIB="$EGL_ROOT/usr/lib/x86_64-linux-gnu"
EGL_DRI="$EGL_LIB/dri"
EGL_JSON="$EGL_ROOT/usr/share/glvnd/egl_vendor.d/50_mesa.json"

[[ -x "$PYTHON_BIN" ]] || {
  echo "Missing isolated RoboCasa Python: $PYTHON_BIN" >&2
  exit 1
}
[[ -d "$ROBOCASA_ROOT/robocasa" ]] || {
  echo "Missing pinned RoboCasa checkout: $ROBOCASA_ROOT" >&2
  exit 1
}
[[ -d "$ROBOSUITE_ROOT/robosuite" ]] || {
  echo "Missing isolated robosuite checkout: $ROBOSUITE_ROOT" >&2
  exit 1
}
[[ -f "$EGL_LIB/libEGL.so.1" && -d "$EGL_DRI" && -f "$EGL_JSON" ]] || {
  echo "Missing isolated EGL runtime: $EGL_ROOT" >&2
  exit 1
}
[[ "$(git -C "$ROBOCASA_ROOT" rev-parse HEAD)" == \
  "$(cat "$ROBOCASA_ROOT/.robocasa_pinned_commit")" ]] || {
  echo "RoboCasa checkout does not match its pinned commit" >&2
  exit 1
}
[[ "$(git -C "$ROBOSUITE_ROOT" rev-parse HEAD)" == \
  "$(cat "$ROBOSUITE_ROOT/.robocasa_pinned_commit")" ]] || {
  echo "RoboCasa robosuite checkout does not match its pinned commit" >&2
  exit 1
}

mkdir -p "$CACHE_ROOT"/{xdg,matplotlib,numba,huggingface}

unset VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONHOME
# A physical CUDA id is not an EGL id after CUDA visibility is narrowed.
# Discard a raw MuJoCo override, but preserve the launcher's explicit
# process-local JEPA EGL id (0 on the DLC Mesa runtime).
unset MUJOCO_EGL_DEVICE_ID
export PYTHONNOUSERSITE=1
if [[ "${ROBOCASA_USE_PPU_TORCH:-0}" == "1" ]]; then
  [[ -d "$PPU_SITE/torch" ]] || {
    echo "Missing verified PPU torch: $PPU_SITE/torch" >&2
    exit 1
  }
  mkdir -p "$PPU_OVERLAY"
  for entry in \
    torch torch-2.6.0.dist-info \
    torchvision torchvision-0.21.0.dist-info \
    torchaudio torchaudio-2.6.0.dist-info \
    functorch torchgen torio triton triton-3.2.0.dist-info \
    sympy mpmath; do
    source_path="$PPU_SITE/$entry"
    target_path="$PPU_OVERLAY/$entry"
    [[ -e "$source_path" ]] || {
      echo "Missing PPU overlay source: $source_path" >&2
      exit 1
    }
    if [[ -L "$target_path" ]]; then
      [[ "$(readlink "$target_path")" == "$source_path" ]] || {
        echo "Unexpected PPU overlay link: $target_path" >&2
        exit 1
      }
    elif [[ -e "$target_path" ]]; then
      echo "Unexpected PPU overlay entry: $target_path" >&2
      exit 1
    else
      ln -s "$source_path" "$target_path"
    fi
  done
  export PYTHONPATH="$PPU_OVERLAY:$REPO:$ROBOCASA_ROOT:$ROBOSUITE_ROOT"
else
  export PYTHONPATH="$REPO:$ROBOCASA_ROOT:$ROBOSUITE_ROOT"
fi
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
export NUMBA_CACHE_DIR="$CACHE_ROOT/numba"
export HF_HOME="$CACHE_ROOT/huggingface"
export ROBOCASA_DATASET_BASE_PATH="$REPO/datasets/robocasa"
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export EGL_PLATFORM="surfaceless"
# Keep the isolated Mesa runtime first. DLC injects its NVIDIA driver through
# LD_LIBRARY_PATH (the concrete directory is node-dependent), so retain system
# entries while dropping paths from all project virtual environments/checkouts.
SYSTEM_LIBRARY_DIRS=()
IFS=: read -ra inherited_library_dirs <<<"${LD_LIBRARY_PATH:-}"
for library_dir in "${inherited_library_dirs[@]}"; do
  [[ -n "$library_dir" ]] || continue
  case "$library_dir" in
    "$ROOT"/venvs/*|"$REPO"/*) continue ;;
  esac
  SYSTEM_LIBRARY_DIRS+=("$library_dir")
done
for driver_dir in /usr/local/nvidia/lib64 /usr/local/nvidia/lib; do
  [[ -d "$driver_dir" ]] && SYSTEM_LIBRARY_DIRS+=("$driver_dir")
done
if (( ${#SYSTEM_LIBRARY_DIRS[@]} )); then
  export LD_LIBRARY_PATH="$EGL_LIB:$(IFS=:; echo "${SYSTEM_LIBRARY_DIRS[*]}")"
else
  export LD_LIBRARY_PATH="$EGL_LIB"
fi
export __EGL_VENDOR_LIBRARY_FILENAMES="$EGL_JSON"
export LIBGL_DRIVERS_PATH="$EGL_DRI"
export LIBGL_ALWAYS_SOFTWARE="1"
export MESA_LOADER_DRIVER_OVERRIDE="llvmpipe"

exec "$PYTHON_BIN" "$@"
