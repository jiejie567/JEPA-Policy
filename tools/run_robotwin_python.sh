#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data_nas/ykj_jepa_policy"
REPO="$ROOT/code/JEPA-Policy"
PYTHON_BIN="${ROBOTWIN_PYTHON_BIN:-$ROOT/venvs/robotwin/bin/python}"
DEPENDENCY_SITE="${ROBOTWIN_DEPENDENCY_SITE:-}"
SYSTEM_LIB_DIR="${ROBOTWIN_SYSTEM_LIB_DIR:-}"
ROBOTWIN_ROOT="$REPO/third_party/robotwin"
ROBOTWIN_ROOT="${ROBOTWIN_SOURCE_ROOT:-$ROBOTWIN_ROOT}"
CACHE_ROOT="${ROBOTWIN_CACHE_ROOT:-$ROOT/cache/robotwin}"
PPU_SITE="$ROOT/venvs/jepa_ppu/lib/python3.12/site-packages"
PPU_OVERLAY="${ROBOTWIN_TORCH_OVERLAY_DIR:-/tmp/jepa_robotwin_torch_overlay}"
PINNED_COMMIT="c3ddfa8b97d5519efa828b075999bd0006778e5e"

[[ -x "$PYTHON_BIN" ]] || {
  echo "Missing isolated RoboTwin Python: $PYTHON_BIN" >&2
  exit 1
}
[[ -d "$ROBOTWIN_ROOT/envs" ]] || {
  echo "Missing pinned RoboTwin checkout: $ROBOTWIN_ROOT" >&2
  exit 1
}
[[ "$(git -C "$ROBOTWIN_ROOT" rev-parse HEAD)" == "$PINNED_COMMIT" ]] || {
  echo "RoboTwin checkout is not at pinned commit $PINNED_COMMIT" >&2
  exit 1
}

mkdir -p "$CACHE_ROOT"/{xdg,matplotlib,huggingface,numba}
unset VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONHOME
export PYTHONNOUSERSITE=1
if [[ -n "$SYSTEM_LIB_DIR" ]]; then
  [[ -f "$SYSTEM_LIB_DIR/libX11.so.6" &&
    -f "$SYSTEM_LIB_DIR/libXext.so.6" &&
    -f "$SYSTEM_LIB_DIR/libEGL.so.1" &&
    -f "$SYSTEM_LIB_DIR/libGL.so.1" &&
    -f "$SYSTEM_LIB_DIR/libGLX.so.0" &&
    -f "$SYSTEM_LIB_DIR/libGLdispatch.so.0" &&
    -f "$SYSTEM_LIB_DIR/libglib-2.0.so.0" &&
    -f "$SYSTEM_LIB_DIR/libgthread-2.0.so.0" &&
    -f "$SYSTEM_LIB_DIR/libpcre2-8.so.0" &&
    -f "$SYSTEM_LIB_DIR/libvulkan.so.1" ]] || {
    echo "Missing packaged RoboTwin system libraries: $SYSTEM_LIB_DIR" >&2
    exit 1
  }
  export LD_LIBRARY_PATH="$SYSTEM_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if [[ "${ROBOTWIN_USE_PPU_TORCH:-1}" == "1" ]]; then
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
    sympy mpmath \
    numba numba-0.61.2.dist-info \
    llvmlite llvmlite-0.44.0.dist-info \
    dill dill-0.3.6.dist-info; do
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
  export PYTHONPATH="$PPU_OVERLAY${DEPENDENCY_SITE:+:$DEPENDENCY_SITE}:$REPO:$ROBOTWIN_ROOT"
else
  export PYTHONPATH="${DEPENDENCY_SITE:+$DEPENDENCY_SITE:}$REPO:$ROBOTWIN_ROOT"
fi

export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
export HF_HOME="$CACHE_ROOT/huggingface"
export NUMBA_CACHE_DIR="$CACHE_ROOT/numba"
export TOKENIZERS_PARALLELISM=false
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility,graphics}"
if [[ -n "$DEPENDENCY_SITE" &&
  -f "$DEPENDENCY_SITE/sapien/vulkan_library/nvidia_icd.json" &&
  -f "$DEPENDENCY_SITE/sapien/vulkan_library/10_nvidia.json" ]]; then
  # PAI Lingjun injects the NVIDIA driver libraries into the container without
  # installing the distro ICD JSON files (or the usual /usr/share/glvnd
  # directory). Point the Vulkan/EGL loaders at SAPIEN's equivalent JSON files;
  # their library names are resolved through the injected driver library path.
  export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-$DEPENDENCY_SITE/sapien/vulkan_library/nvidia_icd.json}"
  export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-$DEPENDENCY_SITE/sapien/vulkan_library/10_nvidia.json}"
fi

cd "$REPO"
exec "$PYTHON_BIN" "$@"
