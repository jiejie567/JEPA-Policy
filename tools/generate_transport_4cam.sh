#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${JEPA_PYTHON:-/mnt/data_nas/ykj_jepa_policy/venvs/jepa_ppu/bin/python}"
CONVERTER="/mnt/data_nas/ykj_jepa_policy/venvs/jepa_ppu/lib/python3.12/site-packages/robomimic/scripts/dataset_states_to_obs.py"
DATA_DIR="/mnt/data_nas/ykj_jepa_policy/datasets/robomimic/transport/ph"
INPUT_PATH="$DATA_DIR/image.hdf5"
OUTPUT_PATH="$DATA_DIR/image_4cam.hdf5"
OUTPUT_TMP="${OUTPUT_PATH}.inprogress"

[[ -f "$INPUT_PATH" ]] || { echo "Missing input: $INPUT_PATH" >&2; exit 1; }
[[ ! -e "$OUTPUT_PATH" ]] || { echo "Refusing to overwrite: $OUTPUT_PATH" >&2; exit 1; }
[[ ! -e "$OUTPUT_TMP" ]] || { echo "Refusing to overwrite incomplete output: $OUTPUT_TMP" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "Python is not executable: $PYTHON_BIN" >&2; exit 1; }
[[ -f "$CONVERTER" ]] || { echo "Missing converter: $CONVERTER" >&2; exit 1; }

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/jepa_matplotlib}"
EGL_RUNTIME_ROOT="${JEPA_EGL_RUNTIME_ROOT:-/mnt/data_nas/ykj_jepa_policy/venvs/egl_noble_x86_64}"
EGL_LIBRARY_DIR="$EGL_RUNTIME_ROOT/usr/lib/x86_64-linux-gnu"
EGL_DRI_DIR="$EGL_LIBRARY_DIR/dri"
EGL_VENDOR_JSON="$EGL_RUNTIME_ROOT/usr/share/glvnd/egl_vendor.d/50_mesa.json"

# Prefer NVIDIA EGL on a GPU node, but retain a CPU-only OSMesa fallback for
# DSW instances without NVIDIA graphics libraries. Override explicitly with
# JEPA_RENDER_BACKEND=egl or JEPA_RENDER_BACKEND=osmesa when needed.
RENDER_BACKEND="${JEPA_RENDER_BACKEND:-auto}"
if [[ "$RENDER_BACKEND" == "auto" ]]; then
  LIBRARY_CACHE="$(ldconfig -p 2>/dev/null || true)"
  if [[ "$LIBRARY_CACHE" == *libEGL_nvidia.so* ]]; then
    RENDER_BACKEND="egl"
  elif [[ "$LIBRARY_CACHE" == *libOSMesa.so* ]]; then
    RENDER_BACKEND="osmesa"
  else
    cat >&2 <<'EOF'
No supported MuJoCo offscreen-rendering library was found.
On an NVIDIA GPU node, expose the NVIDIA EGL driver libraries.
For the CPU fallback on Ubuntu, install: apt-get install libosmesa6
EOF
    exit 1
  fi
fi

case "$RENDER_BACKEND" in
  egl|osmesa) ;;
  *) echo "JEPA_RENDER_BACKEND must be auto, egl, or osmesa (got: $RENDER_BACKEND)" >&2; exit 1 ;;
esac

export MUJOCO_GL="$RENDER_BACKEND"
export PYOPENGL_PLATFORM="$RENDER_BACKEND"

EGL_IMPLEMENTATION=""
if [[ "$RENDER_BACKEND" == "egl" ]]; then
  LIBRARY_CACHE="$(ldconfig -p 2>/dev/null || true)"
  if [[ "$LIBRARY_CACHE" == *libEGL_nvidia.so* ]]; then
    EGL_IMPLEMENTATION="native NVIDIA"
  else
    for required_file in \
      "$EGL_LIBRARY_DIR/libEGL.so.0" \
      "$EGL_LIBRARY_DIR/libEGL.so.1" \
      "$EGL_LIBRARY_DIR/libEGL_mesa.so.0" \
      "$EGL_DRI_DIR/swrast_dri.so" \
      "$EGL_VENDOR_JSON"; do
      [[ -e "$required_file" ]] || {
        echo "Missing bundled Mesa EGL runtime file: $required_file" >&2
        exit 1
      }
    done
    export EGL_PLATFORM="${EGL_PLATFORM:-surfaceless}"
    export LD_LIBRARY_PATH="$EGL_LIBRARY_DIR:${LD_LIBRARY_PATH:-}"
    export __EGL_VENDOR_LIBRARY_FILENAMES="$EGL_VENDOR_JSON"
    export LIBGL_DRIVERS_PATH="$EGL_DRI_DIR"
    export LIBGL_ALWAYS_SOFTWARE=1
    export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
    export LP_NUM_THREADS="${LP_NUM_THREADS:-8}"
    EGL_IMPLEMENTATION="bundled Mesa llvmpipe"
  fi
fi

echo "Generating Transport four-camera dataset"
echo "Input:  $INPUT_PATH"
echo "Output: $OUTPUT_PATH"
echo "Temporary output: $OUTPUT_TMP"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<inherited>}"
echo "Rendering backend: $RENDER_BACKEND"
[[ -z "$EGL_IMPLEMENTATION" ]] || echo "EGL implementation: $EGL_IMPLEMENTATION"
if [[ "$RENDER_BACKEND" == "osmesa" ]]; then
  echo "Warning: OSMesa uses CPU rendering and may be much slower than NVIDIA EGL." >&2
fi

cd "$REPO_ROOT"
"$PYTHON_BIN" "$CONVERTER" \
  --dataset "$INPUT_PATH" \
  --output_name "$(basename "$OUTPUT_TMP")" \
  --camera_names \
    shouldercamera0 \
    shouldercamera1 \
    robot0_eye_in_hand \
    robot1_eye_in_hand \
  --camera_height 84 \
  --camera_width 84 \
  --done_mode 2 \
  --copy_rewards \
  --copy_dones \
  --exclude-next-obs \
  --compress

INPUT_PATH="$INPUT_PATH" OUTPUT_TMP="$OUTPUT_TMP" "$PYTHON_BIN" - <<'PY'
import os

import h5py

camera_keys = {
    "shouldercamera0_image",
    "shouldercamera1_image",
    "robot0_eye_in_hand_image",
    "robot1_eye_in_hand_image",
}
with h5py.File(os.environ["INPUT_PATH"], "r") as source, h5py.File(os.environ["OUTPUT_TMP"], "r") as output:
    source_demos = set(source["data"].keys())
    output_demos = set(output["data"].keys())
    assert output_demos == source_demos, (len(output_demos), len(source_demos))
    expected_total = sum(len(source[f"data/{demo}/actions"]) for demo in source_demos)
    assert int(output["data"].attrs["total"]) == expected_total
    for demo in output_demos:
        group = output[f"data/{demo}"]
        assert "next_obs" not in group, demo
        assert camera_keys.issubset(group["obs"].keys()), (demo, sorted(group["obs"].keys()))
        samples = len(group["actions"])
        for key in camera_keys:
            images = group[f"obs/{key}"]
            assert images.shape == (samples, 84, 84, 3), (demo, key, images.shape)
            assert images.dtype == "uint8", (demo, key, images.dtype)
print(f"Validated {len(output_demos)} demos and {expected_total} samples")
PY

mv -- "$OUTPUT_TMP" "$OUTPUT_PATH"
echo "Completed: $OUTPUT_PATH"
