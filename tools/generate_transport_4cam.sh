#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${JEPA_PYTHON:-python}"
DATA_DIR="${TRANSPORT_DATA_DIR:?Set TRANSPORT_DATA_DIR to the robomimic transport/ph directory}"
INPUT="${TRANSPORT_LOWDIM_PATH:-$DATA_DIR/low_dim.hdf5}"
OUTPUT="${TRANSPORT_4CAM_PATH:-$DATA_DIR/image_4cam.hdf5}"

"$PYTHON_BIN" -m robomimic.scripts.dataset_states_to_obs \
  --dataset "$INPUT" \
  --output_name "$(basename "$OUTPUT")" \
  --done_mode 2 \
  --camera_names agentview robot0_eye_in_hand shoulder_left shoulder_right \
  --camera_height 84 \
  --camera_width 84

echo "Created $OUTPUT"
