#!/usr/bin/env bash

# Run the complete 3-task x 4-setting ablation2 matrix on one 16-GPU node.
# GPU allocation: Tool Hang 0-3, Transport 4-7, Square 8-11.

set -u -o pipefail

BENCHMARK="robomimic"
EXPECTED_IMAGE_SIZE=84
SEED="${SEED:-42}"
SKIP_TRANSPORT="${SKIP_TRANSPORT:-0}"

# Preserve the already-running seed-42 matrix exactly. New seeds use the
# corrected four-camera Transport setup and must wait for its dataset.
if [[ "$SEED" == "42" ]]; then
  TRANSPORT_TASK="transport_ph_image"
else
  TRANSPORT_TASK="transport_ph_image_4cam"
fi

case "$SKIP_TRANSPORT" in
  0)
    TASKS=(tool_hang_ph_image "$TRANSPORT_TASK" square_ph_image)
    ;;
  1)
    TASKS=(tool_hang_ph_image square_ph_image)
    ;;
  *)
    echo "ERROR: SKIP_TRANSPORT must be 0 or 1" >&2
    exit 2
    ;;
esac

declare -A DATASET_PATHS=(
  [tool_hang_ph_image]="/mnt/data_nas/ykj_jepa_policy/datasets/robomimic/tool_hang/ph/image.hdf5"
  [transport_ph_image]="/mnt/data_nas/ykj_jepa_policy/datasets/robomimic/transport/ph/image.hdf5"
  [transport_ph_image_4cam]="/mnt/data_nas/ykj_jepa_policy/datasets/robomimic/transport/ph/image_4cam.hdf5"
  [square_ph_image]="/mnt/data_nas/ykj_jepa_policy/datasets/robomimic/square/ph/image.hdf5"
)

declare -A IMAGE_KEYS=(
  [tool_hang_ph_image]="sideview_image,robot0_eye_in_hand_image"
  [transport_ph_image]="robot1_eye_in_hand_image"
  [transport_ph_image_4cam]="shouldercamera0_image,shouldercamera1_image,robot0_eye_in_hand_image,robot1_eye_in_hand_image"
  [square_ph_image]="agentview_image,robot0_eye_in_hand_image"
)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=tools/ablation2_launcher_common.sh
source "$SCRIPT_DIR/tools/ablation2_launcher_common.sh"
