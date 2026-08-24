#!/usr/bin/env bash

# Run the complete 2-task x 4-setting ablation2 matrix on one 16-GPU node.
# GPU allocation: MugMug 0-3, MokaMoka 4-7.

set -u -o pipefail

BENCHMARK="libero"
EXPECTED_IMAGE_SIZE=128
TASKS=(
  mug_mug_image
  moka_moka_image
)

declare -A DATASET_PATHS=(
  [mug_mug_image]="/mnt/data_nas/ykj_jepa_policy/datasets/libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo.hdf5"
  [moka_moka_image]="/mnt/data_nas/ykj_jepa_policy/datasets/libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5"
)

declare -A IMAGE_KEYS=(
  [mug_mug_image]="agentview_rgb,eye_in_hand_rgb"
  [moka_moka_image]="agentview_rgb,eye_in_hand_rgb"
)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=tools/ablation2_launcher_common.sh
source "$SCRIPT_DIR/tools/ablation2_launcher_common.sh"
