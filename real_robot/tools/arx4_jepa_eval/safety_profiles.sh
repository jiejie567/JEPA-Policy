#!/usr/bin/env bash

# 10Hz per-command task profiles: demonstrated 99.9th percentile + 10%.
# Arm values are radians; gripper values are meters. A 0.005 floor is used
# for task arms that remained effectively stationary in the demonstrations.
arx4_max_delta_for_task() {
  case "$1" in
    cabinet)
      printf '%s\n' '[0.080,0.161,0.187,0.164,0.092,0.080,0.020,0.058,0.141,0.182,0.158,0.067,0.061,0.030]'
      ;;
    cup_stack)
      printf '%s\n' '[0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.078,0.181,0.230,0.164,0.058,0.071,0.018]'
      ;;
    cup_upright)
      printf '%s\n' '[0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.092,0.189,0.214,0.198,0.165,0.089,0.029]'
      ;;
    pen_insert)
      printf '%s\n' '[0.053,0.175,0.199,0.197,0.267,0.097,0.033,0.070,0.171,0.227,0.170,0.092,0.086,0.037]'
      ;;
    plate_grape)
      printf '%s\n' '[0.045,0.162,0.200,0.141,0.169,0.106,0.028,0.097,0.157,0.223,0.159,0.071,0.115,0.030]'
      ;;
    *)
      return 2
      ;;
  esac
}

# A task's checkpoints all use the same 10Hz demonstrations and absolute-qpos
# action semantics. Safety envelopes are therefore task-specific. The method
# and checkpoint are still validated so unsupported combinations cannot
# silently inherit a profile.
arx4_validate_safety_profile_key() {
  local method="$1"
  local task="$2"
  local step="$3"
  [[ "${method}" =~ ^(jepa|mip|diffusion_policy)$ ]] || return 2
  [[ "${task}" =~ ^(cabinet|cup_stack|cup_upright|pen_insert|plate_grape)$ ]] || return 2
  [[ "${step}" =~ ^(050000|060000|080000|100000|140000|180000)$ ]] || return 2
}

# Current cache_10hz_base5050_v2 p95 per-step delta and delta-change, rounded
# upward to 0.001. Arm joints follow demonstrations; grippers retain the hard
# task envelope so grasp/release is not stretched across many commands.
arx4_cabinet_stable_delta() {
  printf '%s\n' '[0.044,0.107,0.108,0.083,0.031,0.020,0.020,0.027,0.084,0.098,0.070,0.013,0.018,0.030]'
}

arx4_cabinet_stable_delta_change() {
  printf '%s\n' '[0.010,0.023,0.023,0.030,0.021,0.018,0.020,0.008,0.019,0.021,0.025,0.012,0.016,0.030]'
}

# Deployment rate limit shared by JEPA, MIP, and DP pen_insert. Arm values are
# rounded-up training p95 per-step deltas at 10Hz; grippers retain the hard
# safety profile so a demonstrated grasp closure is not artificially delayed.
arx4_pen_insert_stable_delta() {
  printf '%s\n' '[0.015,0.115,0.110,0.103,0.066,0.024,0.033,0.036,0.116,0.109,0.101,0.029,0.028,0.037]'
}

# Rounded-up training p95 change in per-step delta at 10Hz. This bounds
# commanded acceleration to the demonstrated distribution for arm joints.
# Grippers retain the hard per-step limits so a demonstrated closure is not
# stretched over an impractically long interval.
arx4_pen_insert_stable_delta_change() {
  printf '%s\n' '[0.007,0.023,0.024,0.033,0.026,0.023,0.033,0.009,0.021,0.023,0.028,0.019,0.020,0.037]'
}

# cup_stack training p95 at 10Hz. The right arm is active; the left arm stays
# effectively stationary. Gripper limits retain the hard task envelope so a
# demonstrated close is not stretched across many inference chunks.
arx4_cup_stack_stable_delta() {
  printf '%s\n' '[0.001,0.001,0.001,0.001,0.001,0.001,0.005,0.044,0.130,0.153,0.091,0.015,0.027,0.018]'
}

# cup_stack p95 change in per-step delta. Starting from measured qvel, this
# ramps the active arm toward its demonstrated speed instead of reaching a
# rare per-step extreme on the second command.
arx4_cup_stack_stable_delta_change() {
  printf '%s\n' '[0.001,0.001,0.001,0.001,0.001,0.001,0.005,0.010,0.024,0.035,0.034,0.014,0.023,0.018]'
}

# cup_upright training p95 at 10Hz, computed from all 100 demonstrations.
# This is a right-arm-only task. The grippers retain the hard task envelope so
# the demonstrated grasp is not stretched across many inference chunks.
arx4_cup_upright_stable_delta() {
  printf '%s\n' '[0.001,0.001,0.001,0.001,0.001,0.001,0.005,0.055,0.138,0.139,0.113,0.101,0.040,0.029]'
}

# cup_upright p95 change in per-step delta at 10Hz. Starting from measured
# qvel prevents the first two commands from jumping directly to a rare speed.
arx4_cup_upright_stable_delta_change() {
  printf '%s\n' '[0.001,0.001,0.001,0.001,0.001,0.001,0.005,0.012,0.027,0.034,0.038,0.032,0.024,0.029]'
}

arx4_plate_grape_stable_delta() {
  printf '%s\n' '[0.012,0.105,0.115,0.071,0.074,0.037,0.028,0.050,0.109,0.115,0.082,0.009,0.042,0.030]'
}

arx4_plate_grape_stable_delta_change() {
  printf '%s\n' '[0.007,0.018,0.022,0.027,0.027,0.021,0.028,0.010,0.018,0.019,0.022,0.010,0.021,0.030]'
}

arx4_stable_delta_for_profile() {
  local method="$1"
  local task="$2"
  local step="$3"
  arx4_validate_safety_profile_key "${method}" "${task}" "${step}" || return 2
  case "${task}" in
    cabinet) arx4_cabinet_stable_delta ;;
    cup_stack) arx4_cup_stack_stable_delta ;;
    cup_upright) arx4_cup_upright_stable_delta ;;
    pen_insert) arx4_pen_insert_stable_delta ;;
    plate_grape) arx4_plate_grape_stable_delta ;;
  esac
}

arx4_stable_delta_change_for_profile() {
  local method="$1"
  local task="$2"
  local step="$3"
  arx4_validate_safety_profile_key "${method}" "${task}" "${step}" || return 2
  case "${task}" in
    cabinet) arx4_cabinet_stable_delta_change ;;
    cup_stack) arx4_cup_stack_stable_delta_change ;;
    cup_upright) arx4_cup_upright_stable_delta_change ;;
    pen_insert) arx4_pen_insert_stable_delta_change ;;
    plate_grape) arx4_plate_grape_stable_delta_change ;;
  esac
}

# Backward-compatible names for older operator commands that source this file.
arx4_dp_pen_insert_stable_delta() {
  arx4_pen_insert_stable_delta
}

arx4_dp_pen_insert_stable_delta_change() {
  arx4_pen_insert_stable_delta_change
}
