#!/usr/bin/env bash

# Build a deterministic Python/native-library environment for the isolated
# Python 3.10 runtime. In particular, do not inherit Python packages from an
# activated Conda environment (the operator terminal commonly uses Python 3.12).
arx4_prepare_runtime_env() {
  local source_root="$1"
  local arx5_sdk_root="$2"
  local ros_distro="${ROS_DISTRO:-humble}"
  local entry

  local runtime_pythonpath="${arx5_sdk_root}/python:${source_root}"
  for entry in \
    "/opt/ros/${ros_distro}/lib/python3.10/site-packages" \
    "/opt/ros/${ros_distro}/local/lib/python3.10/dist-packages"; do
    if [[ -d "${entry}" ]]; then
      runtime_pythonpath+=":${entry}"
    fi
  done
  export PYTHONPATH="${runtime_pythonpath}"
  unset PYTHONHOME
  export PYTHONNOUSERSITE=1
  export PYTHONSAFEPATH=1
  export ARX5_X5_URDF="${arx5_sdk_root}/models/X5.urdf"

  # Preserve system/ROS/CUDA library paths, but exclude native libraries from
  # Conda environments because they can also be ABI-incompatible with the venv.
  local runtime_ld_library_path="${arx5_sdk_root}/lib/x86_64"
  for entry in \
    "/opt/ros/${ros_distro}/opt/rviz_ogre_vendor/lib" \
    "/opt/ros/${ros_distro}/lib/x86_64-linux-gnu" \
    "/opt/ros/${ros_distro}/lib"; do
    if [[ -d "${entry}" ]]; then
      runtime_ld_library_path+=":${entry}"
    fi
  done
  local -a inherited_ld_paths=()
  IFS=':' read -r -a inherited_ld_paths <<<"${LD_LIBRARY_PATH:-}"
  for entry in "${inherited_ld_paths[@]}"; do
    [[ -n "${entry}" ]] || continue
    if [[ -n "${CONDA_PREFIX:-}" && "${entry}" == "${CONDA_PREFIX}"* ]]; then
      continue
    fi
    case "${entry}" in
      */miniconda*/envs/* | */anaconda*/envs/*) continue ;;
    esac
    runtime_ld_library_path+=":${entry}"
  done
  export LD_LIBRARY_PATH="${runtime_ld_library_path}"
}

# Hold one advisory lock for the entire lifetime of every process that can open
# the X5 CAN interfaces. The descriptor is intentionally inherited by exec().
arx4_acquire_hardware_lock() {
  local lock_file="${ARX4_HARDWARE_LOCK_FILE:-/tmp/prometheus_arx5_can1_can3.lock}"
  command -v flock >/dev/null 2>&1 || return 127
  exec {ARX4_HARDWARE_LOCK_FD}>>"${lock_file}" || return 126
  flock -n "${ARX4_HARDWARE_LOCK_FD}" || return 1
}

# Select the known-good FFmpeg build and verify both recording paths before
# any camera or robot process starts.
arx4_prepare_recording_env() {
  local ffmpeg_bin="${ARX4_FFMPEG_BIN:-}"
  local encoders
  if [[ -z "${ffmpeg_bin}" ]]; then
    ffmpeg_bin="$(command -v ffmpeg || true)"
  fi
  if [[ -z "${ffmpeg_bin}" || ! -x "${ffmpeg_bin}" ]]; then
    printf '[arx4_jepa_eval] recording requires an executable ffmpeg; set ARX4_FFMPEG_BIN\n' >&2
    return 1
  fi
  ffmpeg_bin="$(readlink -f "${ffmpeg_bin}")"
  encoders="$("${ffmpeg_bin}" -hide_banner -encoders 2>/dev/null)" || {
    printf '[arx4_jepa_eval] ffmpeg encoder preflight failed: %s\n' "${ffmpeg_bin}" >&2
    return 1
  }
  grep -q 'hevc_nvenc' <<<"${encoders}" || {
    printf '[arx4_jepa_eval] ffmpeg lacks hevc_nvenc: %s\n' "${ffmpeg_bin}" >&2
    return 1
  }
  grep -q 'libx264' <<<"${encoders}" || {
    printf '[arx4_jepa_eval] ffmpeg lacks libx264: %s\n' "${ffmpeg_bin}" >&2
    return 1
  }
  "${ffmpeg_bin}" -hide_banner -loglevel error \
    -f lavfi -i color=size=640x480:rate=30 \
    -frames:v 1 -c:v hevc_nvenc -f null - || {
    printf '[arx4_jepa_eval] 640x480 hevc_nvenc preflight failed\n' >&2
    return 1
  }
  "${ffmpeg_bin}" -hide_banner -loglevel error \
    -f lavfi -i color=size=128x128:rate=30 \
    -frames:v 1 -c:v libx264 -pix_fmt yuv444p -f null - || {
    printf '[arx4_jepa_eval] 128x128 libx264 preflight failed\n' >&2
    return 1
  }
  export PATH="$(dirname "${ffmpeg_bin}"):${PATH}"
  export ARX4_FFMPEG_REPORT="${ffmpeg_bin} original=640x480/hevc_nvenc policy=128x128/libx264"
}
