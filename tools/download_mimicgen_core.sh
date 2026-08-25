#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${MIMICGEN_DATA_ROOT:-$REPO_ROOT/datasets/mimicgen/core}"
BASE_URL="https://huggingface.co/datasets/amandlek/mimicgen_datasets/resolve/main/core"

mkdir -p "$DATA_ROOT"

download_one() {
  local name="$1"
  local expected_size="$2"
  local expected_sha256="$3"
  local task_dir="$DATA_ROOT/$name"
  local output="$task_dir/$name.hdf5"
  local partial="$output.inprogress"

  mkdir -p "$task_dir"

  if [[ -f "$output" ]]; then
    local current_size
    current_size="$(stat -c '%s' "$output")"
    [[ "$current_size" == "$expected_size" ]] || {
      echo "Existing file has the wrong size: $output ($current_size != $expected_size)" >&2
      exit 1
    }
    echo "$expected_sha256  $output" | sha256sum -c -
    echo "Already complete: $output"
    return
  fi

  local partial_size=0
  if [[ -f "$partial" ]]; then
    partial_size="$(stat -c '%s' "$partial")"
    (( partial_size <= expected_size )) || {
      echo "Partial file is larger than expected: $partial ($partial_size > $expected_size)" >&2
      exit 1
    }
  fi

  if (( partial_size < expected_size )); then
    echo "Downloading $name to $partial (resume offset: $partial_size)"
    curl \
      --fail \
      --location \
      --continue-at - \
      --retry 20 \
      --retry-all-errors \
      --retry-delay 3 \
      --connect-timeout 30 \
      --output "$partial" \
      "$BASE_URL/$name.hdf5"
  else
    echo "Partial download already has the expected size: $partial"
  fi

  local actual_size
  actual_size="$(stat -c '%s' "$partial")"
  [[ "$actual_size" == "$expected_size" ]] || {
    echo "Downloaded file has the wrong size: $partial ($actual_size != $expected_size)" >&2
    exit 1
  }
  echo "$expected_sha256  $partial" | sha256sum -c -
  mv -- "$partial" "$output"
  echo "Completed: $output"
}

download_one \
  three_piece_assembly_d1 \
  3237234348 \
  7f5cad32fdf492b210c181b84b4856eaff1a90573ffbc5eeae871cbe8e01e586
download_one \
  coffee_preparation_d1 \
  6923699908 \
  0e9e1eac8d969c05a5fff90358f702b6530ea80bb7a9977497e3a3740b88cf55
download_one \
  kitchen_d1 \
  7069704890 \
  e43e339f85283aca458a2455acfe013a0642d72b117412d3218844ffd7d82dfb
