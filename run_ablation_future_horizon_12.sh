#!/usr/bin/env bash
set -euo pipefail

# Run one benchmark's 3-seed x 2-task x 2-horizon matrix on a 16-GPU node.
# Each seed owns four GPUs: 41 -> 0-3, 42 -> 4-7, 43 -> 8-11.

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_TO_RUN="${1:-}"
SEEDS=(41 42 43)
PIDS=()

case "$BENCHMARK_TO_RUN" in
  libero)
    LAUNCHER="run_ablation2_libero_8.sh"
    SKIP_TRANSPORT_VALUE=0
    ;;
  robomimic)
    LAUNCHER="run_ablation2_robomimic_12.sh"
    SKIP_TRANSPORT_VALUE=1
    ;;
  *)
    echo "Usage: $0 {libero|robomimic}" >&2
    exit 2
    ;;
esac

terminate_children() {
  local pid
  trap - INT TERM
  for pid in "${PIDS[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
}
trap terminate_children INT TERM

for seed_index in "${!SEEDS[@]}"; do
  seed="${SEEDS[$seed_index]}"
  gpu_offset="$((seed_index * 4))"
  (
    cd "$REPO"
    SEED="$seed" \
    FUTURE_HORIZON_ABLATION=1 \
    GPU_OFFSET="$gpu_offset" \
    SKIP_TRANSPORT="$SKIP_TRANSPORT_VALUE" \
    RUN_SUFFIX=future_horizon \
    bash "$LAUNCHER"
  ) &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
