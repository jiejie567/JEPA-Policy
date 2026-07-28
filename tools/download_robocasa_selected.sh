#!/usr/bin/env bash
set -euo pipefail

REPO="/mnt/data_nas/ykj_jepa_policy/code/JEPA-Policy"
DATA_ROOT="$REPO/datasets/robocasa/v1.0/target/composite"
DOWNLOAD_ROOT="$REPO/datasets/robocasa/downloads"

declare -A DATES=(
  [SteamInMicrowave]=20250814
  [StoreLeftoversInBowl]=20250813
  [LoadDishwasher]=20250811
)
declare -A URLS=(
  [SteamInMicrowave]="https://utexas.box.com/shared/static/orr1n70ald5dsep2kpdfnxo7jogcjqls.tar"
  [StoreLeftoversInBowl]="https://utexas.box.com/shared/static/n69gzwudustsz0hc3txhqzhclnbpab6r.tar"
  [LoadDishwasher]="https://utexas.box.com/shared/static/k1qxg8rgjs0le1cnv98xylysh9t0dd8z.tar"
)
declare -A SIZES=(
  [SteamInMicrowave]=2056683520
  [StoreLeftoversInBowl]=2009815040
)

mkdir -p "$DOWNLOAD_ROOT"

download_task() {
  local task="$1"
  local archive="$DOWNLOAD_ROOT/${task}_target_human_lerobot.tar"
  local completed="$archive.complete"
  if [[ -f "$completed" ]]; then
    echo "DOWNLOAD_ALREADY_OK task=$task archive=$archive"
    return
  fi
  # Resolve Box redirects with curl, then use the short-lived boxcloud URL for
  # a resumable multi-connection transfer. aria2 cannot reliably resolve the
  # original Box redirect chain itself.
  local final_url
  if ! final_url="$(
    curl --fail --silent --show-error --location --max-time 60 \
      --retry 5 --retry-delay 5 --retry-all-errors \
      --range 0-0 --output /dev/null --write-out '%{url_effective}' \
      "${URLS[$task]}"
  )"; then
    # Some Box objects reject Range requests with HTTP 500. Follow the same
    # redirect without Range and stop immediately after the boxcloud URL has
    # been resolved; aria2 performs the actual resumable transfer below.
    final_url="$(
      curl --silent --show-error --location --max-time 5 \
        --output /dev/null --write-out '%{url_effective}' \
        "${URLS[$task]}" || true
    )"
  fi
  [[ "$final_url" == https://public.boxcloud.com/* ]] || {
    echo "Could not resolve Box download URL for $task: $final_url" >&2
    return 1
  }
  aria2c --continue=true --max-connection-per-server=8 --split=8 \
    --min-split-size=8M --max-tries=20 --retry-wait=10 \
    --connect-timeout=30 --timeout=120 --auto-file-renaming=false \
    --file-allocation=none --dir="$DOWNLOAD_ROOT" \
    --out="$(basename "$archive")" "$final_url"
  if [[ -n "${SIZES[$task]:-}" ]]; then
    [[ "$(stat -c %s "$archive")" == "${SIZES[$task]}" ]] || {
      echo "Wrong archive size for $task" >&2
      return 1
    }
  fi
  tar -tf "$archive" >/dev/null
  touch "$completed"
  echo "DOWNLOAD_OK task=$task bytes=$(stat -c %s "$archive")"
}

pids=()
for task in SteamInMicrowave StoreLeftoversInBowl LoadDishwasher; do
  download_task "$task" >"$DOWNLOAD_ROOT/$task.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
(( failed == 0 )) || {
  tail -n 30 "$DOWNLOAD_ROOT"/*.log >&2
  exit 1
}

for task in SteamInMicrowave StoreLeftoversInBowl LoadDishwasher; do
  destination="$DATA_ROOT/$task/${DATES[$task]}"
  archive="$DOWNLOAD_ROOT/${task}_target_human_lerobot.tar"
  extract_complete="$destination/.extract.complete"
  if [[ ! -f "$extract_complete" ]]; then
    mkdir -p "$destination"
    # The official archives record uid/gid 1000. NAS does not support changing
    # ownership from this container, and ownership is irrelevant to training.
    tar --no-same-owner --overwrite -xf "$archive" -C "$destination"
  fi
  [[ -f "$destination/lerobot/meta/info.json" ]] || {
    echo "Missing LeRobot metadata after extraction: $destination" >&2
    exit 1
  }
  touch "$extract_complete"
  echo "DATASET_OK task=$task path=$destination/lerobot"
done
