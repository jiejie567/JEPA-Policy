#!/usr/bin/env bash
set -euo pipefail

REPO="/mnt/data_nas/ykj_jepa_policy/code/JEPA-Policy"
ASSET_ROOT="$REPO/third_party/robocasa_v1_0_1/robocasa/models/assets"
DOWNLOAD_ROOT="$REPO/datasets/robocasa/asset_downloads"

declare -A URLS=(
  [textures]="https://utexas.box.com/shared/static/4i85ileasdvstmlln5sbvzptz7keuoy1.zip"
  [generative_textures]="https://utexas.box.com/shared/static/ebaad09k82tmfmlq6ohdkmrh8izl9vn5.zip"
  [fixtures]="https://utexas.box.com/shared/static/idbncsadpnaz1jfl4i6m8qejawk7p9pi.zip"
  [objaverse]="https://utexas.box.com/shared/static/03eionyo8fk3a9dsksq9jb8du5lqfw8h.zip"
  [aigen_objs]="https://utexas.box.com/shared/static/nwi1vrn5pgbo95kushkasa3nx1i012ff.zip"
  [lightwheel]="https://utexas.box.com/shared/static/vckqvvkh1z8t69k8qcpcmee6k66stii4.zip"
)

mkdir -p "$DOWNLOAD_ROOT"

download_asset() {
  local name="$1"
  local archive="$DOWNLOAD_ROOT/$name.zip"
  local completed="$archive.complete"
  [[ -f "$completed" ]] && return
  local final_url
  final_url="$(
    curl --fail --silent --show-error --location --max-time 60 \
      --retry 20 --retry-delay 10 --retry-all-errors \
      --range 0-0 --output /dev/null --write-out '%{url_effective}' \
      "${URLS[$name]}"
  )"
  aria2c --continue=true --max-connection-per-server=8 --split=8 \
    --min-split-size=8M --max-tries=20 --retry-wait=10 \
    --connect-timeout=30 --timeout=120 --auto-file-renaming=false \
    --file-allocation=none --dir="$DOWNLOAD_ROOT" \
    --out="$(basename "$archive")" "$final_url"
  unzip -tq "$archive" >/dev/null
  touch "$completed"
  echo "ASSET_DOWNLOAD_OK name=$name bytes=$(stat -c %s "$archive")"
}

pids=()
for name in "${!URLS[@]}"; do
  download_asset "$name" >"$DOWNLOAD_ROOT/$name.log" 2>&1 &
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

for name in textures generative_textures fixtures; do
  unzip -oq "$DOWNLOAD_ROOT/$name.zip" -d "$ASSET_ROOT"
done
mkdir -p "$ASSET_ROOT/objects"
for name in objaverse aigen_objs lightwheel; do
  unzip -oq "$DOWNLOAD_ROOT/$name.zip" -d "$ASSET_ROOT/objects"
done

for required in \
  "$ASSET_ROOT/textures" \
  "$ASSET_ROOT/generative_textures" \
  "$ASSET_ROOT/fixtures" \
  "$ASSET_ROOT/objects/objaverse" \
  "$ASSET_ROOT/objects/aigen_objs" \
  "$ASSET_ROOT/objects/lightwheel"; do
  [[ -d "$required" ]] || {
    echo "Missing extracted asset directory: $required" >&2
    exit 1
  }
done
echo "ROBOCASA_ASSETS_OK"
