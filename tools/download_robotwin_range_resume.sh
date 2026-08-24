#!/usr/bin/env bash
set -euo pipefail

URL="${1:?URL is required}"
PREFIX="${2:?existing contiguous prefix is required}"
OUTPUT="${3:?output path is required}"
TOTAL_SIZE="${4:?total size is required}"
EXPECTED_SHA256="${5:?expected SHA-256 is required}"
SEGMENT_COUNT="${6:-8}"

[[ -f "$PREFIX" ]] || {
  echo "Missing prefix: $PREFIX" >&2
  exit 2
}
[[ ! -e "$OUTPUT" ]] || {
  echo "Output already exists: $OUTPUT" >&2
  exit 2
}
[[ "$TOTAL_SIZE" =~ ^[1-9][0-9]*$ &&
  "$SEGMENT_COUNT" =~ ^[1-9][0-9]*$ ]] || {
  echo "TOTAL_SIZE and SEGMENT_COUNT must be positive integers" >&2
  exit 2
}
[[ "$EXPECTED_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
  echo "Invalid SHA-256" >&2
  exit 2
}

prefix_size="$(stat -c %s "$PREFIX")"
(( prefix_size > 0 && prefix_size < TOTAL_SIZE )) || {
  echo "Invalid prefix size: $prefix_size of $TOTAL_SIZE" >&2
  exit 2
}
remaining=$((TOTAL_SIZE - prefix_size))
chunk_size=$(((remaining + SEGMENT_COUNT - 1) / SEGMENT_COUNT))
segment_root="$(
  mktemp -d "/mnt/workspace/robotwin-ranges.$(basename "$OUTPUT").XXXXXX"
)"

for index in $(seq 0 $((SEGMENT_COUNT - 1))); do
  segment_start=$((prefix_size + index * chunk_size))
  segment_end=$((segment_start + chunk_size - 1))
  (( segment_end < TOTAL_SIZE )) || segment_end=$((TOTAL_SIZE - 1))
  (( segment_start <= segment_end )) || continue
  curl \
    --fail \
    --location \
    --retry 100 \
    --retry-all-errors \
    --retry-delay 3 \
    --range "${segment_start}-${segment_end}" \
    --output "$segment_root/segment.$index" \
    "$URL" \
    >"$segment_root/segment.$index.log" 2>&1 &
done
wait

for index in $(seq 0 $((SEGMENT_COUNT - 1))); do
  segment_start=$((prefix_size + index * chunk_size))
  segment_end=$((segment_start + chunk_size - 1))
  (( segment_end < TOTAL_SIZE )) || segment_end=$((TOTAL_SIZE - 1))
  (( segment_start <= segment_end )) || continue
  expected_size=$((segment_end - segment_start + 1))
  actual_size="$(stat -c %s "$segment_root/segment.$index")"
  [[ "$actual_size" == "$expected_size" ]] || {
    echo "Segment $index has $actual_size bytes, expected $expected_size" >&2
    exit 1
  }
done

assembling="${OUTPUT}.assembling"
[[ ! -e "$assembling" ]] || {
  echo "Assembly path already exists: $assembling" >&2
  exit 2
}
cp "$PREFIX" "$assembling"
for index in $(seq 0 $((SEGMENT_COUNT - 1))); do
  [[ -f "$segment_root/segment.$index" ]] || continue
  dd \
    if="$segment_root/segment.$index" \
    of="$assembling" \
    bs=4M \
    oflag=append \
    conv=notrunc \
    status=none
done

actual_size="$(stat -c %s "$assembling")"
[[ "$actual_size" == "$TOTAL_SIZE" ]] || {
  echo "Assembled file has $actual_size bytes, expected $TOTAL_SIZE" >&2
  exit 1
}
actual_sha256="$(sha256sum "$assembling")"
actual_sha256="${actual_sha256%% *}"
[[ "$actual_sha256" == "$EXPECTED_SHA256" ]] || {
  echo "SHA-256 mismatch: $actual_sha256" >&2
  exit 1
}
mv "$assembling" "$OUTPUT"
echo \
  "RANGE_DOWNLOAD_OK output=$OUTPUT prefix_bytes=$prefix_size " \
  "segments=$SEGMENT_COUNT sha256=$actual_sha256 segment_root=$segment_root"
