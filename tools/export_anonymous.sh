#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
output_path=${1:-"$repo_root/JEPA-Policy-anonymous.tar.gz"}
anon_url=${ANON_REPOSITORY_URL:-https://anonymous.4open.science/r/JEPA-Policy}
tmp_root=$(mktemp -d)
trap 'rm -rf "$tmp_root"' EXIT

git -C "$repo_root" archive --format=tar --prefix=JEPA-Policy/ HEAD \
  | tar -xf - -C "$tmp_root"

readme="$tmp_root/JEPA-Policy/README.md"
sed -i.bak \
  "s#https://github.com/jiejie567/JEPA-Policy.git#$anon_url#g" \
  "$readme"
rm "$readme.bak"

deny_pattern='jiejie567|Sun-Season|Apjocalypse|wuwoasd811|/mnt/data_nas|/root/'
if grep -RInE --exclude=export_anonymous.sh "$deny_pattern" "$tmp_root/JEPA-Policy"; then
  echo "Anonymous export aborted: identity or internal path detected." >&2
  exit 1
fi

tar -czf "$output_path" -C "$tmp_root" JEPA-Policy
echo "Created $output_path"
