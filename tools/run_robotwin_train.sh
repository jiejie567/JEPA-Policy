#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
exec "$REPO/tools/run_robotwin_python.sh" \
  "$REPO/examples/train_robomimic.py" "$@"
