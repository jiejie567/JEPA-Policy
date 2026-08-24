#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$REPO/tools/run_mimicgen_python.sh" "$REPO/examples/train_robomimic.py" "$@"
