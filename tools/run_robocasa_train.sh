#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# This is the only supported RoboCasa training entry point. The underlying
# wrapper rejects checkout drift and replaces inherited Python/Conda/GL paths.
exec "$REPO/tools/run_robocasa_python.sh" \
  "$REPO/examples/train_robomimic.py" "$@"
