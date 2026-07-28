#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_ROOT="$(cd -- "$REPO/../.." && pwd)"
VENV_ROOT="${JEPA_POLICY_VENV_ROOT:-$POLICY_ROOT/venvs}"
PYTHON="$VENV_ROOT/jepa_ppu/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "robomimic Python not found: $PYTHON" >&2
    exit 2
fi

cd "$REPO"
exec "$PYTHON" "$REPO/examples/train_robomimic.py" "$@"
