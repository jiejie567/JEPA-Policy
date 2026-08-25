#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_ROOT="${JEPA_POLICY_VENV_ROOT:-$REPO/.venvs}"
PYTHON="${LIBERO_PYTHON:-$VENV_ROOT/libero/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
    echo "LIBERO Python not found: $PYTHON" >&2
    echo "Set LIBERO_PYTHON or JEPA_POLICY_VENV_ROOT." >&2
    exit 2
fi

cd "$REPO"
exec "$PYTHON" "$REPO/examples/train_robomimic.py" "$@"
