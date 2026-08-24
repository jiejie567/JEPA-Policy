#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_ROOT="${JEPA_POLICY_VENV_ROOT:-$REPO/.venvs}"
PYTHON="${MIMICGEN_PYTHON:-$VENV_ROOT/mimicgen/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
    echo "MimicGen Python not found: $PYTHON" >&2
    echo "Set MIMICGEN_PYTHON or JEPA_POLICY_VENV_ROOT." >&2
    exit 2
fi

cd "$REPO"
exec "$PYTHON" "$REPO/examples/train_robomimic.py" "$@"
