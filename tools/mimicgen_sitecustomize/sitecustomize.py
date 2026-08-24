"""Append the verified JEPA/PPU packages as a last-resort import fallback."""

from __future__ import annotations

import sys


PPU_SITE = "/mnt/data_nas/ykj_jepa_policy/venvs/jepa_ppu/lib/python3.12/site-packages"
if PPU_SITE not in sys.path:
    # Append rather than prepend: packages installed in the isolated MimicGen
    # venv and its pinned source trees must always win name conflicts.
    sys.path.append(PPU_SITE)
