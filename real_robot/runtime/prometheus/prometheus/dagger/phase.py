from __future__ import annotations

from enum import Enum
from typing import Literal

DAggerSemantic = Literal["pause", "teaching", "cycle"]
SemanticDisposition = Literal["accept", "defer", "ignore"]

DAGGER_SEMANTICS: frozenset[str] = frozenset({"pause", "teaching", "cycle"})


class DAggerPhase(str, Enum):
    IDLE = "idle"
    CYCLE_WAIT = "cycle_wait"
    AUTONOMOUS = "autonomous"
    PAUSED = "paused"
    CORRECTING = "correcting"


def semantic_disposition(phase: DAggerPhase, semantic: str) -> SemanticDisposition:
    if semantic not in DAGGER_SEMANTICS:
        raise ValueError(f"unsupported dagger semantic: {semantic!r}")

    if phase == DAggerPhase.CYCLE_WAIT:
        return "ignore"

    if semantic == "pause":
        if phase == DAggerPhase.AUTONOMOUS:
            return "defer"
        return "ignore"

    if semantic == "teaching":
        if phase in {DAggerPhase.AUTONOMOUS, DAggerPhase.PAUSED}:
            return "accept"
        return "ignore"

    if semantic == "cycle":
        if phase in {
            DAggerPhase.IDLE,
            DAggerPhase.AUTONOMOUS,
            DAggerPhase.PAUSED,
            DAggerPhase.CORRECTING,
        }:
            return "accept"
        return "ignore"

    raise ValueError(f"unsupported dagger semantic: {semantic!r}")
