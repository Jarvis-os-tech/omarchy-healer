"""Backward-compatibility shim. Use omarchy_healer.triage_judge instead."""
from __future__ import annotations

from .triage_judge import LayaTriageJudge, TriageJudge

__all__ = ["TriageJudge", "LayaTriageJudge"]
