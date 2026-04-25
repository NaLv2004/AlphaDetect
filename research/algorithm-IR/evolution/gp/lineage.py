"""MutationRecord — per-operator lineage for traceable typed GP."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MutationRecord:
    operator: str               # e.g. "mut_const"
    seed: int
    parent_hash: str
    parent2_hash: str | None
    diff_summary: str           # human-readable diff
    accepted: bool
    rejection_reason: str | None = None
