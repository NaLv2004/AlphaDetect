"""Opcode whitelist (operator filter) for Push-GP search.

Lets a user restrict the operator pool used for random program
generation, mutation, and crossover.  Two separate whitelists are
supported (V2C and C2V) because the legal opcode sets on the two sides
already differ — e.g. ``Env.GetChannelLLR`` is V2C-only.  The filter is
intersected with the base side-specific allowed set, so it can only
*shrink* the search space, never grow it past what the validator can
handle.

Config file format (JSON)
-------------------------
Either a flat list (applied to BOTH sides):

    {"allowed": ["Float.Add", "Float.Sub", "Exec.DoTimes", ...]}

Or per-side lists (recommended for OMS, since v2c and c2v differ):

    {
      "v2c": ["FVec.Len", "FVec.At", "Float.Add", "Exec.DoTimes"],
      "c2v": ["Bool.False", "Bool.Xor", "FVec.Len", ...]
    }

A `name_pattern` field (regex) is also supported, applied AFTER the
explicit list; matched names are added to the allowed set:

    {"v2c": [], "c2v": [], "name_pattern": "^Float\\.|^Exec\\."}

Unknown opcode names are ignored with a warning (so configs survive
opcode renames without crashing).

Public API
----------
* `OpFilter`        - immutable filter object with `.v2c` / `.c2v` sets
* `load_op_filter`  - load OpFilter from JSON path
* `filter_instr_set`- shrink a Sequence[str] to the filter's allowed set
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import FrozenSet, List, Optional, Sequence, Set

from .instructions import HANDLERS


# ---- side-base sets (mirrors pushgp.random_program) ------------------
_ALL_NAMES: List[str] = sorted(HANDLERS.keys())
_V2C_ONLY: Set[str] = {"Env.GetChannelLLR"}
_C2V_ONLY: Set[str] = set()


def _base_v2c() -> Set[str]:
    return {n for n in _ALL_NAMES if n not in _C2V_ONLY}


def _base_c2v() -> Set[str]:
    return {n for n in _ALL_NAMES if n not in _V2C_ONLY}


@dataclass(frozen=True)
class OpFilter:
    """Immutable opcode whitelist.

    ``v2c`` / ``c2v`` are the **final** allowed sets after intersecting
    user choices with the base side-allowed sets.  An empty frozenset
    means "no filter (use the full base set)" — represented explicitly
    by ``is_active=False``.
    """
    v2c: FrozenSet[str] = field(default_factory=frozenset)
    c2v: FrozenSet[str] = field(default_factory=frozenset)
    is_active: bool = False
    source_path: Optional[str] = None

    def names(self, side: str) -> FrozenSet[str]:
        if side == "v2c":
            return self.v2c
        if side == "c2v":
            return self.c2v
        raise ValueError(f"unknown side: {side!r}")

    def applies(self) -> bool:
        return self.is_active

    def describe(self) -> str:
        if not self.is_active:
            return "[op-filter] inactive (full opcode set)"
        return (
            f"[op-filter] active (src={self.source_path or '?'})  "
            f"v2c={len(self.v2c)}/{len(_base_v2c())}  "
            f"c2v={len(self.c2v)}/{len(_base_c2v())}"
        )

    # serialisation for meta.json / UI display
    def to_dict(self) -> dict:
        return {
            "active": bool(self.is_active),
            "source_path": self.source_path,
            "v2c": sorted(self.v2c),
            "c2v": sorted(self.c2v),
        }


def _validate_names(names: Sequence[str], side_base: Set[str], side: str,
                    src: str) -> Set[str]:
    """Return only names that are present in side_base; warn on others."""
    kept: Set[str] = set()
    unknown: List[str] = []
    excluded: List[str] = []
    for n in names:
        if n in side_base:
            kept.add(n)
        elif n in _ALL_NAMES:
            excluded.append(n)
        else:
            unknown.append(n)
    if unknown:
        print(f"[op-filter] WARNING [{src} {side}] unknown opcode(s) "
              f"ignored: {unknown}", file=sys.stderr, flush=True)
    if excluded:
        print(f"[op-filter] WARNING [{src} {side}] opcode(s) not legal on "
              f"this side, ignored: {excluded}", file=sys.stderr, flush=True)
    return kept


def load_op_filter(path: Optional[str]) -> OpFilter:
    """Load an OpFilter from a JSON file.

    Path may be:
      * None or "" -> returns an INACTIVE filter (full opcode set).
      * "off"      -> same as None.
      * a JSON file path with structure described in the module docstring.
    """
    if path is None or str(path).strip() == "" or str(path).strip().lower() == "off":
        return OpFilter(is_active=False)
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"op filter config not found: {path}")
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"failed to parse op filter config {path}: {e}") from e
    if not isinstance(cfg, dict):
        raise ValueError(f"op filter config root must be an object: {path}")

    v_names: Set[str] = set()
    c_names: Set[str] = set()
    if "allowed" in cfg:
        flat = list(cfg.get("allowed") or [])
        v_names.update(flat)
        c_names.update(flat)
    if "v2c" in cfg:
        v_names.update(list(cfg.get("v2c") or []))
    if "c2v" in cfg:
        c_names.update(list(cfg.get("c2v") or []))

    pat = cfg.get("name_pattern")
    if pat:
        rx = re.compile(pat)
        for n in _ALL_NAMES:
            if rx.search(n):
                v_names.add(n)
                c_names.add(n)

    src = str(p)
    v_final = _validate_names(sorted(v_names), _base_v2c(), "v2c", src)
    c_final = _validate_names(sorted(c_names), _base_c2v(), "c2v", src)
    if not v_final:
        raise ValueError(
            f"op filter {path}: V2C whitelist is empty after validation. "
            f"At least one legal V2C opcode is required."
        )
    if not c_final:
        raise ValueError(
            f"op filter {path}: C2V whitelist is empty after validation."
        )
    return OpFilter(
        v2c=frozenset(v_final),
        c2v=frozenset(c_final),
        is_active=True,
        source_path=src,
    )


def filter_instr_set(side: str, instr_set: Sequence[str],
                     of: Optional[OpFilter]) -> List[str]:
    """Intersect `instr_set` with `of.names(side)` if the filter is active.

    Returns the input unchanged (as a list) when no filter is given.
    Preserves the order of `instr_set`.
    """
    if of is None or not of.applies():
        return list(instr_set)
    allowed = of.names(side)
    return [n for n in instr_set if n in allowed]


__all__ = ["OpFilter", "load_op_filter", "filter_instr_set"]
