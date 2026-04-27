"""Region/slot classification for the M6a graft pipeline.

A graft proposal targets a **host region** ``R`` (a set of op IDs that the
GNN selected as the cut-out window). The relationship between ``R`` and
the host's existing slot regions determines how grafting must be applied:

* **Case I — slot-aligned**: ``R`` exactly matches the transitive op set
  of some slot ``S``. The graft replaces ``S``'s body wholesale; the slot
  identity, ``slot_meta`` entry, and ``SlotPopulation`` are preserved and
  the donor body becomes a new variant inside ``S``.
* **Case II — slot-disjoint**: ``R`` does not overlap any slot's
  transitive op set. The graft is a free-form structural splice that
  leaves all slot annotations untouched.
* **Case III — half-cut**: ``R`` overlaps but is not equal to one or
  more slots' transitive op sets. Each such "half-cut" slot must be
  *dissolved* (annotations + meta + population removed) before the splice
  proceeds, since the resulting IR no longer contains the slot.

This module exposes:

* :class:`BoundarySignature` — the typed entry/exit signature of a
  region, used by the GNN's donor-sampling masks (M6a §6.7).
* :class:`RegionClassification` — the result of :func:`classify_region`.
* :func:`classify_region` — the verbatim §6.4 algorithm.
* :func:`region_op_set` / :func:`slot_op_set` — small accessors used by
  the engine when applying Case I/III bookkeeping.

The classification is purely structural; it never mutates the IR.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from algorithm_ir.ir.model import FunctionIR, SlotMeta
    from algorithm_ir.region.selector import RewriteRegion


__all__ = [
    "BoundarySignature",
    "RegionClassification",
    "classify_region",
    "region_op_set",
    "slot_op_set",
]


# ---------------------------------------------------------------------------
# BoundarySignature — typed entry/exit ports
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BoundarySignature:
    """Type signature of a region's data-flow boundary.

    ``entry_types[i]`` is the static type of the i-th entry value (live-in
    SSA value) and ``exit_types[j]`` is the static type of the j-th exit
    value (live-out SSA value). Used by the GNN donor-sampler to build
    per-step type masks: only donor candidate values whose type is
    compatible (lattice ``is_subtype``) with the corresponding host port
    are eligible at each sampling step (§6.7).

    Two signatures are equal iff their type tuples match positionally.
    """
    entry_types: tuple[str, ...]
    exit_types: tuple[str, ...]

    def arity(self) -> tuple[int, int]:
        return (len(self.entry_types), len(self.exit_types))


def _value_type(ir: "FunctionIR", value_id: str) -> str:
    """Return the static type tag of ``value_id`` (or ``'unknown'``)."""
    val = ir.values.get(value_id)
    if val is None:
        return "unknown"
    # Multiple representations have been used historically; check all.
    type_hint = getattr(val, "type_hint", None)
    if type_hint:
        return str(type_hint)
    attrs = getattr(val, "attrs", None) or {}
    for key in ("type", "static_type", "dtype"):
        if key in attrs and attrs[key] is not None:
            return str(attrs[key])
    return "unknown"


def signature_for_region(
    ir: "FunctionIR", region: "RewriteRegion",
) -> BoundarySignature:
    """Build a :class:`BoundarySignature` from the region's port lists."""
    entry_types = tuple(_value_type(ir, v) for v in region.entry_values)
    exit_types = tuple(_value_type(ir, v) for v in region.exit_values)
    return BoundarySignature(entry_types=entry_types, exit_types=exit_types)


# ---------------------------------------------------------------------------
# Region / slot op-set helpers
# ---------------------------------------------------------------------------

def region_op_set(region: "RewriteRegion") -> frozenset[str]:
    """Frozen set of op-IDs covered by the region."""
    return frozenset(region.op_ids)


def slot_op_set(ir: "FunctionIR", pop_key: str) -> frozenset[str]:
    """Transitive op set for a slot (innermost + descendants)."""
    return frozenset(ir.slot_full_op_ids(pop_key))


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegionClassification:
    """Outcome of :func:`classify_region`.

    ``case`` is the dispatch label.
    For Case I, ``attribution_slot_pop_key`` names the slot whose body
    matches the region exactly.
    For Case III, ``half_cut_slots`` lists every slot that must be
    dissolved before splicing.
    """
    case: Literal["I", "II", "III"]
    attribution_slot_pop_key: str | None = None
    half_cut_slots: tuple[str, ...] = ()


def classify_region(
    region: "RewriteRegion", slot_meta: dict[str, "SlotMeta"], ir: "FunctionIR",
) -> RegionClassification:
    """Classify ``region`` against ``ir.slot_meta`` per §6.4.

    Algorithm (verbatim from major_refactor.md §6.4):

    1. Let ``R = set(region.op_ids)``.
    2. For every slot ``S``, compute its transitive op set
       ``S_full = ir.slot_full_op_ids(S.pop_key)``.
    3. If there exists a slot ``S`` with ``R == S_full`` → Case I.
       The attribution key is ``S.pop_key``. (At most one such ``S``
       can exist because slot regions are pairwise disjoint at the
       innermost-membership level and ``slot_full_op_ids`` is monotone.)
    4. Else collect ``halves = [S for S in slots if R ∩ S_full ≠ ∅
       and R ⊉ S_full and R ⊈ S_full]``. The conditions
       ``R ⊉ S_full`` and ``R ⊈ S_full`` ensure neither side fully
       contains the other — i.e. the region cuts ``S`` in half.
       If ``halves`` is non-empty → Case III.
    5. Otherwise (``R`` is either disjoint from every slot or strictly
       contained in / strictly contains some slot but never with mutual
       overlap) → Case II.

    Notes
    -----
    The Case II catch-all also covers the case where ``R`` strictly
    contains a slot ``S`` (``R ⊃ S_full``) — that scenario does not
    require dissolving ``S`` because the slot's body remains intact
    inside the donor's replacement region; ``slot_meta`` is refreshed
    by the engine after splicing if any slot's op set changed.
    """
    R = region_op_set(region)
    if not R:
        # Empty region — degenerate; treat as Case II so the caller can
        # decide whether to reject.
        return RegionClassification(case="II")

    # Pre-compute every slot's transitive op set once.
    slot_full: dict[str, frozenset[str]] = {
        key: slot_op_set(ir, key) for key in slot_meta.keys()
    }

    # Case I: exact match against any slot's transitive op set.
    for pop_key, S_full in slot_full.items():
        if S_full and R == S_full:
            return RegionClassification(case="I", attribution_slot_pop_key=pop_key)

    # Case III: any slot is half-cut.
    halves: list[str] = []
    for pop_key, S_full in slot_full.items():
        if not S_full:
            continue
        intersect = R & S_full
        if not intersect:
            continue  # disjoint
        if R >= S_full or R <= S_full:
            continue  # one fully contains the other → not half-cut
        halves.append(pop_key)

    if halves:
        return RegionClassification(
            case="III", half_cut_slots=tuple(sorted(halves)),
        )

    return RegionClassification(case="II")
