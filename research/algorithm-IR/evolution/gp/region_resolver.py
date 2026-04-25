"""Region resolver — locate the IR sub-graph corresponding to a slot.

Three-tier strategy (per Phase H+4 §5):

  Tier 1: explicit binding stored on the genome (set during pool admission
          / FII materialization / post-graft). Highest priority.

  Tier 2: provenance annotation ``op.attrs["_provenance"]["from_slot_id"]``
          matching the slot key (the path used by Phase H+3).

  Tier 3: legacy `slot` op-kind (kbest / bp / soft_sic / amp templates
          that still hold AlgSlot ops at FII-time).

Returns ``SlotRegionInfo`` containing the op_ids that constitute the
region, or ``None`` when no resolution succeeds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from algorithm_ir.ir.model import FunctionIR

if TYPE_CHECKING:
    from evolution.pool_types import AlgorithmGenome


@dataclass(frozen=True)
class SlotRegionInfo:
    slot_key: str
    short_name: str
    op_ids: frozenset[str]
    sids: frozenset[str]               # provenance from_slot_id values that matched
    tier: str                          # "binding" | "provenance" | "slot_op"


def _scan_provenance(ir: FunctionIR, short: str) -> tuple[set[str], set[str]]:
    """Return (op_ids, sids) for ops whose provenance matches ``short``."""
    op_ids: set[str] = set()
    sids: set[str] = set()
    for oid, op in ir.ops.items():
        prov = op.attrs.get("_provenance") if op.attrs else None
        if not isinstance(prov, dict):
            continue
        sid = prov.get("from_slot_id")
        if not isinstance(sid, str):
            continue
        if sid.startswith(f"_slot_{short}_") or sid == short or sid == f"_slot_{short}":
            op_ids.add(oid)
            sids.add(sid)
    return op_ids, sids


def _scan_slot_ops(ir: FunctionIR, short: str) -> set[str]:
    """Return op_ids of any ``slot`` opcode whose slot_id matches ``short``."""
    op_ids: set[str] = set()
    for oid, op in ir.ops.items():
        if op.opcode != "slot":
            continue
        sid = op.attrs.get("slot_id") if op.attrs else None
        if isinstance(sid, str) and (
            sid == short or sid.endswith(f".{short}") or sid.endswith(f"_{short}")
        ):
            op_ids.add(oid)
    return op_ids


def resolve_slot_region(
    genome: "AlgorithmGenome",
    slot_key: str,
) -> SlotRegionInfo | None:
    """Return ``SlotRegionInfo`` for ``slot_key`` on ``genome.ir`` or None.

    Note: this resolver never raises — it returns ``None`` for any
    slot that cannot be found, leaving the caller to either skip the
    slot (and prune its phantom population) or fall back to the legacy
    ``map_pop_key_to_from_slot_ids`` path.
    """
    short = slot_key.split(".")[-1]
    ir = getattr(genome, "ir", None)
    if ir is None:
        return None

    # Tier 1: explicit binding
    bindings = genome.metadata.get("slot_bindings", {}) if hasattr(genome, "metadata") else {}
    binding = bindings.get(slot_key) if isinstance(bindings, dict) else None
    if binding is not None and isinstance(binding, dict):
        sids = binding.get("from_slot_ids") or binding.get("sids") or []
        sid_set = set(sids) if sids else set()
        if sid_set:
            op_ids, _ = _scan_provenance(ir, short)
            if op_ids:
                return SlotRegionInfo(
                    slot_key=slot_key,
                    short_name=short,
                    op_ids=frozenset(op_ids),
                    sids=frozenset(sid_set),
                    tier="binding",
                )

    # Tier 2: provenance scan
    op_ids, sids = _scan_provenance(ir, short)
    if op_ids:
        return SlotRegionInfo(
            slot_key=slot_key,
            short_name=short,
            op_ids=frozenset(op_ids),
            sids=frozenset(sids),
            tier="provenance",
        )

    # Tier 3: legacy slot ops
    slot_op_ids = _scan_slot_ops(ir, short)
    if slot_op_ids:
        return SlotRegionInfo(
            slot_key=slot_key,
            short_name=short,
            op_ids=frozenset(slot_op_ids),
            sids=frozenset(),
            tier="slot_op",
        )

    return None


def is_resolvable(genome: "AlgorithmGenome", slot_key: str) -> bool:
    return resolve_slot_region(genome, slot_key) is not None


def prune_phantom_pops(genome: "AlgorithmGenome") -> int:
    """Remove slot populations whose region cannot be resolved.

    Returns the number of populations dropped.
    """
    pops = getattr(genome, "slot_populations", None)
    if not isinstance(pops, dict):
        return 0
    to_drop: list[str] = []
    for slot_key in list(pops.keys()):
        if resolve_slot_region(genome, slot_key) is None:
            to_drop.append(slot_key)
    for k in to_drop:
        del pops[k]
    return len(to_drop)
