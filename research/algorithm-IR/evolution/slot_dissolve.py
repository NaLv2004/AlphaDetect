"""Slot dissolution & stamping — Case III + Track B helpers.

This module owns the two complementary in-place mutations on a
:class:`AlgorithmGenome` that change its slot inventory:

* :func:`dissolve_slot` — Case III (M6a §6.5): a graft region cut a
  slot in half, so we must retire the slot's identity. Strips
  ``op.attrs["slot_id"]`` tags, removes the ``slot_meta`` entry, and
  archives the :class:`SlotPopulation` under
  ``genome.metadata["dissolved_slots"]`` for lineage.
* :func:`apply_slot_stamping` — Track B (M6b §6.7): the GNN found a
  cohesive sub-DAG that is *not yet* a slot. Tags its ops, inserts a
  fresh ``SlotMeta`` entry, and seeds a :class:`SlotPopulation` whose
  first variant is the stamped region's :class:`SubgraphSnapshot`.

Both helpers are pure mutators — they do not validate the surrounding
IR after their changes. The orchestrator is responsible for invoking
``validate_function_ir`` if a downstream step requires it.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from evolution.pool_types import AlgorithmGenome, SlotStampingProposal

logger = logging.getLogger(__name__)

__all__ = ["dissolve_slot", "apply_slot_stamping"]


def dissolve_slot(genome: "AlgorithmGenome", pop_key: str) -> bool:
    """Dissolve ``pop_key`` from ``genome``.

    Parameters
    ----------
    genome
        Genome whose IR + slot_populations + metadata are mutated in place.
    pop_key
        The slot population key to dissolve.

    Returns
    -------
    bool
        ``True`` if the slot was present and dissolved, ``False`` if it
        was already gone (idempotent).
    """
    ir = genome.ir
    if ir is None:
        return False

    op_ids: set[str] = set()
    meta = ir.slot_meta.get(pop_key) if ir.slot_meta else None
    if meta is not None:
        op_ids.update(meta.op_ids)

    for op_id, op in ir.ops.items():
        attrs = getattr(op, "attrs", None) or {}
        if attrs.get("slot_id") == pop_key:
            op_ids.add(op_id)

    if not op_ids and meta is None and pop_key not in genome.slot_populations:
        return False

    for op_id in op_ids:
        op = ir.ops.get(op_id)
        if op is None:
            continue
        attrs = getattr(op, "attrs", None)
        if attrs is None:
            continue
        if attrs.get("slot_id") == pop_key:
            attrs.pop("slot_id", None)

    if ir.slot_meta and pop_key in ir.slot_meta:
        del ir.slot_meta[pop_key]

    if pop_key in genome.slot_populations:
        archive = genome.metadata.setdefault("dissolved_slots", {})
        archive[pop_key] = genome.slot_populations.pop(pop_key)

    logger.debug(
        "dissolve_slot: genome=%s pop_key=%s n_ops_cleared=%d",
        genome.algo_id, pop_key, len(op_ids),
    )
    return True


def apply_slot_stamping(
    genome: "AlgorithmGenome",
    proposal: "SlotStampingProposal",
) -> bool:
    """Stamp a Track B :class:`SlotStampingProposal` onto ``genome``.

    Steps (verbatim from M6b §6.7):

    1. **Reject** if any op in ``proposal.op_ids`` is already tagged
       with a different ``slot_id`` (region overlaps an existing slot).
    2. **Reject** if ``proposal.suggested_pop_key`` already exists in
       ``ir.slot_meta`` (idempotency).
    3. Tag every op in ``proposal.op_ids`` with ``slot_id``.
    4. Insert a fresh :class:`SlotMeta` whose ``op_ids`` is the sorted
       op-set, ``inputs`` / ``outputs`` are taken from the proposal.
    5. Spawn a fresh :class:`SlotPopulation` seeded with a
       :class:`SubgraphSnapshot` extracted from the stamped region.

    Returns
    -------
    bool
        ``True`` on success, ``False`` if the proposal was rejected.
    """
    from algorithm_ir.ir.model import SlotMeta
    from evolution.pool_types import SlotPopulation, SubgraphSnapshot
    from evolution.skeleton_registry import ProgramSpec

    ir = genome.ir
    if ir is None:
        return False

    pop_key = proposal.suggested_pop_key
    if pop_key in (ir.slot_meta or {}):
        return False
    if pop_key in genome.slot_populations:
        return False

    # Reject overlap with existing slot tags.
    op_ids = list(proposal.op_ids)
    for op_id in op_ids:
        op = ir.ops.get(op_id)
        if op is None:
            return False
        attrs = getattr(op, "attrs", None) or {}
        existing = attrs.get("slot_id")
        if existing and existing != pop_key:
            return False

    # 3. Tag ops with slot_id.
    for op_id in op_ids:
        op = ir.ops.get(op_id)
        if op is None:
            continue
        attrs = getattr(op, "attrs", None)
        if attrs is None:
            op.attrs = {"slot_id": pop_key}
        else:
            attrs["slot_id"] = pop_key

    # 4. Insert SlotMeta.
    output_names = tuple(f"out_{i}" for i in range(len(proposal.outputs)))
    ir.slot_meta[pop_key] = SlotMeta(
        pop_key=pop_key,
        op_ids=tuple(op_ids),
        inputs=tuple(proposal.inputs),
        outputs=tuple(proposal.outputs),
        output_names=output_names,
        parent=None,
    )

    # 5. Spawn SlotPopulation, seed with SubgraphSnapshot.
    spec = ProgramSpec(
        name=pop_key,
        param_names=[f"in_{i}" for i in range(len(proposal.inputs))],
        param_types=["object" for _ in proposal.inputs],
        return_type="object",
    )
    population = SlotPopulation(
        slot_id=pop_key,
        spec=spec,
        variants=[],
        fitness=[],
        best_idx=0,
    )
    try:
        snapshot = SubgraphSnapshot.extract(ir, pop_key)
        snapshots = genome.metadata.setdefault("slot_snapshots", {})
        snapshots[pop_key] = snapshot
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(
            "apply_slot_stamping: snapshot extract failed for %s: %r",
            pop_key, exc,
        )
    genome.slot_populations[pop_key] = population
    # Track B reward telemetry — set initial reward to 0; the engine
    # bumps it +1 if a subsequent micro-evolution variant of this slot
    # passes R6 and improves fitness.
    rewards = genome.metadata.setdefault("track_b_rewards", {})
    rewards[pop_key] = 0.0
    return True
