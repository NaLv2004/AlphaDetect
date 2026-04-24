"""Typed bipartite host-donor binding for graft_general.

Decision rule
=============
For every donor argument, score every visible host candidate against a
weighted cost combining

  * **Type distance** in the lattice (``algorithm_ir.ir.type_lattice``):
    0 = exact match, 1 = lattice subtype, 2 = unify-only common ancestor,
    ``∞`` = incompatible (the host candidate's type cannot be coerced
    into the donor argument's type without lying).
  * **Name-hint similarity**: 0 if the hints are equal, 1 if one is a
    prefix or substring of the other, 2 otherwise.
  * **Dataflow recency**: a small bonus for host values defined closer
    to the splice point so the binding does not pull a stale value
    when a fresher one is available.
  * **Callable-result confidence**: a small bonus for values whose
    producing op carries a registered ``qualified_name`` (i.e. the
    type comes from the registry, not the fallback).

The optimal one-to-one assignment is computed via the Hungarian
algorithm (``scipy.optimize.linear_sum_assignment``).  When no feasible
assignment exists (some donor arg has *no* type-compatible host
candidate), the function returns ``None`` and the caller is expected to
either fall back to the legacy name-hint matcher or to reject the
proposal entirely — both behaviours are acceptable, but the strong
default in :func:`bind_typed` is to **fail-closed**: the resulting
mapping is only returned when every donor argument received a
type-compatible host value.

This module performs **no Python-level execution**; it is pure
combinatorial reasoning over the IR's static type tags.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import numpy as np

from algorithm_ir.ir.type_lattice import is_subtype, unify

if TYPE_CHECKING:
    from algorithm_ir.ir.model import FunctionIR, Value

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cost weights — kept module-level so they can be tuned (or A/B'd) without
# editing call-sites.
# ---------------------------------------------------------------------------

WEIGHT_TYPE = 10.0       # dominates: type incompat is the dealbreaker
WEIGHT_NAME = 1.0        # tie-break between equally-typed candidates
WEIGHT_DATAFLOW = 0.1    # very mild preference for recent definitions
WEIGHT_CALL_CONF = 0.5   # mild bonus for registry-typed values
INFEASIBLE = 1e9         # used in lieu of ``inf`` to keep Hungarian numerically stable

# Generic value name_hints emitted by the lifter that carry no semantic
# weight and must not be rewarded by the name-similarity term.
_GENERIC_NAMES = frozenset({
    "binary", "unary", "compare", "call", "get_attr", "get_item",
    "iter_init", "iter_next", "iter_has_next", "phi", "const",
    "const_int", "const_float", "module", "function",
})


@dataclass
class TypedBindingResult:
    """Outcome of :func:`bind_typed`.

    Attributes
    ----------
    mapping
        ``donor_arg_vid -> host_value_vid`` for every donor argument.
        Empty if ``feasible`` is ``False``.
    feasible
        ``True`` iff every donor arg was assigned to a type-compatible
        host candidate (no ``INFEASIBLE`` cell on the optimal diagonal).
    cost
        Total weighted cost of the chosen assignment (``∞`` if infeasible).
    diagnostics
        Per-arg ``(donor_vid, host_vid, donor_type, host_type, cost)``
        rows used for logging / debugging.
    """

    mapping: dict[str, str]
    feasible: bool
    cost: float
    diagnostics: list[tuple[str, str, str, str, float]]


# ---------------------------------------------------------------------------
# Cost terms
# ---------------------------------------------------------------------------

def _type_cost(donor_type: str, host_type: str) -> float:
    """Lattice distance between a donor's required type and a host candidate's type.

    Returns ``INFEASIBLE`` when neither direction of subtyping holds and
    the unify of the two types collapses to the lattice top ("any").
    """
    dt = donor_type or "any"
    ht = host_type or "any"
    if dt == ht:
        return 0.0
    # ``any`` on either side is a wildcard: it matches anything but
    # carries the smallest possible reward so a more specific candidate
    # always wins.
    if dt == "any" or ht == "any":
        return 1.5
    if is_subtype(ht, dt):
        return 1.0
    if is_subtype(dt, ht):
        # Donor is more specific than host candidate — still admissible
        # because the host value will be passed in opaquely; the donor
        # body's narrower assumptions are what matter at runtime.
        return 1.5
    common = unify(dt, ht)
    if common == "any":
        return INFEASIBLE
    return 2.0


def _name_cost(donor_hints: list[str], host_hints: list[str]) -> float:
    """Heuristic name similarity between donor arg hints and host candidate hints."""
    d_clean = [h for h in donor_hints if h and h not in _GENERIC_NAMES]
    h_clean = [h for h in host_hints if h and h not in _GENERIC_NAMES]
    if not d_clean or not h_clean:
        return 2.0
    for d in d_clean:
        if d in h_clean:
            return 0.0
    for d in d_clean:
        for h in h_clean:
            if d == h:
                return 0.0
            if d.startswith(h) or h.startswith(d):
                return 0.5
            if d in h or h in d:
                return 1.0
    return 2.0


def _dataflow_cost(host_block_index: int, splice_block_index: int) -> float:
    """Mildly prefer host values defined nearer (and before) the splice point."""
    if host_block_index < 0 or splice_block_index < 0:
        return 1.0
    if host_block_index > splice_block_index:
        # Defined *after* the splice point: not visible without
        # reordering — treat as infeasible at this layer.
        return INFEASIBLE
    return min(1.0, (splice_block_index - host_block_index) / 100.0)


def _value_hints(value: "Value") -> list[str]:
    if value is None:
        return []
    return [
        value.name_hint,
        value.attrs.get("var_name") if isinstance(value.attrs, dict) else None,
        value.attrs.get("name") if isinstance(value.attrs, dict) else None,
    ]


def _call_conf_bonus(host_ir: "FunctionIR", host_vid: str) -> float:
    """Small reward for host values typed via the callable registry."""
    val = host_ir.values.get(host_vid)
    if val is None or val.def_op is None:
        return 0.0
    op = host_ir.ops.get(val.def_op)
    if op is None or op.opcode != "call":
        return 0.0
    if isinstance(op.attrs, dict) and op.attrs.get("qualified_name"):
        return -WEIGHT_CALL_CONF
    return 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def collect_visible_host_values(
    host_ir: "FunctionIR",
    splice_op_ids: Iterable[str],
) -> list[str]:
    """Return host value ids visible at the graft splice point.

    Visible = function arguments + outputs of any op that strictly
    precedes the first op of the graft region in block order.  This
    is a coarse but correct over-approximation: a true SSA dominance
    analysis is unnecessary because we additionally penalize
    ``host_block_index > splice_block_index`` to ``INFEASIBLE`` in
    :func:`_dataflow_cost`.
    """
    splice_set = set(splice_op_ids)
    if not splice_set:
        # No region: everything is visible (rare but legal).
        return list(host_ir.values.keys())

    # Build a global op-index (block-major linearisation).
    op_index: dict[str, int] = {}
    idx = 0
    for block in host_ir.blocks.values():
        for oid in block.op_ids:
            op_index[oid] = idx
            idx += 1

    splice_indices = [op_index[oid] for oid in splice_set if oid in op_index]
    if not splice_indices:
        return list(host_ir.values.keys())
    splice_first = min(splice_indices)

    visible: list[str] = list(host_ir.arg_values)
    seen = set(visible)
    for oid, opi in op_index.items():
        if opi >= splice_first:
            continue
        op = host_ir.ops.get(oid)
        if op is None:
            continue
        for out in op.outputs:
            if out not in seen:
                visible.append(out)
                seen.add(out)
    return visible


def bind_typed(
    donor_ir: "FunctionIR",
    host_ir: "FunctionIR",
    splice_op_ids: Iterable[str],
    *,
    require_feasible: bool = True,
) -> TypedBindingResult | None:
    """Compute the optimal type-aware donor → host argument binding.

    Parameters
    ----------
    donor_ir
        The trimmed donor function whose ``arg_values`` need binding.
    host_ir
        The host function (already deep-copied for the graft).
    splice_op_ids
        Ops belonging to the host graft region — used to derive the
        visible-value set and the dataflow-recency cost term.
    require_feasible
        When ``True`` (the default), return ``None`` if any donor
        argument cannot be matched to a type-compatible host
        candidate.  When ``False``, return the best-effort assignment
        even if one or more cells were ``INFEASIBLE``.

    Returns
    -------
    TypedBindingResult | None
    """
    donor_args = list(donor_ir.arg_values)
    if not donor_args:
        return TypedBindingResult({}, feasible=True, cost=0.0, diagnostics=[])

    visible = collect_visible_host_values(host_ir, splice_op_ids)
    if not visible:
        logger.debug("typed_binding: no visible host values at splice site")
        return None

    # Splice-point index (used for dataflow cost).  We pick the first op
    # in the graft region as the splice anchor.
    op_index: dict[str, int] = {}
    idx = 0
    for block in host_ir.blocks.values():
        for oid in block.op_ids:
            op_index[oid] = idx
            idx += 1
    splice_indices = [op_index[oid] for oid in splice_op_ids if oid in op_index]
    splice_anchor = min(splice_indices) if splice_indices else len(op_index)

    # Build cost matrix [n_donor, n_host_candidates].
    n_donor = len(donor_args)
    n_host = len(visible)
    cost = np.full((n_donor, max(n_host, n_donor)), INFEASIBLE, dtype=np.float64)

    for di, dvid in enumerate(donor_args):
        dval = donor_ir.values.get(dvid)
        d_type = (dval.type_hint if dval else "any") or "any"
        d_hints = _value_hints(dval)
        for hi, hvid in enumerate(visible):
            hval = host_ir.values.get(hvid)
            if hval is None:
                continue
            h_type = (hval.type_hint or "any")
            h_hints = _value_hints(hval)
            t = _type_cost(d_type, h_type)
            if t >= INFEASIBLE:
                continue
            n = _name_cost(d_hints, h_hints)
            host_op_idx = op_index.get(hval.def_op or "", -1)
            df = _dataflow_cost(host_op_idx, splice_anchor)
            if df >= INFEASIBLE:
                continue
            cb = _call_conf_bonus(host_ir, hvid)
            cost[di, hi] = (
                WEIGHT_TYPE * t
                + WEIGHT_NAME * n
                + WEIGHT_DATAFLOW * df
                + cb
            )

    # Pad with dummy infeasible columns if fewer host candidates than
    # donor args (Hungarian needs a square or wider matrix).
    if n_host < n_donor:
        # Already padded with INFEASIBLE in the np.full above.
        pass

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:  # pragma: no cover — scipy is a hard project dep
        logger.warning("scipy unavailable; typed_binding cannot run")
        return None

    row_ind, col_ind = linear_sum_assignment(cost)

    mapping: dict[str, str] = {}
    diagnostics: list[tuple[str, str, str, str, float]] = []
    total_cost = 0.0
    feasible = True
    for r, c in zip(row_ind, col_ind):
        donor_vid = donor_args[r]
        cell_cost = cost[r, c]
        if c < n_host:
            host_vid = visible[c]
        else:
            host_vid = ""  # padded slot — no real candidate
        if cell_cost >= INFEASIBLE:
            feasible = False
        else:
            mapping[donor_vid] = host_vid
        dval = donor_ir.values.get(donor_vid)
        hval = host_ir.values.get(host_vid) if host_vid else None
        diagnostics.append((
            donor_vid,
            host_vid,
            (dval.type_hint if dval else "?") or "?",
            (hval.type_hint if hval else "?") or "?",
            float(cell_cost),
        ))
        total_cost += float(cell_cost) if cell_cost < INFEASIBLE else 0.0

    if require_feasible and not feasible:
        return None
    return TypedBindingResult(
        mapping=mapping,
        feasible=feasible,
        cost=total_cost,
        diagnostics=diagnostics,
    )


__all__ = [
    "INFEASIBLE",
    "TypedBindingResult",
    "bind_typed",
    "collect_visible_host_values",
]
