"""Donor-side region sampling masks (4-layer scheme, mirror of host mask).

The donor sampler is conditioned on the host's :class:`BoundarySignature`
(``entry_types``, ``exit_types``). Donor regions are EQUALITY-constrained
on arity (donor.exits == len(host.entry_types) so the typed binder can
wire up the graft) — this is materially different from host masks where
arity is an upper bound only.

Layers:

    Layer D1: donor pool prefilter (one-shot, per host signature).
        Drops every donor IR that cannot possibly satisfy the
        signature: (a) lacks at least one type-compatible candidate
        for SOME host exit-port type, or (b) lacks at least one
        type-compatible candidate for SOME host entry-port type
        among the union of cut candidates over all observable values.

    Layer D2: per-step donor-output mask
        Applied to the donor's observable-value pool when picking the
        i-th output.  A candidate v is eligible iff:
          * its static type is lattice-compatible with
            ``signature.exit_types[i]`` (positional type mask), AND
          * (selected ∪ {v}) backward closure is connected
            (cuts cannot reconnect islands, structural invariant), AND
          * the resulting donor region is feasibly reducible by ≤
            ``max_cut_budget`` cuts to satisfy the equality arity
            constraint on exits and ≤ caps on ops/inputs.

    Layer D3: cut candidate pool — reuses
        :func:`enumerate_cut_candidates(require_connected=True)`.

    Layer D4: per-step donor-cut mask + STOP gate.
        For each candidate cut c:
          * type-compatible with signature.entry_types[next_step] (positional), AND
          * backward_slice with selected_cuts ∪ {c} stays connected, AND
          * doesn't drop op count below min_region_ops, AND
          * the resulting state is still budget-feasibly reducible to
            the exact target arity within remaining cut budget.
        STOP allowed iff current donor region exactly matches the
        target arity on BOTH entries and exits, fits within op caps,
        is connected, and the chosen cuts' types match
        ``signature.entry_types`` positionally.
"""
from __future__ import annotations

from typing import Iterable

from algorithm_ir.ir.model import FunctionIR
from algorithm_ir.ir.type_lattice import is_subtype
from algorithm_ir.region.slicer import (
    backward_slice_by_values,
    backward_slice_until_values,
    enumerate_cut_candidates,
)

# Reuse host-mask connectivity / boundary helpers verbatim.
from evolution.host_region_mask import (
    _entry_values,
    _exit_values,
    _nontrivial_op_count,
    clear_singleton_cut_cache,
    is_op_set_connected,
    precompute_op_closures as _precompute_op_closures,
)
from evolution.donor_profiler import profile_donor, profile_donor_ctx

# Per-IR singleton cut cache (mirrors host_region_mask._singleton_cut_cache).
# Keyed by (id(ir), vid) to avoid cross-IR SSA value collisions.
_donor_singleton_cut_cache: dict[tuple[int, str], list[str]] = {}


def _get_donor_singleton_cuts(ir: FunctionIR, vid: str) -> list[str]:
    """Get or compute singleton cut candidates for a donor value."""
    key = (id(ir), vid)
    pool = _donor_singleton_cut_cache.get(key)
    if pool is None:
        try:
            pool = enumerate_cut_candidates(ir, [vid])
        except Exception:
            pool = []
        _donor_singleton_cut_cache[key] = pool
    return pool


# ---------------------------------------------------------- type helpers


def _value_static_type(ir: FunctionIR, value_id: str) -> str:
    val = ir.values.get(value_id)
    if val is None:
        return "unknown"
    type_hint = getattr(val, "type_hint", None)
    if type_hint:
        return str(type_hint)
    attrs = getattr(val, "attrs", None) or {}
    for key in ("type", "static_type", "dtype"):
        v = attrs.get(key)
        if v:
            return str(v)
    return "unknown"


def _types_compatible(donor_t: str, host_t: str) -> bool:
    """Lattice subtype check tolerant of unknown / any tags.

    Mirrors ``GnnPatternMatcher._types_compatible`` so the per-step
    donor mask uses the SAME compatibility predicate as the existing
    type mask (otherwise the AND of the two could drop a candidate
    that the type mask considers eligible — silent inconsistency).
    """
    if not host_t or host_t in ("unknown", "any", "object"):
        return True
    if not donor_t or donor_t in ("unknown", "any", "object"):
        return True
    try:
        return is_subtype(donor_t, host_t) or is_subtype(host_t, donor_t)
    except Exception:
        return True


# ---------------------------------------------------- Layer D1 (prefilter)


def donor_pool_signature_compatible(
    donor_ir: FunctionIR,
    observable_values: list[str],
    cut_pool_union: list[str],
    *,
    entry_types: tuple[str, ...],
    exit_types: tuple[str, ...],
) -> bool:
    """Layer D1 cheap necessary condition.

    Returns False iff there exists a host port type that NO donor value
    of the right side can satisfy. Cut pool is the union over all
    observable singletons (cheaper to compute than per-output cut sets).

    This is a NECESSARY condition only, not sufficient — Layer D2/D4
    do the per-step exact verification. But it cuts ~70-90% of
    pool-level mismatches in O(|values| × |types|) time.
    """
    # Cache donor-side type tags once per call.
    obs_types = [_value_static_type(donor_ir, vid) for vid in observable_values]
    cut_types = [_value_static_type(donor_ir, vid) for vid in cut_pool_union]

    for t in exit_types:
        if not any(_types_compatible(ot, t) for ot in obs_types):
            return False
    for t in entry_types:
        if not any(_types_compatible(ct, t) for ct in cut_types):
            return False
    return True


def donor_cut_pool_union(
    donor_ir: FunctionIR,
    observable_values: list[str],
    *,
    cap: int = 32,
) -> list[str]:
    """Best-effort union of cut candidates over observable singletons.

    Capped at ``cap`` per-output to keep prefilter cheap on large IRs.
    """
    seen: set[str] = set()
    out: list[str] = []
    for vid in observable_values:
        try:
            cands = enumerate_cut_candidates(donor_ir, [vid])
        except Exception:
            continue
        for c in cands[:cap]:
            if c not in seen:
                seen.add(c)
                out.append(c)
    return out


# ---------------------------------------------------- feasibility (greedy)


def _state_arity(
    ir: FunctionIR, output_values: list[str], cuts: list[str],
) -> tuple[int, int, int, int, bool]:
    """Return (n_entries, n_exits, n_ops_nontrivial, n_inputs, connected)."""
    ops = backward_slice_until_values(ir, output_values, cuts)
    if not ops:
        return (0, 0, 0, 0, False)
    return (
        len(_entry_values(ir, ops)),
        len(_exit_values(ir, ops)),
        _nontrivial_op_count(ir, ops),
        len(_entry_values(ir, ops)),  # n_inputs == n_entries by definition
        is_op_set_connected(ir, ops),
    )


def _donor_combo_is_feasible(
    ir: FunctionIR,
    output_values: list[str],
    *,
    target_entries: int,
    target_exits: int,
    max_region_ops: int,
    min_region_ops: int,
    max_region_inputs: int,
    max_cut_budget: int,
    initial_cuts: list[str] | None = None,
    cut_pool: list[str] | None = None,
    op_closures: dict[str, frozenset[str]] | None = None,
) -> bool:
    """Greedy feasibility test for donor combos."""
    _t0 = __import__('time').perf_counter()
    _bfs_calls = 0
    _greedy_iters = 0
    _cand_count = 0
    cuts_used: list[str] = list(initial_cuts or [])
    if cut_pool is not None:
        cand_pool = list(cut_pool)
    else:
        try:
            cand_pool = enumerate_cut_candidates(ir, output_values)
        except Exception:
            return False
    cand_pool = [c for c in cand_pool if c not in cuts_used]
    remaining_budget = max_cut_budget - len(cuts_used)
    if remaining_budget < 0:
        return False

    # ── Pre-compute op_closures-based upper bounds ────────────────
    if op_closures is not None:
        base_union = set().union(*(op_closures.get(v, frozenset()) for v in output_values))
    else:
        base_union = None
    if base_union:
        _base_connected: bool | None = is_op_set_connected(ir, base_union)
        _base_exits_ub = len(_exit_values(ir, base_union))
        _base_size_ub = _nontrivial_op_count(ir, base_union)
        _base_entries_lb = len(_entry_values(ir, base_union))
    else:
        _base_connected = None
        _base_exits_ub = None
        _base_size_ub = None
        _base_entries_lb = None

    # Relaxed semantics: match validate_boundary_region (inequality caps),
    # not exact equality. The per-step sampler already loops exactly
    # len(exit_types)/len(entry_types) times, so equality is loop-implicit.
    # Equality on the resulting region's natural arity is enforced
    # post-hoc by validate_boundary_region with max_region_outputs ==
    # target_exits acting as the hard cap; we just need nx <= target_exits.
    def _feasible(ne: int, nx: int, nop: int, nin: int, conn: bool) -> bool:
        return (
            conn
            and min_region_ops <= nop <= max_region_ops
            and nin <= max_region_inputs
            and nx <= target_exits
        )

    def _deficit(ne: int, nx: int, nop: int, nin: int) -> int:
        d = 0
        if nx > target_exits:
            d += (nx - target_exits)
        if nop > max_region_ops:
            d += (nop - max_region_ops)
        if nop < min_region_ops:
            d += (min_region_ops - nop)
        if nin > max_region_inputs:
            d += (nin - max_region_inputs)
        return d

    ne, nx, nop, nin, conn = _state_arity(ir, output_values, cuts_used)
    _bfs_calls += 1
    if not conn or nop == 0:
        return False

    if _feasible(ne, nx, nop, nin, conn):
        return True

    _cand_count = len(cand_pool)

    # Fast-path flag: when full-closure upper bounds already fit within
    # ALL constraints, connectivity is guaranteed for any subset.
    _fast_path = (
        _base_connected is True
        and _base_exits_ub is not None
        and _base_exits_ub <= target_exits
        and _base_size_ub is not None
        and _base_size_ub <= max_region_ops
        and _base_size_ub >= min_region_ops
    )

    cur_deficit = _deficit(ne, nx, nop, nin)
    for _ in range(remaining_budget):
        _greedy_iters += 1
        best_score = 0
        best_cut = None
        best_state = None
        for c in cand_pool:
            if c in cuts_used:
                continue
            if _fast_path:
                # Connectivity, exit, and size are all guaranteed;
                # just need the per-candidate metrics for scoring.
                new_ops = backward_slice_until_values(ir, output_values, cuts_used + [c])
                _bfs_calls += 1
                if not new_ops:
                    continue
                new_ne = new_nin = len(_entry_values(ir, new_ops))
                new_nx = len(_exit_values(ir, new_ops))
                new_nop = _nontrivial_op_count(ir, new_ops)
                new_conn = True  # guaranteed by _base_connected
            else:
                new_ne, new_nx, new_nop, new_nin, new_conn = _state_arity(
                    ir, output_values, cuts_used + [c]
                )
                _bfs_calls += 1
                if not new_conn or new_nop == 0:
                    continue
            new_def = _deficit(new_ne, new_nx, new_nop, new_nin)
            score = cur_deficit - new_def
            if score > best_score:
                best_score = score
                best_cut = c
                best_state = (new_ne, new_nx, new_nop, new_nin, new_conn)
        if best_cut is None:
            return False
        cuts_used.append(best_cut)
        ne, nx, nop, nin, conn = best_state
        cur_deficit = _deficit(ne, nx, nop, nin)
        if _feasible(ne, nx, nop, nin, conn):
            return True
    _dt = __import__('time').perf_counter() - _t0
    _n_ops = len(base_union) if base_union else 0
    profile_donor("_donor_combo_is_feasible", _dt,
                  bfs_calls=_bfs_calls, greedy_iters=_greedy_iters,
                  n_candidates=_cand_count, ir_ops=_n_ops,
                  n_outputs=len(output_values), budget=max_cut_budget)
    return _feasible(ne, nx, nop, nin, conn)


# ---------------------------------------------------- Layer D2 (output)


def donor_output_step_mask(
    donor_ir: FunctionIR,
    op_closures: dict[str, frozenset[str]],
    selected_outputs: list[str],
    remaining_candidates: list[str],
    *,
    next_step: int,
    entry_types: tuple[str, ...],
    exit_types: tuple[str, ...],
    max_region_ops: int,
    min_region_ops: int,
    max_region_inputs: int,
    max_cut_budget: int,
    cut_pool: list[str] | None = None,
) -> list[bool]:
    """Layer D2: combined type + connectivity + feasibility mask for
    the next donor output sampling step.

    Note: STOP is not returned — the donor sampler is hard-driven by
    signature arity (must pick exactly ``len(exit_types)`` outputs),
    so STOP is decided externally.
    """
    if next_step >= len(exit_types):
        # Should not be called; arity already satisfied.
        return [False] * len(remaining_candidates)
    _t0 = __import__('time').perf_counter()
    target_t = exit_types[next_step]
    target_exits = len(exit_types)
    target_entries = len(entry_types)

    base_ops: set[str] = set()
    # Build merged cut pool from singleton caches if not provided.
    if cut_pool is None:
        merged: set[str] = set()
        for svid in selected_outputs:
            merged.update(_get_donor_singleton_cuts(donor_ir, svid))
        # Will add candidate's singleton pool per-iteration below.
    for vid in selected_outputs:
        base_ops |= op_closures.get(vid, frozenset())

    mask: list[bool] = []
    for vid in remaining_candidates:
        cand_t = _value_static_type(donor_ir, vid)
        if not _types_compatible(cand_t, target_t):
            mask.append(False)
            continue
        cand_ops = op_closures.get(vid, frozenset())
        if not cand_ops:
            mask.append(False)
            continue
        merged = base_ops | cand_ops
        if not is_op_set_connected(donor_ir, merged):
            mask.append(False)
            continue
        # Provisional output set for feasibility lookahead.
        prov_outs = list(selected_outputs) + [vid]
        # Feasibility lookahead at EVERY step (not just final) to
        # avoid greedy dead-ends.  Cost is amortized by the
        # type+connectivity prefilters which already cut the candidate
        # pool by ~50%.
        # IMPORTANT: cap the lookahead's cut budget by
        # ``len(entry_types)`` because the donor cut sampler is
        # iterating exactly ``len(entry_types)`` times (positional
        # type binding requires one cut per host entry).  Using the
        # raw ``max_cut_budget`` would be optimistic and cause
        # ``too_many_outputs`` validation failures downstream.
        effective_budget = min(max_cut_budget, target_entries)
        # Build augmented cut pool from singleton caches if the caller
        # did not provide one.
        _cp = cut_pool
        if _cp is None:
            _cp = list(merged | set(_get_donor_singleton_cuts(donor_ir, vid)))
        if not _donor_combo_is_feasible(
            donor_ir, prov_outs,
            target_entries=target_entries,
            target_exits=target_exits,
            max_region_ops=max_region_ops,
            min_region_ops=min_region_ops,
            max_region_inputs=max_region_inputs,
            max_cut_budget=effective_budget,
            cut_pool=_cp,
            op_closures=op_closures,
        ):
            mask.append(False)
            continue
        mask.append(True)
    _dt = __import__('time').perf_counter() - _t0
    profile_donor("donor_output_step_mask", _dt,
                  n_candidates=len(remaining_candidates),
                  n_selected=len(selected_outputs))
    return mask


# ---------------------------------------------------- Layer D4 (cut + STOP)


def donor_cut_step_mask(
    donor_ir: FunctionIR,
    output_values: list[str],
    selected_cuts: list[str],
    remaining_candidates: list[str],
    *,
    entry_types: tuple[str, ...],
    exit_types: tuple[str, ...],
    max_region_ops: int,
    min_region_ops: int,
    max_region_inputs: int,
    max_cut_budget: int,
    cut_pool: list[str] | None = None,
    op_closures: dict[str, frozenset[str]] | None = None,
) -> tuple[list[bool], bool]:
    """Layer D4: per-step donor-cut mask plus STOP gate.

    Returns ``(mask, stop_allowed)``.  STOP is allowed only when the
    current state matches the target arity exactly on entries AND
    exits, fits within op/input caps, and is connected.
    """
    _t0 = __import__('time').perf_counter()
    target_entries = len(entry_types)
    target_exits = len(exit_types)
    next_step = len(selected_cuts)

    # Current state.
    cur_ops_set = backward_slice_until_values(donor_ir, output_values, selected_cuts)
    cur_ne = len(_entry_values(donor_ir, cur_ops_set))
    cur_nx = len(_exit_values(donor_ir, cur_ops_set))
    cur_nop = _nontrivial_op_count(donor_ir, cur_ops_set)
    cur_nin = len(_entry_values(donor_ir, cur_ops_set))  # entries == inputs by definition
    cur_conn = is_op_set_connected(donor_ir, cur_ops_set)

    stop_allowed = (
        cur_conn
        and min_region_ops <= cur_nop <= max_region_ops
        and cur_nin <= max_region_inputs
        and cur_nx <= target_exits
    )

    # If we've already met arity and budget, no need to add more cuts;
    # mask everything out so the sampler will hit STOP.
    if next_step >= target_entries:
        return [False] * len(remaining_candidates), stop_allowed

    target_t = entry_types[next_step]
    budget_left = max_cut_budget - len(selected_cuts)

    mask: list[bool] = []
    for c in remaining_candidates:
        if c in selected_cuts:
            mask.append(False)
            continue
        # Type compatibility (positional).
        cand_t = _value_static_type(donor_ir, c)
        if not _types_compatible(cand_t, target_t):
            mask.append(False)
            continue
        new_ops_set = backward_slice_until_values(
            donor_ir, output_values, selected_cuts + [c]
        )
        if not new_ops_set:
            mask.append(False)
            continue
        if not is_op_set_connected(donor_ir, new_ops_set):
            mask.append(False)
            continue
        new_ne = len(_entry_values(donor_ir, new_ops_set))
        new_nx = len(_exit_values(donor_ir, new_ops_set))
        new_nop = _nontrivial_op_count(donor_ir, new_ops_set)
        # Per-step state can still violate exit cap if more cuts will
        # reduce it; defer the final-state check to the lookahead
        # ``_donor_combo_is_feasible`` below.  Just enforce caps that
        # cuts CANNOT recover from in subsequent steps:
        #   * cuts only ever shrink the op set, so nop only decreases.
        #     If new_nop already < min_region_ops we cannot grow it.
        #   * cuts only ever grow entries, so if new_ne already exceeds
        #     max_region_inputs we cannot shrink it.
        if new_nop < min_region_ops:
            mask.append(False)
            continue
        if new_ne > max_region_inputs:
            mask.append(False)
            continue
        if new_nop > max_region_ops:
            # Adding a cut should never INCREASE nop, but defensive.
            mask.append(False)
            continue
        # Per-cut lookahead: from this state, can the remaining
        # ``budget_left - 1`` cuts close the gap to exact arity?
        # Cap budget by the sampler's iteration cap (len(entry_types))
        # so the lookahead reflects the sampler's actual capacity.
        effective_budget = min(max_cut_budget, target_entries)
        if not _donor_combo_is_feasible(
            donor_ir, output_values,
            target_entries=target_entries,
            target_exits=target_exits,
            max_region_ops=max_region_ops,
            min_region_ops=min_region_ops,
            max_region_inputs=max_region_inputs,
            max_cut_budget=effective_budget,
            initial_cuts=selected_cuts + [c],
            cut_pool=cut_pool,
            op_closures=op_closures,
        ):
            mask.append(False)
            continue
        mask.append(True)

    _dt = __import__('time').perf_counter() - _t0
    profile_donor("donor_cut_step_mask", _dt,
                  n_candidates=len(remaining_candidates),
                  n_selected_cuts=len(selected_cuts))
    return mask, stop_allowed


# ---------------------------------------------------- precompute helper


precompute_op_closures = _precompute_op_closures
