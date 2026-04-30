"""Host-side region sampling masks (4-layer scheme).

The masks make the host (output, cut) sampler emit ONLY structurally
and numerically valid regions — no post-hoc fallback / greedy repair
required. Layers:

    Layer 1: dead-code output filter (one-shot, on the candidate pool).
        Drops every observable value that is not in the host's
        return-slice (i.e. cannot affect the function's output).

    Layer 2: per-step output mask (connectivity invariant).
        When deciding whether to add a 2nd / 3rd output, mask out any
        candidate whose backward-closure forms a disjoint connected
        component from the closure already chosen. ``cut`` cannot
        re-connect islands, so this is a hard structural rule.

    Layer 3: cut candidate pool (already implemented elsewhere via
        ``enumerate_cut_candidates(require_connected=True)``).

    Layer 4: per-step cut mask + STOP gate (numerical invariants).
        For each candidate cut: simulate the resulting region and mask
        it out if it would (a) disconnect, (b) blow inputs > max_inputs,
        (c) leave 0 ops. ``op`` count is monotone-decreasing in cuts so
        the size upper bound is never violated by adding cuts; only
        the STOP gate enforces it.
        STOP is *forbidden* until the current region is fully valid
        (size in [min, max], inputs <= max, exits <= max, connected).

The masks are designed so that any sequence the sampler can produce
yields a region passing :func:`validate_boundary_region` *without*
the greedy repair fallback in ``_build_boundary_region``.
"""
from __future__ import annotations

from collections import deque
from typing import Iterable

from algorithm_ir.ir.model import FunctionIR
from algorithm_ir.region.slicer import (
    backward_slice_by_values,
    backward_slice_until_values,
)
from algorithm_ir.region.triviality import is_trivial_op


# ---------------------------------------------------------------- singleton cut cache

_singleton_cut_cache: dict[str, list[str]] = {}


def clear_singleton_cut_cache() -> None:
    """Clear the module-level singleton cut cache.

    Call at the beginning of each ``_propose_pairs`` invocation to avoid
    unbounded cross-generation accumulation.
    """
    _singleton_cut_cache.clear()


# ---------------------------------------------------------------- precompute


def filter_dead_code_outputs(
    observable_values: list[str],
    return_slice_values: set[str],
) -> tuple[list[str], list[int]]:
    """Layer 1: drop observable values absent from the return-slice.

    Returns (kept_values, kept_indices_into_original).
    """
    kept_values: list[str] = []
    kept_idx: list[int] = []
    for i, vid in enumerate(observable_values):
        if vid in return_slice_values:
            kept_values.append(vid)
            kept_idx.append(i)
    return kept_values, kept_idx


def precompute_op_closures(
    ir: FunctionIR,
    values: Iterable[str],
) -> dict[str, frozenset[str]]:
    """For each value v, the set of ops in the unbounded backward slice.

    Cached on the host context so connectivity checks are O(|union|)
    rather than re-running BFS each step.
    """
    closures: dict[str, frozenset[str]] = {}
    for vid in values:
        closures[vid] = frozenset(backward_slice_by_values(ir, [vid]))
    return closures


# --------------------------------------------------------------- connectivity


def is_op_set_connected(ir: FunctionIR, op_set: set[str]) -> bool:
    """Undirected-dataflow connectivity over the op subgraph.

    Mirrors ``algorithm_ir.region.slicer._is_region_connected`` but
    scoped to the helper module.
    """
    if not op_set:
        return False
    if len(op_set) == 1:
        return True
    adjacency: dict[str, set[str]] = {oid: set() for oid in op_set}
    for oid in op_set:
        op = ir.ops.get(oid)
        if op is None:
            continue
        for vid in op.inputs:
            v = ir.values.get(vid)
            if v and v.def_op in op_set:
                adjacency[oid].add(v.def_op)
                adjacency[v.def_op].add(oid)
        for vid in op.outputs:
            v = ir.values.get(vid)
            if v is None:
                continue
            for use_op in v.use_ops:
                if use_op in op_set:
                    adjacency[oid].add(use_op)
                    adjacency[use_op].add(oid)
    start = next(iter(op_set))
    seen = {start}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for nxt in adjacency[cur]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen == op_set


def _entry_values(ir: FunctionIR, op_set: set[str]) -> set[str]:
    """Values used by ops in op_set whose def_op is outside op_set."""
    entry: set[str] = set()
    for oid in op_set:
        op = ir.ops.get(oid)
        if op is None:
            continue
        for vid in op.inputs:
            v = ir.values.get(vid)
            if v is None:
                continue
            if v.def_op is None or v.def_op not in op_set:
                entry.add(vid)
    return entry


def _exit_values(
    ir: FunctionIR, op_set: set[str], extra_outputs: Iterable[str] = (),
) -> set[str]:
    """Mirror :func:`define_rewrite_region`'s region.exit_values:
    values defined in op_set whose use_ops include at least one op
    outside op_set. The ``extra_outputs`` argument is *not* added
    automatically (the selector only adds them via the legacy
    ``exit_values`` keyword arg, which boundary-spec mode does not
    use), so we deliberately ignore it here.
    """
    del extra_outputs  # kept for caller-API compatibility
    exit_vals: set[str] = set()
    for oid in op_set:
        op = ir.ops.get(oid)
        if op is None:
            continue
        for vid in op.outputs:
            v = ir.values.get(vid)
            if v is None:
                continue
            for use_op in v.use_ops:
                if use_op not in op_set:
                    exit_vals.add(vid)
                    break
    return exit_vals


def _nontrivial_op_count(ir: FunctionIR, op_set: set[str]) -> int:
    return sum(1 for oid in op_set if oid in ir.ops and not is_trivial_op(ir.ops[oid], ir))


# ------------------------------------------------------------ Layer 2 (output)


def output_step_mask(
    ir: FunctionIR,
    op_closures: dict[str, frozenset[str]],
    selected: list[str],
    remaining: list[str],
    *,
    max_region_ops: int | None = None,
    max_region_outputs: int | None = None,
    max_region_inputs: int | None = None,
    max_cut_budget: int | None = None,
) -> tuple[list[bool], bool]:
    """Layer 2: which candidates are still legal for the next output step.

    A candidate v is masked out if:
      * (selected ∪ {v}) closure is *not connected* (cuts cannot
        re-connect islands — hard structural rule), OR
      * even after applying *all* possible cuts, the resulting region
        would still violate ``max_region_ops`` or ``max_region_outputs``
        (dead-end pruning; cuts are monotone-reducing on both metrics).

    STOP: forbidden at step 0 (must pick at least one output);
    allowed at step ≥ 1.
    """
    if not selected:
        # Step 0 — nothing to compare against; only the per-candidate
        # feasibility check applies.
        if max_region_ops is None and max_region_outputs is None:
            return [True] * len(remaining), False
        mask: list[bool] = []
        for vid in remaining:
            cut_pool = _singleton_cut_cache.get(vid)
            if cut_pool is None:
                from algorithm_ir.region.slicer import enumerate_cut_candidates
                try:
                    cut_pool = enumerate_cut_candidates(ir, [vid])
                except Exception:
                    cut_pool = []
                _singleton_cut_cache[vid] = cut_pool
            mask.append(_combo_is_feasible(
                ir, [vid],
                op_closures,
                max_region_ops=max_region_ops,
                max_region_outputs=max_region_outputs,
                max_region_inputs=max_region_inputs,
                max_cut_budget=max_cut_budget,
                cut_pool=cut_pool,
            ))
        return mask, False

    base_ops: set[str] = set()
    # Pre-merge selected-value cut pools (reused across all candidates).
    merged_cut_pool_set: set[str] = set()
    for svid in selected:
        scp = _singleton_cut_cache.get(svid)
        if scp is None:
            from algorithm_ir.region.slicer import enumerate_cut_candidates
            try:
                scp = enumerate_cut_candidates(ir, [svid])
            except Exception:
                scp = []
            _singleton_cut_cache[svid] = scp
        merged_cut_pool_set.update(scp)
    for vid in selected:
        base_ops |= op_closures.get(vid, frozenset())

    mask: list[bool] = []
    for vid in remaining:
        cand_ops = op_closures.get(vid, frozenset())
        merged = base_ops | cand_ops
        if not merged:
            mask.append(False)
            continue
        if not is_op_set_connected(ir, merged):
            mask.append(False)
            continue
        if (max_region_ops is not None or max_region_outputs is not None):
            outs = list(selected) + [vid]
            # Add candidate's singleton cut pool to the merged set.
            vid_pool = _singleton_cut_cache.get(vid)
            if vid_pool is None:
                from algorithm_ir.region.slicer import enumerate_cut_candidates
                try:
                    vid_pool = enumerate_cut_candidates(ir, [vid])
                except Exception:
                    vid_pool = []
                _singleton_cut_cache[vid] = vid_pool
            cut_pool = list(merged_cut_pool_set | set(vid_pool))
            if not _combo_is_feasible(
                ir, outs,
                op_closures,
                max_region_ops=max_region_ops,
                max_region_outputs=max_region_outputs,
                max_region_inputs=max_region_inputs,
                max_cut_budget=max_cut_budget,
                cut_pool=cut_pool,
            ):
                mask.append(False)
                continue
        mask.append(True)
    return mask, True


def _combo_is_feasible(
    ir: FunctionIR,
    output_values: list[str],
    op_closures: dict[str, frozenset[str]] | None = None,
    *,
    max_region_ops: int | None,
    max_region_outputs: int | None,
    max_region_inputs: int | None = None,
    max_cut_budget: int | None = None,
    _initial_cuts: list[str] | None = None,
    cut_pool: list[str] | None = None,
) -> bool:
    """Greedy budget-aware dead-end test for an output combination.

    1. Apply *all* available cuts simultaneously and check whether the
       resulting minimum-region satisfies ``max_region_ops`` and is
       connected; if not the choice is irrescuable.
    2. If a ``max_cut_budget`` is given, run a greedy ``budget``-step
       cut selection (each step picks the single cut that reduces the
       exit-count the most while preserving connectivity / size limits)
       and verify the resulting region's exit-count fits within
       ``max_region_outputs``. The greedy strategy is a sufficient
       sub-routine: if greedy succeeds, the sampler can also succeed
       (greedy emits a valid concrete cut sequence).

    When ``cut_pool`` is provided it is used as the cut-candidate
    set instead of re-enumerating via :func:`enumerate_cut_candidates`.
    """
    if cut_pool is not None:
        cut_cands = list(cut_pool)
    else:
        from algorithm_ir.region.slicer import enumerate_cut_candidates
        try:
            cut_cands = enumerate_cut_candidates(ir, output_values)
        except Exception:
            return False
    initial_cuts = list(_initial_cuts) if _initial_cuts else []
    # Step 1 — global lower-bound check (all cuts applied)
    all_cuts = list(set(cut_cands) | set(initial_cuts))
    min_ops_set = backward_slice_until_values(ir, output_values, all_cuts)
    if not min_ops_set:
        return False
    if not is_op_set_connected(ir, min_ops_set):
        return False
    if max_region_ops is not None and _nontrivial_op_count(ir, min_ops_set) > max_region_ops:
        return False
    if (
        max_region_outputs is not None
        and len(_exit_values(ir, min_ops_set)) > max_region_outputs
    ):
        return False

    # Step 2 — greedy budget-aware reduction starting from ``initial_cuts``.
    if max_cut_budget is None or max_region_outputs is None:
        return True

    # ── Pre-compute op_closures-based upper bounds ────────────────
    # When ``op_closures`` is available, the union of all output-value
    # closures gives an over-estimate of the no-cut region.  Cuts only
    # ever *remove* ops, so exits and op-count are monotone-decreasing
    # while entries are monotone-increasing.  Upper bounds that already
    # satisfy the constraints let us short-circuit BFS in the greedy
    # lookahead loop.
    if op_closures is not None:
        base_union = set().union(*(op_closures.get(v, frozenset()) for v in output_values))
    else:
        base_union = None

    if base_union:
        _base_connected: bool | None = is_op_set_connected(ir, base_union)
        _base_exits_ub: set[str] | None = _exit_values(ir, base_union)
        _base_size_ub: int | None = _nontrivial_op_count(ir, base_union)
        _base_entries_lb: set[str] | None = _entry_values(ir, base_union)
    else:
        _base_connected = None
        _base_exits_ub = None
        _base_size_ub = None
        _base_entries_lb = None

    def _state_valid(cuts: list[str]) -> bool:
        ops = backward_slice_until_values(ir, output_values, cuts)
        if not ops or not is_op_set_connected(ir, ops):
            return False
        if len(_exit_values(ir, ops)) > max_region_outputs:
            return False
        if max_region_ops is not None and _nontrivial_op_count(ir, ops) > max_region_ops:
            return False
        if max_region_inputs is not None and len(_entry_values(ir, ops)) > max_region_inputs:
            return False
        return True

    cuts_used: list[str] = list(initial_cuts)
    cands = list(cut_cands)
    remaining_budget = max_cut_budget - len(initial_cuts)
    if remaining_budget < 0:
        return False
    for _ in range(remaining_budget):
        if _state_valid(cuts_used):
            return True
        # need more reduction; pick the cut that reduces deficit most
        cur_ops_local = backward_slice_until_values(ir, output_values, cuts_used)
        cur_exits_local = _exit_values(ir, cur_ops_local)
        cur_size_local = _nontrivial_op_count(ir, cur_ops_local)
        deficit_now = max(0, len(cur_exits_local) - max_region_outputs) + max(
            0, cur_size_local - (max_region_ops or cur_size_local)
        )
        # Fast-path flag: when the full-closure upper bounds already fit
        # within ALL constraints, we can skip per-candidate BFS because
        # every subset is also feasible on those dimensions.
        _fast_path = (
            _base_connected is True
            and _base_exits_ub is not None
            and len(_base_exits_ub) <= max_region_outputs
            and _base_size_ub is not None
            and (max_region_ops is None or _base_size_ub <= max_region_ops)
        )
        best_score = 0
        best_cut = None
        for c in cands:
            if c in cuts_used:
                continue
            if _fast_path:
                # Connectivity is guaranteed (subset of connected),
                # exit-count ≤ max_region_outputs is guaranteed, and
                # op-count ≤ max_region_ops is guaranteed.  We still
                # need per-cut metrics for scoring, so we do a BFS
                # but skip the post-BFS connectivity / dim checks.
                new_ops = backward_slice_until_values(ir, output_values, cuts_used + [c])
                if not new_ops:
                    continue
                # Connectivity guaranteed by _base_connected (no check).
                new_exits = _exit_values(ir, new_ops)
                new_size = _nontrivial_op_count(ir, new_ops)
            else:
                new_ops = backward_slice_until_values(ir, output_values, cuts_used + [c])
                if not new_ops or not is_op_set_connected(ir, new_ops):
                    continue
                new_exits = _exit_values(ir, new_ops)
                new_size = _nontrivial_op_count(ir, new_ops)
            new_def = max(0, len(new_exits) - max_region_outputs) + max(
                0, new_size - (max_region_ops or new_size)
            )
            score = deficit_now - new_def
            if score > best_score:
                best_score = score
                best_cut = c
        if best_cut is None:
            return False
        cuts_used.append(best_cut)
    return _state_valid(cuts_used)


# ---------------------------------------------------------------- Layer 4 (cut)


def cut_step_mask(
    ir: FunctionIR,
    output_values: list[str],
    selected_cuts: list[str],
    remaining: list[str],
    *,
    max_region_ops: int,
    min_region_ops: int,
    max_region_inputs: int,
    max_region_outputs: int,
    max_cut_budget: int | None = None,
    cut_pool: list[str] | None = None,
) -> tuple[list[bool], bool]:
    """Layer 4: which cut candidates are still legal + whether STOP
    would yield a valid region.

    For each candidate c:
        legal iff backward_slice_until_values(outputs, selected ∪ {c})
        is non-empty, connected, has |entry_values| <= max_region_inputs,
        and |exit_values| <= max_region_outputs.

        op-count is NOT checked — adding cuts is monotone-decreasing on
        op count, so it never *creates* a too_large violation.

    STOP is allowed iff the *current* region is fully valid in all four
    dimensions (size in [min, max], inputs <= max, exits <= max, connected).
    """
    cur_ops = backward_slice_until_values(ir, output_values, selected_cuts)
    cur_size = _nontrivial_op_count(ir, cur_ops)
    cur_inputs = _entry_values(ir, cur_ops)
    cur_exits = _exit_values(ir, cur_ops)
    cur_connected = is_op_set_connected(ir, cur_ops)

    # STOP gate
    stop_allowed = (
        len(cur_ops) > 0
        and cur_connected
        and min_region_ops <= cur_size <= max_region_ops
        and len(cur_inputs) <= max_region_inputs
        and len(cur_exits) <= max_region_outputs
    )

    # When the cut budget is tight, force per-step "future-feasibility":
    # we mask out any cut whose resulting state cannot subsequently be
    # legalized by the remaining budget (greedy lookahead).  This is
    # strictly stronger than per-step pace-matching: it preserves global
    # feasibility instead of local progress, so the sampler never paints
    # itself into a corner.
    if max_cut_budget is not None:
        budget_left = max_cut_budget - len(selected_cuts)
    else:
        budget_left = None
    # Always look ahead when a cut budget is set: even a state that is
    # currently within all dim limits can become unrecoverable if a
    # particular cut traps us (e.g. drops size below min_region_ops on
    # the next step or breaks connectivity along the only reduction
    # path). Per-cut feasibility check guards every transition.
    enforce_lookahead = budget_left is not None

    mask: list[bool] = []
    for c in remaining:
        if c in selected_cuts:
            mask.append(False)
            continue
        new_ops = backward_slice_until_values(ir, output_values, list(selected_cuts) + [c])
        if not new_ops:
            mask.append(False)
            continue
        if not is_op_set_connected(ir, new_ops):
            mask.append(False)
            continue
        new_size = _nontrivial_op_count(ir, new_ops)
        if new_size < min_region_ops:
            # cutting away too much would leave nothing
            mask.append(False)
            continue
        # NOTE: we deliberately do NOT reject cuts that leave
        # ``new_inputs > max_region_inputs`` or
        # ``new_exits > max_region_outputs``: a single cut may need a
        # follow-up cut to legalize. The greedy lookahead below is the
        # authoritative feasibility test (it accepts the cut iff some
        # ≤ budget_left cut sequence containing it yields a valid
        # region). max_region_ops never violated by adding cuts
        # (monotone ≤), so no upper-bound check needed.

        # Per-cut look-ahead feasibility from post-cut state with
        # budget_left-1 cuts remaining.
        if enforce_lookahead:
            if not _combo_is_feasible(
                ir, list(output_values),
                None,
                max_region_ops=max_region_ops,
                max_region_outputs=max_region_outputs,
                max_region_inputs=max_region_inputs,
                max_cut_budget=max_cut_budget,
                _initial_cuts=list(selected_cuts) + [c],
                cut_pool=cut_pool,
            ):
                mask.append(False)
                continue
        else:
            # No budget => fall back to immediate dim ceilings.
            new_inputs = _entry_values(ir, new_ops)
            if len(new_inputs) > max_region_inputs:
                mask.append(False)
                continue
            new_exits = _exit_values(ir, new_ops)
            if len(new_exits) > max_region_outputs:
                mask.append(False)
                continue
        mask.append(True)

    return mask, stop_allowed


# ---------------------------------------------------------------- feasibility


def is_output_combo_feasible(
    ir: FunctionIR,
    output_values: list[str],
    *,
    max_region_ops: int,
    max_region_outputs: int,
    max_region_inputs: int | None = None,
    max_cut_budget: int | None = None,
    cut_pool: list[str] | None = None,
) -> bool:
    """Public wrapper for :func:`_combo_is_feasible` (callers outside
    this module).
    """
    return _combo_is_feasible(
        ir, output_values,
        max_region_ops=max_region_ops,
        max_region_outputs=max_region_outputs,
        max_region_inputs=max_region_inputs,
        max_cut_budget=max_cut_budget,
        cut_pool=cut_pool,
    )
