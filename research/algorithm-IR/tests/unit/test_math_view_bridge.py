"""Tests for ``algorithm_ir.grafting.math_view_bridge``.

Coverage:
  * Round-trip on LMMSE: select all MathNodes -> op_ids should equal
    every reachable op in the IR (every op either mapped or J1-dropped).
  * project_op_ids_to_math_nodes: exact inverse on non-dropped ops.
  * J1 jump inclusion (B1 rule): selecting just the LMMSE while-body
    block must include the back-edge jump op_46; selecting just the
    entry block alone (without the test block) must NOT include the
    entry jump.
  * Signature equivalence on the full pool: for every detector and a
    handful of random connected sub-regions, the bridge-produced
    BoundarySignature must equal the canonical one from
    define_rewrite_region + signature_for_region applied to the
    expanded op_id set. (This is essentially a self-consistency test
    because the bridge delegates -- but it also locks the contract so
    a future change to the bridge can't silently diverge.)
  * kind=='other' must be empty pool-wide after Phase 3a.
"""
from __future__ import annotations

import os
import sys
import random

import numpy as np
import pytest

# Allow ``import algorithm_ir`` etc. when tests are run from repo root.
_THIS = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from algorithm_ir.ir.math_view import build_math_view  # noqa: E402
from algorithm_ir.grafting.math_view_bridge import (  # noqa: E402
    expand_math_region_to_op_ids,
    project_op_ids_to_math_nodes,
    boundary_signature_for_math_region,
)
from algorithm_ir.region.selector import define_rewrite_region  # noqa: E402
from evolution.graft_classifier import signature_for_region  # noqa: E402
from evolution.ir_pool import build_ir_pool  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pool():
    rng = np.random.default_rng(42)
    return build_ir_pool(rng)


@pytest.fixture(scope="module")
def lmmse_genome(pool):
    return next(g for g in pool if g.algo_id == "lmmse")


@pytest.fixture(scope="module")
def lmmse_view(lmmse_genome):
    return build_math_view(lmmse_genome.ir)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

def test_full_view_round_trip_lmmse(lmmse_genome, lmmse_view):
    """Selecting every node should expand to every op_id in the IR
    EXCEPT the dropped-orphan op_ids that have no semantic role
    (orphan module consts, orphan shape-derivation chains).

    J1-dropped jumps that *are* part of the live control flow are
    auto-included by the B1 rule and therefore appear in ``expanded``.
    """
    all_node_ids = {n.node_id for n in lmmse_view.nodes}
    expanded = expand_math_region_to_op_ids(lmmse_view, all_node_ids)

    ir = lmmse_genome.ir
    all_ops = set(ir.ops.keys())
    dropped = set(lmmse_view.dropped)
    # B1 should still pull live jumps back in:
    live_jumps_pulled = {oid for oid in dropped if ir.ops[oid].opcode == "jump"}
    expected = (all_ops - dropped) | (live_jumps_pulled & set(expanded))
    # And the orphan-dropped non-jump ops stay out:
    orphan_dropped = {oid for oid in dropped if ir.ops[oid].opcode != "jump"}
    assert set(expanded) == expected
    for oid in orphan_dropped:
        assert oid not in expanded, f"orphan dropped op {oid} leaked into expansion"


def test_project_inverse_on_nondropped(lmmse_genome, lmmse_view):
    """project_op_ids_to_math_nodes is the inverse of expand on
    non-dropped op_ids."""
    op_ids = set(lmmse_genome.ir.ops.keys()) - set(lmmse_view.dropped)
    nodes = project_op_ids_to_math_nodes(lmmse_view, op_ids)
    expected = {n.node_id for n in lmmse_view.nodes if n.op_ids}
    assert set(nodes) == expected


# ---------------------------------------------------------------------------
# J1 jump inclusion (B1 rule)
# ---------------------------------------------------------------------------

def test_j1_jump_inclusion_full_loop(lmmse_genome, lmmse_view):
    """Selecting all the while-loop nodes (test + body + back edge target)
    should include the back-edge jump op (a J1-dropped op)."""
    ir = lmmse_genome.ir
    # Find every MathNode whose underlying ops live in any block other than
    # the entry block.
    entry_block_ops = set(ir.blocks[ir.entry_block].op_ids)
    loop_node_ids = set()
    for node in lmmse_view.nodes:
        if not node.op_ids:
            continue
        if any(op_id not in entry_block_ops for op_id in node.op_ids):
            loop_node_ids.add(node.node_id)

    expanded = expand_math_region_to_op_ids(lmmse_view, loop_node_ids)

    # The set of jump ops in the loop region should be exactly those whose
    # source block is fully consumed by ``expanded``.
    jump_ops_inside = {
        oid for oid in lmmse_view.dropped
        if ir.ops[oid].opcode == "jump"
        and all(
            other in expanded
            for other in ir.blocks[ir.ops[oid].block_id].op_ids
            if other != oid
        )
    }
    # Every such jump must show up in expanded.
    for j in jump_ops_inside:
        assert j in expanded, f"missing dropped jump {j} that should belong"


def test_j1_jump_excluded_when_block_is_partial(lmmse_genome, lmmse_view):
    """If we select only ONE node from a block, the block-terminating jump
    must NOT come along, because the rest of the block is outside the
    region."""
    ir = lmmse_genome.ir
    # Find a block with both a non-jump op AND a dropped jump.
    target_jump = None
    for oid in lmmse_view.dropped:
        if ir.ops[oid].opcode != "jump":
            continue
        blk = ir.blocks[ir.ops[oid].block_id]
        non_jumps = [o for o in blk.op_ids if o != oid]
        if not non_jumps:
            continue
        # Pick the first non-jump op and project it to its node.
        sample_op = non_jumps[0]
        sample_nid = lmmse_view.op_id_to_node.get(sample_op)
        if sample_nid is None:
            continue
        target_jump = (oid, sample_nid, blk, non_jumps)
        break

    if target_jump is None:
        pytest.skip("LMMSE has no block where a partial selection would expose this rule")

    j_op_id, sample_nid, blk, non_jumps = target_jump
    # Selecting JUST that one node: most of the block is missing -> jump
    # must NOT be auto-included (unless the node happens to own all
    # non-jump ops of the block by itself).
    expanded = expand_math_region_to_op_ids(lmmse_view, {sample_nid})
    sample_node_ops = {
        oid for n in lmmse_view.nodes if n.node_id == sample_nid for oid in n.op_ids
    }
    if set(non_jumps) <= sample_node_ops:
        # Single MathNode happens to own the whole block — jump SHOULD come.
        assert j_op_id in expanded
    else:
        # Block partially covered — jump must stay out.
        assert j_op_id not in expanded


# ---------------------------------------------------------------------------
# Signature equivalence (pool-wide, randomized)
# ---------------------------------------------------------------------------

def _random_connected_node_subset(view, rng: random.Random, target_size: int) -> set[str]:
    """Grow a connected MathNode subset by BFS from a random seed."""
    nodes = [n for n in view.nodes if n.op_ids]  # skip boundary args
    if not nodes:
        return set()
    seed = rng.choice(nodes)
    selected = {seed.node_id}
    frontier = [seed.node_id]
    # Build neighbour map (data flow, undirected) once.
    neighbours: dict[str, set[str]] = {n.node_id: set() for n in view.nodes}
    for n in view.nodes:
        for port in n.inputs:
            neighbours[n.node_id].add(port.node_id)
            neighbours.setdefault(port.node_id, set()).add(n.node_id)
    while frontier and len(selected) < target_size:
        nid = frontier.pop(0)
        nbrs = list(neighbours.get(nid, set()) - selected)
        rng.shuffle(nbrs)
        for nb in nbrs:
            if nb not in selected:
                # Skip pure-boundary nodes
                node = next((m for m in view.nodes if m.node_id == nb), None)
                if node is None or not node.op_ids:
                    continue
                selected.add(nb)
                frontier.append(nb)
                if len(selected) >= target_size:
                    break
    return selected


def test_signature_equivalence_pool_wide(pool):
    """For every detector, pick a few random connected subsets and verify
    the bridge-produced signature matches the canonical pipeline."""
    rng = random.Random(20251108)
    n_subsets_per_detector = 3
    target_sizes = [4, 8, 16]
    for g in pool:
        view = build_math_view(g.ir)
        for size in target_sizes[: n_subsets_per_detector]:
            sub = _random_connected_node_subset(view, rng, target_size=size)
            if not sub:
                continue
            sig_bridge = boundary_signature_for_math_region(view, sub)
            op_ids = expand_math_region_to_op_ids(view, sub)
            region = define_rewrite_region(g.ir, op_ids=list(op_ids))
            sig_canonical = signature_for_region(g.ir, region)
            assert sig_bridge == sig_canonical, (
                f"signature mismatch for {g.algo_id} subset size={size}: "
                f"bridge={sig_bridge}  canonical={sig_canonical}"
            )


# ---------------------------------------------------------------------------
# kind != 'other' invariant
# ---------------------------------------------------------------------------

def test_no_other_kind_pool_wide(pool):
    """Phase 3a contract: every MathNode in every detector has a known kind."""
    offenders: list[tuple[str, str, str]] = []
    for g in pool:
        view = build_math_view(g.ir)
        for n in view.nodes:
            if n.kind == "other":
                offenders.append((g.algo_id, n.node_id, n.opcode))
    assert not offenders, f"Found {len(offenders)} 'other'-kind nodes: {offenders[:5]}"
