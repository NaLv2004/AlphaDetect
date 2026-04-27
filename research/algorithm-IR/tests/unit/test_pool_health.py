"""Phase H+2 hard gate: every genome admitted by ``build_ir_pool`` must
satisfy ``validate_function_ir``. Stale def/use references and duplicated
uses corrupt every downstream consumer (region selection, GNN graphs,
graft surgery, materialization), so the gate is non-negotiable.
"""
from __future__ import annotations

import numpy as np

from algorithm_ir.ir.validator import validate_function_ir
from evolution.ir_pool import build_ir_pool, _POOL_REJECTIONS


def test_every_admitted_genome_validates():
    pool = build_ir_pool(np.random.default_rng(42))
    assert len(pool) > 0, "build_ir_pool produced no genomes"
    bad = []
    for g in pool:
        errs = validate_function_ir(g.ir)
        if errs:
            bad.append((g.algo_id, len(errs), errs[:3]))
    assert not bad, f"{len(bad)} admitted genomes are invalid: {bad[:5]}"


def test_pool_rejection_manifest_recorded():
    # Manifest is reset per call; after a clean build it should be empty.
    pool = build_ir_pool(np.random.default_rng(42))
    assert isinstance(_POOL_REJECTIONS, list)
    # Every entry has the expected schema (no entries on the happy path).
    for entry in _POOL_REJECTIONS:
        assert {"algo_id", "n_errors", "first_errors"} <= set(entry.keys())
    # Sanity: pool is non-empty and all entries are AlgorithmGenome-shaped.
    assert all(hasattr(g, "ir") and hasattr(g, "algo_id") for g in pool)


def test_pool_size_matches_expected_baseline():
    """Lock current pool size so accidental rejection regressions are loud."""
    pool = build_ir_pool(np.random.default_rng(42))
    # M2 (annotation-only slots): 8 core L3 detectors, all flatten to a valid IR.
    # The legacy ~80 skeleton-library extended specs were dropped.
    assert len(pool) == 8, (
        f"pool size changed from baseline 8 to {len(pool)}; "
        f"rejections: {_POOL_REJECTIONS[:3]}"
    )
