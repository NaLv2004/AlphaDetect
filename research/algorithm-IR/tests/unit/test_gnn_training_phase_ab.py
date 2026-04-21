from __future__ import annotations

import numpy as np
import pytest

from evolution.gnn_pattern_matcher import GNNPatternMatcher
from evolution.ir_pool import build_ir_pool
from evolution.materialize import materialize_to_callable
from evolution.mimo_evaluator import MIMOEvalConfig, MIMOFitnessEvaluator


def test_materialize_to_callable_uses_cache():
    genome = build_ir_pool(np.random.default_rng(0))[0]
    fn1 = materialize_to_callable(genome)
    fn2 = materialize_to_callable(genome)
    assert fn1 is fn2


def test_mimo_evaluate_batch_parallel_matches_sequential():
    genomes = build_ir_pool(np.random.default_rng(1))[:2]
    cfg = dict(
        Nr=4,
        Nt=4,
        mod_order=4,
        snr_db_list=[15.0],
        n_trials=1,
        timeout_sec=0.3,
        complexity_weight=0.01,
        seed=7,
    )
    seq_eval = MIMOFitnessEvaluator(MIMOEvalConfig(**cfg, batch_workers=1))
    par_eval = MIMOFitnessEvaluator(MIMOEvalConfig(**cfg, batch_workers=2))

    seq_results = seq_eval.evaluate_batch(genomes)
    par_results = par_eval.evaluate_batch(genomes)

    assert len(seq_results) == len(par_results) == 2
    for seq_fit, par_fit in zip(seq_results, par_results):
        assert seq_fit.is_valid == par_fit.is_valid
        assert seq_fit.metrics["ser"] == pytest.approx(par_fit.metrics["ser"])
        assert seq_fit.composite_score() == pytest.approx(par_fit.composite_score())


def test_gnn_matcher_warmstart_uses_all_pairs_then_sampling():
    matcher = GNNPatternMatcher(
        max_proposals_per_gen=2,
        warmstart_generations=1,
        pair_temperature=0.7,
        pair_exploration=0.0,
    )
    pair_scores = [
        ("h0", "d0", 0.4),
        ("h0", "d1", 0.3),
        ("h1", "d0", 0.2),
        ("h1", "d1", 0.1),
        ("h2", "d0", 0.5),
        ("h2", "d1", 0.6),
    ]

    matcher._generation = 1
    warmstart_pairs = matcher._select_pair_candidates(pair_scores)  # type: ignore[arg-type]
    assert len(warmstart_pairs) == len(pair_scores)

    matcher._generation = 2
    sampled_pairs = matcher._select_pair_candidates(pair_scores)  # type: ignore[arg-type]
    assert len(sampled_pairs) == 2
