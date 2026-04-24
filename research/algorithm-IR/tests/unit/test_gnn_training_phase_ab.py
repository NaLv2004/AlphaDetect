from __future__ import annotations

import warnings
import numpy as np
import pytest
import torch

from algorithm_ir.ir.model import Block, FunctionIR, Op, Value
from algorithm_ir.regeneration.codegen import emit_python_source
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


def _proposal_signature(proposal):
    return {
        "host_algo_id": proposal.host_algo_id,
        "donor_algo_id": proposal.donor_algo_id,
        "region_op_ids": tuple(proposal.region.op_ids),
        "region_entry_values": tuple(proposal.region.entry_values),
        "region_exit_values": tuple(proposal.region.exit_values),
        "donor_source": emit_python_source(proposal.donor_ir),
        "confidence": proposal.confidence,
        "rationale": proposal.rationale,
    }


def test_gnn_matcher_batched_proposals_match_unbatched_with_same_seed():
    genomes = build_ir_pool(np.random.default_rng(123))[:5]
    entries = [g.to_entry(None) for g in genomes]

    torch.manual_seed(7)
    np.random.seed(7)
    matcher_ref = GNNPatternMatcher(
        max_proposals_per_gen=50,
        warmstart_generations=1,
        train_interval=10,
        enable_batched_proposals=False,
        proposal_batch_size=1,
    )

    matcher_batch = GNNPatternMatcher(
        max_proposals_per_gen=50,
        warmstart_generations=1,
        train_interval=10,
        enable_batched_proposals=True,
        proposal_batch_size=8,
    )
    matcher_batch.encoder.load_state_dict(matcher_ref.encoder.state_dict())
    matcher_batch.scorer.load_state_dict(matcher_ref.scorer.state_dict())
    matcher_batch.region_proposer.load_state_dict(matcher_ref.region_proposer.state_dict())
    matcher_batch.donor_region_selector.load_state_dict(
        matcher_ref.donor_region_selector.state_dict()
    )

    torch.manual_seed(99)
    np.random.seed(99)
    ref_props = matcher_ref(entries, generation=1)
    torch.manual_seed(99)
    np.random.seed(99)
    batched_props = matcher_batch(entries, generation=1)

    assert len(ref_props) == len(batched_props)
    ref_sig = [_proposal_signature(p) for p in ref_props]
    batched_sig = [_proposal_signature(p) for p in batched_props]
    assert ref_sig == batched_sig


def test_precise_evaluation_accumulates_errors_or_hits_cap():
    genome = build_ir_pool(np.random.default_rng(2))[0]
    evaluator = MIMOFitnessEvaluator(MIMOEvalConfig(
        Nr=4,
        Nt=4,
        mod_order=4,
        snr_db_list=[12.0],
        n_trials=1,
        timeout_sec=0.5,
        complexity_weight=0.01,
        seed=9,
    ))
    fit = evaluator.evaluate_precise(
        genome,
        target_errors=4,
        max_symbols=64,
        timeout_sec=3.0,
    )
    assert fit.is_valid
    assert fit.metrics["symbols_total"] > 0
    assert (
        fit.metrics["symbol_errors"] >= 4
        or fit.metrics["symbols_total"] >= 64
    )


def test_codegen_avoids_syntaxwarning_for_non_simple_call_targets():
    values = {
        "v_const": Value(id="v_const", def_op="op_const"),
        "v_ret": Value(id="v_ret", def_op="op_ret_const"),
    }
    ops = {
        "op_const": Op(
            id="op_const",
            opcode="const",
            inputs=[],
            outputs=["v_const"],
            block_id="b0",
            attrs={"literal": 3},
        ),
        "op_call": Op(
            id="op_call",
            opcode="call",
            inputs=["v_const"],
            outputs=[],
            block_id="b0",
            attrs={"n_args": 0, "kwarg_names": []},
        ),
        "op_ret_const": Op(
            id="op_ret_const",
            opcode="const",
            inputs=[],
            outputs=["v_ret"],
            block_id="b0",
            attrs={"literal": 0},
        ),
        "op_return": Op(
            id="op_return",
            opcode="return",
            inputs=["v_ret"],
            outputs=[],
            block_id="b0",
        ),
    }
    block = Block(id="b0", op_ids=["op_const", "op_call", "op_ret_const", "op_return"])
    ir = FunctionIR(
        id="fn:test",
        name="warn_free",
        arg_values=[],
        return_values=["v_ret"],
        values=values,
        ops=ops,
        blocks={"b0": block},
        entry_block="b0",
    )
    source = emit_python_source(ir)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", SyntaxWarning)
        compile(source, "<warn_free>", "exec")
    assert not [w for w in record if issubclass(w.category, SyntaxWarning)]
    assert "__call_target_op_call = 3" in source


def test_codegen_avoids_syntaxwarning_for_non_simple_subscript_targets():
    values = {
        "v_const": Value(id="v_const", def_op="op_const"),
        "v_idx": Value(id="v_idx", def_op="op_idx"),
        "v_item": Value(id="v_item", def_op="op_getitem"),
    }
    ops = {
        "op_const": Op(
            id="op_const",
            opcode="const",
            inputs=[],
            outputs=["v_const"],
            block_id="b0",
            attrs={"literal": 3},
        ),
        "op_idx": Op(
            id="op_idx",
            opcode="const",
            inputs=[],
            outputs=["v_idx"],
            block_id="b0",
            attrs={"literal": 0},
        ),
        "op_getitem": Op(
            id="op_getitem",
            opcode="get_item",
            inputs=["v_const", "v_idx"],
            outputs=["v_item"],
            block_id="b0",
        ),
        "op_return": Op(
            id="op_return",
            opcode="return",
            inputs=["v_item"],
            outputs=[],
            block_id="b0",
        ),
    }
    block = Block(id="b0", op_ids=["op_const", "op_idx", "op_getitem", "op_return"])
    ir = FunctionIR(
        id="fn:test_item",
        name="warn_free_item",
        arg_values=[],
        return_values=["v_item"],
        values=values,
        ops=ops,
        blocks={"b0": block},
        entry_block="b0",
    )
    source = emit_python_source(ir)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", SyntaxWarning)
        compile(source, "<warn_free_item>", "exec")
    assert not [w for w in record if issubclass(w.category, SyntaxWarning)]
    assert "__getitem_target_op_getitem = 3" in source
