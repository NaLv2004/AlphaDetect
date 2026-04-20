"""Debug donor trimming."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from evolution.ir_pool import build_ir_pool
from evolution.pattern_matchers import ExpertPatternMatcher, StaticStructurePatternMatcher, CompositePatternMatcher
from evolution.gnn_pattern_matcher import GNNPatternMatcher
from evolution.pool_types import AlgorithmEvolutionConfig
from evolution.mimo_evaluator import MIMOFitnessEvaluator, MIMOEvalConfig
from evolution.algorithm_engine import AlgorithmEvolutionEngine

rng = np.random.default_rng(42)
pool = build_ir_pool(rng)
print(f"Pool: {len(pool)} genomes")

entries = [g.to_entry(None) for g in pool[:48]]

expert = ExpertPatternMatcher(max_proposals_per_gen=3)
static = StaticStructurePatternMatcher(max_proposals_per_gen=3)
gnn = GNNPatternMatcher(max_proposals_per_gen=4, top_k_pairs=8)
comp = CompositePatternMatcher([expert, static, gnn])

proposals = comp(entries, 1)
print(f"Proposals: {len(proposals)}")
for p in proposals:
    donor_id = p.donor_algo_id[:12] if p.donor_algo_id else "?"
    n_donor = len(p.donor_ir.ops) if p.donor_ir else 0
    print(f"  {p.proposal_id}: host={p.host_algo_id[:12]}, donor={donor_id}, "
          f"region_ops={len(p.region.op_ids)}, donor_ops={n_donor}")

# Create a real engine instance for trimming test
eval_config = MIMOEvalConfig(Nr=4, Nt=4, mod_order=4, snr_db_list=[10.0],
                              n_trials=2, timeout_sec=1.0)
evaluator = MIMOFitnessEvaluator(eval_config)
evo_config = AlgorithmEvolutionConfig(pool_size=4, n_generations=1)
engine = AlgorithmEvolutionEngine(evaluator, evo_config, rng, pattern_matcher=comp)

# Test trimming
for p in proposals:
    try:
        trimmed = engine._trim_donor_for_proposal(p)
        t_ir = trimmed.donor_ir
        print(f"  -> trimmed donor_ops={len(t_ir.ops)}, args={len(t_ir.arg_values)}, "
              f"ret={len(t_ir.return_values)}, blocks={len(t_ir.blocks)}")
    except Exception as e:
        import traceback
        print(f"  -> TRIM FAILED: {e}")
        traceback.print_exc()

# Try executing a graft with trimmed proposal
print("\n--- Testing graft execution ---")
from algorithm_ir.grafting.graft_general import graft_general
for p in proposals[:3]:
    try:
        trimmed = engine._trim_donor_for_proposal(p)
        host_genome = None
        for g in pool:
            if g.algo_id == p.host_algo_id:
                host_genome = g
                break
        if host_genome is None:
            print(f"  Host {p.host_algo_id} not found")
            continue
        artifact = graft_general(host_genome.structural_ir, trimmed)
        print(f"  GRAFT OK: {len(artifact.inlined_op_ids)} inlined, "
              f"{len(artifact.replaced_op_ids)} replaced, "
              f"{len(artifact.new_slot_ids)} new slots")
    except Exception as e:
        import traceback
        print(f"  GRAFT FAILED: {e}")
        traceback.print_exc()
