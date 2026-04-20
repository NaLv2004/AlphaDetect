"""Quick diagnostic: test evaluation on a single genome."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Building pool...")
from evolution.ir_pool import build_ir_pool
import numpy as np
pool = build_ir_pool(np.random.default_rng(42))
print(f"Pool size: {len(pool)}")

print("Testing evaluation...")
from evolution.mimo_evaluator import MIMOFitnessEvaluator, MIMOEvalConfig

eval_cfg = MIMOEvalConfig(Nr=16, Nt=16, mod_order=16, snr_db_list=[24.0], n_trials=50, timeout_sec=10.0)
evaluator = MIMOFitnessEvaluator(eval_cfg)

for g in pool[:3]:
    t0 = time.perf_counter()
    result = evaluator.evaluate(g)
    elapsed = time.perf_counter() - t0
    print(f"{g.algo_id}: score={result.composite_score():.6f} metrics={result.metrics} time={elapsed:.1f}s")
print("Done!")
