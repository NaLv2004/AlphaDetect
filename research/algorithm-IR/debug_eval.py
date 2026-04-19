import numpy as np
from evolution.ir_pool import build_ir_pool
from evolution.mimo_evaluator import MIMOEvalConfig, MIMOFitnessEvaluator

eval_cfg = MIMOEvalConfig(Nr=16, Nt=16, mod_order=16, snr_db_list=[24.0, 26.0], n_trials=10, timeout_sec=8.0, complexity_weight=0.001, seed=42)
evaluator = MIMOFitnessEvaluator(eval_cfg)
pool = build_ir_pool()
for g in pool:
    res = evaluator.evaluate(g)
    s24 = res.metrics.get("ser_24dB", "N/A")
    print(f"{g.algo_id:10s}: score={res.composite_score():.4f}, SER24={s24}, time={res.metrics.get('eval_time',0):.1f}s, valid={res.is_valid}")
