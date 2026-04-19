"""Check why final eval gives SER=1.0 for best genome."""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))
import numpy as np
from evolution.ir_pool import build_ir_pool
from evolution.materialize import materialize_to_callable
from evolution.mimo_evaluator import qam16_constellation, generate_mimo_sample, _nearest_symbols

pool = build_ir_pool()
lmmse = next(g for g in pool if g.algo_id == "lmmse")
constellation = qam16_constellation()
rng = np.random.default_rng(123)

fn = materialize_to_callable(lmmse)
errors = 0; total = 0
for _ in range(10):
    H, x_true, y, sigma2 = generate_mimo_sample(16, 16, constellation, 24.0, rng)
    try:
        x_hat = fn(H, y, sigma2, constellation)
        if x_hat is None or len(x_hat) != 16:
            errors += 16
            print(f"  Trial {_}: x_hat={x_hat}")
        else:
            x_hat = _nearest_symbols(x_hat, constellation)
            errs = int(np.sum(np.abs(x_true - x_hat) > 1e-6))
            errors += errs
    except Exception as e:
        import traceback
        traceback.print_exc()
        errors += 16
    total += 16
print(f"LMMSE from pool: SER={errors/total:.4f} over {total//16} trials")
