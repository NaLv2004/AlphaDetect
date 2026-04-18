"""Debug EP materialization."""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from evolution.ir_pool import build_ir_pool
from evolution.materialize import materialize_to_callable
from evolution.mimo_evaluator import qam16_constellation, generate_mimo_sample

constellation = qam16_constellation()
rng = np.random.default_rng(42)
Nr, Nt = 16, 16
snr_db = 24.0
H, x_true, y, sigma2 = generate_mimo_sample(Nr, Nt, constellation, snr_db, rng)

pool = build_ir_pool()
for g in pool:
    if g.algo_id != "ep":
        continue
    
    # Print materialized source
    from evolution.materialize import _materialize_source_with_override
    source = _materialize_source_with_override(g, {})
    print("=== EP materialized source ===")
    for i, line in enumerate(source.splitlines(), 1):
        print(f"{i:3d}: {line}")
    print("=== END ===\n")
    
    fn = materialize_to_callable(g)
    x_hat = fn(H, y, sigma2, constellation)
    x_hat_arr = np.array(x_hat)
    ser = np.mean(np.abs(x_true - x_hat_arr) > 1e-6)
    print(f"EP SER = {ser:.4f}")
    print(f"x_hat[:4] = {x_hat_arr[:4]}")
    print(f"x_true[:4] = {x_true[:4]}")
