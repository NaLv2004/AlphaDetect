"""Diagnose why kbest/stack/bp/amp/ep are broken at 16×16 16QAM."""
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
print(f"sigma2 = {sigma2:.6f}")
print(f"x_true[:4] = {x_true[:4]}")
print(f"H shape = {H.shape}, H dtype = {H.dtype}")
print(f"y shape = {y.shape}, y dtype = {y.dtype}")

pool = build_ir_pool()
for g in pool:
    name = g.algo_id
    try:
        fn = materialize_to_callable(g)
        x_hat = fn(H, y, sigma2, constellation)
        if x_hat is None:
            print(f"{name:10s}: returned None")
            continue
        x_hat_arr = np.array(x_hat)
        ser = np.mean(np.abs(x_true - x_hat_arr) > 1e-6)
        print(f"{name:10s}: SER={ser:.4f}  x_hat[:4]={x_hat_arr[:4]}")
        
        # Extra diagnostics for broken detectors
        if ser > 0.5 and name not in ('lmmse', 'zf'):
            print(f"  >> x_hat all same? unique={len(np.unique(np.round(x_hat_arr, 4)))}")
            print(f"  >> x_hat range: min={np.min(np.abs(x_hat_arr)):.4f} max={np.max(np.abs(x_hat_arr)):.4f}")
            print(f"  >> x_hat mean={np.mean(x_hat_arr):.4f}")
    except Exception as e:
        import traceback
        print(f"{name:10s}: EXCEPTION")
        traceback.print_exc()
        print()
