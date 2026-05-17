"""Cross-check: run the pushgp OMS adapter (via eval_cpp.py) on the
SAME standard 5G LDPC config that simulate.py just validated:
BG2 set5, Zc=72  →  N=3744, M=3024, K_info=720 (Kb=10).

simulate.py at Eb/N0=2 dB reached BER ~2.7e-3.  If the pushgp path is
implemented correctly, it should reproduce ballpark the same number.
If it instead lands at 5e-2 like our BG1 Zc=8 test did, then the
pushgp adapter has a real bug.  This is the cleanest possible apples-
to-apples check.
"""
from __future__ import annotations

import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ldpc_5g import build_parity
from pushgp_ldpc.eval import FitnessConfig, _channel_inputs, info_bits_count, physical_code_rate
from pushgp_ldpc.eval_cpp import _get_cdce, _get_parity_handle
from pushgp.serialize import program_to_dict
from pushgp_ldpc.adapter import oms_seed_genome


par = build_parity(bgn=2, set_idx=5, zc=72)
K = info_bits_count(par)
print(f"BG2 set5 Zc=72: N={par.cols} M={par.rows} K_info={K}  "
      f"base R={physical_code_rate(par):.4f}")

# Target R = 0.5 → tx_len = ceil(720/0.5) = 1440.  This matches sim's "outlen=1024"
# only loosely because sim uses A=512 (filler-padded to K=720); we use the full
# K=720 as info to keep apples-to-apples for the BP performance check.
snrs = (1.0, 2.0, 3.0, 4.0)
N_FRAMES = 100   # smaller because each frame is 7x larger than before
MAX_ITER = 20

oms = oms_seed_genome()
cdce = _get_cdce()
parH = _get_parity_handle(par)
v_dict = program_to_dict(oms.prog_v2c)
c_dict = program_to_dict(oms.prog_c2v)
evo = oms.evo_const_values().astype(np.float64)

print(f"\nSweep: {len(snrs)} pts, {N_FRAMES} frames each, max_iter={MAX_ITER}, "
      f"target_R=0.5", flush=True)
print(f"\n  SNR | info-BER pushgp |  cw-BER  | sec")
print("-" * 50)

for snr in snrs:
    cfg = FitnessConfig(par=par, snr_list=(snr,), n_frames_per_snr=N_FRAMES,
                        max_iter=MAX_ITER, target_code_rate=0.5,
                        seed_base=20260601)
    pairs = _channel_inputs(cfg, snr)
    n_err_info = n_err_cw = 0
    n_info = n_cw = 0
    t1 = time.time()
    for cw, llr in pairs:
        post, _it = cdce.decode_bp(llr, parH, v_dict, c_dict, evo,
                                    MAX_ITER, 0.25, cfg.effective_code_rate)
        hat = (post < 0.0).astype(np.int8)
        n_err_info += int((hat[:K] != cw[:K]).sum())
        n_info += K
        n_err_cw += int((hat != cw).sum())
        n_cw += cw.size
    dt = time.time() - t1
    print(f"{snr:>5.1f} | {n_err_info/max(1,n_info):>15.3e} | "
          f"{n_err_cw/max(1,n_cw):>8.3e} | {dt:.1f}", flush=True)

print("\nDONE.")
