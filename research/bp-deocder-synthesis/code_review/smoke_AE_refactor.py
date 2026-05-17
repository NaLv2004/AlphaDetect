"""End-to-end CPP smoke for the (A, E) refactor.

Exercises the full stack:
  FitnessConfig(A=176, E=352) -> derive_params (BG2 set1 Zc=32 N=1664)
  -> _channel_inputs (encode_codeblock + cbs_rate_match + bpsk + awgn + bpsk_llr + cbs_rate_recover)
  -> evaluate_genome_cpp_ber (BER over info bits A only)
  -> uncoded baselines

Expected, post-fix (LOG_CONST_MAX=6 → EvoConst1=1e6 sentinel):
  OMS in waterfall (2-4 dB) should crush R=1 uncoded.
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pushgp_ldpc.eval import FitnessConfig, physical_code_rate, info_bits_count
from pushgp_ldpc.eval_cpp import evaluate_genome_cpp_ber
from pushgp_ldpc.adapter import oms_seed_genome
from pushgp_ldpc.baselines import uncoded_rate1_baseline, channel_hard_baseline

snrs = (1.0, 2.0, 3.0, 4.0, 5.0)
A_E_grid = [
    (176, 352),   # short, R=0.5
    (512, 1024),  # the simulate.py reference config
]

for A, E in A_E_grid:
    cfg = FitnessConfig(info_len_A=A, code_length_E=E,
                        snr_list=snrs, n_frames_per_snr=100, max_iter=20,
                        seed_base=20260601)
    par = cfg.par
    print(f"\n=== A={A} E={E}  → BG{par.bgn} set{par.set_idx} Zc={par.zc} "
          f"N={par.cols} K_cb_bit={cfg.K_cb_bit} R={cfg.effective_code_rate:.4f} ===",
          flush=True)

    t0 = time.time()
    r1 = uncoded_rate1_baseline(cfg, bits_per_snr=cfg.K_cb_bit * 100)
    ch = channel_hard_baseline(cfg)
    print(f"baselines: {time.time()-t0:.1f}s", flush=True)

    oms = oms_seed_genome()
    t1 = time.time()
    m = evaluate_genome_cpp_ber(oms, cfg)
    dt = time.time() - t1
    print(f"OMS cpp eval: {dt:.1f}s  fitness={m.fitness:.3f}  iters_avg≈?", flush=True)

    print(f"  SNR | info-BER OMS | ch-hard | R=1 uncoded")
    print("  " + "-"*55)
    for i, s in enumerate(snrs):
        print(f"  {s:>4.1f} | {m.ber_per_snr[i]:>12.3e} | {ch['ber_per_snr'][i]:>9.3e} | "
              f"{r1['ber_per_snr'][i]:>12.3e}", flush=True)

print("\nDONE.")
