"""Test R=0.5 via rate matching on BG1 set1 Zc=8.

BG1 set1 base graph: Kb=22, Nb=68, Mb=46.  With Zc=8:
  K_info = 22*8 = 176
  N_full = 68*8 - 2*8 = 528 (= max tx_len without parity puncturing)
  Base physical rate = 176/528 = 0.333

Target R = 0.5  →  tx_len = ceil(176/0.5) = 352  →  puncture 528-352=176
parity bits past position 352.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pushgp_ldpc.eval import FitnessConfig, physical_code_rate, tx_length, info_bits_count
from pushgp_ldpc.eval_cpp import evaluate_genome_cpp_ber
from pushgp_ldpc.adapter import oms_seed_genome
from pushgp_ldpc.baselines import uncoded_rate1_baseline, channel_hard_baseline
from ldpc_5g import build_parity

par = build_parity(bgn=1, set_idx=1, zc=8)
print(f"BG1 set1 Zc=8: N={par.cols} M={par.rows} "
      f"K_info={info_bits_count(par)}  base R={physical_code_rate(par):.4f}")
for target in (None, 0.5):
    tx = tx_length(par, target)
    print(f"  target_R={target}  tx_len={tx}  actual_R={info_bits_count(par)/tx:.4f}")

snrs = (1.0, 2.0, 3.0, 4.0, 5.0)
cfg = FitnessConfig(par=par, snr_list=snrs, n_frames_per_snr=100,
                    max_iter=20, target_code_rate=0.5)
print(f"\nFitnessConfig: target_R=0.5  effective_R={cfg.effective_code_rate:.4f}  tx_len={cfg.tx_len}")

t0 = time.time()
m = evaluate_genome_cpp_ber(oms_seed_genome(), cfg)
print(f"OMS eval took {time.time()-t0:.1f}s")
ch = channel_hard_baseline(cfg)
r1 = uncoded_rate1_baseline(cfg, bits_per_snr=100 * cfg.tx_len)
print(f"\n{'SNR':>5} | {'OMS BP':>12} | {'ch-hard':>12} | {'R=1':>12} | {'gain':>8}")
print("-" * 60)
for i, s in enumerate(snrs):
    b1 = m.ber_per_snr[i]
    b2 = ch['ber_per_snr'][i]
    b3 = r1['ber_per_snr'][i]
    g = (b2 / b1) if b1 > 0 else float('inf')
    print(f"{s:>5.1f} | {b1:>12.3e} | {b2:>12.3e} | {b3:>12.3e} | {g:>7.2f}x")
