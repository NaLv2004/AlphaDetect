"""Print final OMS vs uncoded comparison after the code_rate fix.
Runs at the canonical physical rate (auto-derived from par).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pushgp_ldpc.eval import FitnessConfig
from pushgp_ldpc.eval_cpp import evaluate_genome_cpp_ber
from pushgp_ldpc.adapter import oms_seed_genome
from pushgp_ldpc.baselines import uncoded_rate1_baseline, channel_hard_baseline
from ldpc_5g import build_parity

par = build_parity(bgn=2, set_idx=1, zc=2)
snrs = tuple(float(x) for x in range(0, 11))
cfg = FitnessConfig(par=par, snr_list=snrs, n_frames_per_snr=200, max_iter=20)

m = evaluate_genome_cpp_ber(oms_seed_genome(), cfg)
r1 = uncoded_rate1_baseline(cfg, bits_per_snr=200 * 100)
ch = channel_hard_baseline(cfg)

print(f"R_eff={cfg.effective_code_rate}, max_iter=20, n_frames=200")
print(f"{'SNR':>5} | {'OMS BP':>12} | {'channel-hard':>14} | "
      f"{'R=1 uncoded':>14} | {'OMS gain vs ch-hard':>20}")
print("-" * 86)
for i, s in enumerate(snrs):
    b1 = m.ber_per_snr[i]
    b2 = ch['ber_per_snr'][i]
    b3 = r1['ber_per_snr'][i]
    g = (b2 / b1) if b1 > 0 else float('inf')
    print(f"{s:>5.1f} | {b1:>12.3e} | {b2:>14.3e} | {b3:>14.3e} | {g:>19.2f}x")
