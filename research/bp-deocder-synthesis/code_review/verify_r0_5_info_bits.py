"""Re-run the R=0.5 sweep but report BER over INFO BITS only
(matching the SEU reference convention in cpp-ldpc-simulation/main.cpp:678,
which counts errors over A info bits, not the full codeword).

Hypothesis: our OMS curve looked uncompetitive because we were dividing
errors by N=544 instead of K_info=176 (so parity-bit errors were inflating
the reported BER while the actually-recovered info was fine).
"""
from __future__ import annotations

import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from ldpc_5g import build_parity
from pushgp_ldpc.eval import FitnessConfig, _channel_inputs, info_bits_count, tx_length, physical_code_rate
from pushgp_ldpc.eval_cpp import _get_cdce, _get_parity_handle
from pushgp.serialize import program_to_dict
from pushgp_ldpc.adapter import oms_seed_genome

par = build_parity(bgn=1, set_idx=1, zc=8)
K = info_bits_count(par)        # 176
N = par.cols                    # 544
M = par.rows                    # 368
print(f"BG1 set1 Zc=8: N={N} M={M} K_info={K}  base R={physical_code_rate(par):.4f}")

target_R = 0.5
tx_len = tx_length(par, target_R)
print(f"  target_R={target_R}  tx_len={tx_len}  actual_R={K/tx_len:.4f}")

snrs = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0)
N_FRAMES = 200
MAX_ITER = 20

oms = oms_seed_genome()
cdce = _get_cdce()

print(f"\nSweep: {len(snrs)} SNR pts, {N_FRAMES} frames each, max_iter={MAX_ITER}")
print("BER over INFO BITS ONLY (denom=K={}); cw-BER for comparison.".format(K))
print()
print("  SNR | info-BER OMS |  cw-BER OMS  | info ch-hard | R=1 uncoded  |  sec")
print("-" * 78)

for snr_db in snrs:
    cfg = FitnessConfig(par=par, snr_list=(snr_db,), n_frames_per_snr=N_FRAMES,
                        max_iter=MAX_ITER, target_code_rate=target_R,
                        seed_base=20260601)
    parH = _get_parity_handle(par)
    v_dict = program_to_dict(oms.prog_v2c)
    c_dict = program_to_dict(oms.prog_c2v)
    evo = oms.evo_const_values().astype(np.float64)
    pairs = _channel_inputs(cfg, snr_db)

    n_err_info = 0
    n_err_cw = 0
    n_err_ch = 0
    n_bits_info = 0
    n_bits_cw = 0
    n_bits_ch = 0
    t1 = time.time()
    for cw, llr in pairs:
        post, _it = cdce.decode_bp(llr, parH, v_dict, c_dict, evo,
                                    MAX_ITER, 0.25, cfg.effective_code_rate)
        hat = (post < 0.0).astype(np.int8)
        # info-bit BER: indices [0, K)
        n_err_info += int((hat[:K] != cw[:K]).sum())
        n_bits_info += K
        # cw BER for reference
        n_err_cw += int((hat != cw).sum())
        n_bits_cw += cw.size
        # channel-hard info-bit BER: only over transmitted info positions
        # info bits live at [0, K); transmitted info = [2*Zc, K) (first 2*Zc are punctured)
        ch_hat = (llr[2*par.zc:K] < 0.0).astype(np.int8)
        n_err_ch += int((ch_hat != cw[2*par.zc:K]).sum())
        n_bits_ch += (K - 2*par.zc)
    dt = time.time() - t1

    # uncoded R=1 reference at same Eb/N0
    rng = np.random.default_rng(int(snr_db * 1000) + 999)
    sigma2_r1 = 1.0 / (2.0 * 1.0 * 10.0 ** (snr_db / 10.0))
    nb = 50_000
    b = rng.integers(0, 2, size=nb, dtype=np.int8)
    tx = 1.0 - 2.0 * b.astype(np.float64)
    rx = tx + np.sqrt(sigma2_r1) * rng.standard_normal(nb)
    ber_r1 = float(np.mean((rx < 0.0).astype(np.int8) != b))

    ber_info = n_err_info / max(1, n_bits_info)
    ber_cw = n_err_cw / max(1, n_bits_cw)
    ber_ch = n_err_ch / max(1, n_bits_ch)
    print(f"{snr_db:>5.1f} | {ber_info:>12.3e} | {ber_cw:>12.3e} | "
          f"{ber_ch:>12.3e} | {ber_r1:>12.3e} | {dt:>5.1f}", flush=True)

print("\nDONE.")
