"""Verify the OMS-rediscovery target before launching the experiment.

We need to know — for the EXACT (A, E, SNR, n_frames, max_iter) the
evolution will use — what fitness/BER/FER the canonical OMS achieves,
because the evolution restricted to OMS-only opcodes should be able to
reach (or get close to) this number.

Also probes the constant-range issue: OMS uses K1=1e6 (log10=6) as the
min-fold sentinel, which is OUTSIDE the default RAND_LOG_CONST_MAX=1.0.
We check whether a smaller sentinel (K1 in {100, 1e3, 1e4}) still
delivers OMS-equivalent BER.  If yes, we can widen the random sampling
range only to log10=2 or 3 instead of 6.

Outputs are printed; no files written.  Run with:
    conda run -n AutoGenOld python research/bp-deocder-synthesis/code_review/verify_oms_rediscover_target.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BPS = _HERE.parent
sys.path.insert(0, str(_BPS))

import numpy as np

from pushgp_ldpc.adapter import oms_seed_genome
from pushgp_ldpc.eval import FitnessConfig
from pushgp_ldpc.eval_logged import evaluate_genome_with_ber


def run_one(A: int, E: int, snrs, n_frames: int, max_iter: int,
            beta: float, sentinel: float, *, use_cpp: bool = True, tag: str = ""):
    g = oms_seed_genome(beta=beta, sentinel=sentinel)
    fit_cfg = FitnessConfig(
        info_len_A=A,
        code_length_E=E,
        snr_list=tuple(snrs),
        n_frames_per_snr=n_frames,
        max_iter=max_iter,
        use_cpp_fitness=use_cpp,
    )
    t0 = time.time()
    m = evaluate_genome_with_ber(g, fit_cfg)
    dt = time.time() - t0
    K0 = 10.0 ** g.log_constants[0]
    K1 = 10.0 ** g.log_constants[1]
    print(f"[{tag}] beta={beta:.3f} sentinel={sentinel:g}  "
          f"K0={K0:.3f} K1={K1:.3g}  fit={m.fitness:+.4f}  "
          f"BER={['%.3e' % b for b in m.ber_per_snr]}  "
          f"FER={['%.3e' % b for b in m.fer_per_snr]}  "
          f"valid={m.valid}  t={dt:.2f}s",
          flush=True)
    return m


def main():
    A = 176
    E = 352
    snrs = [2.0, 3.0, 4.0]
    n_frames = 20
    max_iter = 8

    print(f"=== OMS rediscover target: A={A} E={E} SNRs={snrs} "
          f"n_frames={n_frames} max_iter={max_iter} ===", flush=True)
    print()
    print("--- Canonical OMS (beta=0.25, sentinel=1e6) ---", flush=True)
    m_can = run_one(A, E, snrs, n_frames, max_iter, beta=0.25, sentinel=1e6,
                    tag="OMS-canonical")
    print()
    print("--- Smaller sentinels (does evolution need K1 in [10^6] or just [10^2]?) ---",
          flush=True)
    sentinels = [10.0, 30.0, 100.0, 1000.0, 1e4, 1e5]
    rows = []
    for s in sentinels:
        m = run_one(A, E, snrs, n_frames, max_iter, beta=0.25, sentinel=s,
                    tag=f"OMS-K1={s:g}")
        rows.append((s, m.ber_per_snr[-1], m.fer_per_snr[-1]))

    print()
    print("--- Summary: sentinel sweep at SNR=4dB ---", flush=True)
    print(f"{'sentinel':>10}  {'BER@4dB':>12}  {'FER@4dB':>12}  rel-to-canonical")
    can_ber = m_can.ber_per_snr[-1]
    can_fer = m_can.fer_per_snr[-1]
    for s, b, f in rows:
        rel = (b / max(can_ber, 1e-12))
        print(f"{s:>10g}  {b:>12.3e}  {f:>12.3e}  ber x{rel:.3f}")

    print()
    print("--- Suggested rand-log-const range for evolution ---", flush=True)
    # Find smallest sentinel S such that BER stays within 2x of canonical.
    ok = [s for s, b, f in rows if b <= 2.0 * can_ber]
    if ok:
        min_S = min(ok)
        print(f"   smallest sentinel within 2x canonical BER@4dB: {min_S:g}  "
              f"=> RAND_LOG_CONST_MAX >= {np.log10(min_S):.2f}",
              flush=True)
    else:
        print("   no sentinel <= 1e5 reaches 2x of canonical; "
              "evolution will need full log_const range up to 6.", flush=True)


if __name__ == "__main__":
    main()
