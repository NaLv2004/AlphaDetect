"""Benchmark a champion genome against the default OMS decoder.

Computes BER (and FER) at a list of SNR points using the same
all-zero-codeword channel model as `eval.py`.  Outputs are written as
CSV + PNG (PNG only if matplotlib is available).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldpc_5g import (
    HTYPE,
    LiftedParity,
    bpsk_llr,
    bpsk_modulate,
    decode_bp,
    decode_oms_fast,
    build_oms_context,
    default_c2v_oms,
    default_v2c_oms,
    encode_codeblock,
)
from pushgp.genome import Genome
from pushgp_ldpc.adapter import make_callables


def _random_codeword(par: LiftedParity, htype: int, rng) -> np.ndarray:
    Kb = 10 if par.bgn == 2 else 22
    K = Kb * par.zc
    info = rng.integers(0, 2, size=K, dtype=np.int8)
    cw_punct = encode_codeblock(info, par, htype)
    return np.concatenate([info[: 2 * par.zc], cw_punct]).astype(np.int8)


@dataclass
class BenchResult:
    snr_db: List[float]
    ber_oms: List[float]
    ber_champ: List[float]
    fer_oms: List[float]
    fer_champ: List[float]


def run_benchmark(
    champion: Genome,
    par: LiftedParity,
    snr_list: Sequence[float],
    n_frames: int,
    *,
    max_iter: int = 25,
    code_rate: float = 0.5,
    seed: int = 99,
) -> BenchResult:
    v2c_fn, c2v_fn = make_callables(champion)
    ctx = build_oms_context(par)

    snr_out: List[float] = []
    ber_o, ber_c, fer_o, fer_c = [], [], [], []

    htype = HTYPE[par.bgn - 1][par.set_idx - 1]

    for snr_db in snr_list:
        rng = np.random.default_rng(seed + abs(int(snr_db * 1000)))
        sigma2 = 1.0 / (2.0 * code_rate * 10.0 ** (snr_db / 10.0))
        sigma = float(np.sqrt(sigma2))

        n_bits = par.cols * n_frames
        n_err_o = n_err_c = 0
        n_fer_o = n_fer_c = 0
        for _ in range(n_frames):
            cw = _random_codeword(par, htype, rng)
            tx = bpsk_modulate(cw[2 * par.zc :])
            rx = tx + sigma * rng.standard_normal(tx.shape)
            llr = np.zeros(par.cols, dtype=np.float64)
            llr[2 * par.zc :] = bpsk_llr(rx, sigma2)

            post_o, _ = decode_oms_fast(llr, ctx, max_iter=max_iter, offset=0.25)
            hat_o = (post_o < 0.0).astype(np.int8)
            errs_o = int((hat_o != cw).sum())
            n_err_o += errs_o
            n_fer_o += (1 if errs_o else 0)

            try:
                post_c = decode_bp(llr, par, v2c_fn=v2c_fn, c2v_fn=c2v_fn,
                                   max_iter=max_iter, offset=0.25,
                                   code_rate=code_rate)
                hat_c = (post_c < 0.0).astype(np.int8)
                errs_c = int((hat_c != cw).sum())
            except Exception:
                errs_c = par.cols
            n_err_c += errs_c
            n_fer_c += (1 if errs_c else 0)

        snr_out.append(float(snr_db))
        ber_o.append(n_err_o / n_bits)
        ber_c.append(n_err_c / n_bits)
        fer_o.append(n_fer_o / n_frames)
        fer_c.append(n_fer_c / n_frames)

    return BenchResult(snr_out, ber_o, ber_c, fer_o, fer_c)


def write_csv(res: BenchResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["snr_db", "ber_oms", "ber_champ", "fer_oms", "fer_champ"])
        for i in range(len(res.snr_db)):
            w.writerow([res.snr_db[i], res.ber_oms[i], res.ber_champ[i],
                        res.fer_oms[i], res.fer_champ[i]])


def plot_results(res: BenchResult, png_path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    eps = 1e-7
    ber_o = np.maximum(np.array(res.ber_oms), eps)
    ber_c = np.maximum(np.array(res.ber_champ), eps)
    fer_o = np.maximum(np.array(res.fer_oms), eps)
    fer_c = np.maximum(np.array(res.fer_champ), eps)
    axs[0].semilogy(res.snr_db, ber_o, "o-", label="OMS")
    axs[0].semilogy(res.snr_db, ber_c, "s--", label="Champion")
    axs[0].set_xlabel("SNR (dB)"); axs[0].set_ylabel("BER"); axs[0].grid(True); axs[0].legend()
    axs[1].semilogy(res.snr_db, fer_o, "o-", label="OMS")
    axs[1].semilogy(res.snr_db, fer_c, "s--", label="Champion")
    axs[1].set_xlabel("SNR (dB)"); axs[1].set_ylabel("FER"); axs[1].grid(True); axs[1].legend()
    fig.suptitle("Champion vs OMS")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    return True


__all__ = ["BenchResult", "run_benchmark", "write_csv", "plot_results"]
