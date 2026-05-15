"""Fitness evaluation for evolved decoder Push genomes.

We measure decoder quality on a set of (SNR, n_frames) configurations
on a fixed LDPC code (parity matrix `par`), starting from the same
fixed-seed channel realisations across all genomes (so noise is
*shared* and the only variability is the decoder behaviour).

The fitness returned by `evaluate_genome` is the mean of
`log10(BER + EPS)` across SNR points (smaller is better).  Using log-BER
avoids saturation at 0.5 and gives meaningful gradient even when most
programs fail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldpc_5g import (
    HTYPE,
    LiftedParity,
    bpsk_llr,
    bpsk_modulate,
    decode_bp,
    encode_codeblock,
)
from pushgp.genome import Genome
from pushgp_ldpc.adapter import make_callables


EPS = 1e-6


def _random_codeword(par: LiftedParity, htype: int, rng: np.random.Generator) -> np.ndarray:
    """Encode a random message into a full N-length codeword (int8 0/1).

    Reconstructs the punctured prefix (first 2*Zc info bits) so the
    returned vector covers the full lifted codeword and `H @ cw == 0`.
    """
    Kb = 10 if par.bgn == 2 else 22
    K = Kb * par.zc
    info = rng.integers(0, 2, size=K, dtype=np.int8)
    cw_punct = encode_codeblock(info, par, htype)  # int8, length N - 2*Zc
    cw_full = np.concatenate([info[: 2 * par.zc], cw_punct]).astype(np.int8)
    return cw_full


@dataclass
class FitnessConfig:
    par: LiftedParity
    snr_list: Tuple[float, ...]      # in dB
    n_frames_per_snr: int = 4
    max_iter: int = 8
    code_rate: float = 0.5
    seed_base: int = 12345           # shared channel seed
    info_bits_per_frame: int = 0     # 0 → use par.cols (transmit codeword=0)
    early_fail_threshold: float = 0.45  # bail out if BER > this on first SNR


def _channel_inputs(cfg: FitnessConfig, snr_db: float) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Pre-build (transmitted_codeword, channel_llr) pairs for one SNR.

    Each frame uses a freshly encoded random codeword (so trivial
    constant-output 'decoders' cannot win the GA).
    """
    rng = np.random.default_rng(cfg.seed_base + abs(int(snr_db * 1000)))
    par = cfg.par
    htype = HTYPE[par.bgn - 1][par.set_idx - 1]
    rate = cfg.code_rate
    sigma2 = 1.0 / (2.0 * rate * 10.0 ** (snr_db / 10.0))
    sigma = float(np.sqrt(sigma2))
    pairs = []
    for _ in range(cfg.n_frames_per_snr):
        cw = _random_codeword(par, htype, rng)
        # BPSK only the non-punctured part; first 2*Zc bits get LLR=0.
        tx = bpsk_modulate(cw[2 * par.zc :])
        rx = tx + sigma * rng.standard_normal(tx.shape)
        llr_part = bpsk_llr(rx, sigma2)
        llr = np.zeros(par.cols, dtype=np.float64)
        llr[2 * par.zc :] = llr_part
        pairs.append((cw, llr))
    return pairs


def evaluate_genome(genome: Genome, cfg: FitnessConfig) -> float:
    """Return mean log10(BER+EPS) across SNRs (smaller = better).

    Returns +1.0 (i.e. log10(0.1+EPS) ≈ -1) if the adapter raises or
    yields all-NaN posteriors, but pinned to a large value (1.0) so the
    GA can still rank.  Catastrophic failures (exception) → +6.0.
    """
    try:
        v2c_fn, c2v_fn = make_callables(genome)
    except Exception:
        return 6.0

    log_bers: List[float] = []
    for snr_db in cfg.snr_list:
        pairs = _channel_inputs(cfg, snr_db)
        n_err = 0
        n_bits = 0
        for bits, llr in pairs:
            try:
                post = decode_bp(
                    llr, cfg.par,
                    v2c_fn=v2c_fn, c2v_fn=c2v_fn,
                    max_iter=cfg.max_iter, offset=0.25,
                    code_rate=cfg.code_rate,
                )
            except Exception:
                return 6.0
            hat = (post < 0.0).astype(np.int8)
            n_err += int((hat != bits).sum())
            n_bits += bits.size
        ber = n_err / max(1, n_bits)
        log_bers.append(float(np.log10(ber + EPS)))
        # Early exit if first SNR catastrophically bad — saves time.
        if len(log_bers) == 1 and ber > cfg.early_fail_threshold:
            # Pad the remaining SNRs with the same bad value.
            log_bers.extend([log_bers[0]] * (len(cfg.snr_list) - 1))
            break

    return float(np.mean(log_bers))


__all__ = ["FitnessConfig", "evaluate_genome", "EPS"]
