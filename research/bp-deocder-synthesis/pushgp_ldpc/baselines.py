"""Uncoded reference baselines for the info-bit BER-vs-SNR comparison plot.

Two flavours of "uncoded" are reported alongside the OMS-decoded curve:

1. **rate1_hard**  ─ pure BPSK + AWGN with σ² computed for R = 1
   (i.e. no coding gain advertised to the channel).  This is the
   textbook AWGN reference BER ≈ Q(√(2·10^(SNR/10))).

2. **channel_hard_same_pipeline** ─ uses the *exact* fitness pipeline
   (same FitnessConfig, same random codewords, same rate-matching,
   same σ² derived from A/E) but skips BP completely and hard-decides
   the recovered LLR vector at the K_cb_bit info-bit positions.  The
   first 2*Zc info bits are always punctured (LLR=0) so the hard-slice
   error rate is 0.5 there — exactly the penalty a real receiver would
   pay without a decoder, which makes this a useful "no BP" reference.

Both baselines write `{snr_db, ber_per_snr}` records so the UI can
overlay them on the info-bit BER chart.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from pushgp_ldpc.eval import FitnessConfig, _channel_inputs


def _ber_rate1_hard(snr_db: float, n_bits: int, rng: np.random.Generator) -> float:
    sigma2 = 1.0 / (2.0 * 1.0 * 10.0 ** (snr_db / 10.0))
    bits = rng.integers(0, 2, size=n_bits, dtype=np.int8)
    tx = 1.0 - 2.0 * bits.astype(np.float64)
    rx = tx + np.sqrt(sigma2) * rng.standard_normal(n_bits)
    hat = (rx < 0.0).astype(np.int8)
    return float(np.mean(hat != bits))


def uncoded_rate1_baseline(cfg: FitnessConfig,
                            bits_per_snr: int = 50_000,
                            seed: int = 20260601) -> Dict[str, list]:
    """BPSK + AWGN with R=1 (no coding).  Independent of code config.

    Reports BER over `bits_per_snr` random info bits at each SNR.
    """
    snrs: List[float] = [float(s) for s in cfg.snr_list]
    bers: List[float] = [
        _ber_rate1_hard(s, bits_per_snr, np.random.default_rng(seed + int(s * 1000)))
        for s in snrs
    ]
    return {
        "kind": "uncoded_rate1_hard",
        "snr_list_db": snrs,
        "ber_per_snr": bers,
        "n_bits_per_snr": int(bits_per_snr),
    }


def channel_hard_baseline(cfg: FitnessConfig) -> Dict[str, list]:
    """Same encoder + channel + noise as fitness; decoder = hard slice on
    the recovered LLR vector over the K_cb_bit info-bit positions.

    This is the canonical "no BP" reference: it exposes the 2*Zc
    punctured info prefix as a 50%-error region (LLR=0 → arbitrary
    hard decision).  Reported BER uses the SAME denominator
    (K_cb_bit per frame) as the fitness BER, so the two curves are
    directly comparable.
    """
    K_cb_bit = cfg.K_cb_bit
    snrs: List[float] = [float(s) for s in cfg.snr_list]
    bers: List[float] = []
    for snr in snrs:
        pairs = _channel_inputs(cfg, snr)
        n_err = 0
        n_bits = 0
        for info_payload, llr in pairs:
            # hard decision on llr over the K_cb_bit info-bit positions
            hat = (llr[:K_cb_bit] < 0.0).astype(np.int8)
            n_err += int(np.sum(hat != info_payload))
            n_bits += K_cb_bit
        bers.append(n_err / max(1, n_bits))
    return {
        "kind": "channel_hard_same_pipeline",
        "snr_list_db": snrs,
        "ber_per_snr": bers,
        "n_frames_per_snr": cfg.n_frames_per_snr,
        "code_rate_used": cfg.effective_code_rate,
        "info_len_A": cfg.info_len_A,
        "code_length_E": cfg.code_length_E,
        "K_cb_bit": K_cb_bit,
        "note": "hard-slice recovered LLR over K_cb_bit info bits (2*Zc prefix punctured → 0.5 error there)",
    }


__all__ = ["uncoded_rate1_baseline", "channel_hard_baseline"]

