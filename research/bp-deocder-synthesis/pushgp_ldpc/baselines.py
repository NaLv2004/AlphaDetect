"""Uncoded reference baselines for the BER-vs-SNR comparison plot.

Two flavours of "uncoded" are reported alongside the OMS-decoded curve:

1. **rate1_hard**  ─ pure BPSK + AWGN with σ² computed for R = 1
   (i.e. no coding gain advertised to the channel).  This is the
   textbook AWGN reference BER ≈ Q(√(2·10^(SNR/10))).

2. **channel_hard_same_pipeline** ─ uses the *exact* fitness pipeline
   (same `FitnessConfig`, same random codewords, same seeds, same
   σ² derived from the physical code rate) but skips BP completely and
   hard-decides on the channel LLR alone.  Punctured positions
   (the first 2*Zc info bits, whose LLR is 0) are excluded from the
   error count because their hard decision is undefined.

Both baselines write a `{snr_db, ber_per_snr}` record so the UI can
overlay them on the BER chart without touching the rest of the runner.
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
    """BPSK + AWGN with R=1 (no coding).  Independent of `cfg.par`."""
    rng = np.random.default_rng(seed)
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
    """Same encoder+channel+noise as fitness; decoder = hard slice on channel LLR.

    Bit-comparison is restricted to the transmitted positions (i.e. the
    coded bits with index ≥ 2*Zc).  This makes the curve directly
    comparable to the OMS-decoded BER, which is also reported on the
    full codeword but where the punctured bits are reconstructed by BP.
    """
    par = cfg.par
    skip = 2 * par.zc
    snrs: List[float] = [float(s) for s in cfg.snr_list]
    bers: List[float] = []
    for snr in snrs:
        pairs = _channel_inputs(cfg, snr)
        n_err = 0
        n_bits = 0
        for cw, llr in pairs:
            # hard decision on channel LLR (LLR>0 → bit 0, LLR<0 → bit 1)
            hat = (llr[skip:] < 0.0).astype(np.int8)
            ref = cw[skip:].astype(np.int8)
            n_err += int(np.sum(hat != ref))
            n_bits += int(ref.size)
        bers.append(n_err / max(1, n_bits))
    return {
        "kind": "channel_hard_same_pipeline",
        "snr_list_db": snrs,
        "ber_per_snr": bers,
        "n_frames_per_snr": cfg.n_frames_per_snr,
        "code_rate_used": cfg.effective_code_rate,
        "note": "hard-slice channel LLR; excludes first 2*Zc punctured bits",
    }


__all__ = ["uncoded_rate1_baseline", "channel_hard_baseline"]
