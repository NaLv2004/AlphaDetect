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
from typing import List, Optional, Sequence, Tuple

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


def physical_code_rate(par: LiftedParity) -> float:
    """True transmitted code rate when ONLY the mandatory 2*Zc info bits
    are punctured: K_info / (N - 2*Zc).

    For 5G NR BG2 set1 this is fixed by the base graph at 10/50 = 0.20;
    for BG1 set1 it is 22/66 ≈ 0.333.  To achieve any other rate the
    transmitter must additionally puncture parity bits (rate matching),
    which is controlled by `FitnessConfig.target_code_rate`.
    """
    Kb = 10 if par.bgn == 2 else 22
    K_info = Kb * par.zc
    N_tx = par.cols - 2 * par.zc
    if N_tx <= 0:
        raise ValueError(f"degenerate par (N_tx={N_tx})")
    return float(K_info) / float(N_tx)


def info_bits_count(par: LiftedParity) -> int:
    Kb = 10 if par.bgn == 2 else 22
    return Kb * par.zc


def tx_length(par: LiftedParity, target_code_rate: Optional[float]) -> int:
    """Number of bits actually transmitted per codeword (BPSK symbols).

    Always >= K_info and <= N - 2*Zc.  When `target_code_rate` is None
    we fall back to the full un-puncturable suffix (no extra parity
    puncturing).  When a target rate is requested we compute
    ceil(K_info / target_rate) and clip into the legal range.
    """
    K = info_bits_count(par)
    N_full = par.cols - 2 * par.zc
    if target_code_rate is None:
        return N_full
    if target_code_rate <= 0.0 or target_code_rate > 1.0:
        raise ValueError(f"target_code_rate out of range: {target_code_rate}")
    import math
    tx = int(math.ceil(K / float(target_code_rate)))
    if tx < K:
        raise ValueError(f"target rate {target_code_rate} would shorten below K={K}")
    if tx > N_full:
        raise ValueError(
            f"target rate {target_code_rate} demands tx={tx} > N-2*Zc={N_full}; "
            f"choose a base graph with lower base rate or a larger Zc")
    return tx


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
    # When None (the canonical setting), the channel pipeline derives
    # the rate from `par` via `physical_code_rate(par)`.  Explicit
    # numeric values are tolerated only for legacy/scratch callers; the
    # runner refuses to launch with a hard-coded rate.  Never pass 0.5
    # to a real evolution — it under-estimates σ² by ~2.6× on BG2 Zc=2
    # and makes every decoder look better than it really is.
    code_rate: Optional[float] = None
    # Target transmitted code rate.  When set, the channel pipeline
    # additionally punctures parity bits past position `tx_length(par,
    # target_code_rate)` so the actual transmitted rate is exactly the
    # requested value (up to integer ceiling).  When None, no extra
    # parity puncturing is applied and the rate equals the base graph's
    # physical rate.
    target_code_rate: Optional[float] = None
    seed_base: int = 12345           # shared channel seed
    info_bits_per_frame: int = 0     # 0 → use par.cols (transmit codeword=0)
    early_fail_threshold: float = 0.45  # bail out if BER > this on first SNR
    # When True, route every BP decode through the C++ kernel
    # `pushgp_cpp_dce.decode_bp` instead of the Python `decode_bp`. The
    # cpp path is byte-locked to the Python path to 6 decimals by
    # `cpp_dce/tests/test_bp_equivalence.py` and reproduces the exact
    # SNR/frame/BER bookkeeping in `pushgp_ldpc.eval_cpp`.  Disable to
    # fall back to the legacy Python loop (useful for A/B equivalence
    # tests and for environments where the .pyd cannot be loaded).
    use_cpp_fitness: bool = True

    @property
    def effective_code_rate(self) -> float:
        """Resolved code rate used by `_channel_inputs` and BP kernels.

        Priority (highest first):
            1. explicit `code_rate` (legacy callers only);
            2. computed from `target_code_rate` + actual tx_length;
            3. `physical_code_rate(par)` (no extra parity puncturing).
        """
        if self.code_rate is not None:
            return float(self.code_rate)
        if self.target_code_rate is not None:
            K = info_bits_count(self.par)
            return float(K) / float(tx_length(self.par, self.target_code_rate))
        return physical_code_rate(self.par)

    @property
    def tx_len(self) -> int:
        """Number of transmitted BPSK symbols per codeword."""
        return tx_length(self.par, self.target_code_rate)


def _channel_inputs(cfg: FitnessConfig, snr_db: float) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Pre-build (transmitted_codeword, channel_llr) pairs for one SNR.

    Each frame uses a freshly encoded random codeword (so trivial
    constant-output 'decoders' cannot win the GA).
    """
    rng = np.random.default_rng(cfg.seed_base + abs(int(snr_db * 1000)))
    par = cfg.par
    htype = HTYPE[par.bgn - 1][par.set_idx - 1]
    rate = cfg.effective_code_rate
    sigma2 = 1.0 / (2.0 * rate * 10.0 ** (snr_db / 10.0))
    sigma = float(np.sqrt(sigma2))
    tx_len = cfg.tx_len  # number of BPSK symbols actually transmitted
    N_full = par.cols - 2 * par.zc
    pairs = []
    for _ in range(cfg.n_frames_per_snr):
        cw = _random_codeword(par, htype, rng)
        # Transmit only the first tx_len bits of the non-punctured suffix
        # (per TS 38.212 §5.4.2.1 rate-matching RV0 from circular buffer
        # start).  Remaining parity bits are additionally punctured.
        tx_bits = cw[2 * par.zc : 2 * par.zc + tx_len]
        tx = bpsk_modulate(tx_bits)
        rx = tx + sigma * rng.standard_normal(tx.shape)
        llr_part = bpsk_llr(rx, sigma2)
        llr = np.zeros(par.cols, dtype=np.float64)
        # mandatory 2*Zc prefix already zero; fill transmitted region:
        llr[2 * par.zc : 2 * par.zc + tx_len] = llr_part
        # positions [2*Zc + tx_len : N] stay zero (additional puncturing)
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
                    code_rate=cfg.effective_code_rate,
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


__all__ = [
    "FitnessConfig", "evaluate_genome", "EPS",
    "physical_code_rate", "info_bits_count", "tx_length",
]
