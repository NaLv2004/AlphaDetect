"""Fitness evaluation for evolved decoder Push genomes.

User-facing config is **(A, E)**:
    A = info_len_A  : number of information bits per code block (= A in
                      3GPP TS 38.212; no TB-CRC).
    E = code_length_E : rate-matched output length per code block (= E
                        in 3GPP TS 38.212, the number of transmitted
                        BPSK symbols).

All other LDPC parameters (bgn, set_idx, Zc, Kb, N, K, K_cb_bit,
N_punctured, ...) are derived via `ldpc_5g.derive_params(A, A/E)`
which mirrors the SEU C++ reference implementation in
`code_for_reference/cpp-ldpc-simulation/GAMP_Test/main.cpp`.

Channel pipeline (per frame) follows simulate.py exactly:
    1. encode K-bit info CB with `encode_codeblock`
       (length N_punctured = N - 2*Zc, with -1 filler markers)
    2. `cbs_rate_match` with RV=0  →  length-E hard bit stream
    3. BPSK modulation, AWGN(sigma)
    4. `bpsk_llr` (channel LLR over the E received symbols)
    5. `cbs_rate_recover` re-inserts E LLRs into length-N_punctured
       slots; filler positions filled with +1e9 (known bit-0)
    6. prepend 2*Zc zero LLRs (mandatory NR puncture)  →  length-N
       LLR vector fed to the BP decoder.

BER is computed strictly over the K_cb_bit *information* bits
(matches simulate.py and the SEU reference at
`cpp-ldpc-simulation/GAMP_Test/main.cpp:678`).

The legacy (par=..., target_code_rate=...) API has been removed; pass
A and E instead.  Callers that need a specific lifted code may build
their own `LiftedParity` and use the lower-level BP utilities in
`ldpc_5g.py` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import List, Optional, Sequence, Tuple

import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldpc_5g import (
    HTYPE,
    LiftedParity,
    bpsk_llr,
    bpsk_modulate,
    build_parity,
    cbs_rate_match,
    cbs_rate_recover,
    compute_E_array,
    compute_k0,
    decode_bp,
    derive_params,
    encode_codeblock,
)
from pushgp.genome import Genome
from pushgp_ldpc.adapter import make_callables


EPS = 1e-6


@dataclass(frozen=True)
class FitnessConfig:
    """Code + channel configuration for fitness evaluation.

    Required:
        info_len_A    : K-info bits per CB (= TS 38.212 A, no TB-CRC).
        code_length_E : rate-matched output length per CB (= TS 38.212 E).
        snr_list      : Eb/N0 grid in dB.

    Channel realisations are seeded by `seed_base + |int(snr_db*1000)|`,
    so every genome in a generation sees identical noise/codewords.

    The dataclass is frozen so derived fields can be safely cached.
    """
    info_len_A: int
    code_length_E: int
    snr_list: Tuple[float, ...]
    n_frames_per_snr: int = 8
    max_iter: int = 25
    seed_base: int = 12345
    early_fail_threshold: float = 0.45
    use_cpp_fitness: bool = True

    def __post_init__(self):
        if self.info_len_A <= 0:
            raise ValueError(f"info_len_A must be positive (got {self.info_len_A})")
        if self.code_length_E <= 0:
            raise ValueError(f"code_length_E must be positive (got {self.code_length_E})")
        if self.code_length_E < self.info_len_A:
            raise ValueError(
                f"code_length_E={self.code_length_E} < info_len_A={self.info_len_A}: "
                "would imply rate > 1 (uncoded or worse)")
        rate = float(self.info_len_A) / float(self.code_length_E)
        if not (0.0 < rate <= 1.0):
            raise ValueError(f"derived code_rate {rate} out of (0, 1]")
        p = derive_params(self.info_len_A, rate)
        if p.C != 1:
            raise ValueError(
                f"FitnessConfig requires single-CB configs (C=1); "
                f"A={self.info_len_A} R={rate} produced C={p.C}. "
                "Split into smaller A or use the lower-level ldpc_5g API.")
        # Store derived params via object.__setattr__ (frozen dataclass).
        object.__setattr__(self, "_derived", p)
        object.__setattr__(self, "_par", build_parity(p.bgn, p.set_idx, p.zc))

    # ---- derived accessors -------------------------------------------

    @property
    def derived(self):
        return self._derived  # type: ignore[attr-defined]

    @property
    def par(self) -> LiftedParity:
        return self._par  # type: ignore[attr-defined]

    @property
    def K_cb_bit(self) -> int:
        return self.derived.K_cb_bit

    @property
    def K(self) -> int:
        return self.derived.K

    @property
    def N(self) -> int:
        return self.derived.N

    @property
    def N_punctured(self) -> int:
        return self.derived.N_punctured

    @property
    def zc(self) -> int:
        return self.derived.zc

    @property
    def bgn(self) -> int:
        return self.derived.bgn

    @property
    def set_idx(self) -> int:
        return self.derived.set_idx

    @property
    def effective_code_rate(self) -> float:
        """Standard NR rate: A / E (matches simulate.py and SEU)."""
        return float(self.info_len_A) / float(self.code_length_E)


# Backwards-compatible helpers used by tests and downstream callers.
def physical_code_rate(par: LiftedParity) -> float:
    """Base-graph un-rate-matched physical rate K/(N-2*Zc).  Only used
    for diagnostics; the fitness pipeline always uses A/E.
    """
    Kb = 10 if par.bgn == 2 else 22
    return float(Kb * par.zc) / float(par.cols - 2 * par.zc)


def info_bits_count(par: LiftedParity) -> int:
    """Total K = Kb*Zc info-slot bits in the lifted code (incl. fillers)."""
    return (10 if par.bgn == 2 else 22) * par.zc


# ---------------------------------------------------------------------------
# Channel realiser.
# ---------------------------------------------------------------------------


def _channel_inputs(cfg: FitnessConfig, snr_db: float) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Pre-build (info_payload, channel_llr_full_N) pairs for one SNR.

    Mirrors `simulate.py:run_point` step-for-step:
        info  (K bits with -1 fillers)
          --[encode_codeblock]-->  d   (N_punctured)
          --[cbs_rate_match RV=0]--> e (E)
          --[BPSK + AWGN]-->        rx
          --[bpsk_llr]-->           llr_chan (E)
          --[cbs_rate_recover]-->   llr_punct (N_punctured)
          --[prepend 2*Zc zeros]--> llr (N)   ← consumed by BP

    Returned `info_payload` is the K_cb_bit-long source bit vector
    (0/1 only; -1 fillers replaced with 0).  Consumers compute BER as
    `(post[:K_cb_bit] < 0) != info_payload` over K_cb_bit bits.
    """
    p = cfg.derived
    par = cfg.par
    htype = HTYPE[p.bgn - 1][p.set_idx - 1]
    rate = cfg.effective_code_rate
    sigma2 = 1.0 / (2.0 * rate * 10.0 ** (snr_db / 10.0))
    sigma = float(np.sqrt(sigma2))

    k0 = compute_k0(p.bgn, p.zc, p.N_punctured, rv=0)
    e_arr = compute_E_array(p.C, p.outlen, qm=1, nlayers=1)
    E = e_arr[0]
    Kd = p.K_cb_bit - 2 * p.zc          # filler region start (post-puncture)
    K_after_punct = p.K - 2 * p.zc      # filler region end   (post-puncture)

    rng = np.random.default_rng(cfg.seed_base + abs(int(snr_db * 1000)))

    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    for _ in range(cfg.n_frames_per_snr):
        info = -np.ones(p.K, dtype=np.int64)
        info[: p.K_cb_bit] = rng.integers(0, 2, size=p.K_cb_bit, dtype=np.int64)
        cw_punct = encode_codeblock(info, par, htype)            # (N_punctured,) int8 with -1 fillers
        tx_bits = cbs_rate_match(cw_punct, E, k0, p.N_punctured, qm=1)  # (E,) int8
        tx_sym = bpsk_modulate(tx_bits)
        rx = tx_sym + sigma * rng.standard_normal(tx_sym.shape)
        llr_chan = bpsk_llr(rx, sigma2)                          # (E,)
        llr_punct = cbs_rate_recover(
            llr_chan, p.N_punctured, E, K_after_punct, Kd, k0, p.N_punctured, qm=1
        )                                                         # (N_punctured,)
        llr = np.concatenate(
            (np.zeros(2 * p.zc, dtype=np.float64), llr_punct)
        )                                                         # (N,)
        info_payload = np.where(info[: p.K_cb_bit] == -1, 0,
                                info[: p.K_cb_bit]).astype(np.int8)
        pairs.append((info_payload, llr))
    return pairs


# ---------------------------------------------------------------------------
# Python-path fitness evaluator (Push-VM in pure Python).
# ---------------------------------------------------------------------------


def evaluate_genome(genome: Genome, cfg: FitnessConfig) -> float:
    """Return mean log10(BER+EPS) across SNRs (smaller = better).

    BER is computed over the K_cb_bit information bits (matches
    simulate.py and SEU `main.cpp:678`).  Returns +6.0 on any
    catastrophic decoder exception so the GA can still rank.
    """
    try:
        v2c_fn, c2v_fn = make_callables(genome)
    except Exception:
        return 6.0

    K_cb_bit = cfg.K_cb_bit
    log_bers: List[float] = []
    for snr_db in cfg.snr_list:
        pairs = _channel_inputs(cfg, snr_db)
        n_err = 0
        n_bits = 0
        for info_payload, llr in pairs:
            try:
                post = decode_bp(
                    llr, cfg.par,
                    v2c_fn=v2c_fn, c2v_fn=c2v_fn,
                    max_iter=cfg.max_iter, offset=0.25,
                    code_rate=cfg.effective_code_rate,
                )
            except Exception:
                return 6.0
            hat = (post[:K_cb_bit] < 0.0).astype(np.int8)
            n_err += int((hat != info_payload).sum())
            n_bits += K_cb_bit
        ber = n_err / max(1, n_bits)
        log_bers.append(float(np.log10(ber + EPS)))
        if len(log_bers) == 1 and ber > cfg.early_fail_threshold:
            log_bers.extend([log_bers[0]] * (len(cfg.snr_list) - 1))
            break

    return float(np.mean(log_bers))


__all__ = [
    "FitnessConfig", "evaluate_genome", "EPS",
    "physical_code_rate", "info_bits_count",
]

