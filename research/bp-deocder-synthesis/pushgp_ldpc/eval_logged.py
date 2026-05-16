"""Variant of `evaluate_genome` that returns per-SNR BER and FER as well
as the scalar fitness, so we can log decoder performance per individual
without having to evaluate twice.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from pushgp.genome import Genome
from pushgp_ldpc.adapter import make_callables
from pushgp_ldpc.eval import EPS, FitnessConfig, _channel_inputs

from ldpc_5g import decode_bp


@dataclass
class GenomeMetrics:
    fitness: float                 # mean log10(BER+EPS); smaller better
    ber_per_snr: List[float]       # one BER per cfg.snr_list
    fer_per_snr: List[float]       # one FER per cfg.snr_list
    n_frames_per_snr: int
    valid: bool                    # False on adapter/decoder exception
    error: str = ""


_BAD = GenomeMetrics(fitness=6.0, ber_per_snr=[], fer_per_snr=[],
                     n_frames_per_snr=0, valid=False, error="adapter")


def evaluate_genome_with_ber(genome: Genome, cfg: FitnessConfig) -> GenomeMetrics:
    # Optional C++ fast path. Behaviourally equivalent to the Python
    # loop below (verified by code_review/smoke_cpp_fitness_equiv.py).
    if getattr(cfg, "use_cpp_fitness", False):
        from pushgp_ldpc.eval_cpp import evaluate_genome_cpp_ber
        return evaluate_genome_cpp_ber(genome, cfg)

    try:
        v2c_fn, c2v_fn = make_callables(genome)
    except Exception as e:  # noqa: BLE001
        return GenomeMetrics(6.0, [float("nan")] * len(cfg.snr_list),
                             [float("nan")] * len(cfg.snr_list),
                             cfg.n_frames_per_snr, False, f"adapter:{e!r}")

    bers: List[float] = []
    fers: List[float] = []
    log_bers: List[float] = []

    for snr_idx, snr_db in enumerate(cfg.snr_list):
        pairs = _channel_inputs(cfg, snr_db)
        n_err = 0
        n_bits = 0
        n_frame_err = 0
        for bits, llr in pairs:
            try:
                post = decode_bp(
                    llr, cfg.par,
                    v2c_fn=v2c_fn, c2v_fn=c2v_fn,
                    max_iter=cfg.max_iter, offset=0.25,
                    code_rate=cfg.code_rate,
                )
            except Exception as e:  # noqa: BLE001
                # Pad with worst values for remaining SNRs.
                bers.extend([float("nan")] * (len(cfg.snr_list) - snr_idx))
                fers.extend([float("nan")] * (len(cfg.snr_list) - snr_idx))
                return GenomeMetrics(6.0, bers, fers, cfg.n_frames_per_snr,
                                     False, f"decode:{e!r}")
            hat = (post < 0.0).astype(np.int8)
            errs = int((hat != bits).sum())
            n_err += errs
            n_bits += bits.size
            if errs > 0:
                n_frame_err += 1
        ber = n_err / max(1, n_bits)
        fer = n_frame_err / max(1, len(pairs))
        bers.append(ber)
        fers.append(fer)
        log_bers.append(float(np.log10(ber + EPS)))
        if snr_idx == 0 and ber > cfg.early_fail_threshold:
            # Fill remaining SNRs with the same bad values.
            n_remain = len(cfg.snr_list) - 1
            bers.extend([ber] * n_remain)
            fers.extend([fer] * n_remain)
            log_bers.extend([log_bers[0]] * n_remain)
            break

    fit = float(np.mean(log_bers))
    return GenomeMetrics(fit, bers, fers, cfg.n_frames_per_snr, True)


__all__ = ["GenomeMetrics", "evaluate_genome_with_ber"]
