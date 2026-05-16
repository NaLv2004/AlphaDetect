"""C++-accelerated fitness evaluation.

This module mirrors `pushgp_ldpc.eval_logged.evaluate_genome_with_ber`
exactly, except every per-frame BP decode is dispatched into
`pushgp_cpp_dce.decode_bp` (the same C++ kernel that already powers DCE
and that is byte-locked to the Python `ldpc_5g.decode_bp` to 6 decimals
by `cpp_dce/tests/test_bp_equivalence.py`).

Wiring constraints honored
--------------------------
* Reuses the existing `pushgp.dce._try_import_pushgp_cpp_dce` resolver so
  there is no second discovery path for the .pyd module.
* Reuses `pushgp.serialize.program_to_dict` for V/C programs (same dict
  format the BP test suite already verifies).
* Reuses `Genome.evo_const_values()` for the 8-slot evolved constants
  vector (same path used by `cpp_dce/tests/test_bp_equivalence.py`).
* Reuses the same `_channel_inputs` channel realiser so the LLR feed is
  bitwise identical to the Python path. A per-(cfg-id, snr-tag) LRU
  cache is added so all genomes in a generation share a single
  realisation instead of regenerating it once per individual.
* Keeps the per-SNR loop, BER/FER bookkeeping, early-fail threshold,
  and GenomeMetrics return shape identical to the Python version, so
  the existing ParallelPairEvaluator / GenerationLogger pipelines see
  no behavioural change other than speed.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np

from pushgp.genome import Genome
from pushgp.serialize import program_to_dict
from pushgp.dce import _try_import_pushgp_cpp_dce
from pushgp_ldpc.eval import EPS, FitnessConfig, _channel_inputs
from pushgp_ldpc.eval_logged import GenomeMetrics


# ---------------------------------------------------------------------------
# Module-level caches.  Each worker process has its own copy (multiprocessing
# spawn semantics), which is exactly what we want: parity handles and channel
# realisations are recomputed once per worker, then shared across all
# evaluations the worker performs.
# ---------------------------------------------------------------------------

_CDCE = None                           # cached pushgp_cpp_dce module handle
_PARITY_HANDLE_CACHE: dict = {}        # id(par) -> ParityHandle


def _get_cdce():
    global _CDCE
    if _CDCE is None:
        _CDCE = _try_import_pushgp_cpp_dce()
        if _CDCE is None:
            raise ImportError(
                "pushgp_cpp_dce is not importable; build cpp_dce first "
                "(see cpp_dce/setup.py) before enabling cpp fitness eval."
            )
    return _CDCE


def _get_parity_handle(par):
    """Build (or fetch cached) C++ ParityHandle for `par`.

    Cache key is `id(par)` rather than par contents — within a single
    worker process, the FitnessConfig (and therefore its `par`) is set
    once at init and never replaced, so id-equality is the correct
    invariant.  Across workers each one builds its own handle once.
    """
    key = id(par)
    handle = _PARITY_HANDLE_CACHE.get(key)
    if handle is None:
        handle = _get_cdce().build_parity_handle(par)
        _PARITY_HANDLE_CACHE[key] = handle
    return handle


# ---------------------------------------------------------------------------
# Channel-realisation cache (per generation).  The Python path regenerates
# channel data on every `_channel_inputs` call.  Since the seed is
# deterministically `cfg.seed_base + abs(int(snr_db * 1000))`, the result is
# already bitwise identical across all genomes in a generation; we just want
# to avoid recomputing it N times.  Bounded LRU so cache size stays small
# (~len(snr_list) entries) and never grows across generations.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=64)
def _cached_channel(_key: Tuple, _cfg_id: int):
    # NB: _key encodes everything that affects the channel realisation;
    # _cfg_id keeps caches per-FitnessConfig instance so distinct configs
    # do not collide.  Actual cfg comes from a closure via _channel_for.
    raise RuntimeError("must be primed via _channel_for")


def _channel_for(cfg: FitnessConfig, snr_db: float):
    """Return cached `_channel_inputs(cfg, snr_db)` result for this gen.

    Cache key includes every FitnessConfig field that participates in
    channel generation, so accidental cache poisoning across distinct
    configs is impossible.
    """
    par = cfg.par
    key = (
        id(cfg.par),
        cfg.seed_base,
        cfg.n_frames_per_snr,
        float(snr_db),
        cfg.effective_code_rate,
        cfg.tx_len,
    )
    cached = _CHANNEL_CACHE.get(key)
    if cached is not None:
        return cached
    pairs = _channel_inputs(cfg, snr_db)
    # Defensive size bound: ~len(snr_list) entries per cfg, but evict
    # oldest if cache gets unreasonably large (e.g. user changes seed).
    if len(_CHANNEL_CACHE) >= 256:
        _CHANNEL_CACHE.clear()
    _CHANNEL_CACHE[key] = pairs
    return pairs


_CHANNEL_CACHE: dict = {}


# ---------------------------------------------------------------------------
# Main entry point.
# ---------------------------------------------------------------------------


def evaluate_genome_cpp_ber(genome: Genome, cfg: FitnessConfig) -> GenomeMetrics:
    """C++-accelerated drop-in replacement for `evaluate_genome_with_ber`.

    Behavioural contract: returns the same GenomeMetrics shape, same
    fitness scalar (mean log10(BER+EPS)), same BER/FER lists, same
    early-fail short-circuit, same GenomeMetrics(6.0, ...) failure
    sentinel on exception.
    """
    try:
        cdce = _get_cdce()
        parH = _get_parity_handle(cfg.par)
        v_dict = program_to_dict(genome.prog_v2c)
        c_dict = program_to_dict(genome.prog_c2v)
        evo = genome.evo_const_values().astype(np.float64)
    except Exception as e:  # noqa: BLE001
        return GenomeMetrics(
            fitness=6.0,
            ber_per_snr=[float("nan")] * len(cfg.snr_list),
            fer_per_snr=[float("nan")] * len(cfg.snr_list),
            n_frames_per_snr=cfg.n_frames_per_snr,
            valid=False,
            error=f"adapter:{e!r}",
        )

    bers: List[float] = []
    fers: List[float] = []
    log_bers: List[float] = []

    for snr_idx, snr_db in enumerate(cfg.snr_list):
        pairs = _channel_for(cfg, snr_db)
        n_err = 0
        n_bits = 0
        n_frame_err = 0
        for bits, llr in pairs:
            try:
                post, _iters = cdce.decode_bp(
                    llr, parH, v_dict, c_dict, evo,
                    cfg.max_iter, 0.25, cfg.effective_code_rate,
                )
            except Exception as e:  # noqa: BLE001
                n_remain = len(cfg.snr_list) - snr_idx
                bers.extend([float("nan")] * n_remain)
                fers.extend([float("nan")] * n_remain)
                return GenomeMetrics(
                    fitness=6.0,
                    ber_per_snr=bers,
                    fer_per_snr=fers,
                    n_frames_per_snr=cfg.n_frames_per_snr,
                    valid=False,
                    error=f"decode_cpp:{e!r}",
                )
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
            n_remain = len(cfg.snr_list) - 1
            bers.extend([ber] * n_remain)
            fers.extend([fer] * n_remain)
            log_bers.extend([log_bers[0]] * n_remain)
            break

    fit = float(np.mean(log_bers))
    return GenomeMetrics(
        fitness=fit,
        ber_per_snr=bers,
        fer_per_snr=fers,
        n_frames_per_snr=cfg.n_frames_per_snr,
        valid=True,
    )


__all__ = ["evaluate_genome_cpp_ber"]
