"""Parallel pair-fitness evaluation for two-pop evolution.

We avoid pickling the FitnessConfig per task by using a Pool initializer
that stores it in a worker-side global. Each task only ships the
serialized programs and the log_constants array.
"""
from __future__ import annotations

from multiprocessing import Pool
from typing import List, Tuple, Any

import numpy as np

from pushgp.serialize import dict_to_program, program_to_dict
from pushgp.genome import Genome


# ----- worker globals (one per process) -----
_WORKER_FIT_CFG = None


def _init_worker(fit_cfg_pickle: bytes) -> None:
    """Pool initializer: reconstruct FitnessConfig once per worker."""
    import pickle
    global _WORKER_FIT_CFG
    _WORKER_FIT_CFG = pickle.loads(fit_cfg_pickle)


def _eval_one(args: Tuple[List[dict], List[dict], List[float]]) -> Tuple[
    float, List[float], List[float], int, bool, str
]:
    """Worker: evaluate a single (V2C, C2V, k) triple."""
    from pushgp_ldpc.eval_logged import evaluate_genome_with_ber
    v_dict, c_dict, k_list = args
    g = Genome(
        prog_v2c=dict_to_program(v_dict),
        prog_c2v=dict_to_program(c_dict),
        log_constants=np.asarray(k_list, dtype=np.float64),
    )
    m = evaluate_genome_with_ber(g, _WORKER_FIT_CFG)
    return (
        float(m.fitness),
        [float(x) for x in m.ber_per_snr],
        [float(x) for x in m.fer_per_snr],
        int(m.n_frames_per_snr),
        bool(m.valid),
        str(m.error or ""),
    )


def make_eval_pool(fit_cfg, n_workers: int) -> Pool:
    """Spawn a pool with the FitnessConfig pre-loaded in each worker."""
    import pickle
    payload = pickle.dumps(fit_cfg)
    return Pool(
        processes=max(1, n_workers),
        initializer=_init_worker,
        initargs=(payload,),
    )


class ParallelPairEvaluator:
    """Reusable pair-fitness evaluator backed by a persistent Pool.

    Also fills `metrics_by_id` after each call so the GenerationLogger
    can recover full BER/FER without re-running anything.
    """

    def __init__(self, fit_cfg, n_workers: int):
        self.fit_cfg = fit_cfg
        self.n_workers = max(1, n_workers)
        self.pool = make_eval_pool(fit_cfg, self.n_workers)
        # Per-batch cache: maps id(genome) -> GenomeMetrics-like object
        self.last_metrics: List[Any] = []

    def close(self) -> None:
        try:
            self.pool.close()
            self.pool.join()
        except Exception:
            pass

    def eval_pairs(self, pop_v, pop_c, pop_k, perm) -> List[float]:
        """Evaluate `len(pop_c)` triples (v[perm[i]], c[i], k[i]). Returns list of fits."""
        n = len(pop_c)
        jobs = [
            (
                program_to_dict(pop_v[perm[i]]),
                program_to_dict(pop_c[i]),
                pop_k[i].tolist(),
            )
            for i in range(n)
        ]
        # imap preserves order; chunksize tuned for ~few seconds per task
        results = list(self.pool.imap(_eval_one, jobs, chunksize=1))
        # Build lightweight metrics records
        from pushgp_ldpc.eval_logged import GenomeMetrics
        self.last_metrics = []
        fits: List[float] = []
        for fit, ber, fer, n_fr, valid, err in results:
            fits.append(fit)
            self.last_metrics.append(GenomeMetrics(
                fitness=fit, ber_per_snr=ber, fer_per_snr=fer,
                n_frames_per_snr=n_fr, valid=valid,
                error=err if err else None,
            ))
        return fits
