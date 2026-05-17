"""Parallel multiprocessing helpers for offspring batch validation.

NOTE: The Python random-program seeder (``parallel_fill_random``) was
removed -- random program generation is now done exclusively by the C++
backend (``pushgp.cpp_seeder_adapter.cpp_parallel_fill_random``) which
is 5-10x faster and bit-equivalent.  Only the batch program validator
remains here.

Public API
----------
parallel_validate_programs(programs_serialized, side, *, workers, deg, base_seed)
    -> List[bool]                              # one True/False per input program
"""

from __future__ import annotations

import os
from multiprocessing import Pool
from typing import List, Optional

import numpy as np

from .genome import Instruction
from .serialize import program_to_dict, dict_to_program
from .validators import validate_c2v, validate_v2c

DEFAULT_WORKERS = max(1, (os.cpu_count() or 4) - 1)


# ============================================================ Worker tasks


def _validate_batch(args):
    """Worker: validate a batch of already-serialized programs.

    Returns parallel list of bools (True = valid).
    """
    side, programs_serialized, base_seed, deg = args
    if side == "v2c":
        val = validate_v2c
    elif side == "c2v":
        val = validate_c2v
    else:
        raise ValueError(side)
    out = []
    for i, dprog in enumerate(programs_serialized):
        prog = dict_to_program(dprog)
        sub_seed = (base_seed * 0x9E3779B9 + i) & 0xFFFFFFFF
        ok, _ = val(prog, rng=np.random.default_rng(sub_seed), deg=deg)
        out.append(bool(ok))
    return out


# ============================================================ Public API


def parallel_validate_programs(
    side: str,
    programs: List[List[Instruction]],
    *,
    workers: int = DEFAULT_WORKERS,
    deg: int = 8,
    base_seed: int = 0,
    pool: Optional[Pool] = None,
) -> List[bool]:
    """Validate a list of already-built programs in parallel.

    Useful for batch-validating offspring after mutation/crossover.
    Returns one bool per input program in the same order.
    """
    if not programs:
        return []
    serialized = [program_to_dict(p) for p in programs]
    n = len(serialized)
    n_chunks = max(1, min(workers * 4, n))
    chunk_size = (n + n_chunks - 1) // n_chunks
    jobs = []
    for c in range(n_chunks):
        lo = c * chunk_size
        hi = min(n, lo + chunk_size)
        if lo >= hi:
            break
        jobs.append((side, serialized[lo:hi], base_seed + c, deg))

    own_pool = pool is None
    if own_pool:
        pool = Pool(processes=workers)
    try:
        out_per_chunk = list(pool.imap(_validate_batch, jobs))
        result: List[bool] = []
        for chunk in out_per_chunk:
            result.extend(chunk)
        return result
    finally:
        if own_pool:
            pool.close()
            pool.join()


__all__ = [
    "parallel_validate_programs",
    "DEFAULT_WORKERS",
]
