"""Parallel multiprocessing helpers for random program init + offspring
batch validation.

Two independent program populations (V2C, C2V) are filled by sampling
purely random programs and accepting only those that pass
`validate_v2c` / `validate_c2v`.  No structural priors, no biased
sampling — exactly the same `RandomProgramGenerator` used elsewhere.

Workers communicate by serializing programs to dicts (top-level
`program_to_list`), so all process boundaries are pickle-safe.

Public API
----------
parallel_fill_random_v2c(n_target, *, max_attempts, workers, ...) -> List[List[Instruction]]
parallel_fill_random_c2v(n_target, *, max_attempts, workers, ...) -> List[List[Instruction]]
parallel_validate_programs(programs_serialized, side, *, workers, deg, base_seed)
    -> List[bool]                              # one True/False per input program
"""

from __future__ import annotations

import os
from multiprocessing import Pool
from typing import List, Optional, Tuple

import numpy as np

from .genome import Instruction
from .random_program import RandomProgramGenerator
from .serialize import program_to_dict, dict_to_program
from .validators import validate_c2v, validate_v2c

DEFAULT_WORKERS = max(1, (os.cpu_count() or 4) - 1)


# ============================================================ Worker tasks


def _gen_and_validate_chunk(args):
    """Worker: generate `n_attempts` random programs of `side`, return
    serialized valid ones.

    Args tuple: (side, seed, n_attempts, min_size, max_size, deg)
    """
    side, seed, n_attempts, min_size, max_size, deg = args
    rng = np.random.default_rng(seed)
    rpg = RandomProgramGenerator(rng=rng)
    if side == "v2c":
        gen = rpg.random_v2c
        val = validate_v2c
    elif side == "c2v":
        gen = rpg.random_c2v
        val = validate_c2v
    else:
        raise ValueError(side)

    valid = []
    for i in range(n_attempts):
        prog = gen(min_size=min_size, max_size=max_size)
        sub_seed = (seed * 0x9E3779B9 + i) & 0xFFFFFFFF
        ok, _ = val(prog, rng=np.random.default_rng(sub_seed), deg=deg)
        if ok:
            valid.append(program_to_dict(prog))
    return side, valid, n_attempts


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


def parallel_fill_random(
    side: str,
    n_target: int,
    *,
    max_attempts: int = 10_000_000,
    workers: int = DEFAULT_WORKERS,
    chunk_attempts: int = 5000,
    min_size: int = 4,
    max_size: int = 16,
    deg: int = 8,
    base_seed: int = 0,
    progress_cb=None,
    pool: Optional[Pool] = None,
) -> Tuple[List[List[Instruction]], int]:
    """Generate `n_target` valid random programs of `side` (v2c or c2v).

    Returns `(programs, total_attempts)`.  Raises RuntimeError if
    max_attempts exceeded before `n_target` reached.

    `progress_cb(side, n_valid, n_attempts, elapsed_s)` invoked after
    every chunk wave (optional).

    If `pool` is given it is used (caller manages lifecycle); otherwise
    a fresh Pool is created and torn down here.
    """
    own_pool = pool is None
    if own_pool:
        pool = Pool(processes=workers)
    try:
        valid: List[List[Instruction]] = []
        attempts = 0
        seed_counter = base_seed
        import time as _time
        t0 = _time.time()
        while len(valid) < n_target and attempts < max_attempts:
            jobs = []
            for _ in range(workers):
                seed_counter += 1
                jobs.append((side, seed_counter, chunk_attempts,
                             min_size, max_size, deg))
            for _side, vlist, n_att in pool.imap_unordered(
                _gen_and_validate_chunk, jobs
            ):
                attempts += n_att
                for dprog in vlist:
                    valid.append(dict_to_program(dprog))
            if progress_cb is not None:
                progress_cb(side, len(valid), attempts, _time.time() - t0)
        if len(valid) < n_target:
            raise RuntimeError(
                f"parallel_fill_random({side}) exhausted: got {len(valid)}/{n_target} "
                f"valid in {attempts} attempts"
            )
        return valid[:n_target], attempts
    finally:
        if own_pool:
            pool.close()
            pool.join()


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
    "parallel_fill_random",
    "parallel_validate_programs",
    "DEFAULT_WORKERS",
]
