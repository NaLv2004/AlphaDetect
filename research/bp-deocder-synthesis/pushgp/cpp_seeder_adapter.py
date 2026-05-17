"""Adapter that exposes the C++ `parallel_seed` to Python with the same
shape used by `evolve_from_scratch`.  Always the sole production seeder:
the Python multiprocessing seeder has been removed.

Returns `(programs, total_attempts)`, where `programs` is a list of
`Instruction` lists (decoded from the C++ ProgramHandle via `to_dict` →
`pushgp.serialize.dict_to_program`).

Raises `RuntimeError` if `n_target` is not reached within `max_attempts`.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from .genome import Instruction
from .op_filter import OpFilter
from .serialize import dict_to_program


def _load_module():
    """Import the C++ extension; raise immediately if missing (no fallback)."""
    import sys
    from pathlib import Path
    cpp_dir = Path(__file__).resolve().parent.parent / "cpp_seeder"
    if str(cpp_dir) not in sys.path:
        sys.path.insert(0, str(cpp_dir))
    try:
        import pushgp_cpp_seeder as M  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "pushgp_cpp_seeder is not built. Run "
            "`python setup.py build_ext --inplace` in "
            "research/bp-deocder-synthesis/cpp_seeder/."
        ) from e
    return M


def cpp_parallel_fill_random(
    side: str,
    n_target: int,
    *,
    max_attempts: int = 10_000_000,
    workers: int = 8,
    chunk_attempts: int = 5000,
    min_size: int = 4,
    max_size: int = 16,
    deg: int = 8,
    base_seed: int = 0,
    progress_cb=None,
    pool: Optional[Any] = None,            # ignored; signature compatibility
    seen_fingerprints: Optional[set] = None,  # dedup-as-validation
    op_filter: Optional[OpFilter] = None,
) -> Tuple[List[List[Instruction]], int]:
    """Fill `n_target` valid programs of `side` using the C++ seeder.

    `op_filter`: when provided and active, restricts the random program
    generator to the whitelisted opcodes for this side.  The whitelist
    is intersected with the side's base set on the C++ side (so c2v
    still excludes Env_GetChannelLLR even if you list it).

    If `seen_fingerprints` is provided, the C++ side rejects any
    candidate whose 32-entry behavioral fingerprint is already in the
    set; on success, the returned new fingerprints are inserted into
    the same set (in-place) so callers see the updated state.
    """
    M = _load_module()
    seen_in: List[str] = (
        list(seen_fingerprints) if seen_fingerprints is not None else []
    )
    allowed: List[str] = []
    if op_filter is not None and op_filter.applies():
        names = op_filter.v2c if side == "v2c" else op_filter.c2v
        if names is not None:
            allowed = sorted(names)
    handles, attempts, fps = M.parallel_seed(
        side=side,
        n_target=n_target,
        max_attempts=int(max_attempts),
        threads=int(workers),
        chunk_attempts=int(chunk_attempts),
        min_size=int(min_size),
        max_size=int(max_size),
        deg=int(deg),
        num_configs=3,
        num_permutations=5,
        base_seed=int(base_seed) & 0xFFFFFFFFFFFFFFFF,
        progress_cb=progress_cb,
        seen_fingerprints=seen_in,
        allowed_op_names=allowed,
    )
    if len(handles) < n_target:
        raise RuntimeError(
            f"cpp_parallel_fill_random({side}) exhausted (after dedup): "
            f"got {len(handles)}/{n_target} valid in {attempts} attempts"
        )
    progs: List[List[Instruction]] = []
    for h in handles[:n_target]:
        progs.append(dict_to_program(h.to_dict()))
    if seen_fingerprints is not None:
        for fp in fps[:n_target]:
            seen_fingerprints.add(fp)
    return progs, int(attempts)


__all__ = ["cpp_parallel_fill_random"]
