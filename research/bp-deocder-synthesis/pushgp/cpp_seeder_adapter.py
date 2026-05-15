"""Adapter that exposes the C++ `parallel_seed` with the same signature as
`pushgp.parallel_init.parallel_fill_random`, so `evolve_from_scratch` can
swap implementations transparently.

Returns `(programs, total_attempts)`, where `programs` is a list of
`Instruction` lists (decoded from the C++ ProgramHandle via `to_dict` →
`pushgp.serialize.dict_to_program`).

Raises `RuntimeError` if `n_target` is not reached within `max_attempts`,
matching the Python side's contract.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from .genome import Instruction
from .serialize import dict_to_program

# Lazy import so the rest of the package still works if the .pyd is absent.
def _load_module():
    try:
        import sys
        from pathlib import Path
        # cpp_seeder/ lives next to pushgp/.
        cpp_dir = Path(__file__).resolve().parent.parent / "cpp_seeder"
        if str(cpp_dir) not in sys.path:
            sys.path.insert(0, str(cpp_dir))
        import pushgp_cpp_seeder as M  # type: ignore
        return M
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "pushgp_cpp_seeder is not built. Run "
            "`python setup.py build_ext --inplace` in research/bp-deocder-synthesis/cpp_seeder/."
        ) from e


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
    seen_fingerprints: Optional[set] = None,  # dedup-as-validation (Phase B)
) -> Tuple[List[List[Instruction]], int]:
    """Fill `n_target` valid programs of `side` using the C++ seeder.

    If `seen_fingerprints` is provided, the C++ side rejects any
    candidate whose 32-entry behavioral fingerprint is already in the
    set; on success, the returned new fingerprints are inserted into
    the same set (in-place) so callers see the updated state.
    """
    M = _load_module()
    seen_in: List[str] = (
        list(seen_fingerprints) if seen_fingerprints is not None else []
    )
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
