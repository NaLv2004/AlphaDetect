"""Sanity check: _state_arity_fast must agree with the legacy 5-helper
composition over real donor IRs.

Run:
    python code_review/verify_state_arity_equiv.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithm_ir.region.slicer import (
    backward_slice_until_values,
    enumerate_observable_values,
    enumerate_cut_candidates,
)
from evolution.host_region_mask import (
    _entry_values, _exit_values, _nontrivial_op_count, is_op_set_connected,
)
from evolution.donor_region_mask import _state_arity_fast
from evolution.ir_pool import build_ir_pool


def legacy(ir, ops):
    if not ops:
        return (0, 0, 0, 0, False)
    return (
        len(_entry_values(ir, ops)),
        len(_exit_values(ir, ops)),
        _nontrivial_op_count(ir, ops),
        len(_entry_values(ir, ops)),
        is_op_set_connected(ir, ops),
    )


def main() -> int:
    rng = random.Random(0)
    pool = build_ir_pool()
    n_checks = 0
    n_mismatch = 0
    for genome in pool:
        ir = genome.structural_ir
        if ir is None:
            continue
        obs = enumerate_observable_values(ir)
        if not obs:
            continue
        # Sample a handful of (output, cuts) configurations.
        for _ in range(20):
            k_out = rng.randint(1, min(3, len(obs)))
            outs = rng.sample(obs, k_out)
            cuts_pool = enumerate_cut_candidates(ir, outs)
            if cuts_pool:
                k_cut = rng.randint(0, min(3, len(cuts_pool)))
                cuts = rng.sample(cuts_pool, k_cut)
            else:
                cuts = []
            ops = backward_slice_until_values(ir, outs, cuts)
            a = legacy(ir, ops)
            b = _state_arity_fast(ir, ops)
            n_checks += 1
            if a != b:
                n_mismatch += 1
                print(f"MISMATCH: legacy={a} fast={b} | ir={genome.algorithm_id} outs={outs} cuts={cuts}")
    print(f"checked={n_checks} mismatches={n_mismatch}")
    return 0 if n_mismatch == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
