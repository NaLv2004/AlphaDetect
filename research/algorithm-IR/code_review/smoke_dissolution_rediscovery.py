"""Smoke test: import + invoke the new dissolution + rediscovery modules
on a synthetic genome.  Shouldn't crash; should print a one-line summary.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    print("== smoke_dissolution_rediscovery ==")
    try:
        from evolution.types_lattice import (
            is_subtype, unify, available_ops_for_type, default_value,
            infer_value_type,
        )
    except Exception as exc:
        print(f"FAIL types_lattice import: {exc!r}")
        traceback.print_exc()
        return 1
    print("  types_lattice import: OK")

    # Quick sanity checks for the lattice
    assert is_subtype("int", "float"), "int <: float should hold"
    assert is_subtype("vec_i", "vec_cx"), "vec_i <: vec_cx should hold"
    assert unify("int", "float") == "float"
    assert unify("vec_f", "vec_cx") == "vec_cx"
    assert unify("vec_i", "mat_f") == "any"
    assert unify("tuple<int,float>", "tuple<float,float>") == "tuple<float,float>"
    assert "call" in available_ops_for_type("vec_f")
    import numpy as np
    arr = default_value("vec_cx")
    assert arr.dtype == np.complex128 and arr.shape == (4,)
    assert infer_value_type(np.zeros(8, dtype=np.complex128)) == "vec_cx"
    assert infer_value_type([1, 2, 3]) == "list<int>"
    assert infer_value_type((1, 2.0)) == "tuple<int,float>"
    print("  types_lattice sanity: OK")

    try:
        from evolution.fii import strip_provenance_markers
        from evolution.slot_rediscovery import (
            rediscover_slots, NewSlotProposal,
            apply_rediscovered_slots, maybe_rediscover_slots,
        )
    except Exception as exc:
        print(f"FAIL dissolution/rediscovery import: {exc!r}")
        traceback.print_exc()
        return 1
    print("  slot_dissolution + slot_rediscovery import: OK")

    # Build one detector genome and run rediscovery on its structural_ir
    try:
        from evolution.ir_pool import build_ir_pool
        pool = build_ir_pool()
        if not pool:
            print("FAIL build_ir_pool returned empty")
            return 1
        genome = pool[0]
        print(f"  seeded genome: {genome.algo_id} "
              f"slot_pops={len(genome.slot_populations)}")

        # Strip slot pops to mimic full dissolution, then rediscover
        flat = genome.clone()
        flat.slot_populations = {}
        proposals = rediscover_slots(
            flat.structural_ir,
            min_size=4, max_size=24,
            max_boundary=4, min_cohesion=1.5,
            max_new_per_pass=3,
        )
        print(f"  rediscover_slots: {len(proposals)} proposals")
        for p in proposals[:3]:
            print(f"    - {p.slot_id}: n_ops={len(p.op_ids)} "
                  f"cohesion={p.cohesion:.2f} "
                  f"in/out=({len(p.entry_values)},{len(p.exit_values)})")
        apply_rediscovered_slots(flat, proposals)
        print(f"  apply_rediscovered_slots: slot_pops={len(flat.slot_populations)}")
    except Exception as exc:
        print(f"FAIL rediscovery on real genome: {exc!r}")
        traceback.print_exc()
        return 1

    # strip_provenance_markers no-op test (no markers)
    try:
        ir2 = strip_provenance_markers(genome.structural_ir)
        n_ops_before = len(genome.structural_ir.ops)
        n_ops_after = len(ir2.ops)
        print(f"  strip_provenance_markers: ops {n_ops_before} -> {n_ops_after}")
    except Exception as exc:
        print(f"FAIL strip_provenance_markers: {exc!r}")
        traceback.print_exc()
        return 1

    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
