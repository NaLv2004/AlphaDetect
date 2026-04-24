"""Smoke test for the two new integrations:
  1. types_lattice actively consulted by random_program
  2. const_lifter audit runs at skeleton_library load
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evolution.random_program import random_ir_program
from evolution.skeleton_library import get_extended_specs
from evolution.skeleton_registry import ProgramSpec


def test_random_program_with_tensor_return():
    """Forces types_lattice coercion path (vec_cx is not a subtype of float)."""
    spec = ProgramSpec(
        name="custom_tensor",
        param_names=["x", "y"],
        param_types=["vec_cx", "vec_cx"],
        return_type="vec_cx",
    )
    rng = np.random.default_rng(0)
    ir = random_ir_program(spec, rng, max_depth=2)
    assert ir is not None
    print(f"== random_ir_program(vec_cx) ops: {len(ir.ops)}")


def test_random_program_with_float_return():
    spec = ProgramSpec(
        name="custom_scalar",
        param_names=["x", "y"],
        param_types=["float", "float"],
        return_type="float",
    )
    rng = np.random.default_rng(0)
    ir = random_ir_program(spec, rng, max_depth=2)
    assert ir is not None
    print(f"== random_ir_program(float) ops: {len(ir.ops)}")


def test_const_lift_audit_runs():
    specs = get_extended_specs()
    print(f"== get_extended_specs: {len(specs)} specs")
    audit_path = ROOT / "results" / "const_lift_audit.json"
    assert audit_path.exists(), "audit file not written"
    data = json.loads(audit_path.read_text(encoding="utf-8"))
    totals = data.get("totals", {})
    print(f"== audit totals: {totals}")
    assert totals.get("templates", 0) > 0, "no templates had liftable literals"


if __name__ == "__main__":
    print("== test_random_program_with_tensor_return ==")
    test_random_program_with_tensor_return()
    print("== test_random_program_with_float_return ==")
    test_random_program_with_float_return()
    print("== test_const_lift_audit_runs ==")
    test_const_lift_audit_runs()
    print("\nALL CHECKS PASSED")
