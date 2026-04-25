"""R7: live demo — apply each R5 structural operator to a real genome
and print the IR diff so we can SEE the mutation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algorithm_ir.frontend.ir_builder import compile_source_to_ir
from algorithm_ir.ir.validator import validate_function_ir
from evolution.gp.contract import build_slot_contract
from evolution.gp.operators import OPERATOR_REGISTRY
from evolution.gp.operators.base import GPContext, run_operator_with_gates
from evolution.pool_types import AlgorithmGenome, SlotPopulation
from evolution.skeleton_registry import ProgramSpec


HOST_SRC = """
def f(x: vec_cx, y: vec_cx, snr_db: float):
    a = x + y
    b = a * x
    s = b + 1.0
    n = s * 2.0
    g = n - y
    h = g + a
    out = h * 0.5
    return out
"""


DONOR_SRC = """
def f(x: vec_cx, y: vec_cx, snr_db: float):
    p = x * y
    q = p + x
    r = q - y
    s = r * 0.25
    t = s + p
    u = t * 1.5
    return u
"""


def _short_summary(ir, label):
    lines = [f"--- {label}: {len(ir.ops)} ops ---"]
    for oid, op in ir.ops.items():
        ins = ",".join(getattr(op, "inputs", None) or [])
        outs = ",".join(getattr(op, "outputs", None) or [])
        lines.append(f"  {oid:8s} {op.opcode:18s}  ins=[{ins}]  outs=[{outs}]")
    return "\n".join(lines)


def main() -> int:
    parent_ir = compile_source_to_ir(HOST_SRC, "f")
    donor_ir = compile_source_to_ir(DONOR_SRC, "f")
    assert validate_function_ir(parent_ir) == []
    assert validate_function_ir(donor_ir) == []
    # Tag provenance so resolve_slot_region works.
    for op in parent_ir.ops.values():
        if not isinstance(op.attrs, dict):
            op.attrs = {}
        op.attrs.setdefault("_provenance", {})["from_slot_id"] = "_slot_body"

    spec = ProgramSpec(
        name="body", param_names=["x", "y", "snr_db"],
        param_types=["vec_cx", "vec_cx", "float"], return_type="vec_cx",
    )
    pop = SlotPopulation(slot_id="f.body", spec=spec)
    pop.variants = [parent_ir]
    pop.fitness = [0.1]
    pop.best_idx = 0
    g = AlgorithmGenome(algo_id="demo", ir=parent_ir, slot_populations={"f.body": pop})

    contract = build_slot_contract(pop, slot_key="f.body", complexity_cap=64)
    region_op_ids = frozenset(parent_ir.ops.keys())
    rng = np.random.default_rng(42)

    target_ops = [
        "mut_insert_typed",
        "mut_delete_typed",
        "mut_subtree_replace",
        "mut_primitive_inject",
        "cx_subtree_typed",
    ]

    print(_short_summary(parent_ir, "PARENT IR (before any R5 mutation)"))
    print()

    for name in target_ops:
        entry = OPERATOR_REGISTRY.get(name)
        if entry is None:
            print(f"!! operator {name} not registered\n")
            continue
        factory, _w, _is_cx = entry
        # Try several seeds — operators are stochastic and may early-out.
        success = False
        last_reason = None
        for trial in range(40):
            rng2 = np.random.default_rng(1000 + trial)
            ctx2 = GPContext(rng=rng2, contract=contract, region_op_ids=region_op_ids)
            parent2 = donor_ir if name.startswith("cx_") else None
            op_instance = factory()
            res = run_operator_with_gates(
                op_instance, ctx2, parent_ir, "demo_parent",
                parent2_ir=parent2,
            )
            child = res.child_ir if res is not None else None
            if res is not None and res.rejection_reason:
                last_reason = res.rejection_reason
            if child is None:
                continue
            if len(child.ops) == len(parent_ir.ops):
                # Maybe a same-size structural change still differs.
                same_keys = all(
                    parent_ir.ops.get(k) is not None
                    and parent_ir.ops[k].opcode == child.ops[k].opcode
                    for k in child.ops
                )
                if same_keys:
                    continue
            print(f"### {name} (trial={trial}): {len(parent_ir.ops)} -> {len(child.ops)} ops")
            print(_short_summary(child, f"CHILD via {name}"))
            print()
            success = True
            break
        if not success:
            print(f"### {name}: no structurally-distinct child after 40 trials (last_reason={last_reason})\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
