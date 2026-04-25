"""IR-level genetic programming operators.

All operations work on FunctionIR objects from algorithm_ir.ir.model.
Mutations clone-first, then modify, then validate.

Phase H+4 S0.0 — Single-representation principle:
   This module MUST NOT import or call ``emit_python_source`` /
   ``compile_source_to_ir`` / ``ast.parse`` / ``compile()``.
   Mutations operate exclusively on the FunctionIR data structure.
   Python source materialization happens only at the evaluation
   boundary (evaluator.py). The legacy ``_mutate_via_recompile``
   path that did ``IR -> emit_python_source -> textual edit ->
   compile_source_to_ir -> IR`` was removed in Phase H+4 S0.0 because
   it violated this principle (and harbored a NameError in the
   ``swap`` branch).
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from algorithm_ir.ir.model import FunctionIR, Op, Value


# Binary op names that can be swapped
_BINARY_OPS = ["Add", "Sub", "Mult", "Div"]

# Compare op names
_COMPARE_OPS = ["Lt", "Gt", "LtE", "GtE", "Eq", "NotEq"]


def mutate_ir(
    func_ir: FunctionIR,
    rng: np.random.Generator,
    mutation_type: str | None = None,
) -> FunctionIR:
    """Mutate a FunctionIR. Returns a new (cloned) FunctionIR.

    Mutation types (pure IR, no source roundtrip):
      - "point" : swap binary opcode / compare opcode in-place
      - "constant_perturb" : Gaussian-perturb a numeric literal in-place

    If ``mutation_type`` is None, one is sampled uniformly.

    Phase H+4 S0.0 removed the ``insert``/``delete``/``swap_lines``
    modes because they were implemented via Python source roundtrip
    (see the new typed ``mut_insert_typed`` / ``mut_delete_typed`` /
    ``cx_block_typed`` operators in ``evolution.gp.operators`` for
    the IR-native replacements).
    """
    if mutation_type is None:
        mutation_type = rng.choice(["point", "constant_perturb"], p=[0.5, 0.5])

    # Clone first
    new_ir = copy.deepcopy(func_ir)

    if mutation_type == "point":
        return _mutate_point(new_ir, rng)
    elif mutation_type == "constant_perturb":
        return _mutate_constant_perturb(new_ir, rng)
    else:
        return _mutate_point(new_ir, rng)


def _mutate_point(func_ir: FunctionIR, rng: np.random.Generator) -> FunctionIR:
    """Point mutation: swap binary opcode, swap compare, or swap inputs."""
    # Collect mutable ops
    mutable = []
    for op in func_ir.ops.values():
        if op.opcode == "binary":
            mutable.append(("binary", op))
        elif op.opcode == "compare":
            mutable.append(("compare", op))

    if not mutable:
        return func_ir

    kind, op = mutable[rng.integers(len(mutable))]

    if kind == "binary":
        old = op.attrs.get("operator", "Add")
        choices = [o for o in _BINARY_OPS if o != old]
        if choices:
            op.attrs["operator"] = rng.choice(choices)

    elif kind == "compare":
        old_ops = op.attrs.get("operators", ["Lt"])
        old = old_ops[0] if old_ops else "Lt"
        choices = [o for o in _COMPARE_OPS if o != old]
        if choices:
            op.attrs["operators"] = [rng.choice(choices)]

    return func_ir


def _mutate_constant_perturb(
    func_ir: FunctionIR,
    rng: np.random.Generator,
    sigma: float = 0.3,
) -> FunctionIR:
    """Perturb a random constant literal by Gaussian noise."""
    const_ops = [
        op for op in func_ir.ops.values()
        if op.opcode == "const" and isinstance(op.attrs.get("literal"), (int, float))
        and not isinstance(op.attrs.get("literal"), bool)
    ]

    if not const_ops:
        return func_ir

    op = const_ops[rng.integers(len(const_ops))]
    old_val = float(op.attrs["literal"])
    new_val = old_val + rng.normal(0, sigma)
    op.attrs["literal"] = round(new_val, 6)

    return func_ir


# NOTE: ``_mutate_via_recompile`` was deleted in Phase H+4 S0.0 along with
# the ``insert``/``delete``/``swap_lines`` mutation modes that depended on
# it. Those modes performed a Python-source roundtrip
# (``emit_python_source`` -> textual edit -> ``compile_source_to_ir``)
# which violated the single-representation principle (see
# ``code_review/typed_gp_remediation_plan.md`` §10.0). Their semantics
# are now provided by the IR-native typed operators in
# ``evolution.gp.operators`` (``mut_insert_typed``, ``mut_delete_typed``,
# ``cx_block_typed``).


def crossover_ir(
    ir1: FunctionIR,
    ir2: FunctionIR,
    rng: np.random.Generator,
) -> FunctionIR:
    """Crossover two FunctionIRs.

    Strategy: take the structure of ir1 and replace constants/operators
    from ir2 where compatible. Safe because it preserves ir1's CFG.
    """
    child = copy.deepcopy(ir1)

    # Collect ops by type from ir2
    ir2_binary = [op for op in ir2.ops.values() if op.opcode == "binary"]
    ir2_consts = [
        op for op in ir2.ops.values()
        if op.opcode == "const" and isinstance(op.attrs.get("literal"), (int, float))
        and not isinstance(op.attrs.get("literal"), bool)
    ]

    # Replace some binary operators from ir2
    child_binary = [op for op in child.ops.values() if op.opcode == "binary"]
    for op in child_binary:
        if rng.random() < 0.5 and ir2_binary:
            donor = ir2_binary[rng.integers(len(ir2_binary))]
            op.attrs["operator"] = donor.attrs.get("operator", "Add")

    # Replace some constants from ir2
    child_consts = [
        op for op in child.ops.values()
        if op.opcode == "const" and isinstance(op.attrs.get("literal"), (int, float))
        and not isinstance(op.attrs.get("literal"), bool)
    ]
    for op in child_consts:
        if rng.random() < 0.5 and ir2_consts:
            donor = ir2_consts[rng.integers(len(ir2_consts))]
            op.attrs["literal"] = donor.attrs.get("literal", 0.0)

    return child
