"""IR-level genetic programming operators.

All operations work on FunctionIR objects from algorithm_ir.ir.model.
Mutations clone-first, then modify, then validate.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from algorithm_ir.ir.model import FunctionIR, Op, Value
from algorithm_ir.frontend.ir_builder import compile_source_to_ir
from algorithm_ir.regeneration.codegen import emit_python_source

from evolution.config import EvolutionConfig


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

    Mutation types: "point", "constant_perturb", "insert", "delete".
    If mutation_type is None, one is chosen randomly.
    """
    if mutation_type is None:
        mutation_type = rng.choice(["point", "point", "constant_perturb", "constant_perturb"])

    # Clone first
    new_ir = copy.deepcopy(func_ir)

    if mutation_type == "point":
        return _mutate_point(new_ir, rng)
    elif mutation_type == "constant_perturb":
        return _mutate_constant_perturb(new_ir, rng)
    elif mutation_type == "insert":
        return _mutate_via_recompile(new_ir, rng, "insert")
    elif mutation_type == "delete":
        return _mutate_via_recompile(new_ir, rng, "delete")
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


def _mutate_via_recompile(
    func_ir: FunctionIR,
    rng: np.random.Generator,
    action: str,
) -> FunctionIR:
    """Insert or delete by modifying Python source and recompiling.

    This is a safe approach: regenerate source → edit → recompile.
    If recompilation fails, return original unchanged.
    """
    try:
        source = emit_python_source(func_ir)
        lines = source.split("\n")

        if action == "insert" and len(lines) >= 2:
            # Insert a random assignment at a random position
            # Simple: add "x_tmp = param1 + param2" type line
            indent = "    "
            # Pick a random insertion point (after def line)
            pos = rng.integers(1, max(2, len(lines)))
            param_names = []
            for vid in func_ir.arg_values:
                v = func_ir.values[vid]
                name = v.attrs.get("var_name") or v.name_hint or "x"
                param_names.append(name)
            if param_names:
                a = rng.choice(param_names)
                b = rng.choice(param_names)
                op = rng.choice(["+", "-", "*"])
                new_line = f"{indent}_tmp = {a} {op} {b}"
                lines.insert(pos, new_line)

        elif action == "delete" and len(lines) > 2:
            # Delete a random non-def, non-return line
            deletable = [
                i for i, line in enumerate(lines)
                if i > 0 and "return" not in line and line.strip()
            ]
            if deletable:
                idx = rng.choice(deletable)
                lines.pop(idx)

        new_source = "\n".join(lines)
        namespace: dict[str, Any] = {"__builtins__": __builtins__}

        def _safe_div(a, b): return a / b if abs(b) > 1e-30 else 0.0
        def _safe_log(a):
            import math; return math.log(max(a, 1e-30))
        def _safe_sqrt(a):
            import math; return math.sqrt(max(a, 0.0))

        namespace["_safe_div"] = _safe_div
        namespace["_safe_log"] = _safe_log
        namespace["_safe_sqrt"] = _safe_sqrt

        return compile_source_to_ir(new_source, func_ir.name, namespace)
    except Exception:
        return func_ir


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


def mutate_genome(
    genome,  # IRGenome (avoid circular import)
    config: EvolutionConfig,
    rng: np.random.Generator,
) -> None:
    """Mutate a genome in-place: mutate one program + optionally constants."""
    roles = list(genome.programs.keys())
    if not roles:
        return

    # Mutate one random program
    role = rng.choice(roles)
    genome.programs[role] = mutate_ir(genome.programs[role], rng)

    # Optionally perturb constants
    if len(genome.constants) > 0 and rng.random() < 0.5:
        idx = rng.integers(len(genome.constants))
        genome.constants[idx] += rng.normal(0, config.constant_mutate_sigma)
        # Clamp to range
        lo, hi = config.constant_range
        genome.constants[idx] = np.clip(genome.constants[idx], lo, hi)

    genome.invalidate_cache()


def crossover_genome(
    g1,  # IRGenome
    g2,  # IRGenome
    config: EvolutionConfig,
    rng: np.random.Generator,
):  # -> IRGenome
    """Crossover two genomes. Returns a new genome."""
    child = g1.clone()

    # Per-role crossover
    for role in child.programs:
        if role in g2.programs and rng.random() < 0.5:
            child.programs[role] = crossover_ir(
                child.programs[role], g2.programs[role], rng
            )

    # Constant blending
    if len(child.constants) > 0 and len(g2.constants) > 0:
        alpha = rng.uniform(0.3, 0.7)
        n = min(len(child.constants), len(g2.constants))
        child.constants[:n] = (
            alpha * child.constants[:n] + (1 - alpha) * g2.constants[:n]
        )

    child.parent_ids = [g1.genome_id, g2.genome_id]
    child.invalidate_cache()
    return child
