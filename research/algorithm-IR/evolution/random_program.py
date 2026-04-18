"""Random IR program generation from skeleton specification.

Generates random Python function source matching a ProgramSpec,
then compiles to FunctionIR via algorithm_ir.frontend.ir_builder.
"""

from __future__ import annotations

import textwrap
from typing import Any

import numpy as np

from algorithm_ir.frontend.ir_builder import compile_source_to_ir
from algorithm_ir.ir.model import FunctionIR
from evolution.skeleton_registry import ProgramSpec


# Available binary operators (safe for numerical evolution)
_BINARY_OPS = ["+", "-", "*"]
_SAFE_BINARY_OPS = ["+", "-", "*", "_safe_div"]

# Available unary operators
_UNARY_OPS = ["abs", "-", "_safe_sqrt", "_safe_log"]


def random_ir_program(
    spec: ProgramSpec,
    rng: np.random.Generator,
    max_depth: int = 4,
    use_safe_ops: bool = True,
) -> FunctionIR:
    """Generate a random FunctionIR matching the given ProgramSpec.

    Strategy: build a random Python function source string, then compile
    to FunctionIR via compile_function_to_ir().
    """
    body_expr = _random_expr(spec.param_names, rng, max_depth, use_safe_ops)

    if spec.return_type == "bool":
        # Wrap expression in a comparison
        threshold = round(rng.uniform(-1.0, 1.0), 4)
        body_expr = f"({body_expr}) > {threshold}"

    # Build parameter list with type hints
    params = []
    for name, ptype in zip(spec.param_names, spec.param_types):
        params.append(f"{name}")

    source = f"def {spec.name}({', '.join(params)}):\n    return {body_expr}\n"

    # Compile source string directly to IR (no inspect.getsource needed)
    helpers = _build_helper_namespace()
    try:
        func_ir = compile_source_to_ir(source, spec.name, helpers)
    except Exception:
        # Fallback: trivial function
        fallback = f"def {spec.name}({', '.join(params)}):\n    return 0.0\n"
        func_ir = compile_source_to_ir(fallback, spec.name, helpers)

    return func_ir


def random_loop_program(
    spec: ProgramSpec,
    rng: np.random.Generator,
    loop_var: str = "i",
    loop_count_param: str | None = None,
    array_param: str | None = None,
    max_depth: int = 3,
) -> FunctionIR:
    """Generate a random program with a while loop (for aggregation specs like f_up).

    The loop iterates over elements and accumulates a result.
    """
    # Find the count parameter and array parameter from spec
    if loop_count_param is None:
        # Use last int-typed param as count
        for name, ptype in zip(spec.param_names, spec.param_types):
            if ptype in ("int", "i64"):
                loop_count_param = name
        if loop_count_param is None:
            loop_count_param = spec.param_names[-1]

    if array_param is None:
        # Use first list-typed param
        for name, ptype in zip(spec.param_names, spec.param_types):
            if ptype in ("list", "object"):
                array_param = name
        if array_param is None:
            array_param = spec.param_names[0]

    # Build loop body expression using array element access
    elem_vars = [f"{array_param}[{loop_var}]"]
    init_val = round(rng.uniform(-1.0, 1.0), 4)

    # Random accumulation expression
    accum_expr = _random_expr(
        ["result", f"{array_param}[{loop_var}]"],
        rng, max_depth=max_depth - 1, use_safe_ops=True
    )

    params = [name for name in spec.param_names]
    source = textwrap.dedent(f"""\
        def {spec.name}({', '.join(params)}):
            result = {init_val}
            {loop_var} = 0
            while {loop_var} < {loop_count_param}:
                result = {accum_expr}
                {loop_var} = {loop_var} + 1
            return result
    """)

    helpers = _build_helper_namespace()
    try:
        func_ir = compile_source_to_ir(source, spec.name, helpers)
    except Exception:
        # Fallback: simple sum
        fallback = textwrap.dedent(f"""\
            def {spec.name}({', '.join(params)}):
                return 0.0
        """)
        func_ir = compile_source_to_ir(fallback, spec.name, helpers)

    return func_ir


def _random_expr(
    variables: list[str],
    rng: np.random.Generator,
    max_depth: int,
    use_safe_ops: bool = True,
) -> str:
    """Generate a random expression tree as a string."""
    if max_depth <= 0 or rng.random() < 0.3:
        # Terminal: variable or constant
        if rng.random() < 0.6 and variables:
            return rng.choice(variables)
        else:
            return str(round(rng.uniform(-2.0, 2.0), 4))

    kind = rng.choice(["binary", "unary", "terminal"])

    if kind == "binary":
        left = _random_expr(variables, rng, max_depth - 1, use_safe_ops)
        right = _random_expr(variables, rng, max_depth - 1, use_safe_ops)
        if use_safe_ops:
            op = rng.choice(_SAFE_BINARY_OPS)
        else:
            op = rng.choice(_BINARY_OPS)
        if op == "_safe_div":
            return f"_safe_div({left}, {right})"
        return f"({left} {op} {right})"

    if kind == "unary":
        child = _random_expr(variables, rng, max_depth - 1, use_safe_ops)
        op = rng.choice(_UNARY_OPS)
        if op == "-":
            return f"(-{child})"
        return f"{op}({child})"

    # Terminal
    if variables and rng.random() < 0.7:
        return rng.choice(variables)
    return str(round(rng.uniform(-2.0, 2.0), 4))


def _build_helper_namespace() -> dict[str, Any]:
    """Build namespace with safe math helpers for exec()."""
    import math

    def _safe_div(a, b):
        return a / b if abs(b) > 1e-30 else 0.0

    def _safe_log(a):
        return math.log(max(a, 1e-30))

    def _safe_sqrt(a):
        return math.sqrt(max(a, 0.0))

    return {
        "__builtins__": __builtins__,
        "_safe_div": _safe_div,
        "_safe_log": _safe_log,
        "_safe_sqrt": _safe_sqrt,
        "abs": abs,
        "math": math,
    }
