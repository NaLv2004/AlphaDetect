"""Phase H+4 S0.0 — assert that no GP / micro-evolution module imports
the source-roundtrip helpers (``emit_python_source`` /
``compile_source_to_ir`` / ``ast.parse`` / ``compile``) and that
``mutate_ir`` only exposes pure IR mutation modes.

The single allowed crossings IR -> Python source are:

* ``algorithm_ir/frontend/ir_builder.py``  (parses user-authored Python
  into IR — pre-evolution boundary)
* ``applications/**/evaluator.py`` and the per-detector ``materialize``
  pipeline (executes IR by emitting source then running it — evaluation
  boundary)

Any GP / mutation / crossover / selection module appearing inside
``evolution/`` MUST NOT import or call those helpers.
"""
from __future__ import annotations

import ast
import importlib
from pathlib import Path

import numpy as np
import pytest

import evolution.operators as operators_mod


_FORBIDDEN_NAMES = {"emit_python_source", "compile_source_to_ir"}


def test_mutate_via_recompile_is_gone():
    """The legacy source-roundtrip helper must no longer be defined."""
    assert not hasattr(operators_mod, "_mutate_via_recompile"), (
        "_mutate_via_recompile must be deleted (Phase H+4 S0.0): "
        "it performed a Python-source roundtrip which violates the "
        "single-representation principle."
    )


def test_operators_module_does_not_import_source_helpers():
    """``evolution.operators`` must not import emit_python_source or compile_source_to_ir."""
    module_dict = vars(operators_mod)
    for name in _FORBIDDEN_NAMES:
        assert name not in module_dict, (
            f"`{name}` must not be importable from evolution.operators "
            f"(found in module dict). It violates the single-representation "
            f"principle."
        )


def test_no_source_imports_in_micro_evolution_chain():
    """Static AST scan of the micro-evolution chain rejects any import
    of emit_python_source / compile_source_to_ir.

    Only ``slot_evolution.py`` is allowed to import ``emit_python_source``
    transiently in Phase H+3 (it materializes IR for the evaluator);
    that boundary will be removed once Phase S3.1 fully delegates to
    ``evolution.gp.population`` which calls the evaluator directly.
    For now we only enforce the principle on ``operators.py`` (the
    pure mutation kernel) and ``algorithm_engine.py`` (the macro
    orchestrator), and we accept the slot_evolution.py exception with
    an explicit allow-list.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    targets = [
        repo_root / "evolution" / "operators.py",
        repo_root / "evolution" / "algorithm_engine.py",
    ]
    failures: list[str] = []
    for target in targets:
        text = target.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            pytest.fail(f"{target} did not parse: {exc}")
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in _FORBIDDEN_NAMES:
                        failures.append(
                            f"{target.name}: from {node.module} import {alias.name}"
                        )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in _FORBIDDEN_NAMES:
                        failures.append(f"{target.name}: import {alias.name}")
    assert not failures, (
        "Source-roundtrip imports leaked into the micro-evolution chain:\n  "
        + "\n  ".join(failures)
    )


def test_mutate_ir_default_modes_are_pure_ir():
    """mutate_ir() default mode list must be a subset of pure-IR modes."""
    import inspect
    src = inspect.getsource(operators_mod.mutate_ir)
    # The forbidden modes used to be sampled with non-zero probability.
    for mode in ("\"insert\"", "\"delete\"", "\"swap_lines\""):
        assert mode not in src, (
            f"mutate_ir source still references {mode}; the source-roundtrip "
            f"branches must be removed (Phase H+4 S0.0)."
        )


def test_mutate_ir_smoke_no_nameerror():
    """Calling mutate_ir 200 times on a real detector IR must not raise.

    Specifically, the historical NameError on undefined ``deletable`` in
    the legacy ``swap`` branch must be impossible because that branch is
    gone.
    """
    from evolution.ir_pool import compile_detector_template, _DETECTOR_SPECS

    spec = next(s for s in _DETECTOR_SPECS if s.algo_id == "lmmse")
    ir = compile_detector_template(spec)
    rng = np.random.default_rng(0xDEADBEEF)

    for _ in range(200):
        out = operators_mod.mutate_ir(ir, rng)
        # Returned IR is well-formed (has ops dict, name attribute)
        assert hasattr(out, "ops")
        assert hasattr(out, "name")
