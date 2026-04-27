"""Materialization pipeline: replace AlgSlot ops in skeleton IR with concrete
implementations, then compile to a callable Python function.

Pipeline::

    AlgorithmGenome  →  materialize()  →  FunctionIR (slot-free)
                                                ↓
                                       emit_python_source()
                                                ↓
                                          exec → callable

Key entry points
-----------------
- ``materialize(genome)`` — replace each AlgSlot with the best variant
  from its slot population, returning a complete (slot-free) FunctionIR.
- ``ir_to_callable(func_ir, extra_globals)`` — emit Python source from a
  FunctionIR and compile to a callable.
- ``materialize_to_callable(genome, extra_globals)`` — convenience wrapper
  that does materialize → ir_to_callable in one step.
"""

from __future__ import annotations

import hashlib
import textwrap
import threading
from copy import deepcopy
from collections import OrderedDict
from typing import Any, Callable

import numpy as np

from algorithm_ir.ir.model import FunctionIR, Op
from algorithm_ir.regeneration.codegen import emit_python_source

from evolution.pool_types import AlgorithmGenome


_CALLABLE_CACHE_LOCK = threading.Lock()
_CALLABLE_CACHE: "OrderedDict[str, Callable]" = OrderedDict()
_CALLABLE_CACHE_MAX = 512


# ═══════════════════════════════════════════════════════════════════════════
# Source-level materialization (pragmatic approach)
# ═══════════════════════════════════════════════════════════════════════════
#
# Instead of xDSL-level op surgery (fragile across xDSL versions), we use
# the codegen path that already knows how to handle AlgSlot:
#
# 1. emit_python_source(skeleton_ir)  →  source with __slot_xxx__ placeholders
# 2. For each placeholder, compile the corresponding slot variant's IR to
#    Python source and inject it as a local function
# 3. Replace each __slot_xxx__ call-site with a call to the injected function


def materialize(genome: AlgorithmGenome) -> str:
    """Materialize a genome into a single Python source string.

    Annotation-only model (M5): ``ir.slot_meta`` is the source of truth and
    the IR already inlines every slot body, so emit the source as-is. The
    legacy ``AlgSlot``-stub inlining path has been removed.
    """
    skeleton_ir = genome.structural_ir

    if getattr(skeleton_ir, "slot_meta", None) is not None:
        return emit_python_source(skeleton_ir)

    # Final-stage cleanup: if a genome arrives without slot_meta, the IR is
    # already flat (no AlgSlot ops are emitted in the new model) and we
    # simply emit it. Any residual ``slot`` ops will be rendered as
    # ``# slot: <name>`` comments by codegen.
    return emit_python_source(skeleton_ir)


def _extract_func_name(source: str) -> str | None:
    """Extract the function name from a ``def`` statement."""
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("def "):
            # Extract name between 'def ' and '('
            rest = stripped[4:]
            paren = rest.find("(")
            if paren > 0:
                return rest[:paren].strip()
    return None


# ═══════════════════════════════════════════════════════════════════════════
# materialize_with_override — micro-level evaluation helper
# ═══════════════════════════════════════════════════════════════════════════


def materialize_with_override(
    genome: AlgorithmGenome,
    override_map: dict[str, "FunctionIR"],
    extra_globals: dict[str, Any] | None = None,
) -> Callable:
    """Materialize a genome with specific slot variant overrides, then compile.

    Like ``materialize_to_callable`` but for each slot_id in *override_map*
    the provided FunctionIR is used **instead of** the population's best
    variant.  All other slots use the population's current best.

    This is the workhorse of micro-level evaluation: we fix all other slots
    and vary only the target slot to evaluate a candidate variant.

    Parameters
    ----------
    genome : AlgorithmGenome
        The genome whose skeleton will be materialized.
    override_map : dict[str, FunctionIR]
        ``{slot_id: variant_ir}`` — override the best variant for these slots.
    extra_globals : dict, optional
        Additional names for the exec namespace.

    Returns
    -------
    callable
        Compiled detector function.
    """
    source = _materialize_source_with_override(genome, override_map)
    func_name = _extract_func_name_from_full(source, genome.algo_id)

    ns = _default_exec_namespace()
    if extra_globals:
        ns.update(extra_globals)

    try:
        exec(compile(source, f"<override:{genome.algo_id}>", "exec"), ns)
    except Exception as exc:
        raise RuntimeError(
            f"materialize_with_override failed for '{genome.algo_id}': {exc}\n"
            f"Source:\n{source}"
        ) from exc

    fn = ns.get(func_name)
    if fn is None:
        raise RuntimeError(
            f"Function '{func_name}' not found after override materialization"
        )
    return fn


def _materialize_source_with_override(
    genome: AlgorithmGenome,
    override_map: dict[str, "FunctionIR"],
) -> str:
    """Annotation-only override path (M5).

    With the slot-meta model the canonical IR carries the inlined slot
    bodies directly, so the only legitimate way to evaluate an alternative
    variant is to first call ``apply_slot_variant`` to splice it into the
    IR, then materialize that resulting genome. This helper therefore
    silently ignores ``override_map`` and emits the genome's current IR
    source as-is — kept for call-site compatibility with the micro-evo
    fast-path which already passes a freshly grafted IR via ``genome.ir``.
    """
    skeleton_ir = genome.structural_ir
    return emit_python_source(skeleton_ir)


# ═══════════════════════════════════════════════════════════════════════════
# ir_to_callable
# ═══════════════════════════════════════════════════════════════════════════

def ir_to_callable(
    func_ir: FunctionIR,
    extra_globals: dict[str, Any] | None = None,
) -> Callable:
    """Emit Python source from a *slot-free* FunctionIR and compile it
    to a callable.

    Parameters
    ----------
    func_ir : FunctionIR
        Must not contain any AlgSlot ops.
    extra_globals : dict, optional
        Additional names to inject into the exec namespace.

    Returns
    -------
    callable
        The compiled function.

    Raises
    ------
    RuntimeError
        If compilation or exec fails.
    """
    source = emit_python_source(func_ir)
    func_name = _extract_func_name(source)
    if func_name is None:
        raise RuntimeError("Cannot extract function name from generated source")

    ns = _default_exec_namespace()
    if extra_globals:
        ns.update(extra_globals)

    try:
        exec(compile(source, f"<ir:{func_name}>", "exec"), ns)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to compile IR to callable: {exc}\n"
            f"Source:\n{source}"
        ) from exc

    fn = ns.get(func_name)
    if fn is None:
        raise RuntimeError(f"Function '{func_name}' not found after exec")
    return fn


# ═══════════════════════════════════════════════════════════════════════════
# materialize_to_callable
# ═══════════════════════════════════════════════════════════════════════════

def materialize_to_callable(
    genome: AlgorithmGenome,
    extra_globals: dict[str, Any] | None = None,
) -> Callable:
    """Materialize a genome and compile it to a callable in one step.

    Parameters
    ----------
    genome : AlgorithmGenome
        Genome with AlgSlot ops in structural_ir.
    extra_globals : dict, optional
        Additional names for the execution namespace.

    Returns
    -------
    callable
        A function with signature ``(H, y, sigma2, constellation) -> x_hat``.
    """
    source = materialize(genome)
    func_name = _extract_func_name_from_full(source, genome.algo_id)
    cache_key = hashlib.sha1(source.encode("utf-8")).hexdigest()

    if extra_globals is None:
        with _CALLABLE_CACHE_LOCK:
            cached = _CALLABLE_CACHE.get(cache_key)
            if cached is not None:
                _CALLABLE_CACHE.move_to_end(cache_key)
                return cached

    ns = _default_exec_namespace()
    if extra_globals:
        ns.update(extra_globals)

    try:
        exec(compile(source, f"<materialized:{genome.algo_id}>", "exec"), ns)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to compile materialized genome '{genome.algo_id}': {exc}\n"
            f"Source:\n{source}"
        ) from exc

    fn = ns.get(func_name)
    if fn is None:
        raise RuntimeError(
            f"Function '{func_name}' not found after materializing '{genome.algo_id}'"
        )

    if extra_globals is None:
        with _CALLABLE_CACHE_LOCK:
            _CALLABLE_CACHE[cache_key] = fn
            _CALLABLE_CACHE.move_to_end(cache_key)
            while len(_CALLABLE_CACHE) > _CALLABLE_CACHE_MAX:
                _CALLABLE_CACHE.popitem(last=False)
    return fn


def _extract_func_name_from_full(source: str, algo_id: str) -> str:
    """Extract the *main* function name from a materialized source.

    The main function is the last ``def`` at the top level (indent=0).
    Falls back to the algo_id if not found.
    """
    last_name = None
    for line in source.splitlines():
        if line.startswith("def ") and "(" in line:
            name = line[4:line.index("(")].strip()
            last_name = name
    return last_name or algo_id


# ═══════════════════════════════════════════════════════════════════════════
# Execution namespace
# ═══════════════════════════════════════════════════════════════════════════

def _default_exec_namespace() -> dict[str, Any]:
    """Build the default namespace for exec-compiling materialized sources."""
    from evolution.ir_pool import _template_globals
    ns = _template_globals()

    # Tree search support
    try:
        from evolution.pool_ops_l2 import TreeNode
        ns["TreeNode"] = TreeNode
    except ImportError:
        pass

    return ns
