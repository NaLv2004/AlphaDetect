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

from evolution.pool_types import AlgorithmGenome, SlotPopulation
from evolution.ir_pool import find_algslot_ops


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

    Each ``AlgSlot`` op becomes a call to the best variant from the
    corresponding SlotPopulation, inlined as a local helper.

    Returns the complete Python source with no slot placeholders.
    """
    skeleton_ir = genome.structural_ir
    slot_ops = find_algslot_ops(skeleton_ir)

    # 1. Emit skeleton source (with __slot_xxx__ placeholders)
    skeleton_source = emit_python_source(skeleton_ir)

    # 2. Collect slot implementations
    slot_impls: dict[str, str] = {}   # op_id → impl_source
    slot_names: dict[str, str] = {}   # op_id → helper_func_name

    for slot_op in slot_ops:
        slot_id = slot_op.attrs.get("slot_id", "unknown")
        op_id = slot_op.id

        # Find matching SlotPopulation
        pop = _find_population_for_slot(genome, slot_id)
        if pop is None:
            # Generate a fallback pass-through
            slot_names[op_id] = f"_slot_{slot_id}"
            slot_impls[op_id] = f"def _slot_{slot_id}(*args):\n    return args[0] if args else None\n"
            continue

        # Get best variant IR
        best_idx = pop.best_idx
        best_ir = pop.variants[best_idx] if best_idx < len(pop.variants) else None

        variant_source = None
        if best_ir is not None:
            variant_source = emit_python_source(best_ir)

        if variant_source is None:
            # Fallback: pass-through
            slot_names[op_id] = f"_slot_{slot_id}"
            slot_impls[op_id] = f"def _slot_{slot_id}(*args):\n    return args[0] if args else None\n"
            continue

        # Determine the original function name from the variant
        func_name = _extract_func_name(variant_source)
        if func_name is None:
            func_name = f"_slot_{slot_id}"

        # Create a unique name to avoid collisions
        unique_name = f"_slot_{slot_id}_{op_id}"
        variant_source = variant_source.replace(
            f"def {func_name}(", f"def {unique_name}(", 1
        )

        slot_names[op_id] = unique_name
        slot_impls[op_id] = variant_source

    # 3. Replace __slot_xxx__ placeholders in skeleton source
    materialized = skeleton_source
    for op_id, helper_name in slot_names.items():
        placeholder = f"__slot_{op_id}__"
        materialized = materialized.replace(placeholder, helper_name)

    # 4. Prepend slot helper definitions
    helpers = "\n".join(slot_impls.values())

    parts = [p for p in [helpers, materialized] if p.strip()]
    full_source = "\n\n".join(parts)

    return full_source


def _find_population_for_slot(
    genome: AlgorithmGenome,
    slot_id: str,
) -> SlotPopulation | None:
    """Find the SlotPopulation matching a slot_id.

    Tries exact match on the population key, then matches by the
    short name at the end of the key, then checks if slot_id is a
    suffix of the pop key's last segment (e.g. "bp_sweep" ends with "sweep").
    """
    for pop_key, pop in genome.slot_populations.items():
        if pop.slot_id == slot_id:
            return pop
        # Check if pop_key ends with the slot_id
        if pop_key.endswith(f".{slot_id}"):
            return pop
        # Check short_name extracted from key
        parts = pop_key.split(".")
        short = parts[-1] if parts else ""
        if short == slot_id:
            return pop
        # Check if slot_id ends with the short_name (e.g. "bp_sweep" ends with "sweep")
        if slot_id.endswith(short) or slot_id.endswith(f"_{short}"):
            return pop
    return None


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
    """Like ``materialize()`` but uses overrides for specified slots."""
    skeleton_ir = genome.structural_ir
    slot_ops = find_algslot_ops(skeleton_ir)

    skeleton_source = emit_python_source(skeleton_ir)

    slot_impls: dict[str, str] = {}
    slot_names: dict[str, str] = {}

    for slot_op in slot_ops:
        slot_id = slot_op.attrs.get("slot_id", "unknown")
        op_id = slot_op.id

        # Check if this slot is overridden
        if slot_id in override_map:
            override_ir = override_map[slot_id]
            variant_source = emit_python_source(override_ir)
            func_name = _extract_func_name(variant_source)
            if func_name is None:
                func_name = f"_slot_{slot_id}"
            unique_name = f"_slot_{slot_id}_{op_id}"
            variant_source = variant_source.replace(
                f"def {func_name}(", f"def {unique_name}(", 1
            )
            slot_names[op_id] = unique_name
            slot_impls[op_id] = variant_source
            continue

        # Otherwise, use the population's best variant (same as materialize)
        pop = _find_population_for_slot(genome, slot_id)
        if pop is None:
            slot_names[op_id] = f"_slot_{slot_id}"
            slot_impls[op_id] = (
                f"def _slot_{slot_id}(*args):\n"
                f"    return args[0] if args else None\n"
            )
            continue

        best_idx = pop.best_idx
        best_ir = pop.variants[best_idx] if best_idx < len(pop.variants) else None

        variant_source = None
        if best_ir is not None:
            variant_source = emit_python_source(best_ir)
        if variant_source is None:
            slot_names[op_id] = f"_slot_{slot_id}"
            slot_impls[op_id] = (
                f"def _slot_{slot_id}(*args):\n"
                f"    return args[0] if args else None\n"
            )
            continue

        func_name = _extract_func_name(variant_source)
        if func_name is None:
            func_name = f"_slot_{slot_id}"
        unique_name = f"_slot_{slot_id}_{op_id}"
        variant_source = variant_source.replace(
            f"def {func_name}(", f"def {unique_name}(", 1
        )
        slot_names[op_id] = unique_name
        slot_impls[op_id] = variant_source

    materialized = skeleton_source
    for op_id, helper_name in slot_names.items():
        placeholder = f"__slot_{op_id}__"
        materialized = materialized.replace(placeholder, helper_name)

    helpers = "\n".join(slot_impls.values())
    return helpers + "\n\n" + materialized


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
