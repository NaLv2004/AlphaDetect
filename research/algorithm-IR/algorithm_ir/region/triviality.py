"""Identify "trivial" IR ops that carry no semantic computation.

Trivial ops are pure forwarders/constants whose presence inflates GNN
graph node counts and dilutes the region-proposal action space without
adding real algorithmic structure. They are intentionally **kept** in the
physical IR (so codegen / execution / SubgraphSnapshot are untouched);
the visibility filter only removes them from GNN-facing views.

Categories
----------
- ``const``    : zero-input literal/attribute binding.
- ``get_attr`` : single-input attribute lookup chain (e.g. ``np.linalg.solve``).
- ``assign``   : single-input ``:=`` SSA rebind (pure rename).
- ``jump``     : zero-output unconditional goto (CFG already gives the edge).
- ``return``   : zero-output return-value collector (return values are
  tracked separately via ``FunctionIR.return_values``).
- ``build_tuple`` / ``build_list`` / ``build_slice`` : pure value packers
  whose output carries no novel computation — only a bundling of inputs.
- trivial ``phi`` : every input value-id is identical (loop-header phi
  for a value never reassigned in the loop body).

Edge transitivity
-----------------
When a trivial op is hidden, callers that reason about dataflow edges
must redirect through it. :func:`visible_def_op` walks back through any
chain of trivial defining ops until it hits a non-trivial producer (or
``None`` for function arguments / unresolved values).

For multi-input trivial ops (e.g. ``build_tuple(a, b, c)``) a single
"first predecessor" walk loses the other inputs. Use
:func:`visible_def_ops` to enumerate **all** non-trivial producers
that transitively feed a value.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from algorithm_ir.ir.model import FunctionIR, Op

# Opcodes whose op is *always* trivial regardless of inputs.
# - const/get_attr/assign: single-input forwarders or constants.
# - jump/return: control-flow terminators with no algorithmic content
#   (the CFG block-sequential edges already convey ordering, and
#   ``return`` values are observable via ``FunctionIR.return_values``).
# - build_tuple/build_list/build_slice: pure container packers.
_ALWAYS_TRIVIAL_OPCODES = frozenset({
    "const", "get_attr", "assign",
    "jump", "return",
    "build_tuple", "build_list", "build_slice",
})


_ITER_TRIVIAL_PHI_CACHE_KEY = "_iter_trivial_phi_cache"


def _compute_iterative_trivial_phi(
    ir: "FunctionIR",
) -> tuple[set[str], dict[str, str]]:
    """Iteratively collapse trivial phi to a fixed point.

    A phi is **iteratively trivial** if, after collapsing all other
    iteratively-trivial phi to their canonical inputs, its non-self
    inputs all reduce to the same value. This is the classic
    "redundant phi elimination" used in SSA optimizers (Cytron et al.,
    later refined by Briggs/Cooper).

    The single-input ``is_trivial_op`` check only catches direct
    ``phi(x, x)`` — but in nested loops, an outer phi like
    ``phi(entry, inner_phi_output)`` is dead only after the inner phi
    is recognized as trivial. Iterating to a fixed point handles this.

    Returns
    -------
    (trivial_phi_ids, rep)
        ``trivial_phi_ids`` is the set of phi ``op.id`` values that
        collapse. ``rep`` maps each value-id to its canonical
        representative (a value-id that is *not* defined by any
        iteratively-trivial phi). Walk via ``find()``-style chasing.
    """
    rep: dict[str, str] = {}

    def find(v: str) -> str:
        # Path-compressed lookup.
        path: list[str] = []
        while rep.get(v, v) != v:
            path.append(v)
            v = rep[v]
        for p in path:
            rep[p] = v
        return v

    phi_ops = [op for op in ir.ops.values() if op.opcode == "phi"]
    changed = True
    while changed:
        changed = False
        for op in phi_ops:
            if not op.outputs:
                continue
            out = op.outputs[0]
            if find(out) != out:
                continue  # already collapsed
            # Canonicalize each input, dropping self-references that
            # only refer back to this phi (loop carry of unmodified value).
            canonical: list[str] = []
            for inp in op.inputs:
                fi = find(inp)
                if fi != out:
                    canonical.append(fi)
            if not canonical:
                continue
            first = canonical[0]
            if all(c == first for c in canonical[1:]):
                rep[out] = first
                changed = True

    trivial_ids = {
        op.id for op in phi_ops
        if op.outputs and find(op.outputs[0]) != op.outputs[0]
    }
    return trivial_ids, rep


def _get_iter_trivial_cache(
    ir: "FunctionIR | None",
) -> tuple[set[str], dict[str, str]] | tuple[None, None]:
    """Return cached ``(trivial_phi_ids, rep)`` for ``ir``, computing on demand.

    The cache is stashed on ``ir.attrs`` under a private key so it is
    transparent to the rest of the system. Callers that mutate ``ir``
    must invalidate via :func:`invalidate_iter_trivial_cache`.
    """
    if ir is None:
        return None, None
    cache = ir.attrs.get(_ITER_TRIVIAL_PHI_CACHE_KEY)
    if cache is None:
        cache = _compute_iterative_trivial_phi(ir)
        ir.attrs[_ITER_TRIVIAL_PHI_CACHE_KEY] = cache
    return cache


def invalidate_iter_trivial_cache(ir: "FunctionIR") -> None:
    """Drop the cached iterative-trivial-phi analysis after IR mutation."""
    if ir is None:
        return
    ir.attrs.pop(_ITER_TRIVIAL_PHI_CACHE_KEY, None)


def _canonical_value(ir: "FunctionIR | None", vid: str) -> str:
    """Walk a value through any iteratively-trivial phi rep chain."""
    if ir is None:
        return vid
    _, rep = _get_iter_trivial_cache(ir)
    if not rep:
        return vid
    cur = vid
    seen: set[str] = set()
    while cur in rep and rep[cur] != cur:
        if cur in seen:
            return cur
        seen.add(cur)
        cur = rep[cur]
    return cur


def is_trivial_op(op: "Op", ir: "FunctionIR | None" = None) -> bool:
    """Return True iff ``op`` is a pure forwarder/constant.

    When ``ir`` is provided, phi triviality is also checked against the
    iterative fixed-point analysis (covers nested-loop phi chains where
    the outer phi only becomes trivial after the inner phi collapses).
    Without ``ir``, only the single-step ``phi(x, x)`` check is used.
    """
    if op is None:
        return False
    opcode = op.opcode
    if opcode in _ALWAYS_TRIVIAL_OPCODES:
        return True
    if opcode == "phi":
        inputs = op.inputs or []
        if not inputs:
            return True
        first = inputs[0]
        if all(v == first for v in inputs[1:]):
            return True
        if ir is not None:
            trivial_ids, _ = _get_iter_trivial_cache(ir)
            if trivial_ids and op.id in trivial_ids:
                return True
        return False
    return False


def visible_def_op(ir: "FunctionIR", value_id: str) -> "Op | None":
    """Return the nearest non-trivial op defining ``value_id`` (transitively).

    Walks back through chains of trivial ops. Returns ``None`` if the
    value is a function argument, has no def, or the chain terminates in
    an unresolved id (e.g. dangling phi backedge).
    """
    seen: set[str] = set()
    current = value_id
    while True:
        # Canonicalize through any iteratively-trivial phi chain first.
        current = _canonical_value(ir, current)
        if current in seen:
            return None  # cycle guard (should not happen in well-formed SSA)
        seen.add(current)
        val = ir.values.get(current)
        if val is None or val.def_op is None:
            return None
        op = ir.ops.get(val.def_op)
        if op is None:
            return None
        if not is_trivial_op(op, ir):
            return op
        # Trivial op: hop to its single semantic predecessor.
        if not op.inputs:
            return None  # const has no predecessor
        if op.opcode == "phi":
            # All inputs equal by single-step triviality, OR the phi is
            # iteratively trivial (handled by _canonical_value at the
            # next loop iteration).
            current = op.inputs[0]
        else:
            # const handled above (no inputs); get_attr/assign have one input.
            current = op.inputs[0]


def is_trivial_value(ir: "FunctionIR", value_id: str) -> bool:
    """Return True iff ``value_id`` is defined by a trivial op."""
    val = ir.values.get(value_id)
    if val is None or val.def_op is None:
        return False
    op = ir.ops.get(val.def_op)
    if op is None:
        return False
    return is_trivial_op(op, ir)


def visible_def_ops(ir: "FunctionIR", value_id: str) -> list["Op"]:
    """Return **all** non-trivial ops transitively producing ``value_id``.

    Walks back through arbitrary chains of trivial ops, fanning out at
    multi-input trivial ops (e.g. ``build_tuple(a, b)`` returns the
    visible producers of *both* ``a`` and ``b``).

    Returns an empty list for function arguments, unresolved values, or
    chains that terminate without ever hitting a non-trivial op.
    Order is unspecified; callers should treat it as a set.
    """
    out: list["Op"] = []
    seen_ops: set[str] = set()
    seen_vals: set[str] = set()
    stack: list[str] = [_canonical_value(ir, value_id)]
    while stack:
        vid = _canonical_value(ir, stack.pop())
        if vid in seen_vals:
            continue
        seen_vals.add(vid)
        val = ir.values.get(vid)
        if val is None or val.def_op is None:
            continue  # function arg / unresolved
        op = ir.ops.get(val.def_op)
        if op is None or op.id in seen_ops:
            continue
        seen_ops.add(op.id)
        if not is_trivial_op(op, ir):
            out.append(op)
            continue
        # Trivial op: fan out across every input.
        for inp in op.inputs:
            stack.append(inp)
    return out


__all__ = [
    "is_trivial_op",
    "is_trivial_value",
    "visible_def_op",
    "visible_def_ops",
    "invalidate_iter_trivial_cache",
]
