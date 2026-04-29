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


def is_trivial_op(op: "Op", _ir: "FunctionIR" | None = None) -> bool:
    """Return True iff ``op`` is a pure forwarder/constant.

    The optional ``_ir`` parameter is reserved for future opcode rules
    that may need to look up referenced values; current rules need only
    the op itself.
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
        return all(v == first for v in inputs[1:])
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
            # All inputs equal by triviality definition.
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
    stack: list[str] = [value_id]
    while stack:
        vid = stack.pop()
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
]
