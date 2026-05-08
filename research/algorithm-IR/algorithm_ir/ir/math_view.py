"""
MathView: a flat, math-focused view over FunctionIR for the GNN.

Design (per super_node_view_plan.md):
  - Flat DAG. No hierarchy, no sub-views, no LoopBlock containing inner nodes.
    A loop is just a cycle in the edge structure (mechanism A only — no loop_id tags).
  - No semantic merging of math primitives. H.conj() and .T stay as separate nodes.
  - Only "programming-syntax noise" is absorbed:
        C1: callee/structural-arg consts (modules, functions, classes) -> consumer call attrs
        C2: get_attr(.shape/.dtype/...) + const(idx) + get_item chain -> terminal consumer attrs
        C2c: call(len, tensor) when used as alloc shape arg -> alloc call attrs
        C6: assign(target = src) when src has unique consumer (this assign) -> src node
        Orphan drop: const(callable/module) and shape-chains with no consumer -> drop entirely
  - View only. FunctionIR is unchanged. Built on demand.

The MathView holds:
  - tuple of MathNode (each peer, kind-tagged)
  - op_id_to_node: every SSA op_id maps to exactly one MathNode (covers absorbed too)
  - value_id_to_port: SSA value_id -> output MathPort of producing node
  - boundary_nodes: ids of arg/return nodes
"""

from __future__ import annotations

import types
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


# ------------------------------------------------------------------
# Op classification rules (kept module-level for audit-script reuse)
# ------------------------------------------------------------------

_SHAPE_LIKE_ATTRS: frozenset[str] = frozenset({
    "shape", "dtype", "ndim", "size", "nbytes", "itemsize", "strides", "flags",
})

_ALLOC_QNAMES: frozenset[str] = frozenset({
    "np.zeros", "np.ones", "np.empty", "np.full",
    "numpy.zeros", "numpy.ones", "numpy.empty", "numpy.full",
})


def _literal_is_callable_or_module(lit: Any) -> bool:
    """A const's literal is C1-eligible iff it is a Python callable / module / class."""
    if lit is None:
        return False
    if isinstance(lit, (types.ModuleType, type)):
        return True
    if isinstance(lit, (types.FunctionType, types.BuiltinFunctionType, types.MethodType)):
        return True
    # numpy ufuncs etc.
    if callable(lit) and not isinstance(lit, (int, float, complex, str, bytes, bool, tuple, list, dict)):
        return True
    return False


def _is_c1_const(op) -> bool:
    """const op whose literal is a callable/module/class — i.e. pure name resolution."""
    if op.opcode != "const":
        return False
    return _literal_is_callable_or_module(op.attrs.get("literal"))


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------


@dataclass(frozen=True)
class MathPort:
    node_id: str
    port_idx: int = 0
    type_hint: str = "any"


@dataclass(frozen=True)
class MathNode:
    node_id: str
    kind: str  # "math" | "tensor_struct" | "state_update" | "phi" | "branch"
               # | "const" | "jump" | "boundary" | "collection" | "iter"
               # | "other" (must be empty after Phase 3a)
    opcode: str  # human-readable label, e.g. "binary.MatMult", "call.np.eye"
    inputs: tuple = ()
    n_outputs: int = 1
    attrs: Mapping[str, Any] = field(default_factory=dict)
    op_ids: frozenset = frozenset()


class MathView:
    def __init__(
        self,
        *,
        ir,
        nodes: tuple,
        op_id_to_node: Mapping[str, str],
        value_id_to_port: Mapping[str, MathPort],
        boundary_nodes: tuple,
        absorbed: Mapping[str, str],
        dropped: tuple,
    ) -> None:
        self.ir = ir
        self.nodes: tuple[MathNode, ...] = nodes
        self.nodes_by_id: dict[str, MathNode] = {n.node_id: n for n in nodes}
        self.op_id_to_node: dict[str, str] = dict(op_id_to_node)
        self.value_id_to_port: dict[str, MathPort] = dict(value_id_to_port)
        self.boundary_nodes: tuple[str, ...] = boundary_nodes
        self.absorbed: dict[str, str] = dict(absorbed)
        self.dropped: tuple[str, ...] = dropped

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.nodes)

    def n_ssa_op_nodes(self) -> int:
        """Count of MathNodes that wrap >=1 SSA op (excludes pure boundary arg nodes)."""
        return sum(1 for n in self.nodes if n.op_ids)

    def n_boundary_arg_nodes(self) -> int:
        return sum(1 for n in self.nodes
                   if n.kind == "boundary" and n.opcode.startswith("arg."))

    def coverage_ok(self) -> bool:
        """Every SSA op is either mapped to a node or in `dropped`."""
        all_ops = set(self.ir.ops.keys())
        mapped = set(self.op_id_to_node.keys())
        dropped = set(self.dropped)
        return mapped | dropped == all_ops

    def coverage_report(self) -> dict:
        all_ops = set(self.ir.ops.keys())
        mapped = set(self.op_id_to_node.keys())
        dropped = set(self.dropped)
        missing = all_ops - mapped - dropped
        return {
            "n_ssa_ops": len(all_ops),
            "n_mapped": len(mapped),
            "n_dropped": len(dropped),
            "n_missing": len(missing),
            "missing_op_ids": sorted(missing),
        }


# ------------------------------------------------------------------
# Builder
# ------------------------------------------------------------------


def build_math_view(ir) -> MathView:
    ops = ir.ops

    # 1. value_id -> producing op
    value_def_op: dict[str, str] = {}
    for op_id, op in ops.items():
        for v in op.outputs:
            value_def_op[v] = op_id

    # 2. value_id -> list of consumer op_ids (positional info kept via list of (op_id, in_idx))
    value_consumers: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for op_id, op in ops.items():
        for idx, v in enumerate(op.inputs):
            value_consumers[v].append((op_id, idx))

    # absorbed: op_id -> target_op_id (or sentinel "__DROP__")
    absorbed: dict[str, str] = {}

    # ---------------- C1: callee / structural-arg const ----------------
    # Rule: const with callable/module literal AND every consumer is a `call` op.
    # If 0 consumers -> orphan drop.
    # Otherwise absorb into ALL consumers (encoded in attrs). For port-mapping we
    # pick a single "primary" target = the first consumer (deterministic).
    for op_id, op in ops.items():
        if not _is_c1_const(op):
            continue
        out_v = op.outputs[0]
        consumers = value_consumers.get(out_v, [])
        if not consumers:
            absorbed[op_id] = "__DROP__"
            continue
        # All consumers must be `call` ops
        all_call = all(ops[c_id].opcode == "call" for c_id, _ in consumers)
        if not all_call:
            continue
        # Absorb into the first consumer (used for port mapping); all consumers
        # will pick up the absorbed const's name via the attrs-merge pass below.
        primary = consumers[0][0]
        absorbed[op_id] = primary

    # ---------------- C2: shape derivation chain ----------------
    # get_attr(.shape) -> [const(int idx) + get_item] -> consumer(call/compare/branch)
    # Every link must have unique consumer-of-link, otherwise can't safely absorb.
    for op_id, op in ops.items():
        if op_id in absorbed:
            continue
        if op.opcode != "get_attr":
            continue
        if op.attrs.get("attr") not in _SHAPE_LIKE_ATTRS:
            continue
        out_v = op.outputs[0]
        out_consumers = value_consumers.get(out_v, [])
        # Orphan shape attr (no consumer)
        if not out_consumers:
            absorbed[op_id] = "__DROP__"
            continue
        if len(out_consumers) != 1:
            continue
        gi_op_id, gi_in_idx = out_consumers[0]
        if gi_op_id in absorbed:
            continue
        gi_op = ops[gi_op_id]
        if gi_op.opcode != "get_item":
            continue
        # get_item.inputs = (container, index). container must be our get_attr output.
        if gi_op.inputs[0] != out_v:
            continue
        idx_v = gi_op.inputs[1]
        idx_op_id = value_def_op.get(idx_v)
        if not idx_op_id:
            continue
        if idx_op_id in absorbed:
            continue
        idx_op = ops[idx_op_id]
        if idx_op.opcode != "const":
            continue
        # idx const must have unique consumer = this get_item (else it's shared)
        if [c for c, _ in value_consumers.get(idx_v, [])] != [gi_op_id]:
            continue
        # Walk: get_item output -> single consumer of right kind
        gi_out = gi_op.outputs[0]
        gi_out_consumers = value_consumers.get(gi_out, [])
        # Orphan terminal -> drop entire chain
        if not gi_out_consumers:
            absorbed[op_id] = "__DROP__"
            absorbed[idx_op_id] = "__DROP__"
            absorbed[gi_op_id] = "__DROP__"
            continue
        if len(gi_out_consumers) != 1:
            continue
        target_op_id, _ = gi_out_consumers[0]
        if target_op_id in absorbed:
            continue
        target_op = ops[target_op_id]
        if target_op.opcode not in ("call", "compare", "branch"):
            continue
        # All checks passed; absorb 3-op chain into target
        absorbed[op_id] = target_op_id
        absorbed[idx_op_id] = target_op_id
        absorbed[gi_op_id] = target_op_id

    # ---------------- C2c: call(len, tensor) for alloc shape ----------------
    for op_id, op in ops.items():
        if op_id in absorbed:
            continue
        if op.opcode != "call":
            continue
        if op.attrs.get("qualified_name") != "len":
            continue
        out_v = op.outputs[0]
        consumers = value_consumers.get(out_v, [])
        if len(consumers) != 1:
            continue
        cons_op_id, _ = consumers[0]
        if cons_op_id in absorbed:
            continue
        cons_op = ops[cons_op_id]
        if cons_op.opcode != "call":
            continue
        if cons_op.attrs.get("qualified_name") not in _ALLOC_QNAMES:
            continue
        absorbed[op_id] = cons_op_id

    # ---------------- C6: assign elimination ----------------
    # assign(target=name)  with src having unique consumer (this assign) -> absorb into src
    for op_id, op in ops.items():
        if op_id in absorbed:
            continue
        if op.opcode != "assign":
            continue
        if not op.inputs:
            continue
        src_v = op.inputs[0]
        src_op_id = value_def_op.get(src_v)
        if not src_op_id:
            continue
        if src_op_id in absorbed:
            continue
        src_consumers = [c for c, _ in value_consumers.get(src_v, [])]
        if src_consumers != [op_id]:
            continue
        absorbed[op_id] = src_op_id

    # ---------------- M2: bound-method call merge ----------------
    # get_attr(method)  followed by  call(<that-method>, ...)  -> absorb get_attr into call.
    # The get_attr's output must have a single consumer = the call, and the call must
    # use it as inputs[0] (the callee position).
    for op_id, op in ops.items():
        if op_id in absorbed:
            continue
        if op.opcode != "get_attr":
            continue
        if op.attrs.get("attr") in _SHAPE_LIKE_ATTRS:
            continue  # shape-like already handled by C2
        out_v = op.outputs[0]
        consumers = value_consumers.get(out_v, [])
        if len(consumers) != 1:
            continue
        cons_op_id, cons_in_idx = consumers[0]
        if cons_op_id in absorbed:
            continue
        cons_op = ops[cons_op_id]
        if cons_op.opcode != "call":
            continue
        if cons_in_idx != 0:
            continue  # must be the callee position
        absorbed[op_id] = cons_op_id

    # ---------------- J1: noise jump absorption ----------------
    # Absorb jump ops whose source block has exactly one successor (= target).
    # Both entry-jumps and loop backedges qualify because phi.sources +
    # branch.true/false already encode the control flow exhaustively.
    block_of_op: dict[str, str] = {}
    for blk_id, blk in ir.blocks.items():
        for oid in blk.op_ids:
            block_of_op[oid] = blk_id
    for op_id, op in ops.items():
        if op_id in absorbed:
            continue
        if op.opcode != "jump":
            continue
        blk_id = block_of_op.get(op_id)
        if not blk_id:
            continue
        blk = ir.blocks[blk_id]
        succs = list(getattr(blk, "succs", []) or [])
        if len(succs) != 1:
            continue  # safety: only absorb if there's no real choice
        absorbed[op_id] = "__DROP__"

    # ---------------- Build MathNodes ----------------
    nodes: list[MathNode] = []
    op_id_to_node: dict[str, str] = {}
    value_id_to_port: dict[str, MathPort] = {}
    boundary_node_ids: list[str] = []

    counter = [0]
    def _mk_node_id() -> str:
        counter[0] += 1
        return f"mn_{counter[0]:04d}"

    # 1. Boundary arg nodes (one per function param)
    for arg_v in ir.arg_values:
        nid = _mk_node_id()
        val = ir.values.get(arg_v)
        name = getattr(val, "name_hint", None) or arg_v
        type_hint = getattr(val, "type_hint", "any") or "any"
        n = MathNode(
            node_id=nid, kind="boundary", opcode=f"arg.{name}",
            inputs=(), n_outputs=1,
            attrs={"arg_value_id": arg_v, "type_hint": type_hint},
            op_ids=frozenset(),
        )
        nodes.append(n)
        boundary_node_ids.append(nid)
        value_id_to_port[arg_v] = MathPort(node_id=nid, port_idx=0, type_hint=type_hint)

    # 2. Pre-allocate node ids and output ports for non-absorbed ops (in IR insertion order)
    op_order: list[str] = list(ops.keys())
    pending_node: dict[str, str] = {}  # op_id -> node_id reserved for it
    for op_id in op_order:
        if op_id in absorbed:
            continue
        nid = _mk_node_id()
        pending_node[op_id] = nid
        op_id_to_node[op_id] = nid
        op = ops[op_id]
        for port_idx, out_v in enumerate(op.outputs):
            type_hint = "any"
            val = ir.values.get(out_v)
            if val is not None:
                type_hint = getattr(val, "type_hint", None) or "any"
            value_id_to_port[out_v] = MathPort(
                node_id=nid, port_idx=port_idx, type_hint=type_hint,
            )

    # 3. Forward absorbed ops' value_ids to their target node's primary port
    dropped_op_ids: list[str] = []
    for ab_op_id, target in absorbed.items():
        if target == "__DROP__":
            dropped_op_ids.append(ab_op_id)
            # Outputs of dropped ops -> no port (shouldn't be referenced; if they
            # are, the consumer must also be dropped/absorbed)
            continue
        target_nid = op_id_to_node.get(target)
        if not target_nid:
            # Target was itself absorbed/dropped; bubble up
            # (rare; handle by collapsing to root)
            cur = target
            seen = {ab_op_id}
            while cur in absorbed and absorbed[cur] != "__DROP__":
                if cur in seen:
                    break
                seen.add(cur)
                cur = absorbed[cur]
            target_nid = op_id_to_node.get(cur)
            if not target_nid:
                continue
        op_id_to_node[ab_op_id] = target_nid
        ab_op = ops[ab_op_id]
        for out_v in ab_op.outputs:
            if out_v not in value_id_to_port:
                value_id_to_port[out_v] = MathPort(
                    node_id=target_nid, port_idx=0, type_hint="any",
                )

    # 4. Build the actual MathNode objects with merged attrs and wired inputs
    # Group absorbed ops by target
    absorbed_into: dict[str, list[str]] = defaultdict(list)
    for ab_op_id, target in absorbed.items():
        if target == "__DROP__":
            continue
        # Resolve to root target
        root = op_id_to_node.get(ab_op_id)
        if root:
            absorbed_into[root].append(ab_op_id)

    def _kind_of(op) -> str:
        oc = op.opcode
        if oc in ("binary", "unary", "compare"):
            return "math"
        if oc == "call":
            return "math"
        if oc in ("get_attr", "get_item"):
            return "tensor_struct"
        if oc in ("set_item", "set_attr"):
            return "state_update"
        if oc == "phi":
            return "phi"
        if oc == "branch":
            return "branch"
        if oc == "const":
            return "const"
        if oc == "jump":
            return "jump"
        if oc == "return":
            return "boundary"
        if oc == "assign":
            return "math"  # surviving assigns are kept
        if oc in ("build_list", "build_tuple", "build_dict", "build_slice"):
            return "collection"
        if oc in ("iter_init", "iter_next"):
            return "iter"
        return "other"

    def _opcode_label(op, attrs) -> str:
        oc = op.opcode
        if oc == "binary":
            return f"binary.{op.attrs.get('operator', '?')}"
        if oc == "unary":
            return f"unary.{op.attrs.get('operator', '?')}"
        if oc == "compare":
            ops_l = op.attrs.get("operators", [])
            return f"compare.{ops_l[0] if ops_l else '?'}"
        if oc == "call":
            qn = op.attrs.get("qualified_name") or attrs.get("callee_name")
            if not qn:
                bm = attrs.get("bound_method_name")
                if bm:
                    qn = f"method.{bm}"
                else:
                    # Fallback: walk callee value back to its (non-absorbed) producer
                    if op.inputs:
                        callee_v = op.inputs[0]
                        callee_op_id = value_def_op.get(callee_v)
                        if callee_op_id:
                            callee_op = ops[callee_op_id]
                            if callee_op.opcode == "get_attr":
                                qn = f"method.{callee_op.attrs.get('attr', '?')}"
            return f"call.{qn or '?'}"
        if oc == "get_attr":
            return f"get_attr.{op.attrs.get('attr', '?')}"
        if oc == "const":
            lit = op.attrs.get("literal")
            return f"const.{type(lit).__name__}"
        return oc

    for op_id in op_order:
        if op_id in absorbed:
            continue
        nid = pending_node[op_id]
        op = ops[op_id]
        attrs: dict[str, Any] = {
            k: v for k, v in op.attrs.items() if k != "xdsl_op"
        }

        # Merge absorbed children into attrs.
        # Also remember any "input substitutions" needed when an absorbed child
        # was the producer of one of our inputs (e.g. M2 absorbs get_attr.conj
        # into call(H.conj()): the call's input v_7 should be replaced by H).
        absorbed_value_ids: set[str] = set()
        substitute_for: dict[str, str | None] = {}
        for child_id in absorbed_into.get(nid, []):
            child = ops[child_id]
            absorbed_value_ids.update(child.outputs)
            if child.opcode == "const":
                if "name" in child.attrs:
                    attrs.setdefault("callee_name", child.attrs["name"])
                if "literal" in child.attrs:
                    pc = attrs.setdefault("packed_consts", {})
                    pc[child.outputs[0]] = repr(child.attrs.get("literal"))[:80]
            elif child.opcode == "get_attr":
                child_attr = child.attrs.get("attr")
                if child_attr in _SHAPE_LIKE_ATTRS:
                    attrs.setdefault("shape_derivation_attr", child_attr)
                else:
                    # M2 bound-method absorption: receiver replaces the absorbed value
                    attrs.setdefault("bound_method_name", child_attr)
                    if child.inputs:
                        for child_out in child.outputs:
                            substitute_for[child_out] = child.inputs[0]
            elif child.opcode == "get_item":
                attrs.setdefault("shape_derivation_via_get_item", True)
            elif child.opcode == "call" and child.attrs.get("qualified_name") == "len":
                attrs.setdefault("shape_derivation_via_len", True)
            elif child.opcode == "assign":
                attrs.setdefault("var_name_assigned", child.attrs.get("target"))
                attrs.setdefault("absorbed_assign_op_id", child_id)

        # Build inputs (handling substitutions then skipping pure-absorbed)
        input_ports: list[MathPort] = []
        for in_v in op.inputs:
            if in_v in substitute_for:
                sub = substitute_for[in_v]
                if sub is None:
                    continue
                port = value_id_to_port.get(sub)
                if port is None:
                    continue
                input_ports.append(port)
                continue
            if in_v in absorbed_value_ids:
                continue
            port = value_id_to_port.get(in_v)
            if port is None:
                continue
            input_ports.append(port)

        # n_outputs
        if op.opcode == "branch":
            n_out = 2  # true/false control ports
        else:
            n_out = max(len(op.outputs), 1) if op.opcode != "return" else 0

        kind = _kind_of(op)
        opcode_label = _opcode_label(op, attrs)
        owned = frozenset({op_id} | set(absorbed_into.get(nid, [])))

        nodes.append(MathNode(
            node_id=nid, kind=kind, opcode=opcode_label,
            inputs=tuple(input_ports), n_outputs=n_out,
            attrs=attrs, op_ids=owned,
        ))

    view = MathView(
        ir=ir,
        nodes=tuple(nodes),
        op_id_to_node=op_id_to_node,
        value_id_to_port=value_id_to_port,
        boundary_nodes=tuple(boundary_node_ids),
        absorbed={k: v for k, v in absorbed.items() if v != "__DROP__"},
        dropped=tuple(dropped_op_ids),
    )

    # Invariant: every SSA op is either mapped or dropped
    rep = view.coverage_report()
    if rep["n_missing"] > 0:
        raise ValueError(
            f"MathView coverage failure: {rep['n_missing']} ops not accounted for: "
            f"{rep['missing_op_ids'][:10]}..."
        )
    return view


# ------------------------------------------------------------------
# Compression statistics
# ------------------------------------------------------------------


def compression_stats(view: MathView) -> dict:
    n_ops = len(view.ir.ops)
    n_args = view.n_boundary_arg_nodes()
    n_ssa_nodes = view.n_ssa_op_nodes()
    n_total_nodes = len(view.nodes)
    return {
        "n_ssa_ops": n_ops,
        "n_args": n_args,
        "n_ssa_op_nodes": n_ssa_nodes,
        "n_total_nodes": n_total_nodes,
        "n_absorbed": len(view.absorbed),
        "n_dropped": len(view.dropped),
        "compression_ssa": 1.0 - n_ssa_nodes / max(n_ops, 1),
    }
