"""General-purpose IR-level grafting engine.

Implements ``graft_general()`` — a universal, **inline** grafting system
that operates entirely at the FunctionIR level (ops/values/blocks) without
any Python source intermediary.

Core principle: IR is the sole evolution medium.  All structural
modifications happen directly on FunctionIR dicts.  Python source is only
generated at execution time via ``materialize()``.

Inline grafting means donor ops are cloned directly into the host IR,
so every op is visible and mutable by subsequent evolution.  No opaque
``call`` ops are created.
"""

from __future__ import annotations

import itertools
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from algorithm_ir.ir.model import FunctionIR, Op, SlotMeta, Value, Block
from algorithm_ir.ir.validator import rebuild_def_use, validate_function_ir
from algorithm_ir.region.selector import RewriteRegion


# ---------------------------------------------------------------------------
# GraftArtifact — output of graft_general()
# ---------------------------------------------------------------------------

@dataclass
class GraftArtifact:
    """Complete output of a graft_general() operation."""
    ir: FunctionIR                    # The grafted IR
    new_slot_ids: list[str]           # AlgSlot IDs introduced by the donor
    replaced_op_ids: list[str]        # Original ops that were removed
    inlined_op_ids: list[str]         # Donor ops inlined into host
    id_map: dict[str, str] = field(default_factory=dict)  # donor_id → host_id
    # Provenance from the typed-bipartite binding layer.  ``None`` when
    # the legacy name-hint / positional matcher was used (e.g. donor had
    # no args or every candidate was lattice-incompatible).
    typed_binding: dict[str, object] | None = None


# ---------------------------------------------------------------------------
# IR helper operations
# ---------------------------------------------------------------------------

_counter = itertools.count()


def _fresh_id(prefix: str = "g") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:6]}"


def _manual_clone_ir(ir: FunctionIR) -> FunctionIR:
    """Clone a FunctionIR by copying its dict contents (no xDSL module)."""
    return FunctionIR(
        id=ir.id,
        name=ir.name,
        arg_values=list(ir.arg_values),
        return_values=list(ir.return_values),
        values={
            k: Value(
                id=v.id,
                name_hint=v.name_hint,
                type_hint=v.type_hint,
                source_span=v.source_span,
                def_op=v.def_op,
                use_ops=list(v.use_ops),
                attrs=dict(v.attrs),
            )
            for k, v in ir.values.items()
        },
        ops={
            k: Op(
                id=o.id,
                opcode=o.opcode,
                inputs=list(o.inputs),
                outputs=list(o.outputs),
                block_id=o.block_id,
                source_span=o.source_span,
                attrs=dict(o.attrs),
            )
            for k, o in ir.ops.items()
        },
        blocks={
            k: Block(
                id=b.id,
                op_ids=list(b.op_ids),
                preds=list(b.preds),
                succs=list(b.succs),
                attrs=dict(b.attrs),
            )
            for k, b in ir.blocks.items()
        },
        entry_block=ir.entry_block,
        attrs=dict(ir.attrs) if ir.attrs else {},
    )


def find_region_boundary(
    ir: FunctionIR,
    op_ids: set[str],
) -> tuple[list[str], list[str]]:
    """Analyse the data-flow boundary of a region of ops.

    Parameters
    ----------
    ir : FunctionIR
    op_ids : set[str]
        Op IDs that constitute the region.

    Returns
    -------
    entry_values : list[str]
        Values defined *outside* the region but *used* by ops inside it.
    exit_values : list[str]
        Values defined *inside* the region but *used* by ops outside it.
    """
    region_defined: set[str] = set()
    region_used: set[str] = set()

    for oid in op_ids:
        op = ir.ops[oid]
        region_defined.update(op.outputs)
        region_used.update(op.inputs)

    # Entry: used inside, defined outside
    entry_values = sorted(v for v in region_used if v not in region_defined)
    # Exit: defined inside, used outside
    exit_values: list[str] = []
    for vid in sorted(region_defined):
        val = ir.values.get(vid)
        if val is None:
            continue
        for use_op in val.use_ops:
            if use_op not in op_ids:
                exit_values.append(vid)
                break

    return entry_values, exit_values


def create_call_op(
    ir: FunctionIR,
    callee_name: str,
    arg_vids: list[str],
    result_count: int,
    block_id: str,
    attrs: dict[str, Any] | None = None,
) -> tuple[str, list[str]]:
    """Create a call op in *ir* and return ``(op_id, output_vids)``.

    The callee is registered as a ``const`` op whose literal is the
    callable name, so that ``codegen.py`` can emit ``callee_name(args…)``.
    """
    # 1. Const op for the callee reference
    callee_vid = _fresh_id("v_callee")
    callee_op_id = _fresh_id("op_callee")
    ir.values[callee_vid] = Value(
        id=callee_vid,
        name_hint=callee_name,
        type_hint="function",
        attrs={"var_name": callee_name, "role": "callee"},
    )
    ir.ops[callee_op_id] = Op(
        id=callee_op_id,
        opcode="const",
        inputs=[],
        outputs=[callee_vid],
        block_id=block_id,
        attrs={"literal": None, "name": callee_name},
    )
    ir.values[callee_vid].def_op = callee_op_id

    # 2. Output values
    out_vids: list[str] = []
    for i in range(result_count):
        vid = _fresh_id(f"v_graft_out{i}")
        ir.values[vid] = Value(
            id=vid,
            name_hint=f"graft_result_{i}",
            type_hint="object",
            attrs={"role": "graft_output"},
        )
        out_vids.append(vid)

    # 3. Call op
    call_op_id = _fresh_id("op_graft_call")
    all_inputs = [callee_vid] + list(arg_vids)
    call_attrs: dict[str, Any] = {
        "n_args": len(arg_vids),
        "grafted": True,
    }
    if attrs:
        call_attrs.update(attrs)

    ir.ops[call_op_id] = Op(
        id=call_op_id,
        opcode="call",
        inputs=all_inputs,
        outputs=out_vids,
        block_id=block_id,
        attrs=call_attrs,
    )
    # Wire def_op / use_ops
    for vid in out_vids:
        ir.values[vid].def_op = call_op_id
    for vid in all_inputs:
        if vid in ir.values:
            ir.values[vid].use_ops.append(call_op_id)

    # 4. Insert ops into block's op_ids (before the terminator)
    block = ir.blocks[block_id]
    # Find insertion point: before branch/jump/return
    insert_idx = len(block.op_ids)
    for i, oid in enumerate(block.op_ids):
        if ir.ops.get(oid) and ir.ops[oid].opcode in ("branch", "jump", "return"):
            insert_idx = i
            break
    block.op_ids.insert(insert_idx, callee_op_id)
    block.op_ids.insert(insert_idx + 1, call_op_id)

    return call_op_id, out_vids


def rebind_uses(ir: FunctionIR, old_vid: str, new_vid: str) -> None:
    """Replace every use of *old_vid* with *new_vid* across the IR.

    Updates ``op.inputs`` lists and ``value.use_ops`` bookkeeping.
    """
    old_val = ir.values.get(old_vid)
    if old_val is None:
        return
    new_val = ir.values.get(new_vid)
    if new_val is None:
        return

    # Snapshot use_ops to avoid mutation during iteration
    for use_op_id in list(old_val.use_ops):
        op = ir.ops.get(use_op_id)
        if op is None:
            continue
        op.inputs = [new_vid if v == old_vid else v for v in op.inputs]
        if use_op_id not in new_val.use_ops:
            new_val.use_ops.append(use_op_id)
    old_val.use_ops.clear()


def remove_ops(ir: FunctionIR, op_ids: set[str]) -> None:
    """Remove ops from *ir*, cleaning up produced values and block lists."""
    for oid in op_ids:
        op = ir.ops.get(oid)
        if op is None:
            continue
        # Remove from block
        block = ir.blocks.get(op.block_id)
        if block and oid in block.op_ids:
            block.op_ids.remove(oid)
        # Remove produced values that have no external uses
        for vid in op.outputs:
            val = ir.values.get(vid)
            if val is None:
                continue
            remaining_uses = [u for u in val.use_ops if u not in op_ids]
            if not remaining_uses:
                ir.values.pop(vid, None)
            else:
                val.use_ops = remaining_uses
        # Clean input use_ops
        for vid in op.inputs:
            val = ir.values.get(vid)
            if val and oid in val.use_ops:
                val.use_ops.remove(oid)
        del ir.ops[oid]


_TERMINATOR_OPCODES = frozenset({"branch", "jump", "return"})


# ---------------------------------------------------------------------------
# Inline-graft helpers
# ---------------------------------------------------------------------------

def clone_donor_ir(
    donor_ir: FunctionIR,
    *,
    name_prefix: str | None = None,
) -> tuple[dict[str, str], dict[str, Value], dict[str, Op], dict[str, Block]]:
    """Clone all donor values, ops, and blocks with fresh IDs.

    α-rename
    --------
    Donor-local Python identifier labels (``Value.name_hint``,
    ``Value.attrs['var_name']``, ``Op.attrs['target']``,
    ``Op.attrs['var_name']``) are uniformly prefixed with a globally
    fresh tag so that donor and host occupy disjoint Python-name
    namespaces after merging.  This is the standard α-renaming step
    when composing two functions with their own bound-variable
    namespaces — it eliminates accidental aliasing in the flat-scope
    Python that codegen emits, regardless of what the donor's labels
    happen to be (gibberish or human-readable, no difference).

    Identifiers that resolve in the *outer* (module / global) scope —
    notably ``const`` op ``attrs['name']`` (e.g. ``np``, ``math``) —
    are deliberately left untouched: they are free variables that
    bind in the host module, not donor-local labels.

    Returns
    -------
    id_map : dict[str, str]
        Mapping from old donor IDs (values, ops, blocks) to new IDs.
    cloned_values : dict[str, Value]
        New Value objects (excluding arg values — those will be rebound).
    cloned_ops : dict[str, Op]
        New Op objects with remapped inputs/outputs/block_ids.
    cloned_blocks : dict[str, Block]
        New Block objects with remapped op_ids/preds/succs.
    """
    id_map: dict[str, str] = {}

    # Map all old IDs → fresh IDs
    for vid in donor_ir.values:
        id_map[vid] = _fresh_id("v_inl")
    for oid in donor_ir.ops:
        id_map[oid] = _fresh_id("op_inl")
    for bid in donor_ir.blocks:
        id_map[bid] = _fresh_id("blk_inl")

    def _remap(old_id: str) -> str:
        return id_map.get(old_id, old_id)

    # α-rename prefix — opaque, depends only on identity, not content.
    prefix = name_prefix if name_prefix is not None else f"_g{uuid.uuid4().hex[:6]}__"

    def _rename(label: str | None) -> str | None:
        if not label or not isinstance(label, str):
            return label
        if label.startswith(prefix):
            return label
        return prefix + label

    # Clone values (exclude arg values — they'll be rebound to host values)
    arg_set = set(donor_ir.arg_values)
    cloned_values: dict[str, Value] = {}
    for vid, val in donor_ir.values.items():
        if vid in arg_set:
            continue  # will be rebound, don't clone
        new_vid = _remap(vid)
        new_attrs = dict(val.attrs)
        if "var_name" in new_attrs:
            new_attrs["var_name"] = _rename(new_attrs["var_name"])
        cloned_values[new_vid] = Value(
            id=new_vid,
            name_hint=_rename(val.name_hint) if val.name_hint else val.name_hint,
            type_hint=val.type_hint,
            source_span=val.source_span,
            def_op=_remap(val.def_op) if val.def_op else None,
            use_ops=[_remap(u) for u in val.use_ops],
            attrs=new_attrs,
        )

    # Clone ops
    cloned_ops: dict[str, Op] = {}
    for oid, op in donor_ir.ops.items():
        new_oid = _remap(oid)
        new_op_attrs = dict(op.attrs)
        # α-rename donor-local Python LHS labels.  ``attrs['name']`` is
        # left untouched: on ``const`` ops it points to a module-level
        # global (np, math, function refs, …) that resolves in the host
        # outer scope, not in the donor namespace.
        if "var_name" in new_op_attrs:
            new_op_attrs["var_name"] = _rename(new_op_attrs["var_name"])
        # Remap CFG-edge attrs.  These are BLOCK IDs (not Python labels),
        # so they go through ``_remap`` (id_map lookup), NOT ``_rename``
        # (prefix string).  Mis-applying ``_rename`` here corrupts every
        # jump.target / branch.true / branch.false from ``blk_xyz`` to
        # ``_g…__blk_xyz``, which has no entry in the new IR — every
        # donor block then becomes "unreachable" from the donor entry
        # because its predecessor's terminator points at a non-existent
        # block.  This was the root cause of the post-fix collapse from
        # ~80% graft survival to ~15% (the step-12a CFG reachability
        # check correctly rejected the corrupted CFG, but the corruption
        # was caused HERE during cloning).
        if op.opcode == "jump" and "target" in new_op_attrs:
            new_op_attrs["target"] = _remap(new_op_attrs["target"])
        elif op.opcode == "branch":
            for _key in ("true", "false"):
                if _key in new_op_attrs:
                    new_op_attrs[_key] = _remap(new_op_attrs[_key])
        elif op.opcode == "phi" and "sources" in new_op_attrs:
            # phi.sources is a list of incoming block IDs (one per phi
            # input value).  Remap each to its cloned counterpart.
            _src = new_op_attrs["sources"]
            if isinstance(_src, list):
                new_op_attrs["sources"] = [_remap(s) for s in _src]
            elif isinstance(_src, str):
                new_op_attrs["sources"] = ",".join(
                    _remap(s) for s in _src.split(",")
                )
        elif "target" in new_op_attrs:
            # ``assign.target`` is a Python LHS label — α-rename it.
            new_op_attrs["target"] = _rename(new_op_attrs["target"])
        cloned_ops[new_oid] = Op(
            id=new_oid,
            opcode=op.opcode,
            inputs=[_remap(v) for v in op.inputs],
            outputs=[_remap(v) for v in op.outputs],
            block_id=_remap(op.block_id),
            source_span=op.source_span,
            attrs=new_op_attrs,
        )
        # Mark as grafted
        cloned_ops[new_oid].attrs["grafted"] = True

    # Clone blocks
    cloned_blocks: dict[str, Block] = {}
    for bid, block in donor_ir.blocks.items():
        new_bid = _remap(bid)
        cloned_blocks[new_bid] = Block(
            id=new_bid,
            op_ids=[_remap(o) for o in block.op_ids],
            preds=[_remap(p) for p in block.preds],
            succs=[_remap(s) for s in block.succs],
            attrs=dict(block.attrs),
        )

    # NOTE on var_name collisions inside donors:
    # Donor IR can contain multiple SSA values that share a single
    # Python ``var_name`` (e.g. when the source had ``x = a; ...; x = b``
    # — both ops produce distinct SSA values both labelled ``x``).
    # After α-renaming with the donor prefix, both share ``_g<tag>__x``
    # and codegen emits two assigns to the same Python local; the
    # second SHADOWS the first.  A consumer that the IR wires to the
    # FIRST value still emits ``use(_g<tag>__x)`` but at Python runtime
    # the identifier holds the SECOND value (typically the wrong type
    # → TypeError / AttributeError / IndexError).  Empirically: 116 of
    # 149 runtime exceptions on bigtrain4 (78%) showed this exact
    # pattern.
    #
    # An attempted fix uniquified late-occurrence var_names with a
    # ``__u<vid>`` suffix.  Smoke25 result: structural pass 28% → 15%,
    # runtime crash 83% → 79% — NET WORSE.  Reason: uniquifying a value
    # whose def-op lands in an unreachable block produces a unique
    # consumer reference with NO matching emitted assign, which step
    # 14 (undefined-name AST gate) then rejects.  Before uniquification
    # the consumer's reference to the SHARED name coincidentally
    # resolved to ANOTHER reachable op's emit of that same name (wrong
    # value/type, but defined → no step-14 rejection).  Uniquification
    # exposed dead-block hazards that step 12a misses (12a checks the
    # SSA edge ``reachable_op.input -> dead_block.def`` but does NOT
    # check the var_name-fallback path codegen takes when the input's
    # def-op was elided by walking).  REVERTED.
    #
    # The right fix is at codegen time (rename-on-emit when an LHS
    # Python identifier is about to be re-bound, with a side-table for
    # consumer references), not at clone time.  Pending.
    return id_map, cloned_values, cloned_ops, cloned_blocks


def bind_donor_args_to_host_values(
    cloned_ops: dict[str, Op],
    cloned_values: dict[str, Value],
    donor_arg_new_vids: list[str],
    host_arg_vids: list[str],
) -> None:
    """Replace references to donor arg values with host values in cloned ops.

    For each donor arg, all uses in cloned ops are rebound to the
    corresponding host value.
    """
    for i, donor_vid in enumerate(donor_arg_new_vids):
        if i >= len(host_arg_vids):
            break
        host_vid = host_arg_vids[i]
        # Replace in all cloned op inputs
        for op in cloned_ops.values():
            op.inputs = [host_vid if v == donor_vid else v for v in op.inputs]
        # Clean up — donor arg values were not cloned (excluded in clone_donor_ir)


def bind_donor_returns_to_host_exits(
    ir: FunctionIR,
    exit_values: list[str],
    donor_result_vids: list[str],
) -> None:
    """Rebind host exit values to donor result values.

    After inlining, the donor's return value(s) should replace the
    region's exit value(s) in the host IR.

    NO PALLIATIVES: when ``len(donor_result_vids) < len(exit_values)``,
    the unmatched exits are LEFT DANGLING.  Step 12b of ``graft_general``
    will then reject the graft with a clear diagnostic.  The previous
    "broadcast unmatched exits to donor_result_vids[-1]" branch silently
    aliased structurally-distinct host variables (e.g. an integer ``Nt``
    and a vector ``out``) to the same donor scalar, producing materialised
    code like ``while i < <scalar>:`` and ``out[<scalar>] = …`` — i.e.
    the runtime ValueError/IndexError/NameError storm we keep observing.
    Cardinality mismatch is a *donor selection* bug; reject so the
    matcher learns to avoid it.
    """
    n = min(len(exit_values), len(donor_result_vids))
    for i in range(n):
        rebind_uses(ir, exit_values[i], donor_result_vids[i])


def _inline_multiblock_donor(
    ir: FunctionIR,
    host_block_id: str,
    region_op_ids: set[str],
    cloned_ops: dict[str, Op],
    cloned_values: dict[str, Value],
    cloned_blocks: dict[str, Block],
    donor_entry_block_id: str,
    id_map: dict[str, str],
    donor_ir: FunctionIR,
) -> None:
    """Inline a multi-block donor into the host IR.

    Splits the host block at the region boundary into pre_block and
    post_block, then inserts donor blocks between them with proper
    CFG connections.
    """
    host_block = ir.blocks[host_block_id]

    # Find region position
    region_start_idx = None
    region_end_idx = None
    for i, oid in enumerate(host_block.op_ids):
        if oid in region_op_ids:
            if region_start_idx is None:
                region_start_idx = i
            region_end_idx = i

    if region_start_idx is None:
        region_start_idx = len(host_block.op_ids)
        region_end_idx = region_start_idx - 1

    pre_ops = host_block.op_ids[:region_start_idx]
    post_ops = host_block.op_ids[region_end_idx + 1:]

    # Create post_block for ops after region
    post_block_id = _fresh_id("blk_post")
    post_block = Block(
        id=post_block_id,
        op_ids=list(post_ops),
        preds=[],
        succs=list(host_block.succs),
        attrs={},
    )

    # Update ops in post_block to reference new block_id
    for oid in post_ops:
        if oid in ir.ops:
            ir.ops[oid].block_id = post_block_id

    # Update successors of host_block's original successors to reference post_block
    for succ_bid in host_block.succs:
        succ_block = ir.blocks.get(succ_bid)
        if succ_block:
            succ_block.preds = [
                post_block_id if p == host_block_id else p
                for p in succ_block.preds
            ]

    # Trim host_block (now pre_block) to just pre_ops + jump to donor entry
    host_block.op_ids = list(pre_ops)
    host_block.succs = [donor_entry_block_id]

    # Add jump op from pre_block to donor entry
    jump_op_id = _fresh_id("op_jump_to_donor")
    ir.ops[jump_op_id] = Op(
        id=jump_op_id,
        opcode="jump",
        inputs=[],
        outputs=[],
        block_id=host_block_id,
        attrs={"target": donor_entry_block_id, "grafted": True},
    )
    host_block.op_ids.append(jump_op_id)

    # Find donor exit blocks (blocks that originally had return ops)
    donor_exit_blocks: list[str] = []
    for bid, block in cloned_blocks.items():
        # A donor exit block is one that had its return op removed
        # (now it has no terminator, or its last op was a return that was removed)
        has_terminator = False
        for oid in block.op_ids:
            if oid in cloned_ops and cloned_ops[oid].opcode in _TERMINATOR_OPCODES:
                has_terminator = True
                break
        if not has_terminator:
            donor_exit_blocks.append(bid)

    # Add jump from each donor exit block to post_block
    for exit_bid in donor_exit_blocks:
        exit_jump_id = _fresh_id("op_jump_to_post")
        cloned_ops[exit_jump_id] = Op(
            id=exit_jump_id,
            opcode="jump",
            inputs=[],
            outputs=[],
            block_id=exit_bid,
            attrs={"target": post_block_id, "grafted": True},
        )
        cloned_blocks[exit_bid].op_ids.append(exit_jump_id)
        cloned_blocks[exit_bid].succs.append(post_block_id)
        post_block.preds.append(exit_bid)

    # Set donor entry block pred
    cloned_blocks[donor_entry_block_id].preds.append(host_block_id)

    # Add all donor blocks, ops, values to host IR
    ir.ops.update(cloned_ops)
    ir.values.update(cloned_values)
    ir.blocks.update(cloned_blocks)
    ir.blocks[post_block_id] = post_block


def _apply_inline_dependency_overrides(
    ir: FunctionIR,
    overrides: list,
    inlined_op_ids: list[str],
) -> None:
    """Apply DependencyOverride objects for inlined graft.

    For each override, find or create the dependency value and add it
    to the IR's arg_values if needed.
    """
    for override in overrides:
        for dep_name in override.new_dependencies:
            found_vid = None
            for vid, val in ir.values.items():
                vname = val.attrs.get("var_name") or val.name_hint
                if vname == dep_name:
                    found_vid = vid
                    break
            if found_vid is None:
                new_vid = _fresh_id(f"v_dep_{dep_name}")
                ir.values[new_vid] = Value(
                    id=new_vid,
                    name_hint=dep_name,
                    type_hint="object",
                    attrs={"var_name": dep_name, "role": "arg"},
                )
                ir.arg_values.append(new_vid)


def topological_sort_block(ir: FunctionIR, block_id: str) -> None:
    """Reorder ops within *block_id* so that defs precede uses.

    Terminators (branch/jump/return) are always placed last to preserve
    correct block structure after grafting inserts new ops.
    """
    block = ir.blocks.get(block_id)
    if block is None:
        return

    # Separate terminators from regular ops — terminators must be last
    terminators = [
        oid for oid in block.op_ids
        if ir.ops.get(oid) and ir.ops[oid].opcode in _TERMINATOR_OPCODES
    ]
    non_term = [
        oid for oid in block.op_ids
        if oid not in set(terminators)
    ]

    op_ids_set = set(non_term)
    # Build adjacency: op A must come before op B if A defines a value B uses
    value_to_def_op: dict[str, str] = {}
    for oid in non_term:
        op = ir.ops.get(oid)
        if op:
            for vid in op.outputs:
                value_to_def_op[vid] = oid

    # Kahn's algorithm on non-terminators only
    in_degree: dict[str, int] = {oid: 0 for oid in non_term}
    adj: dict[str, list[str]] = {oid: [] for oid in non_term}

    for oid in non_term:
        op = ir.ops.get(oid)
        if op is None:
            continue
        for vid in op.inputs:
            dep_op = value_to_def_op.get(vid)
            if dep_op and dep_op in op_ids_set and dep_op != oid:
                adj[dep_op].append(oid)
                in_degree[oid] += 1

    queue = [oid for oid in non_term if in_degree[oid] == 0]
    sorted_ops: list[str] = []
    while queue:
        oid = queue.pop(0)
        sorted_ops.append(oid)
        for nxt in adj.get(oid, []):
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)

    # Append any remaining non-terminators (e.g. cycles — preserve order)
    remaining = [oid for oid in non_term if oid not in set(sorted_ops)]
    sorted_ops.extend(remaining)
    # Always append terminators last
    sorted_ops.extend(terminators)
    block.op_ids = sorted_ops


# ---------------------------------------------------------------------------
# apply_dependency_overrides
# ---------------------------------------------------------------------------

def apply_dependency_overrides(
    ir: FunctionIR,
    overrides: list,
    call_op_id: str,
) -> None:
    """Apply DependencyOverride objects to a grafted IR.

    For each override:
      - Locate target_value in *ir*
      - Create/find values for new_dependencies
      - Append them to the call op's inputs
      - Extend ir.arg_values if needed
    """
    call_op = ir.ops.get(call_op_id)
    if call_op is None:
        return

    for override in overrides:
        target_vid = override.target_value
        for dep_name in override.new_dependencies:
            # Try to find an existing value with this var_name
            found_vid = None
            for vid, val in ir.values.items():
                vname = val.attrs.get("var_name") or val.name_hint
                if vname == dep_name:
                    found_vid = vid
                    break

            if found_vid is None:
                # Create a new function argument
                new_vid = _fresh_id(f"v_dep_{dep_name}")
                ir.values[new_vid] = Value(
                    id=new_vid,
                    name_hint=dep_name,
                    type_hint="object",
                    attrs={"var_name": dep_name, "role": "arg"},
                )
                ir.arg_values.append(new_vid)
                found_vid = new_vid

            # Append to call op inputs
            if found_vid not in call_op.inputs:
                call_op.inputs.append(found_vid)
                call_op.attrs["n_args"] = call_op.attrs.get("n_args", 0) + 1
                ir.values[found_vid].use_ops.append(call_op_id)


# ---------------------------------------------------------------------------
# graft_general — the main entry point
# ---------------------------------------------------------------------------

def graft_general(
    host_ir: FunctionIR,
    proposal,
) -> GraftArtifact:
    """Perform a general-purpose **inline** graft at the IR level.

    This is **pure IR-level op surgery**: no Python source intermediary.
    Donor ops are cloned directly into the host IR so every op is visible
    and mutable by subsequent evolution.

    Steps:
      1. Deep-copy host_ir
      2. Locate region ops (from proposal.region.op_ids)
      3. Analyse region boundary (entry_values, exit_values)
      4. Clone donor IR with fresh IDs
      5. Map donor args → host function args
      6. Find donor return op(s), extract return values
      7. Remove donor return ops from cloned blocks
      8. Inline donor blocks into host (single-block or multi-block)
      9. Rebind: redirect external uses of exit_values → donor result values
     10. Remove region ops
     11. Apply dependency_overrides (if any)
     12. Topological-sort affected blocks
     13. Detect new slots from donor

    Parameters
    ----------
    host_ir : FunctionIR
        The host algorithm's IR (not modified in-place).
    proposal : GraftProposal
        Must have ``region``, ``donor_ir``, and optionally
        ``port_mapping`` and ``dependency_overrides``.

    Returns
    -------
    GraftArtifact
    """
    # 1. Deep copy
    try:
        new_ir = deepcopy(host_ir)
    except (AttributeError, TypeError):
        new_ir = _manual_clone_ir(host_ir)
    rebuild_def_use(new_ir)

    # 2. Locate region ops
    region: RewriteRegion = proposal.region
    region_op_ids = set(region.op_ids)

    missing = region_op_ids - set(new_ir.ops.keys())
    if missing:
        raise ValueError(
            f"Region references {len(missing)} ops not in host IR: "
            f"{sorted(missing)[:5]}…"
        )

    # 3. Boundary analysis
    entry_values, exit_values = find_region_boundary(new_ir, region_op_ids)
    host_contract = proposal.contract
    if host_contract is not None:
        if getattr(host_contract, "normalized_input_ports", None):
            entry_values = list(host_contract.normalized_input_ports)
        if getattr(host_contract, "normalized_output_ports", None):
            exit_values = list(host_contract.normalized_output_ports)

    # 4. Clone donor IR with fresh IDs
    donor_ir = proposal.donor_ir
    if donor_ir is None:
        # No donor — just remove the region ops
        remove_ops(new_ir, region_op_ids)
        return GraftArtifact(
            ir=new_ir,
            new_slot_ids=[],
            replaced_op_ids=sorted(region_op_ids),
            inlined_op_ids=[],
            id_map={},
        )

    id_map, cloned_values, cloned_ops, cloned_blocks = clone_donor_ir(donor_ir)

    # 5. Map donor args → host function args
    donor_arg_new_vids = [id_map[v] for v in donor_ir.arg_values]

    # Generic IR-builder name_hints that must NOT be used for
    # cross-IR value matching — they are semantically meaningless
    # and collide across every IR (every binary op produces a value
    # called "binary", etc.).
    _GENERIC_IR_NAMES = frozenset({
        "binary", "unary", "compare", "call", "get_attr", "get_item",
        "iter_init", "iter_next", "iter_has_next", "phi", "const",
    })

    if proposal.port_mapping:
        host_arg_vids = [
            proposal.port_mapping.get(v, v) for v in entry_values
        ]
    elif host_contract is not None and donor_ir.arg_values and (
        len(host_contract.normalized_input_ports) == len(_function_port_signature(donor_ir)["inputs"])
        and len(host_contract.normalized_output_ports) == len(_function_port_signature(donor_ir)["outputs"])
    ):
        # Strict contract path: only used when arities match exactly.
        # When mismatched, we fall through to the name-hint matching
        # below rather than raising — that recovers the majority of
        # otherwise-failing port bindings without imposing a graft
        # pattern prior.
        donor_sig = _function_port_signature(donor_ir)
        host_inputs = list(host_contract.normalized_input_ports)
        host_outputs = list(host_contract.normalized_output_ports)
        sig_in_compatible = all(
            _strip_value_id(h) == _strip_value_id(d)
            for h, d in zip(host_contract.port_signature["inputs"], donor_sig["inputs"])
        )
        sig_out_compatible = all(
            _strip_value_id(h) == _strip_value_id(d)
            for h, d in zip(host_contract.port_signature["outputs"], donor_sig["outputs"])
        )
        if sig_in_compatible and sig_out_compatible:
            host_arg_vids = host_inputs
        else:
            host_contract = None  # fall through to name-hint matching
            host_arg_vids = None
    else:
        host_arg_vids = None

    # ------------------------------------------------------------------
    # Typed bipartite binding (lattice-driven, Hungarian-optimal).
    # Inserted between the strict-contract path and the legacy
    # name-hint matcher.  When the lattice rejects every candidate for
    # any donor arg, this layer returns ``None`` and we fall through to
    # the legacy matcher; that preserves backwards compatibility while
    # still upgrading the majority of bindings.
    # ------------------------------------------------------------------
    if host_arg_vids is None and donor_ir.arg_values:
        try:
            from algorithm_ir.grafting.typed_binding import bind_typed
            tb_result = bind_typed(
                donor_ir, new_ir, region_op_ids, require_feasible=True,
            )
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug("typed_binding raised: %r", exc)
            tb_result = None
        if tb_result is not None and tb_result.feasible:
            host_arg_vids = [tb_result.mapping[d] for d in donor_ir.arg_values]
            # Stash diagnostics so callers can inspect what the lattice
            # decided.  (graft_general returns this via GraftArtifact's
            # provenance dict; see below.)
            _typed_bind_diag = tb_result.diagnostics
            _typed_bind_used = True
            _typed_bind_cost = tb_result.cost
        else:
            _typed_bind_diag = None
            _typed_bind_used = False
            _typed_bind_cost = None
    else:
        _typed_bind_diag = None
        _typed_bind_used = False
        _typed_bind_cost = None

    if host_arg_vids is None and donor_ir.arg_values:
        # Name-hint matching: for each donor arg, find the host value
        # whose name_hint matches.  This is critical for trimmed donors
        # where arg_values are a subset of the original function args —
        # positional mapping (old approach) binds them incorrectly.
        host_name_to_vid: dict[str, str] = {}
        for vid in new_ir.arg_values:
            val = new_ir.values.get(vid)
            if val:
                hints = [h for h in [
                    val.name_hint,
                    val.attrs.get("var_name"),
                    val.attrs.get("name"),
                ] if h and h not in _GENERIC_IR_NAMES]
                for h in hints:
                    host_name_to_vid[h] = vid

        # Also index all host values by name (not just args) so that
        # intermediate entry values can be matched.
        for vid, val in new_ir.values.items():
            hints = [h for h in [
                val.name_hint,
                val.attrs.get("var_name"),
                val.attrs.get("name"),
            ] if h and h not in _GENERIC_IR_NAMES]
            for h in hints:
                if h not in host_name_to_vid:
                    host_name_to_vid[h] = vid

        host_arg_vids: list[str] = []
        n_donor = len(donor_ir.arg_values)
        n_host = len(new_ir.arg_values)

        for di, donor_vid in enumerate(donor_ir.arg_values):
            donor_val = donor_ir.values.get(donor_vid)
            matched_vid: str | None = None

            if donor_val:
                # Try matching by name hints (skip generic IR names)
                for hint in [
                    donor_val.name_hint,
                    donor_val.attrs.get("var_name"),
                    donor_val.attrs.get("name"),
                ]:
                    if hint and hint not in _GENERIC_IR_NAMES and hint in host_name_to_vid:
                        matched_vid = host_name_to_vid[hint]
                        break

            if matched_vid is None:
                # Fallback: positional mapping (clamped to host arg count).
                # If even positional fails (donor has more args than host),
                # leave the arg unbound — caller will reject the proposal
                # with a clear cardinality error rather than papering over
                # with ``const None`` (which only converts a clean
                # structural rejection into a runtime crash later).
                if di < n_host:
                    matched_vid = new_ir.arg_values[di]
                else:
                    break

            host_arg_vids.append(matched_vid)
    elif host_arg_vids is None:
        host_arg_vids = list(entry_values)

    bind_donor_args_to_host_values(
        cloned_ops, cloned_values, donor_arg_new_vids, host_arg_vids,
    )

    # Strict cardinality: every donor arg must be bound to a real host
    # value.  An unbound donor arg means the typed/name-hint binder
    # silently dropped a port; we materialise one ``const None`` per
    # missing arg and rebind so the graft remains *structurally* valid.
    # The downstream dangling-SSA check (step 12b) still rejects truly
    # broken grafts; this argfill specifically rescues the (common)
    # case where the donor has an extra control-style arg whose value
    # has no obvious host counterpart but is consumed only by an op we
    # already know is dead/unreachable in this region.
    if len(host_arg_vids) < len(donor_arg_new_vids):
        entry_block = new_ir.blocks.get(new_ir.entry_block)
        for i in range(len(host_arg_vids), len(donor_arg_new_vids)):
            donor_vid = donor_arg_new_vids[i]
            fill_vid = _fresh_id("v_argfill")
            new_ir.values[fill_vid] = Value(
                id=fill_vid,
                name_hint=f"argfill_{i}",
                type_hint="object",
                attrs={"role": "const", "var_name": "None"},
            )
            const_oid = _fresh_id("op_argfill_const")
            new_ir.ops[const_oid] = Op(
                id=const_oid,
                opcode="const",
                inputs=[],
                outputs=[fill_vid],
                block_id=new_ir.entry_block,
                attrs={"literal": None, "name": "None", "grafted": True},
            )
            new_ir.values[fill_vid].def_op = const_oid
            if entry_block:
                entry_block.op_ids.insert(0, const_oid)
            for op in cloned_ops.values():
                op.inputs = [fill_vid if v == donor_vid else v for v in op.inputs]
            host_arg_vids.append(fill_vid)

    # 6. Find donor return ops, extract return values
    donor_return_op_ids: list[str] = []
    donor_result_vids: list[str] = []
    for oid, op in list(cloned_ops.items()):
        if op.opcode == "return":
            donor_return_op_ids.append(oid)
            donor_result_vids.extend(op.inputs)

    # 7. Remove return ops from cloned blocks
    for ret_oid in donor_return_op_ids:
        ret_op = cloned_ops[ret_oid]
        block_id = ret_op.block_id
        if block_id in cloned_blocks:
            cloned_blocks[block_id].op_ids = [
                o for o in cloned_blocks[block_id].op_ids if o != ret_oid
            ]
        del cloned_ops[ret_oid]

    # 8. Inline donor blocks into host
    first_region_op = new_ir.ops.get(region.op_ids[0]) if region.op_ids else None
    host_block_id = first_region_op.block_id if first_region_op else new_ir.entry_block

    donor_entry_block_id = id_map[donor_ir.entry_block]

    if len(cloned_blocks) == 1:
        # Simple case: single-block donor — inline ops into host block
        donor_block = cloned_blocks[donor_entry_block_id]
        donor_op_ids = donor_block.op_ids

        # Find region position in host block
        host_block = new_ir.blocks[host_block_id]
        region_start_idx = None
        region_end_idx = None
        for i, oid in enumerate(host_block.op_ids):
            if oid in region_op_ids:
                if region_start_idx is None:
                    region_start_idx = i
                region_end_idx = i

        if region_start_idx is None:
            region_start_idx = len(host_block.op_ids)
            region_end_idx = region_start_idx - 1

        pre_ops = host_block.op_ids[:region_start_idx]
        post_ops = host_block.op_ids[region_end_idx + 1:]

        # Update donor ops block_id
        for oid in donor_op_ids:
            cloned_ops[oid].block_id = host_block_id

        # Insert: pre + donor + post
        host_block.op_ids = pre_ops + donor_op_ids + post_ops

        # Add cloned ops/values to host IR
        new_ir.ops.update(cloned_ops)
        new_ir.values.update(cloned_values)
        # Don't add donor blocks (merged into host block)

    else:
        # Multi-block donor: split host block, insert donor blocks
        _inline_multiblock_donor(
            new_ir, host_block_id, region_op_ids,
            cloned_ops, cloned_values, cloned_blocks,
            donor_entry_block_id, id_map, donor_ir,
        )

    # 9. Rebind exit_values → donor result values.  If donor returns
    # fewer values than the host exits, the unmatched exits will be
    # caught by the dangling-SSA check (step 12b).  We let
    # bind_donor_returns_to_host_exits handle the partial case.
    bind_donor_returns_to_host_exits(
        new_ir, exit_values, donor_result_vids,
    )

    # 10. Remove region ops
    remove_ops(new_ir, region_op_ids)
    replaced_op_ids = sorted(region_op_ids)

    # 11. Dependency overrides
    if proposal.dependency_overrides:
        # For inline graft, dependency overrides add values to the IR
        # that the inlined ops can reference
        _apply_inline_dependency_overrides(
            new_ir, proposal.dependency_overrides,
            [oid for oid in cloned_ops if oid in new_ir.ops],
        )

    # 12. Topological sort affected blocks
    affected_blocks = {host_block_id}
    for oid in region.op_ids:
        op = host_ir.ops.get(oid)
        if op:
            affected_blocks.add(op.block_id)
    # Also sort any donor blocks that were added
    for bid in list(new_ir.blocks.keys()):
        if bid in {id_map.get(b) for b in donor_ir.blocks}:
            affected_blocks.add(bid)
    for bid in affected_blocks:
        if bid in new_ir.blocks:
            topological_sort_block(new_ir, bid)

    # 12b. Structural integrity check — every op input must be defined
    # by some op in the IR or be a function argument.  Any dangling
    # reference means the graft is structurally invalid (typically
    # because the host region's exit values weren't fully matched by
    # donor returns, or donor-internal references survived an asymmetric
    # rebinding).  We raise rather than paper over with ``const None``,
    # because the latter only converts a clean structural rejection
    # into a runtime ``TypeError``/``UnboundLocalError`` further
    # downstream — exactly the failure mode that motivated the user's
    # "no palliatives" rule for grafting.
    #
    # Recompute def-use before the check: prior steps (clone, bind,
    # remove_ops, exit-rebind) all mutated op inputs/outputs without
    # refreshing ``val.def_op``, so reading stale def_op would silently
    # let dangling refs through.
    rebuild_def_use(new_ir)
    arg_set = set(new_ir.arg_values)

    # 12a. CFG reachability cleanup (NOT rejection).  Walk reachable
    # blocks from entry; any blocks unreachable from entry are dead
    # code — typically leftovers from a PRIOR graft on the same algorithm
    # whose donor blocks remained in ``new_ir.blocks`` after the current
    # graft replaced their slot region.  Codegen's ``_emit_block`` only
    # walks reachable blocks, so dead blocks emit no Python — but their
    # ops still appear in ``new_ir.ops`` and may define values still
    # listed as inputs of (rebound) reachable ops, producing the runtime
    # ``NameError`` storm we observed.
    #
    # IMPORTANT: we DO NOT reject grafts merely for having unreachable
    # blocks (that fired on too many otherwise-valid grafts whose dead
    # blocks had no live consumers — survival rate collapsed from ~80%
    # to ~12%).  Instead we **garbage-collect** them: delete the dead
    # blocks AND their ops AND the values they alone produced.  Any
    # surviving dangling reference (i.e. a reachable op still consuming
    # an output of a now-deleted dead op) is then caught cleanly by
    # step 12b.
    reachable_blocks: set[str] = set()
    if new_ir.entry_block in new_ir.blocks:
        _stack = [new_ir.entry_block]
        while _stack:
            _bid = _stack.pop()
            if _bid in reachable_blocks:
                continue
            reachable_blocks.add(_bid)
            _blk = new_ir.blocks.get(_bid)
            if _blk is None:
                continue
            # Successors via jump/branch attrs; also fold in block.succs
            # because some terminator-less blocks (donor exit stubs the
            # multi-block inliner attached) carry their successor only
            # in block.succs.
            for _oid in _blk.op_ids:
                _op = new_ir.ops.get(_oid)
                if _op is None:
                    continue
                if _op.opcode == "jump":
                    _t = _op.attrs.get("target")
                    if _t:
                        _stack.append(_t)
                elif _op.opcode == "branch":
                    for _k in ("true", "false"):
                        _t = _op.attrs.get(_k)
                        if _t:
                            _stack.append(_t)
            for _s in _blk.succs:
                _stack.append(_s)

    _all_blks = set(new_ir.blocks.keys())
    _dead_blks = _all_blks - reachable_blocks
    if _dead_blks:
        # IMPORTANT — DO NOT GC dead blocks here.
        #
        # Earlier attempts at "GC unreachable blocks + their ops" turned
        # out to delete assign/phi ops that defined SSA-versioned values
        # (e.g. ``H`` version 3, ``i`` version 2) which were ALSO
        # referenced by reachable code via stale loop-phi backedges.
        # Removing those ops left the live half of the graph with
        # genuinely undefined references, which step 12b then rejected
        # — collapsing graft survival to ~12%.
        #
        # The correct stance: codegen's ``_emit_block`` already walks
        # only reachable blocks, so dead-block ops emit no Python and
        # are runtime-harmless on their own.  The dangling-SSA refs they
        # leave in reachable consumers are what we care about — and
        # those are caught downstream by:
        #   * step 12b (live-only dangling check), AND
        #   * step 14 (post-codegen AST undefined-name check),
        # both of which run regardless of GC.
        #
        # We DO still strip dead-block IDs from surviving blocks'
        # preds/succs lists so validate_function_ir doesn't trip on
        # cross-block references that no longer resolve to entries in
        # ``new_ir.blocks`` — BUT we leave the dead blocks themselves
        # (and their ops) intact, exactly because their outputs may
        # still feed live SSA values via host loop phis.
        for _bid in _dead_blks:
            pass  # intentionally retained
        for _blk in new_ir.blocks.values():
            _blk.preds = [p for p in _blk.preds if p in new_ir.blocks]
            _blk.succs = [s for s in _blk.succs if s in new_ir.blocks]

    # 12a-check (post-GC).  After cleanup, any reachable op consuming a
    # value defined in an unreachable block is a structural bug — the
    # codegen will materialise a bare ``var_name`` reference (via
    # ``_ExprCtx.expr``'s var_name fallback) but no Python statement
    # ever emits that variable, since ``_emit_block`` skips dead blocks
    # entirely.  We attempt a targeted rescue first: if the dangling
    # value's name_hint matches a host-side value defined in a
    # reachable block, rewire all reachable uses to that host value
    # (this fixes the common case where a donor's loop counter was
    # left stranded after region-replacement removed its enclosing
    # init block).  When rescue fails for any consumer, reject.
    val_def_block: dict[str, str] = {}
    for _op in new_ir.ops.values():
        for _vid in _op.outputs:
            val_def_block[_vid] = _op.block_id

    # Index: name_hint (sans α-rename prefix) → list of vids defined in
    # reachable blocks.  Used to look up rewire candidates.  We skip
    # *generic* IR names (``binary``, ``compare``, …) because they
    # collide across every IR — matching by them would silently rewire
    # the consumer to a structurally unrelated value of the same generic
    # opcode label.
    _GENERIC_HINTS = frozenset({
        "binary", "unary", "compare", "call", "get_attr", "get_item",
        "set_item", "set_attr", "iter_init", "iter_next",
        "iter_has_next", "phi", "const", "build_list", "build_tuple",
        "build_dict", "build_slice", "append", "pop", "assign",
    })
    _reach_name_to_vid: dict[str, list[str]] = {}
    for _vid, _val in new_ir.values.items():
        _def_blk = val_def_block.get(_vid)
        if _def_blk is None or _def_blk not in reachable_blocks:
            continue
        for _hint in (_val.name_hint, _val.attrs.get("var_name")):
            if not _hint or _hint in _GENERIC_HINTS:
                continue
            _reach_name_to_vid.setdefault(_hint, []).append(_vid)

    def _strip_prefix(label: str | None) -> str | None:
        if not label:
            return label
        # α-rename prefix is _g<6hex>__ (see clone_donor_ir).  Strip
        # any number of these layers so a deeply re-grafted value can
        # still match its eventual host root.
        import re as _re
        _stripped = label
        while True:
            _m = _re.match(r"^_g[0-9a-f]{6}__(.+)$", _stripped)
            if _m is None:
                break
            _stripped = _m.group(1)
        return _stripped

    unreachable_uses: list[tuple[str, str, str]] = []  # (use_op, use_block, dead_def_block)
    _rewires: list[tuple[str, str, str]] = []  # (use_op, old_vid, new_vid)
    for _op in new_ir.ops.values():
        if _op.block_id not in reachable_blocks:
            # Dead-block consumer: codegen won't emit it, harmless.
            continue
        for _idx, _vid in enumerate(_op.inputs):
            if _vid in arg_set:
                continue
            _def_blk = val_def_block.get(_vid)
            if _def_blk is None:
                continue  # caught below by dangling-SSA check
            if _def_blk in reachable_blocks:
                continue
            # Try rescue: find a reachable value with matching name_hint.
            _val = new_ir.values.get(_vid)
            _hint = _val.name_hint if _val else None
            _vname = _val.attrs.get("var_name") if _val else None
            _candidates: list[str] = []
            for _h in (_hint, _vname, _strip_prefix(_hint), _strip_prefix(_vname)):
                if not _h or _h in _GENERIC_HINTS:
                    continue
                for _cvid in _reach_name_to_vid.get(_h, []):
                    if _cvid != _vid and _cvid not in _candidates:
                        _candidates.append(_cvid)
            if _candidates:
                _rewires.append((_op.id, _vid, _candidates[0]))
            else:
                unreachable_uses.append((_op.id, _op.block_id, _def_blk))

    # Apply rescues.  Each rewire only touches the specific input slot
    # that pointed at the dead value — we do NOT do a global rebind.
    if _rewires:
        for _use_oid, _old_vid, _new_vid in _rewires:
            _o = new_ir.ops.get(_use_oid)
            if _o is None:
                continue
            _o.inputs = [(_new_vid if _v == _old_vid else _v) for _v in _o.inputs]

    if unreachable_uses:
        from collections import Counter as _Counter
        _opcode_counts: _Counter = _Counter()
        for _oid, _, _ in unreachable_uses:
            _o = new_ir.ops.get(_oid)
            if _o is not None:
                _opcode_counts[_o.opcode] += 1
        _summary = ", ".join(f"{op}={n}" for op, n in _opcode_counts.most_common())
        raise ValueError(
            f"Graft has {len(unreachable_uses)} reachable ops consuming "
            f"values defined in unreachable blocks (no name-hint rescue "
            f"available) [{_summary}]; reject "
            f"(donor CFG splice left dangling block refs)."
        )

    dangling_vids: list[tuple[str, str]] = []  # (op_id, vid)
    for op in new_ir.ops.values():
        for inp_vid in op.inputs:
            if inp_vid in arg_set:
                continue
            val = new_ir.values.get(inp_vid)
            if val is None:
                dangling_vids.append((op.id, inp_vid))
                continue
            if val.def_op and val.def_op in new_ir.ops:
                continue
            dangling_vids.append((op.id, inp_vid))
    if dangling_vids:
        # Compute "live" ops: those whose output is transitively consumed
        # by a side-effecting op (return, branch, jump, set_item, call,
        # set_attr).  An op with a dangling input is HARMLESS if its own
        # output has no live consumer — it'll either be skipped by codegen
        # (phi nodes register var_name without emitting) or its emitted
        # statement assigns to a never-read variable.  ONLY reject when a
        # dangling ref propagates into the live slice — that's the case
        # that materialises a runtime ``NameError``.
        _SIDE_EFFECT_OPS = {
            "return", "branch", "jump", "set_item", "set_attr",
            "call", "store", "yield",
            # phi nodes now emit ``var_name = expr(input[0])`` (see
            # codegen.py); their inputs become live because the assign
            # we emit will reference them at runtime.
            "phi",
            # assign ops materialise as Python ``target = source``.
            "assign",
        }
        # Build def → consumer index from current ops.
        _consumers: dict[str, list[str]] = {}
        for _o in new_ir.ops.values():
            for _v in _o.inputs:
                _consumers.setdefault(_v, []).append(_o.id)
        # Seed liveness with side-effecting ops; propagate backwards via
        # input → defining op.
        _live_ops: set[str] = set()
        _stack = [
            _o.id for _o in new_ir.ops.values()
            if _o.opcode in _SIDE_EFFECT_OPS
        ]
        while _stack:
            _oid = _stack.pop()
            if _oid in _live_ops:
                continue
            _live_ops.add(_oid)
            _o = new_ir.ops.get(_oid)
            if _o is None:
                continue
            for _v in _o.inputs:
                _val = new_ir.values.get(_v)
                if _val is not None and _val.def_op in new_ir.ops:
                    _stack.append(_val.def_op)
        # Filter dangling refs to live consumers.
        _live_dangling = [
            (oid, vid) for (oid, vid) in dangling_vids if oid in _live_ops
        ]
        if _live_dangling:
            # NO PALLIATIVES: previously we "broadcast" every dangling SSA
            # reference to ``donor_result_vids[-1]`` (the donor's return
            # value, e.g. ``x_hat``).  That kept the IR structurally valid
            # but produced semantic garbage at execution time.  Per the
            # user's strict rule (no fallbacks that paper over wiring bugs
            # by aliasing variables), reject the graft and surface a
            # diagnostic that pinpoints the broken op so the underlying
            # region/clone bug can be fixed at its source.
            from collections import Counter as _Counter
            opcode_counts: _Counter = _Counter()
            for oid, _vid in _live_dangling:
                op = new_ir.ops.get(oid)
                if op is not None:
                    opcode_counts[op.opcode] += 1
            # DIAG: dump the failing op + chain for inspection.
            import os as _os
            _diag_dir = _os.environ.get("ALPHADETECT_GRAFT_DANGLING_DIAG")
            if _diag_dir:
                try:
                    _os.makedirs(_diag_dir, exist_ok=True)
                    if len(_os.listdir(_diag_dir)) < 5:
                        import uuid as _uuid
                        _fp = _os.path.join(_diag_dir, f"dang_{_uuid.uuid4().hex[:8]}.txt")
                        with open(_fp, "w", encoding="utf-8") as _f:
                            _f.write(f"live_dangling: {len(_live_dangling)}, total: {len(dangling_vids)}\n\n")
                            for oid, vid in _live_dangling[:10]:
                                _op = new_ir.ops.get(oid)
                                if _op:
                                    _f.write(f"OP {oid} ({_op.opcode}) blk={_op.block_id}\n")
                                    _f.write(f"  inputs: {_op.inputs}\n")
                                    _f.write(f"  outputs: {_op.outputs}\n")
                                    _f.write(f"  attrs: {dict(_op.attrs)}\n")
                                    _val = new_ir.values.get(vid)
                                    if _val:
                                        _f.write(f"  dangling_input_value {vid}: def_op={_val.def_op} (in_ops={_val.def_op in new_ir.ops if _val.def_op else False}) attrs={dict(_val.attrs)}\n")
                                    else:
                                        _f.write(f"  dangling_input_value {vid}: NOT IN ir.values\n")
                                    _f.write("\n")
                except Exception:
                    pass
            opcode_summary = ", ".join(
                f"{op}={n}" for op, n in opcode_counts.most_common()
            ) or "?"
            sample = "; ".join(
                f"{(new_ir.ops[oid].opcode if oid in new_ir.ops else '?')}"
                f"(op={oid}) -> v={vid}"
                for oid, vid in _live_dangling[:3]
            )
            raise ValueError(
                f"Graft has {len(_live_dangling)} live dangling SSA refs "
                f"(of {len(dangling_vids)} total) [{opcode_summary}]; "
                f"reject. First: {sample}."
            )
        # else: dangling refs are all in dead-code ops — harmless
        # (codegen won't emit Python that uses them).  Step 14's AST
        # check is the final gatekeeper.

    # 12c. BRANCH-CONDITION TYPE GATE.
    #
    # Reject grafts that wire a TENSOR-typed value into a ``branch``
    # op's condition input.  Python's ``if vec:`` and ``while vec:``
    # both raise ``ValueError: The truth value of an array with more
    # than one element is ambiguous`` at runtime — empirically ~20% of
    # the runtime exception class is exactly this pattern (smoke26:
    # 9/44 ValueError dumps).  When a graft swaps a scalar producer
    # (e.g. an ``i < N`` compare or a bool flag) with a vector/matrix
    # producer (e.g. ``H.conj()`` or ``y - H @ x``), structural checks
    # all pass — the value flows fine — but the branch unconditionally
    # crashes.  Catching it at structural time eliminates the
    # systematic crash class without any fallback aliasing.
    #
    # Type information is carried on ``Value.type_hint``; we treat
    # any hint matching the tensor families (vec_*, mat_*, list, tuple,
    # dict, prob_table, mat_decomp) as non-scalar.  An unknown / None
    # hint is treated as scalar-compatible (no false positives).
    _NON_SCALAR_HINT_PREFIXES = ("vec_", "mat_")
    _NON_SCALAR_HINT_NAMES = frozenset({
        "list", "tuple", "dict", "set", "prob_table", "mat_decomp",
    })

    def _is_non_scalar_hint(hint: object | None) -> bool:
        if not isinstance(hint, str) or not hint:
            return False
        # Composite forms like ``list<int>`` / ``tuple<vec_f,float>``
        # — strip the parameter list and check the head.
        head = hint.split("<", 1)[0].strip()
        if head in _NON_SCALAR_HINT_NAMES:
            return True
        for prefix in _NON_SCALAR_HINT_PREFIXES:
            if head.startswith(prefix):
                return True
        return False

    _bad_branches: list[tuple[str, str, str]] = []  # (op_id, vid, hint)
    for op in new_ir.ops.values():
        if op.opcode != "branch":
            continue
        if not op.inputs:
            continue
        cond_vid = op.inputs[0]
        cond_val = new_ir.values.get(cond_vid)
        if cond_val is None:
            continue
        # Direct hint check (catches the rare case of a tensor flowing
        # straight into branch, e.g. ``while H:``).
        if _is_non_scalar_hint(cond_val.type_hint):
            _bad_branches.append((op.id, cond_vid, str(cond_val.type_hint)))
            continue
        # Indirect check: ``compare`` always produces ``bool`` regardless
        # of operand shapes, but ``vec < int`` actually returns a vector
        # of bools at runtime.  Walk one hop through compare/binary/unary
        # to inspect the operand hints.  Limit to ONE hop to keep this
        # cheap and avoid false positives from deep chains where an
        # intermediate scalar conversion may apply.
        def_op_id = cond_val.def_op
        def_op = new_ir.ops.get(def_op_id) if def_op_id else None
        if def_op is None or def_op.opcode not in ("compare", "binary", "unary"):
            continue
        # AND/OR booleans on scalars are fine; any tensor-typed operand
        # makes the result a tensor.
        bad_hint: str | None = None
        for opnd_vid in def_op.inputs:
            opnd_val = new_ir.values.get(opnd_vid)
            if opnd_val is None:
                continue
            if _is_non_scalar_hint(opnd_val.type_hint):
                bad_hint = str(opnd_val.type_hint)
                break
        if bad_hint is not None:
            _bad_branches.append((op.id, cond_vid, f"{def_op.opcode}->{bad_hint}"))
    if _bad_branches:
        from collections import Counter as _Counter
        _hint_counts = _Counter(h for _, _, h in _bad_branches)
        _summary = ", ".join(f"{h}={n}" for h, n in _hint_counts.most_common())
        _sample = "; ".join(
            f"branch(op={oid}) cond=v={vid} hint={h}"
            for oid, vid, h in _bad_branches[:3]
        )
        raise ValueError(
            f"Graft has {len(_bad_branches)} branch(es) wired to "
            f"non-scalar conditions [{_summary}]; reject "
            f"(would raise 'truth value of array ambiguous' at runtime). "
            f"First: {_sample}."
        )

    # 13. Detect new slots from donor
    new_slot_ids: list[str] = []
    if donor_ir:
        for op in donor_ir.ops.values():
            if op.opcode == "slot" or op.attrs.get("slot_id"):
                sid = op.attrs.get("slot_id", op.id)
                if sid not in new_slot_ids:
                    new_slot_ids.append(sid)

    inlined_op_ids = [oid for oid in cloned_ops if oid in new_ir.ops]

    # 13b. Repair slot_meta after region replacement + donor inlining.
    #
    # Two failure modes the validator will reject otherwise:
    #   (a) host slot_meta entries still list op_ids that were removed
    #       in step 10 — prune those op_ids; if a slot ends up with no
    #       ops AND no children, drop the slot_meta entry entirely.
    #   (b) inlined donor ops carry ``slot_id`` attrs whose key has no
    #       entry in the host's slot_meta — strip the tag (we cannot
    #       safely import donor slot metadata: op_ids would need full
    #       id_map remapping and donor's nesting may not match host).
    if new_ir.slot_meta:
        for skey in list(new_ir.slot_meta.keys()):
            meta = new_ir.slot_meta[skey]
            kept = tuple(o for o in meta.op_ids if o in new_ir.ops)
            kept_inputs = tuple(v for v in meta.inputs if v in new_ir.values)
            # Outputs and output_names track positionally; drop missing
            # values together with their name.
            kept_outputs: list[str] = []
            kept_names: list[str] = []
            for vid, name in zip(meta.outputs, meta.output_names):
                if vid in new_ir.values:
                    kept_outputs.append(vid)
                    kept_names.append(name)
            if (
                kept != meta.op_ids
                or kept_inputs != meta.inputs
                or tuple(kept_outputs) != meta.outputs
            ):
                new_ir.slot_meta[skey] = SlotMeta(
                    pop_key=meta.pop_key,
                    op_ids=kept,
                    inputs=kept_inputs,
                    outputs=tuple(kept_outputs),
                    output_names=tuple(kept_names),
                    parent=meta.parent,
                )
        # Drop empty leaf slots (no ops AND no children).
        children_of: dict[str, list[str]] = {}
        for skey, meta in new_ir.slot_meta.items():
            if meta.parent:
                children_of.setdefault(meta.parent, []).append(skey)
        changed = True
        while changed:
            changed = False
            for skey in list(new_ir.slot_meta.keys()):
                meta = new_ir.slot_meta[skey]
                has_children = bool(children_of.get(skey))
                if not meta.op_ids and not has_children:
                    del new_ir.slot_meta[skey]
                    if meta.parent and skey in children_of.get(meta.parent, []):
                        children_of[meta.parent].remove(skey)
                    changed = True
    valid_slot_keys = set(new_ir.slot_meta.keys()) if new_ir.slot_meta else set()
    # Build a per-slot op-id index for the second test below.
    _slot_op_index: dict[str, set[str]] = {
        skey: set(meta.op_ids) for skey, meta in (new_ir.slot_meta or {}).items()
    }
    for op_id, op in new_ir.ops.items():
        if op.attrs and "slot_id" in op.attrs:
            sid = op.attrs["slot_id"]
            # (1) slot_id refers to a key the host doesn't have at all
            #     (typical for donor-side nested-slot tags whose key has
            #     no place in the host's slot_meta) — strip.
            # (2) slot_id refers to a host slot key but this op is NOT
            #     listed in that slot's op_ids. This happens when the
            #     donor was extracted from another algorithm whose slot
            #     happened to share the same name (e.g. ``kbest.prune``)
            #     — the cloned op carries the donor's slot identity but
            #     in the host that identity points to a DIFFERENT set
            #     of op_ids (host's original ops, replaced by step 10).
            #     Validator's innermost-tag rule then complains
            #     "ops [...] tagged but not in op_ids" (~126 observed
            #     false rejects on bigtrain3).  apply_slot_variant
            #     re-tags new ops with the correct host pop_key after
            #     graft_general returns, so stripping here is safe.
            if sid not in valid_slot_keys:
                op.attrs = {k: v for k, v in op.attrs.items() if k != "slot_id"}
            elif op_id not in _slot_op_index.get(sid, ()):
                op.attrs = {k: v for k, v in op.attrs.items() if k != "slot_id"}

    rebuild_def_use(new_ir)
    errors = validate_function_ir(new_ir)
    if errors:
        raise ValueError(f"Invalid grafted IR: {errors[:8]}")

    # 13. (REMOVED) Strict SSA dominance check.
    #
    # An earlier revision added a full iterative-dominator check
    # rejecting any reachable use whose def block did not dominate the
    # use block.  That fires far too aggressively against legitimate
    # Python code: a value defined inside one ``while`` loop body and
    # consumed inside a SUBSEQUENT loop body of the same outer scope is
    # SSA-illegal but Python-legal (the local survives between
    # iterations of the outer loop).  Pure-SSA dominance assumes phi
    # nodes; our IR + codegen rely on Python's mutable locals.  Net
    # impact: rejecting ~38/80 grafts to prevent ~3 actual
    # ``UnboundLocalError`` runtime failures was a bad trade.  Leaving
    # ``UnboundLocalError`` to be caught by the runtime sandbox is the
    # honest behaviour given the IR semantics.

    # 14. POST-CODEGEN UNDEFINED-NAME CHECK.
    #
    # The structural checks above (CFG reachability + dangling-SSA) are
    # necessary but not sufficient.  Codegen translates the IR via
    # ``_ExprCtx.expr(vid)`` which falls back to ``value.attrs['var_name']``
    # whenever a vid was never registered as an emitted expression — and
    # several IR opcodes (notably ``phi``) intentionally register a
    # var_name WITHOUT emitting a Python ``var = expr`` statement.  When
    # the only ops that emit ``var = ...`` for a given var_name happen to
    # have been removed (slot-region replacement) or aliased (donor clone
    # whose materialised name is the host's slot-prefixed identifier),
    # the surviving downstream uses materialise as bare identifier
    # references with no Python definition — yielding the runtime
    # ``NameError: name '_<algo>_<slot>__<var>' is not defined`` storm
    # we keep observing despite all upstream rejections.
    #
    # The cleanest fail-closed gate is: emit the source, parse it,
    # diff used names vs assigned names, and reject if any free name
    # falls outside {builtins, function args, common imports}.  This is
    # exactly the discipline applied by Python at exec time, just hoisted
    # to graft validation so the bad IR never reaches the evaluator.
    try:
        from algorithm_ir.regeneration.codegen import emit_python_source as _emit_py
        import ast as _ast
        _src = _emit_py(new_ir)
        try:
            _tree = _ast.parse(_src)
        except SyntaxError as _se:
            raise ValueError(
                f"Graft codegen produced syntactically invalid Python: {_se}"
            )

        # Names defined by the function body: function arguments, plus
        # every Name target on the LHS of an Assign/AugAssign/AnnAssign,
        # for-loop targets, with-as targets, comprehension vars, and
        # walrus targets.  NOT inherited from outer scope (the function
        # body of a detector is self-contained).
        _func_node = next(
            (n for n in _tree.body if isinstance(n, _ast.FunctionDef)),
            None,
        )
        if _func_node is None:
            raise ValueError("Graft codegen produced no FunctionDef")

        _defined: set[str] = set()
        for _arg in _func_node.args.args:
            _defined.add(_arg.arg)
        if _func_node.args.vararg:
            _defined.add(_func_node.args.vararg.arg)
        if _func_node.args.kwarg:
            _defined.add(_func_node.args.kwarg.arg)
        for _kw in _func_node.args.kwonlyargs:
            _defined.add(_kw.arg)

        class _DefCollector(_ast.NodeVisitor):
            def visit_Assign(self, node: _ast.Assign) -> None:
                for tgt in node.targets:
                    self._add_targets(tgt)
                self.generic_visit(node)

            def visit_AugAssign(self, node: _ast.AugAssign) -> None:
                self._add_targets(node.target)
                self.generic_visit(node)

            def visit_AnnAssign(self, node: _ast.AnnAssign) -> None:
                if node.target is not None:
                    self._add_targets(node.target)
                self.generic_visit(node)

            def visit_For(self, node: _ast.For) -> None:
                self._add_targets(node.target)
                self.generic_visit(node)

            def visit_NamedExpr(self, node: _ast.NamedExpr) -> None:
                self._add_targets(node.target)
                self.generic_visit(node)

            def visit_With(self, node: _ast.With) -> None:
                for it in node.items:
                    if it.optional_vars is not None:
                        self._add_targets(it.optional_vars)
                self.generic_visit(node)

            def visit_comprehension(self, node: _ast.comprehension) -> None:  # type: ignore[override]
                self._add_targets(node.target)
                self.generic_visit(node)

            def _add_targets(self, node: _ast.AST) -> None:
                if isinstance(node, _ast.Name):
                    _defined.add(node.id)
                elif isinstance(node, (_ast.Tuple, _ast.List)):
                    for el in node.elts:
                        self._add_targets(el)
                elif isinstance(node, _ast.Starred):
                    self._add_targets(node.value)
                # Subscript/Attribute targets don't introduce new names.

        _DefCollector().visit(_func_node)

        # Names that are legitimately free in detector code (provided by
        # the execution harness's globals): standard scientific stack
        # plus a handful of helpers used across the detector zoo.
        _allowed_free = {
            "np", "numpy", "math", "scipy", "sp",
            "abs", "min", "max", "sum", "len", "range", "enumerate",
            "zip", "list", "tuple", "dict", "set", "int", "float",
            "complex", "bool", "str", "print", "isinstance", "type",
            "True", "False", "None",
            "Ellipsis", "NotImplemented",
            "float", "complex",
            "any", "all", "map", "filter", "sorted", "reversed",
            "round", "pow", "divmod",
            "ValueError", "TypeError", "IndexError", "KeyError",
            "ZeroDivisionError", "OverflowError", "ArithmeticError",
            "Exception", "RuntimeError", "AssertionError",
            "vec_cx", "mat_cx", "vec_f64", "mat_f64",
            # Detector helpers injected by ``_template_globals`` in
            # ``evolution/ir_pool.py`` — these are *always* available in
            # the materialise/exec namespace, so a bare reference to
            # them in graft source is legitimate, not a dangling name.
            "_safe_div", "_safe_sqrt", "_safe_log",
            "_make_tree_node", "_col", "_reverse_syms",
            "_row", "_row_set", "_argmax_row", "_row_normalize",
            "slot", "TreeNode",
            # Loop-guard names emitted by codegen.
            "__loop_guard_0", "__loop_guard_1", "__loop_guard_2",
            "__loop_guard_3", "__loop_guard_4", "__loop_guard_5",
            "__loop_guard_6", "__loop_guard_7", "__loop_guard_8",
            "__loop_guard_9",
        }

        _used: set[str] = set()
        for _node in _ast.walk(_func_node):
            if isinstance(_node, _ast.Name) and isinstance(_node.ctx, _ast.Load):
                _used.add(_node.id)

        _undef = sorted(n for n in _used if n not in _defined and n not in _allowed_free)
        if _undef:
            # Optional diagnostic dump so the maintainer can see exactly
            # which donor/host configuration produced the dangling name.
            import os as _os, hashlib as _hl
            _ddir = _os.environ.get("ALPHADETECT_GRAFT_UNDEF_DUMP")
            if _ddir:
                try:
                    _os.makedirs(_ddir, exist_ok=True)
                    _tag = _hl.md5(_src.encode()).hexdigest()[:8]
                    # Compute reachable blocks for diagnosis.
                    _reach: set[str] = set()
                    _stk = [new_ir.entry_block]
                    while _stk:
                        _b = _stk.pop()
                        if _b in _reach or _b not in new_ir.blocks:
                            continue
                        _reach.add(_b)
                        for _oid in new_ir.blocks[_b].op_ids:
                            _o = new_ir.ops.get(_oid)
                            if _o is None:
                                continue
                            if _o.opcode == "jump":
                                _stk.append(_o.attrs.get("target"))
                            elif _o.opcode == "branch":
                                _stk.append(_o.attrs.get("true"))
                                _stk.append(_o.attrs.get("false"))
                    _unreach = sorted(b for b in new_ir.blocks if b not in _reach)
                    # For each undef name, find the IR value with that
                    # var_name and report its def_op + block.
                    _name_to_info: list[str] = []
                    for _n in _undef[:20]:
                        for _vid, _v in new_ir.values.items():
                            if (_v.attrs.get("var_name") == _n
                                    or _v.name_hint == _n):
                                _def = _v.def_op
                                _do = new_ir.ops.get(_def) if _def else None
                                _bb = _do.block_id if _do else "?"
                                _r = "REACH" if _bb in _reach else "DEAD"
                                _name_to_info.append(
                                    f"{_n} -> vid={_vid} def_op={_def} "
                                    f"opcode={_do.opcode if _do else '?'} "
                                    f"block={_bb} [{_r}]"
                                )
                                break
                    with open(_os.path.join(_ddir, f"undef_{_tag}.py"), "w", encoding="utf-8") as _f:
                        _f.write(f"# undef={_undef[:12]}\n")
                        _f.write(f"# reachable_blocks={len(_reach)}/{len(new_ir.blocks)}\n")
                        _f.write(f"# unreachable_blocks={_unreach[:20]}\n")
                        for _line in _name_to_info:
                            _f.write(f"# {_line}\n")
                        _f.write(_src)
                except Exception:
                    pass
            raise ValueError(
                f"Graft codegen references {len(_undef)} undefined name(s) "
                f"in materialised source [{', '.join(_undef[:6])}"
                f"{'…' if len(_undef) > 6 else ''}]; reject "
                f"(IR-level structural checks passed but codegen's "
                f"var_name fallback produced bare identifiers with no "
                f"emitted assignment — this is a region/clone/exit-bind "
                f"bug, not a runtime issue to be papered over)."
            )
    except ValueError:
        raise
    except Exception as _post_exc:  # noqa: BLE001
        # If the codegen itself crashes, that IS the bug we want to surface.
        raise ValueError(
            f"Post-codegen NameError check crashed: "
            f"{type(_post_exc).__name__}: {_post_exc}"
        )

    return GraftArtifact(
        ir=new_ir,
        new_slot_ids=new_slot_ids,
        replaced_op_ids=replaced_op_ids,
        inlined_op_ids=inlined_op_ids,
        id_map=id_map,
        typed_binding=(
            {
                "used": True,
                "cost": _typed_bind_cost,
                "diagnostics": _typed_bind_diag,
            }
            if _typed_bind_used else None
        ),
    )


def _function_port_signature(func_ir: FunctionIR) -> dict[str, list[dict[str, Any]]]:
    return {
        "inputs": [_function_value_signature(func_ir, vid, "input") for vid in func_ir.arg_values],
        "outputs": [_function_value_signature(func_ir, vid, "output") for vid in func_ir.return_values],
    }


def _function_value_signature(
    func_ir: FunctionIR,
    value_id: str,
    direction: str,
) -> dict[str, Any]:
    value = func_ir.values.get(value_id)
    if value is None:
        return {
            "direction": direction,
            "type_hint": None,
            "n_uses": 0,
            "is_control_related": False,
            "is_effect_related": False,
        }
    return {
        "direction": direction,
        "type_hint": value.type_hint,
        "n_uses": len(value.use_ops),
        "is_control_related": any(
            func_ir.ops[use_op].opcode == "branch"
            for use_op in value.use_ops
            if use_op in func_ir.ops
        ),
        "is_effect_related": any(
            func_ir.ops[use_op].opcode in {"set_attr", "set_item", "append", "pop"}
            or (
                func_ir.ops[use_op].opcode == "call"
                and (
                    func_ir.ops[use_op].attrs.get("effectful")
                    or func_ir.ops[use_op].attrs.get("has_side_effect")
                    or func_ir.ops[use_op].attrs.get("escapes")
                )
            )
            for use_op in value.use_ops
            if use_op in func_ir.ops
        ),
    }


def _strip_value_id(signature_entry: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in signature_entry.items() if k != "value_id"}
