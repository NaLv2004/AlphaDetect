from __future__ import annotations

from algorithm_ir.ir.model import Block, FunctionIR, Op, Value
from algorithm_ir.ir.validator import rebuild_def_use
from algorithm_ir.region.selector import RewriteRegion


def extract_region_ir(func_ir: FunctionIR, region: RewriteRegion) -> FunctionIR:
    """Extract an executable FunctionIR for exactly ``region``.

    The extracted function:
    - contains only region ops plus a synthetic return op
    - uses ``region.entry_values`` as function arguments
    - uses ``region.exit_values`` as function returns
    - performs no dependency expansion beyond the region closure
    """
    block_ids = list(dict.fromkeys(region.block_ids)) or [func_ir.entry_block]
    # Choose the *topologically first* in-region block as the entry.
    # ``region.block_ids`` arrives in arbitrary order (typically
    # lexicographic from ``sorted({...})`` in the selector), which can
    # put alphabetically-early blocks like ``b_if_merge_9`` ahead of
    # their natural predecessors and break codegen's CFG walk
    # (which starts emission at ``func_ir.entry_block``) and the
    # graft-time inlining (``donor_entry_block_id =
    # id_map[donor_ir.entry_block]``).
    #
    # Use BFS distance from ``func_ir.entry_block`` in the host CFG:
    # the region block reached *earliest* from the host's true entry
    # is the natural inlining entry for this region.  This works for
    # straight-line, branch, and loop regions alike (loop heads are
    # reached before loop bodies, merge blocks after their branches).
    region_block_set = set(block_ids)
    _dist: dict[str, int] = {func_ir.entry_block: 0}
    _queue = [func_ir.entry_block]
    while _queue:
        _next: list[str] = []
        for _b in _queue:
            for _s in func_ir.blocks[_b].succs:
                if _s not in _dist:
                    _dist[_s] = _dist[_b] + 1
                    _next.append(_s)
        _queue = _next
    entry_block_id = min(
        block_ids,
        key=lambda bid: (_dist.get(bid, 1 << 30), block_ids.index(bid)),
    )
    # Reorder block_ids so entry comes first (preserves remaining order).
    block_ids = [entry_block_id] + [b for b in block_ids if b != entry_block_id]
    new_blocks: dict[str, Block] = {}
    new_ops: dict[str, Op] = {}

    for block_id in block_ids:
        orig_block = func_ir.blocks[block_id]
        selected_ops = [op_id for op_id in orig_block.op_ids if op_id in set(region.op_ids)]
        new_blocks[block_id] = Block(
            id=block_id,
            op_ids=list(selected_ops),
            preds=[pred for pred in orig_block.preds if pred in block_ids],
            succs=[succ for succ in orig_block.succs if succ in block_ids],
            attrs=dict(orig_block.attrs),
        )
        for op_id in selected_ops:
            op = func_ir.ops[op_id]
            new_ops[op_id] = Op(
                id=op.id,
                opcode=op.opcode,
                inputs=list(op.inputs),
                outputs=list(op.outputs),
                block_id=block_id,
                source_span=op.source_span,
                attrs=dict(op.attrs),
            )

    return_block_id = block_ids[-1]
    return_op_id = f"{region.region_id}_return"
    new_ops[return_op_id] = Op(
        id=return_op_id,
        opcode="return",
        inputs=list(region.exit_values),
        outputs=[],
        block_id=return_block_id,
        attrs={"synthetic": True},
    )
    new_blocks[return_block_id].op_ids.append(return_op_id)

    kept_value_ids = set(region.entry_values)
    for op_id in region.op_ids:
        op = func_ir.ops[op_id]
        kept_value_ids.update(op.inputs)
        kept_value_ids.update(op.outputs)
    kept_value_ids.update(region.exit_values)

    new_values: dict[str, Value] = {}
    for value_id in kept_value_ids:
        orig = func_ir.values[value_id]
        new_values[value_id] = Value(
            id=orig.id,
            name_hint=orig.name_hint,
            type_hint=orig.type_hint,
            source_span=orig.source_span,
            def_op=None,
            use_ops=[],
            attrs=dict(orig.attrs),
        )

    extracted = FunctionIR(
        id=f"{func_ir.id}_extract_{region.region_id}",
        name=f"{func_ir.name}_extract",
        arg_values=list(region.entry_values),
        return_values=list(region.exit_values),
        values=new_values,
        ops=new_ops,
        blocks=new_blocks,
        entry_block=entry_block_id,
        attrs=dict(func_ir.attrs) if func_ir.attrs else {},
    )
    rebuild_def_use(extracted)

    # ------------------------------------------------------------------
    # Synthetic entry-block ``assign`` injection
    # ------------------------------------------------------------------
    # Codegen lowers the IR to flat-scope Python where ``var_name``
    # (on Op outputs / ``attrs['target']``) is the Python local that
    # an assignment writes.  Two donor-side situations would otherwise
    # leave a Python local *unbound* after grafting:
    #
    #   (P) A ``phi`` op inside the region merges Python-name ``X``
    #       across predecessors.  ``phi`` itself emits no statement —
    #       only its predecessor branches do.  If one phi input comes
    #       from an outside-region predecessor (i.e. an ``entry_value``),
    #       the corresponding assignment to ``X`` lives outside the
    #       region and is dropped after grafting.  Downstream reads of
    #       the phi-output's name fail with ``UnboundLocalError``.
    #
    #   (L) A Python name ``X`` is assigned by ops *inside* the region
    #       but is *also* assigned by ops *outside* the region (a
    #       loop-init, a branch-init, a flat init line).  After
    #       grafting, only the inside writes are kept; if the donor
    #       relied on the outside init reaching the inside read first,
    #       we get ``UnboundLocalError``.
    #
    # Both cases are repaired by *promoting the outside writer's SSA
    # value to an extra ``arg_value``* of the extracted function and
    # *injecting* a synthetic ``assign`` op at the head of the entry
    # block whose ``target`` is ``X`` and whose source is the lifted
    # arg.  After ``clone_donor_ir`` α-renames the assign's ``target``
    # alongside every other in-region label, the injected statement
    # becomes ``<prefix>_X = <host-bound-value>`` — exactly the binding
    # that the dropped outside writer used to provide.  The
    # ``typed_binding`` pipeline picks the host candidate for the
    # lifted arg through its normal type/name/dataflow scoring.
    region_op_set = set(region.op_ids)

    def _names_from_op(op: Op) -> set[str]:
        names: set[str] = set()
        if op.attrs:
            t = op.attrs.get("target")
            if isinstance(t, str) and t:
                names.add(t)
            v = op.attrs.get("var_name")
            if isinstance(v, str) and v:
                names.add(v)
        for vid in op.outputs:
            val = func_ir.values.get(vid)
            if val is None:
                continue
            v = (val.attrs or {}).get("var_name")
            if isinstance(v, str) and v:
                names.add(v)
        return names

    # Map: name → SSA vid of latest outside writer (block-major linear order).
    op_position: dict[str, int] = {}
    pos_idx = 0
    for block_id_iter in func_ir.blocks:
        for op_id in func_ir.blocks[block_id_iter].op_ids:
            op_position[op_id] = pos_idx
            pos_idx += 1

    latest_writer_vid: dict[str, str] = {}
    latest_writer_pos: dict[str, int] = {}
    for op_id, op in func_ir.ops.items():
        if op_id in region_op_set:
            continue
        pos = op_position.get(op_id, -1)
        for vid in op.outputs:
            val = func_ir.values.get(vid)
            if val is None:
                continue
            vname = (val.attrs or {}).get("var_name")
            if not (isinstance(vname, str) and vname):
                continue
            if pos > latest_writer_pos.get(vname, -1):
                latest_writer_pos[vname] = pos
                latest_writer_vid[vname] = vid

    # Names assigned inside the region.
    region_assigned: set[str] = set()
    for op_id in region.op_ids:
        region_assigned.update(_names_from_op(func_ir.ops[op_id]))

    # (L) Leaked names — assigned both inside and outside the region.
    leaked_names: set[str] = set()
    if region_assigned:
        for op_id, op in func_ir.ops.items():
            if op_id in region_op_set:
                continue
            outside_names = _names_from_op(op) & region_assigned
            leaked_names.update(outside_names)

    # (P) Phi-merged names — collect (phi_var_name, donor_writer_vid) pairs
    # for each phi inside the region whose name has an outside writer.
    # We bind the phi's flat-scope name to the most recent outside SSA
    # writer of *that name*, regardless of which phi input vid happens
    # to be in entry_set (the SSA-derived entry can be a different
    # writer of the same name).
    phi_inits: list[tuple[str, str]] = []
    for op in extracted.ops.values():
        if op.opcode != "phi":
            continue
        phi_var = (op.attrs or {}).get("var_name") if op.attrs else None
        if not isinstance(phi_var, str) or not phi_var:
            continue
        writer_vid = latest_writer_vid.get(phi_var)
        if writer_vid is None:
            continue
        phi_inits.append((phi_var, writer_vid))

    # Build the union of names that need a synthetic init at entry.
    # Per name, we lift exactly one outside writer SSA value as a new
    # arg and emit one ``assign`` at the head of the entry block.
    #
    # Note: even if the SSA-derived entry contract already exposes an
    # arg whose ``var_name`` matches the leaked/phi name, we still
    # inject a synthetic assign — because after grafting, donor arg
    # vids are *rebound* to host vids in op inputs (their donor-side
    # ``var_name`` is irrelevant; codegen uses the host vid's
    # ``var_name`` to render reads of the arg).  The donor's flat-scope
    # ``best_dist`` would otherwise have no host-side binding and a
    # downstream phi-output read would surface as ``UnboundLocalError``.
    init_targets: dict[str, str] = {}  # name → outside writer vid
    for name in leaked_names:
        vid = latest_writer_vid.get(name)
        if vid is not None:
            init_targets[name] = vid
    for name, vid in phi_inits:
        init_targets.setdefault(name, vid)

    if init_targets:
        existing_args = set(extracted.arg_values)
        new_args: list[str] = []
        injected_op_ids: list[str] = []
        synth_idx = 0
        for name in sorted(init_targets):
            writer_vid = init_targets[name]
            # Promote the outside writer value to extracted.values (if missing)
            # and to arg_values (if not already an arg).
            if writer_vid not in extracted.values:
                orig = func_ir.values[writer_vid]
                extracted.values[writer_vid] = Value(
                    id=orig.id,
                    name_hint=orig.name_hint,
                    type_hint=orig.type_hint,
                    source_span=orig.source_span,
                    def_op=None,
                    use_ops=[],
                    attrs=dict(orig.attrs),
                )
            if writer_vid not in existing_args:
                new_args.append(writer_vid)
                existing_args.add(writer_vid)
            # Synthesise an assign op at the entry block head:
            #   target=<leaked/phi name>, inputs=[lifted_arg], outputs=[new_val]
            out_vid = f"{region.region_id}_initv{synth_idx}"
            op_id = f"{region.region_id}_initop{synth_idx}"
            synth_idx += 1
            extracted.values[out_vid] = Value(
                id=out_vid,
                name_hint=name,
                type_hint=None,
                source_span=None,
                def_op=op_id,
                use_ops=[],
                attrs={"var_name": name, "synthetic_init": True},
            )
            extracted.ops[op_id] = Op(
                id=op_id,
                opcode="assign",
                inputs=[writer_vid],
                outputs=[out_vid],
                block_id=entry_block_id,
                attrs={"target": name, "synthetic_init": True},
            )
            injected_op_ids.append(op_id)

        if new_args:
            extracted.arg_values = list(extracted.arg_values) + new_args
        if injected_op_ids:
            entry_block = extracted.blocks[entry_block_id]
            entry_block.op_ids = injected_op_ids + list(entry_block.op_ids)
        rebuild_def_use(extracted)

    # Codegen uses ``var_name`` as the Python LHS for assignments, which
    # means two SSA-distinct ops that share a ``var_name`` end up writing
    # to the *same* Python local in flat scope.  When those ops live on
    # mutually exclusive control-flow paths (e.g. opposite branches of
    # an if/else, or loop-init vs loop-step), the donor's original
    # phi reconciled them: a phi op output carrying the same
    # ``var_name`` is the SSA marker that the duplicate writes are
    # meant to merge.  If the region *contains* such a phi for the
    # duplicated name, the scaffold is intact and the duplicates are
    # legitimate (loop counter, branch merge, etc.).  If no such phi
    # exists inside the region, the duplicates have no merge point
    # after grafting and a downstream read can hit ``UnboundLocalError``.
    name_count: dict[str, int] = {}
    for op_id in region.op_ids:
        for name in _names_from_op(func_ir.ops[op_id]):
            name_count[name] = name_count.get(name, 0) + 1
    in_region_phi_names: set[str] = set()
    for op_id in region.op_ids:
        op = func_ir.ops[op_id]
        if op.opcode != "phi":
            continue
        for name in _names_from_op(op):
            in_region_phi_names.add(name)
    duplicated_unscaffolded = sorted(
        n for n, c in name_count.items()
        if c > 1
        and n not in in_region_phi_names
        and n not in init_targets  # outside writer was lifted + assign-injected
    )
    if duplicated_unscaffolded:
        raise ValueError(
            f"Region has {len(duplicated_unscaffolded)} Python name(s) "
            f"assigned by multiple ops with no in-region phi to reconcile "
            f"them and no outside initialiser to lift (e.g. "
            f"{duplicated_unscaffolded[:5]}); flat-scope codegen would "
            f"alias them and lose SSA scaffolding. Reject."
        )

    return extracted
