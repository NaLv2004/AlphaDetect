"""M1: ast.With → SlotMeta compilation tests.

Covers sequential, nested, multi-block (slot wrapping ast.If) cases plus
authoring-error surfacing (unbound output, missing kwargs, duplicate keys).
"""
from __future__ import annotations

import pytest

from algorithm_ir.frontend.ir_builder import compile_source_to_ir
from algorithm_ir.ir.validator import validate_function_ir, validate_slot_meta


def _compile(src: str, name: str = "f"):
    return compile_source_to_ir(src, name)


def test_sequential_slots_innermost_tag_and_meta():
    src = """
def f(a, b, c):
    with slot("blk.add", inputs=(a, b), outputs=("s",)):
        s = a + b
    with slot("blk.mul", inputs=(s, c), outputs=("p",)):
        p = s * c
    return p
"""
    ir = _compile(src)
    assert set(ir.slot_meta) == {"blk.add", "blk.mul"}
    assert ir.slot_meta["blk.add"].parent is None
    assert ir.slot_meta["blk.mul"].parent is None
    # Inputs come from current SSA name_env at slot entry.
    assert ir.slot_meta["blk.add"].inputs == (
        ir.arg_values[0], ir.arg_values[1],
    )
    # Outputs are defined inside the slot body.
    s_out = ir.slot_meta["blk.add"].outputs[0]
    assert ir.values[s_out].def_op in ir.slot_meta["blk.add"].op_ids
    # Each tagged op belongs to exactly the matching slot.
    for oid in ir.slot_meta["blk.add"].op_ids:
        assert ir.ops[oid].attrs.get("slot_id") == "blk.add"
    for oid in ir.slot_meta["blk.mul"].op_ids:
        assert ir.ops[oid].attrs.get("slot_id") == "blk.mul"
    assert validate_function_ir(ir) == []


def test_nested_slots_innermost_tag_only():
    src = """
def f(a, b, c):
    with slot("outer", inputs=(a, b, c), outputs=("p",)):
        with slot("inner", inputs=(a, b), outputs=("s",)):
            s = a + b
        p = s * c
    return p
"""
    ir = _compile(src)
    assert set(ir.slot_meta) == {"outer", "inner"}
    assert ir.slot_meta["inner"].parent == "outer"
    assert ir.slot_meta["outer"].parent is None
    # Innermost tagging: ops in inner must NOT be in outer.op_ids,
    # but slot_full_op_ids("outer") subsumes both.
    inner_ops = set(ir.slot_meta["inner"].op_ids)
    outer_ops = set(ir.slot_meta["outer"].op_ids)
    assert inner_ops.isdisjoint(outer_ops)
    assert inner_ops <= ir.slot_full_op_ids("outer")
    assert outer_ops <= ir.slot_full_op_ids("outer")
    assert ir.slot_children("outer") == ["inner"]
    assert validate_function_ir(ir) == []


def test_slot_wrapping_if_block():
    src = """
def f(a, b):
    with slot("conditional", inputs=(a, b), outputs=("y",)):
        if a > b:
            y = a - b
        else:
            y = b - a
    return y
"""
    ir = _compile(src)
    assert "conditional" in ir.slot_meta
    # Every op tagged "conditional" must be in slot_meta op_ids set.
    tagged = {oid for oid, op in ir.ops.items()
              if op.attrs.get("slot_id") == "conditional"}
    assert tagged == set(ir.slot_meta["conditional"].op_ids)
    # Output y must be defined inside the slot (i.e. by the merged phi).
    y_id = ir.slot_meta["conditional"].outputs[0]
    assert ir.values[y_id].def_op in ir.slot_full_op_ids("conditional")
    assert validate_function_ir(ir) == []


def test_unbound_output_name_raises():
    src = """
def f(a, b):
    with slot("oops", inputs=(a, b), outputs=("missing",)):
        x = a + b
    return x
"""
    with pytest.raises(SyntaxError, match="declared output 'missing'"):
        _compile(src)


def test_duplicate_pop_key_raises():
    src = """
def f(a, b):
    with slot("k", inputs=(a, b), outputs=("s",)):
        s = a + b
    with slot("k", inputs=(s, a), outputs=("p",)):
        p = s * a
    return p
"""
    with pytest.raises(SyntaxError, match="duplicate slot pop_key"):
        _compile(src)


def test_unknown_input_name_raises():
    src = """
def f(a, b):
    with slot("k", inputs=(a, c), outputs=("s",)):
        s = a + b
    return s
"""
    with pytest.raises(SyntaxError, match="input name 'c'"):
        _compile(src)


def test_slot_dsl_runtime_stub_executes():
    """The source string must remain directly executable at runtime."""
    from algorithm_ir.frontend.slot_dsl import slot
    ns: dict = {"slot": slot}
    src = """
def f(a, b):
    with slot("k", inputs=(a, b), outputs=("s",)):
        s = a + b
    return s + 1
"""
    exec(src, ns)
    assert ns["f"](2, 3) == 6


def test_validator_detects_corrupt_slot_meta():
    """validate_slot_meta surfaces tag/op_ids mismatches."""
    src = """
def f(a, b):
    with slot("k", inputs=(a, b), outputs=("s",)):
        s = a + b
    return s
"""
    ir = _compile(src)
    # Corrupt: drop one op from op_ids while keeping the tag.
    meta = ir.slot_meta["k"]
    bad = meta.__class__(
        pop_key=meta.pop_key,
        op_ids=meta.op_ids[:-1],
        inputs=meta.inputs,
        outputs=meta.outputs,
        output_names=meta.output_names,
        parent=meta.parent,
    )
    ir.slot_meta["k"] = bad
    errs = validate_slot_meta(ir)
    assert any("op_ids" in e or "tagged" in e for e in errs)
