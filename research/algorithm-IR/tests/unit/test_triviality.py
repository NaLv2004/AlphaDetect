"""Unit tests for the IR triviality predicate (Plan B visibility filter)."""

from __future__ import annotations

from algorithm_ir.ir import Block, FunctionIR, Op, Value
from algorithm_ir.ir.validator import rebuild_def_use
from algorithm_ir.region.triviality import (
    is_trivial_op,
    is_trivial_value,
    visible_def_op,
    visible_def_ops,
)


def _make_ir(values: dict[str, Value], ops: dict[str, Op],
             arg_values: list[str], return_values: list[str]) -> FunctionIR:
    ir = FunctionIR(
        id="t", name="t",
        arg_values=arg_values,
        return_values=return_values,
        values=values,
        ops=ops,
        blocks={"b0": Block(id="b0", op_ids=list(ops.keys()))},
        entry_block="b0",
    )
    return rebuild_def_use(ir)


def test_const_is_trivial():
    op = Op(id="c", opcode="const", inputs=[], outputs=["v"], block_id="b0",
            attrs={"literal": 1})
    assert is_trivial_op(op) is True


def test_get_attr_is_trivial():
    op = Op(id="g", opcode="get_attr", inputs=["v0"], outputs=["v1"],
            block_id="b0", attrs={"attr": "T"})
    assert is_trivial_op(op) is True


def test_assign_is_trivial():
    op = Op(id="a", opcode="assign", inputs=["v0"], outputs=["v1"],
            block_id="b0", attrs={"target": "x"})
    assert is_trivial_op(op) is True


def test_phi_with_identical_inputs_is_trivial():
    op = Op(id="p", opcode="phi", inputs=["v0", "v0"], outputs=["v1"],
            block_id="b0")
    assert is_trivial_op(op) is True


def test_phi_with_distinct_inputs_is_not_trivial():
    op = Op(id="p", opcode="phi", inputs=["v0", "v9"], outputs=["v1"],
            block_id="b0")
    assert is_trivial_op(op) is False


def test_phi_single_input_is_trivial():
    op = Op(id="p", opcode="phi", inputs=["v0"], outputs=["v1"], block_id="b0")
    assert is_trivial_op(op) is True


def test_real_compute_is_not_trivial():
    op = Op(id="b", opcode="binary", inputs=["v0", "v1"], outputs=["v2"],
            block_id="b0", attrs={"operator": "Add"})
    assert is_trivial_op(op) is False


def test_visible_def_op_skips_trivial_chain():
    # v0 -- binary --> v1 -- assign --> v2 -- get_attr --> v3 -- assign --> v4
    values = {
        "v0": Value(id="v0"),
        "v_arg": Value(id="v_arg"),
        "v1": Value(id="v1"),
        "v2": Value(id="v2"),
        "v3": Value(id="v3"),
        "v4": Value(id="v4"),
    }
    ops = {
        "compute": Op(id="compute", opcode="binary", inputs=["v_arg", "v_arg"],
                      outputs=["v1"], block_id="b0", attrs={"operator": "Add"}),
        "rebind1": Op(id="rebind1", opcode="assign", inputs=["v1"],
                      outputs=["v2"], block_id="b0", attrs={"target": "x"}),
        "attr":    Op(id="attr", opcode="get_attr", inputs=["v2"],
                      outputs=["v3"], block_id="b0", attrs={"attr": "T"}),
        "rebind2": Op(id="rebind2", opcode="assign", inputs=["v3"],
                      outputs=["v4"], block_id="b0", attrs={"target": "y"}),
        "ret": Op(id="ret", opcode="return", inputs=["v4"], outputs=[], block_id="b0"),
    }
    ir = _make_ir(values, ops, ["v_arg"], ["v4"])
    # v4 -> rebind2 (trivial) -> v3 -> attr (trivial) -> v2 -> rebind1 (trivial)
    # -> v1 -> compute (non-trivial) ✓
    found = visible_def_op(ir, "v4")
    assert found is not None
    assert found.id == "compute"


def test_visible_def_op_returns_none_for_arg():
    values = {"v_arg": Value(id="v_arg")}
    ops = {"ret": Op(id="ret", opcode="return", inputs=["v_arg"], outputs=[], block_id="b0")}
    ir = _make_ir(values, ops, ["v_arg"], ["v_arg"])
    assert visible_def_op(ir, "v_arg") is None


def test_is_trivial_value_uses_def_op():
    values = {
        "v_arg": Value(id="v_arg"),
        "v1": Value(id="v1"),
        "v2": Value(id="v2"),
    }
    ops = {
        "real": Op(id="real", opcode="binary", inputs=["v_arg", "v_arg"],
                   outputs=["v1"], block_id="b0", attrs={"operator": "Add"}),
        "rebind": Op(id="rebind", opcode="assign", inputs=["v1"],
                     outputs=["v2"], block_id="b0", attrs={"target": "x"}),
        "ret": Op(id="ret", opcode="return", inputs=["v2"], outputs=[], block_id="b0"),
    }
    ir = _make_ir(values, ops, ["v_arg"], ["v2"])
    assert is_trivial_value(ir, "v1") is False
    assert is_trivial_value(ir, "v2") is True
    # Function arg has no def_op → considered non-trivial.
    assert is_trivial_value(ir, "v_arg") is False


# ---------------------------------------------------------------------------
# Newly trivial opcodes: jump / return / build_tuple / build_list / build_slice
# ---------------------------------------------------------------------------

def test_jump_is_trivial():
    op = Op(id="j", opcode="jump", inputs=[], outputs=[], block_id="b0",
            attrs={"target": "b1"})
    assert is_trivial_op(op) is True


def test_return_is_trivial():
    op = Op(id="r", opcode="return", inputs=["v0"], outputs=[], block_id="b0")
    assert is_trivial_op(op) is True


def test_build_tuple_is_trivial():
    op = Op(id="t", opcode="build_tuple", inputs=["v0", "v1"], outputs=["v2"],
            block_id="b0", attrs={"n_items": 2})
    assert is_trivial_op(op) is True


def test_build_list_is_trivial():
    op = Op(id="l", opcode="build_list", inputs=[], outputs=["v0"],
            block_id="b0", attrs={"n_items": 0})
    assert is_trivial_op(op) is True


def test_build_slice_is_trivial():
    op = Op(id="s", opcode="build_slice", inputs=["v0", "v1", "v2"],
            outputs=["v3"], block_id="b0")
    assert is_trivial_op(op) is True


# ---------------------------------------------------------------------------
# visible_def_ops: multi-input transitive walk for packers
# ---------------------------------------------------------------------------

def test_visible_def_ops_fans_out_through_build_tuple():
    """build_tuple(real_a, real_b) -> tuple_value: walking back from
    tuple_value must surface BOTH real_a's and real_b's producer ops,
    not just the first input."""
    values = {
        "v_arg": Value(id="v_arg"),
        "va": Value(id="va"),
        "vb": Value(id="vb"),
        "vt": Value(id="vt"),
    }
    ops = {
        "compute_a": Op(id="compute_a", opcode="binary",
                        inputs=["v_arg", "v_arg"], outputs=["va"],
                        block_id="b0", attrs={"operator": "Add"}),
        "compute_b": Op(id="compute_b", opcode="unary",
                        inputs=["v_arg"], outputs=["vb"],
                        block_id="b0", attrs={"operator": "Neg"}),
        "pack": Op(id="pack", opcode="build_tuple", inputs=["va", "vb"],
                   outputs=["vt"], block_id="b0", attrs={"n_items": 2}),
        "ret": Op(id="ret", opcode="return", inputs=["vt"], outputs=[],
                  block_id="b0"),
    }
    ir = _make_ir(values, ops, ["v_arg"], ["vt"])
    producers = visible_def_ops(ir, "vt")
    producer_ids = {op.id for op in producers}
    assert producer_ids == {"compute_a", "compute_b"}, (
        f"expected both compute_a and compute_b, got {producer_ids}"
    )


def test_visible_def_ops_walks_through_nested_trivial_chain():
    """Chain: binary -> assign -> build_tuple -> get_attr (via assign).
    visible_def_ops on the final value must surface the binary op."""
    values = {
        "v_arg": Value(id="v_arg"),
        "v1": Value(id="v1"),
        "v2": Value(id="v2"),
        "v3": Value(id="v3"),
        "v4": Value(id="v4"),
    }
    ops = {
        "compute": Op(id="compute", opcode="binary",
                      inputs=["v_arg", "v_arg"], outputs=["v1"],
                      block_id="b0", attrs={"operator": "Add"}),
        "rebind": Op(id="rebind", opcode="assign", inputs=["v1"],
                     outputs=["v2"], block_id="b0", attrs={"target": "x"}),
        "pack": Op(id="pack", opcode="build_tuple", inputs=["v2"],
                   outputs=["v3"], block_id="b0", attrs={"n_items": 1}),
        "rebind2": Op(id="rebind2", opcode="assign", inputs=["v3"],
                      outputs=["v4"], block_id="b0", attrs={"target": "y"}),
        "ret": Op(id="ret", opcode="return", inputs=["v4"], outputs=[],
                  block_id="b0"),
    }
    ir = _make_ir(values, ops, ["v_arg"], ["v4"])
    producers = visible_def_ops(ir, "v4")
    assert [op.id for op in producers] == ["compute"]


def test_visible_def_ops_returns_empty_for_arg():
    values = {"v_arg": Value(id="v_arg")}
    ops = {"ret": Op(id="ret", opcode="return", inputs=["v_arg"],
                     outputs=[], block_id="b0")}
    ir = _make_ir(values, ops, ["v_arg"], ["v_arg"])
    assert visible_def_ops(ir, "v_arg") == []
