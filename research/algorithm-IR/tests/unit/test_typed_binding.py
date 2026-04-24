"""Unit tests for algorithm_ir.grafting.typed_binding."""
from __future__ import annotations

import pytest

from algorithm_ir.grafting.typed_binding import (
    INFEASIBLE,
    bind_typed,
    collect_visible_host_values,
)
from algorithm_ir.ir.model import Block, FunctionIR, Op, Value
from algorithm_ir.ir.validator import rebuild_def_use


def _ir(name, args, values, ops, block_ops, ret):
    """Tiny IR builder for tests."""
    ir = FunctionIR(
        id=name, name=name,
        arg_values=args, return_values=ret,
        values=values, ops=ops,
        blocks={"b0": Block(id="b0", op_ids=block_ops)},
        entry_block="b0",
    )
    return rebuild_def_use(ir)


def _val(vid, type_hint, name=None, def_op=None):
    return Value(id=vid, name_hint=name or vid, type_hint=type_hint,
                 def_op=def_op, attrs={"var_name": name or vid})


def test_typed_binding_picks_matching_lattice_type():
    """Donor wants vec_cx; host has both mat_cx and vec_cx — must pick vec_cx."""
    donor_vals = {
        "d_y": _val("d_y", "vec_cx", "y"),
        "d_out": _val("d_out", "vec_cx", "out", def_op="d_op"),
    }
    donor_ops = {
        "d_op": Op(id="d_op", opcode="unary", inputs=["d_y"],
                   outputs=["d_out"], block_id="b0"),
        "d_ret": Op(id="d_ret", opcode="return", inputs=["d_out"],
                    outputs=[], block_id="b0"),
    }
    donor = _ir("donor", ["d_y"], donor_vals, donor_ops,
                ["d_op", "d_ret"], ["d_out"])

    host_vals = {
        "h_H": _val("h_H", "mat_cx", "H"),
        "h_y": _val("h_y", "vec_cx", "wrong_name_but_right_type"),
        "h_o": _val("h_o", "vec_cx", "tmp", def_op="h_op"),
    }
    host_ops = {
        "h_op": Op(id="h_op", opcode="unary", inputs=["h_y"],
                   outputs=["h_o"], block_id="b0"),
        "h_ret": Op(id="h_ret", opcode="return", inputs=["h_o"],
                    outputs=[], block_id="b0"),
    }
    host = _ir("host", ["h_H", "h_y"], host_vals, host_ops,
               ["h_op", "h_ret"], ["h_o"])

    result = bind_typed(donor, host, splice_op_ids={"h_op"})
    assert result is not None and result.feasible
    # Must NOT pick h_H (mat_cx) for a vec_cx donor arg.
    assert result.mapping["d_y"] != "h_H"
    assert host.values[result.mapping["d_y"]].type_hint == "vec_cx"


def test_typed_binding_rejects_when_no_compatible_candidate():
    """Donor wants mat_cx; host has only int and float — must return None."""
    donor_vals = {
        "d_M": _val("d_M", "mat_cx", "M"),
        "d_out": _val("d_out", "mat_cx", "out", def_op="d_op"),
    }
    donor_ops = {
        "d_op": Op(id="d_op", opcode="unary", inputs=["d_M"],
                   outputs=["d_out"], block_id="b0"),
        "d_ret": Op(id="d_ret", opcode="return", inputs=["d_out"],
                    outputs=[], block_id="b0"),
    }
    donor = _ir("donor", ["d_M"], donor_vals, donor_ops,
                ["d_op", "d_ret"], ["d_out"])

    host_vals = {
        "h_n": _val("h_n", "int", "n"),
        "h_s": _val("h_s", "float", "s"),
        "h_o": _val("h_o", "int", "tmp", def_op="h_op"),
    }
    host_ops = {
        "h_op": Op(id="h_op", opcode="binary", inputs=["h_n", "h_s"],
                   outputs=["h_o"], block_id="b0"),
        "h_ret": Op(id="h_ret", opcode="return", inputs=["h_o"],
                    outputs=[], block_id="b0"),
    }
    host = _ir("host", ["h_n", "h_s"], host_vals, host_ops,
               ["h_op", "h_ret"], ["h_o"])

    result = bind_typed(donor, host, splice_op_ids={"h_op"},
                        require_feasible=True)
    assert result is None


def test_typed_binding_breaks_ties_by_name():
    """Two host candidates of the right type — name match should win."""
    donor_vals = {
        "d_sigma2": _val("d_sigma2", "float", "sigma2"),
        "d_out": _val("d_out", "float", "out", def_op="d_op"),
    }
    donor_ops = {
        "d_op": Op(id="d_op", opcode="unary", inputs=["d_sigma2"],
                   outputs=["d_out"], block_id="b0"),
        "d_ret": Op(id="d_ret", opcode="return", inputs=["d_out"],
                    outputs=[], block_id="b0"),
    }
    donor = _ir("donor", ["d_sigma2"], donor_vals, donor_ops,
                ["d_op", "d_ret"], ["d_out"])

    host_vals = {
        "h_other": _val("h_other", "float", "lr"),
        "h_sigma2": _val("h_sigma2", "float", "sigma2"),
        "h_o": _val("h_o", "float", "tmp", def_op="h_op"),
    }
    host_ops = {
        "h_op": Op(id="h_op", opcode="binary", inputs=["h_other", "h_sigma2"],
                   outputs=["h_o"], block_id="b0"),
        "h_ret": Op(id="h_ret", opcode="return", inputs=["h_o"],
                    outputs=[], block_id="b0"),
    }
    host = _ir("host", ["h_other", "h_sigma2"], host_vals, host_ops,
               ["h_op", "h_ret"], ["h_o"])

    result = bind_typed(donor, host, splice_op_ids={"h_op"})
    assert result is not None and result.feasible
    assert result.mapping["d_sigma2"] == "h_sigma2"


def test_typed_binding_solve_signature():
    """np.linalg.solve(A, b): two donor args (mat_cx, vec_cx).

    Host has the same two types but in *swapped* declaration order.
    The bipartite assignment must still produce mat_cx→A, vec_cx→b.
    """
    donor_vals = {
        "d_A": _val("d_A", "mat_cx", "A"),
        "d_b": _val("d_b", "vec_cx", "b"),
        "d_x": _val("d_x", "vec_cx", "x", def_op="d_op"),
    }
    donor_ops = {
        "d_op": Op(id="d_op", opcode="call", inputs=["d_A", "d_b"],
                   outputs=["d_x"], block_id="b0"),
        "d_ret": Op(id="d_ret", opcode="return", inputs=["d_x"],
                    outputs=[], block_id="b0"),
    }
    donor = _ir("solve", ["d_A", "d_b"], donor_vals, donor_ops,
                ["d_op", "d_ret"], ["d_x"])

    # Host args declared as (b, A) — opposite order from the donor.
    host_vals = {
        "h_b": _val("h_b", "vec_cx", "received"),
        "h_A": _val("h_A", "mat_cx", "channel"),
        "h_o": _val("h_o", "vec_cx", "tmp", def_op="h_op"),
    }
    host_ops = {
        "h_op": Op(id="h_op", opcode="binary", inputs=["h_b", "h_A"],
                   outputs=["h_o"], block_id="b0"),
        "h_ret": Op(id="h_ret", opcode="return", inputs=["h_o"],
                    outputs=[], block_id="b0"),
    }
    host = _ir("host", ["h_b", "h_A"], host_vals, host_ops,
               ["h_op", "h_ret"], ["h_o"])

    result = bind_typed(donor, host, splice_op_ids={"h_op"})
    assert result is not None and result.feasible
    assert result.mapping["d_A"] == "h_A"
    assert result.mapping["d_b"] == "h_b"


def test_collect_visible_host_values_excludes_post_splice():
    host_vals = {
        "a": _val("a", "int", "a"),
        "b": _val("b", "int", "b"),
        "c": _val("c", "int", "c", def_op="op_c"),
        "d": _val("d", "int", "d", def_op="op_splice"),
        "e": _val("e", "int", "e", def_op="op_after"),
    }
    host_ops = {
        "op_c":      Op(id="op_c", opcode="binary", inputs=["a", "b"],
                        outputs=["c"], block_id="b0"),
        "op_splice": Op(id="op_splice", opcode="binary", inputs=["c", "a"],
                        outputs=["d"], block_id="b0"),
        "op_after":  Op(id="op_after", opcode="binary", inputs=["d", "b"],
                        outputs=["e"], block_id="b0"),
        "op_ret":    Op(id="op_ret", opcode="return", inputs=["e"],
                        outputs=[], block_id="b0"),
    }
    host = _ir("host", ["a", "b"], host_vals, host_ops,
               ["op_c", "op_splice", "op_after", "op_ret"], ["e"])
    visible = collect_visible_host_values(host, ["op_splice"])
    assert "a" in visible and "b" in visible and "c" in visible
    assert "d" not in visible  # produced by splice op itself
    assert "e" not in visible  # produced after the splice point
