from __future__ import annotations

import pytest

from algorithm_ir.ir import Block, FunctionIR, Op, Value
from algorithm_ir.ir.validator import rebuild_def_use, validate_def_use
from algorithm_ir.region.contract import infer_boundary_contract
from algorithm_ir.region.selector import BoundaryRegionSpec, define_rewrite_region
from algorithm_ir.region.slicer import (
    backward_slice_by_values,
    backward_slice_until_values,
    enumerate_cut_candidates,
    enumerate_observable_values,
    validate_boundary_region,
)


def _chain_ir() -> FunctionIR:
    values = {
        "v_arg": Value(id="v_arg", type_hint="f64"),
        "v1": Value(id="v1", type_hint="f64"),
        "v2": Value(id="v2", type_hint="f64"),
        "v3": Value(id="v3", type_hint="f64"),
    }
    ops = {
        "op1": Op(id="op1", opcode="unary", inputs=["v_arg"], outputs=["v1"], block_id="b0"),
        "op2": Op(id="op2", opcode="unary", inputs=["v1"], outputs=["v2"], block_id="b0"),
        "op3": Op(id="op3", opcode="unary", inputs=["v2"], outputs=["v3"], block_id="b0"),
        "ret": Op(id="ret", opcode="return", inputs=["v3"], outputs=[], block_id="b0"),
    }
    ir = FunctionIR(
        id="chain",
        name="chain",
        arg_values=["v_arg"],
        return_values=["v3"],
        values=values,
        ops=ops,
        blocks={"b0": Block(id="b0", op_ids=["op1", "op2", "op3", "ret"])},
        entry_block="b0",
    )
    return rebuild_def_use(ir)


def _diamond_ir() -> FunctionIR:
    values = {
        "v_arg": Value(id="v_arg", type_hint="f64"),
        "v_left": Value(id="v_left", type_hint="f64"),
        "v_right": Value(id="v_right", type_hint="f64"),
        "v_join": Value(id="v_join", type_hint="f64"),
        "v_out": Value(id="v_out", type_hint="f64"),
    }
    ops = {
        "left": Op(id="left", opcode="unary", inputs=["v_arg"], outputs=["v_left"], block_id="b0"),
        "right": Op(id="right", opcode="unary", inputs=["v_arg"], outputs=["v_right"], block_id="b0"),
        "join": Op(
            id="join",
            opcode="binary",
            inputs=["v_left", "v_right"],
            outputs=["v_join"],
            block_id="b0",
        ),
        "post": Op(id="post", opcode="unary", inputs=["v_join"], outputs=["v_out"], block_id="b0"),
        "ret": Op(id="ret", opcode="return", inputs=["v_out"], outputs=[], block_id="b0"),
    }
    ir = FunctionIR(
        id="diamond",
        name="diamond",
        arg_values=["v_arg"],
        return_values=["v_out"],
        values=values,
        ops=ops,
        blocks={"b0": Block(id="b0", op_ids=["left", "right", "join", "post", "ret"])},
        entry_block="b0",
    )
    return rebuild_def_use(ir)


def _multiblock_ir() -> FunctionIR:
    values = {
        "v_arg": Value(id="v_arg", type_hint="bool"),
        "v_cond": Value(id="v_cond", type_hint="bool"),
        "v_then": Value(id="v_then", type_hint="f64"),
        "v_else": Value(id="v_else", type_hint="f64"),
        "v_phi": Value(id="v_phi", type_hint="f64"),
        "v_out": Value(id="v_out", type_hint="f64"),
    }
    ops = {
        "cond": Op(id="cond", opcode="unary", inputs=["v_arg"], outputs=["v_cond"], block_id="entry"),
        "br": Op(
            id="br",
            opcode="branch",
            inputs=["v_cond"],
            outputs=[],
            block_id="entry",
            attrs={"true": "then", "false": "else"},
        ),
        "then_op": Op(id="then_op", opcode="const", inputs=[], outputs=["v_then"], block_id="then"),
        "then_jump": Op(id="then_jump", opcode="jump", inputs=[], outputs=[], block_id="then", attrs={"target": "merge"}),
        "else_op": Op(id="else_op", opcode="const", inputs=[], outputs=["v_else"], block_id="else"),
        "else_jump": Op(id="else_jump", opcode="jump", inputs=[], outputs=[], block_id="else", attrs={"target": "merge"}),
        "phi": Op(id="phi", opcode="phi", inputs=["v_then", "v_else"], outputs=["v_phi"], block_id="merge"),
        "post": Op(id="post", opcode="unary", inputs=["v_phi"], outputs=["v_out"], block_id="merge"),
        "ret": Op(id="ret", opcode="return", inputs=["v_out"], outputs=[], block_id="merge"),
    }
    blocks = {
        "entry": Block(id="entry", op_ids=["cond", "br"], succs=["then", "else"]),
        "then": Block(id="then", op_ids=["then_op", "then_jump"], preds=["entry"], succs=["merge"]),
        "else": Block(id="else", op_ids=["else_op", "else_jump"], preds=["entry"], succs=["merge"]),
        "merge": Block(id="merge", op_ids=["phi", "post", "ret"], preds=["then", "else"]),
    }
    ir = FunctionIR(
        id="multiblock",
        name="multiblock",
        arg_values=["v_arg"],
        return_values=["v_out"],
        values=values,
        ops=ops,
        blocks=blocks,
        entry_block="entry",
    )
    return rebuild_def_use(ir)


def test_rebuild_def_use_reconstructs_chain_metadata():
    ir = _chain_ir()
    ir.values["v1"].def_op = None
    ir.values["v2"].use_ops = []

    rebuild_def_use(ir)

    assert ir.values["v1"].def_op == "op1"
    assert ir.values["v2"].use_ops == ["op3"]
    assert ir.values["v_arg"].use_ops == ["op1"]


def test_validate_def_use_reports_corruption():
    ir = _chain_ir()
    ir.values["v2"].use_ops = ["op2"]
    errors = validate_def_use(ir)
    assert errors
    assert any("v2 use_ops mismatch" in err for err in errors)


def test_bcir_without_cut_matches_full_backward_slice():
    ir = _diamond_ir()
    old_slice = backward_slice_by_values(ir, ["v_out"])
    new_slice = backward_slice_until_values(ir, ["v_out"], [])
    assert new_slice == old_slice


def test_bcir_single_cut_stops_backtrace():
    ir = _chain_ir()
    region = define_rewrite_region(
        ir,
        boundary_spec=BoundaryRegionSpec(output_values=["v3"], cut_values=["v1"]),
    )
    assert region.op_ids == ["op2", "op3"]
    assert region.entry_values == ["v1"]
    assert region.exit_values == ["v3"]
    assert region.provenance["effective_cut_values"] == ["v1"]


def test_bcir_multi_cut_canonicalizes_redundant_values():
    ir = _diamond_ir()
    region = define_rewrite_region(
        ir,
        boundary_spec=BoundaryRegionSpec(
            output_values=["v_out"],
            cut_values=["v_arg", "v_left", "v_arg"],
        ),
    )
    assert sorted(region.entry_values) == ["v_arg", "v_left"]
    assert region.provenance["effective_cut_values"] == ["v_arg", "v_left"]


def test_enumerate_observable_values_finds_return_and_branch_condition():
    ir = _multiblock_ir()
    observed = enumerate_observable_values(ir)
    assert "v_out" in observed
    assert "v_cond" in observed


def test_enumerate_cut_candidates_only_returns_backward_values():
    ir = _diamond_ir()
    candidates = enumerate_cut_candidates(ir, ["v_out"])
    assert "v_join" in candidates
    assert "v_left" in candidates
    assert "v_out" not in candidates


def test_multiblock_boundary_region_tracks_blocks():
    ir = _multiblock_ir()
    region = define_rewrite_region(
        ir,
        boundary_spec=BoundaryRegionSpec(output_values=["v_out"], cut_values=["v_phi"]),
    )
    assert region.op_ids == ["post"]
    assert region.block_ids == ["merge"]
    assert region.entry_values == ["v_phi"]
    assert region.exit_values == ["v_out"]


def test_validate_boundary_region_rejects_invalid_cut():
    ir = _diamond_ir()
    with pytest.raises(ValueError):
        define_rewrite_region(
            ir,
            boundary_spec=BoundaryRegionSpec(output_values=["v_out"], cut_values=["missing"]),
        )


def test_contract_normalizes_single_output_region():
    ir = _chain_ir()
    region = define_rewrite_region(
        ir,
        boundary_spec=BoundaryRegionSpec(output_values=["v3"], cut_values=["v1"]),
    )
    contract = infer_boundary_contract(ir, region)
    assert contract.input_ports == ["v1"]
    assert contract.output_ports == ["v3"]
    assert contract.normalized_input_ports == ["v1"]
    assert contract.normalized_output_ports == ["v3"]
    assert contract.port_signature["inputs"][0]["type_hint"] == "f64"
