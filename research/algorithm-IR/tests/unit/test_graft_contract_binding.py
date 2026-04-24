from __future__ import annotations

import pytest

from algorithm_ir.grafting.graft_general import graft_general
from algorithm_ir.ir import Block, FunctionIR, Op, Value
from algorithm_ir.ir.validator import rebuild_def_use, validate_function_ir
from algorithm_ir.region import BoundaryRegionSpec, define_rewrite_region, extract_region_ir, infer_boundary_contract
from evolution.pool_types import GraftProposal


def _host_ir() -> FunctionIR:
    values = {
        "h_arg": Value(id="h_arg", type_hint="f64"),
        "h_mid": Value(id="h_mid", type_hint="f64"),
        "h_out": Value(id="h_out", type_hint="f64"),
    }
    ops = {
        "h_pre": Op(id="h_pre", opcode="unary", inputs=["h_arg"], outputs=["h_mid"], block_id="b0"),
        "h_region": Op(id="h_region", opcode="unary", inputs=["h_mid"], outputs=["h_out"], block_id="b0"),
        "h_ret": Op(id="h_ret", opcode="return", inputs=["h_out"], outputs=[], block_id="b0"),
    }
    ir = FunctionIR(
        id="host",
        name="host",
        arg_values=["h_arg"],
        return_values=["h_out"],
        values=values,
        ops=ops,
        blocks={"b0": Block(id="b0", op_ids=["h_pre", "h_region", "h_ret"])},
        entry_block="b0",
    )
    return rebuild_def_use(ir)


def _donor_ir() -> FunctionIR:
    values = {
        "d_in": Value(id="d_in", type_hint="f64"),
        "d_mid": Value(id="d_mid", type_hint="f64"),
        "d_out": Value(id="d_out", type_hint="f64"),
    }
    ops = {
        "d_a": Op(id="d_a", opcode="unary", inputs=["d_in"], outputs=["d_mid"], block_id="db0"),
        "d_b": Op(id="d_b", opcode="unary", inputs=["d_mid"], outputs=["d_out"], block_id="db0"),
        "d_ret": Op(id="d_ret", opcode="return", inputs=["d_out"], outputs=[], block_id="db0"),
    }
    ir = FunctionIR(
        id="donor",
        name="donor",
        arg_values=["d_in"],
        return_values=["d_out"],
        values=values,
        ops=ops,
        blocks={"db0": Block(id="db0", op_ids=["d_a", "d_b", "d_ret"])},
        entry_block="db0",
    )
    return rebuild_def_use(ir)


def _donor_two_input_ir() -> FunctionIR:
    values = {
        "x1": Value(id="x1", type_hint="f64"),
        "x2": Value(id="x2", type_hint="f64"),
        "y": Value(id="y", type_hint="f64"),
    }
    ops = {
        "mix": Op(id="mix", opcode="binary", inputs=["x1", "x2"], outputs=["y"], block_id="b0"),
        "ret": Op(id="ret", opcode="return", inputs=["y"], outputs=[], block_id="b0"),
    }
    ir = FunctionIR(
        id="donor2",
        name="donor2",
        arg_values=["x1", "x2"],
        return_values=["y"],
        values=values,
        ops=ops,
        blocks={"b0": Block(id="b0", op_ids=["mix", "ret"])},
        entry_block="b0",
    )
    return rebuild_def_use(ir)


def _signature_without_value_ids(contract_dicts):
    return [
        {k: v for k, v in item.items() if k != "value_id"}
        for item in contract_dicts
    ]


def test_extract_region_ir_preserves_exact_region_boundary():
    donor = _donor_ir()
    region = define_rewrite_region(
        donor,
        boundary_spec=BoundaryRegionSpec(output_values=["d_out"], cut_values=["d_in"]),
    )
    extracted = extract_region_ir(donor, region)

    assert extracted.arg_values == ["d_in"]
    assert extracted.return_values == ["d_out"]
    assert {op_id for op_id, op in extracted.ops.items() if op.opcode != "return"} == {"d_a", "d_b"}
    assert validate_function_ir(extracted) == []


def test_contract_port_order_is_stable_under_value_renaming():
    donor_a = _donor_two_input_ir()
    donor_b = _donor_two_input_ir()
    donor_b.values["x1"].id = "u1"
    donor_b.values["x2"].id = "u2"
    donor_b.values["y"].id = "u3"
    donor_b.ops["mix"].inputs = ["u1", "u2"]
    donor_b.ops["mix"].outputs = ["u3"]
    donor_b.ops["ret"].inputs = ["u3"]
    donor_b.arg_values = ["u1", "u2"]
    donor_b.return_values = ["u3"]
    donor_b.values = {"u1": donor_b.values.pop("x1"), "u2": donor_b.values.pop("x2"), "u3": donor_b.values.pop("y")}
    rebuild_def_use(donor_b)

    region_a = define_rewrite_region(
        donor_a,
        boundary_spec=BoundaryRegionSpec(output_values=["y"], cut_values=["x1", "x2"]),
    )
    region_b = define_rewrite_region(
        donor_b,
        boundary_spec=BoundaryRegionSpec(output_values=["u3"], cut_values=["u1", "u2"]),
    )
    contract_a = infer_boundary_contract(donor_a, region_a)
    contract_b = infer_boundary_contract(donor_b, region_b)

    assert _signature_without_value_ids(contract_a.port_signature["inputs"]) == _signature_without_value_ids(contract_b.port_signature["inputs"])
    assert _signature_without_value_ids(contract_a.port_signature["outputs"]) == _signature_without_value_ids(contract_b.port_signature["outputs"])


def test_graft_general_binds_by_contract_not_name_hint():
    host = _host_ir()
    donor = _donor_ir()
    host_region = define_rewrite_region(
        host,
        boundary_spec=BoundaryRegionSpec(output_values=["h_out"], cut_values=["h_mid"]),
    )
    host_contract = infer_boundary_contract(host, host_region)

    donor_region = define_rewrite_region(
        donor,
        boundary_spec=BoundaryRegionSpec(output_values=["d_out"], cut_values=["d_in"]),
    )
    donor_trim = extract_region_ir(donor, donor_region)

    proposal = GraftProposal(
        proposal_id="p_bc",
        host_algo_id="host",
        donor_algo_id="donor",
        region=host_region,
        contract=host_contract,
        donor_ir=donor_trim,
        donor_region=donor_region,
    )

    artifact = graft_general(host, proposal)
    assert validate_function_ir(artifact.ir) == []
    assert "h_region" not in artifact.ir.ops
    ret_op = next(op for op in artifact.ir.ops.values() if op.opcode == "return")
    assert ret_op.inputs


def test_graft_general_rejects_incompatible_port_signature():
    host = _host_ir()
    donor = _donor_two_input_ir()
    host_region = define_rewrite_region(
        host,
        boundary_spec=BoundaryRegionSpec(output_values=["h_out"], cut_values=["h_mid"]),
    )
    host_contract = infer_boundary_contract(host, host_region)

    donor_region = define_rewrite_region(
        donor,
        boundary_spec=BoundaryRegionSpec(output_values=["y"], cut_values=["x1", "x2"]),
    )
    donor_trim = extract_region_ir(donor, donor_region)

    proposal = GraftProposal(
        proposal_id="p_bad",
        host_algo_id="host",
        donor_algo_id="donor2",
        region=host_region,
        contract=host_contract,
        donor_ir=donor_trim,
        donor_region=donor_region,
    )

    with pytest.raises(ValueError):
        graft_general(host, proposal)
