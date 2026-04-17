from __future__ import annotations

import pathlib
import sys
from dataclasses import asdict

ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TESTS_ROOT = ROOT / "tests"
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from algorithm_ir.factgraph import build_factgraph
from algorithm_ir.frontend import compile_function_to_ir
from algorithm_ir.grafting import graft_skeleton, make_bp_summary_skeleton
from algorithm_ir.ir import render_function_ir, validate_function_ir
from algorithm_ir.projection import annotate_region
from algorithm_ir.region import define_rewrite_region, infer_boundary_contract
from algorithm_ir.runtime import execute_ir
from examples.algorithms import bp_summary_update, simple_branch_loop, stack_decoder_host


def latest_value_for_var(func_ir, var_name: str) -> str:
    matches = [
        value_id
        for value_id, value in func_ir.values.items()
        if value.attrs.get("var_name") == var_name
    ]
    return sorted(matches, key=lambda value_id: func_ir.values[value_id].attrs.get("version", -1))[-1]


def select_contract_output(func_ir, contract, preferred_var: str) -> str:
    for value_id in contract.output_ports:
        if func_ir.values[value_id].attrs.get("var_name") == preferred_var:
            return value_id
    if len(contract.output_ports) == 1:
        return contract.output_ports[0]
    return contract.output_ports[-1]


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_subsection(title: str) -> None:
    print("\n" + "-" * 40)
    print(title)
    print("-" * 40)


def excerpt_region_ops(func_ir, region) -> str:
    lines: list[str] = []
    for op_id in region.op_ids:
        op = func_ir.ops[op_id]
        attrs = f" attrs={op.attrs}" if op.attrs else ""
        lines.append(
            f"{op.id}: {op.opcode} in={op.inputs} out={op.outputs} block={op.block_id}{attrs}"
        )
    return "\n".join(lines)


def render_block(func_ir, block_id: str) -> str:
    block = func_ir.blocks[block_id]
    lines = [f"Block {block.id} preds={block.preds} succs={block.succs}"]
    for op_id in block.op_ids:
        op = func_ir.ops[op_id]
        attrs = f" attrs={op.attrs}" if op.attrs else ""
        lines.append(
            f"  {op.id}: {op.opcode} in={op.inputs} out={op.outputs}{attrs}"
        )
    return "\n".join(lines)


def main() -> None:
    costs = [0.5, 0.25, 0.75]

    print_section("1. Frontend: 编译 simple_branch_loop")
    simple_ir = compile_function_to_ir(simple_branch_loop)
    print(f"validate errors = {validate_function_ir(simple_ir)}")
    print(f"blocks = {len(simple_ir.blocks)}, ops = {len(simple_ir.ops)}, values = {len(simple_ir.values)}")
    print(render_function_ir(simple_ir))

    print_section("2. Frontend: 编译 stack_decoder_host")
    host_ir = compile_function_to_ir(stack_decoder_host)
    print(f"validate errors = {validate_function_ir(host_ir)}")
    print(f"blocks = {len(host_ir.blocks)}, ops = {len(host_ir.ops)}, values = {len(host_ir.values)}")
    print(render_function_ir(host_ir))

    print_section("3. Frontend: 编译 donor bp_summary_update")
    donor_ir = compile_function_to_ir(bp_summary_update)
    print(f"validate errors = {validate_function_ir(donor_ir)}")
    print(f"blocks = {len(donor_ir.blocks)}, ops = {len(donor_ir.ops)}, values = {len(donor_ir.values)}")
    print(render_function_ir(donor_ir))

    print_section("4. Runtime: 执行 simple_branch_loop")
    expected_simple = simple_branch_loop(5)
    actual_simple, simple_trace, simple_runtime_values = execute_ir(simple_ir, [5])
    print(f"python result = {expected_simple}")
    print(f"ir result     = {actual_simple}")
    print(f"trace events  = {len(simple_trace)}")
    print(f"runtime values= {len(simple_runtime_values)}")

    print_section("5. Runtime + FactGraph: 执行 stack_decoder_host")
    expected_host = stack_decoder_host(costs, 4)
    actual_host, host_trace, host_runtime_values = execute_ir(host_ir, [costs, 4])
    fg = build_factgraph(host_ir, host_trace, host_runtime_values)
    print(f"python result = {expected_host}")
    print(f"ir result     = {actual_host}")
    print(f"trace events  = {len(host_trace)}")
    print(f"runtime values= {len(host_runtime_values)}")
    print("static edge counts =", {k: len(v) for k, v in fg.static_edges.items()})
    print("dynamic edge counts =", {k: len(v) for k, v in fg.dynamic_edges.items()})
    print("alignment edge counts =", {k: len(v) for k, v in fg.alignment_edges.items()})

    print_section("6. Region + Contract + Projection")
    score_value = latest_value_for_var(host_ir, "score")
    region = define_rewrite_region(host_ir, exit_values=[score_value])
    contract = infer_boundary_contract(host_ir, region, host_trace, host_runtime_values)
    projections = annotate_region(region, fg)

    print_subsection("RewriteRegion")
    print(asdict(region))

    print_subsection("Region 对应的 host IR 片段")
    print(excerpt_region_ops(host_ir, region))

    print_subsection("BoundaryContract")
    print(asdict(contract))

    print_subsection("Optional Projections")
    for projection in projections:
        print(asdict(projection))

    print_section("7. Grafting: 把 BP donor 嫁接到 stack host")
    source_region = define_rewrite_region(host_ir, source_span=(15, 8, 16, 63))
    source_contract = infer_boundary_contract(host_ir, source_region, host_trace, host_runtime_values)
    skeleton = make_bp_summary_skeleton(bp_summary_update, damping=0.1)
    artifact = graft_skeleton(host_ir, source_region, source_contract, skeleton)
    rewritten_ir = artifact.ir
    rewritten_result, rewritten_trace, _ = execute_ir(rewritten_ir, [costs, 4])
    override_plan = artifact.provenance["override_plan"]

    print_subsection("Source-span RewriteRegion")
    print(asdict(source_region))

    print_subsection("Source-span BoundaryContract")
    print(asdict(source_contract))

    score_output = select_contract_output(host_ir, source_contract, "score")
    original_block_id = host_ir.ops[host_ir.values[score_output].def_op].block_id
    print_subsection("原始 block")
    print(render_block(host_ir, original_block_id))

    print_subsection("改写后的 block")
    print(render_block(rewritten_ir, original_block_id))

    print_subsection("OverridePlan")
    print(asdict(override_plan))

    print_subsection("Grafted op ids")
    grafted_ops = [
        op.id
        for op in rewritten_ir.ops.values()
        if op.attrs.get("grafted")
    ]
    print(grafted_ops)
    for op_id in grafted_ops:
        op = rewritten_ir.ops[op_id]
        print(f"{op.id}: {op.opcode} in={op.inputs} out={op.outputs} attrs={op.attrs}")

    print_subsection("改写前后结果")
    print(f"host original result = {actual_host}")
    print(f"host rewritten result = {rewritten_result}")
    print(f"rewritten trace events = {len(rewritten_trace)}")
    print(f"donor provenance = {artifact.provenance['donor_function']}")


if __name__ == "__main__":
    main()
