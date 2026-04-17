from __future__ import annotations

import inspect
import pathlib
import sys
import textwrap
from dataclasses import asdict
from pprint import pformat

ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TESTS_ROOT = ROOT / "tests"
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from algorithm_ir.factgraph import build_factgraph
from algorithm_ir.frontend import compile_function_to_ir
from algorithm_ir.grafting import (
    graft_skeleton,
    make_bp_summary_skeleton,
    make_bp_tree_runtime_skeleton,
)
from algorithm_ir.ir import render_function_ir, validate_function_ir
from algorithm_ir.projection import annotate_region
from algorithm_ir.region import define_rewrite_region, infer_boundary_contract
from algorithm_ir.runtime import execute_ir
from examples.algorithms import (
    bp_summary_update,
    bp_tree_runtime_update,
    stack_decoder_host,
    stack_decoder_runtime_host,
)


def section(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def subsection(title: str) -> None:
    print("\n" + "-" * 100)
    print(title)
    print("-" * 100)


def source_of(fn) -> str:
    return textwrap.dedent(inspect.getsource(fn)).strip()


def pretty(obj) -> str:
    return pformat(obj, width=110, sort_dicts=False)


def block_text(func_ir, block_id: str) -> str:
    block = func_ir.blocks[block_id]
    lines = [f"Block {block.id}  preds={block.preds}  succs={block.succs}"]
    for op_id in block.op_ids:
        op = func_ir.ops[op_id]
        attrs = f" attrs={op.attrs}" if op.attrs else ""
        lines.append(f"  {op.id}: {op.opcode} in={op.inputs} out={op.outputs}{attrs}")
    return "\n".join(lines)


def excerpt_xdsl(xdsl_text: str, needles: list[str], context: int = 1) -> str:
    lines = xdsl_text.splitlines()
    chosen: set[int] = set()
    for index, line in enumerate(lines):
        if any(needle in line for needle in needles):
            for hit in range(max(0, index - context), min(len(lines), index + context + 1)):
                chosen.add(hit)
    if not chosen:
        return "[no matching xDSL lines found]"
    ordered = []
    previous = None
    for index in sorted(chosen):
        if previous is not None and index != previous + 1:
            ordered.append("  ...")
        ordered.append(lines[index])
        previous = index
    return "\n".join(ordered)


def latest_value_for_var(func_ir, var_name: str) -> str:
    matches = [
        value_id
        for value_id, value in func_ir.values.items()
        if value.attrs.get("var_name") == var_name
    ]
    if not matches:
        raise KeyError(f"Could not find value for variable {var_name!r}")
    return sorted(matches, key=lambda value_id: func_ir.values[value_id].attrs.get("version", -1))[-1]


def select_contract_output(func_ir, contract, preferred_var: str) -> str:
    for value_id in contract.output_ports:
        if func_ir.values[value_id].attrs.get("var_name") == preferred_var:
            return value_id
    if len(contract.output_ports) == 1:
        return contract.output_ports[0]
    return contract.output_ports[-1]


def runtime_expansion_region_ops(func_ir) -> list[str]:
    for block in func_ir.blocks.values():
        targets = {func_ir.ops[op_id].attrs.get("target") for op_id in block.op_ids}
        if {"left", "right"} <= targets:
            return list(block.op_ids)
    raise RuntimeError("Could not locate expansion block for runtime BP graft")


def step_message(step: int, message: str) -> None:
    print(f"[Step {step}] {message}")


def demo_local_score_injection(costs: list[float]) -> None:
    section("Demo A: Inject A BP Summary Into The Stack Decoder Score Region")

    step_message(1, "Show the host algorithm and donor algorithm at source-code level.")
    subsection("Host Python: stack_decoder_host")
    print(source_of(stack_decoder_host))
    subsection("Donor Python: bp_summary_update")
    print(source_of(bp_summary_update))

    step_message(2, "Compile both functions into xDSL-backed FunctionIR objects.")
    host_ir = compile_function_to_ir(stack_decoder_host)
    donor_ir = compile_function_to_ir(bp_summary_update)
    print(
        f"host validate = {validate_function_ir(host_ir)}, "
        f"blocks={len(host_ir.blocks)}, ops={len(host_ir.ops)}, values={len(host_ir.values)}"
    )
    print(
        f"donor validate = {validate_function_ir(donor_ir)}, "
        f"blocks={len(donor_ir.blocks)}, ops={len(donor_ir.ops)}, values={len(donor_ir.values)}"
    )

    subsection("Host FunctionIR (abridged)")
    host_ir_lines = render_function_ir(host_ir).splitlines()
    print("\n".join(host_ir_lines[:28]) + ("\n  ..." if len(host_ir_lines) > 28 else ""))

    subsection("Host xDSL snapshot")
    xdsl_lines = host_ir.attrs["xdsl_text"].splitlines()
    print("\n".join(xdsl_lines[:16]) + ("\n  ..." if len(xdsl_lines) > 16 else ""))

    step_message(3, "Run the host once so we have dynamic evidence for region and boundary inference.")
    original_result, host_trace, host_runtime_values = execute_ir(host_ir, [costs, 4])
    factgraph = build_factgraph(host_ir, host_trace, host_runtime_values)
    print(f"host result = {original_result}")
    print(f"trace events = {len(host_trace)}, runtime values = {len(host_runtime_values)}")
    print(
        "factgraph edge counts = "
        + pretty(
            {
                "static": {key: len(value) for key, value in factgraph.static_edges.items()},
                "dynamic": {key: len(value) for key, value in factgraph.dynamic_edges.items()},
                "alignment": {key: len(value) for key, value in factgraph.alignment_edges.items()},
            }
        )
    )

    step_message(4, "Select the score-computation region and infer its boundary contract.")
    region = define_rewrite_region(host_ir, source_span=(15, 8, 16, 63))
    contract = infer_boundary_contract(host_ir, region, host_trace, host_runtime_values)
    projections = annotate_region(region, factgraph)
    print("RewriteRegion:")
    print(pretty(asdict(region)))
    print("\nBoundaryContract:")
    print(pretty(asdict(contract)))
    print("\nOptional projections:")
    print(pretty([asdict(projection) for projection in projections]))

    score_output = select_contract_output(host_ir, contract, "score")
    original_block_id = host_ir.ops[host_ir.values[score_output].def_op].block_id
    subsection("Original score block")
    print(block_text(host_ir, original_block_id))

    step_message(5, "Build a BP-summary donor skeleton and graft it into the chosen host region.")
    skeleton = make_bp_summary_skeleton(bp_summary_update, damping=0.1)
    artifact = graft_skeleton(host_ir, region, contract, skeleton)
    rewritten_ir = artifact.ir
    rewritten_result, rewritten_trace, _ = execute_ir(rewritten_ir, [costs, 4])

    print("OverridePlan:")
    print(pretty(asdict(artifact.provenance["override_plan"])))

    subsection("Rewritten score block")
    print(block_text(rewritten_ir, original_block_id))

    subsection("xDSL lines that show the injected donor")
    print(excerpt_xdsl(rewritten_ir.attrs["xdsl_text"], ["bp_summary_update", "grafted"], context=1))

    subsection("What changed?")
    print("Original logic:")
    print("  score = candidate['metric'] + costs[candidate['depth']]")
    print("Rewritten logic:")
    print("  bp_summary = bp_summary_update(frontier, costs, damping)")
    print("  score = candidate['metric'] + bp_summary")
    print(f"Original result  = {original_result}")
    print(f"Rewritten result = {rewritten_result}")
    print(f"Rewritten trace events = {len(rewritten_trace)}")


def demo_runtime_nested_injection(costs: list[float]) -> None:
    section("Demo B: Runtime-Nested BP Injection After Each Stack Expansion")

    step_message(1, "Show the runtime host and the donor that operates on the current explored tree.")
    subsection("Host Python: stack_decoder_runtime_host")
    print(source_of(stack_decoder_runtime_host))
    subsection("Donor Python: bp_tree_runtime_update")
    print(source_of(bp_tree_runtime_update))

    step_message(2, "Compile both functions to xDSL-backed IR and run the baseline host.")
    host_ir = compile_function_to_ir(stack_decoder_runtime_host)
    donor_ir = compile_function_to_ir(bp_tree_runtime_update)
    baseline_audit: list[int] = []
    baseline_result, trace, runtime_values = execute_ir(host_ir, [costs, 3, baseline_audit])
    print(
        f"host validate = {validate_function_ir(host_ir)}, donor validate = {validate_function_ir(donor_ir)}"
    )
    print(f"baseline result = {baseline_result}")
    print(f"baseline audit  = {baseline_audit}")
    print(f"trace events    = {len(trace)}")

    step_message(3, "Choose the expansion block as the rewrite region.")
    region = define_rewrite_region(host_ir, op_ids=runtime_expansion_region_ops(host_ir))
    contract = infer_boundary_contract(host_ir, region, trace, runtime_values)
    insertion_block = region.block_ids[-1]
    print("RewriteRegion:")
    print(pretty(asdict(region)))
    print("\nBoundaryContract:")
    print(pretty(asdict(contract)))

    subsection("Original expansion block")
    print(block_text(host_ir, insertion_block))

    step_message(
        4,
        "Graft a runtime BP skeleton so that every expansion is followed by one BP pass over explored nodes.",
    )
    skeleton = make_bp_tree_runtime_skeleton(bp_tree_runtime_update, damping=0.1)
    artifact = graft_skeleton(host_ir, region, contract, skeleton)
    rewritten_ir = artifact.ir
    rewritten_audit: list[int] = []
    rewritten_result, rewritten_trace, _ = execute_ir(rewritten_ir, [costs, 3, rewritten_audit])

    subsection("Rewritten expansion block")
    print(block_text(rewritten_ir, insertion_block))

    subsection("xDSL lines that show the runtime-nested donor")
    print(excerpt_xdsl(rewritten_ir.attrs["xdsl_text"], ["bp_tree_runtime_update", "runtime_bp_pass"], context=1))

    bp_call_ops = [op for op in rewritten_ir.ops.values() if op.attrs.get("runtime_bp_pass")]
    bp_call_op = bp_call_ops[0] if bp_call_ops else None
    bp_events = [event for event in rewritten_trace if bp_call_op is not None and event.static_op_id == bp_call_op.id]

    subsection("Runtime behavior after grafting")
    print("Execution schedule:")
    print("  stack decoder expands one node")
    print("  -> donor bp_tree_runtime_update(explored, frontier, costs, audit, damping)")
    print("  -> donor updates every explored node")
    print("  -> host search continues")
    print(f"\nBaseline result   = {baseline_result}")
    print(f"Rewritten result  = {rewritten_result}")
    print(f"Rewritten audit   = {rewritten_audit}")
    print(f"BP static op id   = {bp_call_op.id if bp_call_op is not None else 'N/A'}")
    print(
        "BP runtime events = "
        + pretty([{"event_id": event.event_id, "timestamp": event.timestamp} for event in bp_events])
    )

    cloned_ir = rewritten_ir.clone()
    cloned_audit: list[int] = []
    cloned_result, _, _ = execute_ir(cloned_ir, [costs, 3, cloned_audit])
    print(f"Cloned IR result  = {cloned_result}")
    print(f"Cloned IR audit   = {cloned_audit}")


def main() -> None:
    costs = [0.5, 0.25, 0.75]
    section("Algorithm-IR Walkthrough")
    print("This demo shows how the system injects BP into a stack decoder at multiple levels.")
    print("Run command:")
    print("  conda run --no-capture-output -n AutoGenOld python research/algorithm-IR/demo.py")
    print(f"Demo costs = {costs}")

    demo_local_score_injection(costs)
    demo_runtime_nested_injection(costs)

    section("Takeaway")
    print("1. The host and donor are first compiled into xDSL-backed IR.")
    print("2. A local rewrite region is selected from the host.")
    print("3. A boundary contract describes how a donor can fit there.")
    print("4. Grafting mutates xDSL blocks directly, then rebuilds FunctionIR from xDSL.")
    print("5. The stronger runtime demo shows true nested execution: expand -> BP pass on explored tree -> continue search.")


if __name__ == "__main__":
    main()
