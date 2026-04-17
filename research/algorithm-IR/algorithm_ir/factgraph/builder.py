from __future__ import annotations

from algorithm_ir.factgraph.model import FactGraph
from algorithm_ir.ir.model import FunctionIR
from algorithm_ir.runtime.frames import RuntimeFrame
from algorithm_ir.runtime.tracer import RuntimeEvent, RuntimeValue


def build_factgraph(
    func_ir: FunctionIR,
    runtime_trace: list[RuntimeEvent],
    runtime_values: dict[str, RuntimeValue],
) -> FactGraph:
    runtime_events = {event.event_id: event for event in runtime_trace}
    runtime_frames: dict[str, RuntimeFrame] = {}
    for runtime_value in runtime_values.values():
        frame_meta = runtime_value.metadata.get("frames", {})
        for frame_id, frame_data in frame_meta.items():
            runtime_frames[frame_id] = RuntimeFrame(
                frame_id=frame_id,
                function_id=frame_data["function_id"],
                parent_frame_id=None,
                callsite_event_id=None,
                locals=frame_data["locals"],
                attrs={},
            )

    fg = FactGraph(
        static_functions={func_ir.id: func_ir},
        static_ops=func_ir.ops,
        static_values=func_ir.values,
        runtime_events=runtime_events,
        runtime_values=runtime_values,
        runtime_frames=runtime_frames,
        static_edges={},
        dynamic_edges={},
        alignment_edges={},
        metadata={"function_name": func_ir.name},
    )

    fg.static_edges["def_use"] = set()
    fg.static_edges["use_def"] = set()
    fg.static_edges["cfg"] = set()
    fg.static_edges["call_static"] = set()
    for op in func_ir.ops.values():
        for inp in op.inputs:
            fg.static_edges["use_def"].add((op.id, inp))
            if func_ir.values[inp].def_op:
                fg.static_edges["def_use"].add((func_ir.values[inp].def_op, op.id))
        if op.opcode == "call" and op.inputs:
            fg.static_edges["call_static"].add((op.id, op.inputs[0]))
    for block in func_ir.blocks.values():
        for succ in block.succs:
            fg.static_edges["cfg"].add((block.id, succ))

    fg.dynamic_edges["event_input"] = set()
    fg.dynamic_edges["event_output"] = set()
    fg.dynamic_edges["temporal"] = set()
    fg.dynamic_edges["control_dynamic"] = set()
    fg.dynamic_edges["call_dynamic"] = set()
    fg.dynamic_edges["frame_nesting"] = set()
    fg.dynamic_edges["container_membership"] = set()
    fg.dynamic_edges["field_slot"] = set()
    for index, event in enumerate(runtime_trace):
        for rid in event.input_rids:
            fg.dynamic_edges["event_input"].add((rid, event.event_id))
        for rid in event.output_rids:
            fg.dynamic_edges["event_output"].add((event.event_id, rid))
        if index + 1 < len(runtime_trace):
            fg.dynamic_edges["temporal"].add((event.event_id, runtime_trace[index + 1].event_id))
        for token in event.control_context:
            fg.dynamic_edges["control_dynamic"].add((token, event.event_id))

    for rid, runtime_value in runtime_values.items():
        fg.alignment_edges.setdefault("instantiates_value", set()).add((rid, runtime_value.static_value_id))
        fg.alignment_edges.setdefault("runtime_in_frame", set()).add((rid, runtime_value.frame_id))
        shadow_meta = runtime_value.metadata.get("shadow_store", {})
        for container_id, members in shadow_meta.get("container_membership", {}).items():
            for member_rid in members:
                fg.dynamic_edges["container_membership"].add((f"container:{container_id}", member_rid))
        for (owner_id, field_name), writer in shadow_meta.get("field_writers", {}).items():
            fg.dynamic_edges["field_slot"].add((writer, f"field:{owner_id}:{field_name}"))

    for event in runtime_trace:
        fg.alignment_edges.setdefault("instantiates_op", set()).add((event.event_id, event.static_op_id))
        branch_taken = event.attrs.get("branch_taken")
        if branch_taken:
            fg.dynamic_edges["control_dynamic"].add((f"branch:{branch_taken}", event.event_id))

    return fg

