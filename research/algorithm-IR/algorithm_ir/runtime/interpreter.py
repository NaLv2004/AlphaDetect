from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from typing import Any

from algorithm_ir.ir.model import FunctionIR, Op
from algorithm_ir.runtime.frames import RuntimeFrame
from algorithm_ir.runtime.shadow_store import ShadowStore
from algorithm_ir.runtime.tracer import RuntimeEvent, RuntimeValue


@dataclass
class ExecutionState:
    func_ir: FunctionIR
    frame: RuntimeFrame
    rid_counter: itertools.count
    event_counter: itertools.count
    runtime_values: dict[str, RuntimeValue]
    rid_to_obj: dict[str, Any]
    trace: list[RuntimeEvent]
    shadow_store: ShadowStore
    loop_iterations: dict[str, int]


def _literal_copy(value):
    if isinstance(value, (int, float, str, bool, type(None))):
        return value
    if callable(value):
        return value
    import types as _types
    if isinstance(value, _types.ModuleType):
        return value
    return copy.deepcopy(value)


def execute_ir(func_ir: FunctionIR, args: list[object]) -> tuple[object, list[RuntimeEvent], dict[str, RuntimeValue]]:
    frame = RuntimeFrame(frame_id="frame_0", function_id=func_ir.id, parent_frame_id=None, callsite_event_id=None)
    state = ExecutionState(
        func_ir=func_ir,
        frame=frame,
        rid_counter=itertools.count(),
        event_counter=itertools.count(),
        runtime_values={},
        rid_to_obj={},
        trace=[],
        shadow_store=ShadowStore(),
        loop_iterations={},
    )
    value_bindings: dict[str, str] = {}
    for static_value_id, py_value in zip(func_ir.arg_values, args):
        rid = _new_runtime_value(state, static_value_id, py_value, "arg_init", None)
        value_bindings[static_value_id] = rid
        name_hint = func_ir.values[static_value_id].attrs.get("var_name") or func_ir.values[static_value_id].name_hint
        if name_hint:
            state.frame.locals[name_hint] = rid

    current_block = func_ir.entry_block
    incoming_block: str | None = None
    result = None
    while True:
        block = func_ir.blocks[current_block]
        state.frame.attrs["incoming_block"] = incoming_block
        branch_taken = False
        for op_id in block.op_ids:
            op = func_ir.ops[op_id]
            output, next_block = _execute_op(state, op, value_bindings)
            if op.opcode == "return":
                result = output
                branch_taken = True
                current_block = "__return__"
                break
            if next_block is not None:
                incoming_block = current_block
                current_block = next_block
                branch_taken = True
                break
        if current_block == "__return__":
            break
        if not branch_taken:
            if len(block.succs) == 1:
                incoming_block = current_block
                current_block = block.succs[0]
            else:
                break

    for runtime_value in state.runtime_values.values():
        runtime_value.metadata.setdefault("frames", {})[state.frame.frame_id] = {
            "function_id": state.frame.function_id,
            "locals": dict(state.frame.locals),
        }
        runtime_value.metadata.setdefault("shadow_store", {})["object_versions"] = dict(state.shadow_store.object_versions)
        runtime_value.metadata["shadow_store"]["field_writers"] = dict(state.shadow_store.field_writers)
        runtime_value.metadata["shadow_store"]["item_writers"] = dict(state.shadow_store.item_writers)
        runtime_value.metadata["shadow_store"]["container_membership"] = {
            key: sorted(value) for key, value in state.shadow_store.container_membership.items()
        }

    return result, state.trace, state.runtime_values


def _execute_op(
    state: ExecutionState,
    op: Op,
    bindings: dict[str, str],
) -> tuple[object | None, str | None]:
    if op.opcode == "phi":
        incoming_block = state.frame.attrs.get("incoming_block")
        block = state.func_ir.blocks[op.block_id]
        chosen_index = 0
        if incoming_block in block.preds:
            chosen_index = block.preds.index(incoming_block)
        chosen_index = min(chosen_index, len(op.inputs) - 1)
        chosen_input = op.inputs[chosen_index]
        input_rids = [bindings[chosen_input]]
        input_objs = [state.rid_to_obj[input_rids[0]]]
    else:
        input_rids = [bindings[value_id] for value_id in op.inputs]
        input_objs = [state.rid_to_obj[rid] for rid in input_rids]

    outputs: list[tuple[str, Any]] = []
    next_block: str | None = None
    result: object | None = None
    event_attrs = dict(op.attrs)

    if op.opcode == "const":
        literal = op.attrs.get("literal")
        outputs.append((op.outputs[0], _literal_copy(literal)))
    elif op.opcode == "assign":
        outputs.append((op.outputs[0], input_objs[0]))
    elif op.opcode == "binary":
        lhs, rhs = input_objs
        outputs.append((op.outputs[0], _apply_binary(op.attrs["operator"], lhs, rhs)))
    elif op.opcode == "unary":
        outputs.append((op.outputs[0], _apply_unary(op.attrs["operator"], input_objs[0])))
    elif op.opcode == "compare":
        outputs.append((op.outputs[0], _apply_compare(op.attrs["operators"], input_objs)))
    elif op.opcode == "phi":
        outputs.append((op.outputs[0], input_objs[0]))
    elif op.opcode == "call":
        fn = input_objs[0]
        args = input_objs[1:]
        outputs.append((op.outputs[0], fn(*args)))
    elif op.opcode == "get_attr":
        outputs.append((op.outputs[0], getattr(input_objs[0], op.attrs["attr"])))
    elif op.opcode == "set_attr":
        setattr(input_objs[0], op.attrs["attr"], input_objs[1])
    elif op.opcode == "get_item":
        outputs.append((op.outputs[0], input_objs[0][input_objs[1]]))
    elif op.opcode == "set_item":
        input_objs[0][input_objs[1]] = input_objs[2]
    elif op.opcode == "build_list":
        outputs.append((op.outputs[0], list(input_objs)))
    elif op.opcode == "build_tuple":
        outputs.append((op.outputs[0], tuple(input_objs)))
    elif op.opcode == "build_dict":
        built = {}
        for index in range(0, len(input_objs), 2):
            built[input_objs[index]] = input_objs[index + 1]
        outputs.append((op.outputs[0], built))
    elif op.opcode == "append":
        input_objs[0].append(input_objs[1])
    elif op.opcode == "pop":
        if len(input_objs) == 2:
            outputs.append((op.outputs[0], input_objs[0].pop(input_objs[1])))
        else:
            outputs.append((op.outputs[0], input_objs[0].pop()))
    elif op.opcode == "iter_init":
        outputs.append((op.outputs[0], iter(input_objs[0])))
    elif op.opcode == "iter_next":
        iterator = input_objs[0]
        try:
            nxt = next(iterator)
            outputs.append((op.outputs[0], nxt))
            outputs.append((op.outputs[1], True))
        except StopIteration:
            outputs.append((op.outputs[0], None))
            outputs.append((op.outputs[1], False))
    elif op.opcode == "branch":
        cond = bool(input_objs[0])
        event_attrs["branch_taken"] = "true" if cond else "false"
        next_block = op.attrs["true"] if cond else op.attrs["false"]
        result = cond
    elif op.opcode == "jump":
        next_block = op.attrs["target"]
    elif op.opcode == "return":
        result = input_objs[0]
    else:
        raise NotImplementedError(f"Unsupported opcode {op.opcode}")

    event_id = f"evt_{next(state.event_counter)}"
    output_rids: list[str] = []
    for static_output_id, py_output in outputs:
        rid = _new_runtime_value(state, static_output_id, py_output, event_id, None)
        bindings[static_output_id] = rid
        output_rids.append(rid)
        value_meta = state.func_ir.values[static_output_id].attrs
        var_name = value_meta.get("var_name")
        if var_name:
            state.frame.locals[var_name] = rid

    event = RuntimeEvent(
        event_id=event_id,
        static_op_id=op.id,
        frame_id=state.frame.frame_id,
        timestamp=len(state.trace),
        input_rids=input_rids,
        output_rids=output_rids,
        control_context=_build_control_context(state, op),
        attrs=event_attrs,
    )
    state.trace.append(event)

    if op.opcode == "set_attr":
        state.shadow_store.note_set_attr(input_objs[0], op.attrs["attr"], event_id)
    elif op.opcode == "set_item":
        state.shadow_store.note_set_item(input_objs[0], input_objs[1], event_id)
    elif op.opcode == "append":
        appended_obj = input_objs[1]
        appended_rid = input_rids[1]
        state.shadow_store.note_append(input_objs[0], appended_rid, event_id)
        state.shadow_store.note_value(appended_obj, appended_rid)
    elif op.opcode == "build_list":
        container_obj = state.rid_to_obj[output_rids[0]]
        for input_rid in input_rids:
            state.shadow_store.note_append(container_obj, input_rid, event_id)
    elif op.opcode == "build_dict":
        container_obj = state.rid_to_obj[output_rids[0]]
        for index in range(0, len(input_objs), 2):
            state.shadow_store.note_set_item(container_obj, input_objs[index], event_id)

    return result, next_block


def _new_runtime_value(
    state: ExecutionState,
    static_value_id: str,
    py_obj: Any,
    created_by_event: str,
    last_writer_event: str | None,
) -> str:
    rid = f"rid_{next(state.rid_counter)}"
    runtime_value = RuntimeValue(
        rid=rid,
        static_value_id=static_value_id,
        py_obj_id=id(py_obj) if py_obj is not None else None,
        created_by_event=created_by_event,
        last_writer_event=last_writer_event,
        frame_id=state.frame.frame_id,
        version=len(state.shadow_store.object_versions.get(id(py_obj), [])) if py_obj is not None else 0,
        metadata={"type_name": type(py_obj).__name__ if py_obj is not None else "NoneType"},
    )
    state.runtime_values[rid] = runtime_value
    state.rid_to_obj[rid] = py_obj
    state.shadow_store.note_value(py_obj, rid)
    return rid


def _apply_binary(operator: str, lhs, rhs):
    if operator == "Add":
        return lhs + rhs
    if operator == "Sub":
        return lhs - rhs
    if operator == "Mult":
        return lhs * rhs
    if operator == "Div":
        return lhs / rhs
    if operator == "FloorDiv":
        return lhs // rhs
    if operator == "Mod":
        return lhs % rhs
    raise NotImplementedError(f"Unsupported binary operator {operator}")


def _apply_unary(operator: str, value):
    if operator == "USub":
        return -value
    if operator == "UAdd":
        return +value
    if operator == "Not":
        return not value
    raise NotImplementedError(f"Unsupported unary operator {operator}")


def _apply_compare(operators: list[str], values: list[object]) -> bool:
    left = values[0]
    for op_name, right in zip(operators, values[1:]):
        if op_name == "Lt":
            okay = left < right
        elif op_name == "LtE":
            okay = left <= right
        elif op_name == "Gt":
            okay = left > right
        elif op_name == "GtE":
            okay = left >= right
        elif op_name == "Eq":
            okay = left == right
        elif op_name == "NotEq":
            okay = left != right
        else:
            raise NotImplementedError(f"Unsupported compare operator {op_name}")
        if not okay:
            return False
        left = right
    return True


def _build_control_context(state: ExecutionState, op: Op) -> tuple[str, ...]:
    block = state.func_ir.blocks[op.block_id]
    contexts: list[str] = [f"block:{block.id}"]
    if "loop_backedge" in op.attrs:
        count = state.loop_iterations.get(block.id, 0) + 1
        state.loop_iterations[block.id] = count
        contexts.append(f"loop:{block.id}:iter={count}")
    return tuple(contexts)
