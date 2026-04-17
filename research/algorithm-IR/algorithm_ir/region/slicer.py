from __future__ import annotations

from collections import deque

from algorithm_ir.ir.model import FunctionIR


def backward_slice_by_values(func_ir: FunctionIR, exit_values: list[str]) -> set[str]:
    seen_values = set(exit_values)
    queue = deque(exit_values)
    selected_ops: set[str] = set()
    while queue:
        value_id = queue.popleft()
        def_op = func_ir.values[value_id].def_op
        if def_op is None or def_op in selected_ops:
            continue
        selected_ops.add(def_op)
        for input_value in func_ir.ops[def_op].inputs:
            if input_value not in seen_values:
                seen_values.add(input_value)
                queue.append(input_value)
    return selected_ops


def forward_slice_from_values(func_ir: FunctionIR, seed_values: list[str]) -> set[str]:
    queue = deque(seed_values)
    seen_values = set(seed_values)
    selected_ops: set[str] = set()
    while queue:
        value_id = queue.popleft()
        for use_op in func_ir.values[value_id].use_ops:
            if use_op in selected_ops:
                continue
            selected_ops.add(use_op)
            for output in func_ir.ops[use_op].outputs:
                if output not in seen_values:
                    seen_values.add(output)
                    queue.append(output)
    return selected_ops

