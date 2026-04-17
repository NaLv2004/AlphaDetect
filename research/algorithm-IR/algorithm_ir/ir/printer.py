from __future__ import annotations

from .model import FunctionIR


def _fmt_list(items: list[str]) -> str:
    return "[" + ", ".join(items) + "]"


def render_function_ir(func_ir: FunctionIR) -> str:
    lines: list[str] = []
    lines.append(f"FunctionIR(name={func_ir.name}, entry={func_ir.entry_block})")
    for block_id in func_ir.blocks:
        block = func_ir.blocks[block_id]
        lines.append(f"  Block {block.id} preds={block.preds} succs={block.succs}")
        for op_id in block.op_ids:
            op = func_ir.ops[op_id]
            attrs = f" attrs={op.attrs}" if op.attrs else ""
            lines.append(
                f"    {op.id}: {op.opcode} in={_fmt_list(op.inputs)} out={_fmt_list(op.outputs)}{attrs}"
            )
    return "\n".join(lines)

