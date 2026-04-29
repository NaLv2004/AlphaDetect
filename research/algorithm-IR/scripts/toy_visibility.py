"""Verify the 'small algorithm' visibility on the toy example."""
from algorithm_ir.frontend.ir_builder import compile_source_to_ir
from algorithm_ir.visualize import build_visible_ir, visibility_stats

src = """
def f(x: vec_cx, y: vec_cx):
    a = x + y
    b = a * x
    c = b + 1.0
    d = c * 2.0
    return d
"""
ir = compile_source_to_ir(src, "f")
tot, vis, hid, pct = visibility_stats(ir)
print(f"toy f(x,y): total={tot} visible={vis} hidden={hid} ({pct:.1f}% hidden)")
print()
print("Visible ops by opcode:")
for op in build_visible_ir(ir).ops.values():
    print(f"  {op.opcode:<10} inputs={op.inputs} outputs={op.outputs}")
