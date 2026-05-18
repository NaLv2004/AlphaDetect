"""Symbolic per-op trace for the false-reject target program.
Build same prog as trace_false_reject_target.py, run symbolic_trace_v2c,
print every step's float-stack top + key intermediate state to find where
X-atom dependency disappears from the float top."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cpp_seeder"))
import pushgp_cpp_seeder as M

ops = [
    {"name": "Float.Sign"},
    {"name": "Float.Log"},
    {"name": "Env.GetCodeRate"},
    {"name": "Float.Const0_1"},
    {"name": "Exec.DoRange", "code_block": [
        {"name": "FVec.At"},
        {"name": "Env.GetChannelLLR"},
        {"name": "Exec.While", "code_block": [
            {"name": "Float.Sign"}, {"name": "Float.LT"},
            {"name": "Env.GetCodeRate"}, {"name": "FVec.Len"},
            {"name": "Bool.FromInt"}, {"name": "Float.Div"},
        ]},
    ]},
    {"name": "Float.Const1"},
    {"name": "FVec.Len"},
    {"name": "Bool.FromInt"},
    {"name": "FVec.Push"},
    {"name": "Float.Sub"},
    {"name": "Float.Div"},
    {"name": "Exec.DoRange", "code_block": [
        {"name": "Float.Tanh"}, {"name": "Float.Exp"},
    ]},
]
prog = M.build_program(ops)
result = M.symbolic_trace_v2c(prog, 8, 0)
print(f"opaque={result['opaque']} reason={result['opaque_reason']!r} steps={result['step_count']} branches={result['branches_seen']}")
print()
prev_fvec_count = None
for i, s in enumerate(result["steps"]):
    f = s["float"]
    iv = s["int"]
    fv = s["fvec"]
    bv = s["bool"]
    f_top = f[-1] if f else "<empty>"
    has_x = any("X[" in x for x in f)
    marker = " *X*" if has_x else "    "
    print(f"[{i:3d}] {s['op']:25s} | F.depth={len(f):2d} I.depth={len(iv):2d} B.depth={len(bv):2d} FV.n={len(fv)} | top={f_top[:70]}{marker}")

# print full final stacks
print()
print("=== FINAL STACKS ===")
final = result["steps"][-1] if result["steps"] else {}
for k in ("float","int","bool","fvec"):
    v = final.get(k, [])
    print(f"{k}: {v}")
