"""Probe concrete VM semantics:
  1) run_program return value: top of float vs something else?
  2) FVec.Push when fvec stack empty: no-op or pops the float?
  3) Does Exec.DoRange really skip with empty int stack?
"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cpp_seeder"))
import pushgp_cpp_seeder as M

evo = np.zeros(8, dtype=np.float64)
X = np.zeros(7, dtype=np.float64)
L = 1.0

def run(ops):
    p = M.build_program(ops)
    return M.run_program(p, "v2c", evo, X, L, 8, 0)

# Test 1: push 1.0, 2.0, 3.0 -> what does run return?
y = run([{"name": "Float.Const1"}, {"name": "Float.Const2"}, {"name": "Float.Const0_5" if False else "Float.ConstHalf"}])
print(f"Test1 push 1,2,0.5 -> y={y}  (top=0.5? bottom=1?)")

# Test 2: empty program
y = run([])
print(f"Test2 empty prog -> y={y}  (init float stack = [LV] only)")

# Test 3: Float.Const2 then nothing
y = run([{"name": "Float.Const2"}])
print(f"Test3 push 2 (after init [LV=1]) -> y={y}  (top=2? bottom=1?)")

# Test 4: Float.Pop then Float.Const2
y = run([{"name": "Float.Pop"}, {"name": "Float.Const2"}])
print(f"Test4 pop init, push 2 -> y={y}  (=2 if top is returned)")

# Test 5: DoRange with empty int stack and body push 99
y = run([{"name": "Exec.DoRange", "code_block": [{"name": "Float.Const2"}]}])
print(f"Test5 DoRange{{Const2}} (int empty) -> y={y}  (init [LV=1], body should NOT run -> still LV=1)")

# Test 6: DoRange with int stack {0,5} prepared, body push 99
y = run([
    {"name": "Int.Const0"}, {"name": "Int.Const2"}, {"name": "Int.Const2"},
    {"name": "Exec.DoRange", "code_block": [{"name": "Float.Const2"}]}
])
print(f"Test6 DoRange [start=0 end=2] {{Const2}} -> y={y}  (body runs twice, push 2 twice)")

# Test 7: Just two Tanh-Exp on a base value
y = run([{"name": "Float.Const2"}, {"name": "Float.Tanh"}, {"name": "Float.Exp"}])
print(f"Test7 push 2, tanh, exp -> y={y}  (exp(tanh(2))~={np.exp(np.tanh(2)):.4f})")

# Test 8: many Tanh-Exp in a row (manual unroll, no DoRange)
ops = [{"name": "Float.Const2"}]
for _ in range(20):
    ops.append({"name": "Float.Tanh"})
    ops.append({"name": "Float.Exp"})
y = run(ops)
print(f"Test8 push 2, (tanh,exp)*20 -> y={y}  (fixed-point of exp(tanh(.)) ~ 2.6936...)")

# Test 9: FVec.Push when fvec empty
# init fvec=[[X0..X6]]; pop with FVec.Len → fvec empty.
y = run([
    {"name": "FVec.Len"},       # pop fvec, push int=7
    {"name": "Float.Const2"},   # float=[LV, 2]
    {"name": "FVec.Push"},      # needs fvec + float; fvec empty
    # now check float: if no-op then float=[LV,2] top=2; if pops float then float=[LV] top=LV=1.
])
print(f"Test9 FVec.Push when fvec empty -> y={y}  (=2 means no-op; =1 means pops float)")

print("\n--- now trace the exact target prog ---")
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
# Trace by running prefixes
for k in range(1, len(ops)+1):
    y = run(ops[:k])
    print(f"  prefix len {k:2d} (last: {ops[k-1]['name']:25s}) -> y = {y}")
