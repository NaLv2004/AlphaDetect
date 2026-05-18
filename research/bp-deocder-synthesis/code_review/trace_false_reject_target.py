"""Verify the FALSE-reject hypothesis empirically.

The target program (from dump_false_reject.py output @ idx=1):
  Float.Sign Float.Log Env.GetCodeRate Float.Const0_1
  Exec.DoRange { FVec.At, Env.GetChannelLLR, Exec.While {Float.Sign, Float.LT,
                 Env.GetCodeRate, FVec.Len, Bool.FromInt, Float.Div} }
  Float.Const1 FVec.Len Bool.FromInt FVec.Push Float.Sub Float.Div
  Exec.DoRange { Float.Tanh, Float.Exp }

Hypothesis: concrete VM output does NOT depend on X at all (DoRange skipped due
to empty int stack); only depends on LV.  Symbolic VM erroneously assigns X[3].

Tests:
  (a) Run concrete with fixed L=1.0, vary X = N random vectors -> all y must equal.
  (b) Run concrete with fixed X = zeros, vary L -> y must change with L.
  (c) Replace every entry of X with completely wild values (1e6, -1e6, NaN-ish) -> y must NOT change.
"""
import os, sys, time, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cpp_seeder"))
import pushgp_cpp_seeder as M

DEG = 8
N_IN = DEG - 1
SIDE = "v2c"

# Re-build the exact program from the dump
def build_target_prog():
    ops = [
        {"name": "Float.Sign"},
        {"name": "Float.Log"},
        {"name": "Env.GetCodeRate"},
        {"name": "Float.Const0_1"},
        {"name": "Exec.DoRange", "code_block": [
            {"name": "FVec.At"},
            {"name": "Env.GetChannelLLR"},
            {"name": "Exec.While", "code_block": [
                {"name": "Float.Sign"},
                {"name": "Float.LT"},
                {"name": "Env.GetCodeRate"},
                {"name": "FVec.Len"},
                {"name": "Bool.FromInt"},
                {"name": "Float.Div"},
            ]},
        ]},
        {"name": "Float.Const1"},
        {"name": "FVec.Len"},
        {"name": "Bool.FromInt"},
        {"name": "FVec.Push"},
        {"name": "Float.Sub"},
        {"name": "Float.Div"},
        {"name": "Exec.DoRange", "code_block": [
            {"name": "Float.Tanh"},
            {"name": "Float.Exp"},
        ]},
    ]
    return M.build_program(ops)

def run(prog, X, L):
    evo = np.zeros(8, dtype=np.float64)
    return M.run_program(prog, SIDE, evo, X.astype(np.float64), float(L), DEG, 0)

prog = build_target_prog()

print("=== Test (a): fix L=1.0, vary X across 200 random vectors ===")
rng = np.random.default_rng(20251128)
L = 1.0
ys = []
for _ in range(200):
    X = rng.uniform(-3, 3, size=N_IN)
    y = run(prog, X, L)
    ys.append(y)
ys_clean = [y for y in ys if y is not None]
print(f"  total = {len(ys)}, returned-y = {len(ys_clean)}, None = {len(ys) - len(ys_clean)}")
if ys_clean:
    arr = np.array(ys_clean)
    print(f"  y stats: min={arr.min():.6g} max={arr.max():.6g} std={arr.std():.6g} unique={len(np.unique(np.round(arr,9)))}")

print("\n=== Test (b): fix X=zeros, vary L in [-2, 2] across 20 values ===")
X = np.zeros(N_IN)
for L in np.linspace(-2, 2, 21):
    y = run(prog, X, L)
    print(f"  L={L:+.2f}  y={y}")

print("\n=== Test (c): WILD X (1e6, -1e6, NaN-ish), same L=1.0 ===")
configs = [
    ("zeros",        np.zeros(N_IN)),
    ("ones",         np.ones(N_IN)),
    ("big-pos",      np.full(N_IN, 1e6)),
    ("big-neg",      np.full(N_IN, -1e6)),
    ("alt-sign",     np.array([(-1)**k * 1e3 for k in range(N_IN)])),
    ("X[3]=huge",    np.array([0,0,0,1e9,0,0,0], dtype=float)),
    ("X[2]=huge",    np.array([0,0,1e9,0,0,0,0], dtype=float)),
]
L = 1.0
for label, X in configs:
    y = run(prog, X, L)
    print(f"  {label:15s} y = {y}")
