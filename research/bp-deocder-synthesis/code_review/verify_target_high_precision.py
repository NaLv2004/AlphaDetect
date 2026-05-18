"""Verify with high precision: target prog y1 vs y2 (swap X[3]<->X[0])
ULP-level diff to confirm symbolic non-invariance is real."""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cpp_seeder"))
import pushgp_cpp_seeder as M

ops = [
    {"name": "Float.Sign"}, {"name": "Float.Log"},
    {"name": "Env.GetCodeRate"}, {"name": "Float.Const0_1"},
    {"name": "Exec.DoRange", "code_block": [
        {"name": "FVec.At"}, {"name": "Env.GetChannelLLR"},
        {"name": "Exec.While", "code_block": [
            {"name": "Float.Sign"}, {"name": "Float.LT"},
            {"name": "Env.GetCodeRate"}, {"name": "FVec.Len"},
            {"name": "Bool.FromInt"}, {"name": "Float.Div"},
        ]},
    ]},
    {"name": "Float.Const1"}, {"name": "FVec.Len"}, {"name": "Bool.FromInt"},
    {"name": "FVec.Push"}, {"name": "Float.Sub"}, {"name": "Float.Div"},
    {"name": "Exec.DoRange", "code_block": [
        {"name": "Float.Tanh"}, {"name": "Float.Exp"},
    ]},
]
prog = M.build_program(ops)
evo = np.zeros(8, dtype=np.float64)
rng = np.random.default_rng(0)
L = 1.0
n_diff = 0
n_total = 100
max_diff = 0.0
for trial in range(n_total):
    X = rng.uniform(-2, 2, 7).astype(np.float64)
    Xs = X.copy()
    Xs[3], Xs[0] = X[0], X[3]
    y1 = M.run_program(prog, "v2c", evo, X, L, 8, 0)
    y2 = M.run_program(prog, "v2c", evo, Xs, L, 8, 0)
    if y1 is None or y2 is None: continue
    d = abs(y1 - y2)
    max_diff = max(max_diff, d)
    if d > 0.0:
        n_diff += 1
        if n_diff <= 5:
            print(f"trial {trial:3d}: y1={y1!r:25s} y2={y2!r:25s}  |diff|={d:.3e}")
print(f"\nTotal: {n_diff}/{n_total} trials show y1 != y2 exactly, max |diff| = {max_diff:.3e}")
print("If n_diff > 0  → program IS structurally X-dependent in a permutation-NON-invariant way")
print("                 → symbolic rejection was CORRECT all along")

# Also try with X far from convergence basin
print("\n--- adversarial X (X[3] near LV to amplify divergence): ---")
for X3 in [1.01, 1.001, 0.999, 0.99, 0.5, 1.5, 2.0]:
    X = np.array([0.1, 0.2, 0.3, X3, 0.5, 0.6, 0.7])
    Xs = X.copy(); Xs[3], Xs[0] = X[0], X[3]
    y1 = M.run_program(prog, "v2c", evo, X, L, 8, 0)
    y2 = M.run_program(prog, "v2c", evo, Xs, L, 8, 0)
    print(f"  X[3]={X3:6.3f}  y1={y1!r:22s}  y2={y2!r:22s}  |diff|={abs(y1-y2):.3e}")
