"""Detailed diagnostics: kbest symbol ordering + stack max_nodes + bp convergence."""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from evolution.ir_pool import _template_globals, KBEST_TEMPLATE, STACK_TEMPLATE, BP_TEMPLATE, AMP_TEMPLATE
from evolution.ir_pool import SLOT_DEFAULTS
from evolution.mimo_evaluator import qam16_constellation, generate_mimo_sample

constellation = qam16_constellation()
rng = np.random.default_rng(42)
Nr, Nt = 16, 16
snr_db = 24.0
H, x_true, y, sigma2 = generate_mimo_sample(Nr, Nt, constellation, snr_db, rng)

g = _template_globals()

# ============================================================
# 1. kbest: check if reversing symbols fixes it
# ============================================================
print("=" * 60)
print("KBEST DIAGNOSIS")
print("=" * 60)

# Compile expand, prune, and kbest
exec(SLOT_DEFAULTS["expand"], g)
exec(SLOT_DEFAULTS["prune"], g)
exec(KBEST_TEMPLATE, g)

x_kbest = g["kbest"](H, y, sigma2, constellation, g["expand"], g["prune"])
ser_fwd = np.mean(np.abs(x_true - x_kbest) > 1e-6)

x_kbest_rev = x_kbest[::-1]
ser_rev = np.mean(np.abs(x_true - x_kbest_rev) > 1e-6)

print(f"  kbest forward  SER = {ser_fwd:.4f}")
print(f"  kbest reversed SER = {ser_rev:.4f}")
print(f"  x_true[:4]  = {x_true[:4]}")
print(f"  x_fwd[:4]   = {x_kbest[:4]}")
print(f"  x_rev[:4]   = {x_kbest_rev[:4]}")
print()

# ============================================================
# 2. Stack: what happens with larger max_nodes?
# ============================================================
print("=" * 60)
print("STACK DIAGNOSIS")
print("=" * 60)

# Direct exec with modified max_nodes
exec(SLOT_DEFAULTS["node_select"], g)
# expand already in g

for max_n in [500, 2000, 10000]:
    stack_code = f"""\
def stack_test(H, y, sigma2, constellation, slot_node_select, slot_expand):
    Nr = H.shape[0]
    Nt = H.shape[1]
    Q = _qr_Q(H)
    R = _qr_R(H)
    y_tilde = Q.conj().T @ y
    root = _make_tree_node(Nt - 1, [], 0.0)
    open_set = [root]
    nodes_expanded = 0
    max_nodes = {max_n}
    _done = 0
    while _done == 0:
        if len(open_set) == 0:
            _done = 1
        if nodes_expanded >= max_nodes:
            _done = 1
        if _done == 0:
            best_idx = slot_node_select(open_set)
            node = open_set.pop(best_idx)
            nodes_expanded = nodes_expanded + 1
            if len(node.symbols) == Nt:
                x_out = _carray(node.symbols)
                return x_out
            children = slot_expand(node, y_tilde, R, constellation)
            open_set = open_set + children
    if len(open_set) > 0:
        best = open_set[0]
        i = 1
        while i < len(open_set):
            if open_set[i].cost < best.cost:
                best = open_set[i]
            i = i + 1
        if len(best.symbols) == Nt:
            return _carray(best.symbols)
    return _czeros(Nt)
"""
    exec(stack_code, g)
    try:
        x_stack = g["stack_test"](H, y, sigma2, constellation, g["node_select"], g["expand"])
        ser = np.mean(np.abs(x_true - x_stack) > 1e-6)
        ser_rev = np.mean(np.abs(x_true - x_stack[::-1]) > 1e-6)
        is_zero = np.allclose(x_stack, 0)
        print(f"  max_nodes={max_n:5d}: SER(fwd)={ser:.4f}  SER(rev)={ser_rev:.4f}  allzero={is_zero}")
    except Exception as e:
        print(f"  max_nodes={max_n:5d}: ERROR {e}")
print()

# ============================================================
# 3. BP: check convergence
# ============================================================
print("=" * 60)
print("BP DIAGNOSIS - checking iterations")
print("=" * 60)

exec(SLOT_DEFAULTS["bp_sweep"], g)
exec(SLOT_DEFAULTS["final_decision"], g)

# Run BP with different iteration counts
for max_iters in [1, 5, 10, 20, 50]:
    # Manual BP call
    x_mf = H.conj().T @ y
    G_diag = np.real(np.sum(np.abs(H) ** 2, axis=0))
    G_diag = np.maximum(G_diag, 1e-30)
    init_mu = x_mf / G_diag
    init_var = sigma2 / G_diag
    mu, var = g["bp_sweep"](H, y, sigma2, init_mu, init_var, constellation, max_iters)
    x_hat = g["final_decision"](mu, constellation)
    ser = np.mean(np.abs(x_true - x_hat) > 1e-6)
    mu_norm = np.linalg.norm(mu)
    var_mean = np.mean(var)
    print(f"  iters={max_iters:3d}: SER={ser:.4f}  |mu|={mu_norm:.4f}  mean(var)={var_mean:.6f}")

print()

# ============================================================
# 4. AMP: check convergence
# ============================================================
print("=" * 60)
print("AMP DIAGNOSIS")
print("=" * 60)

exec(SLOT_DEFAULTS["amp_iterate"], g)

x_hat = np.zeros(Nt, dtype=complex)
s_hat = np.ones(Nt)
z = np.zeros(Nr, dtype=complex)

for it in range(1, 21):
    x_hat, s_hat, z = g["amp_iterate"](H, y, sigma2, x_hat, s_hat, z, constellation)
    x_final = g["final_decision"](x_hat, constellation)
    ser = np.mean(np.abs(x_true - x_final) > 1e-6)
    if it <= 5 or it % 5 == 0:
        print(f"  iter={it:3d}: SER={ser:.4f}  |x|={np.linalg.norm(x_hat):.4f}  mean(s)={np.mean(s_hat):.6f}")

print()

# ============================================================
# 5. EP: check convergence
# ============================================================
print("=" * 60)
print("EP DIAGNOSIS")
print("=" * 60)

exec(SLOT_DEFAULTS["cavity"], g)
exec(SLOT_DEFAULTS["site_update"], g)

from evolution.ir_pool import _template_globals as _tg
exec(EP_TEMPLATE, g)
x_ep = g["ep"](H, y, sigma2, constellation, g["cavity"], g["site_update"], g["final_decision"])
ser = np.mean(np.abs(x_true - x_ep) > 1e-6)
print(f"  EP SER = {ser:.4f}")
