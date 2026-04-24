"""IR-based algorithm pool: template sources compiled to FunctionIR with AlgSlot ops.

This module provides:

1. **Template source strings** for every L3 detector.  Slot positions are
   ordinary function parameters whose names start with ``slot_``.  After
   compilation these are converted to ``AlgSlot`` ops in the xDSL module.

2. **Default implementations** for every slot (as source strings compiled
   to FunctionIR).

3. ``build_ir_pool()`` — the main entry point that returns a list of
   ``AlgorithmGenome`` objects ready for the two-level evolution engine.

Architecture
------------
    template source  →  compile_source_to_ir  →  FunctionIR
                                                  ↓
                                    convert_slot_calls_to_algslot
                                                  ↓
                                    FunctionIR with AlgSlot ops
                                        (structural_ir)
"""

from __future__ import annotations

import re
import textwrap
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from algorithm_ir.frontend.ir_builder import compile_source_to_ir
from algorithm_ir.ir.model import FunctionIR, Op, Value, Block
from algorithm_ir.ir.xdsl_bridge import create_xdsl_op_from_payload

from evolution.pool_types import (
    AlgorithmGenome,
    AlgorithmEntry,
    SlotDescriptor,
    SlotPopulation,
)
from evolution.algorithm_pool import (
    _lmmse_slots, _zf_slots, _osic_slots, _kbest_slots,
    _bp_slots, _ep_slots, _amp_slots, _stack_slots,
)
from evolution.skeleton_registry import ProgramSpec
from evolution.random_program import random_ir_program
from evolution.skeleton_library import get_extended_specs, EXTENDED_SLOT_DEFAULTS, SkeletonSpec


# ═══════════════════════════════════════════════════════════════════════════
# Helper: namespace for template compilation
# ═══════════════════════════════════════════════════════════════════════════

def _template_globals() -> dict[str, Any]:
    """Safe namespace for compiling detector templates to IR.

    Now that the IR builder supports keyword arguments (ast.keyword),
    most helper wrappers are no longer needed. We keep only helpers
    that perform multi-dimensional slicing (not expressible as simple
    keyword calls) and safe-math wrappers.
    """
    import math

    def _safe_div(a, b):
        return a / b if abs(b) > 1e-30 else 0.0

    def _safe_sqrt(a):
        return math.sqrt(max(a, 0.0))

    def _safe_log(a):
        return math.log(max(a, 1e-30))

    def _make_tree_node(level, symbols, cost):
        from evolution.pool_ops_l2 import TreeNode
        return TreeNode(level=level, symbols=symbols, cost=cost)

    def _col(H, j):
        return H[:, j]

    def _reverse_syms(symbols, Nt):
        """Reverse detection-order symbols to column-order."""
        result = np.zeros(Nt, dtype=complex)
        for si in range(len(symbols)):
            result[Nt - 1 - si] = symbols[si]
        return result

    def _row(mat, i):
        """Get row i of a 2D array."""
        return mat[i, :]

    def _row_set(mat, i, row_vals):
        """Set row i of a 2D array."""
        mat[i, :] = row_vals

    def _argmax_row(mat, i):
        """argmax of row i."""
        return int(np.argmax(mat[i, :]))

    def _row_normalize(mat, i):
        """Normalize row i to sum to 1."""
        s = np.sum(np.abs(mat[i, :]))
        if s > 1e-30:
            mat[i, :] = mat[i, :] / s

    return {
        "np": np,
        "math": math,
        "_safe_div": _safe_div,
        "_safe_sqrt": _safe_sqrt,
        "_safe_log": _safe_log,
        "_make_tree_node": _make_tree_node,
        "_col": _col,
        "_reverse_syms": _reverse_syms,
        "_row": _row,
        "_row_set": _row_set,
        "_argmax_row": _argmax_row,
        "_row_normalize": _row_normalize,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Default slot implementation sources
# ═══════════════════════════════════════════════════════════════════════════

# Each default implementation is a self-contained function source string
# that can be compiled to FunctionIR independently.

SLOT_DEFAULTS: dict[str, str] = {
    # ----- LMMSE / ZF -----
    "regularizer": textwrap.dedent("""\
        def regularizer(G, sigma2):
            n = G.shape[0]
            return G + sigma2 * np.eye(n)
    """),
    "hard_decision": textwrap.dedent("""\
        def hard_decision(x_soft, constellation):
            x_hat = np.zeros(len(x_soft), dtype=complex)
            i = 0
            while i < len(x_soft):
                dists = np.abs(constellation - x_soft[i]) ** 2
                x_hat[i] = constellation[np.argmin(dists)]
                i = i + 1
            return x_hat
    """),

    # ----- OSIC -----
    "ordering": textwrap.dedent("""\
        def ordering(H, y, sigma2):
            Nt = H.shape[1]
            G = H.conj().T @ H + sigma2 * np.eye(Nt)
            G_inv = np.linalg.inv(G)
            snr = np.zeros(Nt, dtype=complex)
            i = 0
            while i < Nt:
                snr[i] = 1.0 / max(float(np.real(G_inv[i, i])), 1e-30) - sigma2
                i = i + 1
            order = []
            used = np.zeros(Nt, dtype=complex)
            k = 0
            while k < Nt:
                best = -1
                best_val = -1e30
                j = 0
                while j < Nt:
                    if float(np.real(used[j])) == 0.0:
                        if float(np.real(snr[j])) > best_val:
                            best_val = float(np.real(snr[j]))
                            best = j
                    j = j + 1
                order.append(best)
                used[best] = 1.0
                k = k + 1
            return order
    """),
    "sic_step": textwrap.dedent("""\
        def sic_step(H, y, sigma2, idx, constellation):
            Nr = H.shape[0]
            Nt = H.shape[1]
            G = H.conj().T @ H + sigma2 * np.eye(Nt)
            w = np.linalg.solve(G, H.conj().T @ y)
            x_est = w[idx]
            dists = np.abs(constellation - x_est) ** 2
            x_hard = constellation[np.argmin(dists)]
            y_new = y - _col(H, idx) * x_hard
            H_new = np.delete(H, idx, axis=1)
            return x_hard, H_new, y_new
    """),

    # ----- K-Best -----
    "expand": textwrap.dedent("""\
        def expand(node, y_tilde, R, constellation):
            level = node.level
            Nt = R.shape[1]
            children = []
            interf = 0.0 + 0.0j
            j_rel = 0
            while j_rel < len(node.symbols):
                j = Nt - 1 - j_rel
                interf = interf + R[level, j] * node.symbols[j_rel]
                j_rel = j_rel + 1
            ci = 0
            while ci < len(constellation):
                sym = constellation[ci]
                residual = y_tilde[level] - R[level, level] * sym - interf
                local_cost = float(np.abs(residual) ** 2)
                total = node.cost + local_cost
                child = _make_tree_node(level - 1, node.symbols + [sym], total)
                children.append(child)
                ci = ci + 1
            return children
    """),
    "prune": textwrap.dedent("""\
        def prune(candidates, K):
            n = len(candidates)
            i = 0
            while i < n:
                j = i + 1
                while j < n:
                    if candidates[j].cost < candidates[i].cost:
                        tmp = candidates[i]
                        candidates[i] = candidates[j]
                        candidates[j] = tmp
                    j = j + 1
                i = i + 1
            return candidates[:K]
    """),

    # ----- Stack -----
    "node_select": textwrap.dedent("""\
        def node_select(open_set):
            best_idx = 0
            best_cost = open_set[0].cost
            i = 1
            while i < len(open_set):
                if open_set[i].cost < best_cost:
                    best_cost = open_set[i].cost
                    best_idx = i
                i = i + 1
            return best_idx
    """),

    # ----- BP -----
    "bp_sweep": textwrap.dedent("""\
        def bp_sweep(H, y, sigma2, Px, constellation, max_iters):
            Nr = H.shape[0]
            Nt = H.shape[1]
            M = len(constellation)
            damping = 0.5
            it = 0
            while it < max_iters:
                Muz = np.zeros((Nr, Nt), dtype=complex)
                sigmaz2 = np.zeros((Nr, Nt))
                a = 0
                while a < Nr:
                    j = 0
                    while j < Nt:
                        mu_j = 0.0 + 0.0j
                        var_j = 0.0
                        m = 0
                        while m < M:
                            mu_j = mu_j + Px[j, m] * constellation[m]
                            m = m + 1
                        m = 0
                        while m < M:
                            var_j = var_j + float(np.real(Px[j, m])) * float(np.abs(constellation[m] - mu_j) ** 2)
                            m = m + 1
                        Muz[a, j] = H[a, j] * mu_j
                        sigmaz2[a, j] = float(np.abs(H[a, j]) ** 2) * var_j
                        j = j + 1
                    a = a + 1
                Beta = np.zeros((Nr, Nt, M))
                a = 0
                while a < Nr:
                    j = 0
                    while j < Nt:
                        mu_sum = 0.0 + 0.0j
                        var_sum = sigma2
                        k = 0
                        while k < Nt:
                            if k != j:
                                mu_sum = mu_sum + Muz[a, k]
                                var_sum = var_sum + sigmaz2[a, k]
                            k = k + 1
                        var_sum = max(var_sum, 1e-30)
                        residual = y[a] - mu_sum
                        m = 0
                        while m < M:
                            beta_val = float(np.real(2.0 * np.conj(residual) * H[a, j] * (constellation[m] - constellation[0]) - np.abs(H[a, j]) ** 2 * (np.abs(constellation[m]) ** 2 - np.abs(constellation[0]) ** 2)))
                            Beta[a, j, m] = beta_val / (2.0 * var_sum)
                            m = m + 1
                        j = j + 1
                    a = a + 1
                Alpha = np.zeros((Nt, M))
                j = 0
                while j < Nt:
                    m = 0
                    while m < M:
                        s = 0.0
                        a = 0
                        while a < Nr:
                            s = s + Beta[a, j, m]
                            a = a + 1
                        Alpha[j, m] = s
                        m = m + 1
                    j = j + 1
                Px_new = np.zeros((Nt, M))
                j = 0
                while j < Nt:
                    max_alpha = -1e30
                    m = 0
                    while m < M:
                        if Alpha[j, m] > max_alpha:
                            max_alpha = Alpha[j, m]
                        m = m + 1
                    m = 0
                    s = 0.0
                    while m < M:
                        Px_new[j, m] = np.exp(Alpha[j, m] - max_alpha)
                        s = s + Px_new[j, m]
                        m = m + 1
                    s = max(s, 1e-30)
                    m = 0
                    while m < M:
                        Px_new[j, m] = Px_new[j, m] / s
                        m = m + 1
                    j = j + 1
                j = 0
                while j < Nt:
                    m = 0
                    while m < M:
                        Px[j, m] = (1.0 - damping) * Px_new[j, m] + damping * float(np.real(Px[j, m]))
                        m = m + 1
                    j = j + 1
                it = it + 1
            gamma = np.zeros((Nt, M))
            j = 0
            while j < Nt:
                m = 0
                while m < M:
                    s = 0.0
                    a = 0
                    while a < Nr:
                        s = s + Beta[a, j, m]
                        a = a + 1
                    gamma[j, m] = s
                    m = m + 1
                j = j + 1
            return gamma, Px
    """),
    "final_decision": textwrap.dedent("""\
        def final_decision(mu, constellation):
            x_hat = np.zeros(len(mu), dtype=complex)
            i = 0
            while i < len(mu):
                dists = np.abs(constellation - mu[i]) ** 2
                x_hat[i] = constellation[np.argmin(dists)]
                i = i + 1
            return x_hat
    """),
    "bp_final_decision": textwrap.dedent("""\
        def bp_final_decision(gamma, constellation):
            Nt = gamma.shape[0]
            x_hat = np.zeros(Nt, dtype=complex)
            i = 0
            while i < Nt:
                best_idx = _argmax_row(gamma, i)
                x_hat[i] = constellation[best_idx]
                i = i + 1
            return x_hat
    """),

    # ----- EP -----
    "cavity": textwrap.dedent("""\
        def cavity(t, h2, gamma_i, alpha_i):
            return t, h2
    """),
    "site_update": textwrap.dedent("""\
        def site_update(t, h2, constellation, gamma_i, alpha_i):
            diffs = constellation - t
            exponents = -np.abs(diffs) ** 2 / max(2.0 * h2, 1e-30)
            max_exp = float(np.max(np.real(exponents)))
            weights = np.exp(np.real(exponents) - max_exp)
            w_sum = max(float(np.sum(weights)), 1e-30)
            weights = weights / w_sum
            mu_p = np.sum(weights * constellation)
            sigma2_p = float(np.real(np.sum(weights * np.abs(constellation) ** 2))) - float(np.abs(mu_p) ** 2)
            sigma2_p = max(sigma2_p, 5e-7)
            new_alpha = 1.0 / sigma2_p - 1.0 / max(h2, 1e-30)
            new_gamma = mu_p / sigma2_p - t / max(h2, 1e-30)
            return new_alpha, new_gamma
    """),
    # ----- AMP -----
    "amp_iterate": textwrap.dedent("""\
        def amp_iterate(G, Gtilde, g_scale, gtilde, yMFtilde, sigma2, x_hat, tau_s, z, constellation):
            Nt = len(x_hat)
            M = len(constellation)
            tau_p = g_scale @ tau_s
            theta_s = 0.5
            theta_z = 0.5
            tau_p_safe = np.maximum(tau_p, 1e-30)
            tau_p_plus_s2 = tau_p_safe + sigma2
            tau_z = tau_p_plus_s2 * gtilde
            tau_z = np.maximum(tau_z, 1e-30)
            x_new = np.zeros(Nt, dtype=complex)
            x_var = np.zeros(Nt)
            i = 0
            while i < Nt:
                w = np.zeros(M)
                j = 0
                max_e = -1e30
                while j < M:
                    dist_j = float(np.abs(z[i] - constellation[j]) ** 2)
                    e = 0.0 - dist_j / max(float(tau_z[i]), 1e-30)
                    if e > max_e:
                        max_e = e
                    w[j] = e
                    j = j + 1
                j = 0
                ws = 0.0
                while j < M:
                    w[j] = np.exp(w[j] - max_e)
                    ws = ws + w[j]
                    j = j + 1
                ws = max(ws, 1e-30)
                j = 0
                while j < M:
                    w[j] = w[j] / ws
                    j = j + 1
                mu_i = 0.0 + 0.0j
                j = 0
                while j < M:
                    mu_i = mu_i + w[j] * constellation[j]
                    j = j + 1
                x_new[i] = mu_i
                v_i = 0.0
                j = 0
                while j < M:
                    v_i = v_i + w[j] * float(np.abs(constellation[j] - mu_i) ** 2)
                    j = j + 1
                x_var[i] = v_i
                i = i + 1
            shat_old = x_hat.copy()
            one_m_ts = 1.0 - theta_s
            tau_s_new = theta_s * x_var + one_m_ts * tau_s
            tau_p_new = g_scale @ tau_s_new
            tau_p_old_safe = np.maximum(tau_p, 1e-30)
            tp_denom = tau_p_old_safe + sigma2
            v_scale = tau_p_new / tp_denom
            z_diff = z - shat_old
            v = v_scale * z_diff
            one_m_tz = 1.0 - theta_z
            tp_new_plus_s2 = tau_p_new + sigma2
            tau_z_new = theta_z * tp_new_plus_s2 * gtilde + one_m_tz * tau_z
            z_new = yMFtilde + Gtilde @ x_new + v
            return x_new, tau_s_new, z_new
    """),
}


# ═══════════════════════════════════════════════════════════════════════════
# L3 Detector IR Templates
# ═══════════════════════════════════════════════════════════════════════════
# Parameters prefixed with ``slot_`` are slot function arguments.
# After compilation, call ops to these parameters are converted to AlgSlot.

LMMSE_TEMPLATE = textwrap.dedent("""\
    def lmmse(H, y, sigma2, constellation, slot_regularizer, slot_hard_decision):
        Nt = H.shape[1]
        G = H.conj().T @ H
        G_reg = slot_regularizer(G, sigma2)
        rhs = H.conj().T @ y
        x_soft = np.linalg.solve(G_reg, rhs)
        x_hat = slot_hard_decision(x_soft, constellation)
        return x_hat
""")

ZF_TEMPLATE = textwrap.dedent("""\
    def zf(H, y, sigma2, constellation, slot_hard_decision):
        G = H.conj().T @ H
        rhs = H.conj().T @ y
        x_soft = np.linalg.solve(G, rhs)
        x_hat = slot_hard_decision(x_soft, constellation)
        return x_hat
""")

OSIC_TEMPLATE = textwrap.dedent("""\
    def osic(H, y, sigma2, constellation, slot_ordering, slot_sic_step):
        Nr = H.shape[0]
        Nt = H.shape[1]
        H_cur = H.copy()
        y_cur = y.copy()
        order = slot_ordering(H_cur, y_cur, sigma2)
        detected = np.zeros(Nt, dtype=complex)
        col_map = list(range(Nt))
        step = 0
        while step < Nt:
            orig_idx = order[step]
            cur_idx = col_map.index(orig_idx)
            result = slot_sic_step(H_cur, y_cur, sigma2, cur_idx, constellation)
            detected[orig_idx] = result[0]
            H_cur = result[1]
            y_cur = result[2]
            col_map.pop(cur_idx)
            step = step + 1
        return detected
""")

KBEST_TEMPLATE = textwrap.dedent("""\
    def kbest(H, y, sigma2, constellation, slot_expand, slot_prune):
        Nr = H.shape[0]
        Nt = H.shape[1]
        Q = np.linalg.qr(H)[0]
        R = np.linalg.qr(H)[1]
        y_tilde = Q.conj().T @ y
        root = _make_tree_node(Nt - 1, [], 0.0)
        candidates = [root]
        level = Nt - 1
        while level >= 0:
            new_candidates = []
            ci = 0
            while ci < len(candidates):
                node = candidates[ci]
                node.level = level
                children = slot_expand(node, y_tilde, R, constellation)
                new_candidates = new_candidates + children
                ci = ci + 1
            candidates = slot_prune(new_candidates, 16)
            level = level - 1
        if len(candidates) == 0:
            return np.zeros(Nt, dtype=complex)
        best = candidates[0]
        bi = 1
        while bi < len(candidates):
            if candidates[bi].cost < best.cost:
                best = candidates[bi]
            bi = bi + 1
        return _reverse_syms(best.symbols, Nt)
""")

STACK_TEMPLATE = textwrap.dedent("""\
    def stack(H, y, sigma2, constellation, slot_node_select, slot_expand):
        Nr = H.shape[0]
        Nt = H.shape[1]
        Q = np.linalg.qr(H)[0]
        R = np.linalg.qr(H)[1]
        y_tilde = Q.conj().T @ y
        root = _make_tree_node(Nt - 1, [], 0.0)
        open_set = [root]
        nodes_expanded = 0
        max_nodes = 2000
        best_complete = root
        found = 0
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
                    best_complete = node
                    found = 1
                    _done = 1
                if _done == 0:
                    children = slot_expand(node, y_tilde, R, constellation)
                    open_set = open_set + children
        if found == 1:
            return _reverse_syms(best_complete.symbols, Nt)
        if len(open_set) > 0:
            best = open_set[0]
            i = 1
            while i < len(open_set):
                if open_set[i].cost < best.cost:
                    best = open_set[i]
                i = i + 1
            if len(best.symbols) == Nt:
                return _reverse_syms(best.symbols, Nt)
        return np.zeros(Nt, dtype=complex)
""")

BP_TEMPLATE = textwrap.dedent("""\
    def bp(H, y, sigma2, constellation, slot_bp_sweep, slot_final_decision):
        Nr = H.shape[0]
        Nt = H.shape[1]
        M = len(constellation)
        G = H.conj().T @ H + sigma2 * np.eye(Nt)
        x_mmse = np.linalg.solve(G, H.conj().T @ y)
        Px = np.zeros((Nt, M), dtype=complex)
        i = 0
        while i < Nt:
            dists = np.abs(constellation - x_mmse[i]) ** 2
            min_d = np.min(dists)
            j = 0
            while j < M:
                if float(np.real(dists[j] - min_d)) < 1e-10:
                    Px[i, j] = 1.0
                j = j + 1
            _row_normalize(Px, i)
            i = i + 1
        result = slot_bp_sweep(H, y, sigma2, Px, constellation, 8)
        gamma = result[0]
        x_hat = slot_final_decision(gamma, constellation)
        return x_hat
""")

EP_TEMPLATE = textwrap.dedent("""\
    def ep(H, y, sigma2, constellation, slot_cavity, slot_site_update, slot_final_decision):
        Nr = H.shape[0]
        Nt = H.shape[1]
        alpha = np.ones(Nt) * 2.0
        gamma_ep = np.zeros(Nt, dtype=complex)
        HtH = H.conj().T @ H
        Hty = H.conj().T @ y
        s2inv = 1.0 / max(sigma2, 1e-30)
        Hty_scaled = Hty * s2inv
        HtH_scaled = HtH * s2inv
        rhs = Hty_scaled + gamma_ep
        Sigma_q = np.linalg.inv(HtH_scaled + np.diag(alpha))
        mu_q = Sigma_q @ rhs
        damping = 0.5
        one_m_damp = 1.0 - damping
        it = 0
        while it < 20:
            sig = np.real(np.diag(Sigma_q))
            i = 0
            while i < Nt:
                sig_i = float(sig[i])
                alpha_i = float(alpha[i])
                denom = 1.0 - sig_i * alpha_i
                denom = max(denom, 1e-30)
                h2 = sig_i / denom
                h2 = max(h2, 1e-30)
                sig_inv = 1.0 / max(sig_i, 1e-30)
                ratio = mu_q[i] * sig_inv - gamma_ep[i]
                t = h2 * ratio
                cav_result = slot_cavity(t, h2, gamma_ep[i], alpha_i)
                upd_result = slot_site_update(cav_result[0], cav_result[1], constellation, gamma_ep[i], alpha_i)
                new_alpha = float(np.real(upd_result[0]))
                new_gamma = upd_result[1]
                if new_alpha > 1e-6:
                    alpha[i] = damping * new_alpha + one_m_damp * alpha_i
                    gamma_ep[i] = damping * new_gamma + one_m_damp * gamma_ep[i]
                i = i + 1
            rhs = Hty_scaled + gamma_ep
            Sigma_q = np.linalg.inv(HtH_scaled + np.diag(alpha))
            mu_q = Sigma_q @ rhs
            it = it + 1
        x_hat = slot_final_decision(mu_q, constellation)
        return x_hat
""")

AMP_TEMPLATE = textwrap.dedent("""\
    def amp(H, y, sigma2, constellation, slot_amp_iterate, slot_final_decision):
        Nr = H.shape[0]
        Nt = H.shape[1]
        G = H.conj().T @ H
        g_diag = np.real(np.diag(G))
        g_diag = np.maximum(g_diag, 1e-30)
        Gtilde = np.eye(Nt) - np.diag(1.0 / g_diag) @ G
        g_scale = g_diag / max(float(Nr), 1.0)
        gtilde = 1.0 / g_diag
        yMF = H.conj().T @ y
        yMFtilde = gtilde * yMF
        x_hat = np.zeros(Nt, dtype=complex)
        tau_s = np.ones(Nt)
        z = yMFtilde.copy()
        it = 0
        while it < 20:
            result = slot_amp_iterate(G, Gtilde, g_scale, gtilde, yMFtilde, sigma2, x_hat, tau_s, z, constellation)
            x_hat = result[0]
            tau_s = result[1]
            z = result[2]
            it = it + 1
        x_out = slot_final_decision(z, constellation)
        return x_out
""")


# ═══════════════════════════════════════════════════════════════════════════
# Mapping: template name → (source, slot_names, slot_definitions_fn)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class _DetectorSpec:
    """Specification for one L3 detector template."""
    algo_id: str
    source: str
    func_name: str
    slot_arg_names: list[str]           # slot_regularizer, slot_hard_decision, ...
    slot_defs_fn: Any                   # callable returning dict[str, SlotDescriptor]
    slot_default_keys: dict[str, str]   # slot_arg_name → key in SLOT_DEFAULTS
    tags: set[str]
    extra_globals: dict[str, Any] | None = None


_DETECTOR_SPECS: list[_DetectorSpec] = [
    _DetectorSpec(
        algo_id="lmmse",
        source=LMMSE_TEMPLATE,
        func_name="lmmse",
        slot_arg_names=["slot_regularizer", "slot_hard_decision"],
        slot_defs_fn=_lmmse_slots,
        slot_default_keys={
            "slot_regularizer": "regularizer",
            "slot_hard_decision": "hard_decision",
        },
        tags={"original", "linear"},
    ),
    _DetectorSpec(
        algo_id="zf",
        source=ZF_TEMPLATE,
        func_name="zf",
        slot_arg_names=["slot_hard_decision"],
        slot_defs_fn=_zf_slots,
        slot_default_keys={"slot_hard_decision": "hard_decision"},
        tags={"original", "linear"},
    ),
    _DetectorSpec(
        algo_id="osic",
        source=OSIC_TEMPLATE,
        func_name="osic",
        slot_arg_names=["slot_ordering", "slot_sic_step"],
        slot_defs_fn=_osic_slots,
        slot_default_keys={
            "slot_ordering": "ordering",
            "slot_sic_step": "sic_step",
        },
        tags={"original", "sic"},
    ),
    _DetectorSpec(
        algo_id="kbest",
        source=KBEST_TEMPLATE,
        func_name="kbest",
        slot_arg_names=["slot_expand", "slot_prune"],
        slot_defs_fn=_kbest_slots,
        slot_default_keys={
            "slot_expand": "expand",
            "slot_prune": "prune",
        },
        tags={"original", "tree_search"},
        extra_globals={"TreeNode": None},  # filled at build time
    ),
    _DetectorSpec(
        algo_id="stack",
        source=STACK_TEMPLATE,
        func_name="stack",
        slot_arg_names=["slot_node_select", "slot_expand"],
        slot_defs_fn=_stack_slots,
        slot_default_keys={
            "slot_node_select": "node_select",
            "slot_expand": "expand",
        },
        tags={"original", "tree_search"},
        extra_globals={"TreeNode": None},
    ),
    _DetectorSpec(
        algo_id="bp",
        source=BP_TEMPLATE,
        func_name="bp",
        slot_arg_names=["slot_bp_sweep", "slot_final_decision"],
        slot_defs_fn=_bp_slots,
        slot_default_keys={
            "slot_bp_sweep": "bp_sweep",
            "slot_final_decision": "bp_final_decision",
        },
        tags={"original", "message_passing"},
    ),
    _DetectorSpec(
        algo_id="ep",
        source=EP_TEMPLATE,
        func_name="ep",
        slot_arg_names=["slot_cavity", "slot_site_update", "slot_final_decision"],
        slot_defs_fn=_ep_slots,
        slot_default_keys={
            "slot_cavity": "cavity",
            "slot_site_update": "site_update",
            "slot_final_decision": "final_decision",
        },
        tags={"original", "inference"},
    ),
    _DetectorSpec(
        algo_id="amp",
        source=AMP_TEMPLATE,
        func_name="amp",
        slot_arg_names=["slot_amp_iterate", "slot_final_decision"],
        slot_defs_fn=_amp_slots,
        slot_default_keys={
            "slot_amp_iterate": "amp_iterate",
            "slot_final_decision": "final_decision",
        },
        tags={"original", "inference"},
    ),
]

# ── Extend with skeleton library ──────────────────────────────────────────
SLOT_DEFAULTS.update(EXTENDED_SLOT_DEFAULTS)

for _skel in get_extended_specs():
    _slot_defs_dict = _skel.slot_defs
    _DETECTOR_SPECS.append(_DetectorSpec(
        algo_id=_skel.algo_id,
        source=_skel.source,
        func_name=_skel.func_name,
        slot_arg_names=_skel.slot_arg_names,
        slot_defs_fn=lambda _d=_slot_defs_dict: _d,
        slot_default_keys=_skel.slot_default_keys,
        tags=_skel.tags,
        extra_globals=_skel.extra_globals,
    ))


# ═══════════════════════════════════════════════════════════════════════════
# IR manipulation: xDSL-level helpers (from grafting/rewriter patterns)
# ═══════════════════════════════════════════════════════════════════════════

def _next_prefixed_id(existing: dict[str, Any], prefix: str) -> str:
    """Allocate the next sequential ID with the given prefix."""
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    max_id = -1
    for key in existing:
        m = pattern.match(key)
        if m:
            max_id = max(max_id, int(m.group(1)))
    return f"{prefix}{max_id + 1}"


def _value_payload(value: Value) -> dict[str, Any]:
    return {
        "id": value.id,
        "name_hint": value.name_hint,
        "type_hint": value.type_hint,
        "source_span": value.source_span,
        "attrs": dict(value.attrs),
    }


def _append_xdsl_op(
    func_ir: FunctionIR,
    xdsl_ops: list[Any],
    *,
    opcode: str,
    inputs: list[str],
    outputs: list[str],
    block_id: str,
    attrs: dict[str, Any],
) -> str:
    """Create an op in func_ir and its xDSL representation.

    Returns the new op_id.
    """
    op_id = _next_prefixed_id(func_ir.ops, "op_")
    temp_op = Op(
        id=op_id,
        opcode=opcode,
        inputs=list(inputs),
        outputs=list(outputs),
        block_id=block_id,
        source_span=None,
        attrs=dict(attrs),
    )
    func_ir.ops[op_id] = temp_op
    payload = {
        "id": op_id,
        "opcode": opcode,
        "block_id": block_id,
        "inputs": list(inputs),
        "outputs": list(outputs),
        "source_span": None,
        "attrs": dict(attrs),
        "output_meta": [_value_payload(func_ir.values[vid]) for vid in outputs],
    }
    xdsl_op = create_xdsl_op_from_payload(
        opcode=opcode,
        payload=payload,
        result_type_hints=[func_ir.values[vid].type_hint for vid in outputs],
    )
    func_ir.xdsl_op_map[op_id] = xdsl_op
    xdsl_ops.append(xdsl_op)
    return op_id


def _new_value(func_ir: FunctionIR, name_hint: str, type_hint: str,
               attrs: dict[str, Any] | None = None) -> str:
    """Allocate a new Value in func_ir.  Returns the value_id."""
    value_id = _next_prefixed_id(func_ir.values, "v_")
    func_ir.values[value_id] = Value(
        id=value_id,
        name_hint=name_hint,
        type_hint=type_hint,
        source_span=None,
        attrs=dict(attrs or {}),
    )
    return value_id


# ═══════════════════════════════════════════════════════════════════════════
# Core transform: convert slot_* call ops → AlgSlot ops
# ═══════════════════════════════════════════════════════════════════════════

def convert_slot_calls_to_algslot(
    func_ir: FunctionIR,
    slot_arg_names: list[str],
) -> FunctionIR:
    """Replace ``call`` ops that invoke ``slot_*`` parameters with ``AlgSlot`` ops.

    Steps:
    1. Identify argument value IDs for each ``slot_*`` parameter
    2. Find all ``call`` ops whose callee is one of these slot args
    3. For each such call:
       a. Create an ``AlgSlot`` op at the same position (same block)
       b. slot_inputs = the call's argument values (excluding callee)
       c. slot output = the call's output
       d. Remove the original call op
    4. Remove slot_* args from the function's arg_values
    5. Rebuild from xDSL

    Returns a new FunctionIR with ``AlgSlot`` ops and no slot_* args.
    """
    func_ir = deepcopy(func_ir)

    # --- Step 1: Map slot arg names → arg value IDs ---
    slot_arg_value_ids: dict[str, str] = {}  # arg_name → value_id
    for vid in func_ir.arg_values:
        v = func_ir.values[vid]
        var_name = v.attrs.get("var_name") or v.name_hint
        if var_name in slot_arg_names:
            slot_arg_value_ids[var_name] = vid

    if not slot_arg_value_ids:
        return func_ir  # no slots to convert

    slot_value_ids = set(slot_arg_value_ids.values())
    # Also build a name→arg_name map for matching versioned copies of slot args
    _slot_name_to_arg: dict[str, str] = {}
    for arg_name in slot_arg_names:
        _slot_name_to_arg[arg_name] = arg_name

    # --- Step 2-3: Find call ops → create AlgSlot, collect ops to remove ---
    call_ops_to_remove: set[str] = set()
    new_xdsl_ops_per_block: dict[str, list] = {}  # block_id → list[(anchor_op_id, [xdsl_ops])]

    for op_id, op in list(func_ir.ops.items()):
        if op.opcode != "call":
            continue
        # callee is inputs[0], rest are arguments
        callee_vid = op.inputs[0]

        # Match by direct value ID or by var_name attribute
        slot_arg_name = None
        if callee_vid in slot_value_ids:
            for name, vid in slot_arg_value_ids.items():
                if vid == callee_vid:
                    slot_arg_name = name
                    break
        else:
            # Check if callee is a versioned copy of a slot arg
            callee_val = func_ir.values.get(callee_vid)
            if callee_val:
                var_name = callee_val.attrs.get("var_name") or ""
                if var_name in _slot_name_to_arg:
                    slot_arg_name = _slot_name_to_arg[var_name]

        if slot_arg_name is None:
            continue

        # slot_id: strip "slot_" prefix
        slot_id = slot_arg_name[5:] if slot_arg_name.startswith("slot_") else slot_arg_name

        # Actual arguments (everything after callee)
        call_args = op.inputs[1:]
        call_output = op.outputs[0] if op.outputs else None

        if call_output is None:
            continue

        # Create AlgSlot op
        xdsl_ops: list = []
        _append_xdsl_op(
            func_ir, xdsl_ops,
            opcode="slot",
            inputs=call_args,
            outputs=[call_output],
            block_id=op.block_id,
            attrs={"slot_id": slot_id, "slot_kind": slot_id},
        )

        call_ops_to_remove.add(op_id)

        if op.block_id not in new_xdsl_ops_per_block:
            new_xdsl_ops_per_block[op.block_id] = []
        new_xdsl_ops_per_block[op.block_id].append((op_id, xdsl_ops))

    # --- Step 3: Insert new ops and remove old ---
    for block_id, insertions in new_xdsl_ops_per_block.items():
        xdsl_block = func_ir.xdsl_block_map[block_id]
        for anchor_op_id, xdsl_ops in insertions:
            anchor_xdsl = func_ir.xdsl_op_map.get(anchor_op_id)
            if anchor_xdsl is not None:
                xdsl_block.insert_ops_before(xdsl_ops, anchor_xdsl)

    # Remove old call ops from xDSL
    for op_id in call_ops_to_remove:
        xdsl_op = func_ir.xdsl_op_map.get(op_id)
        if xdsl_op is None:
            continue
        parent = xdsl_op.parent_block()
        if parent is not None:
            parent.erase_op(xdsl_op, safe_erase=False)

    # --- Step 4: Remove slot arg values from arg_values ---
    func_ir.arg_values = [v for v in func_ir.arg_values if v not in slot_value_ids]

    # --- Step 5: Rebuild ---
    try:
        rebuilt = FunctionIR.from_xdsl(func_ir.xdsl_module)
        # Re-apply slot arg removal by name (from_xdsl creates new value IDs)
        slot_names_set = set(slot_arg_names)
        new_arg_values = []
        for vid in rebuilt.arg_values:
            v = rebuilt.values[vid]
            var_name = v.attrs.get("var_name") or v.name_hint
            if var_name not in slot_names_set:
                new_arg_values.append(vid)
        rebuilt.arg_values = new_arg_values
    except Exception:
        # Fallback: return the partially modified IR
        # (from_xdsl may fail if xDSL module is inconsistent;
        #  in that case we keep the dict-based representation which is still valid)
        rebuilt = func_ir

    return rebuilt


# ═══════════════════════════════════════════════════════════════════════════
# Compile a detector template → FunctionIR with AlgSlot
# ═══════════════════════════════════════════════════════════════════════════

def compile_detector_template(spec: _DetectorSpec) -> FunctionIR:
    """Compile a detector template source to FunctionIR, then convert
    slot_* call ops to AlgSlot ops.

    Returns FunctionIR with AlgSlot ops and no slot_* parameters.
    """
    g = _template_globals()
    if spec.extra_globals:
        from evolution.pool_ops_l2 import TreeNode
        g["TreeNode"] = TreeNode
    raw_ir = compile_source_to_ir(spec.source, spec.func_name, g)
    return convert_slot_calls_to_algslot(raw_ir, spec.slot_arg_names)


def compile_slot_default(key: str) -> FunctionIR | None:
    """Try to compile a slot default implementation source to FunctionIR.

    Returns FunctionIR if successful, None if the source uses features
    not supported by the IR builder (keyword args, list comprehensions, etc.).
    """
    source = SLOT_DEFAULTS[key]
    import ast
    tree = ast.parse(source)
    func_name = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            break
    g = _template_globals()
    from evolution.pool_ops_l2 import TreeNode
    g["TreeNode"] = TreeNode
    try:
        return compile_source_to_ir(source, func_name, g)
    except (NotImplementedError, Exception):
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Query helpers on compiled IR
# ═══════════════════════════════════════════════════════════════════════════

def find_algslot_ops(func_ir: FunctionIR) -> list[Op]:
    """Return all AlgSlot ops in the IR."""
    return [op for op in func_ir.ops.values() if op.opcode == "slot"]


def get_slot_ids(func_ir: FunctionIR) -> list[str]:
    """Return all slot_id values from AlgSlot ops."""
    return [op.attrs["slot_id"] for op in func_ir.ops.values()
            if op.opcode == "slot"]


# ═══════════════════════════════════════════════════════════════════════════
# Build the full IR pool
# ═══════════════════════════════════════════════════════════════════════════

def build_ir_pool(
    rng: np.random.Generator | None = None,
    n_random_variants: int = 7,
) -> list[AlgorithmGenome]:
    """Build the initial IR-based algorithm pool.

    For each L3 detector:
      1. Compile template → FunctionIR with AlgSlot ops
      2. Create SlotDescriptor tree (reused from algorithm_pool.py)
      3. For each slot, build a SlotPopulation:
         - variant 0 = default implementation (compiled to FunctionIR)
         - variants 1..N = random FunctionIR programs matching the spec

    Returns a list of AlgorithmGenome objects.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    pool: list[AlgorithmGenome] = []

    for spec in _DETECTOR_SPECS:
        # 1. Compile template (skip on failure — some templates use
        #    unsupported AST nodes like Slice)
        try:
            structural_ir = compile_detector_template(spec)
        except Exception:
            continue

        # 2. Get slot definitions
        slot_tree = spec.slot_defs_fn()

        # 3. Build SlotPopulation for each TOP-LEVEL slot
        #    (only slots that appear as AlgSlot in the IR)
        ir_slot_ids = set(get_slot_ids(structural_ir))
        populations: dict[str, SlotPopulation] = {}

        # Build a lookup: slot_arg_name (minus "slot_" prefix) → slot_arg_name
        # e.g. "bp_sweep" → "slot_bp_sweep"
        arg_name_to_ir_slot = {}
        for arg_name in spec.slot_arg_names:
            ir_sid = arg_name.removeprefix("slot_")
            arg_name_to_ir_slot[arg_name] = ir_sid

        for slot_id, desc in slot_tree.items():
            # Match descriptor to IR slot_id.
            # The IR slot_id = slot_arg_name minus "slot_" prefix (e.g. "bp_sweep")
            # The descriptor short_name might be a suffix (e.g. "sweep").
            # We match via the slot_default_keys mapping: arg_name → default_key.
            matched_ir_sid = None
            matched_arg_name = None

            # Strategy 1: direct short_name match
            if desc.short_name in ir_slot_ids:
                matched_ir_sid = desc.short_name
            # Strategy 2: exact slot_id match
            elif slot_id in ir_slot_ids:
                matched_ir_sid = slot_id
            else:
                # Strategy 3: find via slot_default_keys mapping
                for arg_name, dk in spec.slot_default_keys.items():
                    ir_sid = arg_name.removeprefix("slot_")
                    if ir_sid in ir_slot_ids:
                        # Check if this arg_name's default_key matches desc
                        if dk == desc.short_name or slot_id.endswith(f".{dk}"):
                            matched_ir_sid = ir_sid
                            matched_arg_name = arg_name
                            break
                        # Also check: the arg_name itself ends with desc.short_name
                        if ir_sid.endswith(desc.short_name) or ir_sid.endswith(f"_{desc.short_name}"):
                            matched_ir_sid = ir_sid
                            matched_arg_name = arg_name
                            break

            if matched_ir_sid is None:
                continue

            # Determine the key for the default implementation
            default_key = None
            for arg_name, dk in spec.slot_default_keys.items():
                ir_sid = arg_name.removeprefix("slot_")
                # Match via IR slot_id or short_name
                if ir_sid == matched_ir_sid:
                    default_key = dk
                    break
                if arg_name == f"slot_{desc.short_name}":
                    default_key = dk
                    break

            if default_key is None:
                continue

            # Compile default implementation (may return None)
            default_ir = compile_slot_default(default_key)
            default_source = SLOT_DEFAULTS.get(default_key)

            # Generate random variants
            variants: list[FunctionIR | None] = [default_ir]
            source_variants: list[str | None] = [default_source]
            fitness_vals = [float("inf")]
            for _ in range(n_random_variants):
                try:
                    v = random_ir_program(desc.spec, rng, max_depth=4)
                    variants.append(v)
                    source_variants.append(None)
                    fitness_vals.append(float("inf"))
                except Exception:
                    pass

            populations[slot_id] = SlotPopulation(
                slot_id=slot_id,
                spec=desc.spec,
                variants=variants,
                fitness=fitness_vals,
                best_idx=0,
                source_variants=source_variants,
            )

        genome = AlgorithmGenome(
            algo_id=spec.algo_id,
            ir=structural_ir,  # provisional: replaced by flat annotated IR below
            slot_populations=populations,
            constants=np.zeros(0, dtype=np.float64),
            generation=0,
            parent_ids=[],
            graft_history=[],
            tags=spec.tags | {"original"},
            metadata={
                "level": 3,
                "slot_tree": slot_tree,
                "detector_name": spec.func_name,
            },
        )

        # Single-IR refactor: collapse the (template-with-slot-ops +
        # slot_populations[best]) view into ONE flat annotated IR. After
        # this point ``genome.ir`` is the canonical IR — slot internals
        # are first-class ops, and per-op ``_provenance`` annotations
        # carry slot affiliation. Any failure to inline keeps the raw
        # template IR (purely structural, no annotations).
        try:
            from evolution.fii import build_flat_annotated_ir
            flat_ir = build_flat_annotated_ir(genome)
        except Exception:
            flat_ir = None
        if flat_ir is not None:
            genome.ir = flat_ir
        pool.append(genome)

    return pool
