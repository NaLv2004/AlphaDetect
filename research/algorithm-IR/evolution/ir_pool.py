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

import textwrap
from dataclasses import dataclass
from typing import Any

import numpy as np

from algorithm_ir.frontend.ir_builder import compile_source_to_ir
from algorithm_ir.ir.model import FunctionIR, Op

from evolution.pool_types import (
    AlgorithmGenome,
    SlotPopulation,
)
from evolution.algorithm_pool import (
    _lmmse_slots, _zf_slots, _osic_slots, _kbest_slots,
    _bp_slots, _ep_slots, _amp_slots, _stack_slots,
)


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
    from algorithm_ir.frontend.slot_dsl import slot

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
        "slot": slot,
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

# ═══════════════════════════════════════════════════════════════════════════
# L3 Detector IR Templates (annotation-only ``with slot(...)`` form)
# ═══════════════════════════════════════════════════════════════════════════
# Each template is a self-contained Python function. Slot regions are
# delimited by ``with slot("<algo_id>.<short_name>", inputs=(...), outputs=(...)):``
# blocks. The IR builder recognises these context managers, tags every op
# inside with the slot's pop_key, and records the slot in
# ``FunctionIR.slot_meta``. At runtime the ``slot`` callable is an inert
# context manager (see ``algorithm_ir.frontend.slot_dsl``) so the same
# source can be exec'd directly for evaluation.

LMMSE_TEMPLATE = textwrap.dedent("""\
    def lmmse(H, y, sigma2, constellation):
        Nt = H.shape[1]
        G = H.conj().T @ H
        rhs = H.conj().T @ y
        with slot("lmmse.regularizer", inputs=(G, sigma2), outputs=("G_reg",)):
            n_reg = G.shape[0]
            G_reg = G + sigma2 * np.eye(n_reg)
        x_soft = np.linalg.solve(G_reg, rhs)
        with slot("lmmse.hard_decision", inputs=(x_soft, constellation), outputs=("x_hat",)):
            x_hat = np.zeros(len(x_soft), dtype=complex)
            i_hd = 0
            while i_hd < len(x_soft):
                d_hd = np.abs(constellation - x_soft[i_hd]) ** 2
                x_hat[i_hd] = constellation[np.argmin(d_hd)]
                i_hd = i_hd + 1
        return x_hat
""")

ZF_TEMPLATE = textwrap.dedent("""\
    def zf(H, y, sigma2, constellation):
        G = H.conj().T @ H
        rhs = H.conj().T @ y
        x_soft = np.linalg.solve(G, rhs)
        with slot("zf.hard_decision", inputs=(x_soft, constellation), outputs=("x_hat",)):
            x_hat = np.zeros(len(x_soft), dtype=complex)
            i_hd = 0
            while i_hd < len(x_soft):
                d_hd = np.abs(constellation - x_soft[i_hd]) ** 2
                x_hat[i_hd] = constellation[np.argmin(d_hd)]
                i_hd = i_hd + 1
        return x_hat
""")

OSIC_TEMPLATE = textwrap.dedent("""\
    def osic(H, y, sigma2, constellation):
        Nr = H.shape[0]
        Nt = H.shape[1]
        H_cur = H.copy()
        y_cur = y.copy()
        with slot("osic.ordering", inputs=(H_cur, y_cur, sigma2), outputs=("order",)):
            Nt_o = H_cur.shape[1]
            G_o = H_cur.conj().T @ H_cur + sigma2 * np.eye(Nt_o)
            G_inv_o = np.linalg.inv(G_o)
            snr_o = np.zeros(Nt_o, dtype=complex)
            i_o = 0
            while i_o < Nt_o:
                snr_o[i_o] = 1.0 / max(float(np.real(G_inv_o[i_o, i_o])), 1e-30) - sigma2
                i_o = i_o + 1
            order = []
            used_o = np.zeros(Nt_o, dtype=complex)
            k_o = 0
            while k_o < Nt_o:
                best_o = -1
                bv_o = -1e30
                j_o = 0
                while j_o < Nt_o:
                    if float(np.real(used_o[j_o])) == 0.0:
                        if float(np.real(snr_o[j_o])) > bv_o:
                            bv_o = float(np.real(snr_o[j_o]))
                            best_o = j_o
                    j_o = j_o + 1
                order.append(best_o)
                used_o[best_o] = 1.0
                k_o = k_o + 1
        detected = np.zeros(Nt, dtype=complex)
        col_map = list(range(Nt))
        step = 0
        while step < Nt:
            orig_idx = order[step]
            cur_idx = col_map.index(orig_idx)
            with slot("osic.sic_step",
                      inputs=(H_cur, y_cur, sigma2, cur_idx, constellation),
                      outputs=("x_step", "H_step", "y_step")):
                Nt_s = H_cur.shape[1]
                G_s = H_cur.conj().T @ H_cur + sigma2 * np.eye(Nt_s)
                w_s = np.linalg.solve(G_s, H_cur.conj().T @ y_cur)
                x_est_s = w_s[cur_idx]
                d_s = np.abs(constellation - x_est_s) ** 2
                x_step = constellation[np.argmin(d_s)]
                y_step = y_cur - _col(H_cur, cur_idx) * x_step
                H_step = np.delete(H_cur, cur_idx, axis=1)
            detected[orig_idx] = x_step
            H_cur = H_step
            y_cur = y_step
            col_map.pop(cur_idx)
            step = step + 1
        return detected
""")

KBEST_TEMPLATE = textwrap.dedent("""\
    def kbest(H, y, sigma2, constellation):
        Nr = H.shape[0]
        Nt = H.shape[1]
        Q = np.linalg.qr(H)[0]
        R = np.linalg.qr(H)[1]
        y_tilde = Q.conj().T @ y
        root = _make_tree_node(Nt - 1, [], 0.0)
        candidates = [root]
        level = Nt - 1
        K_keep = 16
        while level >= 0:
            new_candidates = []
            ci = 0
            while ci < len(candidates):
                node = candidates[ci]
                node.level = level
                with slot("kbest.expand",
                          inputs=(node, y_tilde, R, constellation),
                          outputs=("children",)):
                    Nt_e = R.shape[1]
                    children = []
                    interf_e = 0.0 + 0.0j
                    jr_e = 0
                    while jr_e < len(node.symbols):
                        j_e = Nt_e - 1 - jr_e
                        interf_e = interf_e + R[level, j_e] * node.symbols[jr_e]
                        jr_e = jr_e + 1
                    cei = 0
                    while cei < len(constellation):
                        sym_e = constellation[cei]
                        resid_e = y_tilde[level] - R[level, level] * sym_e - interf_e
                        lc_e = float(np.abs(resid_e) ** 2)
                        tot_e = node.cost + lc_e
                        ch_e = _make_tree_node(level - 1, node.symbols + [sym_e], tot_e)
                        children.append(ch_e)
                        cei = cei + 1
                new_candidates = new_candidates + children
                ci = ci + 1
            with slot("kbest.prune",
                      inputs=(new_candidates, K_keep),
                      outputs=("candidates",)):
                n_p = len(new_candidates)
                ip = 0
                while ip < n_p:
                    jp = ip + 1
                    while jp < n_p:
                        if new_candidates[jp].cost < new_candidates[ip].cost:
                            tmp_p = new_candidates[ip]
                            new_candidates[ip] = new_candidates[jp]
                            new_candidates[jp] = tmp_p
                        jp = jp + 1
                    ip = ip + 1
                candidates = new_candidates[:K_keep]
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
    def stack(H, y, sigma2, constellation):
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
                with slot("stack.node_select",
                          inputs=(open_set,),
                          outputs=("best_idx",)):
                    best_idx = 0
                    bc_ns = open_set[0].cost
                    i_ns = 1
                    while i_ns < len(open_set):
                        if open_set[i_ns].cost < bc_ns:
                            bc_ns = open_set[i_ns].cost
                            best_idx = i_ns
                        i_ns = i_ns + 1
                node = open_set.pop(best_idx)
                nodes_expanded = nodes_expanded + 1
                if len(node.symbols) == Nt:
                    best_complete = node
                    found = 1
                    _done = 1
                if _done == 0:
                    level_x = node.level
                    with slot("stack.expand",
                              inputs=(node, y_tilde, R, constellation),
                              outputs=("children",)):
                        Nt_e = R.shape[1]
                        children = []
                        interf_e = 0.0 + 0.0j
                        jr_e = 0
                        while jr_e < len(node.symbols):
                            j_e = Nt_e - 1 - jr_e
                            interf_e = interf_e + R[level_x, j_e] * node.symbols[jr_e]
                            jr_e = jr_e + 1
                        cei = 0
                        while cei < len(constellation):
                            sym_e = constellation[cei]
                            resid_e = y_tilde[level_x] - R[level_x, level_x] * sym_e - interf_e
                            lc_e = float(np.abs(resid_e) ** 2)
                            tot_e = node.cost + lc_e
                            ch_e = _make_tree_node(level_x - 1, node.symbols + [sym_e], tot_e)
                            children.append(ch_e)
                            cei = cei + 1
                    open_set = open_set + children
        if found == 1:
            return _reverse_syms(best_complete.symbols, Nt)
        if len(open_set) > 0:
            best = open_set[0]
            i_b = 1
            while i_b < len(open_set):
                if open_set[i_b].cost < best.cost:
                    best = open_set[i_b]
                i_b = i_b + 1
            if len(best.symbols) == Nt:
                return _reverse_syms(best.symbols, Nt)
        return np.zeros(Nt, dtype=complex)
""")

BP_TEMPLATE = textwrap.dedent("""\
    def bp(H, y, sigma2, constellation):
        Nr = H.shape[0]
        Nt = H.shape[1]
        M = len(constellation)
        G_bp = H.conj().T @ H + sigma2 * np.eye(Nt)
        x_mmse = np.linalg.solve(G_bp, H.conj().T @ y)
        Px = np.zeros((Nt, M), dtype=complex)
        i_init = 0
        while i_init < Nt:
            d_init = np.abs(constellation - x_mmse[i_init]) ** 2
            min_d = np.min(d_init)
            j_init = 0
            while j_init < M:
                if float(np.real(d_init[j_init] - min_d)) < 1e-10:
                    Px[i_init, j_init] = 1.0
                j_init = j_init + 1
            _row_normalize(Px, i_init)
            i_init = i_init + 1
        max_iters = 8
        with slot("bp.bp_sweep",
                  inputs=(H, y, sigma2, Px, constellation, max_iters),
                  outputs=("gamma", "Px_out")):
            Nr_b = H.shape[0]
            Nt_b = H.shape[1]
            M_b = len(constellation)
            damp_b = 0.5
            it_b = 0
            Beta = np.zeros((Nr_b, Nt_b, M_b))
            while it_b < max_iters:
                Muz = np.zeros((Nr_b, Nt_b), dtype=complex)
                sigmaz2 = np.zeros((Nr_b, Nt_b))
                a_b = 0
                while a_b < Nr_b:
                    j_b = 0
                    while j_b < Nt_b:
                        mu_j = 0.0 + 0.0j
                        var_j = 0.0
                        m_b = 0
                        while m_b < M_b:
                            mu_j = mu_j + Px[j_b, m_b] * constellation[m_b]
                            m_b = m_b + 1
                        m_b = 0
                        while m_b < M_b:
                            var_j = var_j + float(np.real(Px[j_b, m_b])) * float(np.abs(constellation[m_b] - mu_j) ** 2)
                            m_b = m_b + 1
                        Muz[a_b, j_b] = H[a_b, j_b] * mu_j
                        sigmaz2[a_b, j_b] = float(np.abs(H[a_b, j_b]) ** 2) * var_j
                        j_b = j_b + 1
                    a_b = a_b + 1
                Beta = np.zeros((Nr_b, Nt_b, M_b))
                a_b = 0
                while a_b < Nr_b:
                    j_b = 0
                    while j_b < Nt_b:
                        mu_sum = 0.0 + 0.0j
                        var_sum = sigma2
                        k_b = 0
                        while k_b < Nt_b:
                            if k_b != j_b:
                                mu_sum = mu_sum + Muz[a_b, k_b]
                                var_sum = var_sum + sigmaz2[a_b, k_b]
                            k_b = k_b + 1
                        var_sum = max(var_sum, 1e-30)
                        resid_b = y[a_b] - mu_sum
                        m_b = 0
                        while m_b < M_b:
                            bv = float(np.real(2.0 * np.conj(resid_b) * H[a_b, j_b] * (constellation[m_b] - constellation[0]) - np.abs(H[a_b, j_b]) ** 2 * (np.abs(constellation[m_b]) ** 2 - np.abs(constellation[0]) ** 2)))
                            Beta[a_b, j_b, m_b] = bv / (2.0 * var_sum)
                            m_b = m_b + 1
                        j_b = j_b + 1
                    a_b = a_b + 1
                Alpha = np.zeros((Nt_b, M_b))
                j_b = 0
                while j_b < Nt_b:
                    m_b = 0
                    while m_b < M_b:
                        s_a = 0.0
                        a_b = 0
                        while a_b < Nr_b:
                            s_a = s_a + Beta[a_b, j_b, m_b]
                            a_b = a_b + 1
                        Alpha[j_b, m_b] = s_a
                        m_b = m_b + 1
                    j_b = j_b + 1
                Px_new = np.zeros((Nt_b, M_b))
                j_b = 0
                while j_b < Nt_b:
                    max_a = -1e30
                    m_b = 0
                    while m_b < M_b:
                        if Alpha[j_b, m_b] > max_a:
                            max_a = Alpha[j_b, m_b]
                        m_b = m_b + 1
                    m_b = 0
                    s_p = 0.0
                    while m_b < M_b:
                        Px_new[j_b, m_b] = np.exp(Alpha[j_b, m_b] - max_a)
                        s_p = s_p + Px_new[j_b, m_b]
                        m_b = m_b + 1
                    s_p = max(s_p, 1e-30)
                    m_b = 0
                    while m_b < M_b:
                        Px_new[j_b, m_b] = Px_new[j_b, m_b] / s_p
                        m_b = m_b + 1
                    j_b = j_b + 1
                j_b = 0
                while j_b < Nt_b:
                    m_b = 0
                    while m_b < M_b:
                        Px[j_b, m_b] = (1.0 - damp_b) * Px_new[j_b, m_b] + damp_b * float(np.real(Px[j_b, m_b]))
                        m_b = m_b + 1
                    j_b = j_b + 1
                it_b = it_b + 1
            gamma = np.zeros((Nt_b, M_b))
            j_b = 0
            while j_b < Nt_b:
                m_b = 0
                while m_b < M_b:
                    s_g = 0.0
                    a_b = 0
                    while a_b < Nr_b:
                        s_g = s_g + Beta[a_b, j_b, m_b]
                        a_b = a_b + 1
                    gamma[j_b, m_b] = s_g
                    m_b = m_b + 1
                j_b = j_b + 1
            Px_out = Px
        with slot("bp.final_decision",
                  inputs=(gamma, constellation),
                  outputs=("x_hat",)):
            Nt_f = gamma.shape[0]
            x_hat = np.zeros(Nt_f, dtype=complex)
            i_f = 0
            while i_f < Nt_f:
                bi_f = _argmax_row(gamma, i_f)
                x_hat[i_f] = constellation[bi_f]
                i_f = i_f + 1
        return x_hat
""")

EP_TEMPLATE = textwrap.dedent("""\
    def ep(H, y, sigma2, constellation):
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
        it_e = 0
        while it_e < 20:
            sig = np.real(np.diag(Sigma_q))
            i_e = 0
            while i_e < Nt:
                sig_i = float(sig[i_e])
                alpha_i = float(alpha[i_e])
                denom = 1.0 - sig_i * alpha_i
                denom = max(denom, 1e-30)
                h2 = sig_i / denom
                h2 = max(h2, 1e-30)
                sig_inv = 1.0 / max(sig_i, 1e-30)
                ratio = mu_q[i_e] * sig_inv - gamma_ep[i_e]
                t_e = h2 * ratio
                gamma_e_i = gamma_ep[i_e]
                with slot("ep.cavity",
                          inputs=(t_e, h2, gamma_e_i, alpha_i),
                          outputs=("cav_t", "cav_h2")):
                    cav_t = t_e
                    cav_h2 = h2
                with slot("ep.site_update",
                          inputs=(cav_t, cav_h2, constellation, gamma_e_i, alpha_i),
                          outputs=("new_alpha", "new_gamma")):
                    diffs_su = constellation - cav_t
                    exps_su = -np.abs(diffs_su) ** 2 / max(2.0 * cav_h2, 1e-30)
                    me_su = float(np.max(np.real(exps_su)))
                    w_su = np.exp(np.real(exps_su) - me_su)
                    ws_su = max(float(np.sum(w_su)), 1e-30)
                    w_su = w_su / ws_su
                    mu_p = np.sum(w_su * constellation)
                    sig2_p = float(np.real(np.sum(w_su * np.abs(constellation) ** 2))) - float(np.abs(mu_p) ** 2)
                    sig2_p = max(sig2_p, 5e-7)
                    new_alpha = 1.0 / sig2_p - 1.0 / max(cav_h2, 1e-30)
                    new_gamma = mu_p / sig2_p - cav_t / max(cav_h2, 1e-30)
                na_real = float(np.real(new_alpha))
                if na_real > 1e-6:
                    alpha[i_e] = damping * na_real + one_m_damp * alpha_i
                    gamma_ep[i_e] = damping * new_gamma + one_m_damp * gamma_ep[i_e]
                i_e = i_e + 1
            rhs = Hty_scaled + gamma_ep
            Sigma_q = np.linalg.inv(HtH_scaled + np.diag(alpha))
            mu_q = Sigma_q @ rhs
            it_e = it_e + 1
        with slot("ep.final_decision",
                  inputs=(mu_q, constellation),
                  outputs=("x_hat",)):
            x_hat = np.zeros(len(mu_q), dtype=complex)
            i_fd = 0
            while i_fd < len(mu_q):
                d_fd = np.abs(constellation - mu_q[i_fd]) ** 2
                x_hat[i_fd] = constellation[np.argmin(d_fd)]
                i_fd = i_fd + 1
        return x_hat
""")

AMP_TEMPLATE = textwrap.dedent("""\
    def amp(H, y, sigma2, constellation):
        Nr = H.shape[0]
        Nt = H.shape[1]
        G_amp = H.conj().T @ H
        g_diag = np.real(np.diag(G_amp))
        g_diag = np.maximum(g_diag, 1e-30)
        Gtilde = np.eye(Nt) - np.diag(1.0 / g_diag) @ G_amp
        g_scale = g_diag / max(float(Nr), 1.0)
        gtilde = 1.0 / g_diag
        yMF = H.conj().T @ y
        yMFtilde = gtilde * yMF
        x_hat_a = np.zeros(Nt, dtype=complex)
        tau_s = np.ones(Nt)
        z = yMFtilde.copy()
        it_a = 0
        while it_a < 20:
            with slot("amp.amp_iterate",
                      inputs=(G_amp, Gtilde, g_scale, gtilde, yMFtilde,
                              sigma2, x_hat_a, tau_s, z, constellation),
                      outputs=("x_new", "tau_s_new", "z_new")):
                Nt_a = len(x_hat_a)
                M_a = len(constellation)
                tau_p = g_scale @ tau_s
                theta_s = 0.5
                theta_z = 0.5
                tau_p_safe = np.maximum(tau_p, 1e-30)
                tau_p_plus_s2 = tau_p_safe + sigma2
                tau_z = tau_p_plus_s2 * gtilde
                tau_z = np.maximum(tau_z, 1e-30)
                x_new = np.zeros(Nt_a, dtype=complex)
                x_var = np.zeros(Nt_a)
                i_a = 0
                while i_a < Nt_a:
                    w_a = np.zeros(M_a)
                    j_a = 0
                    me_a = -1e30
                    while j_a < M_a:
                        dj_a = float(np.abs(z[i_a] - constellation[j_a]) ** 2)
                        e_a = 0.0 - dj_a / max(float(tau_z[i_a]), 1e-30)
                        if e_a > me_a:
                            me_a = e_a
                        w_a[j_a] = e_a
                        j_a = j_a + 1
                    j_a = 0
                    ws_a = 0.0
                    while j_a < M_a:
                        w_a[j_a] = np.exp(w_a[j_a] - me_a)
                        ws_a = ws_a + w_a[j_a]
                        j_a = j_a + 1
                    ws_a = max(ws_a, 1e-30)
                    j_a = 0
                    while j_a < M_a:
                        w_a[j_a] = w_a[j_a] / ws_a
                        j_a = j_a + 1
                    mu_i = 0.0 + 0.0j
                    j_a = 0
                    while j_a < M_a:
                        mu_i = mu_i + w_a[j_a] * constellation[j_a]
                        j_a = j_a + 1
                    x_new[i_a] = mu_i
                    v_i = 0.0
                    j_a = 0
                    while j_a < M_a:
                        v_i = v_i + w_a[j_a] * float(np.abs(constellation[j_a] - mu_i) ** 2)
                        j_a = j_a + 1
                    x_var[i_a] = v_i
                    i_a = i_a + 1
                shat_old = x_hat_a.copy()
                one_m_ts = 1.0 - theta_s
                tau_s_new = theta_s * x_var + one_m_ts * tau_s
                tau_p_new = g_scale @ tau_s_new
                tp_old_safe = np.maximum(tau_p, 1e-30)
                tp_denom = tp_old_safe + sigma2
                v_scale = tau_p_new / tp_denom
                z_diff = z - shat_old
                v_corr = v_scale * z_diff
                one_m_tz = 1.0 - theta_z
                tp_new_plus_s2 = tau_p_new + sigma2
                tau_z_new = theta_z * tp_new_plus_s2 * gtilde + one_m_tz * tau_z
                z_new = yMFtilde + Gtilde @ x_new + v_corr
            x_hat_a = x_new
            tau_s = tau_s_new
            z = z_new
            it_a = it_a + 1
        with slot("amp.final_decision",
                  inputs=(z, constellation),
                  outputs=("x_out",)):
            x_out = np.zeros(len(z), dtype=complex)
            i_fd = 0
            while i_fd < len(z):
                d_fd = np.abs(constellation - z[i_fd]) ** 2
                x_out[i_fd] = constellation[np.argmin(d_fd)]
                i_fd = i_fd + 1
        return x_out
""")


# ═══════════════════════════════════════════════════════════════════════════
# Detector spec table
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class _DetectorSpec:
    """Specification for one L3 detector template."""
    algo_id: str
    source: str
    func_name: str
    slot_defs_fn: Any                   # callable returning dict[str, SlotDescriptor]
    tags: set[str]
    extra_globals: dict[str, Any] | None = None


_DETECTOR_SPECS: list[_DetectorSpec] = [
    _DetectorSpec(
        algo_id="lmmse",
        source=LMMSE_TEMPLATE,
        func_name="lmmse",
        slot_defs_fn=_lmmse_slots,
        tags={"original", "linear"},
    ),
    _DetectorSpec(
        algo_id="zf",
        source=ZF_TEMPLATE,
        func_name="zf",
        slot_defs_fn=_zf_slots,
        tags={"original", "linear"},
    ),
    _DetectorSpec(
        algo_id="osic",
        source=OSIC_TEMPLATE,
        func_name="osic",
        slot_defs_fn=_osic_slots,
        tags={"original", "sic"},
    ),
    _DetectorSpec(
        algo_id="kbest",
        source=KBEST_TEMPLATE,
        func_name="kbest",
        slot_defs_fn=_kbest_slots,
        tags={"original", "tree_search"},
        extra_globals={"TreeNode": None},
    ),
    _DetectorSpec(
        algo_id="stack",
        source=STACK_TEMPLATE,
        func_name="stack",
        slot_defs_fn=_stack_slots,
        tags={"original", "tree_search"},
        extra_globals={"TreeNode": None},
    ),
    _DetectorSpec(
        algo_id="bp",
        source=BP_TEMPLATE,
        func_name="bp",
        slot_defs_fn=_bp_slots,
        tags={"original", "message_passing"},
    ),
    _DetectorSpec(
        algo_id="ep",
        source=EP_TEMPLATE,
        func_name="ep",
        slot_defs_fn=_ep_slots,
        tags={"original", "inference"},
    ),
    _DetectorSpec(
        algo_id="amp",
        source=AMP_TEMPLATE,
        func_name="amp",
        slot_defs_fn=_amp_slots,
        tags={"original", "inference"},
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# Compile a detector template → FunctionIR with slot_meta annotations
# ═══════════════════════════════════════════════════════════════════════════

def compile_detector_template(spec: _DetectorSpec) -> FunctionIR:
    """Compile a detector template source to FunctionIR.

    The template uses ``with slot(...)`` blocks; the IR builder records
    each slot in ``func_ir.slot_meta`` and tags every op inside with
    ``op.attrs["slot_id"]`` (innermost). No AlgSlot ops are emitted —
    slot membership is annotation-only.
    """
    g = _template_globals()
    if spec.extra_globals:
        from evolution.pool_ops_l2 import TreeNode
        g["TreeNode"] = TreeNode
    return compile_source_to_ir(spec.source, spec.func_name, g)


# ═══════════════════════════════════════════════════════════════════════════
# Query helpers on compiled IR
# ═══════════════════════════════════════════════════════════════════════════

def find_algslot_ops(func_ir: FunctionIR) -> list[Op]:
    """Return one representative op per slot in ``func_ir.slot_meta``.

    Backwards-compat shim for callers that previously enumerated
    ``AlgSlot`` ops. In the annotation-only model, slot membership is
    recorded in ``func_ir.slot_meta``; this helper returns the first
    tagged op for each slot so existing call sites can iterate slots.
    """
    if not func_ir.slot_meta:
        return []
    reps: list[Op] = []
    for pop_key, meta in func_ir.slot_meta.items():
        for op_id in meta.op_ids:
            op = func_ir.ops.get(op_id)
            if op is not None:
                reps.append(op)
                break
    return reps


def get_slot_ids(func_ir: FunctionIR) -> list[str]:
    """Return the list of slot pop_keys present in ``func_ir.slot_meta``."""
    if not func_ir.slot_meta:
        return []
    return list(func_ir.slot_meta.keys())


# ═══════════════════════════════════════════════════════════════════════════
# Build the full IR pool
# ═══════════════════════════════════════════════════════════════════════════

# Manifest of pool rejections from the most recent ``build_ir_pool`` call.
# Inspect via ``from evolution.ir_pool import _POOL_REJECTIONS``.
_POOL_REJECTIONS: list[dict] = []


def build_ir_pool(
    rng: np.random.Generator | None = None,
    n_random_variants: int = 7,  # noqa: ARG001  (M3 will wire random variants)
) -> list[AlgorithmGenome]:
    """Build the initial IR-based algorithm pool.

    For each L3 detector:
      1. Compile template → flat ``FunctionIR`` with ``slot_meta``
         (annotation-only slot model; no ``AlgSlot`` ops).
      2. For each ``pop_key`` in ``ir.slot_meta``, create a
         ``SlotPopulation`` whose ``spec`` comes from the matching
         ``SlotDescriptor`` (by ``slot_id``).  Variants list is empty
         in M2; M3 will populate it from the slot's inlined body and
         random programs.
      3. Run the IR validator; reject genomes that fail.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    pool: list[AlgorithmGenome] = []
    _POOL_REJECTIONS.clear()

    for spec in _DETECTOR_SPECS:
        # 1. Compile template
        try:
            ir = compile_detector_template(spec)
        except Exception as exc:
            _POOL_REJECTIONS.append({
                "algo_id": spec.algo_id,
                "n_errors": 1,
                "first_errors": [f"compile failed: {exc!r}"],
            })
            continue

        # 2. Match slot_meta entries against slot descriptor table
        slot_tree = spec.slot_defs_fn()
        populations: dict[str, SlotPopulation] = {}
        slot_meta = ir.slot_meta or {}
        for pop_key in slot_meta.keys():
            desc = slot_tree.get(pop_key)
            if desc is None:
                # No descriptor → skip (no GP spec available)
                continue
            populations[pop_key] = SlotPopulation(
                slot_id=pop_key,
                spec=desc.spec,
                variants=[],
                fitness=[],
                best_idx=0,
            )

        genome = AlgorithmGenome(
            algo_id=spec.algo_id,
            ir=ir,
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

        # 3. Hard validation gate
        try:
            from algorithm_ir.ir.validator import validate_function_ir
            errs = validate_function_ir(genome.ir)
        except Exception as exc:
            errs = [f"validator crashed: {exc!r}"]
        if errs:
            _POOL_REJECTIONS.append({
                "algo_id": spec.algo_id,
                "n_errors": len(errs),
                "first_errors": errs[:5],
            })
            continue
        pool.append(genome)

    return pool
