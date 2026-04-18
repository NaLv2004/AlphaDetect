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


# ═══════════════════════════════════════════════════════════════════════════
# Helper: namespace for template compilation
# ═══════════════════════════════════════════════════════════════════════════

def _template_globals() -> dict[str, Any]:
    """Safe namespace for compiling detector templates to IR.

    Since the IR builder does not support keyword arguments (ast.keyword),
    we provide helper functions that wrap numpy calls using keyword args
    internally.
    """
    import math

    def _safe_div(a, b):
        return a / b if abs(b) > 1e-30 else 0.0

    def _safe_sqrt(a):
        return math.sqrt(max(a, 0.0))

    def _safe_log(a):
        return math.log(max(a, 1e-30))

    # --- dtype helpers (avoid dtype=complex keyword) ---
    def _czeros(n):
        return np.zeros(n, dtype=complex)

    def _czeros2(nr, nc):
        return np.zeros((nr, nc), dtype=complex)

    def _cones(n):
        return np.ones(n, dtype=complex)

    def _cempty(n):
        return np.empty(n, dtype=complex)

    def _cfull(n, val):
        return np.full(n, val, dtype=float)

    def _carray(lst):
        return np.array(lst, dtype=complex)

    # --- axis helpers ---
    def _sum0(x):
        return np.sum(x, axis=0)

    def _sum1(x):
        return np.sum(x, axis=1)

    def _max1_keepdims(x):
        return np.max(x, axis=1, keepdims=True)

    def _sum1_keepdims(x):
        return np.sum(x, axis=1, keepdims=True)

    # --- misc helpers ---
    def _qr(H):
        return np.linalg.qr(H)

    def _qr_Q(H):
        return np.linalg.qr(H)[0]

    def _qr_R(H):
        return np.linalg.qr(H)[1]

    def _delete_col(H, idx):
        return np.delete(H, idx, axis=1)

    def _make_tree_node(level, symbols, cost):
        from evolution.pool_ops_l2 import TreeNode
        return TreeNode(level=level, symbols=symbols, cost=cost)

    def _col(H, j):
        return H[:, j]

    return {
        "np": np,
        "math": math,
        "_safe_div": _safe_div,
        "_safe_sqrt": _safe_sqrt,
        "_safe_log": _safe_log,
        "_czeros": _czeros,
        "_czeros2": _czeros2,
        "_cones": _cones,
        "_cempty": _cempty,
        "_cfull": _cfull,
        "_carray": _carray,
        "_sum0": _sum0,
        "_sum1": _sum1,
        "_max1_keepdims": _max1_keepdims,
        "_sum1_keepdims": _sum1_keepdims,
        "_qr": _qr,
        "_qr_Q": _qr_Q,
        "_qr_R": _qr_R,
        "_delete_col": _delete_col,
        "_make_tree_node": _make_tree_node,
        "_col": _col,
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
            x_hat = _czeros(len(x_soft))
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
            snr = _czeros(Nt)
            i = 0
            while i < Nt:
                snr[i] = 1.0 / max(float(np.real(G_inv[i, i])), 1e-30) - sigma2
                i = i + 1
            order = []
            used = _czeros(Nt)
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
            H_new = _delete_col(H, idx)
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
                j = Nt - len(node.symbols) + j_rel
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
        def bp_sweep(H, y, sigma2, mu, var, constellation, max_iters):
            Nr = H.shape[0]
            Nt = H.shape[1]
            damping = 0.5
            messages_mu = _czeros2(Nr, Nt)
            messages_var = np.ones((Nr, Nt))
            it = 0
            while it < max_iters:
                mu_old = mu.copy()
                a = 0
                while a < Nr:
                    j = 0
                    while j < Nt:
                        s = sigma2
                        k = 0
                        while k < Nt:
                            if k != j:
                                s = s + np.abs(H[a, k]) ** 2 * var[k]
                            k = k + 1
                        s_inv = 1.0 / max(s, 1e-30)
                        messages_var[a, j] = 1.0 / max(np.abs(H[a, j]) ** 2 * s_inv, 1e-30)
                        r = y[a]
                        k = 0
                        while k < Nt:
                            if k != j:
                                r = r - H[a, k] * mu[k]
                            k = k + 1
                        messages_mu[a, j] = r * np.conj(H[a, j]) * s_inv * messages_var[a, j]
                        j = j + 1
                    a = a + 1
                j = 0
                while j < Nt:
                    prec = 0.0
                    wm = 0.0 + 0.0j
                    a = 0
                    while a < Nr:
                        p = 1.0 / max(messages_var[a, j], 1e-30)
                        prec = prec + p
                        wm = wm + messages_mu[a, j] * p
                        a = a + 1
                    new_var = 1.0 / max(prec, 1e-30)
                    new_mu = wm * new_var
                    var[j] = damping * float(new_var) + (1.0 - damping) * var[j]
                    mu[j] = damping * new_mu + (1.0 - damping) * mu[j]
                    j = j + 1
                it = it + 1
            return mu, var
    """),
    "final_decision": textwrap.dedent("""\
        def final_decision(mu, constellation):
            x_hat = _czeros(len(mu))
            i = 0
            while i < len(mu):
                dists = np.abs(constellation - mu[i]) ** 2
                x_hat[i] = constellation[np.argmin(dists)]
                i = i + 1
            return x_hat
    """),

    # ----- EP -----
    "cavity": textwrap.dedent("""\
        def cavity(g_mu, g_var, s_mu, s_var):
            cav_var = 1.0 / max(1.0 / max(g_var, 1e-30) - 1.0 / max(s_var, 1e-30), 1e-30)
            cav_mu = cav_var * (g_mu / max(g_var, 1e-30) - s_mu / max(s_var, 1e-30))
            return cav_mu, cav_var
    """),
    "site_update": textwrap.dedent("""\
        def site_update(cav_mu, cav_var, constellation, s_mu, s_var):
            diffs = constellation - cav_mu
            exponents = -np.abs(diffs) ** 2 / max(2.0 * cav_var, 1e-30)
            max_exp = np.max(np.real(exponents))
            weights = np.exp(np.real(exponents) - max_exp)
            w_sum = max(np.sum(weights), 1e-30)
            weights = weights / w_sum
            post_mu = np.sum(weights * constellation)
            post_var = max(float(np.real(np.sum(weights * np.abs(constellation - post_mu) ** 2))), 1e-30)
            new_var = 1.0 / max(1.0 / max(post_var, 1e-30) - 1.0 / max(cav_var, 1e-30), 1e-30)
            new_mu = new_var * (post_mu / max(post_var, 1e-30) - cav_mu / max(cav_var, 1e-30))
            return new_mu, new_var
    """),
    "damping": textwrap.dedent("""\
        def damping(old, new, iteration):
            alpha = 0.5
            return (alpha * new[0] + (1 - alpha) * old[0],
                    alpha * new[1] + (1 - alpha) * old[1])
    """),

    # ----- AMP -----
    "amp_iterate": textwrap.dedent("""\
        def amp_iterate(H, y, sigma2, x_hat, s_hat, z_prev, constellation):
            Nr = H.shape[0]
            Nt = H.shape[1]
            z = y - H @ x_hat
            onsager_scale = np.mean(s_hat) / max(Nr, 1)
            z = z + onsager_scale * z_prev
            r = x_hat + H.conj().T @ z
            tau = sigma2 * Nt / max(Nr, 1)
            M = len(constellation)
            x_new = _czeros(Nt)
            x_var = np.zeros(Nt)
            i = 0
            while i < Nt:
                w = np.zeros(M)
                j = 0
                max_e = -1e30
                while j < M:
                    e = -float(np.abs(constellation[j] - r[i]) ** 2) / max(2.0 * tau, 1e-30)
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
            return x_new, x_var, z
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
        detected = _czeros(Nt)
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
        Q = _qr_Q(H)
        R = _qr_R(H)
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
            return _czeros(Nt)
        best = candidates[0]
        bi = 1
        while bi < len(candidates):
            if candidates[bi].cost < best.cost:
                best = candidates[bi]
            bi = bi + 1
        return _carray(best.symbols)
""")

STACK_TEMPLATE = textwrap.dedent("""\
    def stack(H, y, sigma2, constellation, slot_node_select, slot_expand):
        Nr = H.shape[0]
        Nt = H.shape[1]
        Q = _qr_Q(H)
        R = _qr_R(H)
        y_tilde = Q.conj().T @ y
        root = _make_tree_node(Nt - 1, [], 0.0)
        open_set = [root]
        nodes_expanded = 0
        max_nodes = 500
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
                    return _carray(node.symbols)
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
""")

BP_TEMPLATE = textwrap.dedent("""\
    def bp(H, y, sigma2, constellation, slot_bp_sweep, slot_final_decision):
        Nr = H.shape[0]
        Nt = H.shape[1]
        x_mf = H.conj().T @ y
        G_diag = np.real(_sum0(np.abs(H) ** 2))
        G_diag = np.maximum(G_diag, 1e-30)
        init_mu = x_mf / G_diag
        init_var = sigma2 / G_diag
        result = slot_bp_sweep(H, y, sigma2, init_mu, init_var, constellation, 20)
        mu = result[0]
        x_hat = slot_final_decision(mu, constellation)
        return x_hat
""")

EP_TEMPLATE = textwrap.dedent("""\
    def ep(H, y, sigma2, constellation, slot_cavity, slot_site_update, slot_final_decision):
        Nr = H.shape[0]
        Nt = H.shape[1]
        site_mu = _czeros(Nt)
        site_var = _cfull(Nt, 1e6)
        HtH = H.conj().T @ H
        Hty = H.conj().T @ y
        it = 0
        while it < 20:
            site_prec = 1.0 / np.maximum(site_var, 1e-30)
            Sigma_inv = HtH / max(sigma2, 1e-30) + np.diag(site_prec)
            Sigma = np.linalg.inv(Sigma_inv + 1e-10 * np.eye(Nt))
            global_mu = Sigma @ (Hty / max(sigma2, 1e-30) + site_mu * site_prec)
            global_var = np.real(np.diag(Sigma))
            i = 0
            while i < Nt:
                cav_result = slot_cavity(global_mu[i], float(global_var[i]),
                                          site_mu[i], float(site_var[i]))
                upd_result = slot_site_update(cav_result[0], cav_result[1], constellation,
                                               site_mu[i], float(site_var[i]))
                site_mu[i] = 0.5 * upd_result[0] + 0.5 * site_mu[i]
                site_var[i] = 0.5 * upd_result[1] + 0.5 * site_var[i]
                i = i + 1
            it = it + 1
        site_prec = 1.0 / np.maximum(site_var, 1e-30)
        Sigma_inv = HtH / max(sigma2, 1e-30) + np.diag(site_prec)
        Sigma = np.linalg.inv(Sigma_inv + 1e-10 * np.eye(Nt))
        final_mu = Sigma @ (Hty / max(sigma2, 1e-30) + site_mu * site_prec)
        x_hat = slot_final_decision(final_mu, constellation)
        return x_hat
""")

AMP_TEMPLATE = textwrap.dedent("""\
    def amp(H, y, sigma2, constellation, slot_amp_iterate, slot_final_decision):
        Nr = H.shape[0]
        Nt = H.shape[1]
        x_hat = _czeros(Nt)
        s_hat = np.ones(Nt)
        z = _czeros(Nr)
        it = 0
        while it < 20:
            result = slot_amp_iterate(H, y, sigma2, x_hat, s_hat, z, constellation)
            x_hat = result[0]
            s_hat = result[1]
            z = result[2]
            it = it + 1
        x_out = slot_final_decision(x_hat, constellation)
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
            "slot_final_decision": "final_decision",
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
        # 1. Compile template
        structural_ir = compile_detector_template(spec)

        # 2. Get slot definitions
        slot_tree = spec.slot_defs_fn()

        # 3. Build SlotPopulation for each TOP-LEVEL slot
        #    (only slots that appear as AlgSlot in the IR)
        ir_slot_ids = set(get_slot_ids(structural_ir))
        populations: dict[str, SlotPopulation] = {}

        for slot_id, desc in slot_tree.items():
            # Find matching slot_id in IR
            # The IR slot_id is the slot_arg_name minus "slot_" prefix.
            # The SlotDescriptor slot_id is e.g. "lmmse.regularizer"
            # We need to match: short_name == IR slot_id
            if desc.short_name not in ir_slot_ids:
                # Check for exact slot_id match too
                if slot_id not in ir_slot_ids:
                    continue

            # Determine the key for the default implementation
            default_key = None
            for arg_name, dk in spec.slot_default_keys.items():
                # arg_name = "slot_regularizer", short_name = "regularizer"
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
            structural_ir=structural_ir,
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
        pool.append(genome)

    return pool
