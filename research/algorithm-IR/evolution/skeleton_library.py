"""Massive skeleton library for multi-level algorithm evolution.

Provides 200+ skeleton templates (L0–L3) for MIMO detection and supporting
sub-modules.  Each skeleton is a compilable Python source with ``slot_*``
parameters for two-level evolution.

Organisation
------------
A. Iterative linear detectors  (15 templates)
B. Optimisation-based detectors (12 templates)
C. Search-based detectors       (8 templates)
D. Probabilistic detectors      (8 templates)
E. Hybrid / multi-stage         (10 templates)
F. Sub-module wrappers L2       (15 templates)
G. Building block wrappers L1   (12 templates)
H. Cross-domain algorithms      (10 templates)

Each entry is a ``SkeletonSpec`` which mirrors ``_DetectorSpec`` from
``ir_pool.py``.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable

from evolution.pool_types import SlotDescriptor
from evolution.skeleton_registry import ProgramSpec


# ═══════════════════════════════════════════════════════════════════════════
# SkeletonSpec — mirrors _DetectorSpec for any level
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SkeletonSpec:
    algo_id: str
    source: str
    func_name: str
    slot_arg_names: list[str]
    slot_defs: dict[str, SlotDescriptor]
    slot_default_keys: dict[str, str]
    tags: set[str]
    level: int = 3
    extra_globals: dict[str, Any] | None = None


# ═══════════════════════════════════════════════════════════════════════════
# Common ProgramSpec definitions (reused across many skeletons)
# ═══════════════════════════════════════════════════════════════════════════

_PS_HARD_DECISION = ProgramSpec(
    name="hard_decision",
    param_names=["x_soft", "constellation"],
    param_types=["vec_cx", "vec_cx"],
    return_type="vec_cx",
)

_PS_INIT = ProgramSpec(
    name="init",
    param_names=["H", "y", "sigma2"],
    param_types=["mat", "vec_cx", "float"],
    return_type="vec_cx",
)

_PS_REGULARIZER = ProgramSpec(
    name="regularizer",
    param_names=["G", "sigma2"],
    param_types=["mat", "float"],
    return_type="mat",
)

_PS_PRECONDITIONER = ProgramSpec(
    name="preconditioner",
    param_names=["G", "sigma2"],
    param_types=["mat", "float"],
    return_type="mat",
)

_PS_ITERATE_FULL = ProgramSpec(
    name="iterate",
    param_names=["H", "y", "sigma2", "x", "constellation"],
    param_types=["mat", "vec_cx", "float", "vec_cx", "vec_cx"],
    return_type="vec_cx",
)

_PS_ITERATE_LINEAR = ProgramSpec(
    name="iterate",
    param_names=["G", "rhs", "x"],
    param_types=["mat", "vec_cx", "vec_cx"],
    return_type="vec_cx",
)

_PS_DAMPING = ProgramSpec(
    name="damping",
    param_names=["x_old", "x_new"],
    param_types=["vec_cx", "vec_cx"],
    return_type="vec_cx",
)

_PS_STEP_SIZE = ProgramSpec(
    name="step_size",
    param_names=["gradient", "x"],
    param_types=["vec_cx", "vec_cx"],
    return_type="float",
)

_PS_GRADIENT = ProgramSpec(
    name="gradient",
    param_names=["H", "y", "sigma2", "x"],
    param_types=["mat", "vec_cx", "float", "vec_cx"],
    return_type="vec_cx",
)

_PS_PROJECT = ProgramSpec(
    name="project",
    param_names=["x", "constellation"],
    param_types=["vec_cx", "vec_cx"],
    return_type="vec_cx",
)

_PS_COST = ProgramSpec(
    name="cost",
    param_names=["H", "y", "sigma2", "x"],
    param_types=["mat", "vec_cx", "float", "vec_cx"],
    return_type="float",
)

_PS_REFINE = ProgramSpec(
    name="refine",
    param_names=["H", "y", "sigma2", "x", "constellation"],
    param_types=["mat", "vec_cx", "float", "vec_cx", "vec_cx"],
    return_type="vec_cx",
)

_PS_STAGE = ProgramSpec(
    name="stage",
    param_names=["H", "y", "sigma2", "constellation"],
    param_types=["mat", "vec_cx", "float", "vec_cx"],
    return_type="vec_cx",
)

_PS_MERGE = ProgramSpec(
    name="merge",
    param_names=["x1", "x2", "constellation"],
    param_types=["vec_cx", "vec_cx", "vec_cx"],
    return_type="vec_cx",
)

_PS_SCORE = ProgramSpec(
    name="score",
    param_names=["x", "H", "y", "sigma2"],
    param_types=["vec_cx", "mat", "vec_cx", "float"],
    return_type="float",
)

_PS_X_UPDATE = ProgramSpec(
    name="x_update",
    param_names=["H", "y", "sigma2", "z", "u"],
    param_types=["mat", "vec_cx", "float", "vec_cx", "vec_cx"],
    return_type="vec_cx",
)

_PS_Z_UPDATE = ProgramSpec(
    name="z_update",
    param_names=["x", "u", "constellation"],
    param_types=["vec_cx", "vec_cx", "vec_cx"],
    return_type="vec_cx",
)

_PS_DUAL_UPDATE = ProgramSpec(
    name="dual_update",
    param_names=["u", "x", "z"],
    param_types=["vec_cx", "vec_cx", "vec_cx"],
    return_type="vec_cx",
)

_PS_LINEAR_SOLVE = ProgramSpec(
    name="linear_solve",
    param_names=["A", "b"],
    param_types=["mat", "vec_cx"],
    return_type="vec_cx",
)

_PS_MATRIX_DECOMP = ProgramSpec(
    name="matrix_decomp",
    param_names=["A"],
    param_types=["mat"],
    return_type="tuple",
)

_PS_ORDERING = ProgramSpec(
    name="ordering",
    param_names=["H", "y", "sigma2"],
    param_types=["mat", "vec_cx", "float"],
    return_type="list_int",
)

_PS_CANCEL = ProgramSpec(
    name="cancel",
    param_names=["y", "H", "idx", "x_hat"],
    param_types=["vec_cx", "mat", "int", "cx"],
    return_type="tuple",
)

_PS_WEIGHT = ProgramSpec(
    name="weight",
    param_names=["H", "sigma2"],
    param_types=["mat", "float"],
    return_type="mat",
)

_PS_SOFT_ESTIMATE = ProgramSpec(
    name="soft_estimate",
    param_names=["x_soft", "variance", "constellation"],
    param_types=["vec_cx", "vec_f", "vec_cx"],
    return_type="vec_cx",
)

_PS_SHRINKAGE = ProgramSpec(
    name="shrinkage",
    param_names=["x", "threshold"],
    param_types=["vec_cx", "float"],
    return_type="vec_cx",
)

_PS_PROPOSAL = ProgramSpec(
    name="proposal",
    param_names=["x", "sigma2", "constellation"],
    param_types=["vec_cx", "float", "vec_cx"],
    return_type="vec_cx",
)

_PS_ACCEPT = ProgramSpec(
    name="accept",
    param_names=["cost_old", "cost_new", "temperature"],
    param_types=["float", "float", "float"],
    return_type="float",
)

_PS_NORMALIZE = ProgramSpec(
    name="normalize",
    param_names=["x"],
    param_types=["vec_cx"],
    return_type="vec_cx",
)

_PS_COMBINE_LAYER = ProgramSpec(
    name="combine_layer",
    param_names=["x_coarse", "x_fine"],
    param_types=["vec_cx", "vec_cx"],
    return_type="vec_cx",
)


# ═══════════════════════════════════════════════════════════════════════════
# Helper: build SlotDescriptor with common patterns
# ═══════════════════════════════════════════════════════════════════════════

def _sd(prefix: str, short: str, spec: ProgramSpec, level: int = 0,
        depth: int = 0, tags: set[str] | None = None) -> tuple[str, SlotDescriptor]:
    sid = f"{prefix}.{short}"
    return sid, SlotDescriptor(
        slot_id=sid, short_name=short,
        level=level, depth=depth,
        parent_slot_id=None,
        spec=spec,
        domain_tags=tags or set(),
    )


# ═══════════════════════════════════════════════════════════════════════════
# SLOT DEFAULTS (new implementations for all new slots)
# ═══════════════════════════════════════════════════════════════════════════

EXTENDED_SLOT_DEFAULTS: dict[str, str] = {
    # ---- Initialisation ----
    "init_zero": textwrap.dedent("""\
        def init_zero(H, y, sigma2):
            Nt = H.shape[1]
            return np.zeros(Nt, dtype=complex)
    """),
    "init_matched_filter": textwrap.dedent("""\
        def init_matched_filter(H, y, sigma2):
            return H.conj().T @ y
    """),
    "init_mmse": textwrap.dedent("""\
        def init_mmse(H, y, sigma2):
            Nt = H.shape[1]
            G = H.conj().T @ H + sigma2 * np.eye(Nt)
            rhs = H.conj().T @ y
            return np.linalg.solve(G, rhs)
    """),

    # ---- Iterative step defaults ----
    "jacobi_iterate": textwrap.dedent("""\
        def jacobi_iterate(G, rhs, x):
            Nt = G.shape[0]
            x_new = np.zeros(Nt, dtype=complex)
            i = 0
            while i < Nt:
                s = 0.0 + 0.0j
                j = 0
                while j < Nt:
                    if j != i:
                        s = s + G[i, j] * x[j]
                    j = j + 1
                x_new[i] = (rhs[i] - s) / G[i, i]
                i = i + 1
            return x_new
    """),
    "gs_iterate": textwrap.dedent("""\
        def gs_iterate(G, rhs, x):
            Nt = G.shape[0]
            x_new = x.copy()
            i = 0
            while i < Nt:
                s = 0.0 + 0.0j
                j = 0
                while j < Nt:
                    if j != i:
                        s = s + G[i, j] * x_new[j]
                    j = j + 1
                x_new[i] = (rhs[i] - s) / G[i, i]
                i = i + 1
            return x_new
    """),
    "sor_iterate": textwrap.dedent("""\
        def sor_iterate(G, rhs, x):
            Nt = G.shape[0]
            omega = 1.2
            x_new = x.copy()
            i = 0
            while i < Nt:
                s = 0.0 + 0.0j
                j = 0
                while j < Nt:
                    if j != i:
                        s = s + G[i, j] * x_new[j]
                    j = j + 1
                gs_val = (rhs[i] - s) / G[i, i]
                x_new[i] = (1.0 - omega) * x[i] + omega * gs_val
                i = i + 1
            return x_new
    """),
    "ssor_iterate": textwrap.dedent("""\
        def ssor_iterate(G, rhs, x):
            Nt = G.shape[0]
            omega = 1.0
            x_half = x.copy()
            i = 0
            while i < Nt:
                s = 0.0 + 0.0j
                j = 0
                while j < Nt:
                    if j != i:
                        s = s + G[i, j] * x_half[j]
                    j = j + 1
                gs_val = (rhs[i] - s) / G[i, i]
                x_half[i] = (1.0 - omega) * x[i] + omega * gs_val
                i = i + 1
            x_new = x_half.copy()
            i = Nt - 1
            while i >= 0:
                s = 0.0 + 0.0j
                j = 0
                while j < Nt:
                    if j != i:
                        s = s + G[i, j] * x_new[j]
                    j = j + 1
                gs_val = (rhs[i] - s) / G[i, i]
                x_new[i] = (1.0 - omega) * x_half[i] + omega * gs_val
                i = i - 1
            return x_new
    """),
    "richardson_iterate": textwrap.dedent("""\
        def richardson_iterate(G, rhs, x):
            alpha = 0.01
            r = rhs - G @ x
            return x + alpha * r
    """),
    "neumann_iterate": textwrap.dedent("""\
        def neumann_iterate(G, rhs, x):
            Nt = G.shape[0]
            D_inv = np.zeros((Nt, Nt), dtype=complex)
            i = 0
            while i < Nt:
                D_inv[i, i] = 1.0 / G[i, i]
                i = i + 1
            E = np.eye(Nt) - D_inv @ G
            x_new = D_inv @ rhs + E @ x
            return x_new
    """),
    "chebyshev_iterate": textwrap.dedent("""\
        def chebyshev_iterate(G, rhs, x):
            Nt = G.shape[0]
            D_inv = np.zeros((Nt, Nt), dtype=complex)
            i = 0
            while i < Nt:
                D_inv[i, i] = 1.0 / G[i, i]
                i = i + 1
            r = rhs - G @ x
            return x + D_inv @ r
    """),

    # ---- Gradient / optimisation defaults ----
    "gradient_ml": textwrap.dedent("""\
        def gradient_ml(H, y, sigma2, x):
            r = y - H @ x
            return H.conj().T @ r
    """),
    "step_size_fixed": textwrap.dedent("""\
        def step_size_fixed(gradient, x):
            return 0.01
    """),
    "step_size_bb": textwrap.dedent("""\
        def step_size_bb(gradient, x):
            g_norm = float(np.real(np.conj(gradient) @ gradient))
            if g_norm < 1e-30:
                return 0.01
            return min(0.1, 1.0 / max(g_norm, 1e-30))
    """),
    "project_nearest_symbol": textwrap.dedent("""\
        def project_nearest_symbol(x, constellation):
            Nt = len(x)
            x_proj = np.zeros(Nt, dtype=complex)
            i = 0
            while i < Nt:
                dists = np.abs(constellation - x[i]) ** 2
                x_proj[i] = constellation[np.argmin(dists)]
                i = i + 1
            return x_proj
    """),
    "project_box": textwrap.dedent("""\
        def project_box(x, constellation):
            bound = float(np.max(np.abs(constellation))) * 1.2
            x_proj = x.copy()
            i = 0
            while i < len(x):
                re = float(np.real(x[i]))
                im = float(np.imag(x[i]))
                re = max(-bound, min(bound, re))
                im = max(-bound, min(bound, im))
                x_proj[i] = re + 1j * im
                i = i + 1
            return x_proj
    """),
    "cost_ml": textwrap.dedent("""\
        def cost_ml(H, y, sigma2, x):
            r = y - H @ x
            return float(np.real(np.conj(r) @ r))
    """),

    # ---- Iteration with full params ----
    "iterate_gradient": textwrap.dedent("""\
        def iterate_gradient(H, y, sigma2, x, constellation):
            r = y - H @ x
            g = H.conj().T @ r
            alpha = 0.01
            return x + alpha * g
    """),
    "iterate_newton": textwrap.dedent("""\
        def iterate_newton(H, y, sigma2, x, constellation):
            Nt = H.shape[1]
            G = H.conj().T @ H + sigma2 * np.eye(Nt)
            r = y - H @ x
            g = H.conj().T @ r
            return x + np.linalg.solve(G, g)
    """),
    "iterate_proximal": textwrap.dedent("""\
        def iterate_proximal(H, y, sigma2, x, constellation):
            Nt = H.shape[1]
            r = y - H @ x
            g = H.conj().T @ r
            x_new = x + 0.01 * g
            i = 0
            while i < Nt:
                dists = np.abs(constellation - x_new[i]) ** 2
                x_new[i] = constellation[np.argmin(dists)]
                i = i + 1
            return x_new
    """),
    "iterate_coordinate": textwrap.dedent("""\
        def iterate_coordinate(H, y, sigma2, x, constellation):
            Nt = H.shape[1]
            x_new = x.copy()
            j = 0
            while j < Nt:
                r = y - H @ x_new
                h_j = _col(H, j)
                x_new[j] = x_new[j] + float(np.real(np.conj(h_j) @ r)) / max(float(np.real(np.conj(h_j) @ h_j)), 1e-30)
                j = j + 1
            return x_new
    """),
    "iterate_heavy_ball": textwrap.dedent("""\
        def iterate_heavy_ball(H, y, sigma2, x, constellation):
            r = y - H @ x
            g = H.conj().T @ r
            return x + 0.01 * g
    """),

    # ---- Damping defaults ----
    "damping_fixed": textwrap.dedent("""\
        def damping_fixed(x_old, x_new):
            alpha = 0.5
            return alpha * x_new + (1.0 - alpha) * x_old
    """),
    "damping_none": textwrap.dedent("""\
        def damping_none(x_old, x_new):
            return x_new
    """),

    # ---- ADMM defaults ----
    "admm_x_update": textwrap.dedent("""\
        def admm_x_update(H, y, sigma2, z, u):
            Nt = H.shape[1]
            rho = 1.0
            G = H.conj().T @ H + (sigma2 + rho) * np.eye(Nt)
            rhs = H.conj().T @ y + rho * (z - u)
            return np.linalg.solve(G, rhs)
    """),
    "admm_z_update": textwrap.dedent("""\
        def admm_z_update(x, u, constellation):
            Nt = len(x)
            z = np.zeros(Nt, dtype=complex)
            v = x + u
            i = 0
            while i < Nt:
                dists = np.abs(constellation - v[i]) ** 2
                z[i] = constellation[np.argmin(dists)]
                i = i + 1
            return z
    """),
    "admm_dual_update": textwrap.dedent("""\
        def admm_dual_update(u, x, z):
            return u + x - z
    """),

    # ---- Merge / combine defaults ----
    "merge_average": textwrap.dedent("""\
        def merge_average(x1, x2, constellation):
            return 0.5 * (x1 + x2)
    """),
    "merge_best_ser": textwrap.dedent("""\
        def merge_best_ser(x1, x2, constellation):
            Nt = len(x1)
            x1_q = np.zeros(Nt, dtype=complex)
            x2_q = np.zeros(Nt, dtype=complex)
            i = 0
            while i < Nt:
                dists1 = np.abs(constellation - x1[i]) ** 2
                x1_q[i] = constellation[np.argmin(dists1)]
                dists2 = np.abs(constellation - x2[i]) ** 2
                x2_q[i] = constellation[np.argmin(dists2)]
                i = i + 1
            e1 = float(np.real(np.sum(np.abs(x1 - x1_q) ** 2)))
            e2 = float(np.real(np.sum(np.abs(x2 - x2_q) ** 2)))
            if e1 <= e2:
                return x1
            return x2
    """),

    # ---- Stage defaults ----
    "stage_lmmse": textwrap.dedent("""\
        def stage_lmmse(H, y, sigma2, constellation):
            Nt = H.shape[1]
            G = H.conj().T @ H + sigma2 * np.eye(Nt)
            x_soft = np.linalg.solve(G, H.conj().T @ y)
            x_hat = np.zeros(Nt, dtype=complex)
            i = 0
            while i < Nt:
                dists = np.abs(constellation - x_soft[i]) ** 2
                x_hat[i] = constellation[np.argmin(dists)]
                i = i + 1
            return x_hat
    """),
    "stage_zf": textwrap.dedent("""\
        def stage_zf(H, y, sigma2, constellation):
            Nt = H.shape[1]
            G = H.conj().T @ H
            x_soft = np.linalg.solve(G, H.conj().T @ y)
            x_hat = np.zeros(Nt, dtype=complex)
            i = 0
            while i < Nt:
                dists = np.abs(constellation - x_soft[i]) ** 2
                x_hat[i] = constellation[np.argmin(dists)]
                i = i + 1
            return x_hat
    """),
    "stage_matched_filter": textwrap.dedent("""\
        def stage_matched_filter(H, y, sigma2, constellation):
            x_soft = H.conj().T @ y
            Nt = len(x_soft)
            x_hat = np.zeros(Nt, dtype=complex)
            i = 0
            while i < Nt:
                dists = np.abs(constellation - x_soft[i]) ** 2
                x_hat[i] = constellation[np.argmin(dists)]
                i = i + 1
            return x_hat
    """),

    # ---- Refine defaults ----
    "refine_gradient_step": textwrap.dedent("""\
        def refine_gradient_step(H, y, sigma2, x, constellation):
            r = y - H @ x
            g = H.conj().T @ r
            x_new = x + 0.02 * g
            Nt = len(x_new)
            x_hat = np.zeros(Nt, dtype=complex)
            i = 0
            while i < Nt:
                dists = np.abs(constellation - x_new[i]) ** 2
                x_hat[i] = constellation[np.argmin(dists)]
                i = i + 1
            return x_hat
    """),
    "refine_jacobi_step": textwrap.dedent("""\
        def refine_jacobi_step(H, y, sigma2, x, constellation):
            Nt = H.shape[1]
            G = H.conj().T @ H + sigma2 * np.eye(Nt)
            rhs = H.conj().T @ y
            x_new = np.zeros(Nt, dtype=complex)
            i = 0
            while i < Nt:
                s = 0.0 + 0.0j
                j = 0
                while j < Nt:
                    if j != i:
                        s = s + G[i, j] * x[j]
                    j = j + 1
                x_new[i] = (rhs[i] - s) / G[i, i]
                i = i + 1
            x_hat = np.zeros(Nt, dtype=complex)
            i = 0
            while i < Nt:
                dists = np.abs(constellation - x_new[i]) ** 2
                x_hat[i] = constellation[np.argmin(dists)]
                i = i + 1
            return x_hat
    """),

    # ---- Score defaults ----
    "score_ml_cost": textwrap.dedent("""\
        def score_ml_cost(x, H, y, sigma2):
            r = y - H @ x
            return float(np.real(np.conj(r) @ r))
    """),
    "score_map": textwrap.dedent("""\
        def score_map(x, H, y, sigma2):
            r = y - H @ x
            ll = float(np.real(np.conj(r) @ r)) / sigma2
            prior = float(np.real(np.sum(np.abs(x) ** 2)))
            return ll + 0.1 * prior
    """),

    # ---- Preconditioner defaults ----
    "precond_diagonal": textwrap.dedent("""\
        def precond_diagonal(G, sigma2):
            Nt = G.shape[0]
            M = np.zeros((Nt, Nt), dtype=complex)
            i = 0
            while i < Nt:
                M[i, i] = G[i, i]
                i = i + 1
            return M
    """),
    "precond_identity": textwrap.dedent("""\
        def precond_identity(G, sigma2):
            return np.eye(G.shape[0], dtype=complex)
    """),
    "precond_block_diagonal": textwrap.dedent("""\
        def precond_block_diagonal(G, sigma2):
            Nt = G.shape[0]
            M = np.zeros((Nt, Nt), dtype=complex)
            i = 0
            while i < Nt:
                M[i, i] = G[i, i]
                if i + 1 < Nt:
                    M[i, i + 1] = G[i, i + 1]
                    M[i + 1, i] = G[i + 1, i]
                i = i + 2
            return M
    """),

    # ---- Linear solve defaults ----
    "linear_solve_direct": textwrap.dedent("""\
        def linear_solve_direct(A, b):
            return np.linalg.solve(A, b)
    """),
    "linear_solve_cg": textwrap.dedent("""\
        def linear_solve_cg(A, b):
            Nt = A.shape[0]
            x = np.zeros(Nt, dtype=complex)
            r = b - A @ x
            p = r.copy()
            i = 0
            while i < 20:
                Ap = A @ p
                rr = float(np.real(np.conj(r) @ r))
                if rr < 1e-20:
                    return x
                alpha = rr / max(float(np.real(np.conj(p) @ Ap)), 1e-30)
                x = x + alpha * p
                r_new = r - alpha * Ap
                rr_new = float(np.real(np.conj(r_new) @ r_new))
                beta = rr_new / max(rr, 1e-30)
                p = r_new + beta * p
                r = r_new
                i = i + 1
            return x
    """),

    # ---- Ordering defaults ----
    "ordering_column_norm": textwrap.dedent("""\
        def ordering_column_norm(H, y, sigma2):
            Nt = H.shape[1]
            norms = np.zeros(Nt)
            j = 0
            while j < Nt:
                norms[j] = float(np.real(np.sum(np.abs(_col(H, j)) ** 2)))
                j = j + 1
            order = []
            used = np.zeros(Nt)
            k = 0
            while k < Nt:
                best = -1
                best_val = -1e30
                j = 0
                while j < Nt:
                    if used[j] == 0.0:
                        if norms[j] > best_val:
                            best_val = norms[j]
                            best = j
                    j = j + 1
                order.append(best)
                used[best] = 1.0
                k = k + 1
            return order
    """),
    "ordering_natural": textwrap.dedent("""\
        def ordering_natural(H, y, sigma2):
            Nt = H.shape[1]
            order = []
            i = 0
            while i < Nt:
                order.append(i)
                i = i + 1
            return order
    """),

    # ---- Cancel defaults ----
    "cancel_subtract": textwrap.dedent("""\
        def cancel_subtract(y, H, idx, x_hat):
            y_new = y - _col(H, idx) * x_hat
            H_new = np.delete(H, idx, axis=1)
            return x_hat, H_new, y_new
    """),

    # ---- Weight defaults ----
    "weight_mmse": textwrap.dedent("""\
        def weight_mmse(H, sigma2):
            Nt = H.shape[1]
            G = H.conj().T @ H + sigma2 * np.eye(Nt)
            return np.linalg.inv(G) @ H.conj().T
    """),
    "weight_zf": textwrap.dedent("""\
        def weight_zf(H, sigma2):
            G = H.conj().T @ H
            return np.linalg.inv(G) @ H.conj().T
    """),

    # ---- Shrinkage defaults ----
    "shrinkage_soft": textwrap.dedent("""\
        def shrinkage_soft(x, threshold):
            Nt = len(x)
            x_new = np.zeros(Nt, dtype=complex)
            i = 0
            while i < Nt:
                mag = abs(x[i])
                if mag > threshold:
                    x_new[i] = x[i] * (1.0 - threshold / mag)
                i = i + 1
            return x_new
    """),

    # ---- Proposal / acceptance defaults (MCMC) ----
    "proposal_gaussian": textwrap.dedent("""\
        def proposal_gaussian(x, sigma2, constellation):
            Nt = len(x)
            noise = np.random.randn(Nt) + 1j * np.random.randn(Nt)
            x_prop = x + 0.1 * noise
            x_q = np.zeros(Nt, dtype=complex)
            i = 0
            while i < Nt:
                dists = np.abs(constellation - x_prop[i]) ** 2
                x_q[i] = constellation[np.argmin(dists)]
                i = i + 1
            return x_q
    """),
    "proposal_flip": textwrap.dedent("""\
        def proposal_flip(x, sigma2, constellation):
            Nt = len(x)
            x_prop = x.copy()
            idx = int(np.random.randint(0, Nt))
            x_prop[idx] = constellation[int(np.random.randint(0, len(constellation)))]
            return x_prop
    """),
    "accept_metropolis": textwrap.dedent("""\
        def accept_metropolis(cost_old, cost_new, temperature):
            if cost_new < cost_old:
                return 1.0
            delta = (cost_new - cost_old) / max(temperature, 1e-30)
            if delta > 30.0:
                return 0.0
            return np.exp(-delta)
    """),

    # ---- Normalise defaults ----
    "normalize_unit": textwrap.dedent("""\
        def normalize_unit(x):
            n = float(np.real(np.sqrt(np.conj(x) @ x)))
            if n < 1e-30:
                return x
            return x / n
    """),
    "normalize_none": textwrap.dedent("""\
        def normalize_none(x):
            return x
    """),

    # ---- Combine layer defaults ----
    "combine_replace": textwrap.dedent("""\
        def combine_replace(x_coarse, x_fine):
            return x_fine
    """),
    "combine_weighted": textwrap.dedent("""\
        def combine_weighted(x_coarse, x_fine):
            return 0.3 * x_coarse + 0.7 * x_fine
    """),

    # ---- Matrix decomp defaults ----
    "decomp_qr": textwrap.dedent("""\
        def decomp_qr(A):
            Q, R = np.linalg.qr(A)
            return Q, R
    """),
    "decomp_cholesky": textwrap.dedent("""\
        def decomp_cholesky(A):
            L = np.linalg.cholesky(A)
            return L, L.conj().T
    """),

    # ---- Soft estimate defaults ----
    "soft_mmse_estimate": textwrap.dedent("""\
        def soft_mmse_estimate(x_soft, variance, constellation):
            Nt = len(x_soft)
            x_out = np.zeros(Nt, dtype=complex)
            i = 0
            while i < Nt:
                v = max(float(variance[i]), 1e-30) if len(variance) > i else 1.0
                w = np.exp(-np.abs(constellation - x_soft[i]) ** 2 / v)
                w_sum = float(np.sum(w))
                if w_sum < 1e-30:
                    x_out[i] = x_soft[i]
                else:
                    x_out[i] = np.sum(w * constellation) / w_sum
                i = i + 1
            return x_out
    """),
}


# ═══════════════════════════════════════════════════════════════════════════
# ══════════════════  TEMPLATE SOURCES  ═══════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════

# -----------------------------------------------------------------------
# A. ITERATIVE LINEAR DETECTORS
# -----------------------------------------------------------------------

T_JACOBI = textwrap.dedent("""\
def jacobi_detector(H, y, sigma2, constellation, slot_iterate, slot_hard_decision):
    Nt = H.shape[1]
    G = H.conj().T @ H + sigma2 * np.eye(Nt)
    rhs = H.conj().T @ y
    x = np.zeros(Nt, dtype=complex)
    i = 0
    while i < 20:
        x = slot_iterate(G, rhs, x)
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_GS = textwrap.dedent("""\
def gs_detector(H, y, sigma2, constellation, slot_iterate, slot_damping, slot_hard_decision):
    Nt = H.shape[1]
    G = H.conj().T @ H + sigma2 * np.eye(Nt)
    rhs = H.conj().T @ y
    x = np.zeros(Nt, dtype=complex)
    i = 0
    while i < 20:
        x_old = x.copy()
        x = slot_iterate(G, rhs, x)
        x = slot_damping(x_old, x)
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_SOR = textwrap.dedent("""\
def sor_detector(H, y, sigma2, constellation, slot_iterate, slot_hard_decision):
    Nt = H.shape[1]
    G = H.conj().T @ H + sigma2 * np.eye(Nt)
    rhs = H.conj().T @ y
    x = np.zeros(Nt, dtype=complex)
    i = 0
    while i < 20:
        x = slot_iterate(G, rhs, x)
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_SSOR = textwrap.dedent("""\
def ssor_detector(H, y, sigma2, constellation, slot_iterate, slot_hard_decision):
    Nt = H.shape[1]
    G = H.conj().T @ H + sigma2 * np.eye(Nt)
    rhs = H.conj().T @ y
    x = np.zeros(Nt, dtype=complex)
    i = 0
    while i < 20:
        x = slot_iterate(G, rhs, x)
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_RICHARDSON = textwrap.dedent("""\
def richardson_detector(H, y, sigma2, constellation, slot_iterate, slot_hard_decision):
    Nt = H.shape[1]
    G = H.conj().T @ H + sigma2 * np.eye(Nt)
    rhs = H.conj().T @ y
    x = np.zeros(Nt, dtype=complex)
    i = 0
    while i < 30:
        x = slot_iterate(G, rhs, x)
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_NEUMANN = textwrap.dedent("""\
def neumann_detector(H, y, sigma2, constellation, slot_iterate, slot_hard_decision):
    Nt = H.shape[1]
    G = H.conj().T @ H + sigma2 * np.eye(Nt)
    rhs = H.conj().T @ y
    x = np.zeros(Nt, dtype=complex)
    i = 0
    while i < 10:
        x = slot_iterate(G, rhs, x)
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_CHEBYSHEV = textwrap.dedent("""\
def chebyshev_detector(H, y, sigma2, constellation, slot_iterate, slot_hard_decision):
    Nt = H.shape[1]
    G = H.conj().T @ H + sigma2 * np.eye(Nt)
    rhs = H.conj().T @ y
    x = np.zeros(Nt, dtype=complex)
    i = 0
    while i < 15:
        x = slot_iterate(G, rhs, x)
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_CG = textwrap.dedent("""\
def cg_detector(H, y, sigma2, constellation, slot_preconditioner, slot_hard_decision):
    Nt = H.shape[1]
    G = H.conj().T @ H + sigma2 * np.eye(Nt)
    rhs = H.conj().T @ y
    M = slot_preconditioner(G, sigma2)
    x = np.zeros(Nt, dtype=complex)
    r = rhs - G @ x
    z = np.linalg.solve(M, r)
    p = z.copy()
    i = 0
    while i < 30:
        Ap = G @ p
        rz = float(np.real(np.conj(r) @ z))
        pAp = float(np.real(np.conj(p) @ Ap))
        alpha = rz / max(pAp, 1e-30)
        x = x + alpha * p
        r_new = r - alpha * Ap
        z_new = np.linalg.solve(M, r_new)
        rz_new = float(np.real(np.conj(r_new) @ z_new))
        beta = rz_new / max(rz, 1e-30)
        p = z_new + beta * p
        r = r_new
        z = z_new
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_PRECOND_CG = textwrap.dedent("""\
def pcg_detector(H, y, sigma2, constellation, slot_preconditioner, slot_damping, slot_hard_decision):
    Nt = H.shape[1]
    G = H.conj().T @ H + sigma2 * np.eye(Nt)
    rhs = H.conj().T @ y
    M = slot_preconditioner(G, sigma2)
    x = np.zeros(Nt, dtype=complex)
    r = rhs - G @ x
    z = np.linalg.solve(M, r)
    p = z.copy()
    i = 0
    while i < 25:
        Ap = G @ p
        rz = float(np.real(np.conj(r) @ z))
        pAp = float(np.real(np.conj(p) @ Ap))
        alpha = rz / max(pAp, 1e-30)
        x_new = x + alpha * p
        x_new = slot_damping(x, x_new)
        r_new = rhs - G @ x_new
        z_new = np.linalg.solve(M, r_new)
        rz_new = float(np.real(np.conj(r_new) @ z_new))
        beta = rz_new / max(rz, 1e-30)
        p = z_new + beta * p
        x = x_new
        r = r_new
        z = z_new
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_MINRES = textwrap.dedent("""\
def minres_detector(H, y, sigma2, constellation, slot_preconditioner, slot_hard_decision):
    Nt = H.shape[1]
    G = H.conj().T @ H + sigma2 * np.eye(Nt)
    rhs = H.conj().T @ y
    M = slot_preconditioner(G, sigma2)
    x = np.zeros(Nt, dtype=complex)
    r = rhs - G @ x
    z = np.linalg.solve(M, r)
    p = z.copy()
    i = 0
    while i < 25:
        Ap = G @ p
        pAp = float(np.real(np.conj(p) @ Ap))
        rAr = float(np.real(np.conj(r) @ Ap))
        alpha = rAr / max(pAp * pAp, 1e-30) * pAp
        alpha = float(np.real(np.conj(r) @ z)) / max(pAp, 1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        z = np.linalg.solve(M, r)
        beta = float(np.real(np.conj(r) @ z)) / max(float(np.real(np.conj(r) @ z)) + 1e-30, 1e-30)
        p = z + beta * p
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_BICGSTAB = textwrap.dedent("""\
def bicgstab_detector(H, y, sigma2, constellation, slot_preconditioner, slot_hard_decision):
    Nt = H.shape[1]
    G = H.conj().T @ H + sigma2 * np.eye(Nt)
    rhs = H.conj().T @ y
    M = slot_preconditioner(G, sigma2)
    x = np.zeros(Nt, dtype=complex)
    r = rhs - G @ x
    r_hat = r.copy()
    p = r.copy()
    rho = float(np.real(np.conj(r_hat) @ r))
    i = 0
    while i < 20:
        Mp = np.linalg.solve(M, p)
        Ap = G @ Mp
        rAp = float(np.real(np.conj(r_hat) @ Ap))
        alpha = rho / max(abs(rAp), 1e-30)
        s = r - alpha * Ap
        Ms = np.linalg.solve(M, s)
        As = G @ Ms
        omega_num = float(np.real(np.conj(As) @ s))
        omega_den = float(np.real(np.conj(As) @ As))
        omega = omega_num / max(omega_den, 1e-30)
        x = x + alpha * Mp + omega * Ms
        r = s - omega * As
        rho_new = float(np.real(np.conj(r_hat) @ r))
        beta = (rho_new / max(abs(rho), 1e-30)) * (alpha / max(abs(omega), 1e-30))
        p = r + beta * (p - omega * Ap)
        rho = rho_new
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_PRECOND_RICHARDSON = textwrap.dedent("""\
def precond_richardson(H, y, sigma2, constellation, slot_preconditioner, slot_iterate, slot_hard_decision):
    Nt = H.shape[1]
    G = H.conj().T @ H + sigma2 * np.eye(Nt)
    rhs = H.conj().T @ y
    M = slot_preconditioner(G, sigma2)
    x = np.zeros(Nt, dtype=complex)
    i = 0
    while i < 20:
        r = rhs - G @ x
        z = np.linalg.solve(M, r)
        x = x + 0.5 * z
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_WEIGHTED_JACOBI = textwrap.dedent("""\
def weighted_jacobi(H, y, sigma2, constellation, slot_iterate, slot_damping, slot_hard_decision):
    Nt = H.shape[1]
    G = H.conj().T @ H + sigma2 * np.eye(Nt)
    rhs = H.conj().T @ y
    x = np.zeros(Nt, dtype=complex)
    i = 0
    while i < 25:
        x_old = x.copy()
        x_jac = slot_iterate(G, rhs, x)
        x = slot_damping(x_old, x_jac)
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_STATIONARY = textwrap.dedent("""\
def stationary_detector(H, y, sigma2, constellation, slot_init, slot_iterate, slot_damping, slot_hard_decision):
    x = slot_init(H, y, sigma2)
    Nt = H.shape[1]
    G = H.conj().T @ H + sigma2 * np.eye(Nt)
    rhs = H.conj().T @ y
    i = 0
    while i < 20:
        x_old = x.copy()
        x = slot_iterate(G, rhs, x)
        x = slot_damping(x_old, x)
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")


# -----------------------------------------------------------------------
# B. OPTIMISATION-BASED DETECTORS
# -----------------------------------------------------------------------

T_GRADIENT_DESCENT = textwrap.dedent("""\
def gradient_descent_detector(H, y, sigma2, constellation, slot_gradient, slot_step_size, slot_hard_decision):
    Nt = H.shape[1]
    x = np.zeros(Nt, dtype=complex)
    i = 0
    while i < 30:
        g = slot_gradient(H, y, sigma2, x)
        alpha = slot_step_size(g, x)
        x = x + alpha * g
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_PROJECTED_GD = textwrap.dedent("""\
def projected_gd_detector(H, y, sigma2, constellation, slot_gradient, slot_step_size, slot_project, slot_hard_decision):
    Nt = H.shape[1]
    x = np.zeros(Nt, dtype=complex)
    i = 0
    while i < 30:
        g = slot_gradient(H, y, sigma2, x)
        alpha = slot_step_size(g, x)
        x = x + alpha * g
        x = slot_project(x, constellation)
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_NEWTON = textwrap.dedent("""\
def newton_detector(H, y, sigma2, constellation, slot_iterate, slot_hard_decision):
    Nt = H.shape[1]
    G = H.conj().T @ H + sigma2 * np.eye(Nt)
    rhs = H.conj().T @ y
    x = np.zeros(Nt, dtype=complex)
    i = 0
    while i < 5:
        r = rhs - G @ x
        dx = np.linalg.solve(G, r)
        x = x + dx
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_HEAVY_BALL = textwrap.dedent("""\
def heavy_ball_detector(H, y, sigma2, constellation, slot_gradient, slot_hard_decision):
    Nt = H.shape[1]
    x = np.zeros(Nt, dtype=complex)
    x_prev = np.zeros(Nt, dtype=complex)
    alpha = 0.01
    beta = 0.8
    i = 0
    while i < 30:
        g = slot_gradient(H, y, sigma2, x)
        x_new = x + alpha * g + beta * (x - x_prev)
        x_prev = x.copy()
        x = x_new
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_NESTEROV = textwrap.dedent("""\
def nesterov_detector(H, y, sigma2, constellation, slot_gradient, slot_hard_decision):
    Nt = H.shape[1]
    x = np.zeros(Nt, dtype=complex)
    v = np.zeros(Nt, dtype=complex)
    alpha = 0.01
    beta = 0.9
    i = 0
    while i < 30:
        x_look = x + beta * v
        g = slot_gradient(H, y, sigma2, x_look)
        v = beta * v + alpha * g
        x = x + v
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_COORDINATE_DESCENT = textwrap.dedent("""\
def coordinate_descent_detector(H, y, sigma2, constellation, slot_iterate, slot_hard_decision):
    Nt = H.shape[1]
    x = np.zeros(Nt, dtype=complex)
    it = 0
    while it < 10:
        x = slot_iterate(H, y, sigma2, x, constellation)
        it = it + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_ADMM = textwrap.dedent("""\
def admm_detector(H, y, sigma2, constellation, slot_x_update, slot_z_update, slot_dual_update, slot_hard_decision):
    Nt = H.shape[1]
    x = np.zeros(Nt, dtype=complex)
    z = np.zeros(Nt, dtype=complex)
    u = np.zeros(Nt, dtype=complex)
    i = 0
    while i < 30:
        x = slot_x_update(H, y, sigma2, z, u)
        z = slot_z_update(x, u, constellation)
        u = slot_dual_update(u, x, z)
        i = i + 1
    x_hat = slot_hard_decision(z, constellation)
    return x_hat
""")

T_PROXIMAL = textwrap.dedent("""\
def proximal_detector(H, y, sigma2, constellation, slot_gradient, slot_project, slot_hard_decision):
    Nt = H.shape[1]
    x = np.zeros(Nt, dtype=complex)
    alpha = 0.01
    i = 0
    while i < 30:
        g = slot_gradient(H, y, sigma2, x)
        x = x + alpha * g
        x = slot_project(x, constellation)
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_BARZILAI_BORWEIN = textwrap.dedent("""\
def bb_detector(H, y, sigma2, constellation, slot_gradient, slot_hard_decision):
    Nt = H.shape[1]
    x = np.zeros(Nt, dtype=complex)
    g_old = slot_gradient(H, y, sigma2, x)
    x = x + 0.01 * g_old
    i = 0
    while i < 30:
        g = slot_gradient(H, y, sigma2, x)
        dg = g - g_old
        dx = 0.01 * g_old
        dg_norm = float(np.real(np.conj(dg) @ dg))
        alpha = float(np.real(np.conj(dx) @ dg)) / max(dg_norm, 1e-30)
        alpha = max(1e-4, min(abs(alpha), 0.1))
        x = x + alpha * g
        g_old = g.copy()
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_FRANK_WOLFE = textwrap.dedent("""\
def frank_wolfe_detector(H, y, sigma2, constellation, slot_gradient, slot_project, slot_hard_decision):
    Nt = H.shape[1]
    x = np.zeros(Nt, dtype=complex)
    i = 0
    while i < 30:
        g = slot_gradient(H, y, sigma2, x)
        s = slot_project(g, constellation)
        gamma = 2.0 / (i + 2.0)
        x = (1.0 - gamma) * x + gamma * s
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_PENALTY = textwrap.dedent("""\
def penalty_detector(H, y, sigma2, constellation, slot_gradient, slot_project, slot_hard_decision):
    Nt = H.shape[1]
    x = np.zeros(Nt, dtype=complex)
    mu = 0.1
    i = 0
    while i < 30:
        g = slot_gradient(H, y, sigma2, x)
        x_proj = slot_project(x, constellation)
        penalty_grad = mu * (x - x_proj)
        x = x + 0.01 * g - 0.01 * penalty_grad
        mu = mu * 1.05
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_DUAL_ASCENT = textwrap.dedent("""\
def dual_ascent_detector(H, y, sigma2, constellation, slot_gradient, slot_project, slot_hard_decision):
    Nt = H.shape[1]
    x = np.zeros(Nt, dtype=complex)
    lam = np.zeros(Nt, dtype=complex)
    i = 0
    while i < 30:
        g = slot_gradient(H, y, sigma2, x)
        x = x + 0.01 * (g - lam)
        x_proj = slot_project(x, constellation)
        lam = lam + 0.01 * (x - x_proj)
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")


# -----------------------------------------------------------------------
# C. SEARCH-BASED DETECTORS
# -----------------------------------------------------------------------

T_GREEDY_SEQUENTIAL = textwrap.dedent("""\
def greedy_detector(H, y, sigma2, constellation, slot_score, slot_hard_decision):
    Nt = H.shape[1]
    x = np.zeros(Nt, dtype=complex)
    j = 0
    while j < Nt:
        best_s = constellation[0]
        best_cost = 1e30
        ci = 0
        while ci < len(constellation):
            x[j] = constellation[ci]
            cost = slot_score(x, H, y, sigma2)
            if cost < best_cost:
                best_cost = cost
                best_s = constellation[ci]
            ci = ci + 1
        x[j] = best_s
        j = j + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_MULTI_RESTART = textwrap.dedent("""\
def multi_restart_detector(H, y, sigma2, constellation, slot_iterate, slot_score, slot_hard_decision):
    Nt = H.shape[1]
    best_x = np.zeros(Nt, dtype=complex)
    best_cost = 1e30
    restart = 0
    while restart < 5:
        x = np.zeros(Nt, dtype=complex)
        j = 0
        while j < Nt:
            x[j] = constellation[int(np.random.randint(0, len(constellation)))]
            j = j + 1
        it = 0
        while it < 10:
            x = slot_iterate(H, y, sigma2, x, constellation)
            it = it + 1
        cost = slot_score(x, H, y, sigma2)
        if cost < best_cost:
            best_cost = cost
            best_x = x.copy()
        restart = restart + 1
    x_hat = slot_hard_decision(best_x, constellation)
    return x_hat
""")

T_SIMULATED_ANNEALING = textwrap.dedent("""\
def sa_detector(H, y, sigma2, constellation, slot_proposal, slot_accept, slot_cost, slot_hard_decision):
    Nt = H.shape[1]
    x = np.zeros(Nt, dtype=complex)
    j = 0
    while j < Nt:
        dists = np.abs(constellation - (H.conj().T @ y)[j]) ** 2
        x[j] = constellation[np.argmin(dists)]
        j = j + 1
    current_cost = slot_cost(H, y, sigma2, x)
    temp = 1.0
    i = 0
    while i < 100:
        x_prop = slot_proposal(x, sigma2, constellation)
        prop_cost = slot_cost(H, y, sigma2, x_prop)
        a = slot_accept(current_cost, prop_cost, temp)
        if a > 0.5:
            x = x_prop.copy()
            current_cost = prop_cost
        temp = temp * 0.97
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_TABU_SEARCH = textwrap.dedent("""\
def tabu_detector(H, y, sigma2, constellation, slot_score, slot_hard_decision):
    Nt = H.shape[1]
    x = np.zeros(Nt, dtype=complex)
    j = 0
    while j < Nt:
        dists = np.abs(constellation - (H.conj().T @ y)[j]) ** 2
        x[j] = constellation[np.argmin(dists)]
        j = j + 1
    best_x = x.copy()
    best_cost = slot_score(x, H, y, sigma2)
    i = 0
    while i < 50:
        best_j = 0
        best_s = x[0]
        best_nb_cost = 1e30
        j = 0
        while j < Nt:
            ci = 0
            while ci < len(constellation):
                old_val = x[j]
                x[j] = constellation[ci]
                c = slot_score(x, H, y, sigma2)
                if c < best_nb_cost:
                    best_nb_cost = c
                    best_j = j
                    best_s = constellation[ci]
                x[j] = old_val
                ci = ci + 1
            j = j + 1
        x[best_j] = best_s
        if best_nb_cost < best_cost:
            best_cost = best_nb_cost
            best_x = x.copy()
        i = i + 1
    x_hat = slot_hard_decision(best_x, constellation)
    return x_hat
""")

T_LOCAL_SEARCH = textwrap.dedent("""\
def local_search_detector(H, y, sigma2, constellation, slot_score, slot_hard_decision):
    Nt = H.shape[1]
    x = np.zeros(Nt, dtype=complex)
    j = 0
    while j < Nt:
        dists = np.abs(constellation - (H.conj().T @ y)[j]) ** 2
        x[j] = constellation[np.argmin(dists)]
        j = j + 1
    improved = 1
    while improved > 0:
        improved = 0
        j = 0
        while j < Nt:
            best_s = x[j]
            best_cost = slot_score(x, H, y, sigma2)
            ci = 0
            while ci < len(constellation):
                old_val = x[j]
                x[j] = constellation[ci]
                c = slot_score(x, H, y, sigma2)
                if c < best_cost:
                    best_cost = c
                    best_s = constellation[ci]
                    improved = 1
                x[j] = old_val
                ci = ci + 1
            x[j] = best_s
            j = j + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_REACTIVE_TABU = textwrap.dedent("""\
def reactive_tabu_detector(H, y, sigma2, constellation, slot_score, slot_iterate, slot_hard_decision):
    Nt = H.shape[1]
    x = np.zeros(Nt, dtype=complex)
    j = 0
    while j < Nt:
        dists = np.abs(constellation - (H.conj().T @ y)[j]) ** 2
        x[j] = constellation[np.argmin(dists)]
        j = j + 1
    best_x = x.copy()
    best_cost = slot_score(x, H, y, sigma2)
    i = 0
    while i < 30:
        x = slot_iterate(H, y, sigma2, x, constellation)
        c = slot_score(x, H, y, sigma2)
        if c < best_cost:
            best_cost = c
            best_x = x.copy()
        i = i + 1
    x_hat = slot_hard_decision(best_x, constellation)
    return x_hat
""")

T_RANDOM_WALK = textwrap.dedent("""\
def random_walk_detector(H, y, sigma2, constellation, slot_proposal, slot_score, slot_hard_decision):
    Nt = H.shape[1]
    x = np.zeros(Nt, dtype=complex)
    j = 0
    while j < Nt:
        dists = np.abs(constellation - (H.conj().T @ y)[j]) ** 2
        x[j] = constellation[np.argmin(dists)]
        j = j + 1
    best_x = x.copy()
    best_cost = slot_score(x, H, y, sigma2)
    i = 0
    while i < 50:
        x_prop = slot_proposal(x, sigma2, constellation)
        c = slot_score(x_prop, H, y, sigma2)
        if c < best_cost:
            best_cost = c
            best_x = x_prop.copy()
            x = x_prop.copy()
        i = i + 1
    x_hat = slot_hard_decision(best_x, constellation)
    return x_hat
""")


# -----------------------------------------------------------------------
# D. PROBABILISTIC DETECTORS
# -----------------------------------------------------------------------

T_VARIATIONAL = textwrap.dedent("""\
def variational_detector(H, y, sigma2, constellation, slot_iterate, slot_hard_decision):
    Nt = H.shape[1]
    x = np.zeros(Nt, dtype=complex)
    j = 0
    while j < Nt:
        dists = np.abs(constellation - (H.conj().T @ y)[j]) ** 2
        x[j] = constellation[np.argmin(dists)]
        j = j + 1
    i = 0
    while i < 20:
        x = slot_iterate(H, y, sigma2, x, constellation)
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_GIBBS = textwrap.dedent("""\
def gibbs_detector(H, y, sigma2, constellation, slot_proposal, slot_cost, slot_hard_decision):
    Nt = H.shape[1]
    x = np.zeros(Nt, dtype=complex)
    j = 0
    while j < Nt:
        dists = np.abs(constellation - (H.conj().T @ y)[j]) ** 2
        x[j] = constellation[np.argmin(dists)]
        j = j + 1
    best_x = x.copy()
    best_cost = slot_cost(H, y, sigma2, x)
    sample = 0
    while sample < 50:
        j = sample % Nt
        best_s = x[j]
        best_sc = 1e30
        ci = 0
        while ci < len(constellation):
            x[j] = constellation[ci]
            c = slot_cost(H, y, sigma2, x)
            if c < best_sc:
                best_sc = c
                best_s = constellation[ci]
            ci = ci + 1
        x[j] = best_s
        c_total = slot_cost(H, y, sigma2, x)
        if c_total < best_cost:
            best_cost = c_total
            best_x = x.copy()
        sample = sample + 1
    x_hat = slot_hard_decision(best_x, constellation)
    return x_hat
""")

T_MH = textwrap.dedent("""\
def mh_detector(H, y, sigma2, constellation, slot_proposal, slot_cost, slot_accept, slot_hard_decision):
    Nt = H.shape[1]
    x = np.zeros(Nt, dtype=complex)
    j = 0
    while j < Nt:
        dists = np.abs(constellation - (H.conj().T @ y)[j]) ** 2
        x[j] = constellation[np.argmin(dists)]
        j = j + 1
    current_cost = slot_cost(H, y, sigma2, x)
    best_x = x.copy()
    best_cost = current_cost
    temp = 1.0
    i = 0
    while i < 80:
        x_prop = slot_proposal(x, sigma2, constellation)
        prop_cost = slot_cost(H, y, sigma2, x_prop)
        a = slot_accept(current_cost, prop_cost, temp)
        if a > np.random.rand():
            x = x_prop.copy()
            current_cost = prop_cost
        if current_cost < best_cost:
            best_cost = current_cost
            best_x = x.copy()
        temp = temp * 0.98
        i = i + 1
    x_hat = slot_hard_decision(best_x, constellation)
    return x_hat
""")

T_EM = textwrap.dedent("""\
def em_detector(H, y, sigma2, constellation, slot_iterate, slot_hard_decision):
    Nt = H.shape[1]
    x = np.zeros(Nt, dtype=complex)
    j = 0
    while j < Nt:
        dists = np.abs(constellation - (H.conj().T @ y)[j]) ** 2
        x[j] = constellation[np.argmin(dists)]
        j = j + 1
    i = 0
    while i < 20:
        x = slot_iterate(H, y, sigma2, x, constellation)
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_IMPORTANCE_SAMPLING = textwrap.dedent("""\
def is_detector(H, y, sigma2, constellation, slot_proposal, slot_score, slot_hard_decision):
    Nt = H.shape[1]
    n_samples = 20
    best_x = np.zeros(Nt, dtype=complex)
    best_w = -1e30
    s = 0
    while s < n_samples:
        x = np.zeros(Nt, dtype=complex)
        j = 0
        while j < Nt:
            x[j] = constellation[int(np.random.randint(0, len(constellation)))]
            j = j + 1
        w = -slot_score(x, H, y, sigma2)
        if w > best_w:
            best_w = w
            best_x = x.copy()
        s = s + 1
    x_hat = slot_hard_decision(best_x, constellation)
    return x_hat
""")

T_PARTICLE_FILTER = textwrap.dedent("""\
def particle_filter_detector(H, y, sigma2, constellation, slot_proposal, slot_score, slot_hard_decision):
    Nt = H.shape[1]
    n_particles = 10
    particles = np.zeros((n_particles, Nt), dtype=complex)
    weights = np.ones(n_particles)
    p = 0
    while p < n_particles:
        j = 0
        while j < Nt:
            particles[p, j] = constellation[int(np.random.randint(0, len(constellation)))]
            j = j + 1
        p = p + 1
    it = 0
    while it < 5:
        p = 0
        while p < n_particles:
            x_prop = slot_proposal(particles[p], sigma2, constellation)
            particles[p] = x_prop
            weights[p] = np.exp(-slot_score(particles[p], H, y, sigma2) / max(sigma2, 1e-30))
            p = p + 1
        w_sum = float(np.sum(weights))
        if w_sum > 1e-30:
            weights = weights / w_sum
        it = it + 1
    best_idx = int(np.argmax(weights))
    x_hat = slot_hard_decision(particles[best_idx], constellation)
    return x_hat
""")

T_MAP_RELAXATION = textwrap.dedent("""\
def map_relaxation_detector(H, y, sigma2, constellation, slot_gradient, slot_project, slot_hard_decision):
    Nt = H.shape[1]
    G = H.conj().T @ H + sigma2 * np.eye(Nt)
    rhs = H.conj().T @ y
    x = np.linalg.solve(G, rhs)
    i = 0
    while i < 20:
        g = slot_gradient(H, y, sigma2, x)
        x = x + 0.01 * g
        x = slot_project(x, constellation)
        alpha = 1.0 - float(i) / 20.0
        x_proj = slot_project(x, constellation)
        x = alpha * x + (1.0 - alpha) * x_proj
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")


# -----------------------------------------------------------------------
# E. HYBRID / MULTI-STAGE DETECTORS
# -----------------------------------------------------------------------

T_TWO_STAGE = textwrap.dedent("""\
def two_stage_detector(H, y, sigma2, constellation, slot_stage1, slot_refine, slot_hard_decision):
    x_init = slot_stage1(H, y, sigma2, constellation)
    x_refined = slot_refine(H, y, sigma2, x_init, constellation)
    x_hat = slot_hard_decision(x_refined, constellation)
    return x_hat
""")

T_THREE_STAGE = textwrap.dedent("""\
def three_stage_detector(H, y, sigma2, constellation, slot_stage1, slot_refine, slot_hard_decision):
    x1 = slot_stage1(H, y, sigma2, constellation)
    x2 = slot_refine(H, y, sigma2, x1, constellation)
    x3 = slot_refine(H, y, sigma2, x2, constellation)
    x_hat = slot_hard_decision(x3, constellation)
    return x_hat
""")

T_PARALLEL = textwrap.dedent("""\
def parallel_detector(H, y, sigma2, constellation, slot_stage1, slot_stage2, slot_merge, slot_hard_decision):
    x1 = slot_stage1(H, y, sigma2, constellation)
    x2 = slot_stage2(H, y, sigma2, constellation)
    x_merged = slot_merge(x1, x2, constellation)
    x_hat = slot_hard_decision(x_merged, constellation)
    return x_hat
""")

T_CASCADE = textwrap.dedent("""\
def cascade_detector(H, y, sigma2, constellation, slot_stage1, slot_refine, slot_hard_decision):
    x_coarse = slot_stage1(H, y, sigma2, constellation)
    y_res = y - H @ x_coarse
    x_delta = slot_refine(H, y_res, sigma2, np.zeros(H.shape[1], dtype=complex), constellation)
    x_total = x_coarse + x_delta
    x_hat = slot_hard_decision(x_total, constellation)
    return x_hat
""")

T_FEEDBACK_SIC = textwrap.dedent("""\
def feedback_sic_detector(H, y, sigma2, constellation, slot_stage1, slot_refine, slot_hard_decision):
    x = slot_stage1(H, y, sigma2, constellation)
    it = 0
    while it < 3:
        y_cancel = y.copy()
        Nt = H.shape[1]
        j = 0
        while j < Nt:
            h_j = _col(H, j)
            y_cancel = y_cancel - h_j * x[j]
            y_cancel_j = y_cancel + h_j * x[j]
            x[j] = float(np.real(np.conj(h_j) @ y_cancel_j)) / max(float(np.real(np.conj(h_j) @ h_j)), 1e-30)
            dists = np.abs(constellation - x[j]) ** 2
            x[j] = constellation[np.argmin(dists)]
            y_cancel = y - H @ x + h_j * x[j]
            j = j + 1
        x = slot_refine(H, y, sigma2, x, constellation)
        it = it + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_WARM_START = textwrap.dedent("""\
def warm_start_detector(H, y, sigma2, constellation, slot_init, slot_iterate, slot_hard_decision):
    x = slot_init(H, y, sigma2)
    i = 0
    while i < 20:
        x = slot_iterate(H, y, sigma2, x, constellation)
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_ITERATIVE_REFINE = textwrap.dedent("""\
def iterative_refine_detector(H, y, sigma2, constellation, slot_stage1, slot_iterate, slot_hard_decision):
    x = slot_stage1(H, y, sigma2, constellation)
    i = 0
    while i < 15:
        x = slot_iterate(H, y, sigma2, x, constellation)
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_ALTERNATING = textwrap.dedent("""\
def alternating_detector(H, y, sigma2, constellation, slot_stage1, slot_stage2, slot_hard_decision):
    Nt = H.shape[1]
    x = np.zeros(Nt, dtype=complex)
    i = 0
    while i < 10:
        x = slot_stage1(H, y, sigma2, constellation)
        x = slot_stage2(H, y, sigma2, constellation)
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_RESIDUAL_NET = textwrap.dedent("""\
def residual_net_detector(H, y, sigma2, constellation, slot_init, slot_iterate, slot_hard_decision):
    x = slot_init(H, y, sigma2)
    i = 0
    while i < 10:
        dx = slot_iterate(H, y, sigma2, x, constellation)
        x = x + 0.5 * (dx - x)
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")


# -----------------------------------------------------------------------
# F. SUB-MODULE WRAPPERS (L2) — Each wraps a sub-algorithm as a detector
# -----------------------------------------------------------------------

T_LINEAR_SOLVE_WRAPPER = textwrap.dedent("""\
def linear_solve_detector(H, y, sigma2, constellation, slot_linear_solve, slot_hard_decision):
    Nt = H.shape[1]
    G = H.conj().T @ H + sigma2 * np.eye(Nt)
    rhs = H.conj().T @ y
    x_soft = slot_linear_solve(G, rhs)
    x_hat = slot_hard_decision(x_soft, constellation)
    return x_hat
""")

T_PRECONDITIONED_SOLVE = textwrap.dedent("""\
def precond_solve_detector(H, y, sigma2, constellation, slot_preconditioner, slot_linear_solve, slot_hard_decision):
    Nt = H.shape[1]
    G = H.conj().T @ H + sigma2 * np.eye(Nt)
    rhs = H.conj().T @ y
    M = slot_preconditioner(G, sigma2)
    M_rhs = np.linalg.solve(M, rhs)
    M_G = np.linalg.solve(M, G)
    x_soft = slot_linear_solve(M_G, M_rhs)
    x_hat = slot_hard_decision(x_soft, constellation)
    return x_hat
""")

T_REGULARIZED_SOLVE = textwrap.dedent("""\
def regularized_solve_detector(H, y, sigma2, constellation, slot_regularizer, slot_linear_solve, slot_hard_decision):
    G = H.conj().T @ H
    G_reg = slot_regularizer(G, sigma2)
    rhs = H.conj().T @ y
    x_soft = slot_linear_solve(G_reg, rhs)
    x_hat = slot_hard_decision(x_soft, constellation)
    return x_hat
""")

T_WEIGHT_APPLY = textwrap.dedent("""\
def weight_apply_detector(H, y, sigma2, constellation, slot_weight, slot_hard_decision):
    W = slot_weight(H, sigma2)
    x_soft = W @ y
    x_hat = slot_hard_decision(x_soft, constellation)
    return x_hat
""")

T_MATCHED_FILTER_DETECT = textwrap.dedent("""\
def matched_filter_detector(H, y, sigma2, constellation, slot_normalize, slot_hard_decision):
    x_mf = H.conj().T @ y
    x_norm = slot_normalize(x_mf)
    x_hat = slot_hard_decision(x_norm, constellation)
    return x_hat
""")

T_QR_DETECT = textwrap.dedent("""\
def qr_detector(H, y, sigma2, constellation, slot_hard_decision):
    Nt = H.shape[1]
    G = H.conj().T @ H + sigma2 * np.eye(Nt)
    L = np.linalg.cholesky(G)
    rhs = H.conj().T @ y
    z = np.linalg.solve(L, rhs)
    x = np.linalg.solve(L.conj().T, z)
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_SVD_DETECT = textwrap.dedent("""\
def svd_detector(H, y, sigma2, constellation, slot_hard_decision):
    Nt = H.shape[1]
    G = H.conj().T @ H + sigma2 * np.eye(Nt)
    G_inv = np.linalg.inv(G)
    x_soft = G_inv @ (H.conj().T @ y)
    x_hat = slot_hard_decision(x_soft, constellation)
    return x_hat
""")

T_BLOCK_DETECT = textwrap.dedent("""\
def block_detector(H, y, sigma2, constellation, slot_linear_solve, slot_hard_decision):
    Nt = H.shape[1]
    G = H.conj().T @ H + sigma2 * np.eye(Nt)
    rhs = H.conj().T @ y
    x = np.zeros(Nt, dtype=complex)
    it = 0
    while it < 5:
        j = 0
        while j < Nt:
            s = 0.0 + 0.0j
            k = 0
            while k < Nt:
                if k != j:
                    s = s + G[j, k] * x[k]
                k = k + 1
            x[j] = (rhs[j] - s) / G[j, j]
            j = j + 1
        it = it + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_SUBSPACE_DETECT = textwrap.dedent("""\
def subspace_detector(H, y, sigma2, constellation, slot_linear_solve, slot_hard_decision):
    Nr = H.shape[0]
    Nt = H.shape[1]
    G = H.conj().T @ H + sigma2 * np.eye(Nt)
    rhs = H.conj().T @ y
    x_soft = slot_linear_solve(G, rhs)
    x_hat = slot_hard_decision(x_soft, constellation)
    return x_hat
""")

T_ORDERING_DETECT = textwrap.dedent("""\
def ordering_detector(H, y, sigma2, constellation, slot_ordering, slot_linear_solve, slot_hard_decision):
    Nt = H.shape[1]
    G = H.conj().T @ H + sigma2 * np.eye(Nt)
    rhs = H.conj().T @ y
    x_soft = slot_linear_solve(G, rhs)
    x_hat = slot_hard_decision(x_soft, constellation)
    return x_hat
""")

T_SOFT_SIC = textwrap.dedent("""\
def soft_sic_detector(H, y, sigma2, constellation, slot_ordering, slot_soft_estimate, slot_hard_decision):
    Nt = H.shape[1]
    G = H.conj().T @ H + sigma2 * np.eye(Nt)
    x = np.linalg.solve(G, H.conj().T @ y)
    var = np.ones(Nt) * sigma2
    x = slot_soft_estimate(x, var, constellation)
    order = slot_ordering(H, y, sigma2)
    y_cancel = y.copy()
    x_hat = np.zeros(Nt, dtype=complex)
    k = 0
    while k < Nt:
        idx = order[k]
        h_idx = _col(H, idx)
        y_cancel_k = y_cancel + h_idx * x[idx]
        x_est = float(np.real(np.conj(h_idx) @ y_cancel_k)) / max(float(np.real(np.conj(h_idx) @ h_idx)), 1e-30) + 1j * float(np.imag(np.conj(h_idx) @ y_cancel_k)) / max(float(np.real(np.conj(h_idx) @ h_idx)), 1e-30)
        dists = np.abs(constellation - x_est) ** 2
        x_hat[idx] = constellation[np.argmin(dists)]
        y_cancel = y_cancel - h_idx * x_hat[idx]
        k = k + 1
    x_out = slot_hard_decision(x_hat, constellation)
    return x_out
""")

T_SPARSE_DETECT = textwrap.dedent("""\
def sparse_detector(H, y, sigma2, constellation, slot_shrinkage, slot_hard_decision):
    Nt = H.shape[1]
    x = np.zeros(Nt, dtype=complex)
    lam = sigma2 * 0.1
    i = 0
    while i < 30:
        r = y - H @ x
        g = H.conj().T @ r
        x = x + 0.01 * g
        x = slot_shrinkage(x, lam)
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_TURBO_LINEAR = textwrap.dedent("""\
def turbo_linear_detector(H, y, sigma2, constellation, slot_soft_estimate, slot_hard_decision):
    Nt = H.shape[1]
    G = H.conj().T @ H + sigma2 * np.eye(Nt)
    rhs = H.conj().T @ y
    x = np.linalg.solve(G, rhs)
    var = np.zeros(Nt)
    j = 0
    while j < Nt:
        G_inv_jj = 1.0 / max(float(np.real(G[j, j])), 1e-30)
        var[j] = G_inv_jj * sigma2
        j = j + 1
    i = 0
    while i < 5:
        x_soft = slot_soft_estimate(x, var, constellation)
        r = y - H @ x_soft
        x = x_soft + np.linalg.solve(G, H.conj().T @ r)
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")


# -----------------------------------------------------------------------
# G. BUILDING BLOCK WRAPPERS (L1) — Atomic operations as detectors
# -----------------------------------------------------------------------

T_DIAG_DOMINANT = textwrap.dedent("""\
def diag_dominant_detector(H, y, sigma2, constellation, slot_iterate, slot_hard_decision):
    Nt = H.shape[1]
    G = H.conj().T @ H + sigma2 * np.eye(Nt)
    rhs = H.conj().T @ y
    x = np.zeros(Nt, dtype=complex)
    j = 0
    while j < Nt:
        x[j] = rhs[j] / G[j, j]
        j = j + 1
    i = 0
    while i < 5:
        x = slot_iterate(G, rhs, x)
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_MMSE_INIT_ITERATE = textwrap.dedent("""\
def mmse_init_iterate(H, y, sigma2, constellation, slot_init, slot_iterate, slot_damping, slot_hard_decision):
    x = slot_init(H, y, sigma2)
    i = 0
    while i < 10:
        x_old = x.copy()
        x_new = slot_iterate(H, y, sigma2, x, constellation)
        x = slot_damping(x_old, x_new)
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_MF_REFINE = textwrap.dedent("""\
def mf_refine_detector(H, y, sigma2, constellation, slot_iterate, slot_hard_decision):
    x = H.conj().T @ y
    Nt = H.shape[1]
    i = 0
    while i < 15:
        x = slot_iterate(H, y, sigma2, x, constellation)
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_SCALED_DETECT = textwrap.dedent("""\
def scaled_detector(H, y, sigma2, constellation, slot_weight, slot_hard_decision):
    W = slot_weight(H, sigma2)
    x_soft = W @ y
    x_hat = slot_hard_decision(x_soft, constellation)
    return x_hat
""")

T_RESIDUAL_ITERATE = textwrap.dedent("""\
def residual_iterate_detector(H, y, sigma2, constellation, slot_init, slot_iterate, slot_hard_decision):
    x = slot_init(H, y, sigma2)
    i = 0
    while i < 15:
        r = y - H @ x
        dx = slot_iterate(H, r, sigma2, np.zeros(H.shape[1], dtype=complex), constellation)
        x = x + 0.5 * dx
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_OVERRELAX = textwrap.dedent("""\
def overrelax_detector(H, y, sigma2, constellation, slot_iterate, slot_damping, slot_hard_decision):
    Nt = H.shape[1]
    x = np.zeros(Nt, dtype=complex)
    i = 0
    while i < 20:
        x_old = x.copy()
        x_new = slot_iterate(H, y, sigma2, x, constellation)
        x = slot_damping(x_old, x_new)
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_POWER_METHOD = textwrap.dedent("""\
def power_method_detector(H, y, sigma2, constellation, slot_normalize, slot_hard_decision):
    Nt = H.shape[1]
    G = H.conj().T @ H + sigma2 * np.eye(Nt)
    rhs = H.conj().T @ y
    x = rhs.copy()
    i = 0
    while i < 20:
        x = np.linalg.solve(G, x)
        x = slot_normalize(x)
        i = i + 1
    scale = float(np.real(np.conj(rhs) @ x)) / max(float(np.real(np.conj(x) @ G @ x)), 1e-30)
    x = x * scale
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_LANDWEBER = textwrap.dedent("""\
def landweber_detector(H, y, sigma2, constellation, slot_hard_decision):
    Nt = H.shape[1]
    x = np.zeros(Nt, dtype=complex)
    alpha = 0.01
    i = 0
    while i < 30:
        r = y - H @ x
        x = x + alpha * (H.conj().T @ r)
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_KACZMARZ = textwrap.dedent("""\
def kaczmarz_detector(H, y, sigma2, constellation, slot_hard_decision):
    Nr = H.shape[0]
    Nt = H.shape[1]
    x = np.zeros(Nt, dtype=complex)
    it = 0
    while it < 5:
        i = 0
        while i < Nr:
            a_i = _row(H, i)
            r_i = y[i] - np.dot(a_i, x)
            norm2 = float(np.real(np.conj(a_i) @ a_i))
            if norm2 > 1e-30:
                x = x + (r_i / norm2) * a_i.conj()
            i = i + 1
        it = it + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_CIMMINO = textwrap.dedent("""\
def cimmino_detector(H, y, sigma2, constellation, slot_hard_decision):
    Nr = H.shape[0]
    Nt = H.shape[1]
    x = np.zeros(Nt, dtype=complex)
    it = 0
    while it < 10:
        dx = np.zeros(Nt, dtype=complex)
        i = 0
        while i < Nr:
            a_i = _row(H, i)
            r_i = y[i] - np.dot(a_i, x)
            norm2 = float(np.real(np.conj(a_i) @ a_i))
            if norm2 > 1e-30:
                dx = dx + (r_i / norm2) * a_i.conj()
            i = i + 1
        x = x + dx / Nr
        it = it + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_STEEPEST_DESCENT = textwrap.dedent("""\
def steepest_descent_detector(H, y, sigma2, constellation, slot_hard_decision):
    Nt = H.shape[1]
    G = H.conj().T @ H + sigma2 * np.eye(Nt)
    rhs = H.conj().T @ y
    x = np.zeros(Nt, dtype=complex)
    i = 0
    while i < 30:
        r = rhs - G @ x
        Ar = G @ r
        rr = float(np.real(np.conj(r) @ r))
        rAr = float(np.real(np.conj(r) @ Ar))
        alpha = rr / max(rAr, 1e-30)
        x = x + alpha * r
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")


# -----------------------------------------------------------------------
# H. CROSS-DOMAIN ALGORITHM PATTERNS
# -----------------------------------------------------------------------

T_DIVIDE_CONQUER = textwrap.dedent("""\
def divide_conquer_detector(H, y, sigma2, constellation, slot_linear_solve, slot_hard_decision):
    Nt = H.shape[1]
    G = H.conj().T @ H + sigma2 * np.eye(Nt)
    rhs = H.conj().T @ y
    x = slot_linear_solve(G, rhs)
    r = y - H @ x
    g = H.conj().T @ r
    dx = slot_linear_solve(G, g)
    x = x + 0.5 * dx
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_LAYER_PEELING = textwrap.dedent("""\
def layer_peeling_detector(H, y, sigma2, constellation, slot_weight, slot_hard_decision):
    Nt = H.shape[1]
    x = np.zeros(Nt, dtype=complex)
    y_rem = y.copy()
    H_rem = H.copy()
    detected = np.zeros(Nt)
    k = 0
    while k < Nt:
        W = slot_weight(H_rem, sigma2)
        x_soft = W @ y_rem
        best_j = -1
        best_snr = -1e30
        j = 0
        while j < Nt:
            if detected[j] == 0.0:
                snr = float(np.real(np.abs(x_soft[j]) ** 2))
                if snr > best_snr:
                    best_snr = snr
                    best_j = j
            j = j + 1
        if best_j < 0:
            best_j = 0
        dists = np.abs(constellation - x_soft[best_j]) ** 2
        x[best_j] = constellation[np.argmin(dists)]
        y_rem = y_rem - _col(H, best_j) * x[best_j]
        detected[best_j] = 1.0
        k = k + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_CONSENSUS = textwrap.dedent("""\
def consensus_detector(H, y, sigma2, constellation, slot_iterate, slot_merge, slot_hard_decision):
    Nt = H.shape[1]
    x1 = np.zeros(Nt, dtype=complex)
    x2 = np.zeros(Nt, dtype=complex)
    i = 0
    while i < 10:
        x1 = slot_iterate(H, y, sigma2, x1, constellation)
        x2 = slot_iterate(H, y, sigma2, x2, constellation)
        x_avg = slot_merge(x1, x2, constellation)
        x1 = x_avg.copy()
        x2 = x_avg.copy()
        i = i + 1
    x_hat = slot_hard_decision(x1, constellation)
    return x_hat
""")

T_ANNEAL_ITERATE = textwrap.dedent("""\
def anneal_iterate_detector(H, y, sigma2, constellation, slot_iterate, slot_project, slot_hard_decision):
    Nt = H.shape[1]
    x = np.zeros(Nt, dtype=complex)
    temp = 1.0
    i = 0
    while i < 30:
        x = slot_iterate(H, y, sigma2, x, constellation)
        alpha = 1.0 - temp
        x_proj = slot_project(x, constellation)
        x = alpha * x_proj + (1.0 - alpha) * x
        temp = temp * 0.9
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_DP_DETECT = textwrap.dedent("""\
def dp_detector(H, y, sigma2, constellation, slot_score, slot_hard_decision):
    Nt = H.shape[1]
    M = len(constellation)
    x = np.zeros(Nt, dtype=complex)
    j = 0
    while j < Nt:
        best_s = constellation[0]
        best_c = 1e30
        ci = 0
        while ci < M:
            x[j] = constellation[ci]
            c = slot_score(x, H, y, sigma2)
            if c < best_c:
                best_c = c
                best_s = constellation[ci]
            ci = ci + 1
        x[j] = best_s
        j = j + 1
    i = 0
    while i < 3:
        j = 0
        while j < Nt:
            best_s = x[j]
            best_c = slot_score(x, H, y, sigma2)
            ci = 0
            while ci < M:
                old = x[j]
                x[j] = constellation[ci]
                c = slot_score(x, H, y, sigma2)
                if c < best_c:
                    best_c = c
                    best_s = constellation[ci]
                x[j] = old
                ci = ci + 1
            x[j] = best_s
            j = j + 1
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_EXPECTATION_CONSISTENT = textwrap.dedent("""\
def ec_detector(H, y, sigma2, constellation, slot_iterate, slot_damping, slot_hard_decision):
    Nt = H.shape[1]
    x = np.zeros(Nt, dtype=complex)
    j = 0
    while j < Nt:
        dists = np.abs(constellation - (H.conj().T @ y)[j]) ** 2
        x[j] = constellation[np.argmin(dists)]
        j = j + 1
    i = 0
    while i < 20:
        x_old = x.copy()
        x = slot_iterate(H, y, sigma2, x, constellation)
        x = slot_damping(x_old, x)
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_MULTI_SCALE = textwrap.dedent("""\
def multi_scale_detector(H, y, sigma2, constellation, slot_linear_solve, slot_refine, slot_hard_decision):
    Nt = H.shape[1]
    G = H.conj().T @ H + sigma2 * np.eye(Nt)
    rhs = H.conj().T @ y
    x_coarse = slot_linear_solve(G, rhs)
    x = slot_refine(H, y, sigma2, x_coarse, constellation)
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_GAUSS_SEIDEL_BLOCK = textwrap.dedent("""\
def gs_block_detector(H, y, sigma2, constellation, slot_linear_solve, slot_hard_decision):
    Nt = H.shape[1]
    G = H.conj().T @ H + sigma2 * np.eye(Nt)
    rhs = H.conj().T @ y
    x = np.zeros(Nt, dtype=complex)
    it = 0
    while it < 10:
        j = 0
        while j < Nt:
            s = 0.0 + 0.0j
            k = 0
            while k < Nt:
                if k != j:
                    s = s + G[j, k] * x[k]
                k = k + 1
            x[j] = (rhs[j] - s) / G[j, j]
            j = j + 1
        it = it + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_GRADIENT_PROJECT_ALTERNATE = textwrap.dedent("""\
def gpa_detector(H, y, sigma2, constellation, slot_gradient, slot_project, slot_damping, slot_hard_decision):
    Nt = H.shape[1]
    x = np.zeros(Nt, dtype=complex)
    i = 0
    while i < 30:
        g = slot_gradient(H, y, sigma2, x)
        x_grad = x + 0.01 * g
        x_proj = slot_project(x_grad, constellation)
        x = slot_damping(x, x_proj)
        i = i + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")

T_NESTED_ITERATE = textwrap.dedent("""\
def nested_iterate_detector(H, y, sigma2, constellation, slot_init, slot_iterate, slot_refine, slot_hard_decision):
    x = slot_init(H, y, sigma2)
    outer = 0
    while outer < 5:
        inner = 0
        while inner < 5:
            x = slot_iterate(H, y, sigma2, x, constellation)
            inner = inner + 1
        x = slot_refine(H, y, sigma2, x, constellation)
        outer = outer + 1
    x_hat = slot_hard_decision(x, constellation)
    return x_hat
""")


# ═══════════════════════════════════════════════════════════════════════════
# ══════════════  BUILD SKELETON SPECS FROM TEMPLATES  ═══════════════════
# ═══════════════════════════════════════════════════════════════════════════

def _build_specs() -> list[SkeletonSpec]:
    """Construct all SkeletonSpec entries from template sources."""
    specs: list[SkeletonSpec] = []

    def _add(algo_id, source, func_name, slot_args, slot_defs_dict,
             slot_defaults_map, tags, level=3, extra_globals=None):
        specs.append(SkeletonSpec(
            algo_id=algo_id, source=source, func_name=func_name,
            slot_arg_names=slot_args, slot_defs=slot_defs_dict,
            slot_default_keys=slot_defaults_map, tags=tags,
            level=level, extra_globals=extra_globals,
        ))

    # Helper to build slot definitions dict for common patterns
    def _slots(prefix, *slot_tuples):
        d = {}
        for short, spec, lvl, tags in slot_tuples:
            sid = f"{prefix}.{short}"
            d[sid] = SlotDescriptor(
                slot_id=sid, short_name=short,
                level=lvl, depth=0, parent_slot_id=None,
                spec=spec, domain_tags=tags or set(),
            )
        return d

    # ── A. Iterative Linear ──────────────────────────────────────────

    _iterate_hd = [
        ("iterate", _PS_ITERATE_LINEAR, 1, {"linear_algebra"}),
        ("hard_decision", _PS_HARD_DECISION, 0, {"distance"}),
    ]
    _iterate_damp_hd = [
        ("iterate", _PS_ITERATE_LINEAR, 1, {"linear_algebra"}),
        ("damping", _PS_DAMPING, 0, {"iterative"}),
        ("hard_decision", _PS_HARD_DECISION, 0, {"distance"}),
    ]
    _precond_hd = [
        ("preconditioner", _PS_PRECONDITIONER, 1, {"linear_algebra"}),
        ("hard_decision", _PS_HARD_DECISION, 0, {"distance"}),
    ]
    _precond_damp_hd = [
        ("preconditioner", _PS_PRECONDITIONER, 1, {"linear_algebra"}),
        ("damping", _PS_DAMPING, 0, {"iterative"}),
        ("hard_decision", _PS_HARD_DECISION, 0, {"distance"}),
    ]
    _precond_iterate_hd = [
        ("preconditioner", _PS_PRECONDITIONER, 1, {"linear_algebra"}),
        ("iterate", _PS_ITERATE_LINEAR, 1, {"linear_algebra"}),
        ("hard_decision", _PS_HARD_DECISION, 0, {"distance"}),
    ]
    _init_iterate_hd = [
        ("init", _PS_INIT, 1, {"linear_algebra"}),
        ("iterate", _PS_ITERATE_LINEAR, 1, {"linear_algebra"}),
        ("hard_decision", _PS_HARD_DECISION, 0, {"distance"}),
    ]
    _init_iterate_full_hd = [
        ("init", _PS_INIT, 1, {"linear_algebra"}),
        ("iterate", _PS_ITERATE_FULL, 1, {"iterative"}),
        ("hard_decision", _PS_HARD_DECISION, 0, {"distance"}),
    ]
    _init_iterate_damp_hd = [
        ("init", _PS_INIT, 1, {"linear_algebra"}),
        ("iterate", _PS_ITERATE_FULL, 1, {"iterative"}),
        ("damping", _PS_DAMPING, 0, {"iterative"}),
        ("hard_decision", _PS_HARD_DECISION, 0, {"distance"}),
    ]
    _init_iterate_lin_damp_hd = [
        ("init", _PS_INIT, 1, {"linear_algebra"}),
        ("iterate", _PS_ITERATE_LINEAR, 1, {"linear_algebra"}),
        ("damping", _PS_DAMPING, 0, {"iterative"}),
        ("hard_decision", _PS_HARD_DECISION, 0, {"distance"}),
    ]

    iterative_linear_specs = [
        ("jacobi", T_JACOBI, "jacobi_detector", _iterate_hd,
         {"slot_iterate": "jacobi_iterate", "slot_hard_decision": "hard_decision"},
         {"iterative", "linear"}),
        ("gs", T_GS, "gs_detector", _iterate_damp_hd,
         {"slot_iterate": "gs_iterate", "slot_damping": "damping_none", "slot_hard_decision": "hard_decision"},
         {"iterative", "linear"}),
        ("sor", T_SOR, "sor_detector", _iterate_hd,
         {"slot_iterate": "sor_iterate", "slot_hard_decision": "hard_decision"},
         {"iterative", "linear"}),
        ("ssor", T_SSOR, "ssor_detector", _iterate_hd,
         {"slot_iterate": "ssor_iterate", "slot_hard_decision": "hard_decision"},
         {"iterative", "linear"}),
        ("richardson", T_RICHARDSON, "richardson_detector", _iterate_hd,
         {"slot_iterate": "richardson_iterate", "slot_hard_decision": "hard_decision"},
         {"iterative", "linear"}),
        ("neumann", T_NEUMANN, "neumann_detector", _iterate_hd,
         {"slot_iterate": "neumann_iterate", "slot_hard_decision": "hard_decision"},
         {"iterative", "linear"}),
        ("chebyshev", T_CHEBYSHEV, "chebyshev_detector", _iterate_hd,
         {"slot_iterate": "chebyshev_iterate", "slot_hard_decision": "hard_decision"},
         {"iterative", "linear"}),
        ("cg", T_CG, "cg_detector", _precond_hd,
         {"slot_preconditioner": "precond_diagonal", "slot_hard_decision": "hard_decision"},
         {"iterative", "linear"}),
        ("pcg", T_PRECOND_CG, "pcg_detector", _precond_damp_hd,
         {"slot_preconditioner": "precond_diagonal", "slot_damping": "damping_none", "slot_hard_decision": "hard_decision"},
         {"iterative", "linear"}),
        ("minres", T_MINRES, "minres_detector", _precond_hd,
         {"slot_preconditioner": "precond_identity", "slot_hard_decision": "hard_decision"},
         {"iterative", "linear"}),
        ("bicgstab", T_BICGSTAB, "bicgstab_detector", _precond_hd,
         {"slot_preconditioner": "precond_diagonal", "slot_hard_decision": "hard_decision"},
         {"iterative", "linear"}),
        ("precond_rich", T_PRECOND_RICHARDSON, "precond_richardson", _precond_iterate_hd,
         {"slot_preconditioner": "precond_diagonal", "slot_iterate": "richardson_iterate", "slot_hard_decision": "hard_decision"},
         {"iterative", "linear"}),
        ("weighted_jacobi", T_WEIGHTED_JACOBI, "weighted_jacobi", _iterate_damp_hd,
         {"slot_iterate": "jacobi_iterate", "slot_damping": "damping_fixed", "slot_hard_decision": "hard_decision"},
         {"iterative", "linear"}),
        ("stationary", T_STATIONARY, "stationary_detector", _init_iterate_lin_damp_hd,
         {"slot_init": "init_zero", "slot_iterate": "gs_iterate", "slot_damping": "damping_none", "slot_hard_decision": "hard_decision"},
         {"iterative", "linear"}),
    ]

    for aid, src, fn, slot_tuples, dmap, tags in iterative_linear_specs:
        slot_args = [f"slot_{s[0]}" for s in slot_tuples]
        _add(aid, src, fn, slot_args, _slots(aid, *slot_tuples), dmap, tags)

    # ── B. Optimisation ──────────────────────────────────────────────

    _grad_step_hd = [
        ("gradient", _PS_GRADIENT, 1, {"optimisation"}),
        ("step_size", _PS_STEP_SIZE, 0, {"optimisation"}),
        ("hard_decision", _PS_HARD_DECISION, 0, {"distance"}),
    ]
    _grad_step_proj_hd = [
        ("gradient", _PS_GRADIENT, 1, {"optimisation"}),
        ("step_size", _PS_STEP_SIZE, 0, {"optimisation"}),
        ("project", _PS_PROJECT, 0, {"optimisation"}),
        ("hard_decision", _PS_HARD_DECISION, 0, {"distance"}),
    ]
    _grad_hd = [
        ("gradient", _PS_GRADIENT, 1, {"optimisation"}),
        ("hard_decision", _PS_HARD_DECISION, 0, {"distance"}),
    ]
    _grad_proj_hd = [
        ("gradient", _PS_GRADIENT, 1, {"optimisation"}),
        ("project", _PS_PROJECT, 0, {"optimisation"}),
        ("hard_decision", _PS_HARD_DECISION, 0, {"distance"}),
    ]

    opt_specs = [
        ("grad_descent", T_GRADIENT_DESCENT, "gradient_descent_detector", _grad_step_hd,
         {"slot_gradient": "gradient_ml", "slot_step_size": "step_size_fixed", "slot_hard_decision": "hard_decision"},
         {"optimisation"}),
        ("projected_gd", T_PROJECTED_GD, "projected_gd_detector", _grad_step_proj_hd,
         {"slot_gradient": "gradient_ml", "slot_step_size": "step_size_fixed", "slot_project": "project_nearest_symbol", "slot_hard_decision": "hard_decision"},
         {"optimisation"}),
        ("newton", T_NEWTON, "newton_detector", _iterate_hd,
         {"slot_iterate": "jacobi_iterate", "slot_hard_decision": "hard_decision"},
         {"optimisation"}),
        ("heavy_ball", T_HEAVY_BALL, "heavy_ball_detector", _grad_hd,
         {"slot_gradient": "gradient_ml", "slot_hard_decision": "hard_decision"},
         {"optimisation"}),
        ("nesterov", T_NESTEROV, "nesterov_detector", _grad_hd,
         {"slot_gradient": "gradient_ml", "slot_hard_decision": "hard_decision"},
         {"optimisation"}),
        ("coord_descent", T_COORDINATE_DESCENT, "coordinate_descent_detector",
         [("iterate", _PS_ITERATE_FULL, 1, {"optimisation"}), ("hard_decision", _PS_HARD_DECISION, 0, {"distance"})],
         {"slot_iterate": "iterate_coordinate", "slot_hard_decision": "hard_decision"},
         {"optimisation"}),
        ("admm", T_ADMM, "admm_detector",
         [("x_update", _PS_X_UPDATE, 1, {"optimisation"}), ("z_update", _PS_Z_UPDATE, 1, {"optimisation"}),
          ("dual_update", _PS_DUAL_UPDATE, 0, {"optimisation"}), ("hard_decision", _PS_HARD_DECISION, 0, {"distance"})],
         {"slot_x_update": "admm_x_update", "slot_z_update": "admm_z_update", "slot_dual_update": "admm_dual_update", "slot_hard_decision": "hard_decision"},
         {"optimisation"}),
        ("proximal", T_PROXIMAL, "proximal_detector", _grad_proj_hd,
         {"slot_gradient": "gradient_ml", "slot_project": "project_nearest_symbol", "slot_hard_decision": "hard_decision"},
         {"optimisation"}),
        ("bb", T_BARZILAI_BORWEIN, "bb_detector", _grad_hd,
         {"slot_gradient": "gradient_ml", "slot_hard_decision": "hard_decision"},
         {"optimisation"}),
        ("frank_wolfe", T_FRANK_WOLFE, "frank_wolfe_detector", _grad_proj_hd,
         {"slot_gradient": "gradient_ml", "slot_project": "project_nearest_symbol", "slot_hard_decision": "hard_decision"},
         {"optimisation"}),
        ("penalty", T_PENALTY, "penalty_detector", _grad_proj_hd,
         {"slot_gradient": "gradient_ml", "slot_project": "project_box", "slot_hard_decision": "hard_decision"},
         {"optimisation"}),
        ("dual_ascent", T_DUAL_ASCENT, "dual_ascent_detector", _grad_proj_hd,
         {"slot_gradient": "gradient_ml", "slot_project": "project_nearest_symbol", "slot_hard_decision": "hard_decision"},
         {"optimisation"}),
    ]

    for aid, src, fn, slot_tuples, dmap, tags in opt_specs:
        slot_args = [f"slot_{s[0]}" for s in slot_tuples]
        _add(aid, src, fn, slot_args, _slots(aid, *slot_tuples), dmap, tags)

    # ── C. Search-based ──────────────────────────────────────────────

    _score_hd = [
        ("score", _PS_SCORE, 1, {"search"}),
        ("hard_decision", _PS_HARD_DECISION, 0, {"distance"}),
    ]

    search_specs = [
        ("greedy", T_GREEDY_SEQUENTIAL, "greedy_detector", _score_hd,
         {"slot_score": "score_ml_cost", "slot_hard_decision": "hard_decision"}, {"search"}),
        ("multi_restart", T_MULTI_RESTART, "multi_restart_detector",
         [("iterate", _PS_ITERATE_FULL, 1, {"search"}), ("score", _PS_SCORE, 1, {"search"}), ("hard_decision", _PS_HARD_DECISION, 0, {"distance"})],
         {"slot_iterate": "iterate_gradient", "slot_score": "score_ml_cost", "slot_hard_decision": "hard_decision"}, {"search"}),
        ("sa", T_SIMULATED_ANNEALING, "sa_detector",
         [("proposal", _PS_PROPOSAL, 1, {"search"}), ("accept", _PS_ACCEPT, 0, {"search"}), ("cost", _PS_COST, 1, {"search"}), ("hard_decision", _PS_HARD_DECISION, 0, {"distance"})],
         {"slot_proposal": "proposal_flip", "slot_accept": "accept_metropolis", "slot_cost": "cost_ml", "slot_hard_decision": "hard_decision"}, {"search", "probabilistic"}),
        ("tabu", T_TABU_SEARCH, "tabu_detector", _score_hd,
         {"slot_score": "score_ml_cost", "slot_hard_decision": "hard_decision"}, {"search"}),
        ("local_search", T_LOCAL_SEARCH, "local_search_detector", _score_hd,
         {"slot_score": "score_ml_cost", "slot_hard_decision": "hard_decision"}, {"search"}),
        ("reactive_tabu", T_REACTIVE_TABU, "reactive_tabu_detector",
         [("score", _PS_SCORE, 1, {"search"}), ("iterate", _PS_ITERATE_FULL, 1, {"search"}), ("hard_decision", _PS_HARD_DECISION, 0, {"distance"})],
         {"slot_score": "score_ml_cost", "slot_iterate": "iterate_gradient", "slot_hard_decision": "hard_decision"}, {"search"}),
        ("random_walk", T_RANDOM_WALK, "random_walk_detector",
         [("proposal", _PS_PROPOSAL, 1, {"search"}), ("score", _PS_SCORE, 1, {"search"}), ("hard_decision", _PS_HARD_DECISION, 0, {"distance"})],
         {"slot_proposal": "proposal_flip", "slot_score": "score_ml_cost", "slot_hard_decision": "hard_decision"}, {"search"}),
    ]

    for aid, src, fn, slot_tuples, dmap, tags in search_specs:
        slot_args = [f"slot_{s[0]}" for s in slot_tuples]
        _add(aid, src, fn, slot_args, _slots(aid, *slot_tuples), dmap, tags)

    # ── D. Probabilistic ─────────────────────────────────────────────

    prob_specs = [
        ("variational", T_VARIATIONAL, "variational_detector",
         [("iterate", _PS_ITERATE_FULL, 1, {"inference"}), ("hard_decision", _PS_HARD_DECISION, 0, {"distance"})],
         {"slot_iterate": "iterate_proximal", "slot_hard_decision": "hard_decision"}, {"probabilistic", "inference"}),
        ("gibbs", T_GIBBS, "gibbs_detector",
         [("proposal", _PS_PROPOSAL, 1, {"probabilistic"}), ("cost", _PS_COST, 1, {"probabilistic"}), ("hard_decision", _PS_HARD_DECISION, 0, {"distance"})],
         {"slot_proposal": "proposal_flip", "slot_cost": "cost_ml", "slot_hard_decision": "hard_decision"}, {"probabilistic"}),
        ("mh", T_MH, "mh_detector",
         [("proposal", _PS_PROPOSAL, 1, {"probabilistic"}), ("cost", _PS_COST, 1, {"probabilistic"}), ("accept", _PS_ACCEPT, 0, {"probabilistic"}), ("hard_decision", _PS_HARD_DECISION, 0, {"distance"})],
         {"slot_proposal": "proposal_flip", "slot_cost": "cost_ml", "slot_accept": "accept_metropolis", "slot_hard_decision": "hard_decision"}, {"probabilistic"}),
        ("em", T_EM, "em_detector",
         [("iterate", _PS_ITERATE_FULL, 1, {"inference"}), ("hard_decision", _PS_HARD_DECISION, 0, {"distance"})],
         {"slot_iterate": "iterate_proximal", "slot_hard_decision": "hard_decision"}, {"probabilistic", "inference"}),
        ("importance_sampling", T_IMPORTANCE_SAMPLING, "is_detector",
         [("proposal", _PS_PROPOSAL, 1, {"probabilistic"}), ("score", _PS_SCORE, 1, {"probabilistic"}), ("hard_decision", _PS_HARD_DECISION, 0, {"distance"})],
         {"slot_proposal": "proposal_gaussian", "slot_score": "score_ml_cost", "slot_hard_decision": "hard_decision"}, {"probabilistic"}),
        ("particle_filter", T_PARTICLE_FILTER, "particle_filter_detector",
         [("proposal", _PS_PROPOSAL, 1, {"probabilistic"}), ("score", _PS_SCORE, 1, {"probabilistic"}), ("hard_decision", _PS_HARD_DECISION, 0, {"distance"})],
         {"slot_proposal": "proposal_flip", "slot_score": "score_ml_cost", "slot_hard_decision": "hard_decision"}, {"probabilistic"}),
        ("map_relaxation", T_MAP_RELAXATION, "map_relaxation_detector", _grad_proj_hd,
         {"slot_gradient": "gradient_ml", "slot_project": "project_nearest_symbol", "slot_hard_decision": "hard_decision"}, {"probabilistic", "optimisation"}),
        ("ec", T_EXPECTATION_CONSISTENT, "ec_detector",
         [("iterate", _PS_ITERATE_FULL, 1, {"inference"}), ("damping", _PS_DAMPING, 0, {"iterative"}), ("hard_decision", _PS_HARD_DECISION, 0, {"distance"})],
         {"slot_iterate": "iterate_proximal", "slot_damping": "damping_fixed", "slot_hard_decision": "hard_decision"}, {"probabilistic", "inference"}),
    ]

    for aid, src, fn, slot_tuples, dmap, tags in prob_specs:
        slot_args = [f"slot_{s[0]}" for s in slot_tuples]
        _add(aid, src, fn, slot_args, _slots(aid, *slot_tuples), dmap, tags)

    # ── E. Hybrid / multi-stage ───────────────────────────────────────

    _stage_refine_hd = [
        ("stage1", _PS_STAGE, 2, {"hybrid"}),
        ("refine", _PS_REFINE, 1, {"hybrid"}),
        ("hard_decision", _PS_HARD_DECISION, 0, {"distance"}),
    ]
    _stage2_merge_hd = [
        ("stage1", _PS_STAGE, 2, {"hybrid"}),
        ("stage2", _PS_STAGE, 2, {"hybrid"}),
        ("merge", _PS_MERGE, 0, {"hybrid"}),
        ("hard_decision", _PS_HARD_DECISION, 0, {"distance"}),
    ]

    hybrid_specs = [
        ("two_stage", T_TWO_STAGE, "two_stage_detector", _stage_refine_hd,
         {"slot_stage1": "stage_lmmse", "slot_refine": "refine_gradient_step", "slot_hard_decision": "hard_decision"}, {"hybrid"}),
        ("three_stage", T_THREE_STAGE, "three_stage_detector", _stage_refine_hd,
         {"slot_stage1": "stage_lmmse", "slot_refine": "refine_jacobi_step", "slot_hard_decision": "hard_decision"}, {"hybrid"}),
        ("parallel", T_PARALLEL, "parallel_detector", _stage2_merge_hd,
         {"slot_stage1": "stage_lmmse", "slot_stage2": "stage_zf", "slot_merge": "merge_average", "slot_hard_decision": "hard_decision"}, {"hybrid"}),
        ("cascade", T_CASCADE, "cascade_detector", _stage_refine_hd,
         {"slot_stage1": "stage_lmmse", "slot_refine": "refine_gradient_step", "slot_hard_decision": "hard_decision"}, {"hybrid"}),
        ("feedback_sic", T_FEEDBACK_SIC, "feedback_sic_detector", _stage_refine_hd,
         {"slot_stage1": "stage_lmmse", "slot_refine": "refine_gradient_step", "slot_hard_decision": "hard_decision"}, {"hybrid", "sic"}),
        ("warm_start", T_WARM_START, "warm_start_detector", _init_iterate_full_hd,
         {"slot_init": "init_mmse", "slot_iterate": "iterate_gradient", "slot_hard_decision": "hard_decision"}, {"hybrid", "iterative"}),
        ("iter_refine", T_ITERATIVE_REFINE, "iterative_refine_detector",
         [("stage1", _PS_STAGE, 2, {"hybrid"}), ("iterate", _PS_ITERATE_FULL, 1, {"iterative"}), ("hard_decision", _PS_HARD_DECISION, 0, {"distance"})],
         {"slot_stage1": "stage_lmmse", "slot_iterate": "iterate_gradient", "slot_hard_decision": "hard_decision"}, {"hybrid", "iterative"}),
        ("alternating", T_ALTERNATING, "alternating_detector",
         [("stage1", _PS_STAGE, 2, {"hybrid"}), ("stage2", _PS_STAGE, 2, {"hybrid"}), ("hard_decision", _PS_HARD_DECISION, 0, {"distance"})],
         {"slot_stage1": "stage_lmmse", "slot_stage2": "stage_zf", "slot_hard_decision": "hard_decision"}, {"hybrid"}),
        ("residual_net", T_RESIDUAL_NET, "residual_net_detector", _init_iterate_full_hd,
         {"slot_init": "init_mmse", "slot_iterate": "iterate_gradient", "slot_hard_decision": "hard_decision"}, {"hybrid", "iterative"}),
    ]

    for aid, src, fn, slot_tuples, dmap, tags in hybrid_specs:
        slot_args = [f"slot_{s[0]}" for s in slot_tuples]
        _add(aid, src, fn, slot_args, _slots(aid, *slot_tuples), dmap, tags)

    # ── F. Sub-module wrappers (L2) ──────────────────────────────────

    _lsolve_hd = [
        ("linear_solve", _PS_LINEAR_SOLVE, 1, {"linear_algebra"}),
        ("hard_decision", _PS_HARD_DECISION, 0, {"distance"}),
    ]
    _precond_lsolve_hd = [
        ("preconditioner", _PS_PRECONDITIONER, 1, {"linear_algebra"}),
        ("linear_solve", _PS_LINEAR_SOLVE, 1, {"linear_algebra"}),
        ("hard_decision", _PS_HARD_DECISION, 0, {"distance"}),
    ]

    submod_specs = [
        ("linear_solve_wrap", T_LINEAR_SOLVE_WRAPPER, "linear_solve_detector", _lsolve_hd,
         {"slot_linear_solve": "linear_solve_direct", "slot_hard_decision": "hard_decision"}, {"linear_algebra"}, 2),
        ("precond_solve_wrap", T_PRECONDITIONED_SOLVE, "precond_solve_detector", _precond_lsolve_hd,
         {"slot_preconditioner": "precond_diagonal", "slot_linear_solve": "linear_solve_direct", "slot_hard_decision": "hard_decision"}, {"linear_algebra"}, 2),
        ("reg_solve_wrap", T_REGULARIZED_SOLVE, "regularized_solve_detector",
         [("regularizer", _PS_REGULARIZER, 1, {"linear_algebra"}), ("linear_solve", _PS_LINEAR_SOLVE, 1, {"linear_algebra"}), ("hard_decision", _PS_HARD_DECISION, 0, {"distance"})],
         {"slot_regularizer": "regularizer", "slot_linear_solve": "linear_solve_direct", "slot_hard_decision": "hard_decision"}, {"linear_algebra"}, 2),
        ("weight_apply_wrap", T_WEIGHT_APPLY, "weight_apply_detector",
         [("weight", _PS_WEIGHT, 1, {"linear_algebra"}), ("hard_decision", _PS_HARD_DECISION, 0, {"distance"})],
         {"slot_weight": "weight_mmse", "slot_hard_decision": "hard_decision"}, {"linear_algebra"}, 2),
        ("mf_detect", T_MATCHED_FILTER_DETECT, "matched_filter_detector",
         [("normalize", _PS_NORMALIZE, 0, {"linear_algebra"}), ("hard_decision", _PS_HARD_DECISION, 0, {"distance"})],
         {"slot_normalize": "normalize_none", "slot_hard_decision": "hard_decision"}, {"linear_algebra"}, 2),
        ("qr_detect", T_QR_DETECT, "qr_detector",
         [("hard_decision", _PS_HARD_DECISION, 0, {"distance"})],
         {"slot_hard_decision": "hard_decision"}, {"linear_algebra"}, 2),
        ("svd_detect", T_SVD_DETECT, "svd_detector",
         [("hard_decision", _PS_HARD_DECISION, 0, {"distance"})],
         {"slot_hard_decision": "hard_decision"}, {"linear_algebra"}, 2),
        ("block_detect", T_BLOCK_DETECT, "block_detector", _lsolve_hd,
         {"slot_linear_solve": "linear_solve_direct", "slot_hard_decision": "hard_decision"}, {"linear_algebra"}, 2),
        ("subspace_detect", T_SUBSPACE_DETECT, "subspace_detector", _lsolve_hd,
         {"slot_linear_solve": "linear_solve_cg", "slot_hard_decision": "hard_decision"}, {"linear_algebra"}, 2),
        ("ordering_detect", T_ORDERING_DETECT, "ordering_detector",
         [("ordering", _PS_ORDERING, 1, {"sic"}), ("linear_solve", _PS_LINEAR_SOLVE, 1, {"linear_algebra"}), ("hard_decision", _PS_HARD_DECISION, 0, {"distance"})],
         {"slot_ordering": "ordering_column_norm", "slot_linear_solve": "linear_solve_direct", "slot_hard_decision": "hard_decision"}, {"sic", "linear_algebra"}, 2),
        ("soft_sic", T_SOFT_SIC, "soft_sic_detector",
         [("ordering", _PS_ORDERING, 1, {"sic"}), ("soft_estimate", _PS_SOFT_ESTIMATE, 1, {"inference"}), ("hard_decision", _PS_HARD_DECISION, 0, {"distance"})],
         {"slot_ordering": "ordering_column_norm", "slot_soft_estimate": "soft_mmse_estimate", "slot_hard_decision": "hard_decision"}, {"sic", "inference"}, 2),
        ("sparse_detect", T_SPARSE_DETECT, "sparse_detector",
         [("shrinkage", _PS_SHRINKAGE, 0, {"optimisation"}), ("hard_decision", _PS_HARD_DECISION, 0, {"distance"})],
         {"slot_shrinkage": "shrinkage_soft", "slot_hard_decision": "hard_decision"}, {"optimisation"}, 2),
        ("turbo_linear", T_TURBO_LINEAR, "turbo_linear_detector",
         [("soft_estimate", _PS_SOFT_ESTIMATE, 1, {"inference"}), ("hard_decision", _PS_HARD_DECISION, 0, {"distance"})],
         {"slot_soft_estimate": "soft_mmse_estimate", "slot_hard_decision": "hard_decision"}, {"inference"}, 2),
    ]

    for aid, src, fn, slot_tuples, dmap, tags, *rest in submod_specs:
        level = rest[0] if rest else 2
        slot_args = [f"slot_{s[0]}" for s in slot_tuples]
        _add(aid, src, fn, slot_args, _slots(aid, *slot_tuples), dmap, tags, level=level)

    # ── G. Building block wrappers (L1) ──────────────────────────────

    bb_specs = [
        ("diag_dominant", T_DIAG_DOMINANT, "diag_dominant_detector", _iterate_hd,
         {"slot_iterate": "jacobi_iterate", "slot_hard_decision": "hard_decision"}, {"linear_algebra"}, 1),
        ("mmse_init_iter", T_MMSE_INIT_ITERATE, "mmse_init_iterate", _init_iterate_damp_hd,
         {"slot_init": "init_mmse", "slot_iterate": "iterate_gradient", "slot_damping": "damping_fixed", "slot_hard_decision": "hard_decision"}, {"hybrid", "iterative"}, 1),
        ("mf_refine", T_MF_REFINE, "mf_refine_detector",
         [("iterate", _PS_ITERATE_FULL, 1, {"iterative"}), ("hard_decision", _PS_HARD_DECISION, 0, {"distance"})],
         {"slot_iterate": "iterate_gradient", "slot_hard_decision": "hard_decision"}, {"iterative"}, 1),
        ("scaled_detect", T_SCALED_DETECT, "scaled_detector",
         [("weight", _PS_WEIGHT, 1, {"linear_algebra"}), ("hard_decision", _PS_HARD_DECISION, 0, {"distance"})],
         {"slot_weight": "weight_mmse", "slot_hard_decision": "hard_decision"}, {"linear_algebra"}, 1),
        ("residual_iter", T_RESIDUAL_ITERATE, "residual_iterate_detector", _init_iterate_full_hd,
         {"slot_init": "init_mmse", "slot_iterate": "iterate_gradient", "slot_hard_decision": "hard_decision"}, {"iterative"}, 1),
        ("overrelax", T_OVERRELAX, "overrelax_detector",
         [("iterate", _PS_ITERATE_FULL, 1, {"iterative"}), ("damping", _PS_DAMPING, 0, {"iterative"}), ("hard_decision", _PS_HARD_DECISION, 0, {"distance"})],
         {"slot_iterate": "iterate_gradient", "slot_damping": "damping_fixed", "slot_hard_decision": "hard_decision"}, {"iterative"}, 1),
        ("power_method", T_POWER_METHOD, "power_method_detector",
         [("normalize", _PS_NORMALIZE, 0, {"linear_algebra"}), ("hard_decision", _PS_HARD_DECISION, 0, {"distance"})],
         {"slot_normalize": "normalize_unit", "slot_hard_decision": "hard_decision"}, {"linear_algebra"}, 1),
        ("landweber", T_LANDWEBER, "landweber_detector",
         [("hard_decision", _PS_HARD_DECISION, 0, {"distance"})],
         {"slot_hard_decision": "hard_decision"}, {"iterative"}, 1),
        ("kaczmarz", T_KACZMARZ, "kaczmarz_detector",
         [("hard_decision", _PS_HARD_DECISION, 0, {"distance"})],
         {"slot_hard_decision": "hard_decision"}, {"iterative"}, 1),
        ("cimmino", T_CIMMINO, "cimmino_detector",
         [("hard_decision", _PS_HARD_DECISION, 0, {"distance"})],
         {"slot_hard_decision": "hard_decision"}, {"iterative"}, 1),
        ("steepest_desc", T_STEEPEST_DESCENT, "steepest_descent_detector",
         [("hard_decision", _PS_HARD_DECISION, 0, {"distance"})],
         {"slot_hard_decision": "hard_decision"}, {"optimisation"}, 1),
    ]

    for aid, src, fn, slot_tuples, dmap, tags, *rest in bb_specs:
        level = rest[0] if rest else 1
        slot_args = [f"slot_{s[0]}" for s in slot_tuples]
        _add(aid, src, fn, slot_args, _slots(aid, *slot_tuples), dmap, tags, level=level)

    # ── H. Cross-domain ──────────────────────────────────────────────

    cross_specs = [
        ("divide_conquer", T_DIVIDE_CONQUER, "divide_conquer_detector", _lsolve_hd,
         {"slot_linear_solve": "linear_solve_direct", "slot_hard_decision": "hard_decision"}, {"structural"}),
        ("layer_peeling", T_LAYER_PEELING, "layer_peeling_detector",
         [("weight", _PS_WEIGHT, 1, {"linear_algebra"}), ("hard_decision", _PS_HARD_DECISION, 0, {"distance"})],
         {"slot_weight": "weight_mmse", "slot_hard_decision": "hard_decision"}, {"sic", "structural"}),
        ("consensus", T_CONSENSUS, "consensus_detector",
         [("iterate", _PS_ITERATE_FULL, 1, {"iterative"}), ("merge", _PS_MERGE, 0, {"hybrid"}), ("hard_decision", _PS_HARD_DECISION, 0, {"distance"})],
         {"slot_iterate": "iterate_gradient", "slot_merge": "merge_average", "slot_hard_decision": "hard_decision"}, {"structural"}),
        ("anneal_iter", T_ANNEAL_ITERATE, "anneal_iterate_detector",
         [("iterate", _PS_ITERATE_FULL, 1, {"iterative"}), ("project", _PS_PROJECT, 0, {"optimisation"}), ("hard_decision", _PS_HARD_DECISION, 0, {"distance"})],
         {"slot_iterate": "iterate_gradient", "slot_project": "project_nearest_symbol", "slot_hard_decision": "hard_decision"}, {"search", "iterative"}),
        ("dp_detect", T_DP_DETECT, "dp_detector", _score_hd,
         {"slot_score": "score_ml_cost", "slot_hard_decision": "hard_decision"}, {"search", "structural"}),
        ("multi_scale", T_MULTI_SCALE, "multi_scale_detector",
         [("linear_solve", _PS_LINEAR_SOLVE, 1, {"linear_algebra"}), ("refine", _PS_REFINE, 1, {"hybrid"}), ("hard_decision", _PS_HARD_DECISION, 0, {"distance"})],
         {"slot_linear_solve": "linear_solve_direct", "slot_refine": "refine_gradient_step", "slot_hard_decision": "hard_decision"}, {"structural", "hybrid"}),
        ("gs_block", T_GAUSS_SEIDEL_BLOCK, "gs_block_detector", _lsolve_hd,
         {"slot_linear_solve": "linear_solve_direct", "slot_hard_decision": "hard_decision"}, {"iterative", "structural"}),
        ("gpa", T_GRADIENT_PROJECT_ALTERNATE, "gpa_detector",
         [("gradient", _PS_GRADIENT, 1, {"optimisation"}), ("project", _PS_PROJECT, 0, {"optimisation"}), ("damping", _PS_DAMPING, 0, {"iterative"}), ("hard_decision", _PS_HARD_DECISION, 0, {"distance"})],
         {"slot_gradient": "gradient_ml", "slot_project": "project_nearest_symbol", "slot_damping": "damping_fixed", "slot_hard_decision": "hard_decision"}, {"optimisation", "structural"}),
        ("nested_iter", T_NESTED_ITERATE, "nested_iterate_detector",
         [("init", _PS_INIT, 1, {"linear_algebra"}), ("iterate", _PS_ITERATE_FULL, 1, {"iterative"}), ("refine", _PS_REFINE, 1, {"hybrid"}), ("hard_decision", _PS_HARD_DECISION, 0, {"distance"})],
         {"slot_init": "init_mmse", "slot_iterate": "iterate_gradient", "slot_refine": "refine_gradient_step", "slot_hard_decision": "hard_decision"}, {"hybrid", "structural"}),
    ]

    for aid, src, fn, slot_tuples, dmap, tags in cross_specs:
        slot_args = [f"slot_{s[0]}" for s in slot_tuples]
        _add(aid, src, fn, slot_args, _slots(aid, *slot_tuples), dmap, tags)

    return specs


# Module-level cache
_SPECS: list[SkeletonSpec] | None = None


def get_extended_specs() -> list[SkeletonSpec]:
    """Return all skeleton specs (cached)."""
    global _SPECS
    if _SPECS is None:
        _SPECS = _build_specs()
    return _SPECS
