"""Level-3 complete MIMO detection algorithms.

Each function is a fully functional detector that accepts standard MIMO
inputs (H, y, sigma2, constellation) and returns the detected symbol vector.

These are *real* algorithms — not toy stubs.  They use the L0/L1/L2
building blocks and expose slot parameters for evolution.

Algorithms
----------
  - LMMSE  (linear MMSE)
  - ZF     (zero-forcing, special case of LMMSE with σ²=0)
  - OSIC   (ordered successive interference cancellation)
  - K-Best (K-best tree search)
  - Stack  (stack / best-first tree search)
  - BP     (Gaussian belief propagation)
  - EP     (expectation propagation)
  - AMP    (approximate message passing)
"""

from __future__ import annotations

import numpy as np

from evolution.pool_ops_l0 import (
    _EPS,
    m_conj_transpose, m_gram, m_eye, m_solve, m_matvec, m_qr,
    v_norm_sq,
)
from evolution.pool_ops_l1 import (
    regularized_solve,
    symbol_distance,
    cumulative_metric,
    moment_match,
    cavity_distribution,
    linear_equalize,
)
from evolution.pool_ops_l2 import (
    TreeNode,
    expand_node,
    prune_kbest,
    best_first_step,
    full_bp_sweep,
    ep_site_update,
    amp_iteration_step,
    sic_detect_one,
    fixed_point_iterate,
)


# ═══════════════════════════════════════════════════════════════════════════
# Utility — nearest-point hard decision
# ═══════════════════════════════════════════════════════════════════════════

def _hard_decide(x_soft: np.ndarray, constellation: np.ndarray) -> np.ndarray:
    """Per-element nearest-constellation-point slicing."""
    x_hat = np.empty(len(x_soft), dtype=complex)
    for i in range(len(x_soft)):
        dists = np.abs(constellation - x_soft[i]) ** 2
        x_hat[i] = constellation[np.argmin(dists)]
    return x_hat


# ═══════════════════════════════════════════════════════════════════════════
# 3.1  LMMSE Detector
# ═══════════════════════════════════════════════════════════════════════════

def lmmse_detector(
    H: np.ndarray,
    y: np.ndarray,
    sigma2: float,
    constellation: np.ndarray,
    *,
    regularizer=None,
    hard_decision_fn=None,
) -> np.ndarray:
    """LMMSE (Linear Minimum Mean Square Error) detector.

    x̂ = (H^H H + σ²I)^{-1} H^H y, then hard decision.

    Slots
    -----
    regularizer : (G, sigma2) -> G_reg          [L3-A]
    hard_decision_fn : (x_soft, constellation) -> x_hat  [L3-B]
    """
    G = m_gram(H)
    Nt = G.shape[0]
    if regularizer is None:
        G_reg = G + sigma2 * m_eye(Nt)
    else:
        G_reg = regularizer(G, sigma2)

    rhs = m_matvec(m_conj_transpose(H), y)
    x_soft = m_solve(G_reg, rhs)

    if hard_decision_fn is None:
        return _hard_decide(x_soft, constellation)
    return hard_decision_fn(x_soft, constellation)


# ═══════════════════════════════════════════════════════════════════════════
# 3.1b  ZF Detector  (LMMSE with σ²=0)
# ═══════════════════════════════════════════════════════════════════════════

def zf_detector(
    H: np.ndarray,
    y: np.ndarray,
    sigma2: float,
    constellation: np.ndarray,
    *,
    hard_decision_fn=None,
) -> np.ndarray:
    """Zero-Forcing detector: x̂ = (H^H H)^{-1} H^H y.

    Equivalent to LMMSE with σ²=0.
    """
    G = m_gram(H)
    rhs = m_matvec(m_conj_transpose(H), y)
    x_soft = m_solve(G, rhs)

    if hard_decision_fn is None:
        return _hard_decide(x_soft, constellation)
    return hard_decision_fn(x_soft, constellation)


# ═══════════════════════════════════════════════════════════════════════════
# 3.2  OSIC (Ordered SIC) Detector
# ═══════════════════════════════════════════════════════════════════════════

def osic_detector(
    H: np.ndarray,
    y: np.ndarray,
    sigma2: float,
    constellation: np.ndarray,
    *,
    ordering_fn=None,
    sic_step_fn=None,
) -> np.ndarray:
    """OSIC (Ordered Successive Interference Cancellation) detector.

    Detects one layer at a time, subtracting its contribution from y.
    Uses MMSE post-detection SNR ordering by default.

    Slots
    -----
    ordering_fn : (H, y, sigma2) -> list[int]   [L3-A]
    sic_step_fn : (H, y, sigma2, idx, const) -> (sym, H_new, y_new)  [L3-B]
    """
    Nr, Nt = H.shape
    H_cur = H.copy()
    y_cur = y.copy()

    # Ordering: MMSE-based SNR
    if ordering_fn is None:
        G = m_gram(H_cur) + sigma2 * m_eye(Nt)
        G_inv = np.linalg.inv(G)
        post_snr = np.array([
            1.0 / max(float(np.real(G_inv[i, i])), _EPS) - sigma2
            for i in range(Nt)
        ])
        order = list(np.argsort(-post_snr))  # highest SNR first
    else:
        order = ordering_fn(H_cur, y_cur, sigma2)

    detected = np.zeros(Nt, dtype=complex)
    col_map = list(range(Nt))  # track original column indices

    for step in range(Nt):
        # Find which current column corresponds to the next ordered layer
        orig_idx = order[step]
        cur_idx = col_map.index(orig_idx)

        if sic_step_fn is None:
            sym, H_cur, y_cur = sic_detect_one(
                H_cur, y_cur, sigma2, cur_idx, constellation,
            )
        else:
            sym, H_cur, y_cur = sic_step_fn(
                H_cur, y_cur, sigma2, cur_idx, constellation,
            )

        detected[orig_idx] = sym
        col_map.pop(cur_idx)

    return detected


# ═══════════════════════════════════════════════════════════════════════════
# 3.3  K-Best Detector
# ═══════════════════════════════════════════════════════════════════════════

def kbest_detector(
    H: np.ndarray,
    y: np.ndarray,
    sigma2: float,
    constellation: np.ndarray,
    K: int = 16,
    *,
    expand_fn=None,
    prune_fn=None,
    child_score_fn=None,
) -> np.ndarray:
    """K-Best tree search detector.

    1. QR decompose H
    2. Process tree from level Nt-1 down to 0
    3. At each level, expand all candidates and keep K-best

    Slots
    -----
    expand_fn : (node, y_tilde, R, constellation) -> children   [L3-A]
    prune_fn : (candidates, K) -> survivors                     [L3-C]
    child_score_fn : (child) -> corrected_score                 [L3-B]
    """
    Nr, Nt = H.shape
    Q, R = m_qr(H)
    y_tilde = m_matvec(m_conj_transpose(Q), y)

    # Initialise with a root node at level Nt-1
    root = TreeNode(level=Nt - 1, symbols=[], cost=0.0)
    candidates = [root]

    for level in range(Nt - 1, -1, -1):
        new_candidates = []
        for node in candidates:
            node.level = level  # ensure correct level
            if expand_fn is None:
                children = expand_node(node, y_tilde, R, constellation)
            else:
                children = expand_fn(node, y_tilde, R, constellation)

            if child_score_fn is not None:
                for ch in children:
                    ch.cost = child_score_fn(ch)

            new_candidates.extend(children)

        # Prune to K-best
        if prune_fn is None:
            candidates = prune_kbest(new_candidates, K)
        else:
            candidates = prune_fn(new_candidates, K)

    # Best candidate
    if not candidates:
        return np.zeros(Nt, dtype=complex)

    best = min(candidates, key=lambda n: n.cost)
    x_hat = np.array(best.symbols, dtype=complex)

    # Symbols were accumulated from top (Nt-1) to bottom (0),
    # so they should be in the right order already.
    if len(x_hat) < Nt:
        # Pad if some levels were skipped
        full = np.zeros(Nt, dtype=complex)
        full[:len(x_hat)] = x_hat
        return full

    return x_hat


# ═══════════════════════════════════════════════════════════════════════════
# 3.4  Stack (Best-First) Detector
# ═══════════════════════════════════════════════════════════════════════════

def stack_detector(
    H: np.ndarray,
    y: np.ndarray,
    sigma2: float,
    constellation: np.ndarray,
    max_nodes: int = 500,
    *,
    node_select_fn=None,
    expand_fn=None,
) -> np.ndarray:
    """Stack (best-first search) detector.

    Maintains a priority queue (open set), always expanding the best node
    first.  Returns the first complete path found.

    Slots
    -----
    node_select_fn : open_set -> index   [L3-A]
    expand_fn : (node, y_tilde, R, const) -> children  [L3-B]
    """
    Nr, Nt = H.shape
    Q, R = m_qr(H)
    y_tilde = m_matvec(m_conj_transpose(Q), y)

    root = TreeNode(level=Nt - 1, symbols=[], cost=0.0)
    open_set = [root]
    nodes_expanded = 0

    while open_set and nodes_expanded < max_nodes:
        # Select best node
        if node_select_fn is None:
            best_idx = min(range(len(open_set)), key=lambda i: open_set[i].cost)
        else:
            best_idx = node_select_fn(open_set)

        node = open_set.pop(best_idx)
        nodes_expanded += 1

        # Check if complete
        if len(node.symbols) == Nt:
            return np.array(node.symbols, dtype=complex)

        # Expand
        if expand_fn is None:
            children = expand_node(node, y_tilde, R, constellation)
        else:
            children = expand_fn(node, y_tilde, R, constellation)

        open_set.extend(children)

    # Fallback: best incomplete path or LMMSE
    if open_set:
        best = min(open_set, key=lambda n: n.cost)
        if len(best.symbols) == Nt:
            return np.array(best.symbols, dtype=complex)

    # Fallback to LMMSE
    return lmmse_detector(H, y, sigma2, constellation)


# ═══════════════════════════════════════════════════════════════════════════
# 3.5  BP (Gaussian Belief Propagation) Detector
# ═══════════════════════════════════════════════════════════════════════════

def bp_detector(
    H: np.ndarray,
    y: np.ndarray,
    sigma2: float,
    constellation: np.ndarray,
    max_iters: int = 20,
    *,
    bp_sweep_fn=None,
    final_decision_fn=None,
    damping: float = 0.5,
) -> np.ndarray:
    """Gaussian BP (Belief Propagation) detector.

    Iterative message passing on the MIMO factor graph y = Hx + n.

    Slots
    -----
    bp_sweep_fn : (H, y, σ², mu, var, const, iters) -> (mu, var)  [L3-A]
    final_decision_fn : (mu, constellation) -> x_hat              [L3-B]
    """
    Nr, Nt = H.shape

    # Initial beliefs: matched-filter estimate
    x_mf = m_matvec(m_conj_transpose(H), y)
    G_diag = np.real(np.sum(np.abs(H) ** 2, axis=0))
    init_mu = x_mf / np.maximum(G_diag, _EPS)
    init_var = sigma2 / np.maximum(G_diag, _EPS)

    if bp_sweep_fn is None:
        mu, var = full_bp_sweep(
            H, y, sigma2, init_mu, init_var, constellation,
            max_iters=max_iters, damping=damping,
        )
    else:
        mu, var = bp_sweep_fn(
            H, y, sigma2, init_mu, init_var, constellation, max_iters,
        )

    if final_decision_fn is None:
        return _hard_decide(mu, constellation)
    return final_decision_fn(mu, constellation)


# ═══════════════════════════════════════════════════════════════════════════
# 3.6  EP (Expectation Propagation) Detector
# ═══════════════════════════════════════════════════════════════════════════

def ep_detector(
    H: np.ndarray,
    y: np.ndarray,
    sigma2: float,
    constellation: np.ndarray,
    max_iters: int = 20,
    *,
    cavity_fn=None,
    site_update_fn=None,
    damping_fn=None,
    final_decision_fn=None,
    damping: float = 0.5,
) -> np.ndarray:
    """EP (Expectation Propagation) MIMO detector.

    Iteratively refines Gaussian site approximations to the
    constellation-constrained posterior.

    Slots
    -----
    cavity_fn : (global_mu, global_var, site_mu, site_var) -> (cav_mu, cav_var)  [L3-B]
    site_update_fn : (cav_mu, cav_var, const, site_mu, site_var) -> (new_mu, new_var)  [L3-C]
    damping_fn : (old, new, it) -> damped  [L3-D]
    final_decision_fn : (mu, constellation) -> x_hat  [L3-E]
    """
    Nr, Nt = H.shape

    # Initialise site parameters (natural form)
    site_mu = np.zeros(Nt, dtype=complex)
    site_var = np.full(Nt, 1e6, dtype=float)  # very large = uninformative

    # Precompute channel Gram
    HtH = m_gram(H)
    Hty = m_matvec(m_conj_transpose(H), y)

    for it in range(max_iters):
        site_mu_old = site_mu.copy()

        # Compute global posterior: posterior precision = H^H H / σ² + Σ_site^{-1}
        site_prec = 1.0 / np.maximum(site_var, _EPS)
        Sigma_inv = HtH / max(sigma2, _EPS) + np.diag(site_prec)
        try:
            Sigma = np.linalg.inv(Sigma_inv)
        except np.linalg.LinAlgError:
            Sigma = np.linalg.inv(Sigma_inv + 1e-10 * m_eye(Nt))

        global_mu = Sigma @ (Hty / max(sigma2, _EPS) + site_mu * site_prec)
        global_var = np.real(np.diag(Sigma))

        # Update each site
        for i in range(Nt):
            # Cavity distribution
            if cavity_fn is None:
                cav_mu, cav_var = cavity_distribution(
                    global_mu[i], float(global_var[i]),
                    site_mu[i], float(site_var[i]),
                )
            else:
                cav_mu, cav_var = cavity_fn(
                    global_mu[i], float(global_var[i]),
                    site_mu[i], float(site_var[i]),
                )

            # Site update via moment matching
            if site_update_fn is None:
                new_mu, new_var = ep_site_update(
                    cav_mu, cav_var, constellation,
                    site_mu[i], float(site_var[i]),
                )
            else:
                new_mu, new_var = site_update_fn(
                    cav_mu, cav_var, constellation,
                    site_mu[i], float(site_var[i]),
                )

            # Damping
            if damping_fn is None:
                site_mu[i] = (1 - damping) * site_mu[i] + damping * new_mu
                site_var[i] = (1 - damping) * site_var[i] + damping * new_var
            else:
                site_mu[i], site_var[i] = damping_fn(
                    (site_mu[i], site_var[i]),
                    (new_mu, new_var),
                    it,
                )

        # Convergence check
        if np.max(np.abs(site_mu - site_mu_old)) < 1e-6:
            break

    # Final global posterior
    site_prec = 1.0 / np.maximum(site_var, _EPS)
    Sigma_inv = HtH / max(sigma2, _EPS) + np.diag(site_prec)
    try:
        Sigma = np.linalg.inv(Sigma_inv)
    except np.linalg.LinAlgError:
        Sigma = np.linalg.inv(Sigma_inv + 1e-10 * m_eye(Nt))
    final_mu = Sigma @ (Hty / max(sigma2, _EPS) + site_mu * site_prec)

    if final_decision_fn is None:
        return _hard_decide(final_mu, constellation)
    return final_decision_fn(final_mu, constellation)


# ═══════════════════════════════════════════════════════════════════════════
# 3.7  AMP (Approximate Message Passing) Detector
# ═══════════════════════════════════════════════════════════════════════════

def amp_detector(
    H: np.ndarray,
    y: np.ndarray,
    sigma2: float,
    constellation: np.ndarray,
    max_iters: int = 30,
    *,
    iterate_fn=None,
    final_decision_fn=None,
    damping: float = 0.0,
) -> np.ndarray:
    """AMP (Approximate Message Passing) MIMO detector.

    Implements the canonical AMP algorithm with MMSE denoiser.

    Slots
    -----
    iterate_fn : (H,y,σ²,x,s,z,const) -> (x_new,s_new,z_new)  [L3-A]
    final_decision_fn : (x_hat, constellation) -> x_hat         [L3-B]
    """
    Nr, Nt = H.shape

    x_hat = np.zeros(Nt, dtype=complex)
    s_hat = np.ones(Nt, dtype=float)
    z = y.copy()

    for it in range(max_iters):
        x_old = x_hat.copy()

        if iterate_fn is None:
            x_hat, s_hat, z = amp_iteration_step(
                H, y, sigma2, x_hat, s_hat, z, constellation,
            )
        else:
            x_hat, s_hat, z = iterate_fn(
                H, y, sigma2, x_hat, s_hat, z, constellation,
            )

        # Optional damping
        if damping > 0:
            x_hat = (1 - damping) * x_old + damping * x_hat

        # Convergence
        if np.max(np.abs(x_hat - x_old)) < 1e-6:
            break

    if final_decision_fn is None:
        return _hard_decide(x_hat, constellation)
    return final_decision_fn(x_hat, constellation)


# ═══════════════════════════════════════════════════════════════════════════
# Detector registry
# ═══════════════════════════════════════════════════════════════════════════

DETECTOR_REGISTRY: dict[str, callable] = {
    "lmmse": lmmse_detector,
    "zf": zf_detector,
    "osic": osic_detector,
    "kbest": kbest_detector,
    "stack": stack_detector,
    "bp": bp_detector,
    "ep": ep_detector,
    "amp": amp_detector,
}
