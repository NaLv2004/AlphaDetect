"""Level-1 composite operations for MIMO detection.

Each operation is a self-contained callable implementing a real signal-processing
sub-routine.  These correspond to the L1 entries in the algorithm pool.

Slots are represented as **callback parameters** so that higher-level code
can inject different L0 primitives or alternative sub-routines.
"""

from __future__ import annotations

import numpy as np

from evolution.pool_ops_l0 import (
    m_conj_transpose, m_gram, m_eye, m_solve, m_matvec,
    v_sub, v_norm_sq, c_abs2, _EPS,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1.1  Linear-algebra patterns
# ═══════════════════════════════════════════════════════════════════════════

def regularized_solve(
    H: np.ndarray,
    y: np.ndarray,
    sigma2: float,
    *,
    regularizer=None,
) -> np.ndarray:
    """Solve (H^H H + Λ)^{-1} H^H y.

    Parameters
    ----------
    H : (Nr, Nt) complex channel matrix
    y : (Nr,) received vector
    sigma2 : noise variance
    regularizer : optional callable (G, sigma2) -> G_reg
        Default: G + sigma2 * I  (MMSE regulariser)
    """
    G = m_gram(H)
    Nt = G.shape[0]
    if regularizer is None:
        G_reg = G + sigma2 * m_eye(Nt)
    else:
        G_reg = regularizer(G, sigma2)
    rhs = m_matvec(m_conj_transpose(H), y)
    return m_solve(G_reg, rhs)


def whitening_transform(
    H: np.ndarray,
    y: np.ndarray,
    sigma2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Whiten observation: y_w = R^{-H} y, H_w = R^{-H} H where Cn = σ²I.

    For additive white Gaussian noise this is trivial (scaling), but the
    slot-based design allows evolution to discover non-trivial transforms.
    """
    scale = 1.0 / np.sqrt(max(sigma2, _EPS))
    return H * scale, y * scale


def matched_filter(H: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Matched filter: x_mf = H^H y."""
    return m_matvec(m_conj_transpose(H), y)


# ═══════════════════════════════════════════════════════════════════════════
# 1.2  Distance / metric patterns
# ═══════════════════════════════════════════════════════════════════════════

def symbol_distance(
    y_tilde_k: complex,
    r_kk: complex,
    partial_interference: complex,
    symbol: complex,
    *,
    residual_fn=None,
    distance_fn=None,
) -> float:
    """Distance of a symbol hypothesis at one tree level.

    residual = y_tilde_k - r_kk * symbol - partial_interference
    cost = |residual|²

    Parameters
    ----------
    y_tilde_k : transformed received signal at level k
    r_kk : diagonal element of R at level k
    partial_interference : accumulated interference from already-decided symbols
    symbol : constellation point being tested
    residual_fn : (y, r, interf, sym) -> residual.  Default: y - r*sym - interf
    distance_fn : residual -> cost.  Default: |r|²
    """
    if residual_fn is None:
        residual = y_tilde_k - r_kk * symbol - partial_interference
    else:
        residual = residual_fn(y_tilde_k, r_kk, partial_interference, symbol)
    if distance_fn is None:
        return c_abs2(residual)
    return distance_fn(residual)


def cumulative_metric(
    parent_cost: float,
    local_cost: float,
    *,
    accumulate_fn=None,
) -> float:
    """Accumulate cost from parent node.

    Default: simple sum.
    """
    if accumulate_fn is None:
        return parent_cost + local_cost
    return accumulate_fn(parent_cost, local_cost)


def log_likelihood_distance(
    y: np.ndarray,
    H: np.ndarray,
    x: np.ndarray,
    sigma2: float,
) -> float:
    """Negative log-likelihood ‖y - Hx‖² / σ²."""
    residual = v_sub(y, m_matvec(H, x))
    return v_norm_sq(residual) / max(sigma2, _EPS)


# ═══════════════════════════════════════════════════════════════════════════
# 1.3  Filtering / equalisation
# ═══════════════════════════════════════════════════════════════════════════

def linear_equalize(
    H: np.ndarray,
    y: np.ndarray,
    sigma2: float,
    *,
    weight_fn=None,
) -> np.ndarray:
    """Linear equalisation x = W y.

    Default weight_fn computes MMSE weight matrix:
      W = (H^H H + σ²I)^{-1} H^H
    """
    if weight_fn is None:
        G = m_gram(H)
        Nt = G.shape[0]
        G_reg = G + sigma2 * m_eye(Nt)
        W = m_solve(G_reg, m_conj_transpose(H))
    else:
        W = weight_fn(H, sigma2)
    return m_matvec(W, y)


# ═══════════════════════════════════════════════════════════════════════════
# 1.4  Distribution / inference patterns
# ═══════════════════════════════════════════════════════════════════════════

def moment_match(
    likelihoods: np.ndarray,
    constellation: np.ndarray,
    *,
    mean_fn=None,
    var_fn=None,
) -> tuple[complex, float]:
    """Moment-match likelihoods over a constellation to (mean, var).

    Parameters
    ----------
    likelihoods : probability/weight for each constellation point
    constellation : complex constellation points
    mean_fn : weights, points -> mean.  Default: weighted mean
    var_fn : weights, points, mean -> var.  Default: weighted variance
    """
    # Normalise
    w = np.real(likelihoods).copy()
    w_sum = np.sum(w)
    if w_sum < _EPS:
        w = np.ones(len(w)) / len(w)
    else:
        w = w / w_sum

    if mean_fn is None:
        mu = np.sum(w * constellation)
    else:
        mu = mean_fn(w, constellation)

    if var_fn is None:
        var = float(np.real(np.sum(w * np.abs(constellation - mu) ** 2)))
    else:
        var = var_fn(w, constellation, mu)

    return mu, var


def cavity_distribution(
    global_mu: complex,
    global_var: float,
    site_mu: complex,
    site_var: float,
    *,
    cavity_var_fn=None,
    cavity_mu_fn=None,
) -> tuple[complex, float]:
    """Compute cavity distribution by removing one site.

    cavity_precision = 1/global_var - 1/site_var
    cavity_mean = (global_mu / global_var - site_mu / site_var) / cavity_precision
    """
    g_prec = 1.0 / max(global_var, _EPS)
    s_prec = 1.0 / max(site_var, _EPS)

    if cavity_var_fn is None:
        cav_prec = g_prec - s_prec
        cav_prec = max(cav_prec, _EPS)
        cav_var = 1.0 / cav_prec
    else:
        cav_var = cavity_var_fn(global_var, site_var)

    if cavity_mu_fn is None:
        cav_prec = max(g_prec - s_prec, _EPS)
        cav_mu = (global_mu * g_prec - site_mu * s_prec) / cav_prec
    else:
        cav_mu = cavity_mu_fn(global_mu, global_var, site_mu, site_var)

    return cav_mu, max(float(np.real(cav_var)), _EPS)


def kl_projection(
    posterior_mu: complex,
    posterior_var: float,
    prior_mu: complex,
    prior_var: float,
) -> tuple[complex, float]:
    """Project posterior onto Gaussian family (KL minimisation = moment match)."""
    return posterior_mu, posterior_var
