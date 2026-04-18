"""Level-2 algorithm modules for MIMO detection.

These are mid-level building blocks that compose L0/L1 primitives and expose
their own slots for fine-grained evolution.  Each module is a real
signal-processing sub-routine (not a toy).

Modules:
  2.1  Tree-search modules (expand_node, frontier_scoring, prune)
  2.2  Graph / message-passing modules (message_up, message_down, bp_sweep)
  2.3  Inference modules (ep_site_update, amp_iteration_step, sic_detect_one)
  2.4  Iterative control (fixed_point_iterate)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from evolution.pool_ops_l0 import _EPS, c_abs2, v_norm_sq, v_sub, m_matvec
from evolution.pool_ops_l1 import (
    symbol_distance,
    cumulative_metric,
    moment_match,
    cavity_distribution,
    linear_equalize,
)


# ═══════════════════════════════════════════════════════════════════════════
# Common data types
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TreeNode:
    """A node in a search tree for MIMO detection."""
    level: int                         # current tree level (layer index)
    symbols: list[complex]             # symbols chosen so far (from level Nt-1 down to level+1)
    cost: float = 0.0                  # accumulated path cost
    parent: TreeNode | None = None
    children: list[TreeNode] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# 2.1  Tree-search modules
# ═══════════════════════════════════════════════════════════════════════════

def expand_node(
    node: TreeNode,
    y_tilde: np.ndarray,
    R: np.ndarray,
    constellation: np.ndarray,
    *,
    local_cost_fn=None,
    cumulative_cost_fn=None,
) -> list[TreeNode]:
    """Expand a tree node by testing all constellation points at the current level.

    For each symbol s in the constellation, computes:
      partial_interference = sum_{j>level} R[level,j] * symbols[j]
      local_cost = symbol_distance(y_tilde[level], R[level,level], interf, s)
      total_cost = cumulative_metric(parent_cost, local_cost)
    and creates a child TreeNode.

    Parameters
    ----------
    local_cost_fn : optional (y_k, r_kk, interf, sym) -> float
    cumulative_cost_fn : optional (parent_cost, local_cost) -> float
    """
    level = node.level
    Nt = R.shape[1]

    # Compute partial interference from already decided symbols
    partial_interf = 0.0 + 0.0j
    for j_rel, sym in enumerate(node.symbols):
        j = Nt - len(node.symbols) + j_rel
        if j < Nt:
            partial_interf += R[level, j] * sym

    children = []
    for s in constellation:
        if local_cost_fn is None:
            lc = symbol_distance(
                y_tilde[level], R[level, level], partial_interf, s,
            )
        else:
            lc = local_cost_fn(y_tilde[level], R[level, level], partial_interf, s)

        if cumulative_cost_fn is None:
            tc = cumulative_metric(node.cost, lc)
        else:
            tc = cumulative_cost_fn(node.cost, lc)

        child = TreeNode(
            level=level - 1,
            symbols=[s] + node.symbols,
            cost=tc,
            parent=node,
        )
        children.append(child)

    node.children = children
    return children


def frontier_scoring(
    candidates: list[TreeNode],
    *,
    score_fn=None,
) -> list[float]:
    """Score each candidate node.  Default: raw accumulated cost."""
    if score_fn is None:
        return [n.cost for n in candidates]
    return [score_fn(n) for n in candidates]


def prune_kbest(
    candidates: list[TreeNode],
    K: int,
    *,
    ranking_fn=None,
    selection_fn=None,
) -> list[TreeNode]:
    """Keep K-best candidates.

    Parameters
    ----------
    ranking_fn : optional list[TreeNode] -> list[float]
        Default: sort by accumulated cost (ascending)
    selection_fn : optional (sorted_candidates, K) -> list[TreeNode]
        Default: take first K
    """
    if ranking_fn is None:
        scores = [n.cost for n in candidates]
    else:
        scores = ranking_fn(candidates)

    sorted_pairs = sorted(zip(scores, candidates), key=lambda p: p[0])
    sorted_candidates = [c for _, c in sorted_pairs]

    if selection_fn is None:
        return sorted_candidates[:K]
    return selection_fn(sorted_candidates, K)


def best_first_step(
    open_set: list[TreeNode],
    y_tilde: np.ndarray,
    R: np.ndarray,
    constellation: np.ndarray,
    *,
    node_select_fn=None,
    expand_fn=None,
) -> tuple[list[TreeNode], TreeNode | None]:
    """One step of best-first tree search.

    1. Select best node from open set
    2. Expand it
    3. Add children to open set
    Returns (updated_open_set, completed_node_or_None)
    """
    if not open_set:
        return [], None

    # Select
    if node_select_fn is None:
        # Pop node with smallest cost
        best_idx = min(range(len(open_set)), key=lambda i: open_set[i].cost)
    else:
        best_idx = node_select_fn(open_set)

    node = open_set.pop(best_idx)

    # If leaf, this is a completed path
    if node.level < 0:
        return open_set, node

    # Expand
    if expand_fn is None:
        children = expand_node(node, y_tilde, R, constellation)
    else:
        children = expand_fn(node, y_tilde, R, constellation)

    open_set.extend(children)
    return open_set, None


# ═══════════════════════════════════════════════════════════════════════════
# 2.2  Graph / message-passing modules
# ═══════════════════════════════════════════════════════════════════════════

def message_up(
    node_beliefs: np.ndarray,
    child_indices: list[int],
    edge_weights: np.ndarray | None = None,
    *,
    leaf_value_fn=None,
    child_contrib_fn=None,
    aggregate_fn=None,
) -> complex:
    """Upward (leaf→root) message computation on a factor graph.

    Parameters
    ----------
    node_beliefs : current belief vector (one per variable node)
    child_indices : indices of child nodes
    edge_weights : optional weight per edge
    leaf_value_fn : i -> value (for leaf nodes)
    child_contrib_fn : (belief, weight) -> contribution
    aggregate_fn : list[contributions] -> aggregated value
    """
    if not child_indices:
        # Leaf
        if leaf_value_fn is not None:
            return leaf_value_fn(0)
        return complex(0.0)

    contributions = []
    for idx, ci in enumerate(child_indices):
        belief = node_beliefs[ci]
        w = edge_weights[idx] if edge_weights is not None else 1.0
        if child_contrib_fn is None:
            contributions.append(belief * w)
        else:
            contributions.append(child_contrib_fn(belief, w))

    if aggregate_fn is None:
        return sum(contributions)
    return aggregate_fn(contributions)


def message_down(
    parent_belief: complex,
    sibling_msgs: list[complex],
    *,
    down_rule_fn=None,
) -> complex:
    """Downward (root→leaf) message computation.

    Default: parent_belief - sum(sibling_msgs)
    """
    if down_rule_fn is None:
        return parent_belief - sum(sibling_msgs)
    return down_rule_fn(parent_belief, sibling_msgs)


def gaussian_bp_message(
    mu_in: np.ndarray,
    prec_in: np.ndarray,
    H_col: np.ndarray,
    sigma2: float,
    *,
    precision_fn=None,
    mean_fn=None,
) -> tuple[complex, float]:
    """One Gaussian BP variable-to-factor message.

    Default: LMMSE-style local belief update.
    mu_in, prec_in: incoming message means and precisions
    H_col: column of H for this variable
    sigma2: noise variance

    Returns (mean, precision) of outgoing message.
    """
    if precision_fn is None:
        prec_out = float(np.real(np.vdot(H_col, H_col))) / max(sigma2, _EPS)
        prec_out += float(np.sum(np.real(prec_in)))
    else:
        prec_out = precision_fn(H_col, sigma2, prec_in)

    if mean_fn is None:
        # Weighted combination of incoming info
        info = np.sum(mu_in * prec_in)
        mu_out = info / max(prec_out, _EPS)
    else:
        mu_out = mean_fn(mu_in, prec_in, H_col, sigma2)

    return mu_out, max(prec_out, _EPS)


def full_bp_sweep(
    H: np.ndarray,
    y: np.ndarray,
    sigma2: float,
    beliefs_mu: np.ndarray,
    beliefs_var: np.ndarray,
    constellation: np.ndarray,
    max_iters: int = 10,
    *,
    message_fn=None,
    belief_update_fn=None,
    halt_fn=None,
    damping: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Full belief-propagation sweep on the MIMO factor graph.

    Implements Gaussian EP / BP over the factor graph defined by y = Hx + n.

    Parameters
    ----------
    H : (Nr, Nt) channel
    y : (Nr,) observation
    sigma2 : noise variance
    beliefs_mu : (Nt,) initial belief means
    beliefs_var : (Nt,) initial belief variances
    constellation : constellation points
    max_iters : max BP iterations
    message_fn : optional per-variable message update
    belief_update_fn : optional belief update rule
    halt_fn : optional (old_mu, new_mu) -> bool convergence check
    damping : damping factor for belief updates
    """
    Nr, Nt = H.shape
    mu = beliefs_mu.copy()
    var = beliefs_var.copy()

    # Messages: factor-to-variable (Nr factors, Nt variables)
    msg_mu = np.zeros((Nr, Nt), dtype=complex)
    msg_prec = np.zeros((Nr, Nt), dtype=float)

    for it in range(max_iters):
        mu_old = mu.copy()

        # Variable-to-factor messages & factor-to-variable messages
        for f in range(Nr):
            for v in range(Nt):
                # Cavity: remove factor f's contribution to variable v
                cav_prec = max(1.0 / max(var[v], _EPS) - msg_prec[f, v], _EPS)
                cav_mu = (mu[v] / max(var[v], _EPS) - msg_mu[f, v] * msg_prec[f, v]) / cav_prec

                # Factor-to-variable message via residual
                residual = y[f] - sum(H[f, j] * mu[j] for j in range(Nt) if j != v)
                new_prec = float(np.abs(H[f, v]) ** 2) / max(sigma2, _EPS)
                if abs(H[f, v]) > _EPS:
                    new_mu = residual / H[f, v]
                else:
                    new_mu = 0.0 + 0.0j

                # Damping
                msg_prec[f, v] = (1 - damping) * msg_prec[f, v] + damping * new_prec
                msg_mu[f, v] = (1 - damping) * msg_mu[f, v] + damping * new_mu

        # Update beliefs
        for v in range(Nt):
            total_prec = np.sum(msg_prec[:, v])
            if total_prec < _EPS:
                continue
            info = np.sum(msg_mu[:, v] * msg_prec[:, v])
            new_mu_v = info / total_prec
            new_var_v = 1.0 / total_prec

            if belief_update_fn is not None:
                new_mu_v, new_var_v = belief_update_fn(
                    new_mu_v, new_var_v, constellation,
                )

            mu[v] = (1 - damping) * mu_old[v] + damping * new_mu_v
            var[v] = max(new_var_v, _EPS)

        # Convergence check
        if halt_fn is not None:
            if halt_fn(mu_old, mu):
                break
        else:
            if np.max(np.abs(mu - mu_old)) < 1e-6:
                break

    return mu, var


# ═══════════════════════════════════════════════════════════════════════════
# 2.3  Inference modules
# ═══════════════════════════════════════════════════════════════════════════

def ep_site_update(
    cavity_mu: complex,
    cavity_var: float,
    constellation: np.ndarray,
    site_mu: complex,
    site_var: float,
    *,
    tilted_fn=None,
    moment_fn=None,
    site_prec_fn=None,
    site_mean_fn=None,
) -> tuple[complex, float]:
    """EP site update for one variable.

    1. Compute tilted distribution = cavity × likelihood
    2. Moment-match to Gaussian → (post_mu, post_var)
    3. New site = posterior − cavity
    """
    # Tilted distribution: likelihood of each constellation point under cavity
    if tilted_fn is None:
        log_lik = -np.abs(constellation - cavity_mu) ** 2 / (2 * max(cavity_var, _EPS))
        log_lik -= np.max(log_lik)  # numerical stability
        tilted = np.exp(log_lik)
    else:
        tilted = tilted_fn(cavity_mu, cavity_var, constellation)

    # Moment matching
    if moment_fn is None:
        post_mu, post_var = moment_match(tilted, constellation)
    else:
        post_mu, post_var = moment_fn(tilted, constellation)

    post_var = max(post_var, _EPS)

    # Site update: new_site = posterior - cavity (in natural parameters)
    if site_prec_fn is None:
        new_site_prec = 1.0 / post_var - 1.0 / max(cavity_var, _EPS)
        new_site_prec = max(new_site_prec, _EPS)
        new_site_var = 1.0 / new_site_prec
    else:
        new_site_var = site_prec_fn(post_var, cavity_var)

    if site_mean_fn is None:
        new_site_prec = 1.0 / max(new_site_var, _EPS)
        post_prec = 1.0 / post_var
        cav_prec = 1.0 / max(cavity_var, _EPS)
        new_site_mu = (post_mu * post_prec - cavity_mu * cav_prec) / max(new_site_prec, _EPS)
    else:
        new_site_mu = site_mean_fn(post_mu, post_var, cavity_mu, cavity_var)

    return new_site_mu, max(new_site_var, _EPS)


def amp_iteration_step(
    H: np.ndarray,
    y: np.ndarray,
    sigma2: float,
    x_hat: np.ndarray,
    s_hat: np.ndarray,
    z: np.ndarray,
    constellation: np.ndarray,
    *,
    residual_fn=None,
    onsager_fn=None,
    effective_obs_fn=None,
    denoiser_fn=None,
    divergence_fn=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One AMP iteration step.

    Implements the canonical AMP iteration:
      z = y - H x_hat + z_prev * <η'(...)> / Nr   (Onsager correction)
      r = x_hat + H^H z                            (effective observation)
      x_hat_new = η(r)                              (denoiser)

    Returns (x_hat_new, s_hat_new, z_new)
    """
    Nr, Nt = H.shape

    # Residual
    if residual_fn is None:
        z_new = y - H @ x_hat
    else:
        z_new = residual_fn(y, H, x_hat)

    # Onsager correction
    if onsager_fn is None:
        # Compute divergence of denoiser (approximate)
        onsager_term = np.mean(np.real(s_hat)) / max(Nr, 1)
        z_new = z_new + z * onsager_term
    else:
        z_new = onsager_fn(z_new, z, s_hat, Nr)

    # Effective observation
    if effective_obs_fn is None:
        r = x_hat + H.conj().T @ z_new
    else:
        r = effective_obs_fn(x_hat, H, z_new)

    # Estimated noise variance per element
    tau = float(v_norm_sq(z_new)) / max(Nr, 1)

    # Denoiser: MMSE estimation of x from r = x + N(0, tau)
    if denoiser_fn is None:
        x_new = np.zeros(Nt, dtype=complex)
        s_new = np.zeros(Nt, dtype=float)
        for i in range(Nt):
            # MMSE denoiser over constellation
            log_lik = -np.abs(constellation - r[i]) ** 2 / (2 * max(tau, _EPS))
            log_lik -= np.max(log_lik)
            weights = np.exp(log_lik)
            w_sum = np.sum(weights)
            if w_sum < _EPS:
                weights = np.ones(len(constellation)) / len(constellation)
            else:
                weights /= w_sum
            x_new[i] = np.sum(weights * constellation)
            s_new[i] = np.sum(weights * np.abs(constellation - x_new[i]) ** 2) / max(tau, _EPS)
    else:
        x_new, s_new = denoiser_fn(r, tau, constellation)

    return x_new, s_new, z_new


def sic_detect_one(
    H: np.ndarray,
    y: np.ndarray,
    sigma2: float,
    layer_idx: int,
    constellation: np.ndarray,
    *,
    detect_fn=None,
    hard_decision_fn=None,
    cancel_fn=None,
) -> tuple[complex, np.ndarray, np.ndarray]:
    """Detect one layer and cancel its interference.

    Returns (detected_symbol, H_deflated, y_deflated)
    """
    Nt = H.shape[1]

    # Detect this layer
    if detect_fn is None:
        x_soft = linear_equalize(H, y, sigma2)
        x_layer = x_soft[layer_idx]
    else:
        x_layer = detect_fn(H, y, sigma2, layer_idx)

    # Hard decision
    if hard_decision_fn is None:
        dists = np.abs(constellation - x_layer) ** 2
        x_hard = constellation[np.argmin(dists)]
    else:
        x_hard = hard_decision_fn(x_layer, constellation)

    # Cancel interference
    if cancel_fn is None:
        y_new = y - H[:, layer_idx] * x_hard
        H_new = np.delete(H, layer_idx, axis=1)
    else:
        y_new, H_new = cancel_fn(y, H, layer_idx, x_hard)

    return x_hard, H_new, y_new


# ═══════════════════════════════════════════════════════════════════════════
# 2.4  Iterative control
# ═══════════════════════════════════════════════════════════════════════════

def fixed_point_iterate(
    state: Any,
    update_fn,
    converged_fn=None,
    max_iters: int = 20,
    damping: float = 0.0,
) -> Any:
    """Generic fixed-point iteration.

    Repeatedly applies state = update_fn(state) until converged or max_iters.
    If damping > 0 and state supports arithmetic (ndarray), applies:
      state = (1-damping) * old + damping * new
    """
    for _ in range(max_iters):
        new_state = update_fn(state)

        if converged_fn is not None and converged_fn(state, new_state):
            return new_state

        if damping > 0 and isinstance(state, np.ndarray):
            state = (1 - damping) * state + damping * new_state
        else:
            state = new_state

    return state
