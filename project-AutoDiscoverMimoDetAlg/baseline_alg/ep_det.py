"""
EP (Expectation Propagation) MIMO Detector.
Iterative Bayesian detector with Gaussian cavity approximation.
Matches the MATLAB reference implementation.
"""
import numpy as np


def ep_detect(tx_ant, rx_ant, slen, y, H, Nv, sym, delta, iter_num):
    """EP detector (real-valued formulation).
    tx_ant:   number of TX antennas
    rx_ant:   number of RX antennas
    slen:     constellation size
    y:        received signal (2*rx_ant,)
    H:        real-valued channel (2*rx_ant x 2*tx_ant)
    Nv:       noise variance (complex)
    sym:      PAM constellation (slen,)
    delta:    damping factor
    iter_num: number of iterations
    Returns:  detected symbol vector (2*tx_ant,)
    """
    N = 2 * tx_ant

    Alpha = 2.0 * np.ones(N)
    Gamma = np.zeros(N)
    Alpha_new = np.zeros(N)
    Gamma_new = np.zeros(N)

    HtH = H.T @ H
    Hty = H.T @ y

    # MMSE-initialised posterior
    Sigma_q = np.linalg.inv(HtH / Nv + np.diag(Alpha))
    Mu_q = Sigma_q @ (Hty / Nv + Gamma)

    for k in range(iter_num):
        sig = np.diag(Sigma_q)                             # marginal variances (N,)
        h2 = sig / (1.0 - sig * Alpha)                     # cavity variance
        t = h2 * (Mu_q / sig - Gamma)                      # cavity mean

        # Moment matching over discrete prior
        prob = np.exp(-(t[:, None] - sym[None, :]) ** 2 / (2.0 * h2[:, None]))
        prob = prob / np.sum(prob, axis=1, keepdims=True)

        mu_p = prob @ sym                                   # posterior mean  (N,)
        sigma2_p = prob @ (sym ** 2) - mu_p ** 2            # posterior var   (N,)
        sigma2_p = np.maximum(sigma2_p, 5e-7)

        tempAlpha = 1.0 / sigma2_p - 1.0 / h2
        tempGamma = mu_p / sigma2_p - t / h2

        mask = tempAlpha > 5e-7
        Alpha_new[mask] = tempAlpha[mask]
        Gamma_new[mask] = tempGamma[mask]

        # Damped update
        Alpha = delta * Alpha_new + (1 - delta) * Alpha
        Gamma = delta * Gamma_new + (1 - delta) * Gamma

        # Recompute posterior
        Sigma_q = np.linalg.inv(HtH / Nv + np.diag(Alpha))
        Mu_q = Sigma_q @ (Hty / Nv + Gamma)

    # Hard decision on posterior mean
    indices = np.argmin(np.abs(Mu_q[:, None] - sym[None, :]), axis=1)
    return sym[indices]
