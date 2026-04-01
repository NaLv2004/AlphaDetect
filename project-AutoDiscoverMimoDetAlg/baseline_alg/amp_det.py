"""
AMP (Approximate Message Passing) MIMO Detector.
Iterative detector with Onsager correction and damped variance tracking.
Matches the MATLAB reference implementation.
"""
import numpy as np


def amp_detect(tx_ant, rx_ant, slen, y, H, Nv, sym, delta, iter_num):
    """AMP detector (real-valued formulation).
    tx_ant:   number of TX antennas
    rx_ant:   number of RX antennas
    slen:     constellation size
    y:        received signal (2*rx_ant,)
    H:        real-valued channel (2*rx_ant x 2*tx_ant)
    Nv:       noise variance (complex)
    sym:      PAM constellation (slen,)
    delta:    damping factor (unused in AMP, internal damping used)
    iter_num: number of iterations
    Returns:  detected symbol vector (2*tx_ant,)
    """
    N = 2 * tx_ant
    M = 2 * rx_ant

    G = H.T @ H                                    # (N, N)
    g_diag = np.diag(G)                             # (N,)
    Gtilde = np.eye(N) - np.diag(1.0 / g_diag) @ G # off-diagonal structure
    g = g_diag / M                                  # (N,)
    gtilde = 1.0 / g_diag                           # (N,)

    yMF = H.T @ y                                   # matched filter output (N,)
    yMFtilde = gtilde * yMF                         # diagonally scaled (N,)

    shat = np.zeros(N)
    tau_s = np.ones(N)
    tau_p = np.dot(g, tau_s)                        # scalar
    tau_z = (tau_p + Nv) * gtilde                   # (N,)
    theta_tau_s = 0.5
    theta_tau_z = 0.5

    z = yMFtilde.copy()

    for k in range(iter_num):
        # Distance to each constellation point
        input_symM = np.abs(z[:, None] - sym[None, :]) ** 2   # (N, slen)
        expo = np.min(input_symM, axis=1, keepdims=True) - input_symM  # normalized

        # Posterior pdf
        pdf = np.exp(expo / tau_z[:, None])

        # Weights (posterior probabilities)
        w = pdf / np.sum(pdf, axis=1, keepdims=True)

        # Posterior mean
        shat_old = shat.copy()
        shat = w @ sym                                         # (N,)

        # Posterior variance
        tau_s = np.sum(w * np.abs(sym[None, :] - shat[:, None]) ** 2, axis=1)

        # Damped tau_p update
        tau_p_old = tau_p
        tau_p = theta_tau_s * np.dot(g, tau_s) + (1 - theta_tau_s) * tau_p

        # Onsager correction term
        v = tau_p / (tau_p_old + Nv) * (z - shat_old)

        # Damped tau_z update
        tau_z = theta_tau_z * (tau_p + Nv) * gtilde + (1 - theta_tau_z) * tau_z

        # Update estimate
        z = yMFtilde + Gtilde @ shat + v

    # Hard decision on final z
    indices = np.argmin(np.abs(z[:, None] - sym[None, :]), axis=1)
    return sym[indices]
