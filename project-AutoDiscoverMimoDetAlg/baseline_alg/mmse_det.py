"""
MMSE (Minimum Mean Square Error) MIMO Detector.
Linear detector: x_hat = (H^T H + Nv I)^{-1} H^T y
"""
import numpy as np


def mmse_detect(tx_ant, rx_ant, slen, y, H, Nv, sym, delta, iter_num):
    """MMSE detector (real-valued formulation).
    tx_ant:   number of TX antennas
    rx_ant:   number of RX antennas
    slen:     constellation size
    y:        received signal (2*rx_ant,)
    H:        real-valued channel (2*rx_ant x 2*tx_ant)
    Nv:       noise variance (complex)
    sym:      PAM constellation (slen,)
    delta:    damping factor (unused)
    iter_num: iterations (unused)
    Returns:  detected symbol vector (2*tx_ant,)
    """
    N = 2 * tx_ant
    S_mmse = np.linalg.solve(H.T @ H + Nv * np.eye(N), H.T @ y)
    indices = np.argmin(np.abs(S_mmse[:, None] - sym[None, :]), axis=1)
    return sym[indices]
