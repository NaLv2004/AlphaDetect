def _slot_regularizer_op_24(G, sigma2):
    n = G.shape[0]
    return G + sigma2 * np.eye(n)

def _slot_hard_decision_op_25(x_soft, constellation):
    x_hat = np.zeros(len(x_soft), dtype=complex)
    i = 0
    while i < len(x_soft):
        dists = np.abs(constellation - x_soft[i]) ** 2
        x_hat[i] = constellation[np.argmin(dists)]
        i = i + 1
    return x_hat


def lmmse(H, y, sigma2, constellation):
    Nt = H.shape[1]
    G = H.conj().T @ H
    G_reg = _slot_regularizer_op_24(G, sigma2)
    rhs = H.conj().T @ y
    x_soft = np.linalg.solve(G_reg, rhs)
    x_hat = _slot_hard_decision_op_25(x_soft, constellation)
    return x_hat