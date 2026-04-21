def _slot_soft_estimate_op_99(x_soft, variance, constellation):
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

def _slot_hard_decision_op_100(x_soft, constellation):
    x_hat = np.zeros(len(x_soft), dtype=complex)
    i = 0
    while i < len(x_soft):
        dists = np.abs(constellation - x_soft[i]) ** 2
        x_hat[i] = constellation[np.argmin(dists)]
        i = i + 1
    return x_hat


def turbo_linear_detector(H, y, sigma2, constellation):
    Nt = H.shape[1]
    G = H.conj().T @ H + sigma2 * np.eye(Nt)
    rhs = H.conj().T @ y
    x = np.linalg.solve(G, rhs)
    var = np.zeros(Nt)
    j = 0
    while j < Nt:
        G_inv_jj = 1.0 / max(float(np.real(G[(j, j)])), 1e-30)
        var[j] = G_inv_jj * sigma2
        j = j + 1
    i = 0
    while i < 5:
        x_soft = _slot_soft_estimate_op_99(x, var, constellation)
        r = y - H @ x_soft
        x = x_soft + np.linalg.solve(G, H.conj().T @ r)
        i = i + 1
    x_hat = _slot_hard_decision_op_100(x, constellation)
    return x_hat