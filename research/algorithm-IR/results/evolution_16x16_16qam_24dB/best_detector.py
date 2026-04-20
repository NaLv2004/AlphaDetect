def _slot_final_decision_op_197(mu, constellation):
    x_hat = np.zeros(len(mu), dtype=complex)
    i = 0
    while i < len(mu):
        dists = np.abs(constellation - mu[i]) ** 2
        x_hat[i] = constellation[np.argmin(dists)]
        i = i + 1
    return x_hat

def _slot_cavity_op_198(t, h2, gamma_i, alpha_i):
    return t, h2

def _slot_site_update_op_199(t, h2, constellation, gamma_i, alpha_i):
    diffs = constellation - t
    exponents = -np.abs(diffs) ** 2 / max(2.0 * h2, 1e-30)
    max_exp = float(np.max(np.real(exponents)))
    weights = np.exp(np.real(exponents) - max_exp)
    w_sum = max(float(np.sum(weights)), 1e-30)
    weights = weights / w_sum
    mu_p = np.sum(weights * constellation)
    sigma2_p = float(np.real(np.sum(weights * np.abs(constellation) ** 2))) - float(np.abs(mu_p) ** 2)
    sigma2_p = max(sigma2_p, 5e-7)
    new_alpha = 1.0 / sigma2_p - 1.0 / max(h2, 1e-30)
    new_gamma = mu_p / sigma2_p - t / max(h2, 1e-30)
    return new_alpha, new_gamma


def ep(H, y, sigma2, constellation):
    Nr = H.shape[0]
    Nt = H.shape[1]
    alpha = np.ones(Nt) * 2.0
    gamma_ep = np.zeros(Nt, dtype=complex)
    HtH = H.conj().T @ H
    Hty = H.conj().T @ y
    s2inv = 1.0 / max(sigma2, 1e-30)
    Hty_scaled = Hty * s2inv
    HtH_scaled = HtH * s2inv
    rhs = Hty_scaled + gamma_ep
    Sigma_q = np.linalg.inv(HtH_scaled + np.diag(alpha))
    mu_q = Sigma_q @ rhs
    damping = 0.5
    one_m_damp = 1.0 - damping
    it = 0
    while it < 20:
        sig = np.real(np.diag(Sigma_q))
        i = 0
        while i < Nt:
            sig_i = float(sig[i])
            alpha_i = float(alpha[i])
            denom = 1.0 - sig_i * alpha_i
            denom = max(denom, 1e-30)
            h2 = sig_i / denom
            h2 = max(h2, 1e-30)
            sig_inv = 1.0 / max(sig_i, 1e-30)
            ratio = mu_q[i] * sig_inv - gamma_ep[i]
            t = h2 * ratio
            cav_result = _slot_cavity_op_198(t, h2, gamma_ep[i], alpha_i)
            upd_result = _slot_site_update_op_199(cav_result[0], cav_result[1], constellation, gamma_ep[i], alpha_i)
            new_alpha = float(np.real(upd_result[0]))
            new_gamma = upd_result[1]
            if new_alpha > 1e-06:
                alpha[i] = damping * new_alpha + one_m_damp * alpha_i
                gamma_ep[i] = damping * new_gamma + one_m_damp * gamma_ep[i]
            i = i + 1
        rhs = Hty_scaled + gamma_ep
        Sigma_q = np.linalg.inv(HtH_scaled + np.diag(alpha))
        mu_q = Sigma_q @ rhs
        it = it + 1
    x_hat = _slot_final_decision_op_197(mu_q, constellation)
    return x_hat