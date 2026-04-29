def amp(H: mat_cx, y: vec_cx, sigma2: float, constellation: vec_cx):
    Nr = H.shape[0]
    Nt = H.shape[1]
    G_amp = (H.conj().T @ H)
    yMF = (H.conj().T @ y)
    g_diag = np.real(np.diag(G_amp))
    g_diag = np.maximum(g_diag, 1e-30)
    g_scale = (g_diag / max(float(Nr), 1.0))
    gtilde = (1.0 / g_diag)
    yMFtilde = (gtilde * yMF)
    Gtilde = (np.eye(Nt) - (np.diag((1.0 / g_diag)) @ G_amp))
    _g46a8c1___feedback_sic_hard_decision__out = np.zeros(Nt, dtype=complex)
    tau_s = np.ones(Nt)
    z = yMFtilde.copy()
    it_a = 0
    __loop_guard_0 = 0
    while (it_a < 20):
        __loop_guard_0 += 1
        if __loop_guard_0 > 100000: break
        Nt_a = len(_g46a8c1___feedback_sic_hard_decision__out)
        M_a = len(constellation)
        tau_p = (g_scale @ tau_s)
        theta_s = 0.5
        theta_z = 0.5
        tau_p_safe = np.maximum(tau_p, 1e-30)
        tau_p_plus_s2 = (tau_p_safe + sigma2)
        tau_z = (tau_p_plus_s2 * gtilde)
        tau_z = np.maximum(tau_z, 1e-30)
        x_new = np.zeros(Nt_a, dtype=complex)
        x_var = np.zeros(Nt_a)
        i_a = 0
        __loop_guard_1 = 0
        while (i_a < Nt_a):
            __loop_guard_1 += 1
            if __loop_guard_1 > 100000: break
            w_a = np.zeros(M_a)
            j_a = 0
            me_a = (-1e+30)
            __loop_guard_2 = 0
            while (j_a < M_a):
                __loop_guard_2 += 1
                if __loop_guard_2 > 100000: break
                dj_a = float((np.abs((z[i_a] - constellation[j_a])) ** 2))
                e_a = (0.0 - (dj_a / max(float(tau_z[i_a]), 1e-30)))
                if (e_a > me_a):
                    me_a = e_a
                w_a[j_a] = e_a
                j_a = (j_a + 1)
            j_a = 0
            ws_a = 0.0
            __loop_guard_3 = 0
            while (j_a < M_a):
                __loop_guard_3 += 1
                if __loop_guard_3 > 100000: break
                w_a[j_a] = np.exp((w_a[j_a] - me_a))
                ws_a = (ws_a + w_a[j_a])
                j_a = (j_a + 1)
            ws_a = max(ws_a, 1e-30)
            j_a = 0
            __loop_guard_4 = 0
            while (j_a < M_a):
                __loop_guard_4 += 1
                if __loop_guard_4 > 100000: break
                w_a[j_a] = (w_a[j_a] / ws_a)
                j_a = (j_a + 1)
            mu_i = (0.0 + 0j)
            j_a = 0
            __loop_guard_5 = 0
            while (j_a < M_a):
                __loop_guard_5 += 1
                if __loop_guard_5 > 100000: break
                mu_i = (mu_i + (w_a[j_a] * constellation[j_a]))
                j_a = (j_a + 1)
            x_new[i_a] = mu_i
            v_i = 0.0
            j_a = 0
            __loop_guard_6 = 0
            while (j_a < M_a):
                __loop_guard_6 += 1
                if __loop_guard_6 > 100000: break
                v_i = (v_i + (w_a[j_a] * float((np.abs((constellation[j_a] - mu_i)) ** 2))))
                j_a = (j_a + 1)
            x_var[i_a] = v_i
            i_a = (i_a + 1)
        shat_old = _g46a8c1___feedback_sic_hard_decision__out
        z_diff = (z - shat_old)
        one_m_ts = (1.0 - theta_s)
        tp_old_safe = np.maximum(tau_p, 1e-30)
        one_m_tz = (1.0 - theta_z)
        it_a = (it_a + 1)
        tp_denom = (tp_old_safe + sigma2)
        tau_s_new = ((theta_s * x_var) + (one_m_ts * tau_s))
        tau_s = tau_s_new
        tau_p_new = (g_scale @ tau_s_new)
        v_scale = (tau_p_new / tp_denom)
        tp_new_plus_s2 = (tau_p_new + sigma2)
        v_corr = (v_scale * z_diff)
        z_new = ((yMFtilde + (Gtilde @ x_new)) + v_corr)
        tau_z_new = (((theta_z * tp_new_plus_s2) * gtilde) + (one_m_tz * tau_z))
        z = z_new
    x_out = np.zeros(len(z), dtype=complex)
    i_fd = 0
    x_hat_a = _g46a8c1___feedback_sic_hard_decision__out
    __loop_guard_7 = 0
    while (i_fd < len(z)):
        __loop_guard_7 += 1
        if __loop_guard_7 > 100000: break
        d_fd = (np.abs((constellation - z[i_fd])) ** 2)
        x_out[i_fd] = constellation[np.argmin(d_fd)]
        i_fd = (i_fd + 1)
    return x_out