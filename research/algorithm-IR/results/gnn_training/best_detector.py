def admm_detector(H: any, y: any, sigma2: float, constellation: any):
    Nt = H.shape[1]
    x = np.zeros(Nt, dtype=complex)
    z = np.zeros(Nt, dtype=complex)
    u = np.zeros(Nt, dtype=complex)
    i = 0
    while i < 30:
        __fii_admm_H_9 = H
        __fii_admm_y_2 = y
        __fii_admm_sigma2_4 = sigma2
        __fii_admm_z_8 = z
        __fii_admm_u_5 = u
        __fii_admm_Nt_1 = __fii_admm_H_9.shape[1]
        __fii_admm_rho_6 = 1.0
        __fii_admm_G_3 = __fii_admm_H_9.conj().T @ __fii_admm_H_9 + __fii_admm_sigma2_4 + __fii_admm_rho_6 * np.eye(__fii_admm_Nt_1)
        __fii_admm_rhs_7 = __fii_admm_H_9.conj().T @ __fii_admm_y_2 + __fii_admm_rho_6 * __fii_admm_z_8 - __fii_admm_u_5
        x = np.linalg.solve(__fii_admm_G_3, __fii_admm_rhs_7)
        __fii_admm_x_13 = x
        __fii_admm_u_14 = u
        __fii_admm_constellation_12 = constellation
        __fii_admm_Nt_10 = len(__fii_admm_x_13)
        __fii_admm_z_16 = np.zeros(__fii_admm_Nt_10, dtype=complex)
        __fii_admm_v_17 = __fii_admm_x_13 + __fii_admm_u_14
        __fii_admm_i_11 = 0
        while __fii_admm_i_11 < __fii_admm_Nt_10:
            __fii_admm_dists_15 = np.abs(__fii_admm_constellation_12 - __fii_admm_v_17[__fii_admm_i_11]) ** 2
            __fii_admm_z_16[__fii_admm_i_11] = __fii_admm_constellation_12[np.argmin(__fii_admm_dists_15)]
            __fii_admm_i_11 = __fii_admm_i_11 + 1
        z = __fii_admm_z_16
        __fii_admm_u_20 = u
        __fii_admm_x_18 = x
        __fii_admm_z_19 = z
        u = __fii_admm_u_20 + __fii_admm_x_18 - __fii_admm_z_19
        i = i + 1
    __fii_admm_x_soft_23 = z
    __fii_admm_constellation_22 = constellation
    __fii_admm_x_hat_24 = np.zeros(len(__fii_admm_x_soft_23), dtype=complex)
    __fii_admm_i_21 = 0
    while __fii_admm_i_21 < len(__fii_admm_x_soft_23):
        __fii_admm_dists_25 = np.abs(__fii_admm_constellation_22 - __fii_admm_x_soft_23[__fii_admm_i_21]) ** 2
        __fii_admm_x_hat_24[__fii_admm_i_21] = __fii_admm_constellation_22[np.argmin(__fii_admm_dists_25)]
        __fii_admm_i_21 = __fii_admm_i_21 + 1
    x_hat = __fii_admm_x_hat_24
    return x_hat