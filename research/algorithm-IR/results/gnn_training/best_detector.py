def kbest(H: mat_cx, y: vec_cx, sigma2: float, constellation: vec_cx):
    Nr = H.shape[0]
    Nt = H.shape[1]
    Q = (np.linalg.qr(H))[0]
    R = (np.linalg.qr(H))[1]
    y_tilde = (Q.conj().T @ y)
    root = _make_tree_node((Nt - 1), [], 0.0)
    candidates = [root]
    level = (Nt - 1)
    while level >= 0:
        new_candidates = []
        ci = 0
        while ci < len(candidates):
            node = candidates[ci]
            node.level = level
            __fii_kbest_node_10 = node
            __fii_kbest_y_tilde_13 = y_tilde
            __fii_kbest_R_4 = R
            __fii_kbest_constellation_7 = constellation
            __fii_kbest_level_12 = __fii_kbest_node_10.level
            __fii_kbest_Nt_1 = __fii_kbest_R_4.shape[1]
            __fii_kbest_children_5 = []
            __fii_kbest_interf_14 = (0.0 + 0j)
            __fii_kbest_j_rel_8 = 0
            while __fii_kbest_j_rel_8 < len(__fii_kbest_node_10.symbols):
                __fii_kbest_j_11 = ((__fii_kbest_Nt_1 - 1) - __fii_kbest_j_rel_8)
                __fii_kbest_interf_14 = (__fii_kbest_interf_14 + (__fii_kbest_R_4[(__fii_kbest_level_12, __fii_kbest_j_11)] * __fii_kbest_node_10.symbols[__fii_kbest_j_rel_8]))
                __fii_kbest_j_rel_8 = (__fii_kbest_j_rel_8 + 1)
            __fii_kbest_ci_2 = 0
            while __fii_kbest_ci_2 < len(__fii_kbest_constellation_7):
                __fii_kbest_sym_3 = __fii_kbest_constellation_7[__fii_kbest_ci_2]
                __fii_kbest_residual_16 = ((__fii_kbest_y_tilde_13[__fii_kbest_level_12] - (__fii_kbest_R_4[(__fii_kbest_level_12, __fii_kbest_level_12)] * __fii_kbest_sym_3)) - __fii_kbest_interf_14)
                __fii_kbest_local_cost_9 = float((np.abs(__fii_kbest_residual_16) ** 2))
                __fii_kbest_total_6 = (__fii_kbest_node_10.cost + __fii_kbest_local_cost_9)
                __fii_kbest_child_15 = _make_tree_node((__fii_kbest_level_12 - 1), (__fii_kbest_node_10.symbols + [__fii_kbest_sym_3]), __fii_kbest_total_6)
                __fii_kbest_children_5.append(__fii_kbest_child_15)
                __fii_kbest_ci_2 = (__fii_kbest_ci_2 + 1)
            children = __fii_kbest_children_5
            new_candidates = (new_candidates + children)
            ci = (ci + 1)
        __fii_kbest_candidates_19 = new_candidates
        __fii_kbest_K_20 = 16
        __fii_kbest_n_21 = len(__fii_kbest_candidates_19)
        __fii_kbest_i_17 = 0
        while __fii_kbest_i_17 < __fii_kbest_n_21:
            __fii_kbest_j_22 = (__fii_kbest_i_17 + 1)
            while __fii_kbest_j_22 < __fii_kbest_n_21:
                if __fii_kbest_candidates_19[__fii_kbest_j_22].cost < __fii_kbest_candidates_19[__fii_kbest_i_17].cost:
                    __fii_kbest_tmp_18 = __fii_kbest_candidates_19[__fii_kbest_i_17]
                    __fii_kbest_candidates_19[__fii_kbest_i_17] = __fii_kbest_candidates_19[__fii_kbest_j_22]
                    __fii_kbest_candidates_19[__fii_kbest_j_22] = __fii_kbest_tmp_18
                __fii_kbest_j_22 = (__fii_kbest_j_22 + 1)
            __fii_kbest_i_17 = (__fii_kbest_i_17 + 1)
        candidates = __fii_kbest_candidates_19[:__fii_kbest_K_20]
        level = (level - 1)
    if len(candidates) == 0:
        return np.zeros(Nt, dtype=complex)
    best = candidates[0]
    bi = 1
    while bi < len(candidates):
        if candidates[bi].cost < best.cost:
            best = candidates[bi]
        bi = (bi + 1)
    return _reverse_syms(best.symbols, Nt)