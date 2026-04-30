def kbest(H: mat_cx, y: vec_cx, sigma2: float, constellation: vec_cx):
    Nr = H.shape[0]
    Nt = H.shape[1]
    Q = np.linalg.qr(H)[0]
    R = np.linalg.qr(H)[1]
    y_tilde = (Q.conj().T @ y)
    root = _make_tree_node((Nt - 1), [], 0.0)
    candidates = [root]
    level = (Nt - 1)
    K_keep = 16
    __loop_guard_0 = 0
    while (level >= 0):
        __loop_guard_0 += 1
        if __loop_guard_0 > 100000: break
        new_candidates = []
        ci = 0
        __loop_guard_1 = 0
        while (ci < len(candidates)):
            __loop_guard_1 += 1
            if __loop_guard_1 > 100000: break
            node = candidates[ci]
            node.level = level
            Nt_e = R.shape[1]
            children = []
            interf_e = (0.0 + 0j)
            jr_e = 0
            __loop_guard_2 = 0
            while (jr_e < len(node.symbols)):
                __loop_guard_2 += 1
                if __loop_guard_2 > 100000: break
                j_e = ((Nt_e - 1) - jr_e)
                interf_e = (interf_e + (R[(level, j_e)] * node.symbols[jr_e]))
                jr_e = (jr_e + 1)
            cei = 0
            __loop_guard_3 = 0
            while (cei < len(constellation)):
                __loop_guard_3 += 1
                if __loop_guard_3 > 100000: break
                sym_e = constellation[cei]
                resid_e = ((y_tilde[level] - (R[(level, level)] * sym_e)) - interf_e)
                lc_e = float((np.abs(resid_e) ** 2))
                tot_e = (node.cost + lc_e)
                ch_e = _make_tree_node((level - 1), (node.symbols + [sym_e]), tot_e)
                children.append(ch_e)
                cei = (cei + 1)
            new_candidates = (new_candidates + children)
            ci = (ci + 1)
        n_p = len(new_candidates)
        ip = 0
        __loop_guard_4 = 0
        while (ip < n_p):
            __loop_guard_4 += 1
            if __loop_guard_4 > 100000: break
            jp = (ip + 1)
            __loop_guard_5 = 0
            while (jp < n_p):
                __loop_guard_5 += 1
                if __loop_guard_5 > 100000: break
                if (new_candidates[jp].cost < new_candidates[ip].cost):
                    tmp_p = new_candidates[ip]
                    new_candidates[ip] = new_candidates[jp]
                    new_candidates[jp] = tmp_p
                jp = (jp + 1)
            ip = (ip + 1)
        candidates = new_candidates[:K_keep]
        level = (level - 1)
    if (len(candidates) == 0):
        return np.zeros(Nt, dtype=complex)
    best = candidates[0]
    bi = 1
    __loop_guard_6 = 0
    while (bi < len(candidates)):
        __loop_guard_6 += 1
        if __loop_guard_6 > 100000: break
        if (candidates[bi].cost < best.cost):
            best = candidates[bi]
        bi = (bi + 1)
    return _reverse_syms(best.symbols, Nt)