def _slot_expand_op_149(node, y_tilde, R, constellation):
    level = node.level
    Nt = R.shape[1]
    children = []
    interf = 0.0 + 0.0j
    j_rel = 0
    while j_rel < len(node.symbols):
        j = Nt - 1 - j_rel
        interf = interf + R[level, j] * node.symbols[j_rel]
        j_rel = j_rel + 1
    ci = 0
    while ci < len(constellation):
        sym = constellation[ci]
        residual = y_tilde[level] - R[level, level] * sym - interf
        local_cost = float(np.abs(residual) ** 2)
        total = node.cost + local_cost
        child = _make_tree_node(level - 1, node.symbols + [sym], total)
        children.append(child)
        ci = ci + 1
    return children

def _slot_prune_op_150(candidates, K):
    n = len(candidates)
    i = 0
    while i < n:
        j = i + 1
        while j < n:
            if candidates[j].cost < candidates[i].cost:
                tmp = candidates[i]
                candidates[i] = candidates[j]
                candidates[j] = tmp
            j = j + 1
        i = i + 1
    return candidates[:K]


def kbest(H, y, sigma2, constellation):
    Nr = H.shape[0]
    Nt = H.shape[1]
    Q = np.linalg.qr(H)[0]
    R = np.linalg.qr(H)[1]
    y_tilde = Q.conj().T @ y
    root = _make_tree_node(Nt - 1, [], 0.0)
    candidates = [root]
    level = Nt - 1
    while level >= 0:
        new_candidates = []
        ci = 0
        while ci < len(candidates):
            node = candidates[ci]
            node.level = level
            children = _slot_expand_op_149(node, y_tilde, R, constellation)
            new_candidates = new_candidates + children
            ci = ci + 1
        candidates = _slot_prune_op_150(new_candidates, 16)
        level = level - 1
    if len(candidates) == 0:
        return np.zeros(Nt, dtype=complex)
    best = candidates[0]
    bi = 1
    while bi < len(candidates):
        if candidates[bi].cost < best.cost:
            best = candidates[bi]
        bi = bi + 1
    return _reverse_syms(best.symbols, Nt)