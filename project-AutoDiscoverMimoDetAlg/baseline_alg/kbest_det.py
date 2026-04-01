"""
K-Best Tree Search MIMO Detector.
Breadth-first tree search with K surviving paths per layer.
Matches the MATLAB reference implementation.
"""
import numpy as np


def kbest_detect(R, sym, z, k):
    """K-Best tree search detector.
    R:   upper triangular matrix (N x N) from QR decomposition
    sym: PAM constellation (1D array, slen elements)
    z:   rotated received vector (N,)  [z = Q^T y]
    k:   number of surviving paths per layer
    Returns: detected symbol vector (N,) in natural order
    """
    N = len(z)
    con_size = len(sym)
    current_paths = None

    for step in range(1, N + 1):
        if current_paths is None:
            # First layer: enumerate all constellation points
            candidate_paths = np.arange(con_size).reshape(1, -1)
        else:
            num_survivors = current_paths.shape[1]
            # Expand each survivor with all constellation points
            new_row = np.tile(np.arange(con_size), num_survivors)
            old_rows = np.repeat(current_paths, con_size, axis=1)
            candidate_paths = np.vstack([new_row.reshape(1, -1), old_rows])

        # Map indices to symbol values
        sym_values = sym[candidate_paths]           # (step, num_candidates)

        # R submatrix and z subvector for current processing window
        z_sub = z[N - step:]                        # (step,)
        R_sub = R[N - step:, N - step:]             # (step, step)

        # Partial Euclidean distance
        residual = z_sub[:, None] - R_sub @ sym_values
        PED = np.sum(residual ** 2, axis=0)

        # Retain K best paths
        num_keep = min(k, len(PED))
        if num_keep < len(PED):
            best = np.argpartition(PED, num_keep)[:num_keep]
            best = best[np.argsort(PED[best])]
        else:
            best = np.argsort(PED)

        current_paths = candidate_paths[:, best]

    # Return the best path's symbols (in natural order: x[0], ..., x[N-1])
    return sym[current_paths[:, 0]]


def kbest_detect_wrapper(tx_ant, rx_ant, slen, y, H, Nv, sym, delta, iter_num, k=128):
    """Wrapper with same interface as other detectors.
    Performs QR decomposition then calls K-Best search.
    """
    Q, R = np.linalg.qr(H, mode='reduced')   # Q: (2Nr, 2Nt), R: (2Nt, 2Nt)
    z = Q.T @ y                                # (2Nt,)
    return kbest_detect(R, sym, z, k)
