"""Level-0 atomic operations for MIMO detection.

Contains numpy-based implementations of all primitive operations used by
higher-level algorithms.  Each function is a self-contained callable that
can be used directly or registered in the primitive pool.

Categories:
  - Scalar ops
  - Vector ops
  - Matrix ops
  - Complex ops
  - Statistical / probability ops
"""

from __future__ import annotations

import numpy as np


# ── Helper constants ──────────────────────────────────────────────────────

_EPS = 1e-30

# ═══════════════════════════════════════════════════════════════════════════
# Scalar ops
# ═══════════════════════════════════════════════════════════════════════════

def s_add(a: float, b: float) -> float:
    return a + b

def s_sub(a: float, b: float) -> float:
    return a - b

def s_mul(a: float, b: float) -> float:
    return a * b

def s_div(a: float, b: float) -> float:
    return a / b if abs(b) > _EPS else 0.0

def s_neg(a: float) -> float:
    return -a

def s_abs(a: float) -> float:
    return abs(a)

def s_sqrt(a: float) -> float:
    return np.sqrt(max(a, 0.0))

def s_log(a: float) -> float:
    return np.log(max(a, _EPS))

def s_exp(a: float) -> float:
    return np.exp(np.clip(a, -500, 500))

def s_inv(a: float) -> float:
    return 1.0 / a if abs(a) > _EPS else 0.0

def s_max(a: float, b: float) -> float:
    return max(a, b)

def s_min(a: float, b: float) -> float:
    return min(a, b)

def s_clamp(a: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, a))

def s_sigmoid(a: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(a, -500, 500)))

def s_relu(a: float) -> float:
    return max(a, 0.0)

def s_tanh(a: float) -> float:
    return np.tanh(a)


# ═══════════════════════════════════════════════════════════════════════════
# Complex scalar ops
# ═══════════════════════════════════════════════════════════════════════════

def c_add(a: complex, b: complex) -> complex:
    return a + b

def c_sub(a: complex, b: complex) -> complex:
    return a - b

def c_mul(a: complex, b: complex) -> complex:
    return a * b

def c_conj(a: complex) -> complex:
    return np.conj(a)

def c_abs2(a: complex) -> float:
    """Squared magnitude |a|²."""
    return float(np.real(a * np.conj(a)))

def c_abs(a: complex) -> float:
    return float(np.abs(a))

def c_real(a: complex) -> float:
    return float(np.real(a))

def c_imag(a: complex) -> float:
    return float(np.imag(a))

def c_phase(a: complex) -> float:
    return float(np.angle(a))


# ═══════════════════════════════════════════════════════════════════════════
# Vector ops
# ═══════════════════════════════════════════════════════════════════════════

def v_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b

def v_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a - b

def v_scale(a: np.ndarray, s: complex) -> np.ndarray:
    return a * s

def v_dot(a: np.ndarray, b: np.ndarray) -> complex:
    """Conjugate dot product a^H b."""
    return np.vdot(a, b)

def v_norm(a: np.ndarray) -> float:
    return float(np.linalg.norm(a))

def v_norm_sq(a: np.ndarray) -> float:
    return float(np.real(np.vdot(a, a)))

def v_conj(a: np.ndarray) -> np.ndarray:
    return np.conj(a)

def v_elementwise_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a * b

def v_elementwise_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(np.abs(b) > _EPS, a / b, 0.0)

def v_slice(a: np.ndarray, start: int, end: int) -> np.ndarray:
    return a[start:end]

def v_zeros(n: int, dtype: type = complex) -> np.ndarray:
    return np.zeros(n, dtype=dtype)

def v_ones(n: int, dtype: type = complex) -> np.ndarray:
    return np.ones(n, dtype=dtype)

def v_sum(a: np.ndarray) -> complex:
    return np.sum(a)

def v_max(a: np.ndarray) -> float:
    return float(np.max(np.real(a)))

def v_argmin(a: np.ndarray) -> int:
    return int(np.argmin(np.real(a)))

def v_argmax(a: np.ndarray) -> int:
    return int(np.argmax(np.real(a)))

def v_sort_indices(a: np.ndarray) -> np.ndarray:
    """Return indices that would sort the array (ascending by real part)."""
    return np.argsort(np.real(a))


# ═══════════════════════════════════════════════════════════════════════════
# Matrix ops
# ═══════════════════════════════════════════════════════════════════════════

def m_add(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A + B

def m_sub(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A - B

def m_mul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Matrix product A @ B."""
    return A @ B

def m_scale(A: np.ndarray, s: complex) -> np.ndarray:
    return A * s

def m_conj_transpose(A: np.ndarray) -> np.ndarray:
    """Conjugate transpose A^H."""
    return A.conj().T

def m_transpose(A: np.ndarray) -> np.ndarray:
    return A.T

def m_matvec(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Matrix-vector product A @ x."""
    return A @ x

def m_gram(A: np.ndarray) -> np.ndarray:
    """Gram matrix A^H A."""
    return A.conj().T @ A

def m_diag(v: np.ndarray) -> np.ndarray:
    """Create diagonal matrix from vector."""
    return np.diag(v)

def m_diag_extract(A: np.ndarray) -> np.ndarray:
    """Extract diagonal as vector."""
    return np.diag(A)

def m_eye(n: int, dtype: type = complex) -> np.ndarray:
    return np.eye(n, dtype=dtype)

def m_zeros(m: int, n: int, dtype: type = complex) -> np.ndarray:
    return np.zeros((m, n), dtype=dtype)

def m_inv(A: np.ndarray) -> np.ndarray:
    """Matrix inverse (regularised if near-singular)."""
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        n = A.shape[0]
        return np.linalg.inv(A + 1e-10 * np.eye(n, dtype=A.dtype))

def m_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Ax = b.  Regularises if near-singular."""
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        n = A.shape[0]
        return np.linalg.solve(
            A + 1e-10 * np.eye(n, dtype=A.dtype), b,
        )

def m_qr(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Economy QR decomposition."""
    return np.linalg.qr(A, mode="reduced")

def m_cholesky(A: np.ndarray) -> np.ndarray:
    """Cholesky factor L such that A = L L^H.  Regularises if needed."""
    try:
        return np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        n = A.shape[0]
        return np.linalg.cholesky(
            A + 1e-10 * np.eye(n, dtype=A.dtype),
        )

def m_det(A: np.ndarray) -> complex:
    return np.linalg.det(A)

def m_trace(A: np.ndarray) -> complex:
    return np.trace(A)

def m_col(A: np.ndarray, j: int) -> np.ndarray:
    """Extract column j."""
    return A[:, j].copy()

def m_row(A: np.ndarray, i: int) -> np.ndarray:
    """Extract row i."""
    return A[i, :].copy()

def m_delete_col(A: np.ndarray, j: int) -> np.ndarray:
    """Delete column j."""
    return np.delete(A, j, axis=1)

def m_rank1_update(A: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """A + u v^H."""
    return A + np.outer(u, np.conj(v))

def m_schur_complement(M: np.ndarray, i: int) -> np.ndarray:
    """Schur complement eliminating row/col i from M."""
    n = M.shape[0]
    idx = [k for k in range(n) if k != i]
    A = M[np.ix_(idx, idx)]
    b = M[np.ix_(idx, [i])]
    c = M[np.ix_([i], idx)]
    d = M[i, i]
    if abs(d) < _EPS:
        return A
    return A - (b @ c) / d


# ═══════════════════════════════════════════════════════════════════════════
# Statistical / probability ops
# ═══════════════════════════════════════════════════════════════════════════

def stat_mean(a: np.ndarray) -> complex:
    return np.mean(a)

def stat_var(a: np.ndarray) -> float:
    return float(np.var(a))

def stat_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(logits - np.max(np.real(logits)))
    return e / np.sum(e)

def stat_log_sum_exp(a: np.ndarray) -> float:
    """Log-sum-exp with numerical stability."""
    mx = float(np.max(np.real(a)))
    return mx + float(np.log(np.sum(np.exp(a - mx))))

def stat_gaussian_pdf(x: float, mu: float, var: float) -> float:
    """Univariate Gaussian PDF."""
    if var < _EPS:
        return 1.0 if abs(x - mu) < _EPS else 0.0
    return float(np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2 * np.pi * var))

def stat_gaussian_logpdf(x: float, mu: float, var: float) -> float:
    """Univariate Gaussian log-PDF."""
    if var < _EPS:
        return 0.0 if abs(x - mu) < _EPS else -1e10
    return float(-0.5 * np.log(2 * np.pi * var) - 0.5 * (x - mu) ** 2 / var)

def stat_weighted_mean(values: np.ndarray, weights: np.ndarray) -> complex:
    """Weighted mean."""
    w_sum = np.sum(weights)
    if abs(w_sum) < _EPS:
        return np.mean(values)
    return np.sum(values * weights) / w_sum

def stat_weighted_var(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted variance."""
    mu = stat_weighted_mean(values, weights)
    w_sum = np.sum(weights)
    if abs(w_sum) < _EPS:
        return float(np.var(values))
    return float(np.real(np.sum(weights * (values - mu) * np.conj(values - mu)) / w_sum))


# ═══════════════════════════════════════════════════════════════════════════
# Primitive registry — maps names → callables for evolution sampling
# ═══════════════════════════════════════════════════════════════════════════

PRIMITIVE_REGISTRY: dict[str, callable] = {}

def _register_all() -> None:
    import inspect
    _mod = inspect.getmodule(_register_all)
    for name, fn in inspect.getmembers(_mod, inspect.isfunction):
        if name.startswith(("s_", "c_", "v_", "m_", "stat_")):
            PRIMITIVE_REGISTRY[name] = fn

_register_all()
