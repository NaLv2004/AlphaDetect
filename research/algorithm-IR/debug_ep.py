"""Debug EP detector."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from evolution.ir_pool import build_ir_pool
from evolution.mimo_evaluator import generate_mimo_sample, qam16_constellation

rng = np.random.default_rng(42)
Nr, Nt, mod = 16, 16, 16
constellation = qam16_constellation()
snr_db = 24.0
H, x_true, y, sigma2 = generate_mimo_sample(Nr, Nt, constellation, snr_db, rng)

# Manual EP with debug prints
HtH = H.conj().T @ H
Hty = H.conj().T @ y

alpha = np.ones(Nt) * 2.0
gamma = np.zeros(Nt, dtype=complex)

Sigma_q = np.linalg.inv(HtH / max(sigma2, 1e-30) + np.diag(alpha))
mu_q = Sigma_q @ (Hty / max(sigma2, 1e-30) + gamma)

print(f"sigma2 = {sigma2:.6f}")
print(f"Initial mu_q[:4] = {mu_q[:4]}")
print(f"Initial sig[:4] = {np.real(np.diag(Sigma_q))[:4]}")

# Do 1 EP iteration 
sig = np.real(np.diag(Sigma_q))
for i in range(min(4, Nt)):
    h2 = float(sig[i]) / max(1.0 - float(sig[i]) * float(alpha[i]), 1e-30)
    h2 = max(h2, 1e-30)
    t = h2 * (mu_q[i] / max(float(sig[i]), 1e-30) - gamma[i])
    
    # Site update
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
    
    print(f"\nSymbol {i}: sig={sig[i]:.6f}, h2={h2:.6f}, t={t:.4f}")
    print(f"  mu_p={mu_p:.4f}, sigma2_p={sigma2_p:.8f}")
    print(f"  new_alpha={new_alpha:.4f}, new_gamma={new_gamma:.4f}")
    print(f"  alpha[i]={alpha[i]:.4f}, check={new_alpha > 1e-6}")

# Run 20 iterations and check
damping = 0.5
for it in range(20):
    sig = np.real(np.diag(Sigma_q))
    for i in range(Nt):
        h2 = float(sig[i]) / max(1.0 - float(sig[i]) * float(alpha[i]), 1e-30)
        h2 = max(h2, 1e-30)
        t = h2 * (mu_q[i] / max(float(sig[i]), 1e-30) - gamma[i])
        
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
        
        if new_alpha > 1e-6:
            alpha[i] = damping * new_alpha + (1.0 - damping) * float(alpha[i])
            gamma[i] = damping * new_gamma + (1.0 - damping) * gamma[i]
    
    Sigma_q = np.linalg.inv(HtH / max(sigma2, 1e-30) + np.diag(alpha))
    mu_q = Sigma_q @ (Hty / max(sigma2, 1e-30) + gamma)

print(f"\nAfter 20 iters:")
print(f"mu_q[:4] = {mu_q[:4]}")
print(f"alpha[:4] = {alpha[:4]}")
print(f"gamma[:4] = {gamma[:4]}")

# Final decision
x_hat = np.zeros(Nt, dtype=complex)
for i in range(Nt):
    dists = np.abs(constellation - mu_q[i]) ** 2
    x_hat[i] = constellation[np.argmin(dists)]
print(f"\nx_hat[:4] = {x_hat[:4]}")
print(f"x_true[:4] = {x_true[:4]}")
ser = np.mean(np.abs(x_hat - x_true) > 1e-6)
print(f"SER = {ser:.4f}")
