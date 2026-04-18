"""MIMO channel simulation and dataset generation.

Generates random MIMO channel realizations, transmitted symbols,
and received vectors for use in detector evaluation.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


def qam16_constellation() -> np.ndarray:
    """Normalized 16-QAM constellation."""
    levels = np.array([-3, -1, 1, 3], dtype=np.float64)
    norm = np.sqrt(10.0)
    points = []
    for re in levels:
        for im in levels:
            points.append(complex(re / norm, im / norm))
    return np.array(points, dtype=np.complex128)


def qpsk_constellation() -> np.ndarray:
    """Normalized QPSK constellation."""
    norm = np.sqrt(2.0)
    return np.array([
        complex(1 / norm, 1 / norm),
        complex(1 / norm, -1 / norm),
        complex(-1 / norm, 1 / norm),
        complex(-1 / norm, -1 / norm),
    ], dtype=np.complex128)


def get_constellation(mod_order: int) -> np.ndarray:
    """Return constellation for given modulation order."""
    if mod_order == 4:
        return qpsk_constellation()
    elif mod_order == 16:
        return qam16_constellation()
    else:
        raise ValueError(f"Unsupported mod_order: {mod_order}")


@dataclass
class MIMODataset:
    """Collection of MIMO channel samples for evaluation."""
    H: np.ndarray        # (n_samples, Nr, Nt) complex channel matrices
    y: np.ndarray        # (n_samples, Nr) complex received vectors
    x_true: np.ndarray   # (n_samples, Nt) complex transmitted symbols
    noise_var: np.ndarray # (n_samples,) per-sample noise variance
    snr_db: float        # SNR in dB used to generate the dataset


def generate_dataset(
    n_samples: int = 200,
    Nt: int = 16,
    Nr: int = 16,
    mod_order: int = 16,
    snr_db: float = 24.0,
    rng: np.random.Generator | None = None,
) -> MIMODataset:
    """Generate a MIMO dataset.

    Model: y = H @ x + n, with i.i.d. Rayleigh fading H,
    uniform random QAM symbols x, and AWGN noise n.

    Args:
        n_samples: Number of channel realizations.
        Nt: Number of transmit antennas.
        Nr: Number of receive antennas.
        mod_order: Modulation order (4=QPSK, 16=16QAM).
        snr_db: Signal-to-noise ratio in dB.
        rng: NumPy random generator.

    Returns:
        MIMODataset with H, y, x_true, noise_var arrays.
    """
    if rng is None:
        rng = np.random.default_rng()

    constellation = get_constellation(mod_order)

    # SNR → noise variance (per-component)
    # SNR = E[|x|^2] / sigma^2, E[|x|^2] = 1 for normalized constellation
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_var_scalar = 1.0 / snr_linear

    H = np.zeros((n_samples, Nr, Nt), dtype=np.complex128)
    y = np.zeros((n_samples, Nr), dtype=np.complex128)
    x_true = np.zeros((n_samples, Nt), dtype=np.complex128)
    noise_var = np.full(n_samples, noise_var_scalar, dtype=np.float64)

    for s in range(n_samples):
        # Rayleigh fading channel
        H[s] = (rng.standard_normal((Nr, Nt)) +
                1j * rng.standard_normal((Nr, Nt))) / np.sqrt(2.0)

        # Random transmitted symbols
        indices = rng.integers(0, mod_order, size=Nt)
        x_true[s] = constellation[indices]

        # Noise
        n = (rng.standard_normal(Nr) +
             1j * rng.standard_normal(Nr)) * np.sqrt(noise_var_scalar / 2.0)

        # Received signal
        y[s] = H[s] @ x_true[s] + n

    return MIMODataset(H=H, y=y, x_true=x_true,
                       noise_var=noise_var, snr_db=snr_db)


def dataset_to_dict(ds: MIMODataset) -> dict[str, np.ndarray]:
    """Convert MIMODataset to dict for C++ evaluator."""
    return {
        'H': ds.H,
        'y': ds.y,
        'x_true': ds.x_true,
        'noise_var': ds.noise_var,
    }
