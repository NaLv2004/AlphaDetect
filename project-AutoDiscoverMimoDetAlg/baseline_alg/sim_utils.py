"""
Simulation utilities for MIMO detection baseline.
Provides QAM modulation, channel generation, noise model, and bit mapping
matching the MATLAB reference implementation exactly.
"""
import numpy as np


def get_qam_params(mod_type):
    """Get PAM constellation and normalization factor.
    mod_type: bits per complex symbol (2=QPSK, 4=16QAM, 6=64QAM, 8=256QAM)
    Returns: (sym, norm_f) where sym is the normalized PAM constellation.
    """
    if mod_type == 2:
        norm_f = 1.0 / np.sqrt(2)
        s = np.array([-1, 1], dtype=float)
    elif mod_type == 4:
        norm_f = 1.0 / np.sqrt(10)
        s = np.array([-1, -3, 1, 3], dtype=float)
    elif mod_type == 6:
        norm_f = 1.0 / np.sqrt(42)
        s = np.array([-3, -1, -5, -7, 3, 1, 5, 7], dtype=float)
    elif mod_type == 8:
        norm_f = 1.0 / np.sqrt(170)
        s = np.array([-5, -7, -3, -1, -11, -9, -13, -15, 5, 7, 3, 1, 11, 9, 13, 15], dtype=float)
    else:
        raise ValueError(f"Unsupported mod_type: {mod_type}")
    sym = s * norm_f
    return sym, norm_f


def gray_map_qam(bits, mod_type, sym):
    """Map bit vector to complex QAM symbols using Gray coding.
    bits:     1D array of length mod_type * n_symbols
    mod_type: bits per complex symbol
    sym:      PAM constellation (from get_qam_params)
    Returns:  complex symbol vector of length n_symbols
    """
    bits_per_pam = mod_type // 2
    n_complex = len(bits) // mod_type
    bits_2d = bits.reshape(n_complex, mod_type)
    re_bits = bits_2d[:, :bits_per_pam]
    im_bits = bits_2d[:, bits_per_pam:]
    powers = 2 ** np.arange(bits_per_pam - 1, -1, -1)
    re_idx = (re_bits @ powers).astype(int)
    im_idx = (im_bits @ powers).astype(int)
    return sym[re_idx] + 1j * sym[im_idx]


def gray_demap(symest_real, sym, tx_ant):
    """Demap real-valued PAM symbol estimates to bits.
    symest_real: 2*tx_ant real-valued estimates [Re; Im]
    sym:         PAM constellation
    tx_ant:      number of transmit antennas
    Returns:     bit vector of length tx_ant * 2 * bits_per_pam
    """
    bits_per_pam = int(np.log2(len(sym)))
    n_real = len(symest_real)
    # Find closest constellation point
    indices = np.argmin(np.abs(symest_real[:, None] - sym[None, :]), axis=1)
    # Convert indices to binary (MSB first)
    shift = np.arange(bits_per_pam - 1, -1, -1)
    bits_matrix = ((indices[:, None] >> shift) & 1).astype(int)
    # Split real / imaginary parts
    re_bits = bits_matrix[:tx_ant]       # tx_ant x bits_per_pam
    im_bits = bits_matrix[tx_ant:]       # tx_ant x bits_per_pam
    # Combine per antenna: [re_bits | im_bits]
    combined = np.hstack([re_bits, im_bits])  # tx_ant x (2*bits_per_pam)
    # Column-major flatten (matches MATLAB's (:) on transposed matrix)
    return combined.T.flatten(order='F')


def acwgn_eb_n0(tx_sig, es, snr_db, nt, nr, mc, fec_rate):
    """Add AWGN noise based on Eb/N0.
    Eb/N0 = Es*Nr / (sigma2 * Mc * FECRate)
    => sigma2 = Es*Nr / (10^(SNR/10) * Mc * FECRate)

    Parameters:
        tx_sig:   noiseless received signal (complex, Nr x 1)
        es:       transmit symbol energy
        snr_db:   Eb/N0 in dB
        nt:       number of TX antennas
        nr:       number of RX antennas
        mc:       bits per symbol (ModType)
        fec_rate: FEC code rate (1 for uncoded)
    Returns:
        (rx_sig, Nv): noisy signal and noise variance
    """
    tx_e = es * nr * nt
    sigma2 = tx_e / (10 ** (snr_db / 10) * mc * fec_rate * nt)
    noise = np.sqrt(sigma2 / 2) * (np.random.randn(*tx_sig.shape) +
                                    1j * np.random.randn(*tx_sig.shape))
    rx_sig = tx_sig + noise
    return rx_sig, sigma2


def generate_channel(nr, nt):
    """Generate i.i.d. Rayleigh fading channel (Nr x Nt complex)."""
    return np.sqrt(0.5) * (np.random.randn(nr, nt) + 1j * np.random.randn(nr, nt))


def complex_to_real(H):
    """Convert complex channel matrix to real-valued block form.
    [Re(H), -Im(H); Im(H), Re(H)]
    """
    return np.block([
        [np.real(H), -np.imag(H)],
        [np.imag(H),  np.real(H)]
    ])
