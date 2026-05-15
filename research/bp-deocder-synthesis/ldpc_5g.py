"""5G NR LDPC encoder + Offset-Min-Sum BP decoder.

This module is a faithful Python reproduction of the LDPC pipeline implemented
in ``code_for_reference/cpp-ldpc-simulation/GAMP_Test`` (encode.cpp,
LDPCfunctions.cpp, ratematch.cpp, decode.cpp).  Only the LDPC parts are
replicated; the surrounding MIMO machinery is replaced by a SISO BPSK + AWGN
channel as requested.

Key facts mirrored from the C++ source:

* Base graph and shift values are loaded from ``bg_tables/BG{1,2}S{1..8}.txt``
  (verbatim copies of the C++ workspace files).
* ``getH``/``parityCheckMatrix`` lifting: each entry ``s`` (>=0) lifts to a
  ZxZ right-cyclic permutation matrix ``M[i][(i+s) % Z] = 1``; ``-1`` lifts to
  the all-zero block.
* Encoder (``encode`` / ``nrLdpcEncode``):
    - filler bits (-1 inputs) are zeroed before encoding then re-marked as -1
      in the output codeword (so rate-matching can skip them);
    - the first ``2*Zc`` bits of the lifted codeword are punctured;
    - parity bits ``m1..m4`` for the 4x4 core block are computed using the
      ``Htype`` switch from ``encode.cpp``.  For ``Htype=0`` (the default
      branch, used by BG2 sets {1,2,3,5,6,7} and BG1 set 7's "case 2", etc.)
      the C++ rule m1 = right-cyclic-shift(sum(d0), 1) is preserved verbatim.
      It is algebraically equivalent to ``m1 = P^{-1} sum(d0)`` because the
      sum of the four core-row equations collapses to ``P^1 * m1 = sum(d0)``
      for those base graphs.
* Rate matching: starting at ``k0`` (rv-dependent) we cycle through the
  un-punctured codeword (length ``Ncb = N - 2*Zc``) skipping filler bits
  (-1) until ``E`` output bits are produced.  Bit interleaving uses ``Qm``
  rows (identity for Qm=1 / BPSK).  Inverse operation deinterleaves and then
  scatters LLRs back to the original codeword positions, leaving the
  unfilled positions at zero and filler positions at +inf (known to be 0).

Decoder is Offset Min-Sum exactly as in
``decodeLogDomainMinSum_converge`` in ``decode.cpp``: V2C/C2V are exposed as
the standalone helpers ``v2c_message`` and ``c2v_message`` per the user
request.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

_BG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bg_tables")

# ---------------------------------------------------------------------------
# 3GPP TS 38.212 §5.3.2 lifting size table (Zlist) and Zsets index.  Direct
# copy of the C++ ``Zlist`` and ``Zsets`` arrays.
# ---------------------------------------------------------------------------

ZLIST: Tuple[int, ...] = (
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    18, 20, 22, 24, 26, 28, 30, 32,
    36, 40, 44, 48, 52, 56, 60, 64,
    72, 80, 88, 96, 104, 112, 120, 128,
    144, 160, 176, 192, 208, 224, 240, 256,
    288, 320, 352, 384,
)

ZSETS: Tuple[Tuple[int, ...], ...] = (
    (2, 4, 8, 16, 32, 64, 128, 256),
    (3, 6, 12, 24, 48, 96, 192, 384),
    (5, 10, 20, 40, 80, 160, 320, 0),
    (7, 14, 28, 56, 112, 224, 0, 0),
    (9, 18, 36, 72, 144, 288, 0, 0),
    (11, 22, 44, 88, 176, 352, 0, 0),
    (13, 26, 52, 104, 208, 0, 0, 0),
    (15, 30, 60, 120, 240, 0, 0, 0),
)

# Htype tables from encode.cpp (indexed [bgn-1][setIdx-1]).  Values pick the
# branch of the parity-computation switch in ``encode``: 1, 2, 3 hit the
# corresponding case; any other value (e.g. 4 for BG2 sets 1/2/3/5/6/7) hits
# the ``default`` branch.
HTYPE: Tuple[Tuple[int, ...], ...] = (
    (3, 3, 3, 3, 3, 3, 2, 3),  # BG1
    (4, 4, 4, 1, 4, 4, 4, 1),  # BG2
)


def select_set_index(zc: int) -> int:
    """Return ``setIdx`` (1..8) for a given lifting size, matching C++."""
    for i, row in enumerate(ZSETS):
        for v in row:
            if v == zc:
                return i + 1
    raise ValueError(f"Lifting size {zc} not in 3GPP table")


def load_base_graph(bgn: int, set_idx: int) -> np.ndarray:
    """Load BG matrix from disk (BG1: 46x68, BG2: 42x52)."""
    if bgn == 1:
        rows, cols = 46, 68
    elif bgn == 2:
        rows, cols = 42, 52
    else:
        raise ValueError("bgn must be 1 or 2")
    path = os.path.join(_BG_DIR, f"BG{bgn}S{set_idx}.txt")
    bg = np.loadtxt(path, dtype=np.int64)
    assert bg.shape == (rows, cols), f"BG file {path} shape {bg.shape} != {(rows, cols)}"
    return bg


# ---------------------------------------------------------------------------
# Parity check matrix construction — mirrors ``parityCheckMatrix`` / ``getH``
# in LDPCfunctions.cpp.  We keep the H matrix sparse via row-wise lists of
# variable-node indices (same data structure used by the C++ decoder).
# ---------------------------------------------------------------------------


@dataclass
class LiftedParity:
    bgn: int
    set_idx: int
    zc: int
    rows: int  # M = Mb * Zc
    cols: int  # N = Nb * Zc
    bg: np.ndarray  # Mb x Nb, raw shifts (-1 or non-negative)
    bg_mod: np.ndarray  # Mb x Nb, shifts reduced mod Zc (-1 preserved)
    cn_to_vn: List[np.ndarray]  # rows[r] = sorted vn indices touched by check r


def build_parity(bgn: int, set_idx: int, zc: int) -> LiftedParity:
    bg = load_base_graph(bgn, set_idx)
    bg_mod = bg.copy()
    nz = bg_mod >= 0
    bg_mod[nz] = bg_mod[nz] % zc

    mb, nb = bg.shape
    M = mb * zc
    N = nb * zc

    cn_to_vn: List[np.ndarray] = [None] * M  # type: ignore[assignment]
    for i_b in range(mb):
        for r_off in range(zc):
            row = i_b * zc + r_off
            cols: List[int] = []
            for j_b in range(nb):
                s = int(bg_mod[i_b, j_b])
                if s >= 0:
                    # Lifted block M[r][(r+s) % zc] = 1; for row r=r_off the
                    # only non-zero column is (r_off + s) % zc inside block j_b.
                    cols.append(j_b * zc + (r_off + s) % zc)
            cn_to_vn[row] = np.asarray(cols, dtype=np.int64)
    return LiftedParity(
        bgn=bgn, set_idx=set_idx, zc=zc,
        rows=M, cols=N,
        bg=bg, bg_mod=bg_mod,
        cn_to_vn=cn_to_vn,
    )


def parity_check_syndrome(par: LiftedParity, codeword: np.ndarray) -> np.ndarray:
    """Compute H @ codeword over GF(2).  ``codeword`` shape (N,)."""
    syn = np.zeros(par.rows, dtype=np.int8)
    cw = codeword.astype(np.int8)
    for r, vn in enumerate(par.cn_to_vn):
        if vn.size:
            syn[r] = int(cw[vn].sum() & 1)
    return syn


# ---------------------------------------------------------------------------
# 3GPP code-block parameter derivation — mirrors the parameter-selection
# block at the top of ``main.cpp``.
# ---------------------------------------------------------------------------


@dataclass
class CodeBlockParams:
    A: int           # info bits per TB (without TB-CRC)
    code_rate: float
    L: int           # TB-CRC length (16 or 24)
    B: int           # A + L
    bgn: int
    Kcb: int
    C: int           # number of code-block segments
    B_cb: int        # B + C*L (=B for C==1)
    K_cb_bit: int    # info bits per CB including CB-CRC, no fillers
    Kb: int
    zc: int
    K: int           # 22*Zc (BG1) or 10*Zc (BG2)
    num_filler_per_cb: int
    Nb: int
    Mb: int
    N: int           # full parity-check codeword length (Nb * Zc)
    M: int           # parity rows (Mb * Zc)
    N_punctured: int # transmitted codeword length (= N - 2*Zc)
    set_idx: int
    outlen: int      # rate-matched output length per TB


def derive_params(A: int, code_rate: float) -> CodeBlockParams:
    L = 24 if A > 3824 else 16
    B = A + L
    if A < 293 or (A < 3825 and code_rate <= 0.67) or code_rate <= 0.25:
        bgn = 2
    else:
        bgn = 1
    Kcb = 8448 if bgn == 1 else 3840
    if B < Kcb:
        C = 1
        B_cb = B
    else:
        C = math.ceil(B / (Kcb - L))
        B_cb = B + C * L
    K_cb_bit = math.ceil(B_cb / C)
    if bgn == 1:
        Kb = 22
    else:
        if B > 640:
            Kb = 10
        elif B > 560:
            Kb = 9
        elif B > 192:
            Kb = 8
        else:
            Kb = 6
    zc = 0
    for v in ZLIST:
        if Kb * v >= K_cb_bit:
            zc = v
            break
    if zc == 0:
        # mirror C++: walk through and pick last assigned
        for v in ZLIST:
            if Kb * v >= K_cb_bit:
                zc = v
                break
    K = 22 * zc if bgn == 1 else 10 * zc
    num_filler = K - K_cb_bit
    if bgn == 1:
        Nb, Mb = 68, 46
    else:
        Nb, Mb = 52, 42
    N = Nb * zc
    M = Mb * zc
    N_punctured = N - 2 * zc
    set_idx = select_set_index(zc)
    outlen = math.ceil(A / code_rate)
    return CodeBlockParams(
        A=A, code_rate=code_rate, L=L, B=B, bgn=bgn, Kcb=Kcb,
        C=C, B_cb=B_cb, K_cb_bit=K_cb_bit, Kb=Kb, zc=zc, K=K,
        num_filler_per_cb=num_filler, Nb=Nb, Mb=Mb,
        N=N, M=M, N_punctured=N_punctured, set_idx=set_idx, outlen=outlen,
    )


# ---------------------------------------------------------------------------
# Encoder.  Mirrors ``encode``/``nrLdpcEncode`` in encode.cpp exactly.
# Operates on a single CB.  ``infobits`` is shape (K,), int values 0/1 with
# ``-1`` marking filler positions.  Returns a length-N_punctured numpy array
# of int8 with filler positions marked as -1 (consumed downstream by the
# rate matcher).
# ---------------------------------------------------------------------------


def _circ_shift_acc(infovec_col: np.ndarray, shift: int) -> np.ndarray:
    """Reproduce the C++ inner loop:

        tmp[k] = infovec_col[(shift + k) % Zc]    for k in 0..Zc-1

    which is the action of the right-cyclic permutation block with shift
    parameter ``shift`` on the message column.
    """
    z = infovec_col.shape[0]
    return np.concatenate((infovec_col[shift:z], infovec_col[0:shift]))


def _compute_core_parity(d0: np.ndarray, htype: int, zc: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute m1..m4 (each length Zc) per the C++ ``encode`` switch.

    ``d0`` is shape (Zc, 4) and equals ``H_A * msg`` for the four core rows.
    Operations are done in plain Python ints (we apply mod 2 later when
    writing to the codeword, identical to the C++ which stores raw integer
    accumulations and only applies ``% 2`` at the final ``out[][]`` write).
    """
    z = d0.shape[0]
    if htype == 1:
        # BG2 sets {4, 8}.  See encode.cpp case 1.
        m1 = d0.sum(axis=1)
        # m1tmp = m1 left-shifted by 1
        m1tmp = np.concatenate((m1[1:], m1[:1]))
        m2 = d0[:, 0] + m1tmp
        m3 = d0[:, 1] + m2
        m4 = d0[:, 2] + m1 + m3
        return m1, m2, m3, m4
    if htype == 2:
        # BG1 set 7 (`shift = 105 % Zc`).  See encode.cpp case 2.
        m1tmp = d0.sum(axis=1)
        shift = 105 % z
        if shift > 0:
            m1 = np.concatenate((m1tmp[z - shift:z], m1tmp[0:z - shift]))
        else:
            m1 = m1tmp.copy()
        m2 = d0[:, 0] + m1
        m4 = d0[:, 3] + m1
        m3 = d0[:, 2] + m4
        return m1, m2, m3, m4
    if htype == 3:
        # BG1 sets {1..6, 8}.  See encode.cpp case 3.
        m1 = d0.sum(axis=1)
        m1tmp = np.concatenate((m1[1:], m1[:1]))
        m2 = d0[:, 0] + m1tmp
        m3 = d0[:, 1] + m1 + m2
        m4 = d0[:, 2] + m3
        return m1, m2, m3, m4
    # default branch (Htype 4 / 0): BG2 sets {1,2,3,5,6,7}.
    m1_sum = d0.sum(axis=1)
    # right cyclic shift by 1: new[0] = old[Z-1], new[k] = old[k-1]
    m1 = np.concatenate((m1_sum[-1:], m1_sum[:-1]))
    m2 = d0[:, 0] + m1
    m3 = d0[:, 1] + m2
    m4 = d0[:, 3] + m1
    return m1, m2, m3, m4


def encode_codeblock(infobits: np.ndarray, par: LiftedParity, htype: int) -> np.ndarray:
    """Encode a single CB.  Filler bits in ``infobits`` are marked as -1.

    Returns int8 array of length ``N - 2*Zc`` (after puncturing first 2*Zc),
    with filler positions restored to -1 (per nrLdpcEncode behaviour).
    """
    zc = par.zc
    bg = par.bg
    Mb, Nb = bg.shape
    if par.bgn == 1:
        Kb = 22
        ncwnodes = 66
    else:
        Kb = 10
        ncwnodes = 50
    K = Kb * zc
    if infobits.shape[0] != K:
        raise ValueError(f"infobits length {infobits.shape[0]} != K={K}")

    # Track filler-bit positions and zero them for the encoder.
    filler_mask = (infobits == -1)
    msg = infobits.copy()
    msg[filler_mask] = 0
    msg = msg.astype(np.int64)

    # infoVec: Zc x Kb, infoVec[k][j] = msg[Zc*j + k]
    infovec = msg.reshape(Kb, zc).T  # column-major reshape into (Zc, Kb)

    # bg_mod for the message section P1 = bg_mod[:, :Kb]; for parity columns
    # we still use bg.  Both have shifts already; only the message part is
    # accumulated into ``d`` per the C++ inner loop.
    bg_mod = par.bg_mod

    # d: (Zc, Mb).  d[:, i] = sum_j (P^{shift_ij} * infovec[:, j])
    d = np.zeros((zc, Mb), dtype=np.int64)
    P1 = bg_mod[:, :Kb]
    for i in range(Mb):
        for j in range(Kb):
            s = int(P1[i, j])
            if s != -1:
                d[:, i] += _circ_shift_acc(infovec[:, j], s)

    # Core parity m1..m4 from rows 0..3 of d.
    d0 = d[:, :4]
    m1, m2, m3, m4 = _compute_core_parity(d0, htype, zc)

    # Extension parity p (Zc x (Mb-4)).  P3 = bg_mod[4:, Kb:Kb+4] — shifts
    # acting on [m1, m2, m3, m4].  The C++ accumulates: p[:, i] += P3 * tmpm
    # then adds d[:, 4+i].
    P3 = bg_mod[4:, Kb:Kb + 4]
    tmpm = np.stack((m1, m2, m3, m4), axis=1)  # (Zc, 4)
    p = np.zeros((zc, Mb - 4), dtype=np.int64)
    for i in range(Mb - 4):
        for j in range(4):
            s = int(P3[i, j])
            if s != -1:
                p[:, i] += _circ_shift_acc(tmpm[:, j], s)
        p[:, i] += d[:, 4 + i]

    # Assemble the full lifted codeword (length Nb * Zc) column-by-column.
    out_full = np.empty(Nb * zc, dtype=np.int64)
    out_full[:K] = msg                                # message
    out_full[K + 0 * zc:K + 1 * zc] = m1
    out_full[K + 1 * zc:K + 2 * zc] = m2
    out_full[K + 2 * zc:K + 3 * zc] = m3
    out_full[K + 3 * zc:K + 4 * zc] = m4
    # extension parity columns Kb+4 .. Nb-1 (= ncwnodes+2 .. )
    for i in range(Mb - 4):
        out_full[K + (4 + i) * zc:K + (5 + i) * zc] = p[:, i]
    out_full = (out_full % 2).astype(np.int8)

    # Re-mark filler positions as -1.
    out_full[:K][filler_mask] = -1

    # Puncture the first 2*Zc bits (per nrLdpcEncode).
    return out_full[2 * zc:]


# ---------------------------------------------------------------------------
# Rate matching / recovery (BPSK so Qm=1, identity bit interleaver, single CB).
# Mirrors ``modified_RateMatchLDPC`` / ``modified_RateRecoverLDPC`` in
# ratematch.cpp for the ``rv=0`` / ``Qm=1`` / single-CB case.  Multi-CB
# concatenation is implemented identically but exercised only when C>1.
# ---------------------------------------------------------------------------


def compute_k0(bgn: int, zc: int, ncb: int, rv: int) -> int:
    if bgn == 1:
        if rv == 0:
            return 0
        if rv == 1:
            return (17 * ncb // (66 * zc)) * zc
        if rv == 2:
            return (33 * ncb // (66 * zc)) * zc
        if rv == 3:
            return (56 * ncb // (66 * zc)) * zc
    else:
        if rv == 0:
            return 0
        if rv == 1:
            return (13 * ncb // (50 * zc)) * zc
        if rv == 2:
            return (25 * ncb // (50 * zc)) * zc
        if rv == 3:
            return (43 * ncb // (50 * zc)) * zc
    raise ValueError("invalid rv")


def compute_E_array(C: int, outlen: int, qm: int, nlayers: int) -> List[int]:
    base = nlayers * qm
    cbs_floor = base * (outlen // (base * C))
    cbs_ceil = base * math.ceil(outlen / (base * C))
    e_arr = []
    threshold = C - (outlen // base) % C - 1
    for r in range(C):
        e_arr.append(cbs_floor if r <= threshold else cbs_ceil)
    return e_arr


def cbs_rate_match(d: np.ndarray, E: int, k0: int, ncb: int, qm: int) -> np.ndarray:
    """Single CB rate matching.  ``d`` is the un-punctured codeword (with
    -1 filler markers).  Returns an int8 array of length E.
    """
    e = np.empty(E, dtype=np.int8)
    k = 0
    j = 0
    while k < E:
        idx = (k0 + j) % ncb
        v = int(d[idx])
        if v != -1:  # skip filler bits
            e[k] = v
            k += 1
        j += 1
    if qm == 1:
        return e
    e1 = np.empty(E, dtype=np.int8)
    sub = E // qm
    for i in range(qm):
        for jj in range(sub):
            e1[jj * qm + i] = e[i * sub + jj]
    return e1


def cbs_rate_recover(d_llr: np.ndarray, N: int, E: int, K: int, Kd: int,
                     k0: int, ncb: int, qm: int) -> np.ndarray:
    """Inverse of cbs_rate_match for soft LLRs.

    ``Kd`` is the start of the filler-bit region inside the lifted codeword
    AFTER puncturing (= K_cb_bit - 2*Zc), ``K`` is the end of the same
    region (= K - 2*Zc).  Filler positions are filled with +infinity (known
    bit-0 → very confident +LLR), all other unfilled positions remain 0.
    """
    e = np.zeros(E, dtype=np.float64)
    if qm == 1:
        e[:] = d_llr[:E]
    else:
        sub = E // qm
        for i in range(qm):
            for jj in range(sub):
                e[i * sub + jj] = d_llr[jj * qm + i]
    indices = np.empty(E, dtype=np.int64)
    k = 0
    j = 0
    while k < E:
        idx = (k0 + j) % ncb
        if not (Kd <= idx < K):
            indices[k] = idx
            k += 1
        j += 1
    out = np.zeros(N, dtype=np.float64)
    out[Kd:K] = 1e9  # filler bits known to be 0 -> huge positive LLR
    # Use np.add.at for repeat-index accumulation (mirrors C++ "+= e[n]").
    np.add.at(out, indices, e)
    return out


# ---------------------------------------------------------------------------
# BPSK + AWGN channel.
# ---------------------------------------------------------------------------


def bpsk_modulate(bits: np.ndarray) -> np.ndarray:
    """0 -> +1, 1 -> -1."""
    return 1.0 - 2.0 * bits.astype(np.float64)


def awgn(symbols: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return symbols + sigma * rng.standard_normal(symbols.shape)


def bpsk_llr(rx: np.ndarray, sigma2: float) -> np.ndarray:
    """LLR convention used by the C++ decoder: positive LLR → bit 0."""
    return 2.0 * rx / sigma2


# ---------------------------------------------------------------------------
# Offset Min-Sum decoder.  V2C / C2V exposed as standalone functions per the
# user request.  Decoder loop calls them per-edge.
# ---------------------------------------------------------------------------


def v2c_message(llr_vn: float, llr_cn_to_vn_old: float) -> float:
    """Variable-to-Check message along edge (v -> c).

    Mirrors C++ ``x1 = LLR_VN[CNtoVN[k][j]] - LLR_CNtoVN[k][j]``.  In a flooded
    schedule this is the extrinsic VN belief: total VN belief minus the
    incoming message from the same check node.
    """
    return llr_vn - llr_cn_to_vn_old


def c2v_message(incoming_v2c: np.ndarray, offset: float = 0.25) -> np.ndarray:
    """Check-to-Variable update for one check node.

    Given the V2C messages along all edges incident to a check, returns the
    new C2V message along each edge.  Mirrors the C++ inner double loop:

        sig  = prod_{j!=i} sign(x1_j)
        |y|  = max( min_{j!=i} |x1_j| - offset, 0 )
        y_i  = sig * |y|
    """
    n = incoming_v2c.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        sig = 1
        tanhtemp = float("inf")
        for j in range(n):
            if j == i:
                continue
            x1 = incoming_v2c[j]
            if x1 > 0:
                pass
            elif x1 < 0:
                sig = -sig
            else:
                sig = 0
            ax = abs(x1)
            if ax < tanhtemp:
                tanhtemp = ax
        mag = tanhtemp - offset
        if mag < 0.0:
            mag = 0.0
        out[i] = sig * mag
    return out


def decode_oms(rx_llr: np.ndarray, par: LiftedParity, max_iter: int,
               offset: float = 0.25) -> Tuple[np.ndarray, int]:
    """Offset Min-Sum BP decoder mirroring decodeLogDomainMinSum_converge.

    Parameters
    ----------
    rx_llr : channel LLR per VN (length N)
    par    : lifted parity-check structure
    max_iter : maximum number of BP iterations
    offset : OMS offset (C++ uses 0.25)

    Returns
    -------
    llr_out : per-VN posterior LLRs (length N)
    iters   : number of iterations performed (early-stop on syndrome=0)
    """
    N = par.cols
    M = par.rows
    cn_to_vn = par.cn_to_vn

    # Per-edge state. Use lists of np.float64 vectors for fast vectorised
    # updates inside each check.
    llr_c2v: List[np.ndarray] = [np.zeros(vn.size, dtype=np.float64) for vn in cn_to_vn]
    llr_vn = rx_llr.astype(np.float64).copy()

    iters_run = 0
    for it in range(max_iter):
        iters_run = it + 1
        # ---- Check node update (C2V) ----
        for k in range(M):
            vn = cn_to_vn[k]
            if vn.size == 0:
                continue
            old = llr_c2v[k]
            v2c = llr_vn[vn] - old  # extrinsic VN beliefs along edges into c=k
            llr_c2v[k] = c2v_message(v2c, offset)

        # ---- Variable node update (VN) ----
        y = np.zeros(N, dtype=np.float64)
        for k in range(M):
            vn = cn_to_vn[k]
            if vn.size:
                np.add.at(y, vn, llr_c2v[k])
        llr_vn = y + rx_llr

        # ---- Syndrome check ----
        bits = (llr_vn < 0.0).astype(np.int8)
        ok = True
        for k in range(M):
            vn = cn_to_vn[k]
            if vn.size and (int(bits[vn].sum()) & 1):
                ok = False
                break
        if ok:
            break

    return llr_vn, iters_run


# ---------------------------------------------------------------------------
# Vectorised OMS decoder (functionally equivalent to ``decode_oms`` but much
# faster for Monte-Carlo sweeps).  Uses a dense (M, dc_max) edge buffer with
# per-row valid-length masks, computes the running min1/min2/sign-product per
# check in pure numpy, then scatters the C2V messages back.  Verified to
# produce bit-identical outputs to ``decode_oms`` on random noise.
# ---------------------------------------------------------------------------


@dataclass
class OmsContext:
    par: LiftedParity
    edge_vn: np.ndarray          # (M, dc_max) int64; vn index per edge slot
    edge_mask: np.ndarray        # (M, dc_max) bool;  True where slot is real
    row_lengths: np.ndarray      # (M,) int64
    edge_flat_vn: np.ndarray     # (E_total,) int64; vn index per real edge
    edge_flat_row: np.ndarray    # (E_total,) int64; row index per real edge
    dc_max: int


def build_oms_context(par: LiftedParity) -> OmsContext:
    M = par.rows
    row_lengths = np.array([vn.size for vn in par.cn_to_vn], dtype=np.int64)
    dc_max = int(row_lengths.max())
    edge_vn = np.zeros((M, dc_max), dtype=np.int64)
    edge_mask = np.zeros((M, dc_max), dtype=bool)
    for k, vn in enumerate(par.cn_to_vn):
        edge_vn[k, :vn.size] = vn
        edge_mask[k, :vn.size] = True
    flat_row = np.repeat(np.arange(M, dtype=np.int64), row_lengths)
    flat_vn = np.concatenate(par.cn_to_vn).astype(np.int64) if M else np.zeros(0, dtype=np.int64)
    return OmsContext(par=par, edge_vn=edge_vn, edge_mask=edge_mask,
                      row_lengths=row_lengths, edge_flat_vn=flat_vn,
                      edge_flat_row=flat_row, dc_max=dc_max)


def decode_oms_fast(rx_llr: np.ndarray, ctx: OmsContext, max_iter: int,
                    offset: float = 0.25) -> Tuple[np.ndarray, int]:
    par = ctx.par
    M = par.rows
    N = par.cols
    edge_vn = ctx.edge_vn
    edge_mask = ctx.edge_mask
    flat_row = ctx.edge_flat_row
    flat_vn = ctx.edge_flat_vn

    # Edge state stored densely; padding slots are masked out.
    llr_c2v = np.zeros((M, ctx.dc_max), dtype=np.float64)
    llr_vn = rx_llr.astype(np.float64).copy()

    INF = 1e30
    iters_run = 0
    for it in range(max_iter):
        iters_run = it + 1

        # ---- Build V2C along every edge in one shot ----
        v2c = llr_vn[edge_vn] - llr_c2v
        v2c = np.where(edge_mask, v2c, 0.0)

        # Per-row min1, min2, argmin1, sign product.
        absv = np.where(edge_mask, np.abs(v2c), INF)
        # argmin along axis 1
        argmin1 = np.argmin(absv, axis=1)              # (M,)
        rows = np.arange(M)
        min1 = absv[rows, argmin1]
        # mask out the min position to find the second minimum
        absv2 = absv.copy()
        absv2[rows, argmin1] = INF
        min2 = np.min(absv2, axis=1)

        signs = np.where(v2c >= 0.0, 1, -1)
        signs = np.where(edge_mask, signs, 1)
        # product of all signs in the row
        sign_prod_row = np.prod(signs, axis=1)         # (M,)
        # extrinsic sign for edge i = total / sign(edge i) = total * sign(edge i)
        sign_ext = sign_prod_row[:, None] * signs       # (M, dc_max)

        # magnitude: edges at argmin use min2, others use min1
        mag = np.broadcast_to(min1[:, None], (M, ctx.dc_max)).copy()
        mag[rows, argmin1] = min2[rows]
        mag = np.maximum(mag - offset, 0.0)

        new_c2v = sign_ext * mag
        llr_c2v = np.where(edge_mask, new_c2v, 0.0)

        # ---- VN posterior ----
        y = np.zeros(N, dtype=np.float64)
        # accumulate llr_c2v values at flat_vn positions
        np.add.at(y, flat_vn, llr_c2v[edge_mask])
        llr_vn = y + rx_llr

        # ---- Syndrome ----
        hard = (llr_vn < 0.0).astype(np.int8)
        # parity per row = XOR of hard[edge_vn] over real edges
        per_edge_bits = hard[edge_vn] & edge_mask.astype(np.int8)
        row_parity = np.bitwise_xor.reduce(per_edge_bits, axis=1)
        if not row_parity.any():
            break

    return llr_vn, iters_run


__all__ = [
    "ZLIST", "ZSETS", "HTYPE",
    "LiftedParity", "CodeBlockParams",
    "select_set_index", "load_base_graph", "build_parity",
    "parity_check_syndrome", "derive_params",
    "encode_codeblock",
    "compute_k0", "compute_E_array",
    "cbs_rate_match", "cbs_rate_recover",
    "bpsk_modulate", "awgn", "bpsk_llr",
    "v2c_message", "c2v_message",
    "decode_oms", "build_oms_context", "decode_oms_fast",
    "default_v2c_oms", "default_c2v_oms", "decode_bp",
]


# ---------------------------------------------------------------------------
# Generic BP decoder with pluggable per-edge V2C / C2V callables.
#
# The contract for the callables (matching the Push-GP stack-seeding rule
# defined in pushgp/validators.py):
#
#     v2c_fn(L_v, incoming_c2v, deg, iter_idx, ctx) -> float
#         L_v          : channel LLR for this variable node
#         incoming_c2v : np.ndarray of shape (deg-1,) of C2V messages from
#                        the OTHER (deg-1) checks (the edge being computed
#                        is excluded)
#         deg          : variable-node degree
#         iter_idx     : 0-based BP iteration index
#         ctx          : dict with at least 'max_iter', 'code_rate', 'offset'
#         returns      : scalar V2C message along the excluded edge.
#
#     c2v_fn(incoming_v2c, deg, iter_idx, ctx) -> float
#         incoming_v2c : np.ndarray of shape (deg-1,) of V2C messages from
#                        the OTHER (deg-1) variable nodes incident to this
#                        check (excluding the edge being computed)
#         (other args identical)
#         returns      : scalar C2V message along the excluded edge.
#
# Default callables reproduce the Offset-Min-Sum decoder above so that
#     decode_bp(..., default_v2c_oms, default_c2v_oms, ...)
# is bit-for-bit equal to decode_oms_fast on the same input.  This is
# verified by the PR5 regression test.
# ---------------------------------------------------------------------------


def _build_vn_to_cn(par: "LiftedParity") -> List[List[Tuple[int, int]]]:
    """For each VN, return list of (check_index, edge_position_in_check)."""
    N = par.cols
    vn_to_cn: List[List[Tuple[int, int]]] = [[] for _ in range(N)]
    for c, vns in enumerate(par.cn_to_vn):
        for pos, v in enumerate(vns):
            vn_to_cn[int(v)].append((c, int(pos)))
    return vn_to_cn


def default_v2c_oms(L_v: float, incoming_c2v: np.ndarray, deg: int,
                    iter_idx: int, ctx: dict) -> float:
    """OMS V2C: extrinsic VN belief = L_v + sum of all c2v EXCEPT this edge.

    Since `incoming_c2v` already excludes the current edge, this reduces to
    `L_v + sum(incoming_c2v)`.
    """
    return float(L_v) + float(np.sum(incoming_c2v))


def default_c2v_oms(incoming_v2c: np.ndarray, deg: int, iter_idx: int,
                    ctx: dict) -> float:
    """OMS C2V: sign_product(other edges) * max(min|other edges| - β, 0)."""
    beta = float(ctx.get("offset", 0.25))
    n = incoming_v2c.shape[0]
    if n == 0:
        return 0.0
    sig = 1
    min_abs = float("inf")
    for j in range(n):
        x = float(incoming_v2c[j])
        if x > 0.0:
            pass
        elif x < 0.0:
            sig = -sig
        else:
            sig = 0
        ax = abs(x)
        if ax < min_abs:
            min_abs = ax
    mag = min_abs - beta
    if mag < 0.0:
        mag = 0.0
    return float(sig * mag)


def decode_bp(rx_llr: np.ndarray, par: "LiftedParity", *,
              v2c_fn=default_v2c_oms, c2v_fn=default_c2v_oms,
              max_iter: int = 25, offset: float = 0.25,
              code_rate: float = 0.5,
              return_iters: bool = False):
    """Generic flooded-schedule BP decoder with pluggable V2C/C2V.

    Functionally equivalent to ``decode_oms`` when the default callables
    are used (verified by regression test).  Pure Python loops, intended
    for evolution (correctness >> speed).
    """
    N = par.cols
    M = par.rows
    cn_to_vn = par.cn_to_vn
    vn_to_cn = _build_vn_to_cn(par)

    llr_c2v: List[np.ndarray] = [np.zeros(vn.size, dtype=np.float64) for vn in cn_to_vn]
    llr_v2c: List[np.ndarray] = [np.zeros(vn.size, dtype=np.float64) for vn in cn_to_vn]

    ctx = {
        "max_iter": int(max_iter),
        "offset": float(offset),
        "code_rate": float(code_rate),
    }

    rx_llr = rx_llr.astype(np.float64, copy=False)
    iters_run = 0
    llr_post = rx_llr.copy()

    for it in range(max_iter):
        iters_run = it + 1

        # ---- V2C update: per VN, per incident edge ----
        for v in range(N):
            edges_in = vn_to_cn[v]
            dv = len(edges_in)
            if dv == 0:
                continue
            c2v_vec = np.empty(dv, dtype=np.float64)
            for ei, (c, p) in enumerate(edges_in):
                c2v_vec[ei] = llr_c2v[c][p]
            L_v = float(rx_llr[v])
            for ei, (c, p) in enumerate(edges_in):
                if dv == 1:
                    incoming = c2v_vec[:0]
                else:
                    incoming = np.concatenate((c2v_vec[:ei], c2v_vec[ei + 1:]))
                try:
                    msg = v2c_fn(L_v, incoming, dv, it, ctx)
                except Exception:
                    msg = 0.0
                if msg is None or not np.isfinite(msg):
                    msg = 0.0
                llr_v2c[c][p] = float(msg)

        # ---- C2V update: per check, per incident edge ----
        for c in range(M):
            v2c_vec = llr_v2c[c]
            dc = v2c_vec.size
            if dc == 0:
                continue
            new_c2v = np.zeros(dc, dtype=np.float64)
            for ei in range(dc):
                if dc == 1:
                    incoming = v2c_vec[:0]
                else:
                    incoming = np.concatenate((v2c_vec[:ei], v2c_vec[ei + 1:]))
                try:
                    msg = c2v_fn(incoming, dc, it, ctx)
                except Exception:
                    msg = 0.0
                if msg is None or not np.isfinite(msg):
                    msg = 0.0
                new_c2v[ei] = float(msg)
            llr_c2v[c] = new_c2v

        # ---- VN posterior ----
        llr_post = rx_llr.copy()
        for v in range(N):
            s = 0.0
            for (c, p) in vn_to_cn[v]:
                s += llr_c2v[c][p]
            llr_post[v] += s

        # ---- Syndrome check (early stop) ----
        bits = (llr_post < 0.0).astype(np.int8)
        ok = True
        for c in range(M):
            vn = cn_to_vn[c]
            if vn.size and (int(bits[vn].sum()) & 1):
                ok = False
                break
        if ok:
            break

    if return_iters:
        return llr_post, iters_run
    return llr_post
