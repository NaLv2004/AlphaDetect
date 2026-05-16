"""Verify: does the OMS baseline's V2C lack the extrinsic subtraction?

Hypothesis: pushgp_ldpc.adapter._oms_v2c_program sends
   v->c = L_v + sum(all incoming c->v)        (== belief)
instead of the standard
   v->c = L_v + sum(incoming) - this_c->v     (== extrinsic).

Test:
  A. evaluate the OMS seed genome through fitness pipeline (cpp BP)
  B. run a numpy OMS without extrinsic subtraction (matches adapter)
  C. run a numpy OMS with extrinsic subtraction (textbook)
Expect:  A ~= B  >>  C.
"""
from __future__ import annotations
import os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np
from ldpc_5g import build_parity
from pushgp.dce import _try_import_pushgp_cpp_dce
from pushgp_ldpc.eval import FitnessConfig, _channel_inputs
from pushgp_ldpc.adapter import oms_seed_genome
from pushgp.serialize import program_to_dict

cdce = _try_import_pushgp_cpp_dce()
par = build_parity(bgn=2, set_idx=1, zc=2)
parH = cdce.build_parity_handle(par)
N, M = par.cols, par.rows
cn_to_vn = par.cn_to_vn
vn_to_cn: list[list[tuple[int,int]]] = [[] for _ in range(N)]
for c, vns in enumerate(cn_to_vn):
    for pos, v in enumerate(vns):
        vn_to_cn[int(v)].append((c, pos))

# ---- OMS seed genome serialized to dict programs ----
g = oms_seed_genome()
PROG_V2C = [program_to_dict([instr])[0] if hasattr(instr, "name") else instr
            for instr in g.prog_v2c]
PROG_C2V = [program_to_dict([instr])[0] if hasattr(instr, "name") else instr
            for instr in g.prog_c2v]
evo = (10.0 ** np.clip(g.log_constants, -8, 8)).astype(np.float64)

def make_channel(snr_db, n_frames, seed, code_rate=0.5):
    cfg = FitnessConfig(par=par, snr_list=(snr_db,), n_frames_per_snr=n_frames,
                        max_iter=20, code_rate=code_rate, seed_base=seed)
    return _channel_inputs(cfg, snr_db)

def uncoded_ber(snr_db, n_frames, seed, code_rate=0.5):
    # σ² uses same convention as fitness pipeline
    sigma2 = 1.0 / (2.0 * code_rate * (10.0 ** (snr_db / 10.0)))
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=n_frames * N).astype(np.uint8)
    bpsk = 1.0 - 2.0 * bits.astype(np.float64)  # 0->+1, 1->-1
    noise = rng.standard_normal(bits.size) * np.sqrt(sigma2)
    rx = bpsk + noise
    hard = (rx < 0).astype(np.uint8)
    return float(np.mean(hard != bits))

def decode_oms_genome(llr, max_iter=20):
    post, _ = cdce.decode_bp(llr.astype(np.float64), parH,
                              PROG_V2C, PROG_C2V, evo, int(max_iter), 0.25, 0.5)
    return (np.asarray(post) < 0).astype(np.uint8)

def decode_oms_no_extrinsic(llr, iters=20, offset=0.25):
    """V2C sends FULL belief (no subtract self c->v) -> matches adapter bug."""
    L = llr.astype(np.float64).copy()
    c2v = [np.zeros(len(vns)) for vns in cn_to_vn]
    for it in range(iters):
        # belief
        post = L + np.array([sum(c2v[c][p] for (c, p) in vn_to_cn[v]) for v in range(N)])
        # v2c (no extrinsic)
        v2c = [np.zeros(len(vns)) for vns in cn_to_vn]
        for v in range(N):
            for (c, p) in vn_to_cn[v]:
                v2c[c][p] = post[v]            # <-- bug: no minus c2v[c][p]
        # c2v offset min-sum
        for c in range(M):
            msgs = v2c[c]; n = len(msgs)
            if n == 0: continue
            signs = np.sign(msgs); signs[signs == 0] = 1.0
            absm = np.abs(msgs); sp_all = np.prod(signs)
            for ei in range(n):
                sp = sp_all * signs[ei]
                others = np.delete(absm, ei)
                mn = others.min() if n > 1 else 0.0
                c2v[c][ei] = sp * max(mn - offset, 0.0)
        # early stop
        post = L + np.array([sum(c2v[c][p] for (c, p) in vn_to_cn[v]) for v in range(N)])
        bits = (post < 0).astype(np.uint8)
        ok = all(((sum(int(bits[v]) for v in cn_to_vn[c]) & 1) == 0) for c in range(M))
        if ok: break
    return (post < 0).astype(np.uint8)

def decode_oms_correct(llr, iters=20, offset=0.25):
    """Textbook OMS with extrinsic subtraction."""
    L = llr.astype(np.float64).copy()
    c2v = [np.zeros(len(vns)) for vns in cn_to_vn]
    for it in range(iters):
        post = L + np.array([sum(c2v[c][p] for (c, p) in vn_to_cn[v]) for v in range(N)])
        v2c = [np.zeros(len(vns)) for vns in cn_to_vn]
        for v in range(N):
            for (c, p) in vn_to_cn[v]:
                v2c[c][p] = post[v] - c2v[c][p]       # correct extrinsic
        for c in range(M):
            msgs = v2c[c]; n = len(msgs)
            if n == 0: continue
            signs = np.sign(msgs); signs[signs == 0] = 1.0
            absm = np.abs(msgs); sp_all = np.prod(signs)
            for ei in range(n):
                sp = sp_all * signs[ei]
                others = np.delete(absm, ei)
                mn = others.min() if n > 1 else 0.0
                c2v[c][ei] = sp * max(mn - offset, 0.0)
        post = L + np.array([sum(c2v[c][p] for (c, p) in vn_to_cn[v]) for v in range(N)])
        bits = (post < 0).astype(np.uint8)
        ok = all(((sum(int(bits[v]) for v in cn_to_vn[c]) & 1) == 0) for c in range(M))
        if ok: break
    return (post < 0).astype(np.uint8)


N_FRAMES = 200
TRUE_R = 20.0 / 104.0  # K_info / N for BG2 set1 Zc=2

for R_used, tag in [(0.5, "硬编 R=0.5 (当前 fitness)"), (TRUE_R, f"真实 R={TRUE_R:.4f}")]:
    print(f"\n=== {N_FRAMES} frames, σ² uses code_rate={R_used:.4f}  [{tag}] ===\n")
    print(f"{'SNR':>5} | {'OMS genome':>14} | {'OMS textbook':>14} | "
          f"{'uncoded hard':>14} | {'coding gain':>11}")
    print("-" * 78)
    for snr in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        pairs = make_channel(snr, N_FRAMES, seed=42 + int(snr*10), code_rate=R_used)
        e_geno = e_ok = 0; tot = 0
        for cw, llr in pairs:
            e_geno += int(np.sum(decode_oms_genome(llr)   != cw))
            e_ok   += int(np.sum(decode_oms_correct(llr)  != cw))
            tot += N
        ber_geno = e_geno / tot
        ber_ok   = e_ok / tot
        ber_unc  = uncoded_ber(snr, N_FRAMES, seed=999 + int(snr*10), code_rate=R_used)
        gain = (ber_unc / ber_geno) if ber_geno > 0 else float('inf')
        print(f"{snr:>5.1f} | {ber_geno:>14.4e} | {ber_ok:>14.4e} | "
              f"{ber_unc:>14.4e} | {gain:>10.2f}x")
