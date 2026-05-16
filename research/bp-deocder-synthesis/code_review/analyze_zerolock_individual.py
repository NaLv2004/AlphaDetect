"""Reproduce + analyse the zero-lock individual.

V2C program = []   (empty)
  -> at run time the Float stack seed pushes L_v as the only value
  -> output = L_v  (independent of c->v messages and iter)

C2V program = sign(div(sign(div(...7-level chain... PopBack ... PopBack))))
  -> output = prod_{v' in N(c)\v} sign( m_{v'->c} ) in {-1, 0, +1}

Dynamics:
  iter 0: v->c_v = L_v ;  c->v = sign-product of L's = ±1 (0 if any L=0)
  iter 1+: v->c stays L_v (V2C ignores incoming) -> c->v stays ±1
  => effective iters = 1, all later iters are identical (early-stop wastes)

Final post = L_v + sum_{c in N(v)} prod_{v' in N(c)\\v} sign(L_{v'})
   -- a single-step weighted-majority / Gallager-A style decoder
   -- channel magnitude |L_v| competes against d_v (variable degree)
   -- when |L_v| >> d_v -> equivalent to channel-hard decision
   -- when |L_v| ~ d_v  -> votes can flip the hard decision
"""
from __future__ import annotations
import os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np

from pushgp_ldpc.eval import FitnessConfig, _channel_inputs

from pushgp.dce import _try_import_pushgp_cpp_dce
from ldpc_5g import build_parity

cdce = _try_import_pushgp_cpp_dce()

par = build_parity(bgn=2, set_idx=1, zc=2)
N = par.cols; M = par.rows
Kb = 10 if par.bgn == 2 else 22
K_info = Kb * par.zc
print(f"BG2 set1 zc={par.zc}: N={N} M={M} K_info={K_info}  "
      f"actual_rate={K_info/N:.3f}  fitness_uses_rate=0.5")
parH = cdce.build_parity_handle(par)

# adjacency
cn_to_vn = par.cn_to_vn
vn_to_cn = [[] for _ in range(N)]
for c, vns in enumerate(cn_to_vn):
    for pos, v in enumerate(vns):
        vn_to_cn[int(v)].append((c, pos))
deg_v = np.array([len(x) for x in vn_to_cn])
print(f"variable-node degree: min={deg_v.min()} mean={deg_v.mean():.2f} max={deg_v.max()}")

# Programs
PROG_V2C: list[dict] = []   # EMPTY
PROG_C2V = [
    {"name": "Exec.DoTimes",
     "code_block": [
         {"name": "Float.Sign"},
         {"name": "FVec.PopBack"},
         {"name": "Float.Div"},
     ]},
    # tail: trailing Sign per the user's expansion
    # (in the dumped form the 7 nested sign-div are produced via the
    # DoTimes loop count = 25, but Float.Div NOOPs once Float stack
    # has only one element; sign/div alternate until incoming exhausted)
]
# The user explicitly gave the live form for C2V as:
#   Float.Sign(Float.Div(... 7x ...(Float.Sign(FVec.PopBack), FVec.PopBack)))
# That is equivalent to the loop-version above producing
#   prod_{j} sign(m_j)   (with the very first sign discarded by Div NOOP
#   chains).  We use the loop form directly since the binding tests
#   match it.

evo = np.zeros(16, dtype=np.float64)


def make_channel(snr_db: float, n_frames: int, seed: int):
    """Use fitness pipeline EXACTLY: code_rate=0.5 in sigma2, random codewords."""
    cfg = FitnessConfig(par=par, snr_list=(snr_db,), n_frames_per_snr=n_frames,
                        max_iter=20, code_rate=0.5, seed_base=seed)
    return _channel_inputs(cfg, snr_db)   # list of (cw, llr)


def decode_prog(llr, max_iter=20):
    post, _ = cdce.decode_bp(llr.astype(np.float64),
                              parH, PROG_V2C, PROG_C2V, evo,
                              int(max_iter), 0.25, 0.5)
    return (np.asarray(post) < 0).astype(np.uint8)


def decode_hard(llr):
    return (llr < 0).astype(np.uint8)


def decode_analytic(llr, max_iter=20):
    """Pure numpy reproduction of:
       post = L_v + sum_{c} prod_{v'!=v} sign(L_{v'})
       Then 1 BP-style verify; if syndrome fails, no further refinement
       (v->c does not depend on c->v, so iters do nothing).
    """
    L = llr.astype(np.float64)
    # c->v messages: depend on L only (not on c->v from previous iter)
    c2v = []
    for c in range(M):
        vns = cn_to_vn[c]
        signs = np.sign(L[vns])
        signs[signs == 0] = 0.0
        # for each ei: product of signs over others
        prod_all = np.prod(signs) if np.all(signs != 0) else 0.0
        msgs = np.zeros(len(vns))
        for ei in range(len(vns)):
            if signs[ei] != 0 and prod_all != 0:
                msgs[ei] = prod_all * signs[ei]   # = prod over others
            else:
                others = np.delete(signs, ei)
                msgs[ei] = np.prod(others) if (len(others) and np.all(others != 0)) else 0.0
        c2v.append(msgs)
    post = L.copy()
    for v in range(N):
        s = 0.0
        for (c, p) in vn_to_cn[v]:
            s += c2v[c][p]
        post[v] += s
    return (post < 0).astype(np.uint8)


def decode_oms(llr, iters=20, offset=0.5):
    L = llr.astype(np.float64).copy()
    c2v = [np.zeros(len(vns)) for vns in cn_to_vn]
    bits = (L < 0).astype(np.uint8)
    for it in range(iters):
        post = L + np.array([sum(c2v[c][p] for (c, p) in vn_to_cn[v])
                              for v in range(N)])
        v2c = [np.zeros(len(vns)) for vns in cn_to_vn]
        for v in range(N):
            for (c, p) in vn_to_cn[v]:
                v2c[c][p] = post[v] - c2v[c][p]
        for c in range(M):
            msgs = v2c[c]; n = len(msgs)
            if n == 0: continue
            signs = np.sign(msgs); signs[signs == 0] = 1.0
            absm = np.abs(msgs)
            sp_all = np.prod(signs)
            for ei in range(n):
                sp = sp_all * signs[ei]
                others = np.delete(absm, ei)
                mn = others.min() if n > 1 else 0.0
                c2v[c][ei] = sp * max(mn - offset, 0.0)
        post = L + np.array([sum(c2v[c][p] for (c, p) in vn_to_cn[v])
                              for v in range(N)])
        bits = (post < 0).astype(np.uint8)
        ok = True
        for c in range(M):
            if (sum(int(bits[v]) for v in cn_to_vn[c]) & 1) != 0:
                ok = False; break
        if ok:
            break
    return bits


# ============ experiments ============
N_FRAMES = 50
print(f"\n=== BER over {N_FRAMES} frames (random codewords, fitness pipeline) ===\n")
for snr in (7.0, 8.0):
    pairs = make_channel(snr, N_FRAMES, seed=42 + int(snr*10))
    e_prog = e_hard = e_anal = e_oms = 0; tot = 0
    t0 = time.time()
    for cw, llr in pairs:
        # Per-bit errors against the true codeword (not 0!).
        e_prog += int(np.sum(decode_prog(llr) != cw))
        e_hard += int(np.sum(decode_hard(llr) != cw))
        e_anal += int(np.sum(decode_analytic(llr) != cw))
        e_oms  += int(np.sum(decode_oms(llr)      != cw))
        tot += N
    dt = time.time() - t0
    print(f"SNR={snr:>4.1f} dB ({dt:5.1f}s)  "
          f"prog={e_prog/tot:.4e}  "
          f"analytic={e_anal/tot:.4e}  "
          f"channel-hard={e_hard/tot:.4e}  "
          f"OMS(20)={e_oms/tot:.4e}")

# ============ single-frame trace ============
print("\n--- single-frame trace at SNR=7 dB ---")
cw0, llr0 = make_channel(7.0, 1, seed=999)[0]
post_arr, iters_run = cdce.decode_bp(llr0.astype(np.float64), parH,
                                  PROG_V2C, PROG_C2V, evo,
                                  20, 0.25, 0.5)
post = np.asarray(post_arr)
print(f"iters_run = {iters_run}")
print(f"max|post - L_v|              = {np.max(np.abs(post-llr0)):.4f}")
print(f"max|post - L_v| / mean|L_v|  = {np.max(np.abs(post-llr0))/np.mean(np.abs(llr0)):.4f}")
print(f"# bits where sign(post) != sign(L_v): "
      f"{int(np.sum(np.sign(post) != np.sign(llr0)))}")
print(f"first 10 L_v : {np.round(llr0[:10], 3)}")
print(f"first 10 post: {np.round(post[:10], 3)}")
