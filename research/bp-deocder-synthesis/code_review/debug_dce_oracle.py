"""Deeper DCE oracle inspection.

For each size-0 individual:
  - Run decode_bp with `return_iters=True` to see if BP early-stops at
    iter 1 (meaning messages don't matter).
  - Instrument c2v_fn / v2c_fn with a counter that records how many
    times they return 0.0 (the silent fallback in decode_bp + adapter).
  - Print llr_c2v after iter 0 to see whether peer messages ever made it
    in.
"""
import pickle
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cpp_seeder"))
sys.path.insert(0, str(ROOT / "cpp_dce"))

from pushgp.genome import Genome, dict_to_instruction, N_EVO_CONSTS
from pushgp_ldpc.adapter import make_callables
from ldpc_5g import decode_bp

DUMP_DIR = ROOT / "code_review"


def deser_prog(lst):
    return [dict_to_instruction(d) for d in lst]


def wrap_count(fn, label, counters):
    def w(*a, **kw):
        try:
            r = fn(*a, **kw)
        except Exception:
            counters[label + "_exc"] += 1
            return 0.0
        if r is None:
            counters[label + "_none"] += 1
            return 0.0
        if not np.isfinite(r):
            counters[label + "_nonfinite"] += 1
            return 0.0
        counters[label + "_calls"] += 1
        if r == 0.0:
            counters[label + "_zero"] += 1
        else:
            counters[label + "_nz_max"] = max(counters[label + "_nz_max"], abs(r))
        return r
    return w


for tag in ("init", "gen0"):
    pkl = DUMP_DIR / f"_dce_dump.{tag}.pkl"
    if not pkl.exists():
        continue
    rec = pickle.load(open(pkl, "rb"))
    par = rec["dce_par"]; rx_llrs = rec["rx_llrs"]
    max_iter = rec["max_iter"]; decimals = rec["decimals"]
    pre_v = [deser_prog(p) for p in rec["pre_pop_v"]]
    pre_c = [deser_prog(p) for p in rec["pre_pop_c"]]
    post_v = [deser_prog(p) for p in rec["post_pop_v"]]
    post_c = [deser_prog(p) for p in rec["post_pop_c"]]
    pop_k = rec["pop_k"]; perm_v = rec["perm_v"]; perm_c = rec["perm_c"]

    zero_v = [i for i, p in enumerate(post_v) if len(p) == 0]
    zero_c = [i for i, p in enumerate(post_c) if len(p) == 0]

    for side, idxs, pre_pop, peer_pop, perm in [
        ("v2c", zero_v, pre_v, pre_c, perm_v),
        ("c2v", zero_c, pre_c, pre_v, perm_c),
    ]:
        for i in idxs:
            prog = pre_pop[i]
            if len(prog) == 0:
                continue
            peer = peer_pop[perm[i]]
            print("\n" + "=" * 70)
            print(f"[{tag}] {side} #{i}  pre-size={len(prog)}  "
                  f"peer #{perm[i]} size={len(peer)}")

            for variant_name, candidate in [("ORIGINAL", prog),
                                             ("EMPTY",    [])]:
                v = candidate if side == "v2c" else peer
                c = peer       if side == "v2c" else candidate
                g = Genome(prog_v2c=v, prog_c2v=c, log_constants=pop_k[i])
                v_fn, c_fn = make_callables(g)
                cnt = {k: 0 for k in
                       ["v_calls","v_zero","v_exc","v_none","v_nonfinite",
                        "c_calls","c_zero","c_exc","c_none","c_nonfinite"]}
                cnt["v_nz_max"] = 0.0; cnt["c_nz_max"] = 0.0
                wv = wrap_count(v_fn, "v", cnt)
                wc = wrap_count(c_fn, "c", cnt)
                for frame_i, rx in enumerate(rx_llrs):
                    post, iters = decode_bp(
                        rx.copy(), par,
                        v2c_fn=wv, c2v_fn=wc,
                        max_iter=max_iter, offset=0.25, code_rate=0.5,
                        return_iters=True)
                    rx_rounded = np.round(rx, decimals)
                    post_rounded = np.round(post, decimals)
                    eq_to_rx = np.array_equal(rx_rounded, post_rounded)
                    print(f"  frame{frame_i} [{variant_name}]: iters_run={iters}"
                          f"  post==rx? {eq_to_rx}  "
                          f"max|post-rx|={float(np.max(np.abs(post-rx))):.3e}")
                print(f"    counters({variant_name}): "
                      f"V calls={cnt['v_calls']} zero={cnt['v_zero']} "
                      f"exc={cnt['v_exc']} none={cnt['v_none']} "
                      f"nz_max={cnt['v_nz_max']:.3g} | "
                      f"C calls={cnt['c_calls']} zero={cnt['c_zero']} "
                      f"exc={cnt['c_exc']} none={cnt['c_none']} "
                      f"nz_max={cnt['c_nz_max']:.3g}")
