"""Debug DCE size-0 reduction step by step.

Loads the pkl dumped by `_apply_dce_bp` and, for every size-0 individual,
1. Reruns `behavioral_reduce_bp` in single-thread Python mode (use_cpp=False)
   with verbose hooks so we see EXACTLY which deletions pass equivalence.
2. For the final empty program AND the original program, computes the
   decode_bp post-LLR on every oracle frame and prints rounded values to
   show why they were judged equal.
3. Also probes the message function: prints v2c_fn / c2v_fn outputs on a
   few representative (L_v, incoming) inputs, both for the original and
   the empty program, so we see the actual message magnitudes that BP
   sees.
"""
import pickle
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cpp_seeder"))
sys.path.insert(0, str(ROOT / "cpp_dce"))

from pushgp.genome import (Genome, dict_to_instruction, program_to_list,
                            N_EVO_CONSTS)
from pushgp.program import program_length
from pushgp.dce import behavioral_reduce_bp
from pushgp_ldpc.adapter import make_callables
from ldpc_5g import decode_bp

DUMP_DIR = ROOT / "code_review"

def deser_prog(lst):
    return [dict_to_instruction(d) for d in lst]

def render(prog):
    out = []
    for j, ins in enumerate(prog):
        nm = ins.get("name", "?") if isinstance(ins, dict) else getattr(ins, "name", "?")
        out.append(f"  {j:2d}: {nm}")
    return "\n".join(out)

def post_llr_pair(side, prog, peer, log_consts, par, rx_llrs, max_iter):
    v = prog if side == "v2c" else peer
    c = peer if side == "v2c" else prog
    g = Genome(prog_v2c=v, prog_c2v=c, log_constants=log_consts)
    v_fn, c_fn = make_callables(g)
    outs = []
    for rx in rx_llrs:
        outs.append(decode_bp(rx.copy(), par, v2c_fn=v_fn, c2v_fn=c_fn,
                              max_iter=max_iter, offset=0.25, code_rate=0.5))
    return outs

def probe_message_fn(side, prog, peer, log_consts, rng):
    """Print a few sampled outputs of the message function for `prog`."""
    g = Genome(prog_v2c=(prog if side == "v2c" else peer),
               prog_c2v=(peer if side == "v2c" else prog),
               log_constants=log_consts)
    v_fn, c_fn = make_callables(g)
    fn = v_fn if side == "v2c" else c_fn
    print(f"  -- message fn samples ({side}) --")
    for trial in range(5):
        incoming = rng.uniform(-3.0, 3.0, size=7)
        L_v = float(rng.uniform(-2.0, 2.0))
        if side == "v2c":
            out = fn(L_v, incoming, 8, trial, {"max_iter": 25})
        else:
            out = fn(incoming, 8, trial, {"max_iter": 25})
        print(f"     trial {trial}: L_v={L_v:+.3f}  inc[:3]={incoming[:3].round(3)}"
              f"  out = {out:+.6e}")

for tag in ("init", "gen0"):
    pkl = DUMP_DIR / f"_dce_dump.{tag}.pkl"
    if not pkl.exists():
        continue
    rec = pickle.load(open(pkl, "rb"))
    par = rec["dce_par"]
    rx_llrs = rec["rx_llrs"]
    max_iter = rec["max_iter"]
    decimals = rec["decimals"]
    pre_v = [deser_prog(p) for p in rec["pre_pop_v"]]
    pre_c = [deser_prog(p) for p in rec["pre_pop_c"]]
    post_v = [deser_prog(p) for p in rec["post_pop_v"]]
    post_c = [deser_prog(p) for p in rec["post_pop_c"]]
    pop_k = rec["pop_k"]
    perm_v = rec["perm_v"]; perm_c = rec["perm_c"]

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
                  f"peer={'c' if side=='v2c' else 'v'}#{perm[i]} "
                  f"peer-size={len(peer)}")
            print("PRE-DCE program:")
            print(render(prog))

            log_consts = pop_k[i]

            # 1. Baseline post-LLR (original prog) vs empty-program post-LLR
            print("\n-- post-LLR comparison --")
            base_outs  = post_llr_pair(side, prog,  peer, log_consts, par, rx_llrs, max_iter)
            empty_outs = post_llr_pair(side, [],    peer, log_consts, par, rx_llrs, max_iter)
            for fi, (bo, eo) in enumerate(zip(base_outs, empty_outs)):
                bo_r = np.round(bo, decimals)
                eo_r = np.round(eo, decimals)
                eq = np.array_equal(bo_r, eo_r)
                diff = bo - eo
                print(f"  frame {fi}: equal_after_round={eq}  "
                      f"max|base-empty|={float(np.max(np.abs(diff))):.3e}  "
                      f"baseline[:6]={bo_r[:6]}  empty[:6]={eo_r[:6]}")

            # 2. Probe message-function outputs for both
            rng = np.random.default_rng(20260518)
            print("\n-- message function probes (ORIGINAL prog) --")
            probe_message_fn(side, prog, peer, log_consts, rng)
            print("\n-- message function probes (EMPTY prog) --")
            rng = np.random.default_rng(20260518)
            probe_message_fn(side, [], peer, log_consts, rng)

            # 3. Re-run Python DCE step-by-step (it instruments .removed_positions)
            from pushgp.dce import DCEStats
            st = DCEStats(side=side, size_before=len(prog))
            red = behavioral_reduce_bp(
                prog, side,
                peer_prog=peer, log_constants=log_consts,
                par=par, rx_llrs=rx_llrs, max_iter=max_iter,
                max_passes=800, max_decode_evals=None, decimals=decimals,
                stats=st, use_cpp=False,
            )
            print(f"\n-- Python DCE replay: size {len(prog)} -> {len(red)}  "
                  f"passes={st.passes} fp_evals={st.fp_evals} --")
