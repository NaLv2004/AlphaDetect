"""Find the actual semantic divergence between an original program and
its DCE-reduced version that supposedly passes the 32-point fingerprint
but fails BER equivalence.

Procedure:
  1. Load pair 0 (v 30->9, c 50->15) from the production log.
  2. Reduce v and c using the genome's actual K values and iter ∈ {0,2,4}.
  3. Verify the 32-point fingerprint is bit-equal on both reduced sides.
  4. Wrap v2c_fn / c2v_fn (orig and reduced) so every (L_v, incoming, it)
     call is recorded with its float output.
  5. Run one BP frame at SNR=-2dB with both adapters fed the SAME (cw, llr).
  6. Print the FIRST input where orig output != reduced output (per side).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pushgp.dce import behavioral_reduce
from pushgp.evolution import _behav_fingerprint
from pushgp.serialize import dict_to_program
from pushgp.program import program_length
from pushgp.genome import Genome
from pushgp_ldpc.adapter import make_callables
from ldpc_5g import (
    HTYPE, build_parity, bpsk_modulate, bpsk_llr, decode_bp,
)
from pushgp_ldpc.eval import _random_codeword


def load_pair(idx: int):
    log_path = ROOT / "results" / "logged_evolution" / "fromscratch_pop100_dedup" / "individuals.jsonl"
    n = 0
    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if not rec.get("valid"):
                continue
            if rec.get("v2c_size", 0) < 8 or rec.get("c2v_size", 0) < 8:
                continue
            if n == idx:
                v = dict_to_program(rec["v2c"])
                c = dict_to_program(rec["c2v"])
                k = np.asarray(rec["log_constants"], dtype=np.float64)
                return v, c, k
            n += 1
    raise RuntimeError("not enough pairs")


def fp_multi(side, prog, k_real, iters=(0, 2, 4)):
    return "||".join(_behav_fingerprint(side, prog, evo_consts=k_real, iter_idx=it) for it in iters)


def make_recording(genome):
    """Wrap make_callables so every call is recorded.

    Returns (v_fn_recording, c_fn_recording, trace_v, trace_c) where
    trace_* is a list of (L_v_or_None, incoming_copy, deg, it, output).
    """
    v_fn, c_fn = make_callables(genome)
    trace_v = []
    trace_c = []

    def v_rec(L_v, incoming, deg, it, ctx):
        out = v_fn(L_v, incoming, deg, it, ctx)
        trace_v.append((float(L_v), incoming.copy(), int(deg), int(it), float(out)))
        return out

    def c_rec(incoming, deg, it, ctx):
        out = c_fn(incoming, deg, it, ctx)
        trace_c.append((None, incoming.copy(), int(deg), int(it), float(out)))
        return out

    return v_rec, c_rec, trace_v, trace_c


def main():
    print("=" * 70)
    pair_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print(f"DCE divergence debug — pair {pair_idx}")
    print("=" * 70)

    v, c, k = load_pair(pair_idx)
    # Match Genome.evo_const_values() exactly: clip then 10**x
    g_for_consts = Genome(prog_v2c=v, prog_c2v=c, log_constants=k.copy())
    k_real = g_for_consts.evo_const_values()
    print(f"orig sizes: v={program_length(v)}  c={program_length(c)}")

    # --- Step 1: reduce
    v_red = behavioral_reduce(v, "v2c", evo_consts=k_real)
    c_red = behavioral_reduce(c, "c2v", evo_consts=k_real)
    print(f"red  sizes: v={program_length(v_red)}  c={program_length(c_red)}")

    # --- Step 2: verify fingerprint equality (the claim DCE is making)
    fpv_o = fp_multi("v2c", v,     k_real)
    fpv_r = fp_multi("v2c", v_red, k_real)
    fpc_o = fp_multi("c2v", c,     k_real)
    fpc_r = fp_multi("c2v", c_red, k_real)
    print(f"\n[panel check] v2c fingerprints equal? {fpv_o == fpv_r}")
    print(f"[panel check] c2v fingerprints equal? {fpc_o == fpc_r}")
    if fpv_o != fpv_r or fpc_o != fpc_r:
        print("  PANEL ALREADY DISAGREES — DCE produced a non-equivalent reduction!")
        # Find which of the 32*3 panel entries first disagrees
        for label, fo, fr in [("v2c", fpv_o, fpv_r), ("c2v", fpc_o, fpc_r)]:
            if fo == fr:
                continue
            parts_o = fo.split("||")
            parts_r = fr.split("||")
            for it_idx, (po, pr) in enumerate(zip(parts_o, parts_r)):
                if po != pr:
                    print(f"  [{label}] iter={it_idx*2}: panel-string differs")
                    items_o = po.split("|")
                    items_r = pr.split("|")
                    for i, (a, b) in enumerate(zip(items_o, items_r)):
                        if a != b:
                            print(f"    panel point {i}: orig={a!r}  red={b!r}")
                    break
        return

    # --- Step 3: build identical channel data
    par = build_parity(bgn=2, set_idx=1, zc=2)
    htype = HTYPE[par.bgn - 1][par.set_idx - 1]
    snr_db = -2.0
    rng = np.random.default_rng(10_000 + abs(int(snr_db * 1000)))
    sigma2 = 1.0 / (2.0 * 0.5 * 10.0 ** (snr_db / 10.0))
    sigma = float(np.sqrt(sigma2))
    cw = _random_codeword(par, htype, rng)
    tx = bpsk_modulate(cw[2 * par.zc:])
    rx = tx + sigma * rng.standard_normal(tx.shape)
    llr = np.zeros(par.cols, dtype=np.float64)
    llr[2 * par.zc:] = bpsk_llr(rx, sigma2)

    # --- Step 4: record both runs
    g_o = Genome(prog_v2c=v,     prog_c2v=c,     log_constants=k.copy())
    g_r = Genome(prog_v2c=v_red, prog_c2v=c_red, log_constants=k.copy())
    v_o, c_o, tv_o, tc_o = make_recording(g_o)
    v_r, c_r, tv_r, tc_r = make_recording(g_r)

    post_o = decode_bp(llr, par, v2c_fn=v_o, c2v_fn=c_o,
                       max_iter=4, offset=0.25, code_rate=0.5)
    post_r = decode_bp(llr, par, v2c_fn=v_r, c2v_fn=c_r,
                       max_iter=4, offset=0.25, code_rate=0.5)
    print(f"\n[trace] v2c calls:  orig={len(tv_o)}  red={len(tv_r)}")
    print(f"[trace] c2v calls:  orig={len(tc_o)}  red={len(tc_r)}")

    if len(tv_o) != len(tv_r) or len(tc_o) != len(tc_r):
        print("  WARNING: call counts differ — divergence already affected control flow")

    # --- Step 5: find first divergence in v2c
    print("\n--- searching for first v2c divergence ---")
    n = min(len(tv_o), len(tv_r))
    found = False
    for i in range(n):
        Lo, inc_o, deg_o, it_o, out_o = tv_o[i]
        Lr, inc_r, deg_r, it_r, out_r = tv_r[i]
        # Inputs should be bit-equal up to the divergence point
        same_input = (Lo == Lr and deg_o == deg_r and it_o == it_r
                      and np.array_equal(inc_o, inc_r))
        if not same_input:
            print(f"  call {i}: INPUTS already differ (downstream of an earlier divergence)")
            print(f"    orig: L_v={Lo}  it={it_o}  inc={inc_o}")
            print(f"    red:  L_v={Lr}  it={it_r}  inc={inc_r}")
            found = True
            break
        if out_o != out_r:
            print(f"  call {i}: SAME INPUT, DIFFERENT OUTPUT")
            print(f"    L_v        = {Lo!r}")
            print(f"    incoming   = {inc_o.tolist()}")
            print(f"    deg        = {deg_o}, it = {it_o}")
            print(f"    orig output = {out_o!r}")
            print(f"    red  output = {out_r!r}")
            print(f"    diff       = {out_o - out_r:.6e}")
            # Verify panel does NOT cover this exact (L_v, incoming) row
            from pushgp.evolution import _BEHAV_PANEL_V2C_LV, _BEHAV_PANEL_V2C_INC
            covered = False
            for j in range(32):
                if (_BEHAV_PANEL_V2C_LV[j] == Lo
                        and np.array_equal(_BEHAV_PANEL_V2C_INC[j][:deg_o], inc_o)):
                    covered = True
                    print(f"    !! this input IS row {j} of the panel — fingerprint is wrong")
                    break
            if not covered:
                print(f"    confirmed: this input is NOT in the 32-point panel")
            found = True
            break
    if not found:
        print(f"  no v2c divergence found in {n} calls — checking c2v")

    # --- Step 6: same for c2v
    print("\n--- searching for first c2v divergence ---")
    n = min(len(tc_o), len(tc_r))
    found = False
    for i in range(n):
        _, inc_o, deg_o, it_o, out_o = tc_o[i]
        _, inc_r, deg_r, it_r, out_r = tc_r[i]
        same_input = (deg_o == deg_r and it_o == it_r
                      and np.array_equal(inc_o, inc_r))
        if not same_input:
            print(f"  call {i}: INPUTS already differ")
            break
        if out_o != out_r:
            print(f"  call {i}: SAME INPUT, DIFFERENT OUTPUT")
            print(f"    incoming   = {inc_o.tolist()}")
            print(f"    deg        = {deg_o}, it = {it_o}")
            print(f"    orig output = {out_o!r}")
            print(f"    red  output = {out_r!r}")
            print(f"    diff       = {out_o - out_r:.6e}")
            found = True
            break
    if not found:
        print(f"  no c2v divergence found in {n} calls")

    # Summary: post LLR equality
    print(f"\n[BP post-LLR] np.allclose(post_o, post_r, atol=0)? "
          f"{np.array_equal(post_o, post_r)}")
    if not np.array_equal(post_o, post_r):
        diff = np.abs(post_o - post_r)
        idx = int(np.argmax(diff))
        print(f"  max abs diff = {diff[idx]:.6e} at idx {idx}: "
              f"orig={post_o[idx]!r}  red={post_r[idx]!r}")


if __name__ == "__main__":
    main()
