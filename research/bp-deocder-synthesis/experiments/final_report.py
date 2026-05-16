"""Pull the best individual of each gen and the BEST-OF-RUN; print
their live (dead-code-stripped) symbolic expressions via TraceVM."""
from __future__ import annotations
import json, sys, os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ))

import numpy as np
from pushgp.serialize import dict_to_program
from pushgp.trace import trace_program

DIR = PROJ / "results" / "logged_evolution" / "fromscratch_pop100_dedup"
recs = [json.loads(l) for l in (DIR / "individuals.jsonl").open(encoding="utf-8")]


def trace(side_dicts, k, has_llr):
    try:
        prog = dict_to_program(side_dicts)
        evo = np.asarray([10.0 ** float(x) for x in k], dtype=np.float64)
        return trace_program(
            prog,
            ctx_channel_llr=0.5,
            ctx_incoming=np.array([0.6, -0.4, 0.3, -0.7, 0.1, -0.2, 0.5, -0.1], dtype=np.float64),
            ctx_noise_var=1.0, ctx_iter=2, ctx_max_iter=8, ctx_deg=8,
            ctx_edge_index=3, ctx_evo_constants=evo, ctx_has_channel_llr=has_llr,
        )
    except Exception as exc:
        return {"value": None, "live_expr": None, "fault": f"{type(exc).__name__}: {exc}"}


def show(tag, r):
    k = r["log_constants"]
    K = [10**x for x in k]
    print(f"\n{'='*78}\n{tag}\n  fit={r['fitness']:+.5f}  BER={[round(b,4) for b in r['ber_per_snr']]}  v_size={r['v2c_size']}  c_size={r['c2v_size']}  v_idx={r.get('v_idx')} c_idx={r.get('c_idx')}")
    print(f"  K=[{', '.join(f'{x:.3g}' for x in K)}]   (log10={[round(x,2) for x in k]})")
    v_t = trace(r["v2c"], k, has_llr=True)
    c_t = trace(r["c2v"], k, has_llr=False)
    v_expr = v_t.get("live_expr") or "<empty float-stack>"
    c_expr = c_t.get("live_expr") or "<empty float-stack>"
    v_val = v_t.get("value")
    c_val = c_t.get("value")
    print(f"  V2C live ({len(v_expr)} chars, sample_out={v_val}):")
    print(f"    {v_expr}")
    print(f"  C2V live ({len(c_expr)} chars, sample_out={c_val}):")
    print(f"    {c_expr}")


# best of each gen
print("### Best of each generation ###")
for g in sorted({r["gen"] for r in recs}):
    g_recs = [r for r in recs if r["gen"] == g]
    best = min(g_recs, key=lambda r: r["fitness"])
    show(f"Gen {g} BEST", best)

# overall best
overall = min(recs, key=lambda r: r["fitness"])
print("\n\n### OVERALL BEST OF RUN ###")
show("OVERALL BEST", overall)

# also show gen 9 top 3 (the final pop's best)
g9 = sorted([r for r in recs if r["gen"] == 9], key=lambda r: r["fitness"])[:3]
print("\n\n### Final pop (gen 9) top 3 ###")
for i, r in enumerate(g9):
    show(f"Gen 9  rank {i}", r)
