"""Probe whether peer C2V programs really emit constant 0 on broad
input distributions, not just on BP-iteration inputs.

For each peer that paired with a size-0 V2C:
  1. Print the full peer program structure (including code blocks).
  2. Run c2v_fn(incoming, deg, it, ctx) on N draws from several input
     distributions and print stats: count, %zero, %nan-via-vmrun (None),
     min, max, mean, std, first 8 sample outputs.
"""
import pickle
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cpp_seeder"))

from pushgp.genome import Genome, dict_to_instruction
from pushgp_ldpc.adapter import make_callables

DUMP_DIR = ROOT / "code_review"


def deser_prog(lst):
    return [dict_to_instruction(d) for d in lst]


def render(prog, indent=0):
    out = []
    pad = "  " * indent
    for j, ins in enumerate(prog):
        if isinstance(ins, dict):
            nm = ins.get("name", "?")
            cb1 = ins.get("code_block")
            cb2 = ins.get("code_block2")
        else:
            nm = getattr(ins, "name", "?")
            cb1 = getattr(ins, "code_block", None)
            cb2 = getattr(ins, "code_block2", None)
        line = f"{pad}{j:2d}: {nm}"
        if cb1:
            line += " [block1]"
        if cb2:
            line += " [block2]"
        out.append(line)
        if cb1:
            out.append(render(cb1, indent + 2))
        if cb2:
            out.append(render(cb2, indent + 2))
    return "\n".join(out)


def probe_c2v(c2v_fn, draws, deg, it):
    outs = []
    n_none_via_vmrun = 0
    for inc in draws:
        try:
            r = c2v_fn(inc, deg, it, {"max_iter": 25})
        except Exception:
            r = None
        outs.append(r)
    # raw outputs (before adapter's None->0.0)
    return outs


def probe_via_vm(prog, evo_consts, draws, deg, it, side):
    """Bypass make_callables to see RAW vm.run output (None when faulty)."""
    from pushgp.vm import VM
    raws = []
    for inc in draws:
        vm = VM()
        vm.state.ctx_evo_constants = np.asarray(evo_consts, dtype=np.float64)
        vm.state.ctx_incoming = np.asarray(inc, dtype=np.float64)
        vm.state.ctx_has_channel_llr = (side == "v2c")
        vm.state.ctx_channel_llr = 0.0
        vm.state.ctx_deg = deg
        vm.state.ctx_iter = it
        vm.state.ctx_max_iter = 25
        vm.state.ctx_edge_index = 0
        vm.state.ctx_noise_var = 1.0
        vm.state.ctx_code_rate = 0.5
        # seed stacks (c2v)
        vm.state.ints.push(0); vm.state.ints.push(deg)
        vm.state.ints.push(it); vm.state.ints.push(25)
        vm.state.fvecs.push(vm.state.ctx_incoming.copy())
        raw = vm.run(prog)
        raws.append(raw)
    return raws


def stats_line(label, raw, post_fb):
    n = len(raw)
    n_none = sum(1 for x in raw if x is None)
    n_zero_raw = sum(1 for x in raw if x is not None and x == 0.0)
    finite = [float(x) for x in raw if x is not None and np.isfinite(x)]
    print(f"  [{label}] n={n}  raw_None={n_none}  raw_zero={n_zero_raw}  "
          f"finite={len(finite)}")
    if finite:
        a = np.array(finite)
        print(f"    finite stats: min={a.min():+.3e} max={a.max():+.3e} "
              f"mean={a.mean():+.3e} std={a.std():.3e}")
    print(f"    first 8 raw outputs: {[None if x is None else round(float(x), 6) for x in raw[:8]]}")


for tag in ("init", "gen0"):
    pkl = DUMP_DIR / f"_dce_dump.{tag}.pkl"
    if not pkl.exists():
        continue
    rec = pickle.load(open(pkl, "rb"))
    pre_v = [deser_prog(p) for p in rec["pre_pop_v"]]
    pre_c = [deser_prog(p) for p in rec["pre_pop_c"]]
    post_v = [deser_prog(p) for p in rec["post_pop_v"]]
    pop_k = rec["pop_k"]; perm_v = rec["perm_v"]

    zero_v_idx = [i for i, p in enumerate(post_v) if len(p) == 0]
    for v_i in zero_v_idx:
        peer_idx = perm_v[v_i]
        peer = pre_c[peer_idx]
        if len(peer) == 0:
            continue
        print("\n" + "=" * 70)
        print(f"[{tag}] Peer C2V #{peer_idx} (paired with V #{v_i}) — size={len(peer)}")
        print("Program structure:")
        print(render(peer))
        log_consts = pop_k[v_i]

        # The cpp seeder validates with evo_default = {1,1,1,1,1,1,1,1}
        # (raw 1.0 on the float stack, NOT 10**1).  My genome-derived
        # log_consts produce 10**log_consts.  Probe BOTH so we can tell
        # which evo regime made the validator accept this program.
        evo_seed = np.ones(8, dtype=np.float64)            # seeder default
        evo_genome = (10.0 ** np.asarray(log_consts)).astype(np.float64)

        # adapter-wrapped function
        g = Genome(prog_v2c=pre_v[v_i], prog_c2v=peer, log_constants=log_consts)
        _, c2v_fn = make_callables(g)

        deg = 8
        rng = np.random.default_rng(20260518)
        N = 1000
        distributions = [
            ("all-zero (deg-1=7)",          [np.zeros(7) for _ in range(N)]),
            ("tiny U(-0.01,0.01)",          [rng.uniform(-0.01, 0.01, 7) for _ in range(N)]),
            ("small U(-0.3,0.3)",           [rng.uniform(-0.3, 0.3, 7) for _ in range(N)]),
            ("medium U(-1,1)",              [rng.uniform(-1, 1, 7) for _ in range(N)]),
            ("validator U(-3,3)",           [rng.uniform(-3, 3, 7) for _ in range(N)]),
            ("BP-like: all-zero + tiny noise", [rng.normal(0, 0.05, 7) for _ in range(N)]),
        ]
        for label, draws in distributions:
            for evo_name, evo in [("seed-default(all-1)", evo_seed),
                                  ("genome(10**log)",     evo_genome)]:
                raw = probe_via_vm(peer, evo, draws, deg, 0, "c2v")
                stats_line(f"{label} | evo={evo_name}", raw, None)
