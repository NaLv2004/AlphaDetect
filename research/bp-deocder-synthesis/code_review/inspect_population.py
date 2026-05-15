"""
Inspect the OMS seed stack execution trace and survey the final population
from a small re-run.

Run from the bp-deocder-synthesis directory:
    python -B code_review/inspect_population.py
"""

from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ldpc_5g import build_parity
from pushgp.genome import Genome
from pushgp.evolution import EvolutionConfig, evolve
from pushgp.validators import validate_genome, _make_vm, _seed_v2c_stacks, _seed_c2v_stacks, _run
from pushgp_ldpc.adapter import oms_seed_genome, make_callables
from pushgp_ldpc.eval import FitnessConfig, evaluate_genome
from pushgp.vm import VM

DIVIDER = "=" * 70


def _show_stacks(vm: VM, label=""):
    """Print float / int / bool / fvec stack contents (top=right, up to 4)."""
    def top(stack, n=4):
        items = stack.as_list()[-n:]
        return [round(x, 5) if isinstance(x, (float, np.floating)) else x for x in items]

    print("  [%s]" % label)
    print("    float : %s  (bottom→top)" % top(vm.state.floats))
    print("    int   : %s" % top(vm.state.ints))
    print("    bool  : %s" % top(vm.state.bools))
    fvecs_data = vm.state.fvecs.as_list()
    print("    fvec  : %s" % [list(np.round(v, 3)) for v in fvecs_data[-2:]])


def trace_oms_seed():
    seed_g = oms_seed_genome()

    # Fixed example inputs
    deg = 6
    L_v = 2.5          # channel LLR at variable node
    it = 3
    max_iter = 25
    incoming_c2v = np.array([1.2, -0.8, 3.1, -1.5, 0.6])   # deg-1 = 5 values

    print(DIVIDER)
    print("OMS SEED — V2C  (program: FVec.Len ; DoTimes[FVec.At, Float.Add])")
    print(DIVIDER)
    print("Inputs: L_v=%.2f  deg=%d  iter=%d  max_iter=%d" % (L_v, deg, it, max_iter))
    print("incoming_c2v = %s" % list(incoming_c2v))
    print()
    print("Stack layout after seeding")
    print("  float_stack : [L_v]                       (bottom → top)")
    print("  int_stack   : [edge_idx, deg, iter, max_iter]  (max_iter on top)")
    print("  fvec_stack  : [incoming_c2v]")
    print()

    vm = _make_vm(incoming_c2v, channel_llr=L_v, deg=deg, iter_idx=it,
                  max_iter=max_iter, evo_consts=seed_g.evo_const_values())
    _seed_v2c_stacks(vm)
    _show_stacks(vm, "initial stacks")
    print()

    print("Execution trace:")
    print("  1. FVec.Len")
    print("       → pops fvec top (incoming, len=5)")
    print("         fvec_stack is now EMPTY (fvec consumed)")
    print("         pushes 5 onto int_stack")
    print("         int_stack: [..., max_iter=25, 5]")
    print()
    print("  2. Exec.DoTimes  N=int_stack.pop() = 5")
    print("       body runs 5 times: [FVec.At, Float.Add]")
    print("       BUT fvec is gone!  FVec.At on empty fvec = no-op / pops 0.0")
    print("       Actually the fvec was NOT consumed: FVec.Len is non-destructive.")
    print("       Let's check by actually running the program:")
    print()

    result_v2c = vm.run(seed_g.prog_v2c)
    print("  result (float_stack.top) = %.6f" % (result_v2c if result_v2c is not None else float("nan")))
    expected = L_v + sum(incoming_c2v)
    print("  expected (L_v + sum(incoming)) = %.6f" % expected)
    print()

    print(DIVIDER)
    print("OMS SEED — C2V  (11 instructions)")
    print(DIVIDER)
    incoming_v2c = np.array([1.4, -2.1, 3.3, -0.9, 0.7])   # deg-1 = 5
    print("Inputs: deg=%d  iter=%d  max_iter=%d" % (deg, it, max_iter))
    print("incoming_v2c = %s" % list(incoming_v2c))
    print()

    vm2 = _make_vm(incoming_v2c, has_channel_llr=False, deg=deg, iter_idx=it,
                   max_iter=max_iter, evo_consts=seed_g.evo_const_values())
    _seed_c2v_stacks(vm2)
    _show_stacks(vm2, "initial stacks (no L_v in float_stack)")
    print()

    print("Program walkthrough:")
    print("  ┌─────────────────────────────────────────────────────────────────┐")
    print("  │ Phase 1 – SIGN PRODUCT                                          │")
    print("  │   Bool.False      → push False           bool=[False]           │")
    print("  │   FVec.Len        → push N=5             int=[..., 5]           │")
    print("  │   DoTimes N=5 [                                                 │")
    print("  │     FVec.At       → idx from int_stack; push incoming[idx]      │")
    print("  │     Float.Const0  → push 0.0                                    │")
    print("  │     Float.LT      → (incoming[i] < 0) → True if negative        │")
    print("  │     Bool.Xor      → XOR with running parity                     │")
    print("  │   ]  → bool_stack.top = (number of negatives is odd)            │")
    print("  │                                                                  │")
    print("  │ Phase 2 – MIN |x|                                                │")
    print("  │   Float.EvoConst1 → push sentinel=1000 (=10^log_c[1])           │")
    print("  │   FVec.Len        → push N=5                                     │")
    print("  │   DoTimes N=5 [                                                  │")
    print("  │     FVec.At       → push incoming[i]                             │")
    print("  │     Float.Abs     → |incoming[i]|                                │")
    print("  │     Float.Min     → running min over float_stack top pair        │")
    print("  │   ]  → float_stack.top = min(|x_i|)                             │")
    print("  │                                                                  │")
    print("  │ Phase 3 – OFFSET CORRECTION  (OMS: subtract β, clamp ≥ 0)       │")
    print("  │   Float.EvoConst0 → push β=0.25 (=10^log_c[0])                  │")
    print("  │   Float.Sub       → min_abs − β                                  │")
    print("  │   Float.Const0    → push 0.0                                     │")
    print("  │   Float.Max       → max(0, min_abs − β)                          │")
    print("  │                                                                  │")
    print("  │ Phase 4 – APPLY SIGN                                             │")
    print("  │   Exec.If b1=[Float.Neg] b2=[]                                  │")
    print("  │     pops bool_stack; if True → negate magnitude                  │")
    print("  └─────────────────────────────────────────────────────────────────┘")
    print()

    result_c2v = vm2.run(seed_g.prog_c2v)
    n_neg = sum(1 for x in incoming_v2c if x < 0)
    sign = -1.0 if (n_neg % 2 == 1) else 1.0
    mag = max(0.0, min(abs(x) for x in incoming_v2c) - 0.25)
    expected_c2v = sign * mag
    print("  result (float_stack.top) = %.6f" % (result_c2v if result_c2v is not None else float("nan")))
    print("  expected (sign×max(0,min|x|−β)) = %.6f" % expected_c2v)
    print("    sign: %d negatives → sign=%.1f" % (n_neg, sign))
    print("    min|x|=%.4f  β=0.25  magnitude=%.4f" % (min(abs(x) for x in incoming_v2c), mag))
    print()
    print("EvoConst values from seed log_constants:")
    for i, lc in enumerate(seed_g.log_constants):
        if abs(lc) > 1e-9:
            print("  log_c[%d]=%.5f  → 10^lc = %.6g" % (i, lc, 10**lc))


# ─────────────────────────────────────────────────────────────────────────────
# Part 2 – Re-run tiny evolution; dump entire final population
# ─────────────────────────────────────────────────────────────────────────────

def _program_str(prog, indent=0) -> str:
    lines = []
    for ins in prog:
        lines.append("  " * indent + ins.name)
        if ins.code_block:
            lines.append("  " * indent + "  [b1]:")
            lines.append(_program_str(ins.code_block, indent + 2))
        if ins.code_block2:
            lines.append("  " * indent + "  [b2]:")
            lines.append(_program_str(ins.code_block2, indent + 2))
    return "\n".join(lines)


def survey_population():
    print(DIVIDER)
    print("TINY EVOLUTION — FINAL POPULATION SURVEY")
    print(DIVIDER)

    par = build_parity(bgn=2, set_idx=1, zc=2)
    seed = oms_seed_genome()

    fit_cfg = FitnessConfig(
        par=par,
        snr_list=(-3.0, -2.0),
        n_frames_per_snr=4,
        max_iter=6,
        code_rate=0.5,
    )
    cfg = EvolutionConfig(
        pop_size=12,
        generations=8,
        elitism=2,
        tournament_k=3,
        n_mutations=2,
        p_const_tweak=0.2,
        p_crossover=0.5,
        max_retries=15,
        seed=777,
    )

    result = None
    final_pop = []
    final_fits = []

    def on_gen(log, pop, fits):
        nonlocal final_pop, final_fits
        final_pop = [g.copy() for g in pop]
        final_fits = list(fits)
        print(f"  gen={log.gen:02d}  best={log.best_fit:.4f}  mean={log.mean_fit:.4f}  valid={log.n_valid}")

    result = evolve(
        fitness_fn=lambda g: evaluate_genome(g, fit_cfg),
        seeds=[seed],
        cfg=cfg,
        on_generation=on_gen,
    )

    # Sort final pop by fitness
    order = np.argsort([x if np.isfinite(x) else 1e9 for x in final_fits])
    print()
    print("Final population (sorted by fitness):")
    for rank, idx in enumerate(order):
        g = final_pop[idx]
        f = final_fits[idx]
        v2c_len = len(g.prog_v2c)
        c2v_len = len(g.prog_c2v)
        betas = [round(10**lc, 4) for lc in g.log_constants if lc != 0.0]
        nonzero_lc = [(i, round(10**lc, 5)) for i, lc in enumerate(g.log_constants) if abs(lc) > 0.01]
        valid_ok, _ = validate_genome(g, rng=np.random.default_rng(42))
        print(f"\n  [{rank+1:2d}] fit={f:.4f}  V2C_len={v2c_len}  C2V_len={c2v_len}  valid={valid_ok}")
        print(f"       nonzero log_consts: {nonzero_lc}")
        print(f"  --- V2C ---")
        print(_program_str(g.prog_v2c, indent=2))
        print(f"  --- C2V ---")
        print(_program_str(g.prog_c2v, indent=2))


if __name__ == "__main__":
    trace_oms_seed()
    survey_population()
