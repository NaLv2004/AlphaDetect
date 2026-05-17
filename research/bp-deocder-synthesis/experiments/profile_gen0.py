"""Profile pop=10 gen-0 fitness eval to test the user's hypotheses:

  H1: mutation slow because validator needs many tries
      -> we measure init seeding + dedup top-up time and rejection counts.
  H2: a few individuals have pathological complexity and dominate fitness eval
      -> we time each individual's fitness eval separately and report the
         distribution (min, median, max, top-5).

Uses --cpp-seeder to skip the slow Python seeding phase.

Output: prints a structured timing report to stdout. Does NOT compete with
production (production already killed).
"""
from __future__ import annotations

import sys
import time
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # bp-deocder-synthesis
sys.path.insert(0, str(ROOT))

from pushgp.evolution import EvolutionConfig
from pushgp.cpp_seeder_adapter import cpp_parallel_fill_random
from pushgp.genome import Genome
from pushgp.random_program import RandomProgramGenerator
from pushgp.program import program_length

from pushgp_ldpc.eval import FitnessConfig
from pushgp_ldpc.eval_logged import evaluate_genome_with_ber
from ldpc_5g import build_parity


def main():
    POP = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    SEED = 7777
    print(f"[profile] pop={POP} seed={SEED}", flush=True)

    fit_cfg = FitnessConfig(
        info_len_A=20,
        code_length_E=100,
        snr_list=[-3.0, -2.0, -1.0],
        n_frames_per_snr=6,
        max_iter=8,
        early_fail_threshold=0.4,
    )
    par = fit_cfg.par
    print(f"[code] BG{fit_cfg.bgn} set{fit_cfg.set_idx} Zc={fit_cfg.zc}  "
          f"N={par.cols}  M={par.rows}  K_cb_bit={fit_cfg.K_cb_bit}", flush=True)

    # ------ 1. Seeding (C++) -----------------------------------------
    t0 = time.perf_counter()
    pop_v, v_att = cpp_parallel_fill_random(
        "v2c", POP, max_attempts=20_000_000, workers=8, chunk_attempts=2000,
        min_size=4, max_size=30, deg=8, base_seed=SEED * 17 + 1,
    )
    pop_c, c_att = cpp_parallel_fill_random(
        "c2v", POP, max_attempts=20_000_000, workers=8, chunk_attempts=2000,
        min_size=4, max_size=30, deg=8, base_seed=SEED * 17 + 2,
    )
    t_seed = time.perf_counter() - t0
    print(f"[seed] {POP} v + {POP} c in {t_seed:.2f}s  "
          f"(v_attempts={v_att}, c_attempts={c_att})", flush=True)

    rng = np.random.default_rng(SEED)
    rpg = RandomProgramGenerator(rng=rng)
    pop_k = [rpg.random_log_constants() for _ in range(POP)]

    # ------ 2. Per-individual fitness eval (sequential, timed) -------
    print(f"[eval] starting sequential gen-0 fitness eval (pop={POP}) ...", flush=True)
    perm = list(rng.permutation(POP))
    times = []
    fits = []
    sizes = []
    for i in range(POP):
        v = pop_v[perm[i]]
        c = pop_c[i]
        k = pop_k[i]
        sz_v = program_length(v)
        sz_c = program_length(c)
        sizes.append((sz_v, sz_c))
        g = Genome(prog_v2c=v, prog_c2v=c, log_constants=k)

        t1 = time.perf_counter()
        m = evaluate_genome_with_ber(g, fit_cfg)
        dt = time.perf_counter() - t1
        times.append(dt)
        fits.append(m.fitness)
        print(f"[eval] pair {i:2d}/{POP}  size_v={sz_v:3d} size_c={sz_c:3d}  "
              f"fit={m.fitness:+.4f}  time={dt:7.2f}s  valid={m.valid}",
              flush=True)

    # ------ 3. Distribution report -----------------------------------
    times_sorted = sorted(times, reverse=True)
    print()
    print(f"[summary] total fitness time: {sum(times):.1f}s  "
          f"(seq, single worker)")
    print(f"[summary] mean = {statistics.mean(times):.2f}s  "
          f"median = {statistics.median(times):.2f}s  "
          f"stdev = {statistics.pstdev(times):.2f}s")
    print(f"[summary] min = {min(times):.2f}s  max = {max(times):.2f}s  "
          f"max/median = {max(times)/max(0.001, statistics.median(times)):.1f}x")
    print(f"[summary] top-5 by time:")
    for rank, t in enumerate(times_sorted[:5]):
        idx = times.index(t)
        sv, sc = sizes[idx]
        print(f"          #{rank+1}  pair {idx:2d}  size_v={sv:3d} size_c={sc:3d}  "
              f"time={t:.2f}s  fit={fits[idx]:+.4f}")

    # ------ 4. Per-call breakdown for the slowest individual ---------
    slowest = max(range(POP), key=lambda i: times[i])
    sv, sc = sizes[slowest]
    print()
    print(f"[deep] re-running slowest individual (pair {slowest}, "
          f"size_v={sv}, size_c={sc}) with VM-call counter ...")
    g = Genome(
        prog_v2c=pop_v[perm[slowest]],
        prog_c2v=pop_c[slowest],
        log_constants=pop_k[slowest],
    )
    # Wrap make_callables to count calls and total instruction count.
    # NOTE: must patch the symbol re-exported into pushgp_ldpc.eval_logged,
    # because that module imports `make_callables` directly at load time.
    from pushgp_ldpc import eval_logged as _el
    n_v2c_calls = [0]
    n_c2v_calls = [0]
    n_v2c_steps = [0]
    n_c2v_steps = [0]
    n_v2c_faults = [0]
    n_c2v_faults = [0]
    orig = _el.make_callables

    def wrapped(genome, **kw):
        v_fn, c_fn = orig(genome, **kw)
        # Hook the underlying VMs.
        from pushgp_ldpc.adapter import _seed_v2c, _seed_c2v  # noqa
        evo = genome.evo_const_values()
        from pushgp.vm import VM
        vm_v = VM(); vm_c = VM()
        def vfn(L_v, incoming, deg, it, ctx):
            _seed_v2c(vm_v, L_v, incoming, deg, it, int(ctx.get("max_iter", 25)), evo)
            out = vm_v.run(genome.prog_v2c)
            n_v2c_calls[0] += 1
            n_v2c_steps[0] += getattr(vm_v.state, "step_count", 0)
            if getattr(vm_v.state, "fault", False):
                n_v2c_faults[0] += 1
            return 0.0 if out is None else float(out)
        def cfn(incoming, deg, it, ctx):
            _seed_c2v(vm_c, incoming, deg, it, int(ctx.get("max_iter", 25)), evo)
            out = vm_c.run(genome.prog_c2v)
            n_c2v_calls[0] += 1
            n_c2v_steps[0] += getattr(vm_c.state, "step_count", 0)
            if getattr(vm_c.state, "fault", False):
                n_c2v_faults[0] += 1
            return 0.0 if out is None else float(out)
        return vfn, cfn

    _el.make_callables = wrapped
    try:
        t1 = time.perf_counter()
        m = evaluate_genome_with_ber(g, fit_cfg)
        dt = time.perf_counter() - t1
    finally:
        _el.make_callables = orig

    print(f"[deep] time={dt:.2f}s  fit={m.fitness:+.4f}  valid={m.valid}")
    print(f"[deep] V2C: {n_v2c_calls[0]:8d} calls,  "
          f"avg_steps={n_v2c_steps[0]/max(1,n_v2c_calls[0]):7.1f},  "
          f"faults={n_v2c_faults[0]}")
    print(f"[deep] C2V: {n_c2v_calls[0]:8d} calls,  "
          f"avg_steps={n_c2v_steps[0]/max(1,n_c2v_calls[0]):7.1f},  "
          f"faults={n_c2v_faults[0]}")
    total_calls = n_v2c_calls[0] + n_c2v_calls[0]
    total_steps = n_v2c_steps[0] + n_c2v_steps[0]
    if dt > 0:
        print(f"[deep] {total_calls/dt:.0f} VM calls/sec, "
              f"{total_steps/dt:.0f} VM instructions/sec")


if __name__ == "__main__":
    main()
