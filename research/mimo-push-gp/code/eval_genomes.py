"""
Evaluate interesting genomes from sbp_evolution_cpp_test6.log against baselines.
Uses min N bit errors per SNR point for statistical significance.

Usage:
    # Pure-Python (slow, small system):
    python eval_genomes.py --nt 8 --nr 8 --min-bit-errors 100

    # C++ accelerated (16×16, full eval):
    python eval_genomes.py --nt 16 --nr 16 --min-bit-errors 500 --use-cpp
"""
import argparse
import argparse
import numpy as np
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bp_main_v2 import (
    full_evaluation, baseline_evaluation, print_genome_formulas,
    Genome, N_EVO_CONSTS
)
from vm import Instruction


# ── Define interesting genomes from test6 log ───────────────────────────

def make_genome(down, up, bel, halt, logc=None):
    if logc is None:
        logc = [0.0] * N_EVO_CONSTS
    return Genome(
        prog_down=[Instruction(n) for n in down],
        prog_up=[Instruction(n) for n in up],
        prog_belief=[Instruction(n) for n in bel],
        prog_halt=[Instruction(n) for n in halt],
        log_constants=np.array(logc))


GENOMES = {
    # Dominant genome (Gen 1-60 rank #1): F_down = Re(R[i,j]) via Mat.PeekAt
    "test6_dominant": make_genome(
        down=['Float.Max', 'Mat.PeekAt', 'Node.Pop', 'Mat.PeekAt'],
        up=['Float.EvoConst0', 'Float.Swap', 'Int.GetNumSymbols',
            'Float.EvoConst0', 'Float.Max'],
        bel=['Node.SetMDown', 'Float.Min'],
        halt=['Float.Const1', 'Float.GT'],
        logc=[0.3737, -0.3951, 1.5414, -1.7020]),

    # Gen 59 #4: F_down=max(M_par, exp(C_i))*(MMSE_LB*M_up), F_up=log(max)
    "gen59_interesting": make_genome(
        down=['Float.Exp', 'Float.Max', 'Float.GetMMSELB',
              'Node.GetMUp', 'Float.Mul', 'Float.Max'],
        up=['Float.Max', 'Float.Log'],
        bel=['Node.SetMDown', 'Float.Min'],
        halt=['Float.GetNoiseVar', 'Float.GT'],
        logc=[0.3737, -0.3951, 1.5414, -1.7020]),

    # Gen 59 #3: F_down=C_i-M_up, F_up=log(max), F_belief=min(D_i,M_down)
    "gen59_3_sub_mup": make_genome(
        down=['Node.GetMUp', 'Float.Sub'],
        up=['Float.Max', 'Float.Log'],
        bel=['Node.SetMDown', 'Float.Min'],
        halt=['Float.Const1', 'Float.GT'],
        logc=[0.3737, -0.3951, 1.5414, -1.7020]),
}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate evolved genomes vs baselines")
    p.add_argument('--nt', type=int, default=16, help='Number of Tx antennas')
    p.add_argument('--nr', type=int, default=16, help='Number of Rx antennas')
    p.add_argument('--mod-order', type=int, default=16, help='Modulation order (4/16)')
    p.add_argument('--snrs', type=str, default='16,18,20,22,24',
                   help='Comma-separated SNR values in dB')
    p.add_argument('--n-trials', type=int, default=1000,
                   help='Max samples per SNR (hard cap = 10×)')
    p.add_argument('--min-bit-errors', type=int, default=200,
                   help='Stop when all detectors reach this many bit errors')
    p.add_argument('--max-nodes', type=int, default=2000,
                   help='Max tree nodes for evolved decoder')
    p.add_argument('--flops-max', type=int, default=10_000_000,
                   help='Max flops budget for evolved decoder')
    p.add_argument('--step-max', type=int, default=3000,
                   help='Max VM steps per program execution')
    p.add_argument('--use-cpp', action='store_true',
                   help='Use C++ accelerated evaluator (much faster for 16×16)')
    p.add_argument('--eval-baseline', action='store_true',
                   help='Also evaluate LMMSE/K-Best baselines (slower)')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    p.add_argument('--genomes', type=str, default=None,
                   help='Comma-separated genome keys to evaluate (default: all). '
                        'Keys: test6_dominant,gen59_interesting,gen59_3_sub_mup')
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    Nt = args.nt
    Nr = args.nr
    snr_dbs = [float(s) for s in args.snrs.split(',')]

    print("=" * 70)
    print(f"  Genome Evaluation — min {args.min_bit_errors} bit errors per SNR")
    print(f"  {Nt}×{Nr} MIMO, {args.mod_order}-QAM, SNRs={snr_dbs}")
    print(f"  max_nodes={args.max_nodes}, flops_max={args.flops_max}, "
          f"step_max={args.step_max}")
    print(f"  use_cpp={args.use_cpp}, eval_baseline={args.eval_baseline}")
    print("=" * 70)

    # ── Build C++ evaluator if requested ────────────────────────────────
    cpp_evaluator = None
    if args.use_cpp:
        try:
            from cpp_bridge import CppBPEvaluator
            cpp_evaluator = CppBPEvaluator(
                Nt=Nt, Nr=Nr, mod_order=args.mod_order,
                max_nodes=args.max_nodes,
                flops_max=args.flops_max,
                step_max=args.step_max)
            print(f"  C++ evaluator ready (max_nodes={args.max_nodes})\n")
        except Exception as e:
            print(f"  WARNING: could not load C++ evaluator: {e}")
            print("  Falling back to pure Python.\n")

    # ── Select genomes to evaluate ───────────────────────────────────────
    genome_keys = (args.genomes.split(',') if args.genomes
                   else list(GENOMES.keys()))
    unknown = [k for k in genome_keys if k not in GENOMES]
    if unknown:
        print(f"ERROR: unknown genome keys: {unknown}")
        print(f"Available: {list(GENOMES.keys())}")
        sys.exit(1)

    # 1. Baseline eval
    print("\n--- Baselines ---", flush=True)
    bl = []
    
    # bl = baseline_evaluation(Nt=Nt, Nr=Nr, mod_order=args.mod_order,
    #                          n_trials=args.n_trials, snr_dbs=snr_dbs,
    #                          min_bit_errors=args.min_bit_errors)
    # for r in bl:
    #     print(f"  SNR={r['snr_db']:5.1f}  "
    #           f"LMMSE={r['lmmse_ber']:.6f}({r['lmmse_bit_errors']}err)  "
    #           f"KB16={r['kbest16_ber']:.6f}({r['kbest16_bit_errors']}err)  "
    #           f"KB32={r['kbest32_ber']:.6f}({r['kbest32_bit_errors']}err)  "
    #           f"[{r['samples']}samp]", flush=True)

    # 2. Evaluate each genome
    all_results = {'args': vars(args), 'baselines': bl, 'genomes': {}}
    for name in genome_keys:
        genome = GENOMES[name]
        print(f"\n--- {name} ---", flush=True)
        print_genome_formulas(genome, name)
        t0 = time.time()
        ev = full_evaluation(
            genome, Nt=Nt, Nr=Nr, mod_order=args.mod_order,
            n_trials=args.n_trials, snr_dbs=snr_dbs,
            max_nodes=args.max_nodes, flops_max=args.flops_max,
            step_max=args.step_max, cpp_evaluator=cpp_evaluator,
            min_bit_errors=args.min_bit_errors,
            if_eval_baseline=args.eval_baseline)
        dt = time.time() - t0
        for r in ev:
            line = f"  SNR={r['snr_db']:5.1f}  " \
                   f"Evo={r['evolved_ber']:.6f}({r.get('evolved_bit_errors','?')}err," \
                   f"{r['evolved_flops']:.0f}fl)"
            if args.eval_baseline:
                line += f"  LMMSE={r['lmmse_ber']:.6f}  " \
                        f"KB16={r['kbest16_ber']:.6f}  " \
                        f"KB32={r['kbest32_ber']:.6f}"
            print(line, flush=True)
        print(f"  [{dt:.1f}s]", flush=True)
        all_results['genomes'][name] = ev

    # Save results
    topic = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(topic, 'results')
    os.makedirs(results_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    suffix = f"{'cpp' if args.use_cpp else 'py'}_{Nt}x{Nr}"
    out_path = os.path.join(results_dir, f'genome_eval_{suffix}_{ts}.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
