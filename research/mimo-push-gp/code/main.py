"""
Main Research Runner v2 鈥?MIMO-Push GP
16脳8  16-QAM from-scratch algorithm discovery.

Run:
    conda run -n AutoGenOld python -B main.py --continuous
"""
import argparse
import os
import json
import time
import math
import random
from datetime import datetime
from typing import List, Tuple

import numpy as np

from vm import MIMOPushVM, Instruction, program_to_string, program_to_oneliner
from stack_decoder import (StackDecoder, lmmse_detect, kbest_detect,
                           qpsk_constellation, qam16_constellation)
from evolution import (
    random_program, mutate, crossover, deep_copy_program, program_length,
    Individual, FitnessResult,
    tournament_select, lexicase_select,
)


GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def complex_gaussian(shape, rng):
    return (rng.randn(*shape) + 1j * rng.randn(*shape)) / np.sqrt(2.0)


def generate_mimo_sample(Nr, Nt, constellation, snr_db, rng):
    H = complex_gaussian((Nr, Nt), rng)
    x_idx = rng.randint(0, len(constellation), size=Nt)
    x = constellation[x_idx]
    sig_power = float(np.mean(np.abs(H @ x) ** 2))
    noise_var = sig_power / (10 ** (snr_db / 10.0))
    noise = np.sqrt(noise_var / 2.0) * (rng.randn(Nr) + 1j * rng.randn(Nr))
    y = H @ x + noise
    return H, x, y, noise_var


def ber_calc(x_true, x_hat):
    return float(np.mean(x_true != x_hat))


def mse_calc(x_true, x_hat):
    return float(np.mean(np.abs(x_true - x_hat) ** 2))


def constellation_for(mod_order):
    if mod_order == 4:
        return qpsk_constellation()
    if mod_order == 16:
        return qam16_constellation()
    raise ValueError(mod_order)


# --------------------------------------------------------------------------
# Logger
# --------------------------------------------------------------------------

class Logger:
    def __init__(self, path):
        self.path = path
        ensure_dir(os.path.dirname(path))
        self._w("=" * 80 + f"\nMIMO-Push GP v2  started {ts()}\n" + "=" * 80 + "\n")

    def _w(self, s):
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(s)

    def gen(self, g, fit, prog, note=""):
        self._w(f"\n[{ts()}] Gen {g}  {fit}")
        if note:
            self._w(f"  ({note})")
        self._w(f"\n  {program_to_oneliner(prog)}\n")

    def info(self, title, body):
        self._w(f"\n[{ts()}] {title}\n{body}\n" + "-" * 60 + "\n")


# --------------------------------------------------------------------------
# Evaluator
# --------------------------------------------------------------------------

class MIMOEvaluator:
    def __init__(self, Nt=8, Nr=16, mod_order=16,
                 flops_max=80000, max_nodes=300,
                 train_samples=8, snr_choices=None):
        self.Nt = Nt
        self.Nr = Nr
        self.mod_order = mod_order
        self.constellation = constellation_for(mod_order)
        self.vm = MIMOPushVM(flops_max=flops_max, step_max=800)
        self.train_samples = train_samples
        self.snr_choices = snr_choices or [12.0, 16.0, 20.0]
        # Training decoder: NO rescoring (fast eval)
        self.decoder = StackDecoder(
            Nt=Nt, Nr=Nr, constellation=self.constellation,
            max_nodes=max_nodes, vm=self.vm, rescore_interval=0)
        # Challenge decoder: also no rescore, tighter budget
        self.challenge_decoder = StackDecoder(
            Nt=Nt, Nr=Nr, constellation=self.constellation,
            max_nodes=max(32, max_nodes // 2), vm=self.vm, rescore_interval=0)
        # Distance-only program = zero correction (residual scoring)
        self._dist_prog = [Instruction('Float.Const0')]
        self._dist_decoder = StackDecoder(
            Nt=Nt, Nr=Nr, constellation=self.constellation,
            max_nodes=max_nodes, vm=MIMOPushVM(flops_max=flops_max, step_max=100),
            rescore_interval=0)

    def build_dataset(self, seed, n=None, snrs=None):
        rng = np.random.RandomState(seed)
        ds = []
        n = n or self.train_samples
        snrs = snrs or self.snr_choices
        for _ in range(n):
            snr = float(rng.choice(snrs))
            ds.append(generate_mimo_sample(self.Nr, self.Nt,
                                           self.constellation, snr, rng))
        return ds

    def _eval_per_sample(self, prog, dataset, decoder):
        """Returns per-sample BER list and per-sample FLOPs list."""
        bers, flops_list, faults = [], [], 0
        for H, x_true, y, nv in dataset:
            try:
                x_hat, fl = decoder.detect(H, y, prog, noise_var=float(nv))
                bers.append(ber_calc(x_true, x_hat))
                flops_list.append(float(fl))
            except Exception:
                bers.append(1.0)
                flops_list.append(float(self.vm.flops_max * 10))
                faults += 1
        return bers, flops_list, faults

    def evaluate(self, prog, ds_main, ds_hold, dist_bers=None):
        bers_m, fl_m, f1 = self._eval_per_sample(prog, ds_main, self.decoder)
        bers_h, fl_h, f2 = self._eval_per_sample(prog, ds_hold, self.challenge_decoder)

        # LMMSE baseline
        lmmse_bers = []
        for H, x_true, y, nv in ds_main:
            xl, _ = lmmse_detect(H, y, nv, self.constellation)
            lmmse_bers.append(ber_calc(x_true, xl))
        baseline_ber = float(np.mean(lmmse_bers)) if lmmse_bers else 1.0

        ber1 = float(np.mean(bers_m))
        ber2 = float(np.mean(bers_h))
        avg_ber = 0.55 * ber1 + 0.45 * ber2
        avg_fl = 0.55 * float(np.mean(fl_m)) + 0.45 * float(np.mean(fl_h))
        frac_f = (0.55 * f1 + 0.45 * f2) / max(1, len(ds_main))
        gap = abs(ber1 - ber2)
        ratio = avg_ber / max(baseline_ber, 1e-6)

        # Compute improvement over distance-only baseline (per sample)
        dist_improvement = 0.0
        if dist_bers is not None and len(dist_bers) == len(bers_m):
            improvements = [max(0, db - eb) for db, eb in zip(dist_bers, bers_m)]
            dist_improvement = float(np.mean(improvements))

        return FitnessResult(
            ber=avg_ber, mse=0.0, avg_flops=avg_fl,
            code_length=program_length(prog),
            frac_faults=frac_f, baseline_ber=baseline_ber,
            ber_ratio=ratio, generalization_gap=gap),\
               bers_m  # return per-sample BER for lexicase


# --------------------------------------------------------------------------
# Evolution Engine
# --------------------------------------------------------------------------

class EvolutionEngine:
    def __init__(self, pop_size=200, tournament_size=5, elitism=6,
                 mutation_rate=0.75, crossover_rate=0.25, seed=0,
                 evaluator: MIMOEvaluator = None,
                 fresh_injection_rate: float = 0.15,
                 hall_of_fame_seeds: List = None):
        self.pop_size = pop_size
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.rng = np.random.RandomState(seed)
        self.evaluator = evaluator or MIMOEvaluator()
        self.fresh_injection_rate = fresh_injection_rate
        # Hall of fame: best-ever discovered programs, injected as mutations
        self._hall_of_fame = []
        if hall_of_fame_seeds:
            self._hall_of_fame = [deep_copy_program(p) for p in hall_of_fame_seeds]

    def init_population(self) -> List[Individual]:
        """100 % random — no seeds. Varied sizes for diversity."""
        pop = []
        while len(pop) < self.pop_size:
            # Vary program size: 30% short (1-4), 40% medium (5-12), 30% long (12-25)
            r = self.rng.rand()
            if r < 0.3:
                mn, mx = 1, 4
            elif r < 0.7:
                mn, mx = 5, 12
            else:
                mn, mx = 12, 25
            # 30% bias towards environment-access instructions
            prog = random_program(min_size=mn, max_size=mx,
                                  max_depth=2, rng=self.rng, env_bias=0.3)
            pop.append(Individual(program=prog))
        return pop

    def evaluate_pop(self, pop: List[Individual], seed: int):
        # Use FIXED datasets per epoch (changes every 3 gens for fresh evaluation)
        epoch_seed = 7000 + (seed // 3) * 3
        ds = self.evaluator.build_dataset(epoch_seed)
        ds_hold = self.evaluator.build_dataset(epoch_seed + 10000,
                                               n=max(3, self.evaluator.train_samples // 2))
        # Compute distance-only baseline BER per sample (cached per epoch)
        if not hasattr(self, '_cached_epoch') or self._cached_epoch != epoch_seed:
            self._cached_dist_bers, _, _ = self.evaluator._eval_per_sample(
                self.evaluator._dist_prog, ds, self.evaluator._dist_decoder)
            self._cached_epoch = epoch_seed
            self._cached_ds = ds
            self._cached_ds_hold = ds_hold
        dist_bers = self._cached_dist_bers

        n_eval = 0
        for ind in pop:
            if ind.fitness is None:
                fit, per_sample_bers = self.evaluator.evaluate(
                    ind.program, ds, ds_hold, dist_bers)
                ind.fitness = fit
                ind.per_sample_bers = per_sample_bers
                n_eval += 1

    def next_gen(self, pop: List[Individual]) -> List[Individual]:
        ranked = sorted(pop, key=lambda x: x.fitness.composite_score()
                        if x.fitness else 1e9)
        nxt = []
        # Elitism
        for ind in ranked[:self.elitism]:
            e = Individual(program=deep_copy_program(ind.program),
                           fitness=ind.fitness)
            e.age = ind.age + 1
            nxt.append(e)

        # Fresh random injection (diversity maintenance)
        n_fresh = int(self.pop_size * self.fresh_injection_rate)
        for _ in range(n_fresh):
            r = self.rng.rand()
            if r < 0.3:
                mn, mx = 1, 4
            elif r < 0.7:
                mn, mx = 5, 12
            else:
                mn, mx = 12, 25
            prog = random_program(min_size=mn, max_size=mx,
                                  max_depth=2, rng=self.rng, env_bias=0.35)
            nxt.append(Individual(program=prog))

        # Hall-of-fame mutations (aggressively mutate best-ever programs)
        if self._hall_of_fame:
            n_hof = max(2, self.pop_size // 20)  # 5% from hall of fame
            for _ in range(n_hof):
                base = self._hall_of_fame[self.rng.randint(len(self._hall_of_fame))]
                # Heavy mutation to explore AROUND the best
                nm = self.rng.randint(2, 6)
                child = mutate(deep_copy_program(base), self.rng, nm)
                nxt.append(Individual(program=child))

        # Update hall of fame with current best
        if ranked[0].fitness is not None:
            best_score = ranked[0].fitness.composite_score()
            hof_scores = []
            for hp in self._hall_of_fame:
                # Quick check: is this program already in HoF?
                pass
            if len(self._hall_of_fame) < 5:
                self._hall_of_fame.append(deep_copy_program(ranked[0].program))
            elif best_score < min(
                (r.fitness.composite_score() for r in ranked[:5]),
                default=1e9):
                self._hall_of_fame.append(deep_copy_program(ranked[0].program))
                if len(self._hall_of_fame) > 10:
                    self._hall_of_fame = self._hall_of_fame[-5:]

        # Fill rest with offspring
        while len(nxt) < self.pop_size:
            if self.rng.rand() < self.crossover_rate:
                p1 = self._per_sample_lexicase(pop)
                p2 = self._per_sample_lexicase(pop)
                child = crossover(p1.program, p2.program, self.rng)
            else:
                parent = tournament_select(pop, self.tournament_size, self.rng)
                # More aggressive mutation: 40% chance of 2-4 mutations
                nm = 1 if self.rng.rand() < 0.4 else self.rng.randint(2, 5)
                child = mutate(parent.program, self.rng, nm)
            nxt.append(Individual(program=child))
        return nxt[:self.pop_size]

    def _per_sample_lexicase(self, pop: List[Individual]) -> Individual:
        """Per-sample lexicase: shuffle sample indices, filter by epsilon-best on each."""
        cands = [ind for ind in pop if ind.fitness is not None
                 and hasattr(ind, 'per_sample_bers') and ind.per_sample_bers]
        if not cands:
            return pop[self.rng.randint(len(pop))]

        n_samples = len(cands[0].per_sample_bers)
        order = list(range(n_samples)) + ['flops', 'length']
        self.rng.shuffle(order)

        for case in order:
            if len(cands) <= 1:
                break
            if isinstance(case, int):
                vals = []
                for c in cands:
                    if case < len(c.per_sample_bers):
                        vals.append(c.per_sample_bers[case])
                    else:
                        vals.append(1.0)
            elif case == 'flops':
                vals = [c.fitness.avg_flops for c in cands]
            elif case == 'length':
                vals = [float(c.fitness.code_length) for c in cands]
            else:
                continue

            best = min(vals)
            eps = 0.02 * (max(vals) - best + 1e-10)
            survivors = [c for c, v in zip(cands, vals) if v <= best + eps]
            if survivors:
                cands = survivors

        return cands[self.rng.randint(len(cands))]


# --------------------------------------------------------------------------
# Target-system evaluation (full rigorous)
# --------------------------------------------------------------------------

def full_evaluation(prog, Nt, Nr, mod_order, n_trials=200,
                    snr_dbs=None, max_nodes=2000, flops_max=200000):
    if snr_dbs is None:
        snr_dbs = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
    constellation = constellation_for(mod_order)
    vm = MIMOPushVM(flops_max=flops_max, step_max=4000)
    decoder = StackDecoder(Nt=Nt, Nr=Nr, constellation=constellation,
                           max_nodes=max_nodes, vm=vm, rescore_interval=3)
    rng = np.random.RandomState(2026)
    results = []

    for snr in snr_dbs:
        evo_ber, evo_fl = [], []
        lm_ber, lm_fl = [], []
        kb16_ber, kb16_fl = [], []
        kb32_ber, kb32_fl = [], []

        for _ in range(n_trials):
            H, x, y, nv = generate_mimo_sample(Nr, Nt, constellation, snr, rng)

            xh, fl = decoder.detect(H, y, prog, noise_var=float(nv))
            evo_ber.append(ber_calc(x, xh))
            evo_fl.append(fl)

            xl, fl_l = lmmse_detect(H, y, nv, constellation)
            lm_ber.append(ber_calc(x, xl))
            lm_fl.append(fl_l)

            xk16, fl16 = kbest_detect(H, y, constellation, K=16)
            kb16_ber.append(ber_calc(x, xk16))
            kb16_fl.append(fl16)

            xk32, fl32 = kbest_detect(H, y, constellation, K=32)
            kb32_ber.append(ber_calc(x, xk32))
            kb32_fl.append(fl32)

        results.append({
            'snr_db': snr,
            'evolved_ber': float(np.mean(evo_ber)),
            'evolved_flops': float(np.mean(evo_fl)),
            'lmmse_ber': float(np.mean(lm_ber)),
            'lmmse_flops': float(np.mean(lm_fl)),
            'kbest16_ber': float(np.mean(kb16_ber)),
            'kbest16_flops': float(np.mean(kb16_fl)),
            'kbest32_ber': float(np.mean(kb32_ber)),
            'kbest32_flops': float(np.mean(kb32_fl)),
        })
    return results


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def save_json(path, data):
    ensure_dir(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--generations', type=int, default=20)
    p.add_argument('--population', type=int, default=200)
    p.add_argument('--train-samples', type=int, default=8)
    p.add_argument('--train-nt', type=int, default=8)
    p.add_argument('--train-nr', type=int, default=16)
    p.add_argument('--mod-order', type=int, default=16)
    p.add_argument('--train-max-nodes', type=int, default=250)
    p.add_argument('--train-flops-max', type=int, default=80000)
    p.add_argument('--eval-trials', type=int, default=200)
    p.add_argument('--eval-max-nodes', type=int, default=2000)
    p.add_argument('--eval-flops-max', type=int, default=200000)
    p.add_argument('--train-snrs', type=str, default='12,16,20')
    p.add_argument('--eval-snrs', type=str, default='10,12,14,16,18,20')
    p.add_argument('--seed', type=int, default=GLOBAL_SEED)
    p.add_argument('--continuous', action='store_true')
    p.add_argument('--batch-gens', type=int, default=5)
    p.add_argument('--log-suffix', type=str, default='v4',
                   help='Suffix for log file name (algorithm_evolution_<suffix>.log)')
    return p.parse_args()


def parse_snrs(s):
    return [float(x.strip()) for x in s.split(',') if x.strip()]


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

def run_batch(engine, logger, gens, pop=None, start_gen=0):
    if pop is None:
        pop = engine.init_population()
        print(f"Evaluating initial population ({len(pop)} individuals)...",
              flush=True)
        t0 = time.time()
        engine.evaluate_pop(pop, seed=100 + start_gen)
        print(f"  initial eval done in {time.time()-t0:.1f}s", flush=True)
        best = min(pop, key=lambda x: x.fitness.composite_score()
                   if x.fitness else 1e9)
        logger.gen(start_gen, best.fitness, best.program, "initial")
    else:
        best = min(pop, key=lambda x: x.fitness.composite_score()
                   if x.fitness else 1e9)

    history = []
    g = start_gen
    for _ in range(gens):
        g += 1
        t0 = time.time()
        pop = engine.next_gen(pop)
        engine.evaluate_pop(pop, seed=100 + g)
        dt = time.time() - t0
        cur = min(pop, key=lambda x: x.fitness.composite_score()
                  if x.fitness else 1e9)
        if cur.fitness.composite_score() < best.fitness.composite_score():
            best = cur
            note = "NEW BEST"
        else:
            note = ""
        history.append({
            'gen': g,
            'ber': cur.fitness.ber,
            'flops': cur.fitness.avg_flops,
            'len': cur.fitness.code_length,
            'faults': cur.fitness.frac_faults,
            'prog': program_to_oneliner(cur.program),
        })
        logger.gen(g, cur.fitness, cur.program, note)
        # Count unique programs in population
        n_unique = len(set(program_to_oneliner(ind.program) for ind in pop
                          if ind.fitness is not None))
        print(f"Gen {g:04d} [{dt:.1f}s] | BER={cur.fitness.ber:.5f} "
              f"ratio={cur.fitness.ber_ratio:.3f} "
              f"FLOPs={cur.fitness.avg_flops:.0f} "
              f"len={cur.fitness.code_length} "
              f"uniq={n_unique}/{len(pop)} "
              f"faults={cur.fitness.frac_faults:.2f}"
              + (f"  *** {note}" if note else ""))
        if note:
            print(f"  prog: {program_to_oneliner(cur.program)}")
    return best, history, pop, g


def main():
    args = parse_args()
    base = os.path.dirname(os.path.abspath(__file__))
    topic = os.path.dirname(base)
    logs_dir = os.path.join(topic, 'logs')
    results_dir = os.path.join(topic, 'results')
    ensure_dir(logs_dir)
    ensure_dir(results_dir)

    log = Logger(os.path.join(logs_dir, f'algorithm_evolution_{args.log_suffix}.log'))
    train_snrs = parse_snrs(args.train_snrs)
    eval_snrs = parse_snrs(args.eval_snrs)

    log.info("Configuration", json.dumps(vars(args), indent=2))

    evaluator = MIMOEvaluator(
        Nt=args.train_nt, Nr=args.train_nr, mod_order=args.mod_order,
        flops_max=args.train_flops_max, max_nodes=args.train_max_nodes,
        train_samples=args.train_samples, snr_choices=train_snrs)

    # Previously discovered program (gen 10, BER=0.02083, matches KB16)
    # Effective correction: min(2*num_siblings, local_dist + Re(symbol))
    # NOT a hand-designed seed — this was evolved from scratch.
    discovered_v1 = [
        Instruction('Exec.DoTimes', code_block=[
            Instruction('Mat.ElementAt'),
            Instruction('Float.FromInt'),
            Instruction('Float.Min'),
            Instruction('Node.GetLocalDist'),
        ]),
        Instruction('Node.GetSymRe'),
        Instruction('Node.ForEachAncestor', code_block=[
            Instruction('Bool.Dup'),
            Instruction('Matrix.Pop'),
        ]),
        Instruction('Node.ReadMem'),
        Instruction('Vec.Sub'),
        Instruction('Node.ForEachSibling', code_block=[
            Instruction('Node.ForEachSibling', code_block=[
                Instruction('Float.Const2'),
            ]),
            Instruction('Float.Min'),
            Instruction('Node.ChildAt'),
            Instruction('Node.ForEachSibling', code_block=[
                Instruction('Vec.Sub'),
            ]),
        ]),
    ]

    # Discovered second (v4 gen9):
    # Removed from HoF — counterproductive (v4 gen9 is WORSE than distance-only at SNR>=10)
    # kept here for reference only
    discovered_v4_ref = [
        Instruction('Node.GetParent'), Instruction('Float.FromInt'),
    ]  # truncated stub — do not use

    # NEW: Float.GetMMSELB program — a SINGLE instruction that achieves KB16 BER at 60 nodes!
    # This is the MMSE lower bound: min_{x[0:k-1]} ||y_tilde[0:k] - R[0:k,:k]@x - R[0:k,k:]@x_partial||^2
    # It is provably admissible (LB <= actual remaining cost) and gives BER = KB16 at 60 nodes.
    # Adding as HoF seed to help GP discover improvements beyond this.
    discovered_mmse_lb = [Instruction('Float.GetMMSELB')]

    engine = EvolutionEngine(
        pop_size=args.population, tournament_size=6, elitism=8,
        mutation_rate=0.75, crossover_rate=0.25,
        seed=args.seed, evaluator=evaluator,
        hall_of_fame_seeds=[discovered_v1, discovered_mmse_lb])  # v1 + MMSE-LB HoF seeds

    history = []
    pop = None
    g = 0
    best = None

    if args.continuous:
        try:
            while True:
                best, bh, pop, g = run_batch(engine, log, args.batch_gens, pop, g)
                history.extend(bh)

                # Periodic full eval
                print(f"\n--- Full evaluation after gen {g} ---")
                ev = full_evaluation(
                    best.program, Nt=args.train_nt, Nr=args.train_nr,
                    mod_order=args.mod_order, n_trials=args.eval_trials,
                    snr_dbs=eval_snrs, max_nodes=args.eval_max_nodes,
                    flops_max=args.eval_flops_max)
                for r in ev:
                    print(f"  SNR={r['snr_db']:5.1f}  "
                          f"Evo={r['evolved_ber']:.5f}({r['evolved_flops']:.0f})  "
                          f"LMMSE={r['lmmse_ber']:.5f}({r['lmmse_flops']:.0f})  "
                          f"KB16={r['kbest16_ber']:.5f}({r['kbest16_flops']:.0f})  "
                          f"KB32={r['kbest32_ber']:.5f}({r['kbest32_flops']:.0f})")
                log.info(f"Eval gen {g}", json.dumps(ev, indent=2))

                save_json(os.path.join(results_dir, 'run_summary_v2.json'), {
                    'ts': ts(), 'gen': g,
                    'best_fit': repr(best.fitness),
                    'best_prog': program_to_oneliner(best.program),
                    'best_prog_hr': program_to_string(best.program),
                    'eval': ev, 'history': history,
                })
        except KeyboardInterrupt:
            print(f"\nStopped at gen {g}")
    else:
        best, history, pop, g = run_batch(engine, log, args.generations)
        ev = full_evaluation(
            best.program, Nt=args.train_nt, Nr=args.train_nr,
            mod_order=args.mod_order, n_trials=args.eval_trials,
            snr_dbs=eval_snrs, max_nodes=args.eval_max_nodes,
            flops_max=args.eval_flops_max)
        for r in ev:
            print(f"  SNR={r['snr_db']:5.1f}  "
                  f"Evo={r['evolved_ber']:.5f}({r['evolved_flops']:.0f})  "
                  f"LMMSE={r['lmmse_ber']:.5f}  "
                  f"KB16={r['kbest16_ber']:.5f}  "
                  f"KB32={r['kbest32_ber']:.5f}")

    if best:
        save_json(os.path.join(results_dir, 'run_summary_v2.json'), {
            'ts': ts(), 'gen': g,
            'best_fit': repr(best.fitness),
            'best_prog': program_to_oneliner(best.program),
            'best_prog_hr': program_to_string(best.program),
            'eval': ev if 'ev' in dir() else [],
            'history': history,
        })
        log.info("Final best program", program_to_string(best.program))
        print(f"\nBest: {best.fitness}")
        print(f"Program: {program_to_oneliner(best.program)}")


if __name__ == '__main__':
    main()
