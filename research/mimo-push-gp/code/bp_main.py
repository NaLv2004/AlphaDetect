"""
Program-Controlled Stack Decoder — Main Experiment Runner.

Goal: Discover MIMO detection algorithms via GP evolution.
The evolved Push program runs on EVERY new node in the stack decoder.
The program controls BP timing, node selection, and stopping — the
framework provides NO hardcoded BP sweep schedule.

Run:
    conda run -n AutoGenOld python -u -B bp_main.py --continuous --log-suffix v2_1
"""
import argparse
import os
import json
import time
import random
from datetime import datetime
from typing import List, Tuple

import numpy as np

from vm import MIMOPushVM, Instruction, program_to_string, program_to_oneliner
from bp_decoder import BPStackDecoder, qam16_constellation, qpsk_constellation
from stack_decoder import lmmse_detect, kbest_detect
from evolution import (
    random_program, random_bp_pattern, mutate, crossover,
    deep_copy_program, program_length,
    Individual, FitnessResult,
    tournament_select, lexicase_select,
)


GLOBAL_SEED = 42


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


def constellation_for(mod_order):
    if mod_order == 4:
        return qpsk_constellation()
    if mod_order == 16:
        return qam16_constellation()
    raise ValueError(mod_order)


def program_has_instruction(prog: List[Instruction], targets) -> bool:
    target_set = set(targets)
    for ins in prog:
        if ins.name in target_set:
            return True
        if ins.code_block and program_has_instruction(ins.code_block, target_set):
            return True
        if ins.code_block2 and program_has_instruction(ins.code_block2, target_set):
            return True
    return False


def program_has_nonlocal_bp_structure(prog: List[Instruction]) -> bool:
    return (program_has_instruction(prog, {'Node.SetScore'}) and
            program_has_instruction(
                prog,
                {
                    'Node.ForEachChild', 'Node.ForEachSibling',
                    'Node.ForEachAncestor', 'Node.GetParent', 'Node.ChildAt'
                }
            ))


# --------------------------------------------------------------------------
# Logger
# --------------------------------------------------------------------------

class Logger:
    def __init__(self, path):
        self.path = path
        ensure_dir(os.path.dirname(path))
        self._w("=" * 80 + f"\nBP-Stack-Decoder GP  started {ts()}\n" + "=" * 80 + "\n")

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
# BP-aware Instruction list for evolution
# --------------------------------------------------------------------------

# Instructions that are especially relevant for BP message passing.
# Biased during random program generation to increase probability
# of discovering BP-like patterns.
BP_FOCUS_INSTRUCTIONS = [
    # Memory read/write — THE message channel
    'Node.ReadMem', 'Node.WriteMem',
    # Graph traversal — visit neighbors to read/write their messages
    'Node.GetParent', 'Node.ChildAt', 'Node.NumChildren',
    'Graph.GetRoot',
    # MapReduce over children / siblings / ancestors — natural BP aggregation
    'Node.ForEachChild', 'Node.ForEachSibling', 'Node.ForEachAncestor',
    # Score access — read/write beliefs
    'Node.GetScore', 'Node.SetScore',
    'Node.GetCumDist', 'Node.GetLocalDist',
    'Node.IsExpanded',
    # Layer / symbol info
    'Node.GetLayer', 'Node.GetSymRe', 'Node.GetSymIm',
    # Channel matrix access
    'Mat.PeekAt', 'Mat.PeekAtIm', 'Mat.Row', 'Mat.VecMul',
    'Vec.PeekAt', 'Vec.PeekAtIm',
    'Vec.Dot', 'Vec.Norm2', 'Vec.Add', 'Vec.Sub', 'Vec.Scale',
    'Mat.ElementAt', 'Vec.ElementAt',
    # Stack management needed for BP
    'Node.Dup', 'Node.Swap', 'Node.Pop',
    'Float.Dup', 'Float.Swap',
    'Int.Dup', 'Int.Swap',
    'Matrix.Dup', 'Vector.Dup',
    # Noise variance
    'Float.GetNoiseVar',
    # Graph info
    'Graph.NodeCount', 'Graph.FrontierCount',
]


# --------------------------------------------------------------------------
# Evaluator
# --------------------------------------------------------------------------

class BPEvaluator:
    """Fitness evaluator using program-controlled stack decoder."""

    def __init__(self, Nt=8, Nr=16, mod_order=16,
                 flops_max=2_000_000, max_nodes=500,
                 train_samples=16, snr_choices=None,
                 step_max=1500):
        self.Nt = Nt
        self.Nr = Nr
        self.mod_order = mod_order
        self.constellation = constellation_for(mod_order)
        self.train_samples = train_samples
        self.snr_choices = snr_choices or [10.0, 12.0, 14.0]
        self.max_nodes = max_nodes

        # Training decoder (program-controlled BP)
        self.decoder = BPStackDecoder(
            Nt=Nt, Nr=Nr, constellation=self.constellation,
            max_nodes=max_nodes,
            vm=MIMOPushVM(flops_max=flops_max, step_max=step_max),
            allow_score_writes=True)

        # Challenge decoder (smaller budget)
        self.challenge_decoder = BPStackDecoder(
            Nt=Nt, Nr=Nr, constellation=self.constellation,
            max_nodes=max(64, max_nodes // 3),
            vm=MIMOPushVM(flops_max=flops_max, step_max=step_max),
            allow_score_writes=True)

        # Ablation decoders: identical search, but Node.SetScore is disabled.
        self.no_bp_decoder = BPStackDecoder(
            Nt=Nt, Nr=Nr, constellation=self.constellation,
            max_nodes=max_nodes,
            vm=MIMOPushVM(flops_max=flops_max, step_max=step_max),
            allow_score_writes=False)
        self.no_bp_challenge_decoder = BPStackDecoder(
            Nt=Nt, Nr=Nr, constellation=self.constellation,
            max_nodes=max(64, max_nodes // 3),
            vm=MIMOPushVM(flops_max=flops_max, step_max=step_max),
            allow_score_writes=False)

        # Distance-only program (cum_dist + 0 correction)
        self._dist_prog = [Instruction('Float.Const0')]
        self._dist_vm = MIMOPushVM(flops_max=flops_max, step_max=100)
        self._dist_decoder = BPStackDecoder(
            Nt=Nt, Nr=Nr, constellation=self.constellation,
            max_nodes=max_nodes, vm=self._dist_vm,
            allow_score_writes=False)

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
        bers, flops_list, faults, bp_total, nonlocal_bp_total = [], [], 0, 0, 0
        for H, x_true, y, nv in dataset:
            try:
                x_hat, fl = decoder.detect(H, y, prog, noise_var=float(nv))
                bers.append(ber_calc(x_true, x_hat))
                flops_list.append(float(fl))
                bp_total += decoder.bp_updates
                nonlocal_bp_total += decoder.nonlocal_bp_updates
            except Exception:
                bers.append(1.0)
                flops_list.append(float(decoder.vm.flops_max * 10))
                faults += 1
        return bers, flops_list, faults, bp_total, nonlocal_bp_total

    def evaluate(self, prog, ds_main, ds_hold, dist_bers=None):
        bers_m, fl_m, f1, bp1, nlbp1 = self._eval_per_sample(
            prog, ds_main, self.decoder)
        bers_h, fl_h, f2, bp2, nlbp2 = self._eval_per_sample(
            prog, ds_hold, self.challenge_decoder)

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
        n_samples = len(ds_main) + len(ds_hold)
        avg_bp = (bp1 + bp2) / max(1, n_samples)
        avg_nonlocal_bp = (nlbp1 + nlbp2) / max(1, n_samples)

        bp_gain = 0.0
        if avg_nonlocal_bp > 0 and program_has_nonlocal_bp_structure(prog):
            bers_nb_m, _, _, _, _ = self._eval_per_sample(
                prog, ds_main, self.no_bp_decoder)
            bers_nb_h, _, _, _, _ = self._eval_per_sample(
                prog, ds_hold, self.no_bp_challenge_decoder)
            avg_ber_nobp = (0.55 * float(np.mean(bers_nb_m)) +
                            0.45 * float(np.mean(bers_nb_h)))
            bp_gain = avg_ber_nobp - avg_ber

        return FitnessResult(
            ber=avg_ber, mse=0.0, avg_flops=avg_fl,
            code_length=program_length(prog),
            frac_faults=frac_f, baseline_ber=baseline_ber,
            ber_ratio=ratio, generalization_gap=gap,
            bp_updates=avg_bp,
            nonlocal_bp_updates=avg_nonlocal_bp,
            bp_gain=bp_gain), bers_m


# --------------------------------------------------------------------------
# Evolution Engine
# --------------------------------------------------------------------------

class BPEvolutionEngine:
    """GP engine specialized for BP algorithm discovery."""

    def __init__(self, pop_size=100, tournament_size=5, elitism=6,
                 mutation_rate=0.75, crossover_rate=0.25, seed=0,
                 evaluator: BPEvaluator = None,
                 fresh_injection_rate: float = 0.20,
                 hall_of_fame_seeds: List = None):
        self.pop_size = pop_size
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.rng = np.random.RandomState(seed)
        self.evaluator = evaluator or BPEvaluator()
        self.fresh_injection_rate = fresh_injection_rate
        self._hall_of_fame = []
        if hall_of_fame_seeds:
            self._hall_of_fame = [deep_copy_program(p) for p in hall_of_fame_seeds]

    def init_population(self) -> List[Individual]:
        """Generate initial random population with BP-seeded subpopulation."""
        pop = []
        # 20% BP-seeded programs (contain SetScore inside traversal blocks)
        n_bp_seed = int(self.pop_size * 0.20)
        for _ in range(n_bp_seed):
            prog = random_bp_pattern(self.rng)
            pop.append(Individual(program=prog))

        # Remaining: diverse random programs
        while len(pop) < self.pop_size:
            r = self.rng.rand()
            if r < 0.3:
                mn, mx, depth = 2, 6, 1
            elif r < 0.7:
                mn, mx, depth = 5, 15, 2
            else:
                mn, mx, depth = 10, 25, 2
            # High env_bias (0.45) to encourage BP-relevant instructions
            prog = random_program(min_size=mn, max_size=mx,
                                  max_depth=depth, rng=self.rng, env_bias=0.45)
            pop.append(Individual(program=prog))
        return pop

    def evaluate_pop(self, pop: List[Individual], seed: int):
        epoch_seed = 7000 + (seed // 3) * 3
        ds = self.evaluator.build_dataset(epoch_seed)
        ds_hold = self.evaluator.build_dataset(
            epoch_seed + 10000,
            n=max(3, self.evaluator.train_samples // 2))

        if not hasattr(self, '_cached_epoch') or self._cached_epoch != epoch_seed:
            self._cached_dist_bers, _, _, _, _ = self.evaluator._eval_per_sample(
                self.evaluator._dist_prog, ds, self.evaluator._dist_decoder)
            self._cached_epoch = epoch_seed
        dist_bers = self._cached_dist_bers

        for ind in pop:
            if ind.fitness is None:
                fit, per_sample = self.evaluator.evaluate(
                    ind.program, ds, ds_hold, dist_bers)
                ind.fitness = fit
                ind.per_sample_bers = per_sample

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

        # Fresh random injection (aggressive diversity for BP discovery)
        n_fresh = int(self.pop_size * self.fresh_injection_rate)
        for i in range(n_fresh):
            # 40% of fresh injection: BP-pattern programs
            if i < int(n_fresh * 0.40):
                prog = random_bp_pattern(self.rng)
            else:
                r = self.rng.rand()
                if r < 0.3:
                    mn, mx, depth = 2, 6, 1
                elif r < 0.7:
                    mn, mx, depth = 5, 15, 2
                else:
                    mn, mx, depth = 10, 25, 2
                prog = random_program(min_size=mn, max_size=mx,
                                      max_depth=depth, rng=self.rng, env_bias=0.45)
            nxt.append(Individual(program=prog))

        # Hall-of-fame mutations
        if self._hall_of_fame:
            n_hof = max(2, self.pop_size // 15)
            for _ in range(n_hof):
                base = self._hall_of_fame[self.rng.randint(len(self._hall_of_fame))]
                nm = self.rng.randint(2, 8)
                child = mutate(deep_copy_program(base), self.rng, nm)
                nxt.append(Individual(program=child))

        # Update hall of fame
        if ranked[0].fitness is not None:
            if len(self._hall_of_fame) < 5:
                self._hall_of_fame.append(deep_copy_program(ranked[0].program))
            elif ranked[0].fitness.composite_score() < min(
                (r.fitness.composite_score() for r in ranked[:5]),
                default=1e9):
                self._hall_of_fame.append(deep_copy_program(ranked[0].program))
                if len(self._hall_of_fame) > 10:
                    self._hall_of_fame = self._hall_of_fame[-5:]

        # Fill with offspring
        while len(nxt) < self.pop_size:
            if self.rng.rand() < self.crossover_rate:
                p1 = self._per_sample_lexicase(pop)
                p2 = self._per_sample_lexicase(pop)
                child = crossover(p1.program, p2.program, self.rng)
            else:
                parent = tournament_select(pop, self.tournament_size, self.rng)
                nm = 1 if self.rng.rand() < 0.4 else self.rng.randint(2, 6)
                child = mutate(parent.program, self.rng, nm)
            nxt.append(Individual(program=child))
        return nxt[:self.pop_size]

    def _per_sample_lexicase(self, pop: List[Individual]) -> Individual:
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
                vals = [c.per_sample_bers[case] if case < len(c.per_sample_bers) else 1.0
                        for c in cands]
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
# Full evaluation (rigorous, with baselines)
# --------------------------------------------------------------------------

def full_evaluation(prog, Nt, Nr, mod_order, n_trials=200,
                    snr_dbs=None, max_nodes=2000, flops_max=5_000_000):
    if snr_dbs is None:
        snr_dbs = [8.0, 10.0, 12.0, 14.0, 16.0]
    constellation = constellation_for(mod_order)
    vm = MIMOPushVM(flops_max=flops_max, step_max=8000)
    decoder = BPStackDecoder(Nt=Nt, Nr=Nr, constellation=constellation,
                             max_nodes=max_nodes, vm=vm)
    rng = np.random.RandomState(2026)
    results = []

    for snr in snr_dbs:
        evo_ber, evo_fl = [], []
        lm_ber, kb16_ber, kb32_ber = [], [], []

        for _ in range(n_trials):
            H, x, y, nv = generate_mimo_sample(Nr, Nt, constellation, snr, rng)

            xh, fl = decoder.detect(H, y, prog, noise_var=float(nv))
            evo_ber.append(ber_calc(x, xh))
            evo_fl.append(fl)

            xl, _ = lmmse_detect(H, y, nv, constellation)
            lm_ber.append(ber_calc(x, xl))

            xk16, _ = kbest_detect(H, y, constellation, K=16)
            kb16_ber.append(ber_calc(x, xk16))

            xk32, _ = kbest_detect(H, y, constellation, K=32)
            kb32_ber.append(ber_calc(x, xk32))

        results.append({
            'snr_db': snr,
            'evolved_ber': float(np.mean(evo_ber)),
            'evolved_flops': float(np.mean(evo_fl)),
            'lmmse_ber': float(np.mean(lm_ber)),
            'kbest16_ber': float(np.mean(kb16_ber)),
            'kbest32_ber': float(np.mean(kb32_ber)),
        })
    return results


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="BP-Stack-Decoder Algorithm Discovery")
    p.add_argument('--generations', type=int, default=30)
    p.add_argument('--population', type=int, default=100)
    p.add_argument('--train-samples', type=int, default=32)
    p.add_argument('--train-nt', type=int, default=8)
    p.add_argument('--train-nr', type=int, default=16)
    p.add_argument('--mod-order', type=int, default=16)
    p.add_argument('--train-max-nodes', type=int, default=500)
    p.add_argument('--train-flops-max', type=int, default=4_000_000)
    p.add_argument('--step-max', type=int, default=3000)
    p.add_argument('--eval-trials', type=int, default=200)
    p.add_argument('--eval-max-nodes', type=int, default=2000)
    p.add_argument('--eval-flops-max', type=int, default=5_000_000)
    p.add_argument('--train-snrs', type=str, default='8,10,12')
    p.add_argument('--eval-snrs', type=str, default='8,10,12,14,16')
    p.add_argument('--seed', type=int, default=GLOBAL_SEED)
    p.add_argument('--continuous', action='store_true')
    p.add_argument('--batch-gens', type=int, default=5)
    p.add_argument('--log-suffix', type=str, default='bp1')
    return p.parse_args()


def parse_snrs(s):
    return [float(x.strip()) for x in s.split(',') if x.strip()]


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
            'gen': g, 'ber': cur.fitness.ber,
            'flops': cur.fitness.avg_flops,
            'len': cur.fitness.code_length,
            'bp_gain': cur.fitness.bp_gain,
            'nonlocal_bp': cur.fitness.nonlocal_bp_updates,
        })
        logger.gen(g, cur.fitness, cur.program, note)
        n_unique = len(set(program_to_oneliner(ind.program) for ind in pop
                          if ind.fitness is not None))
        n_bp_active = sum(1 for ind in pop if ind.fitness is not None
                          and ind.fitness.nonlocal_bp_updates > 0)
        n_bp_helpful = sum(1 for ind in pop if ind.fitness is not None
                           and ind.fitness.nonlocal_bp_updates > 0
                           and ind.fitness.bp_gain > 0)
        print(f"Gen {g:04d} [{dt:.1f}s] | BER={cur.fitness.ber:.5f} "
              f"ratio={cur.fitness.ber_ratio:.3f} "
              f"FLOPs={cur.fitness.avg_flops:.0f} "
              f"len={cur.fitness.code_length} "
              f"uniq={n_unique}/{len(pop)} "
              f"faults={cur.fitness.frac_faults:.2f} "
              f"BPnl={cur.fitness.nonlocal_bp_updates:.1f}"
              f"({n_bp_helpful}/{n_bp_active}) "
              f"gain={cur.fitness.bp_gain:.3f}"
              + (f"  *** {note}" if note else ""))
        if note:
            print(f"  prog: {program_to_oneliner(cur.program)}")
    return best, history, pop, g


def main():
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    base = os.path.dirname(os.path.abspath(__file__))
    topic = os.path.dirname(base)
    logs_dir = os.path.join(topic, 'logs')
    results_dir = os.path.join(topic, 'results')
    ensure_dir(logs_dir)
    ensure_dir(results_dir)

    log = Logger(os.path.join(logs_dir,
                              f'bp_evolution_{args.log_suffix}.log'))
    train_snrs = parse_snrs(args.train_snrs)
    eval_snrs = parse_snrs(args.eval_snrs)

    log.info("Configuration", json.dumps(vars(args), indent=2))
    print(f"\nProgram-Controlled Stack Decoder — Algorithm Discovery")
    print(f"  max_nodes={args.train_max_nodes}, step_max={args.step_max}")
    print(f"  flops_max={args.train_flops_max}, pop={args.population}")
    print(f"  train_snrs={train_snrs}, eval_snrs={eval_snrs}\n")

    evaluator = BPEvaluator(
        Nt=args.train_nt, Nr=args.train_nr, mod_order=args.mod_order,
        flops_max=args.train_flops_max, max_nodes=args.train_max_nodes,
        train_samples=args.train_samples, snr_choices=train_snrs,
        step_max=args.step_max)

    # NO pre-seeded programs — 100% from scratch
    engine = BPEvolutionEngine(
        pop_size=args.population, tournament_size=5, elitism=6,
        mutation_rate=0.75, crossover_rate=0.25,
        seed=args.seed, evaluator=evaluator,
        hall_of_fame_seeds=None)

    history = []
    pop = None
    g = 0
    best = None

    if args.continuous:
        try:
            while True:
                best, bh, pop, g = run_batch(engine, log, args.batch_gens,
                                             pop, g)
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
                          f"LMMSE={r['lmmse_ber']:.5f}  "
                          f"KB16={r['kbest16_ber']:.5f}  "
                          f"KB32={r['kbest32_ber']:.5f}")
                log.info("Full eval", json.dumps(ev, indent=2))

                save_path = os.path.join(results_dir, f'bp_{args.log_suffix}_gen{g}.json')
                with open(save_path, 'w') as f:
                    json.dump({
                        'gen': g,
                        'best_prog': program_to_oneliner(best.program),
                        'best_fitness': {
                            'ber': best.fitness.ber,
                            'flops': best.fitness.avg_flops,
                            'ratio': best.fitness.ber_ratio,
                        },
                        'eval_results': ev,
                        'history': history[-20:],
                    }, f, indent=2)
                print()

        except KeyboardInterrupt:
            print("\nInterrupted.")
    else:
        best, history, pop, g = run_batch(engine, log, args.generations)
        print(f"\n--- Full evaluation ---")
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

    print(f"\nBest program:\n{program_to_string(best.program)}")


if __name__ == '__main__':
    main()
