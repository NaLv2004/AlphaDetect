"""
Main Research Runner for MIMO-Push GP.
Implements a harder zero-prior discovery loop on 32x16 16-QAM MIMO with
dynamic search-tree rescoring and graph-relational program pressure.

Run with:
    conda run -n AutoGenOld python -B main.py
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
from stack_decoder import StackDecoder, lmmse_detect, qpsk_constellation, qam16_constellation
from evolution import (
    random_program, mutate, crossover,
    Individual, FitnessResult,
    tournament_select, lexicase_select,
    program_length, seeded_programs, program_relational_ratio,
)


# ============================================================
# Reproducibility
# ============================================================
GLOBAL_SEED = 1337
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)


# ============================================================
# Utility functions
# ============================================================

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def current_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def complex_gaussian(shape, rng: np.random.RandomState):
    """CN(0,1) complex Gaussian entries."""
    return (rng.randn(*shape) + 1j * rng.randn(*shape)) / np.sqrt(2.0)


def generate_mimo_sample(Nr: int, Nt: int, constellation: np.ndarray,
                         snr_db: float, rng: np.random.RandomState):
    """Generate a single flat-fading MIMO sample."""
    H = complex_gaussian((Nr, Nt), rng)
    x_indices = rng.randint(0, len(constellation), size=Nt)
    x = constellation[x_indices]

    signal_power = np.mean(np.abs(H @ x) ** 2)
    noise_var = signal_power / (10 ** (snr_db / 10.0))
    noise = np.sqrt(noise_var / 2.0) * (rng.randn(Nr) + 1j * rng.randn(Nr))
    y = H @ x + noise
    return H, x, y, noise_var


def ml_detect_exhaustive(H: np.ndarray, y: np.ndarray,
                         constellation: np.ndarray) -> np.ndarray:
    """
    Exact ML detector by exhaustive search.
    Used only for small-scale training cases (e.g., 4x4 QPSK).
    """
    Nt = H.shape[1]
    M = len(constellation)

    # Enumerate all M^Nt combinations.
    # For 4x4 QPSK: 4^4 = 256, which is fine.
    grids = np.meshgrid(*([np.arange(M)] * Nt), indexing='ij')
    indices = np.stack([g.reshape(-1) for g in grids], axis=1)
    symbols = constellation[indices]

    residuals = y[None, :] - np.einsum('ij,bj->bi', H, symbols)
    metrics = np.sum(np.abs(residuals) ** 2, axis=1)
    best_idx = np.argmin(metrics)
    return symbols[best_idx]


def ber(x_true: np.ndarray, x_hat: np.ndarray) -> float:
    """Symbol error rate approximated as BER proxy at symbol level."""
    return float(np.mean(x_true != x_hat))


def mse(x_true: np.ndarray, x_hat: np.ndarray) -> float:
    return float(np.mean(np.abs(x_true - x_hat) ** 2))


def constellation_for_order(mod_order: int) -> np.ndarray:
    if mod_order == 4:
        return qpsk_constellation()
    if mod_order == 16:
        return qam16_constellation()
    raise ValueError(f"Unsupported modulation order: {mod_order}")


def estimate_lmmse_flops(Nr: int, Nt: int) -> float:
    """Rough arithmetic cost for LMMSE solve and hard decision."""
    gram_flops = 8.0 * Nr * Nt * Nt
    regularized_solve_flops = 2.0 * Nt * Nt * Nt
    projection_flops = 16.0 * Nr * Nt
    slicing_flops = 4.0 * Nt
    return gram_flops + regularized_solve_flops + projection_flops + slicing_flops


# ============================================================
# Logging
# ============================================================

class HumanReadableLogger:
    """Continuously log the learned algorithms in human-readable form."""
    def __init__(self, log_file: str):
        self.log_file = log_file
        ensure_dir(os.path.dirname(log_file))
        self._write_header()

    def _write_header(self):
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"MIMO-Push GP Research Log started at {current_timestamp()}\n")
            f.write("=" * 80 + "\n\n")

    def log_generation(self, generation: int, best_fitness: FitnessResult,
                       best_program: List[Instruction], note: str = ""):
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n[{current_timestamp()}] Generation {generation}\n")
            f.write(f"Best fitness: {best_fitness}\n")
            if note:
                f.write(f"Note: {note}\n")
            f.write("Program summary:\n")
            f.write(program_to_oneliner(best_program) + "\n")
            f.write("Program structure:\n")
            f.write(program_to_string(best_program) + "\n")
            f.write("-" * 80 + "\n")

    def log_evaluation(self, title: str, content: str):
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n[{current_timestamp()}] {title}\n")
            f.write(content + "\n")
            f.write("-" * 80 + "\n")


# ============================================================
# Evaluator
# ============================================================

class MIMOEvaluator:
    """Fitness evaluator for the evolved programs."""
    def __init__(self, train_nt: int = 16, train_nr: int = 32,
                 mod_order: int = 16, flops_max: int = 120000,
                 max_nodes: int = 400, train_samples: int = 18,
                 snr_choices: List[float] = None):
        self.train_nt = train_nt
        self.train_nr = train_nr
        self.mod_order = mod_order
        self.vm = MIMOPushVM(flops_max=flops_max, step_max=4000)
        self.constellation_train = constellation_for_order(mod_order)
        self.train_samples = train_samples
        self.snr_choices = snr_choices or [12.0, 16.0, 20.0]
        self.challenge_samples = max(4, math.ceil(0.75 * self.train_samples))
        self.challenge_snr_choices = sorted({float(snr) for snr in (self.snr_choices + [min(self.snr_choices)])})
        self.train_decoder = StackDecoder(
            Nt=self.train_nt, Nr=self.train_nr,
            constellation=self.constellation_train,
            max_nodes=max_nodes,
            vm=self.vm,
        )
        self.challenge_decoder = StackDecoder(
            Nt=self.train_nt, Nr=self.train_nr,
            constellation=self.constellation_train,
            max_nodes=max(48, max_nodes // 2),
            vm=self.vm,
        )

    def _evaluate_dataset(self, program: List[Instruction],
                          dataset: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]],
                          decoder: StackDecoder) -> dict:
        ber_list = []
        mse_list = []
        flops_list = []
        baseline_ber_list = []
        dynamic_delta_list = []
        faults = 0

        for H, x_true, y, noise_var in dataset:
            try:
                x_hat, flops = decoder.detect(H, y, program)
                x_hat_lmmse = lmmse_detect(H, y, noise_var, self.constellation_train)

                ber_list.append(ber(x_true, x_hat))
                mse_list.append(mse(x_true, x_hat))
                flops_list.append(float(flops))
                baseline_ber_list.append(ber(x_true, x_hat_lmmse))
                dynamic_delta_list.append(decoder.last_run_stats.get('avg_rank_change', 0.0))
                if np.any(np.isnan(x_hat)):
                    faults += 1
            except Exception:
                ber_list.append(1.0)
                mse_list.append(10.0)
                flops_list.append(float(self.vm.flops_max * 10))
                baseline_ber_list.append(1.0)
                dynamic_delta_list.append(0.0)
                faults += 1

        avg_ber = float(np.mean(ber_list)) if ber_list else 1.0
        baseline_ber = float(np.mean(baseline_ber_list)) if baseline_ber_list else 1.0
        return {
            'ber': avg_ber,
            'mse': float(np.mean(mse_list)) if mse_list else 10.0,
            'avg_flops': float(np.mean(flops_list)) if flops_list else float(self.vm.flops_max),
            'baseline_ber': baseline_ber,
            'ber_ratio': avg_ber / max(baseline_ber, 1e-6),
            'dynamic_delta': float(np.mean(dynamic_delta_list)) if dynamic_delta_list else 0.0,
            'frac_faults': faults / max(1, len(dataset)),
        }

    def evaluate_program(self, program: List[Instruction],
                         dataset: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]],
                         challenge_dataset: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]) -> FitnessResult:
        """
        Evaluate a program on a dataset.
        Primary metric: BER against the transmitted symbols on the hard 32x16 16-QAM task.
        Secondary metric: beat LMMSE under the same sample distribution.
        Additional pressure: exhibit dynamic rescoring and use relational tree state.
        """
        relational_score = program_relational_ratio(program)
        primary = self._evaluate_dataset(program, dataset, self.train_decoder)
        challenge = self._evaluate_dataset(program, challenge_dataset, self.challenge_decoder)

        avg_ber = 0.55 * primary['ber'] + 0.45 * challenge['ber']
        avg_mse = 0.55 * primary['mse'] + 0.45 * challenge['mse']
        avg_flops = 0.55 * primary['avg_flops'] + 0.45 * challenge['avg_flops']
        baseline_ber = 0.55 * primary['baseline_ber'] + 0.45 * challenge['baseline_ber']
        ber_ratio = max(primary['ber_ratio'], challenge['ber_ratio'])
        dynamic_delta = 0.5 * (primary['dynamic_delta'] + challenge['dynamic_delta'])
        frac_faults = 0.55 * primary['frac_faults'] + 0.45 * challenge['frac_faults']
        generalization_gap = abs(primary['ber'] - challenge['ber'])

        return FitnessResult(
            ber=avg_ber,
            mse=avg_mse,
            avg_flops=avg_flops,
            code_length=program_length(program),
            frac_faults=frac_faults,
            baseline_ber=baseline_ber,
            ber_ratio=ber_ratio,
            dynamic_delta=dynamic_delta,
            relational_score=relational_score,
            generalization_gap=generalization_gap,
        )

    def build_training_dataset(self, seed: int, n_samples: int = None,
                               snr_choices: List[float] = None) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
        rng = np.random.RandomState(seed)
        dataset = []
        if n_samples is None:
            n_samples = self.train_samples
        active_snrs = snr_choices or self.snr_choices
        for _ in range(n_samples):
            snr_db = float(rng.choice(active_snrs))
            dataset.append(generate_mimo_sample(self.train_nr, self.train_nt, self.constellation_train, snr_db, rng))
        return dataset


# ============================================================
# Evolutionary search engine
# ============================================================

class EvolutionEngine:
    """Minimal GP engine for evolving emergent MIMO metrics."""
    def __init__(self,
                 population_size: int = 80,
                 tournament_size: int = 5,
                 elitism: int = 4,
                 mutation_rate: float = 0.75,
                 crossover_rate: float = 0.25,
                 seed: int = 0,
                 evaluator_flops_max: int = 6000,
                 train_max_nodes: int = 400,
                 train_samples: int = 18,
                 train_nt: int = 16,
                 train_nr: int = 32,
                 mod_order: int = 16,
                 train_snr_choices: List[float] = None):
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.rng = np.random.RandomState(seed)
        self.evaluator = MIMOEvaluator(
            train_nt=train_nt,
            train_nr=train_nr,
            mod_order=mod_order,
            flops_max=evaluator_flops_max,
            max_nodes=train_max_nodes,
            train_samples=train_samples,
            snr_choices=train_snr_choices,
        )

    def initialize_population(self) -> List[Individual]:
        population = []
        for prog in seeded_programs():
            population.append(Individual(program=prog))
        while len(population) < self.population_size:
            prog = random_program(min_size=4, max_size=18, max_depth=2, rng=self.rng)
            population.append(Individual(program=prog))
        return population

    def evaluate_population(self, population: List[Individual], dataset_seed: int):
        dataset = self.evaluator.build_training_dataset(seed=dataset_seed)
        challenge_dataset = self.evaluator.build_training_dataset(
            seed=10000 + dataset_seed,
            n_samples=self.evaluator.challenge_samples,
            snr_choices=self.evaluator.challenge_snr_choices,
        )
        for ind in population:
            if ind.fitness is None:
                ind.fitness = self.evaluator.evaluate_program(ind.program, dataset, challenge_dataset)

    def make_next_generation(self, population: List[Individual]) -> List[Individual]:
        # Sort by fitness
        sorted_pop = sorted(population, key=lambda ind: ind.fitness.composite_score())

        next_pop = []
        # Elitism
        for ind in sorted_pop[:self.elitism]:
            elite = Individual(program=ind.program, fitness=ind.fitness)
            elite.age = ind.age + 1
            next_pop.append(elite)

        # Fill rest
        while len(next_pop) < self.population_size:
            if self.rng.rand() < self.crossover_rate:
                p1 = lexicase_select(population, self.rng)
                p2 = lexicase_select(population, self.rng)
                child_prog = crossover(p1.program, p2.program, self.rng)
            else:
                parent = tournament_select(population, self.tournament_size, self.rng)
                n_mut = 1 if self.rng.rand() < 0.7 else 2
                child_prog = mutate(parent.program, self.rng, n_mutations=n_mut)

            next_pop.append(Individual(program=child_prog))

        return next_pop[:self.population_size]

    def run(self, generations: int, logger: HumanReadableLogger):
        population = self.initialize_population()
        self.evaluate_population(population, dataset_seed=100)

        best_overall = min(population, key=lambda ind: ind.fitness.composite_score())
        history = []

        logger.log_generation(0, best_overall.fitness, best_overall.program, note="Initial population")

        for gen in range(1, generations + 1):
            population = self.make_next_generation(population)
            self.evaluate_population(population, dataset_seed=100 + gen)

            best = min(population, key=lambda ind: ind.fitness.composite_score())
            if best.fitness.composite_score() < best_overall.fitness.composite_score():
                best_overall = best
                note = "New overall best"
            else:
                note = "Best of current generation"

            history.append({
                'generation': gen,
                'ber': best.fitness.ber,
                'mse': best.fitness.mse,
                'flops': best.fitness.avg_flops,
                'length': best.fitness.code_length,
                'faults': best.fitness.frac_faults,
                'program': program_to_oneliner(best.program),
            })

            logger.log_generation(gen, best.fitness, best.program, note=note)

            print(
                f"Gen {gen:03d} | best BER={best.fitness.ber:.4f} "
                f"BER/LMMSE={best.fitness.ber_ratio:.3f} dyn={best.fitness.dynamic_delta:.3f} "
                f"FLOPs={best.fitness.avg_flops:.1f} len={best.fitness.code_length} "
                f"faults={best.fitness.frac_faults:.2f}"
            )

        return best_overall, history


# ============================================================
# Target-system evaluation
# ============================================================

def evaluate_on_target_system(best_program: List[Instruction],
                              eval_nt: int, eval_nr: int, mod_order: int,
                              n_trials: int = 120,
                              snr_dbs: List[float] = None,
                              max_nodes: int = 1500,
                              flops_max: int = 20000):
    if snr_dbs is None:
        snr_dbs = [12.0, 16.0, 20.0]

    constellation = constellation_for_order(mod_order)
    vm = MIMOPushVM(flops_max=flops_max, step_max=3000)
    decoder = StackDecoder(Nt=eval_nt, Nr=eval_nr, constellation=constellation, max_nodes=max_nodes, vm=vm)

    rng = np.random.RandomState(2026)
    results = []
    lmmse_flops = estimate_lmmse_flops(eval_nr, eval_nt)

    for snr_db in snr_dbs:
        evo_ber = []
        evo_mse = []
        evo_flops = []
        evo_dynamic = []
        lmmse_ber = []
        lmmse_mse = []

        for _ in range(n_trials):
            H, x_true, y, noise_var = generate_mimo_sample(eval_nr, eval_nt, constellation, snr_db, rng)

            x_hat_evo, flops = decoder.detect(H, y, best_program)
            x_hat_lmmse = lmmse_detect(H, y, noise_var, constellation)

            evo_ber.append(ber(x_true, x_hat_evo))
            evo_mse.append(mse(x_true, x_hat_evo))
            evo_flops.append(flops)
            evo_dynamic.append(decoder.last_run_stats.get('avg_rank_change', 0.0))
            lmmse_ber.append(ber(x_true, x_hat_lmmse))
            lmmse_mse.append(mse(x_true, x_hat_lmmse))

        results.append({
            'snr_db': snr_db,
            'evo_ber': float(np.mean(evo_ber)),
            'evo_mse': float(np.mean(evo_mse)),
            'evo_flops': float(np.mean(evo_flops)),
            'evo_dynamic_delta': float(np.mean(evo_dynamic)),
            'lmmse_ber': float(np.mean(lmmse_ber)),
            'lmmse_mse': float(np.mean(lmmse_mse)),
            'lmmse_flops_est': float(lmmse_flops),
        })

    return results


# ============================================================
# Persistence
# ============================================================

def save_json(path: str, data):
    ensure_dir(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Evolve typed stack programs for MIMO stack-decoder scoring.")
    parser.add_argument('--generations', type=int, default=12)
    parser.add_argument('--population', type=int, default=64)
    parser.add_argument('--train-samples', type=int, default=10)
    parser.add_argument('--train-nt', type=int, default=16)
    parser.add_argument('--train-nr', type=int, default=32)
    parser.add_argument('--eval-nt', type=int, default=16)
    parser.add_argument('--eval-nr', type=int, default=32)
    parser.add_argument('--mod-order', type=int, default=16)
    parser.add_argument('--train-max-nodes', type=int, default=320)
    parser.add_argument('--train-flops-max', type=int, default=120000)
    parser.add_argument('--eval-trials', type=int, default=24)
    parser.add_argument('--eval-max-nodes', type=int, default=2500)
    parser.add_argument('--eval-flops-max', type=int, default=300000)
    parser.add_argument('--train-snrs', type=str, default='12,16,20')
    parser.add_argument('--snrs', type=str, default='12,16,20')
    parser.add_argument('--seed', type=int, default=GLOBAL_SEED)
    parser.add_argument('--continuous', action='store_true')
    parser.add_argument('--batch-generations', type=int, default=4)
    return parser.parse_args()


def parse_snrs(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(',') if item.strip()]


def run_evolution_search(engine: EvolutionEngine, logger: HumanReadableLogger,
                         generations: int, population: List[Individual] = None,
                         start_generation: int = 0):
    if population is None:
        population = engine.initialize_population()
        engine.evaluate_population(population, dataset_seed=100 + start_generation)
        best_overall = min(population, key=lambda ind: ind.fitness.composite_score())
        logger.log_generation(start_generation, best_overall.fitness, best_overall.program, note="Initial population")
    else:
        best_overall = min(population, key=lambda ind: ind.fitness.composite_score())

    history = []
    current_generation = start_generation

    for _ in range(generations):
        current_generation += 1
        population = engine.make_next_generation(population)
        engine.evaluate_population(population, dataset_seed=100 + current_generation)

        best = min(population, key=lambda ind: ind.fitness.composite_score())
        if best.fitness.composite_score() < best_overall.fitness.composite_score():
            best_overall = best
            note = "New overall best"
        else:
            note = "Best of current generation"

        history.append({
            'generation': current_generation,
            'ber': best.fitness.ber,
            'mse': best.fitness.mse,
            'flops': best.fitness.avg_flops,
            'length': best.fitness.code_length,
            'faults': best.fitness.frac_faults,
            'program': program_to_oneliner(best.program),
        })

        logger.log_generation(current_generation, best.fitness, best.program, note=note)
        print(
            f"Gen {current_generation:03d} | best BER={best.fitness.ber:.4f} "
            f"MSE={best.fitness.mse:.4f} FLOPs={best.fitness.avg_flops:.1f} "
            f"len={best.fitness.code_length} faults={best.fitness.frac_faults:.2f}"
        )

    return best_overall, history, population, current_generation


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    topic_dir = os.path.dirname(base_dir)
    logs_dir = os.path.join(topic_dir, 'logs')
    results_dir = os.path.join(topic_dir, 'results')
    ensure_dir(logs_dir)
    ensure_dir(results_dir)

    human_log = os.path.join(logs_dir, 'algorithm_evolution.log')
    logger = HumanReadableLogger(human_log)
    train_snr_dbs = parse_snrs(args.train_snrs)
    eval_snr_dbs = parse_snrs(args.snrs)

    logger.log_evaluation(
        "Run configuration",
        json.dumps({
            'population': args.population,
            'generations': args.generations,
            'train_samples': args.train_samples,
            'train_nt': args.train_nt,
            'train_nr': args.train_nr,
            'eval_nt': args.eval_nt,
            'eval_nr': args.eval_nr,
            'mod_order': args.mod_order,
            'train_max_nodes': args.train_max_nodes,
            'train_flops_max': args.train_flops_max,
            'eval_trials': args.eval_trials,
            'eval_max_nodes': args.eval_max_nodes,
            'eval_flops_max': args.eval_flops_max,
            'train_snr_dbs': train_snr_dbs,
            'eval_snr_dbs': eval_snr_dbs,
            'seed': args.seed,
            'continuous': args.continuous,
            'batch_generations': args.batch_generations,
        }, indent=2, ensure_ascii=False)
    )

    engine = EvolutionEngine(
        population_size=args.population,
        tournament_size=6,
        elitism=6,
        mutation_rate=0.75,
        crossover_rate=0.25,
        seed=args.seed,
        evaluator_flops_max=args.train_flops_max,
        train_max_nodes=args.train_max_nodes,
        train_samples=args.train_samples,
        train_nt=args.train_nt,
        train_nr=args.train_nr,
        mod_order=args.mod_order,
        train_snr_choices=train_snr_dbs,
    )

    history = []
    population = None
    current_generation = 0
    best_individual = None

    if args.continuous:
        try:
            while True:
                best_individual, batch_history, population, current_generation = run_evolution_search(
                    engine=engine,
                    logger=logger,
                    generations=args.batch_generations,
                    population=population,
                    start_generation=current_generation,
                )
                history.extend(batch_history)

                eval_results = evaluate_on_target_system(
                    best_individual.program,
                    eval_nt=args.eval_nt,
                    eval_nr=args.eval_nr,
                    mod_order=args.mod_order,
                    n_trials=args.eval_trials,
                    snr_dbs=eval_snr_dbs,
                    max_nodes=args.eval_max_nodes,
                    flops_max=args.eval_flops_max,
                )
                logger.log_evaluation(
                    f"{args.eval_nr}x{args.eval_nt} {args.mod_order}-QAM evaluation after generation {current_generation}",
                    json.dumps(eval_results, indent=2, ensure_ascii=False)
                )

                artifacts = {
                    'timestamp': current_timestamp(),
                    'mode': 'continuous',
                    'current_generation': current_generation,
                    'best_fitness': {
                        'ber': best_individual.fitness.ber,
                        'mse': best_individual.fitness.mse,
                        'avg_flops': best_individual.fitness.avg_flops,
                        'code_length': best_individual.fitness.code_length,
                        'frac_faults': best_individual.fitness.frac_faults,
                    },
                    'best_program_oneliner': program_to_oneliner(best_individual.program),
                    'best_program_human': program_to_string(best_individual.program),
                    'history': history,
                    'target_config': {
                        'nr': args.eval_nr,
                        'nt': args.eval_nt,
                        'mod_order': args.mod_order,
                    },
                    'eval_target': eval_results,
                    'human_log_file': human_log,
                }
                save_json(os.path.join(results_dir, 'run_summary.json'), artifacts)
        except KeyboardInterrupt:
            print(f"Continuous evolution stopped at generation {current_generation}.")
            if best_individual is None:
                return
            eval_results = evaluate_on_target_system(
                best_individual.program,
                eval_nt=args.eval_nt,
                eval_nr=args.eval_nr,
                mod_order=args.mod_order,
                n_trials=args.eval_trials,
                snr_dbs=eval_snr_dbs,
                max_nodes=args.eval_max_nodes,
                flops_max=args.eval_flops_max,
            )
    else:
        best_individual, history, population, current_generation = run_evolution_search(
            engine=engine,
            logger=logger,
            generations=args.generations,
            population=None,
            start_generation=0,
        )

        eval_results = evaluate_on_target_system(
            best_individual.program,
            eval_nt=args.eval_nt,
            eval_nr=args.eval_nr,
            mod_order=args.mod_order,
            n_trials=args.eval_trials,
            snr_dbs=eval_snr_dbs,
            max_nodes=args.eval_max_nodes,
            flops_max=args.eval_flops_max,
        )

    logger.log_evaluation(
        f"{args.eval_nr}x{args.eval_nt} {args.mod_order}-QAM evaluation",
        json.dumps(eval_results, indent=2, ensure_ascii=False)
    )

    artifacts = {
        'timestamp': current_timestamp(),
        'best_fitness': {
            'ber': best_individual.fitness.ber,
            'mse': best_individual.fitness.mse,
            'avg_flops': best_individual.fitness.avg_flops,
            'code_length': best_individual.fitness.code_length,
            'frac_faults': best_individual.fitness.frac_faults,
        },
        'best_program_oneliner': program_to_oneliner(best_individual.program),
        'best_program_human': program_to_string(best_individual.program),
        'history': history,
        'final_generation': current_generation,
        'target_config': {
            'nr': args.eval_nr,
            'nt': args.eval_nt,
            'mod_order': args.mod_order,
        },
        'eval_target': eval_results,
        'human_log_file': human_log,
    }
    save_json(os.path.join(results_dir, 'run_summary.json'), artifacts)

    print("Best program:")
    print(program_to_string(best_individual.program))
    print(f"\n{args.eval_nr}x{args.eval_nt} {args.mod_order}-QAM evaluation:")
    print(json.dumps(eval_results, indent=2, ensure_ascii=False))
    print(f"\nHuman-readable log written to: {human_log}")


if __name__ == '__main__':
    main()
