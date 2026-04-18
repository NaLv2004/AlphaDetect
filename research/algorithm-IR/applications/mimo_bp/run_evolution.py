"""Main entry point for MIMO BP detector evolution.

Usage:
    cd research/algorithm-IR
    conda run -n AutoGenOld python -m applications.mimo_bp.run_evolution [options]

Options:
    --nt            Transmit antennas (default: 16)
    --nr            Receive antennas (default: 16)
    --mod-order     Modulation order (default: 16)
    --snr-db        Training SNR in dB (default: 24)
    --pop-size      Population size (default: 100)
    --generations   Number of generations (default: 500)
    --seed          Random seed (default: 42)
    --max-nodes     Max tree nodes per detection (default: 500)
    --max-bp-iters  Max BP iterations per expansion (default: 5)
    --n-train       Training samples (default: 200)
    --n-test        Test samples (default: 100)
    --log-dir       Output directory (default: results/)
"""
from __future__ import annotations

import argparse
import json
import sys
import pathlib
import time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evolution.config import EvolutionConfig
from evolution.engine import EvolutionEngine
from evolution.genome import IRGenome
from algorithm_ir.frontend.ir_builder import compile_source_to_ir
from applications.mimo_bp.bp_skeleton import bp_skeleton
from applications.mimo_bp.evaluator import MIMOBPFitnessEvaluator


def _make_seed_genomes() -> list[IRGenome]:
    """Create a few hand-crafted seed genomes with known-good structure."""
    seeds = []

    # Seed 1: Classic stack decoder (f_belief = cum_dist)
    seeds.append(_build_genome(
        f_down="def f_down(parent_m_down, local_dist):\n    return parent_m_down + local_dist\n",
        f_up="def f_up(sum_child_ld, sum_child_m_up, n_children):\n    return sum_child_ld + sum_child_m_up\n",
        f_belief="def f_belief(cum_dist, m_down, m_up):\n    return cum_dist\n",
        h_halt="def h_halt(old_root_m_up, new_root_m_up):\n    return 0.0\n",
    ))

    # Seed 2: BP-enhanced (belief uses all three inputs)
    seeds.append(_build_genome(
        f_down="def f_down(parent_m_down, local_dist):\n    return parent_m_down + local_dist\n",
        f_up="def f_up(sum_child_ld, sum_child_m_up, n_children):\n    return sum_child_ld + sum_child_m_up\n",
        f_belief="def f_belief(cum_dist, m_down, m_up):\n    return cum_dist + m_down + m_up\n",
        h_halt="def h_halt(old_root_m_up, new_root_m_up):\n    return 0.0\n",
    ))

    # Seed 3: Weighted belief with subtraction
    seeds.append(_build_genome(
        f_down="def f_down(parent_m_down, local_dist):\n    return parent_m_down + local_dist\n",
        f_up="def f_up(sum_child_ld, sum_child_m_up, n_children):\n    return sum_child_m_up\n",
        f_belief="def f_belief(cum_dist, m_down, m_up):\n    return cum_dist + m_down - m_up\n",
        h_halt="def h_halt(old_root_m_up, new_root_m_up):\n    return 0.0\n",
    ))

    # Seed 4: Normalized message passing
    seeds.append(_build_genome(
        f_down="def f_down(parent_m_down, local_dist):\n    return parent_m_down * 0.5 + local_dist\n",
        f_up="def f_up(sum_child_ld, sum_child_m_up, n_children):\n    return (sum_child_ld + sum_child_m_up) * 0.5\n",
        f_belief="def f_belief(cum_dist, m_down, m_up):\n    return cum_dist + m_down * 0.3 + m_up * 0.3\n",
        h_halt="def h_halt(old_root_m_up, new_root_m_up):\n    return 0.0\n",
    ))

    # Seed 5: Convergence-based halt
    seeds.append(_build_genome(
        f_down="def f_down(parent_m_down, local_dist):\n    return parent_m_down + local_dist\n",
        f_up="def f_up(sum_child_ld, sum_child_m_up, n_children):\n    return sum_child_ld + sum_child_m_up\n",
        f_belief="def f_belief(cum_dist, m_down, m_up):\n    return cum_dist + m_down + m_up\n",
        h_halt="def h_halt(old_root_m_up, new_root_m_up):\n    return abs(old_root_m_up - new_root_m_up) < 0.01\n",
    ))

    return seeds


def _build_genome(f_down: str, f_up: str, f_belief: str, h_halt: str) -> IRGenome:
    """Compile source strings to an IRGenome."""
    programs = {
        "f_down": compile_source_to_ir(f_down, "f_down"),
        "f_up": compile_source_to_ir(f_up, "f_up"),
        "f_belief": compile_source_to_ir(f_belief, "f_belief"),
        "h_halt": compile_source_to_ir(h_halt, "h_halt"),
    }
    return IRGenome(
        programs=programs,
        constants=np.zeros(4),
    )


def main():
    parser = argparse.ArgumentParser(description="MIMO BP Detector Evolution")
    parser.add_argument("--nt", type=int, default=16)
    parser.add_argument("--nr", type=int, default=16)
    parser.add_argument("--mod-order", type=int, default=16)
    parser.add_argument("--snr-db", type=float, default=24.0)
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-nodes", type=int, default=500)
    parser.add_argument("--max-bp-iters", type=int, default=5)
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--log-dir", type=str, default="results")
    args = parser.parse_args()

    print("=" * 60)
    print("MIMO BP Detector Evolution (Algorithm-IR)")
    print("=" * 60)
    print(f"  System:       {args.nt}×{args.nr} {'16QAM' if args.mod_order == 16 else 'QPSK'}")
    print(f"  Training SNR: {args.snr_db} dB")
    print(f"  Population:   {args.pop_size}")
    print(f"  Generations:  {args.generations}")
    print(f"  Max nodes:    {args.max_nodes}")
    print(f"  Max BP iters: {args.max_bp_iters}")
    print(f"  Seed:         {args.seed}")
    print()

    # Setup skeleton
    skeleton = bp_skeleton()
    program_roles = list(skeleton.roles)

    # Setup evolution config
    config = EvolutionConfig(
        population_size=args.pop_size,
        n_generations=args.generations,
        seed=args.seed,
        tournament_size=5,
        elite_count=3,
        mutation_rate=0.8,
        crossover_rate=0.3,
        constant_mutate_sigma=0.1,
        stagnation_threshold=20,
        hard_restart_after=50,
        hall_of_fame_size=10,
        program_roles=program_roles,
        n_constants=4,
        constant_range=(-3.0, 3.0),
        use_cpp=True,
        metric_weights={"ber": 1.0, "avg_flops": 1e-6, "gen_gap": 0.3},
    )

    # Setup fitness evaluator
    print("Initializing C++ evaluator and generating datasets...")
    t0 = time.time()
    evaluator = MIMOBPFitnessEvaluator(
        Nt=args.nt,
        Nr=args.nr,
        mod_order=args.mod_order,
        snr_db=args.snr_db,
        n_train=args.n_train,
        n_test=args.n_test,
        max_nodes=args.max_nodes,
        max_bp_iters=args.max_bp_iters,
        seed=args.seed,
    )
    print(f"  LMMSE baseline BER: {evaluator._lmmse_ber:.6f}")
    print(f"  Setup time: {time.time() - t0:.1f}s")
    print()

    # Setup evolution engine
    engine = EvolutionEngine(config, evaluator, skeleton)

    # Setup logging
    log_dir = pathlib.Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"evolution_log_{args.seed}.jsonl"

    # Generation callback
    def on_generation(gen: int, best_fitness, population):
        if best_fitness is None:
            return
        metrics = best_fitness.metrics
        line = {
            "gen": gen,
            "ber": metrics.get("ber", 1.0),
            "avg_flops": metrics.get("avg_flops", 0),
            "gen_gap": metrics.get("gen_gap", 0),
            "composite": best_fitness.composite_score(),
            "pop_size": len(population),
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(line) + "\n")

        if gen % 10 == 0 or gen < 5:
            print(f"  Gen {gen:4d}  BER={metrics.get('ber', 1.0):.6f}  "
                  f"Flops={metrics.get('avg_flops', 0):.0f}  "
                  f"Gap={metrics.get('gen_gap', 0):.4f}  "
                  f"Score={best_fitness.composite_score():.6f}")

    # Run evolution
    print("Starting evolution...")
    t_start = time.time()

    seed_genomes = _make_seed_genomes()
    print(f"  Seeding with {len(seed_genomes)} hand-crafted genomes")
    best_genome = engine.run(callback=on_generation, seed_genomes=seed_genomes)

    elapsed = time.time() - t_start
    print()
    print("=" * 60)
    print(f"Evolution completed in {elapsed:.1f}s")
    print("=" * 60)

    if best_genome and engine.hall_of_fame:
        best = engine.hall_of_fame[0]
        best_fit = best[1]
        if best_fit:
            print(f"  Best BER:    {best_fit.metrics.get('ber', 1.0):.6f}")
            print(f"  Best Flops:  {best_fit.metrics.get('avg_flops', 0):.0f}")
            print(f"  Gen gap:     {best_fit.metrics.get('gen_gap', 0):.4f}")
            print(f"  LMMSE BER:   {best_fit.metrics.get('lmmse_ber', 0):.6f}")
            print()

            # Save best genome
            best_path = log_dir / f"best_genome_{args.seed}.json"
            with open(best_path, "w") as f:
                json.dump(best[0].serialize(), f, indent=2)
            print(f"  Best genome saved to: {best_path}")

            # Print program formulas
            print("\nEvolved programs:")
            for name, prog_ir in best[0].programs.items():
                from algorithm_ir.regeneration.codegen import emit_python_source
                try:
                    src = emit_python_source(prog_ir)
                    lines = src.strip().split("\n")
                    print(f"\n  {name}:")
                    for line in lines:
                        print(f"    {line}")
                except Exception:
                    print(f"\n  {name}: <failed to decompile>")

    print()


if __name__ == "__main__":
    main()
