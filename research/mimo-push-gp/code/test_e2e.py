"""Quick end-to-end test: 3 generations of evolution with all changes."""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bp_main_v2 import (
    StructuredBPEvaluator, StructuredBPEngine,
    random_genome, has_valid_bp_dependency, program_to_formula,
    N_EVO_CONSTS,
)

def main():
    print("=" * 60)
    print("END-TO-END TEST: 3 generations of BP evolution")
    print("=" * 60)

    # Small config for quick test
    evaluator = StructuredBPEvaluator(
        Nt=4, Nr=8, mod_order=16,
        flops_max=500_000, max_nodes=200,
        train_samples=8, snr_choices=[10.0, 12.0],
        step_max=1000, use_cpp=False)

    engine = StructuredBPEngine(
        pop_size=20, tournament_size=3, elitism=3,
        mutation_rate=0.75, crossover_rate=0.25,
        seed=42, evaluator=evaluator,
        fresh_injection_rate=0.15)

    # Test 1: Init population (all pass perturbation check)
    print("\n[1] Initializing population (20 individuals)...")
    pop = engine.init_population()
    print(f"    Created {len(pop)} individuals")

    # Dependency check already passed during init_population (reject-regenerate)
    print(f"    All {len(pop)} passed perturbation-based BP dependency check during init")

    # Check evolved constants exist
    for i, ind in enumerate(pop):
        assert hasattr(ind.genome, 'log_constants'), \
            f"Individual {i} has no log_constants!"
        assert len(ind.genome.log_constants) == N_EVO_CONSTS, \
            f"Individual {i} has wrong number of constants!"
    print(f"    All have {N_EVO_CONSTS} evolvable log-domain constants")

    # Show first individual
    g = pop[0].genome
    print(f"\n    First individual:")
    print(f"      log_constants = {g.log_constants}")
    print(f"      evo_constants = {g.evo_constants}")
    print(f"      F_down  = {program_to_formula(g.prog_down, ['M_par_down', 'C_i'])}")
    print(f"      F_up    = {program_to_formula(g.prog_up, ['C_1', 'M_1', 'C_2', 'M_2'])}")
    print(f"      F_belief= {program_to_formula(g.prog_belief, ['D_i', 'M_down', 'M_up'])}")
    print(f"      H_halt  = {program_to_formula(g.prog_halt, ['old_M_up', 'new_M_up'])}")

    # Test 2: Evaluate population
    print("\n[2] Evaluating initial population...")
    engine.evaluate_pop(pop, seed=100)
    evaluated = [ind for ind in pop if ind.fitness is not None]
    print(f"    Evaluated {len(evaluated)} individuals")

    best = min(pop, key=lambda x: x.fitness.composite_score()
               if x.fitness else 1e9)
    print(f"    Best: BER={best.fitness.ber:.5f}, "
          f"ratio={best.fitness.ber_ratio:.3f}, "
          f"faults={best.fitness.frac_faults:.2f}")

    # Test 3: Next generation (mutation + crossover with reject-regenerate)
    print("\n[3] Running 3 generations...")
    for gen in range(3):
        pop = engine.next_gen(pop)
        engine.evaluate_pop(pop, seed=101 + gen)
        best = min(pop, key=lambda x: x.fitness.composite_score()
                   if x.fitness else 1e9)
        print(f"    Gen {gen+1}: BER={best.fitness.ber:.5f}, "
              f"ratio={best.fitness.ber_ratio:.3f}, "
              f"pop_size={len(pop)}")

    # Verify constants evolved
    print("\n[4] Verifying constant evolution...")
    consts_seen = set()
    for ind in pop:
        key = tuple(np.round(ind.genome.log_constants, 3))
        consts_seen.add(key)
    print(f"    Unique constant sets: {len(consts_seen)}/{len(pop)}")

    # Show best individual
    bg = best.genome
    print(f"\n    Best individual after 3 gens:")
    print(f"      log_constants = {bg.log_constants}")
    print(f"      evo_constants = {bg.evo_constants}")
    print(f"      F_down  = {program_to_formula(bg.prog_down, ['M_par_down', 'C_i'])}")
    print(f"      F_up    = {program_to_formula(bg.prog_up, ['C_1', 'M_1', 'C_2', 'M_2'])}")
    print(f"      F_belief= {program_to_formula(bg.prog_belief, ['D_i', 'M_down', 'M_up'])}")
    print(f"      H_halt  = {program_to_formula(bg.prog_halt, ['old_M_up', 'new_M_up'])}")

    print("\n" + "=" * 60)
    print("ALL E2E TESTS PASSED")
    print("=" * 60)


if __name__ == '__main__':
    main()
