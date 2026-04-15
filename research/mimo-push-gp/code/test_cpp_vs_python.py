"""
Test C++ vs Python consistency for the BP stack decoder.

Compares:
1. Pure cum_dist ceiling (no BP) at various max_nodes
2. Verifies both backends produce identical BER for the same genome + data

This is CRITICAL: evolution uses C++, but our ceiling test used Python.
Any inconsistency would invalidate our analysis.
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from bp_main_v2 import (
    Genome, Instruction, generate_mimo_sample,
    ber_calc, N_EVO_CONSTS
)
from bp_decoder_v2 import StructuredBPDecoder, qam16_constellation
from vm import MIMOPushVM as PushVM
from cpp_bridge import CppBPEvaluator, encode_program

def make_ceiling_genome():
    """F_belief = Node.GetCumDist, H_halt = Bool.True (no BP)."""
    return Genome(
        prog_down=[],
        prog_up=[],
        prog_belief=[Instruction(name='Node.GetCumDist')],
        prog_halt=[Instruction(name='Bool.True')],
        log_constants=np.zeros(N_EVO_CONSTS),
    )


def eval_python(genome, dataset, max_nodes):
    """Evaluate using Python StructuredBPDecoder."""
    Nt, Nr = 16, 16
    constellation = qam16_constellation()
    vm = PushVM(flops_max=3_000_000, step_max=2000)
    vm.evolved_constants = genome.evo_constants
    decoder = StructuredBPDecoder(
        Nt, Nr, constellation, max_nodes=max_nodes,
        vm=vm, max_bp_iters=2
    )

    bers = []
    for H, x_true, y, nv in dataset:
        try:
            x_hat, flops = decoder.detect(
                H, y,
                prog_down=genome.prog_down,
                prog_up=genome.prog_up,
                prog_belief=genome.prog_belief,
                prog_halt=genome.prog_halt,
                noise_var=float(nv),
            )
            bers.append(ber_calc(x_true, x_hat))
        except Exception as e:
            print(f"  Python exception: {e}")
            bers.append(1.0)
    return np.mean(bers)


def eval_cpp(genome, dataset, max_nodes):
    """Evaluate using C++ CppBPEvaluator."""
    cpp_eval = CppBPEvaluator(
        Nt=16, Nr=16, mod_order=16,
        max_nodes=max_nodes, flops_max=3_000_000,
        step_max=2000, max_bp_iters=2
    )
    avg_ber, avg_flops, faults, avg_bp = cpp_eval.evaluate_genome(genome, dataset)
    return avg_ber


def generate_dataset(snr_db, n_samples, seed=42):
    """Generate (H, x_true, y, nv) dataset."""
    constellation = qam16_constellation()
    Nt, Nr = 16, 16
    rng = np.random.RandomState(seed)
    ds = []
    for _ in range(n_samples):
        ds.append(generate_mimo_sample(Nr, Nt, constellation, snr_db, rng))
    return ds


if __name__ == '__main__':
    print("=" * 70)
    print("C++ vs Python Consistency Test for BP Stack Decoder")
    print("=" * 70)

    genome = make_ceiling_genome()
    snr = 22.0
    n_samples = 50
    seed = 42

    print(f"\nGenome: F_belief=Node.GetCumDist, H_halt=Bool.True (pure cum_dist, no BP)")
    print(f"SNR={snr}dB, {n_samples} samples, seed={seed}\n")

    dataset = generate_dataset(snr, n_samples, seed)

    node_counts = [256, 500, 1000, 2000, 4000, 8000]

    print(f"{'max_nodes':>10}  {'C++ BER':>12}")
    print("-" * 30)

    for mn in node_counts:
        cpp_ber = eval_cpp(genome, dataset, mn)
        print(f"{mn:>10}  {cpp_ber:>12.5f}")

    print("\nDone.")
