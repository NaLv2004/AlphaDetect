"""Test the BER ceiling of the tree-search framework with F_belief = cum_dist only.

This creates a handcrafted genome where:
- F_belief = Node.GetCumDist (returns cum_dist as the score)
- F_down = no-op (empty)
- F_up = no-op (empty)
- H_halt = Bool.True (halt immediately, no BP iterations)

This should behave like a best-first stack decoder with max_nodes expansions,
which is close to ML performance. The BER gives us the framework ceiling.
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from bp_main_v2 import (
    Genome, Instruction, generate_mimo_sample,
    ber_calc, N_EVO_CONSTS
)
from bp_decoder_v2 import StructuredBPDecoder
from vm import MIMOPushVM as PushVM

def make_ceiling_genome():
    """F_belief = Node.GetCumDist (pushes cum_dist on top of stack)."""
    return Genome(
        prog_down=[],   # no-op
        prog_up=[],     # no-op
        prog_belief=[Instruction(name='Node.GetCumDist')],  # return cum_dist
        prog_halt=[Instruction(name='Bool.True')],  # halt immediately (no BP)
        log_constants=np.zeros(N_EVO_CONSTS),
    )

def evaluate_ceiling(snr_db, n_samples=500, max_nodes=1000, seed=42):
    """Evaluate the ceiling genome."""
    constellation = np.array([-3, -1, 1, 3], dtype=float)
    Nt, Nr = 16, 16
    rng = np.random.RandomState(seed)
    
    genome = make_ceiling_genome()
    vm = PushVM(flops_max=3_000_000, step_max=2000)
    vm.evolved_constants = genome.evo_constants
    decoder = StructuredBPDecoder(
        Nt, Nr, constellation, max_nodes=max_nodes, 
        vm=vm, max_bp_iters=2
    )
    
    bers = []
    bit_errors = 0
    total_bits = 0
    for i in range(n_samples):
        H, x_true, y, nv = generate_mimo_sample(Nr, Nt, constellation, snr_db, rng)
        try:
            x_hat, flops = decoder.detect(
                H, y,
                prog_down=genome.prog_down,
                prog_up=genome.prog_up,
                prog_belief=genome.prog_belief,
                prog_halt=genome.prog_halt,
                noise_var=float(nv),
            )
            b = ber_calc(x_true, x_hat)
            bers.append(b)
            # Count bit errors
            bits_per_sym = int(np.log2(len(constellation)))
            n_err = int(round(b * Nt * bits_per_sym))
            bit_errors += n_err
            total_bits += Nt * bits_per_sym
        except Exception as e:
            bers.append(1.0)
            total_bits += Nt * int(np.log2(len(constellation)))
            bit_errors += Nt * int(np.log2(len(constellation)))
        
        if (i + 1) % 100 == 0:
            avg = np.mean(bers)
            print(f"  {i+1}/{n_samples}  avg BER={avg:.5f}  bit_errors={bit_errors}")
    
    avg_ber = np.mean(bers)
    return avg_ber, bit_errors, total_bits

if __name__ == '__main__':
    print("=== Framework Ceiling Test: F_belief = cum_dist only ===")
    print("  max_nodes=1000, no BP (H_halt=True immediately)")
    print()
    
    for snr in [20.0, 22.0, 24.0]:
        n = {20.0: 500, 22.0: 1000, 24.0: 2000}[snr]
        print(f"SNR={snr} dB  ({n} samples)...")
        avg_ber, bit_err, tot_bits = evaluate_ceiling(snr, n_samples=n)
        print(f"  => avg BER={avg_ber:.5f}  bit_errors={bit_err}  total_bits={tot_bits}")
        print()
