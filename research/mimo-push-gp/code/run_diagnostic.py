"""
Diagnostic: Compare evolved BP (v7c) vs pure MMSE-LB stack decoder.

Tests whether BP actually contributes beyond the MMSE-LB bound.
Runs at multiple node limits and two SNR points.

Usage:
    conda run -n AutoGenOld python -u run_diagnostic.py
"""
import os, sys, json, time
import numpy as np

# Ensure imports work
sys.path.insert(0, os.path.dirname(__file__))
from vm import Instruction
from cpp_bridge import CppBPEvaluator

def _make_instr(item):
    """Build Instruction, handling ForEachChild with body."""
    if isinstance(item, dict):
        body = [_make_instr(b) for b in item.get('body', [])]
        return Instruction(name=item['name'], code_block=body)
    return Instruction(name=item)

def generate_mimo_sample(Nr, Nt, constellation, snr_db, rng):
    H = (rng.randn(Nr, Nt) + 1j * rng.randn(Nr, Nt)) / np.sqrt(2)
    x_idx = rng.randint(0, len(constellation), size=Nt)
    x = constellation[x_idx]
    sig_power = float(np.mean(np.abs(H @ x) ** 2))
    noise_var = sig_power / (10 ** (snr_db / 10.0))
    noise = np.sqrt(noise_var / 2.0) * (rng.randn(Nr) + 1j * rng.randn(Nr))
    y = H @ x + noise
    return H, x, y, noise_var

def load_genome_from_json(path):
    with open(path) as f:
        data = json.load(f)

    class G:
        pass

    g = G()
    g.prog_down = [_make_instr(n) for n in data['prog_down']]
    g.prog_up = [_make_instr(n) for n in data['prog_up']]
    g.prog_belief = [_make_instr(n) for n in data['prog_belief']]
    g.prog_halt = [_make_instr(n) for n in data['prog_halt']]
    g.log_constants = np.array(data.get('log_constants', [0]*4))
    g.evo_constants = np.clip(np.power(10.0, g.log_constants), 1e-6, 1e6)
    return g

def main():
    Nt, Nr = 16, 16
    mod_order = 16
    n_samples = 5000
    snr_list = [22, 24]
    node_limits = [50, 100, 200, 300, 500, 750, 1000, 1500, 2000]
    max_bp_iters = 5
    step_max = 500
    flops_max = 5_000_000

    # Constellation
    levels = np.array([-3, -1, 1, 3]) / np.sqrt(10)
    constellation = np.array([r + 1j*i for r in levels for i in levels])

    # Load v7c genome
    genome_path = os.path.join(os.path.dirname(__file__), 'seed_v7c_best.json')
    genome = load_genome_from_json(genome_path)
    print(f"Loaded genome from {genome_path}")
    print(f"  prog_down: {[i.name for i in genome.prog_down]}")
    print(f"  evo_constants: {genome.evo_constants}")

    # Create evaluator (use large max_nodes for multi-node sweep)
    dll_path = os.path.join(os.path.dirname(__file__), 'cpp', 'evaluator_bp.dll')
    ev = CppBPEvaluator(Nt=Nt, Nr=Nr, mod_order=mod_order,
                        max_nodes=max(node_limits),
                        flops_max=flops_max, step_max=step_max,
                        max_bp_iters=max_bp_iters, dll_path=dll_path)

    rng = np.random.RandomState(12345)

    for snr_db in snr_list:
        print(f"\n{'='*70}")
        print(f"  SNR = {snr_db} dB  |  {n_samples} samples  |  {Nt}x{Nr} 16QAM")
        print(f"{'='*70}")

        # Generate fresh dataset
        dataset = [generate_mimo_sample(Nr, Nt, constellation, snr_db, rng)
                   for _ in range(n_samples)]

        # Baselines
        t0 = time.time()
        ber_lmmse, ber_kb16, ber_kb32 = ev.evaluate_baselines(dataset)
        t_base = time.time() - t0
        print(f"\n  Baselines ({t_base:.1f}s):")
        print(f"    LMMSE:    BER = {ber_lmmse:.6f}")
        print(f"    K-Best16: BER = {ber_kb16:.6f}")
        print(f"    K-Best32: BER = {ber_kb32:.6f}")

        # Evolved BP at multiple node limits
        print(f"\n  Evolved BP (v7c) at varying node limits:")
        t0 = time.time()
        bers_bp = ev.evaluate_genome_multi_nodes(genome, dataset, node_limits)
        t_bp = time.time() - t0
        for i, nl in enumerate(node_limits):
            tag = " ***" if bers_bp[i] < ber_kb16 else ""
            print(f"    nodes={nl:5d}:  BER = {bers_bp[i]:.6f}{tag}")
        print(f"    (elapsed: {t_bp:.1f}s)")

        # MMSE-LB stack decoder at same node limits
        print(f"\n  MMSE-LB Stack Decoder (no BP) at varying node limits:")
        t0 = time.time()
        bers_mmselb = ev.evaluate_mmselb_multi_nodes(dataset, node_limits)
        t_mmselb = time.time() - t0
        for i, nl in enumerate(node_limits):
            tag = " ***" if bers_mmselb[i] < ber_kb16 else ""
            print(f"    nodes={nl:5d}:  BER = {bers_mmselb[i]:.6f}{tag}")
        print(f"    (elapsed: {t_mmselb:.1f}s)")

        # Comparison
        print(f"\n  BP vs MMSE-LB difference:")
        for i, nl in enumerate(node_limits):
            diff = bers_bp[i] - bers_mmselb[i]
            pct = (diff / max(bers_mmselb[i], 1e-10)) * 100
            better = "BP better" if diff < 0 else "MMSE-LB better" if diff > 0 else "equal"
            print(f"    nodes={nl:5d}:  diff = {diff:+.6f} ({pct:+.1f}%)  [{better}]")

    print(f"\n{'='*70}")
    print("CONCLUSION: If evolved BP ≈ MMSE-LB at all node counts,")
    print("            then BP adds NO value — it's just an MMSE-LB stack decoder.")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
