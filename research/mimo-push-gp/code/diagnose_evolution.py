"""
Diagnostic script: trace the execution of evolved BP programs to understand
why they underperform K-Best.

Key questions this script answers:
1. Does F_up actually aggregate children's messages? Or is it dead code?
2. Does F_belief actually use m_up/m_down? Or does it degenerate to cum_dist?
3. Does H_halt ever allow multiple BP iterations?
4. Are BP messages changing across iterations?
5. What fraction of the program is dead code?
"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from vm import MIMOPushVM, Instruction, program_to_string
from bp_decoder_v2 import StructuredBPDecoder, qam16_constellation
from stack_decoder import lmmse_detect, kbest_detect
from stacks import TreeNode, SearchTreeGraph
from bp_main_v2 import (
    Genome, generate_mimo_sample, ber_calc,
    program_to_formula_trace, print_genome_formulas,
    _format_pseudocode_block, random_genome,
    has_valid_bp_dependency, N_EVO_CONSTS,
)


def trace_single_detection(genome, H, y, noise_var, Nt, Nr, max_nodes=500):
    """Run one detection with FULL tracing of BP messages."""
    constellation = qam16_constellation()
    vm = MIMOPushVM(flops_max=5_000_000, step_max=2000)
    vm.evolved_constants = genome.evo_constants
    
    decoder = StructuredBPDecoder(
        Nt=Nt, Nr=Nr, constellation=constellation,
        max_nodes=max_nodes, vm=vm)
    
    # Monkey-patch to trace BP messages
    original_up = decoder._run_f_up
    original_down = decoder._run_down_pass
    original_belief = decoder._run_belief
    original_halt = decoder._run_halt
    
    trace_log = {
        'up_calls': [],
        'down_calls': [],
        'belief_calls': [],
        'halt_calls': [],
        'expansion_order': [],
    }
    
    def traced_up(node, prog_up):
        old_m_up = node.m_up
        flops = original_up(node, prog_up)
        new_m_up = node.m_up
        trace_log['up_calls'].append({
            'node_id': node.node_id,
            'layer': node.layer,
            'n_children': len(node.children),
            'old_m_up': old_m_up,
            'new_m_up': new_m_up,
            'changed': abs(new_m_up - old_m_up) > 1e-12,
            'children_m_up': [c.m_up for c in node.children[:5]],
            'children_local_dist': [c.local_dist for c in node.children[:5]],
        })
        return flops
    
    def traced_down(child, prog_down):
        old_m_down = child.m_down
        flops = original_down(child, prog_down)
        new_m_down = child.m_down
        trace_log['down_calls'].append({
            'node_id': child.node_id,
            'layer': child.layer,
            'parent_m_down': child.parent.m_down if child.parent else None,
            'local_dist': child.local_dist,
            'old_m_down': old_m_down,
            'new_m_down': new_m_down,
            'changed': abs(new_m_down - old_m_down) > 1e-12,
        })
        return flops
    
    def traced_belief(node, prog_belief):
        old_score = node.score
        flops = original_belief(node, prog_belief)
        new_score = node.score
        trace_log['belief_calls'].append({
            'node_id': node.node_id,
            'layer': node.layer,
            'cum_dist': node.cum_dist,
            'm_down': node.m_down,
            'm_up': node.m_up,
            'old_score': old_score,
            'new_score': new_score,
            'score_vs_cumdist': new_score - node.cum_dist if np.isfinite(new_score) else None,
        })
        return flops
    
    def traced_halt(node, old_m_up, prog_halt):
        result = original_halt(node, old_m_up, prog_halt)
        trace_log['halt_calls'].append({
            'old_m_up': old_m_up,
            'new_m_up': node.m_up,
            'result': result,
        })
        return result
    
    decoder._run_f_up = traced_up
    decoder._run_down_pass = traced_down
    decoder._run_belief = traced_belief
    decoder._run_halt = traced_halt
    
    x_hat, flops = decoder.detect(
        H, y,
        prog_down=genome.prog_down,
        prog_up=genome.prog_up,
        prog_belief=genome.prog_belief,
        prog_halt=genome.prog_halt,
        noise_var=noise_var)
    
    return x_hat, flops, trace_log


def analyze_trace(trace_log):
    """Analyze traced BP execution to identify issues."""
    print("\n" + "="*70)
    print("  TRACE ANALYSIS")
    print("="*70)
    
    # 1. F_up analysis
    up_calls = trace_log['up_calls']
    if up_calls:
        n_changed = sum(1 for c in up_calls if c['changed'])
        print(f"\n  F_up: {len(up_calls)} calls, {n_changed} changed m_up ({100*n_changed/len(up_calls):.1f}%)")
        # Check if m_up values are diverse
        m_ups = [c['new_m_up'] for c in up_calls]
        print(f"    m_up range: [{min(m_ups):.4f}, {max(m_ups):.4f}], std={np.std(m_ups):.4f}")
        # Show first few
        for c in up_calls[:3]:
            print(f"    node {c['node_id']} (layer {c['layer']}): "
                  f"m_up {c['old_m_up']:.4f} -> {c['new_m_up']:.4f}, "
                  f"children_m_up={[f'{v:.3f}' for v in c['children_m_up']]}")
    
    # 2. F_down analysis
    down_calls = trace_log['down_calls']
    if down_calls:
        n_changed = sum(1 for c in down_calls if c['changed'])
        print(f"\n  F_down: {len(down_calls)} calls, {n_changed} changed m_down ({100*n_changed/len(down_calls):.1f}%)")
        m_downs = [c['new_m_down'] for c in down_calls]
        print(f"    m_down range: [{min(m_downs):.4f}, {max(m_downs):.4f}], std={np.std(m_downs):.4f}")
        # Check if m_down correlates with local_dist
        lds = [c['local_dist'] for c in down_calls]
        if len(set(m_downs)) > 1 and len(set(lds)) > 1:
            corr = np.corrcoef(m_downs, lds)[0,1]
            print(f"    correlation(m_down, local_dist) = {corr:.4f}")
    
    # 3. F_belief analysis
    belief_calls = trace_log['belief_calls']
    if belief_calls:
        scores = [c['new_score'] for c in belief_calls if np.isfinite(c['new_score'])]
        cum_dists = [c['cum_dist'] for c in belief_calls if np.isfinite(c['new_score'])]
        diffs = [c['score_vs_cumdist'] for c in belief_calls if c['score_vs_cumdist'] is not None]
        print(f"\n  F_belief: {len(belief_calls)} calls")
        if scores:
            print(f"    score range: [{min(scores):.4f}, {max(scores):.4f}]")
            print(f"    cum_dist range: [{min(cum_dists):.4f}, {max(cum_dists):.4f}]")
        if diffs:
            print(f"    score - cum_dist: mean={np.mean(diffs):.4f}, std={np.std(diffs):.4f}")
            # Check if score == cum_dist (degenerate)
            n_equals = sum(1 for d in diffs if abs(d) < 1e-8)
            print(f"    score == cum_dist: {n_equals}/{len(diffs)} ({100*n_equals/len(diffs):.1f}%)")
        if scores and cum_dists and len(set(scores)) > 1:
            corr = np.corrcoef(scores, cum_dists)[0,1]
            print(f"    correlation(score, cum_dist) = {corr:.4f}")
            # Check if score uses m_up/m_down
            m_ups_b = [c['m_up'] for c in belief_calls if np.isfinite(c['new_score'])]
            m_downs_b = [c['m_down'] for c in belief_calls if np.isfinite(c['new_score'])]
            if len(set(m_ups_b)) > 1:
                corr_mu = np.corrcoef(scores, m_ups_b)[0,1]
                print(f"    correlation(score, m_up) = {corr_mu:.4f}")
            if len(set(m_downs_b)) > 1:
                corr_md = np.corrcoef(scores, m_downs_b)[0,1]
                print(f"    correlation(score, m_down) = {corr_md:.4f}")
    
    # 4. H_halt analysis
    halt_calls = trace_log['halt_calls']
    if halt_calls:
        n_continue = sum(1 for c in halt_calls if not c['result'])
        print(f"\n  H_halt: {len(halt_calls)} calls, {n_continue} continue ({100*n_continue/len(halt_calls):.1f}%)")
        for c in halt_calls[:5]:
            print(f"    old_m_up={c['old_m_up']:.4f}, new_m_up={c['new_m_up']:.4f} -> {'HALT' if c['result'] else 'CONTINUE'}")
    
    print()


def diagnose_genome(genome, label="", n_samples=5, Nt=16, Nr=16, snr_db=22.0):
    """Full diagnosis of a single genome."""
    print(f"\n{'#'*70}")
    print(f"  DIAGNOSING: {label}")
    print(f"{'#'*70}")
    
    # Print formulas
    print_genome_formulas(genome, label)
    
    # Print pseudocode for each program  
    print("\n  === Pseudocode ===")
    for pt_label, pt_type, pt_prog in [
        ('F_down', 'down', genome.prog_down),
        ('F_up', 'up', genome.prog_up),
        ('F_belief', 'belief', genome.prog_belief),
        ('H_halt', 'halt', genome.prog_halt),
    ]:
        plines = _format_pseudocode_block(pt_prog, pt_type, genome, base_indent=0)
        print(f"\n  [{pt_label}] ({len(pt_prog)} instrs):")
        for pl in plines:
            print(f"    {pl}")
    
    # Print raw instructions
    print("\n  === Raw Instructions ===")
    for pt_label, pt_prog in [
        ('F_down', genome.prog_down),
        ('F_up', genome.prog_up),
        ('F_belief', genome.prog_belief),
        ('H_halt', genome.prog_halt),
    ]:
        print(f"  {pt_label}: {' ; '.join(str(i) for i in pt_prog)}")
    
    constellation = qam16_constellation()
    rng = np.random.RandomState(42)
    
    total_errors = 0
    total_bits = 0
    kb16_total_errors = 0
    kb32_total_errors = 0
    
    for s in range(n_samples):
        H, x_true, y, nv = generate_mimo_sample(Nr, Nt, constellation, snr_db, rng)
        
        x_hat, flops, trace_log = trace_single_detection(
            genome, H, y, nv, Nt, Nr, max_nodes=1000)
        
        errors = int(np.sum(x_true != x_hat))
        total_errors += errors
        total_bits += Nt
        
        xk16, _ = kbest_detect(H, y, constellation, K=16)
        kb16_total_errors += int(np.sum(x_true != xk16))
        xk32, _ = kbest_detect(H, y, constellation, K=32)
        kb32_total_errors += int(np.sum(x_true != xk32))
        
        if s < 2:  # Only trace first 2 samples in detail
            print(f"\n  --- Sample {s+1} ---")
            print(f"  Errors: {errors}/{Nt}, FLOPs: {flops}")
            analyze_trace(trace_log)
    
    ber = total_errors / total_bits
    kb16_ber = kb16_total_errors / total_bits
    kb32_ber = kb32_total_errors / total_bits
    print(f"\n  Overall ({n_samples} samples, SNR={snr_db}dB):")
    print(f"    Evolved BER:  {ber:.5f} ({total_errors} errors)")
    print(f"    K-Best-16 BER: {kb16_ber:.5f} ({kb16_total_errors} errors)")
    print(f"    K-Best-32 BER: {kb32_ber:.5f} ({kb32_total_errors} errors)")
    print(f"    Ratio vs KB16: {ber/max(kb16_ber, 1e-6):.2f}x")


def diagnose_random_population(n=20, Nt=16, Nr=16, snr_db=22.0):
    """Diagnose a set of random genomes to understand initialization quality."""
    print(f"\n{'#'*70}")
    print(f"  RANDOM POPULATION ANALYSIS ({n} genomes)")
    print(f"{'#'*70}")
    
    rng = np.random.RandomState(42)
    constellation = qam16_constellation()
    
    H, x_true, y, nv = generate_mimo_sample(Nr, Nt, constellation, snr_db, rng)
    
    results = []
    for i in range(n):
        g = random_genome(rng)
        
        vm = MIMOPushVM(flops_max=3_000_000, step_max=2000)
        vm.evolved_constants = g.evo_constants
        decoder = StructuredBPDecoder(
            Nt=Nt, Nr=Nr, constellation=constellation,
            max_nodes=500, vm=vm)
        
        try:
            x_hat, flops = decoder.detect(
                H, y, g.prog_down, g.prog_up, g.prog_belief, g.prog_halt,
                noise_var=float(nv))
            errors = int(np.sum(x_true != x_hat))
            results.append({
                'idx': i, 'errors': errors, 'flops': flops,
                'belief_formula': program_to_formula_trace(g.prog_belief, 'belief', g),
                'up_formula': program_to_formula_trace(g.prog_up, 'up', g),
                'down_formula': program_to_formula_trace(g.prog_down, 'down', g),
                'halt_formula': program_to_formula_trace(g.prog_halt, 'halt', g),
            })
        except Exception as e:
            results.append({'idx': i, 'errors': Nt, 'flops': 0, 
                          'belief_formula': f'FAULT: {e}',
                          'up_formula': '', 'down_formula': '', 'halt_formula': ''})
    
    results.sort(key=lambda x: x['errors'])
    
    print(f"\n  Top 10 by errors (lower is better):")
    for r in results[:10]:
        print(f"    #{r['idx']:2d}  errors={r['errors']:2d}  flops={r['flops']:8.0f}")
        print(f"        belief={r['belief_formula']}")
        print(f"        up={r['up_formula']}")
    
    print(f"\n  Error distribution: mean={np.mean([r['errors'] for r in results]):.1f}, "
          f"min={min(r['errors'] for r in results)}, "
          f"max={max(r['errors'] for r in results)}")


if __name__ == '__main__':
    # Test with a simple known genome
    from bp_main_v2 import _DOWN_INSTR, _UP_INSTR, _BELIEF_INSTR, _HALT_INSTR
    
    rng = np.random.RandomState(42)
    
    print("Generating random genome for diagnosis...")
    g = random_genome(rng)
    diagnose_genome(g, "Random genome", n_samples=3, snr_db=22.0)
    
    # Also diagnose a few more
    print("\n\nGenerating more random genomes...")
    diagnose_random_population(n=10, snr_db=22.0)
