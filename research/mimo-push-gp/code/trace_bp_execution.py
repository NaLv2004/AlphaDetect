"""
Deep trace of BP+Stack Decoder execution.
Examines a genome's actual execution:
- What does each F_down/F_up/F_belief actually compute?
- Are BP messages actually changing scores vs pure cum_dist?
- How many BP iterations happen?
- What's the correlation between node.score and node.cum_dist?

Usage: python trace_bp_execution.py
"""
import sys, os, numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__))

from bp_main_v2 import (
    Genome, Instruction, generate_mimo_sample, N_EVO_CONSTS,
    random_genome, has_valid_bp_dependency, print_genome_formulas,
)
from bp_decoder_v2 import StructuredBPDecoder, qam16_constellation, TreeNode
from vm import MIMOPushVM as PushVM
from cpp_bridge import CppBPEvaluator


def trace_detect(decoder, H, y, genome, noise_var, verbose=True):
    """Run detection step-by-step, tracing BP message updates."""
    Nr, Nt = H.shape
    Q, R = np.linalg.qr(H, mode='reduced')
    y_tilde = Q.conj().T @ y
    decoder._R = R
    decoder._y_tilde = y_tilde
    decoder._noise_var = noise_var
    decoder.vm.evolved_constants = genome.evo_constants

    from bp_decoder_v2 import SearchTreeGraph
    decoder.search_tree = SearchTreeGraph()
    root = decoder.search_tree.create_root(layer=Nt)
    root.m_up = 0.0
    root.m_down = 0.0
    root.score = 0.0

    # Expand root
    k0 = Nt - 1
    for sym in decoder.constellation:
        residual = y_tilde[k0] - R[k0, k0] * sym
        ld = float(np.abs(residual) ** 2)
        decoder.search_tree.add_child(
            parent=root, layer=k0, symbol=sym,
            local_dist=ld, cum_dist=ld,
            partial_symbols=np.array([sym]),
        )

    stats = {
        'n_expansions': 0,
        'bp_cycles': 0,
        'score_vs_cumdist': [],  # (score, cum_dist) pairs
        'score_changes': [],     # abs change per BP cycle
        'mup_values': [],
        'mdown_values': [],
        'halt_results': [],
        'expansion_layers': [],
    }

    import heapq
    pq = []
    counter = 0

    # Track how BP changes scores
    def snapshot_frontier():
        """Get {node_id: (score, cum_dist, m_up, m_down)} for frontier."""
        result = {}
        for i, nd in enumerate(decoder.search_tree.nodes):
            if not nd.is_expanded and nd is not root:
                result[id(nd)] = (nd.score, nd.cum_dist, nd.m_up, nd.m_down)
        return result

    # Initial BP cycle
    pre_bp = snapshot_frontier()
    decoder._full_bp_cycle(genome.prog_down, genome.prog_up,
                           genome.prog_belief, genome.prog_halt)
    post_bp = snapshot_frontier()
    stats['bp_cycles'] += 1

    # Analyze BP effect
    score_deltas = []
    for nid in post_bp:
        if nid in pre_bp:
            delta = abs(post_bp[nid][0] - pre_bp[nid][0])
            score_deltas.append(delta)
            stats['score_vs_cumdist'].append((post_bp[nid][0], post_bp[nid][1]))
            stats['mup_values'].append(post_bp[nid][2])
            stats['mdown_values'].append(post_bp[nid][3])

    if verbose and score_deltas:
        print(f"  Initial BP: avg_score_delta={np.mean(score_deltas):.6f}, "
              f"max={np.max(score_deltas):.6f}")
        fronts = list(post_bp.values())
        scores = [f[0] for f in fronts]
        cds = [f[1] for f in fronts]
        if np.std(scores) > 0 and np.std(cds) > 0:
            corr = np.corrcoef(scores, cds)[0, 1]
            print(f"  Score-CumDist correlation: {corr:.4f}")
        print(f"  Score range: [{min(scores):.4f}, {max(scores):.4f}]")
        print(f"  CumDist range: [{min(cds):.4f}, {max(cds):.4f}]")
        mups = [f[2] for f in fronts]
        mdowns = [f[3] for f in fronts]
        print(f"  M_up range: [{min(mups):.6f}, {max(mups):.6f}]")
        print(f"  M_down range: [{min(mdowns):.6f}, {max(mdowns):.6f}]")

    # Build PQ
    for nd in decoder.search_tree.nodes:
        if not nd.is_expanded and nd is not root:
            heapq.heappush(pq, (nd.score, nd.queue_version, counter, nd))
            counter += 1

    n_complete = 0
    max_expansions = 20  # trace first 20 expansions

    while pq and len(decoder.search_tree.nodes) < decoder.max_nodes + 1:
        sc, ver, _, node = heapq.heappop(pq)
        if ver != node.queue_version:
            continue

        if node.layer == 0:
            n_complete += 1
            if verbose and n_complete <= 3:
                print(f"  Found complete solution #{n_complete}: "
                      f"score={sc:.4f}, cum_dist={node.cum_dist:.4f}")
            if n_complete == 1:
                return node.partial_symbols[::-1], stats

        stats['n_expansions'] += 1
        stats['expansion_layers'].append(node.layer)

        decoder.search_tree.mark_expanded(node)
        next_layer = node.layer - 1

        for sym in decoder.constellation:
            new_partial = np.append(node.partial_symbols, sym)
            n_dec = len(new_partial)
            interference = 0.0 + 0.0j
            for j in range(n_dec):
                col = Nt - 1 - j
                interference += R[next_layer, col] * new_partial[j]
            residual = y_tilde[next_layer] - interference
            ld = float(np.abs(residual) ** 2)
            cd = node.cum_dist + ld
            decoder.search_tree.add_child(
                parent=node, layer=next_layer, symbol=sym,
                local_dist=ld, cum_dist=cd,
                partial_symbols=new_partial)

        # BP cycle
        pre_bp = snapshot_frontier()
        decoder._full_bp_cycle(genome.prog_down, genome.prog_up,
                               genome.prog_belief, genome.prog_halt)
        post_bp = snapshot_frontier()
        stats['bp_cycles'] += 1

        # Analyze change
        n_changed = 0
        deltas = []
        for nid in post_bp:
            if nid in pre_bp:
                delta = abs(post_bp[nid][0] - pre_bp[nid][0])
                if delta > 1e-12:
                    n_changed += 1
                deltas.append(delta)

        if verbose and stats['n_expansions'] <= max_expansions:
            n_frontier = len(post_bp)
            print(f"  Exp#{stats['n_expansions']:3d} layer={node.layer:2d} "
                  f"nodes={len(decoder.search_tree.nodes):4d} "
                  f"frontier={n_frontier:4d} "
                  f"score_changed={n_changed}/{n_frontier} "
                  f"avg_delta={np.mean(deltas):.6f}")

        # Rebuild PQ
        pq = []
        counter = 0
        for nd in decoder.search_tree.nodes:
            if not nd.is_expanded and nd is not root:
                heapq.heappush(pq, (nd.score, nd.queue_version, counter, nd))
                counter += 1

    # Fallback
    best_node = None
    best_score = float('inf')
    for nd in decoder.search_tree.nodes:
        if not nd.is_expanded and nd is not root:
            if nd.score < best_score:
                best_score = nd.score
                best_node = nd
    if best_node:
        x_hat, _ = decoder._complete_path(best_node, R, y_tilde)
    else:
        x_hat = np.zeros(Nt, dtype=complex)

    if verbose:
        print(f"  Total: {stats['n_expansions']} expansions, "
              f"{stats['bp_cycles']} BP cycles, "
              f"{len(decoder.search_tree.nodes)} nodes")
        layers = stats['expansion_layers']
        if layers:
            from collections import Counter
            lc = Counter(layers)
            print(f"  Expansion layer distribution: "
                  f"{dict(sorted(lc.items()))}")
    return x_hat, stats


def main():
    constellation = qam16_constellation()
    Nt, Nr = 16, 16

    # First, generate a random valid genome
    rng = np.random.RandomState(42)
    print("Generating random genome...")
    genome = random_genome(rng)
    print_genome_formulas(genome, "Random Genome")

    # Check C++ vs Python on a few samples
    print("\n--- Testing genome on single sample ---")
    rng2 = np.random.RandomState(100)
    H, x_true, y, nv = generate_mimo_sample(Nr, Nt, constellation, 22.0, rng2)

    vm = PushVM(flops_max=3_000_000, step_max=2000)
    vm.evolved_constants = genome.evo_constants
    decoder = StructuredBPDecoder(
        Nt, Nr, constellation, max_nodes=1000, vm=vm, max_bp_iters=2)

    x_hat, stats = trace_detect(decoder, H, y, genome, float(nv), verbose=True)

    ber = float(np.mean(x_true != x_hat))
    print(f"\n  BER for this sample: {ber:.4f}")
    print(f"  x_true[:4]: {x_true[:4]}")
    print(f"  x_hat[:4]:  {x_hat[:4]}")

    # Now evaluate on more samples with C++
    print("\n--- C++ evaluation (200 samples) ---")
    ds = [generate_mimo_sample(Nr, Nt, constellation, 22.0,
                               np.random.RandomState(i)) for i in range(200)]
    cpp = CppBPEvaluator(Nt=16, Nr=16, mod_order=16, max_nodes=1000,
                          flops_max=3_000_000, step_max=2000, max_bp_iters=2)
    ber, fl, fa, bp = cpp.evaluate_genome(genome, ds)
    print(f"  C++ BER={ber:.5f}, flops={fl:.0f}, faults={fa}, bp_calls={bp:.0f}")

    # Compare with pure cum_dist baseline
    cum_genome = Genome(
        prog_down=[], prog_up=[],
        prog_belief=[Instruction(name='Node.GetCumDist')],
        prog_halt=[Instruction(name='Bool.True')],
        log_constants=np.zeros(N_EVO_CONSTS))
    ber_cd, fl_cd, fa_cd, bp_cd = cpp.evaluate_genome(cum_genome, ds)
    print(f"  Pure cum_dist BER={ber_cd:.5f}")
    print(f"  Improvement over cum_dist: {ber_cd - ber:.5f}")


if __name__ == '__main__':
    main()
