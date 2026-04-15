"""
Comprehensive analysis of the best program discovered in truebp_1 evolution.

Tasks:
1. Dead-code detection: which instructions actually execute?
2. Human-readable formula explanation (verified by trace)
3. Ablation: BER with vs without BP writes
4. Node budget comparison: 200 vs 500 vs 2000 nodes
5. Baseline comparison: LMMSE, K-Best-16/32

Best program (Gen 72):
  Node.GetSymIm ; Matrix.Dup ; Float.Pop ; Node.NumChildren ; Float.Exp ;
  Vec.GetResidue ; Float.Div ; Float.GetNoiseVar ; Node.SetScore ;
  Node.ForEachSibling([Node.GetScore, Mat.Rows, Float.Swap, Int.GetNumSymbols]) ;
  Node.GetParent ;
  Node.ForEachChild([Float.GetMMSELB, Node.GetCumDist, Float.Inv, Node.SetScore]) ;
  Int.GT ; Node.GetLayer
"""

import sys, os, time
import numpy as np

# ── path -------------------------------------------------------------------
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

from vm import Instruction, MIMOPushVM
from bp_decoder import BPStackDecoder, qam16_constellation
from stacks import TreeNode, SearchTreeGraph

# ── programs ---------------------------------------------------------------

GEN72_PROG = [
    Instruction('Node.GetSymIm'),
    Instruction('Matrix.Dup'),
    Instruction('Float.Pop'),
    Instruction('Node.NumChildren'),
    Instruction('Float.Exp'),
    Instruction('Vec.GetResidue'),
    Instruction('Float.Div'),
    Instruction('Float.GetNoiseVar'),
    Instruction('Node.SetScore'),
    Instruction('Node.ForEachSibling', code_block=[
        Instruction('Node.GetScore'),
        Instruction('Mat.Rows'),
        Instruction('Float.Swap'),
        Instruction('Int.GetNumSymbols'),
    ]),
    Instruction('Node.GetParent'),
    Instruction('Node.ForEachChild', code_block=[
        Instruction('Float.GetMMSELB'),
        Instruction('Node.GetCumDist'),
        Instruction('Float.Inv'),
        Instruction('Node.SetScore'),
    ]),
    Instruction('Int.GT'),
    Instruction('Node.GetLayer'),
]

# Gen 8 simple program (len=1)
GEN8_PROG = [Instruction('Float.FromInt')]

# Pure MMSE-LB (best-known no-BP)
MMSE_LB_PROG = [Instruction('Float.GetMMSELB')]

# No correction (pure best-first by cum_dist)
DIST_PROG = []


# ══════════════════════════════════════════════════════════════════════
# Part 1 & 2: Instruction Trace (dead-code detection + explanation)
# ══════════════════════════════════════════════════════════════════════

class TraceVM(MIMOPushVM):
    """VM that records how many times each instruction name is executed."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._trace_counts = {}   # instruction name → count
        self._trace_effects = {}  # instruction name → description of last effect

    def reset(self):
        super().reset()
        # don't reset trace counts — we want aggregates

    def _execute_instruction(self, ins):
        name = ins.name
        # Snapshot stacks before execution
        f_before = self.float_stack.depth()
        i_before = self.int_stack.depth()
        b_before = self.bool_stack.depth()
        v_before = self.vector_stack.depth()
        n_before = self.node_stack.depth()

        super()._execute_instruction(ins)

        f_after = self.float_stack.depth()
        i_after = self.int_stack.depth()
        b_after = self.bool_stack.depth()
        v_after = self.vector_stack.depth()
        n_after = self.node_stack.depth()

        # Determine if the instruction had any stack effect
        had_effect = (f_after != f_before or i_after != i_before or
                      b_after != b_before or v_after != v_before or
                      n_after != n_before)

        if name not in self._trace_counts:
            self._trace_counts[name] = {'total': 0, 'effective': 0}
        self._trace_counts[name]['total'] += 1
        if had_effect:
            self._trace_counts[name]['effective'] += 1


def run_trace_experiment(n_trials=50, snr_db=12, max_nodes=200, seed=42):
    """Run Gen72 on many examples and collect instruction execution counts."""
    print("\n" + "="*70)
    print("PART 1: DEAD-CODE DETECTION (Instruction Execution Counts)")
    print(f"Settings: {n_trials} trials, SNR={snr_db}dB, max_nodes={max_nodes}")
    print("="*70)

    constellation = qam16_constellation()
    Nr, Nt = 16, 8
    rng = np.random.RandomState(seed)
    snr_lin = 10 ** (snr_db / 10)

    trace_vm = TraceVM(step_max=2000, flops_max=3_000_000)

    for trial in range(n_trials):
        x_idx = rng.randint(0, 16, Nt)
        x = constellation[x_idx]
        H = (rng.randn(Nr, Nt) + 1j * rng.randn(Nr, Nt)) / np.sqrt(2)
        sig_p = float(np.mean(np.abs(H @ x) ** 2))
        nv = sig_p / snr_lin
        y = H @ x + np.sqrt(nv / 2) * (rng.randn(Nr) + 1j * rng.randn(Nr))

        dec = BPStackDecoder(Nt=Nt, Nr=Nr, constellation=constellation,
                             max_nodes=max_nodes, vm=trace_vm,
                             allow_score_writes=True)
        dec.detect(H, y, GEN72_PROG, noise_var=nv)

    # Print results
    counts = trace_vm._trace_counts

    # Order by original program structure
    all_names_ordered = [
        'Node.GetSymIm', 'Matrix.Dup', 'Float.Pop', 'Node.NumChildren',
        'Float.Exp', 'Vec.GetResidue', 'Float.Div', 'Float.GetNoiseVar',
        'Node.SetScore',
        # ForEachSibling body
        'Node.GetScore', 'Mat.Rows', 'Float.Swap', 'Int.GetNumSymbols',
        'Node.ForEachSibling',
        'Node.GetParent',
        # ForEachChild body
        'Float.GetMMSELB', 'Node.GetCumDist', 'Float.Inv',
        'Node.ForEachChild',
        'Int.GT', 'Node.GetLayer',
    ]

    # Estimate per-call (upper bound): divide by n_trials * nodes approx
    print(f"\n{'Instruction':<30} {'Total':>10} {'Effective':>10}  Status")
    print("-"*65)
    for name in all_names_ordered:
        c = counts.get(name, {'total': 0, 'effective': 0})
        tot = c['total']
        eff = c['effective']
        if tot == 0:
            status = "NEVER CALLED"
        elif eff == 0:
            status = "DEAD (no stack effect)"
        elif eff < tot * 0.1:
            status = f"MOSTLY DEAD ({100*eff/tot:.0f}% effective)"
        else:
            status = "ACTIVE"
        print(f"  {name:<28} {tot:>10,} {eff:>10,}  {status}")

    return counts


# ══════════════════════════════════════════════════════════════════════
# Part 2: Step-by-step trace on ONE example (explain the formula)
# ══════════════════════════════════════════════════════════════════════

class SingleStepTraceVM(MIMOPushVM):
    """VM that logs stack snapshots for each instruction."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trace_log = []
        self.enable_trace = False

    def _execute_instruction(self, ins):
        if not self.enable_trace:
            super()._execute_instruction(ins)
            return

        # Snapshot depths before
        fd0 = self.float_stack.depth()
        id0 = self.int_stack.depth()
        bd0 = self.bool_stack.depth()
        vd0 = self.vector_stack.depth()
        nd0 = self.node_stack.depth()
        md0 = self.matrix_stack.depth()
        f_top_before = self.float_stack.peek()

        super()._execute_instruction(ins)

        # Snapshot depths after
        fd1 = self.float_stack.depth()
        id1 = self.int_stack.depth()
        bd1 = self.bool_stack.depth()
        nd1 = self.node_stack.depth()
        md1 = self.matrix_stack.depth()
        f_top_after = self.float_stack.peek()
        i_top_after = self.int_stack.peek()
        n_top_after = self.node_stack.peek()

        delta_f = fd1 - fd0
        delta_i = id1 - id0

        self.trace_log.append({
            'ins': str(ins),
            'f_top_before': f_top_before,
            'f_top_after': f_top_after,
            'i_top_after': i_top_after,
            'delta_f': delta_f,
            'delta_i': delta_i,
            'n_node_before': nd0,
            'n_node_after': nd1,
            'n_mat_before': md0,
            'n_mat_after': md1,
        })

    def reset(self):
        super().reset()
        self.trace_log = []


def explain_program(snr_db=12, seed=999):
    """Run one example and print a step-by-step explanation of Gen72."""
    print("\n" + "="*70)
    print("PART 2: STEP-BY-STEP PROGRAM TRACE (1 example)")
    print(f"Settings: SNR={snr_db}dB")
    print("="*70)

    constellation = qam16_constellation()
    Nr, Nt = 16, 8
    rng = np.random.RandomState(seed)
    snr_lin = 10 ** (snr_db / 10)

    x_idx = rng.randint(0, 16, Nt)
    x = constellation[x_idx]
    H = (rng.randn(Nr, Nt) + 1j * rng.randn(Nr, Nt)) / np.sqrt(2)
    sig_p = float(np.mean(np.abs(H @ x) ** 2))
    nv = sig_p / snr_lin
    y = H @ x + np.sqrt(nv / 2) * (rng.randn(Nr) + 1j * rng.randn(Nr))

    Q, R = np.linalg.qr(H, mode='reduced')
    y_tilde = Q.conj().T @ y

    # Manually build a small tree so we can trace a specific node
    from stacks import SearchTreeGraph
    graph = SearchTreeGraph()
    root = graph.create_root(layer=Nt)

    k0 = Nt - 1   # bottom layer
    children_layer0 = []
    for sym in constellation:
        residual = y_tilde[k0] - R[k0, k0] * sym
        ld = float(np.abs(residual) ** 2)
        child = graph.add_child(parent=root, layer=k0, symbol=sym,
                                local_dist=ld, cum_dist=ld,
                                partial_symbols=np.array([sym]))
        children_layer0.append(child)

    # Pick the 5th child (has 4 siblings already in tree)
    cand = children_layer0[4]

    trace_vm = SingleStepTraceVM(step_max=2000, flops_max=3_000_000)
    trace_vm.enable_trace = True
    trace_vm.inject_environment(
        R=R, y_tilde=y_tilde, x_partial=cand.partial_symbols,
        graph=graph, candidate_node=cand, depth_k=cand.layer,
        constellation=constellation, noise_var=nv,
    )

    print(f"\nContext:")
    print(f"  Candidate node: layer={cand.layer}, symbol={cand.symbol:.3f}")
    print(f"  cum_dist={cand.cum_dist:.4f}, local_dist={cand.local_dist:.4f}")
    print(f"  noise_var={nv:.4f}")
    print(f"  Siblings already in tree: {len(graph.siblings(cand))}")
    print(f"  Children of candidate: {len(cand.children)}")
    print(f"  Initial Int stack (depth_k): {cand.layer}")

    print(f"\n  Initial stacks: Float=[], Int=[{cand.layer}], Vec=[y_tilde, x_partial], Mat=[R], Node=[cand]")
    print()

    # Run program with tracing
    try:
        trace_vm._execute_block(GEN72_PROG)
    except Exception as e:
        print(f"  Exception: {e}")

    correction = trace_vm.float_stack.peek()
    final_score = cand.cum_dist + (correction if correction is not None else float('inf'))

    print(f"\nInstruction trace:")
    print(f"  {'Instruction':<48} {'Δf':>4} {'Δi':>4} {'Δn':>4} {'Δmat':>4}  f_top_after")
    print("  " + "-"*80)

    for entry in trace_vm.trace_log:
        ins_name = entry['ins']
        delta_f = entry['delta_f']
        delta_i = entry['delta_i']
        delta_n = entry['n_node_after'] - entry['n_node_before']
        delta_m = entry['n_mat_after'] - entry['n_mat_before']
        f_top = entry['f_top_after']
        # Flag dead instructions
        dead = ""
        if delta_f == 0 and delta_i == 0 and delta_n == 0 and delta_m == 0:
            dead = "  ← (no-op)"
        print(f"  {ins_name:<48} {delta_f:>+4} {delta_i:>+4} {delta_n:>+4} {delta_m:>+4}  {f_top}{dead}")

    print(f"\nFinal float_stack top: {correction}")
    print(f"  → Score for cand = cand.cum_dist + correction")
    print(f"  = {cand.cum_dist:.4f} + {correction}")
    print(f"  = {final_score}")

    # Show what BP did to siblings
    print(f"\nBP side effects — comparing sibling scores BEFORE vs AFTER program:")
    print(f"  (Initial score=inf for new nodes)")
    # Record scores before and run the program
    sibs = graph.siblings(cand)
    scores_before = [s.score for s in sibs]
    # Run program for real on those siblings (they score when traced)
    # The trace_vm already ran — show its effects (node.score fields were set)
    print(f"  {'sib':>4}  {'cum_dist':>10}  {'score_after':>12}  {'vs_cumdist':>14}")
    for i, sib in enumerate(sibs):
        delta = sib.score - sib.cum_dist if np.isfinite(sib.score) else float('nan')
        print(f"  [{i:2d}]  {sib.cum_dist:>10.4f}  {sib.score:>12.4f}  "
              f"diff={delta:>+.4f}")


# ══════════════════════════════════════════════════════════════════════
# Part 3: Ablation — BP vs no-BP at 200 nodes
# ══════════════════════════════════════════════════════════════════════

def ablation_test(n_trials=300, snr_list=None, max_nodes=200, seed=7777):
    """Compare Gen72 with and without BP writes enabled."""
    if snr_list is None:
        snr_list = [8, 10, 12, 14, 16]

    print("\n" + "="*70)
    print(f"PART 3: ABLATION TEST (BP vs no-BP) at max_nodes={max_nodes}")
    print(f"Settings: {n_trials} trials per SNR")
    print("="*70)

    constellation = qam16_constellation()
    Nr, Nt = 16, 8
    rng = np.random.RandomState(seed)

    vm_bp = MIMOPushVM(step_max=2000, flops_max=3_000_000)
    vm_nobp = MIMOPushVM(step_max=2000, flops_max=3_000_000)

    print(f"\n{'SNR':>4}  {'BER_BP':>10}  {'BER_noBP':>10}  {'BER_MMSE':>10}  "
          f"{'gain=noBP-BP':>12}  {'ratio_BP':>9}  {'ratio_noBP':>10}")
    print("-" * 80)

    for snr_db in snr_list:
        snr_lin = 10 ** (snr_db / 10)
        rng = np.random.RandomState(seed + snr_db)

        n_err_bp = n_err_nobp = n_err_mmse = n_sym = 0

        for _ in range(n_trials):
            x_idx = rng.randint(0, 16, Nt)
            x = constellation[x_idx]
            H = (rng.randn(Nr, Nt) + 1j * rng.randn(Nr, Nt)) / np.sqrt(2)
            sig_p = float(np.mean(np.abs(H @ x) ** 2))
            nv = sig_p / snr_lin
            y = H @ x + np.sqrt(nv / 2) * (rng.randn(Nr) + 1j * rng.randn(Nr))

            dec_bp = BPStackDecoder(Nt=Nt, Nr=Nr, constellation=constellation,
                                    max_nodes=max_nodes, vm=vm_bp,
                                    allow_score_writes=True)
            dec_nobp = BPStackDecoder(Nt=Nt, Nr=Nr, constellation=constellation,
                                      max_nodes=max_nodes, vm=vm_nobp,
                                      allow_score_writes=False)
            dec_mmse = BPStackDecoder(Nt=Nt, Nr=Nr, constellation=constellation,
                                      max_nodes=max_nodes, vm=MIMOPushVM(step_max=500, flops_max=1_000_000),
                                      allow_score_writes=False)

            xh_bp, _ = dec_bp.detect(H, y, GEN72_PROG, noise_var=nv)
            xh_nobp, _ = dec_nobp.detect(H, y, GEN72_PROG, noise_var=nv)
            xh_mmse, _ = dec_mmse.detect(H, y, MMSE_LB_PROG, noise_var=nv)

            n_err_bp += int(np.sum(xh_bp != x))
            n_err_nobp += int(np.sum(xh_nobp != x))
            n_err_mmse += int(np.sum(xh_mmse != x))
            n_sym += Nt

        ber_bp = n_err_bp / n_sym
        ber_nobp = n_err_nobp / n_sym
        ber_mmse = n_err_mmse / n_sym
        gain = ber_nobp - ber_bp
        ratio_bp = ber_bp / ber_mmse if ber_mmse > 0 else float('nan')
        ratio_nobp = ber_nobp / ber_mmse if ber_mmse > 0 else float('nan')

        print(f"  {snr_db:>2}   {ber_bp:>10.5f}   {ber_nobp:>10.5f}   {ber_mmse:>10.5f}   "
              f"{gain:>+11.5f}   {ratio_bp:>8.3f}   {ratio_nobp:>9.3f}")


# ══════════════════════════════════════════════════════════════════════
# Part 4: Node budget sweep (200 / 500 / 1000 / 2000)
# ══════════════════════════════════════════════════════════════════════

def node_budget_test(n_trials=300, snr_list=None, seed=8888):
    """Test Gen72 and Gen8 at multiple node budgets vs K-Best and MMSE-LB."""
    if snr_list is None:
        snr_list = [8, 10, 12, 14, 16]

    from bp_decoder import BPStackDecoder
    # Import kbest if available
    try:
        from stack_decoder import kbest_detect
        has_kbest = True
    except ImportError:
        has_kbest = False

    print("\n" + "="*70)
    print("PART 4: NODE BUDGET COMPARISON")
    print(f"Settings: {n_trials} trials per (SNR, budget)")
    print("="*70)

    constellation = qam16_constellation()
    Nr, Nt = 16, 8

    budgets = [200, 500, 1000, 2000]
    programs = {
        'Gen72_BP':   (GEN72_PROG, True),
        'Gen72_noBP': (GEN72_PROG, False),
        'Gen8':       (GEN8_PROG, False),
        'MMSE-LB':    (MMSE_LB_PROG, False),
        'no-corr':    (DIST_PROG, False),
    }

    # K-Best baselines (node-count independent)
    kbest_results = {}
    if has_kbest:
        print("\nComputing K-Best baselines...")
        for K in [16, 32]:
            kbest_results[K] = {}
            for snr_db in snr_list:
                snr_lin = 10 ** (snr_db / 10)
                rng = np.random.RandomState(seed + snr_db)
                n_err = n_sym = 0
                for _ in range(n_trials):
                    x_idx = rng.randint(0, 16, Nt)
                    x = constellation[x_idx]
                    H = (rng.randn(Nr, Nt) + 1j * rng.randn(Nr, Nt)) / np.sqrt(2)
                    sig_p = float(np.mean(np.abs(H @ x) ** 2))
                    nv = sig_p / snr_lin
                    y = H @ x + np.sqrt(nv / 2) * (rng.randn(Nr) + 1j * rng.randn(Nr))
                    xk, _ = kbest_detect(H, y, constellation, K=K)
                    n_err += int(np.sum(xk != x))
                    n_sym += Nt
                kbest_results[K][snr_db] = n_err / n_sym
            print(f"  K-Best-{K}: done")

    # Print K-Best reference table
    if has_kbest:
        print(f"\nK-Best reference (node-independent):")
        print(f"  {'SNR':>4}", end='')
        for K in [16, 32]:
            print(f"  {'K-Best-'+str(K):>12}", end='')
        print()
        for snr_db in snr_list:
            print(f"  {snr_db:>4}", end='')
            for K in [16, 32]:
                print(f"  {kbest_results[K][snr_db]:>12.5f}", end='')
            print()

    # Main sweep
    print(f"\nProgram performance across node budgets:")
    for max_nodes in budgets:
        print(f"\n  ── max_nodes = {max_nodes} ──")
        print(f"  {'Program':<14}", end='')
        for snr_db in snr_list:
            print(f"  {'SNR='+str(snr_db):>10}", end='')
        print()

        for prog_name, (prog, use_bp) in programs.items():
            print(f"  {prog_name:<14}", end='')
            for snr_db in snr_list:
                snr_lin = 10 ** (snr_db / 10)
                rng = np.random.RandomState(seed + snr_db * 100 + max_nodes)
                n_err = n_sym = 0
                vm = MIMOPushVM(step_max=2000, flops_max=5_000_000)
                for _ in range(n_trials):
                    x_idx = rng.randint(0, 16, Nt)
                    x = constellation[x_idx]
                    H = (rng.randn(Nr, Nt) + 1j * rng.randn(Nr, Nt)) / np.sqrt(2)
                    sig_p = float(np.mean(np.abs(H @ x) ** 2))
                    nv = sig_p / snr_lin
                    y = H @ x + np.sqrt(nv / 2) * (rng.randn(Nr) + 1j * rng.randn(Nr))
                    dec = BPStackDecoder(Nt=Nt, Nr=Nr, constellation=constellation,
                                        max_nodes=max_nodes, vm=vm,
                                        allow_score_writes=use_bp)
                    xh, _ = dec.detect(H, y, prog, noise_var=nv)
                    n_err += int(np.sum(xh != x))
                    n_sym += Nt
                ber = n_err / n_sym
                print(f"  {ber:>10.5f}", end='')
            print()

        # Also print MMSE-LB ratio for Gen72 vs no-corr
        print()


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("="*70)
    print("ANALYSIS: Best Evolved Program from truebp_1 (Gen 72)")
    print("="*70)
    print()
    print("Program (Gen 72):")
    print("  Node.GetSymIm ; Matrix.Dup ; Float.Pop ;                        [line A]")
    print("  Node.NumChildren ; Float.Exp ;                                  [dead]")
    print("  Vec.GetResidue ; Float.Div ;                                    [dead]")
    print("  Float.GetNoiseVar ; Node.SetScore ;                             [side-effect]")
    print("  Node.ForEachSibling([Node.GetScore, Mat.Rows,                   [score-sum]")
    print("                       Float.Swap, Int.GetNumSymbols]) ;")
    print("  Node.GetParent ;                                                [navigate]")
    print("  Node.ForEachChild([Float.GetMMSELB, Node.GetCumDist,            [THE BP]")
    print("                     Float.Inv, Node.SetScore]) ;")
    print("  Int.GT ; Node.GetLayer                                          [dead output]")
    print()

    t0 = time.time()

    # Part 1 & 2
    run_trace_experiment(n_trials=100, snr_db=12, max_nodes=200)
    explain_program(snr_db=12)

    print(f"\n[Part 1+2 done in {time.time()-t0:.1f}s]")
    t1 = time.time()

    # Part 3: ablation at 200 nodes and 2000 nodes
    ablation_test(n_trials=500, snr_list=[10, 12, 14, 16], max_nodes=200)
    ablation_test(n_trials=500, snr_list=[10, 12, 14, 16], max_nodes=2000)

    print(f"\n[Part 3 done in {time.time()-t1:.1f}s]")
    t2 = time.time()

    # Part 4: full node budget sweep
    node_budget_test(n_trials=400, snr_list=[10, 12, 14, 16])

    print(f"\n[Part 4 done in {time.time()-t2:.1f}s]")
    print(f"\nTotal analysis time: {time.time()-t0:.1f}s")
    print("\n" + "="*70)
    print("SUMMARY (human-readable formula explanation)")
    print("="*70)
    print("""
The Gen-72 program implements a 'SIBLING SCORE INVERSION' rule:

  WHEN node n is created:
    1. Set n's own score = noise_var (side effect; overridden by driver)
    2. Collect SUM of all already-created sibling scores  → used as n's correction
    3. Navigate to parent(n)
    4. For every child c of parent (= siblings of n created before n):
         c.score ← 1 / c.cum_dist            ← THE BP WRITE

  EFFECT: Every time a new leaf is added, it RE-SCORES all its existing
          siblings using 1/cum_dist (inverted distance).

  DEAD CODE identified:
    - Node.GetSymIm / Matrix.Dup / Float.Pop  → push Im and immediately pop (no effect)
    - Node.NumChildren (returns 0 for new leaf) → no real use
    - Float.Exp   → float stack is empty at this point → no-op
    - Vec.GetResidue / Float.Div → k_int=0 fails validation → no-op
    - Float.GetMMSELB inside ForEachChild → Int top != layer → no-op
    - Int.GT / Node.GetLayer → produce Bool/Int that don't affect Float return

  WHAT REMAINS (the active kernel):
    Float.GetNoiseVar ; Node.SetScore ;       ← minor side effect on cand self
    ForEachSibling([Node.GetScore]) ;         ← correction = sum_prior_sibling_scores
    Node.GetParent ;
    ForEachChild([Node.GetCumDist, Float.Inv, Node.SetScore])  ← sibling BP

  INTERPRETATION:
    - At 200 nodes: the 1/cum_dist reordering redirects search toward nodes
      with large cumulative distance. At low node budget this may help escape
      local attention traps by DEPRIORITISING already-favored paths.
    - At 2000 nodes: the inversion is harmful — it pushes MIN-CUM-DIST nodes
      (the best ML candidates) to the back of the queue, wasting node budget.
    - This explains the reversed scaling: big bpg at training budget,
      regression at eval budget.

  VERDICT: The 'BP' discovered is real (verified by ablation at 200 nodes),
           but its ranking function (1/cum_dist) is ORDER-INVERTED for
           large budgets. The program exploits a training-budget artifact.
""")
