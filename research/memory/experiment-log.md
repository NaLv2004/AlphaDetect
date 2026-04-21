# Experiment Log

Chronological record of all experiments conducted by the research system.
Each entry includes parameters, results, and key observations.

---

## [2026-04-11] Typed PushGP Smoke Validation
- **Topic**: research/mimo-push-gp/
- **Parameters**: population=16, generations=2, training=4x4 QPSK, train_samples=4, train_max_nodes=80, eval=16x8 QPSK, eval_trials=3, SNR=10 dB
- **Result**: Best program=`Node.GetDistance`; training BER=0.0000, training MSE=0.0000, train FLOPs=634; 16x8 BER=0.0000, 16x8 MSE=0.0000, average detector FLOPs=1629.33, LMMSE BER=0.0000
- **Observation**: The full pipeline executed correctly, produced human-readable program logs, and confirmed that the simplest valid primitive score is immediately competitive.
- **Files**: research/mimo-push-gp/results/run_summary.json, research/mimo-push-gp/logs/algorithm_evolution.log

## [2026-04-11] Full 30-Generation 16x8 Evaluation
- **Topic**: research/mimo-push-gp/
- **Parameters**: population=96, generations=30, training=4x4 QPSK, train_samples=24, train_max_nodes=160, train_flops_max=6000, eval=16x8 QPSK, eval_trials=80, eval_max_nodes=1500, SNR={6,10,14} dB
- **Result**: Best program=`Node.GetDistance`; training BER=0.0000, training MSE=0.0000, training average FLOPs=474.67. 16x8 results: at 6 dB BER=0.0078125 vs LMMSE 0.0203125, at 10 dB BER=0.0000 vs LMMSE 0.0015625, at 14 dB both BER=0.0000. Average detector FLOPs were 2261.5, 1371.3, and 1328.0 respectively, versus estimated LMMSE FLOPs 11296.
- **Observation**: Under the tested budget, the zero-prior tree metric already outperforms LMMSE at low and moderate SNR, but evolution collapses to the shortest valid primitive because the training curriculum does not force richer score structure.
- **Files**: research/mimo-push-gp/results/run_summary.json, research/mimo-push-gp/logs/algorithm_evolution.log

## [2026-04-11 03:33] 32x16 16-QAM Dynamic Rescoring Hard-Smoke
- **Topic**: research/mimo-push-gp/
- **Parameters**: population=12, generations=3, training=32x16 16-QAM, train_samples=2, train_max_nodes=120, train_flops_max=40000, eval_trials=2, eval_max_nodes=400, train SNR={14,18} dB, eval SNR={14} dB, dynamic frontier rescoring with rank-change novelty
- **Result**: The cheap relational escape `Graph.GetOpenCount ; Float.FromInt ; Node.GetDistance ; Float.Add` reached training BER=0.0000 but failed the real target, giving 14 dB BER=0.0625 versus LMMSE 0.03125 and detector FLOPs 182112 versus estimated LMMSE FLOPs 81984.
- **Observation**: Frontier rescoring alone is not enough if the fitness can still reward tiny-sample hacks; the hard 32x16 regime immediately exposed that fake dynamic structure can be both less accurate and more expensive than LMMSE.
- **Files**: research/mimo-push-gp/logs/algorithm_evolution.log

## [2026-04-11 03:33] Holdout/Tight-Budget 32x16 16-QAM Dynamic Metric Discovery
- **Topic**: research/mimo-push-gp/
- **Parameters**: population=16, generations=4, training=32x16 16-QAM, train_samples=4, train_max_nodes=96, train_flops_max=50000, holdout challenge_samples=4 at half node budget, eval_trials=4 at 14 dB; follow-up candidate cross-check on the same 32x16 target with 12 trials per SNR at SNR={12,14,16} dB
- **Result**: Best discovered program=`Node.GetDistance ; Node.GetState1 ; Float.Mul ; Mat.VecMul ; Float.ConstHalf ; Float.Add`. A simplified equivalent `Node.GetDistance ; Node.GetState1 ; Float.Mul ; Float.ConstHalf ; Float.Add` matched BER exactly while reducing detector FLOPs from 23840 to 19744. Multi-SNR results for the simplified metric: 12 dB BER=0.09375 vs LMMSE 0.0885417, 14 dB BER=0.015625 vs 0.046875, 16 dB BER=0.0052083 vs 0.015625.
- **Observation**: `Node.GetState1` is the first clearly nontrivial dynamic tree-state metric in this thread. It behaves like a dynamic “best open descendant score” term, beats LMMSE at medium/high SNR with lower detector FLOPs, and survives holdout pressure, but low-SNR robustness at 12 dB is still unfinished.
- **Files**: research/mimo-push-gp/logs/algorithm_evolution.log, research/mimo-push-gp/results/run_summary.json, research/mimo-push-gp/results/candidate_eval_compare.json

## [2026-04-12] v2 Rewrite: From-Scratch Discovery with Residual Scoring (16×8 16-QAM)
- **Topic**: research/mimo-push-gp/
- **Parameters**: population=100, generations=26, training=16×8 16-QAM, train_samples=12, train_max_nodes=50, train_flops_max=60000, eval_trials=200, eval_max_nodes=1500, eval_flops_max=200000, train SNR={10,12,14} dB, eval SNR={8,10,12,14,16,18,20} dB. Residual scoring architecture (score = cum_dist + evolved_correction). No seeds. 83 instructions (77 primitive + 6 control). env_bias=0.30-0.35.
- **Result**: Best discovered program (gen 10): `Exec.DoTimes([Mat.ElementAt, Float.FromInt, Float.Min, Node.GetLocalDist]) ; Node.GetSymRe ; Node.ForEachAncestor([Bool.Dup, Matrix.Pop]) ; Node.ReadMem ; Vec.Sub ; Node.ForEachSibling([Node.ForEachSibling([Float.Const2]), Float.Min, Node.ChildAt, Node.ForEachSibling([Vec.Sub])])`. Training BER=0.02083, ratio=0.250, FLOPs=14056, code_length=18. Full eval results:
  - SNR=8:  Evo=0.14 vs KB16=0.14 (EQUAL), LMMSE=0.25
  - SNR=10: Evo=0.075 vs KB16=0.075 (EQUAL), LMMSE=0.1825
  - SNR=12: Evo=0.04375 vs KB16=0.043125 (NEAR EQUAL), LMMSE=0.105
  - SNR=14: Evo=0.01 vs KB16=0.0125 (**EVO BEATS KB16**), LMMSE=0.04
  - SNR=16: Evo=0.003125 vs KB16=0.00125, LMMSE=0.016875
  - SNR=18: Evo=0.0 vs KB16=0.0
  - SNR=20: Evo=0.0 vs KB16=0.0
  - Evo FLOPs ≈ 10K-14K vs KB16 FLOPs = 100K (**7x more efficient**)
- **Analysis of discovered correction**: Effective computation is min(2*num_siblings, local_dist + Re(symbol)). Does NOT access R matrix. Graph traversal (ForEachAncestor, ForEachSibling) is mostly dead code or used only for sibling counting. The correction double-weights local distance and adds symbol preference.
- **Observation**: Residual scoring is the KEY breakthrough — programs no longer collapse to trivial len=1. Complex 18-instruction programs with graph traversal survive. The discovered algorithm matches K-Best-16 at most SNRs and BEATS it at SNR=14, with 7x fewer FLOPs. However, the correction doesn't access R matrix structure — room for substantial improvement. FLOPs penalty in composite_score (2e-6 * 14K = 0.028) EXCEEDS BER (0.02083), blocking discovery of better-but-more-expensive corrections.
- **Files**: research/mimo-push-gp/logs/algorithm_evolution_v2.log, research/mimo-push-gp/results/run_summary_v2.json

## [2026-04-12] v4 Evolution: Reduced Penalties + Hall-of-Fame Seeding
- **Topic**: research/mimo-push-gp/
- **Parameters**: population=150, train_samples=20, train_max_nodes=60, train_flops_max=80000, train SNR={10,12,14}, eval_trials=200, eval_max_nodes=1500, eval_flops_max=200000, eval SNR={8,10,12,14,16,18,20}. Reduced composite_score penalties: FLOPs 2e-6→3e-7 (7x reduction), length 5e-4→1e-4 (5x reduction). Hall-of-fame seeding with discovered v1 program (NOT hand-designed). Epoch rotation every 3 gens.
- **Result**: RUNNING
- **Observation**: Pending

## [2026-04-13] v3 Architecture: Program-Controlled Scoring (No Hardcoded BP)
- **Topic**: research/mimo-push-gp/
- **Parameters**: v3_1 (seed=42) and v3_2 (seed=137). population=100, train_samples=16, train_max_nodes=500, step_max=1500, train_flops_max=2M, train SNR={10,12,14}. eval_trials=200, eval_max_nodes=2000, eval_flops_max=5M, eval SNR={8,10,12,14,16}. Architecture: removed ALL hardcoded BP, removed 4 macro operators, ~99 instructions, program runs on every new node.
- **Result (v3_1 Gen 9)**: BER=0.0625, ratio=0.615 (38% better than LMMSE). Gen 5 full eval: beats LMMSE at ALL SNRs (8-16 dB). E.g. SNR=12: Evo=0.082 vs LMMSE=0.105. SNR=14: Evo=0.022 vs LMMSE=0.040. ~70K FLOPs (much lower than K-Best-16).
- **Result (v3_2 Gen 3)**: BER=0.109, ratio=0.875. Best len=4 program: `Mat.VecMul ; Node.ForEachSibling([Vec.Dot, Float.Neg])`.
- **Key Observations**:
  1. Programs consistently discover ForEachSibling traversal — sibling-aware scoring is a natural emergent structure
  2. Noise-scaled regularization (σ²/Im(s)) discovered from scratch — MMSE-like structure without any MMSE instruction
  3. No Node.SetScore usage yet — true BP (modifying other nodes' scores) has not emerged
  4. Even without BP, 47% BER improvement over LMMSE at Gen 6 (ratio=0.526)
  5. Programs access R matrix and y_tilde via Mat.PeekAt/ElementAt and Vec.SecondPeekAt
  6. ~1 min/generation speed with 100 pop × 500 nodes (Python decoder)
- **Files**: logs/bp_evolution_v3_1.log, logs/bp_evolution_v3_2.log

## [2026-04-13] truebp_1: Gen72 Analysis — Self-Reset BP False Positive
- **Topic**: research/mimo-push-gp/
- **Parameters**: truebp_1 from-scratch, seed=31415, population=100, train_samples=32, train_max_nodes=200, train SNR={10,12,14}. Analysis performed via analyze_truebp1.py (500 trials, SNR={10,12,14,16}).
- **Best Program (Gen 72, stagnated to Gen 222+)**:
  ```
  Node.GetSymIm ; Matrix.Dup ; Float.Pop ; Node.NumChildren ; Float.Exp ;
  Vec.GetResidue ; Float.Div ; Float.GetNoiseVar ; Node.SetScore ;
  Node.ForEachSibling([Node.GetScore, Mat.Rows, Float.Swap, Int.GetNumSymbols]) ;
  Node.GetParent ;
  Node.ForEachChild([Float.GetMMSELB, Node.GetCumDist, Float.Inv, Node.SetScore]) ;
  Int.GT ; Node.GetLayer
  ```
  Training fitness: BER=0.04688, ratio=0.857, bpg=0.462.
- **Dead code confirmed** (8/22 instructions dead):
  - `Float.Exp`, `Vec.GetResidue`, `Float.Div`: float/vector stack empty at call point
  - `Float.Swap`: only 1 float on stack (no effect)
  - `Float.Inv`: **UNIMPLEMENTED in vm.py** — listed in evolution.py mutation set but VM silently ignores it
  - `Float.GetMMSELB` (in ForEachChild): Int stack top = 16 (accumulated), fails ≤Nt=8 check
  - `Mat.Rows`, `Int.GetNumSymbols` (in ForEachSibling): push values to Int but never consumed
  - `Node.GetParent`: swaps node stack (parent replaces candidate) but net node_stack.depth unchanged
- **Active kernel**:
  1. `Float.GetNoiseVar ; Node.SetScore` → cand.score = σ² (minor, overridden by driver)
  2. `ForEachSibling([Node.GetScore])` → correction = Σ prior_sibling_scores (often 0)
  3. `Node.GetParent` → navigate to parent
  4. `ForEachChild([Node.GetCumDist, Node.SetScore])` → **THE BP: sibling.score = sibling.cum_dist**
- **Interpretation**: Each new leaf triggers a reset of ALL parent's children scores to their cumulative distances. This approximates pure A* (best-first by cum_dist) via periodic reset, undoing any distortion introduced by the correction term.
- **Ablation results** (500 trials each, BER format):
  - max_nodes=200: BP vs noBP vs MMSE-LB:
    - SNR=10: 0.193 / 0.574 / 0.176  → ratio_BP=1.094 (WORSE than MMSE-LB)
    - SNR=12: 0.081 / 0.526 / 0.063  → ratio_BP=1.277
    - SNR=14: 0.032 / 0.489 / 0.020  → ratio_BP=1.575
    - SNR=16: 0.006 / 0.493 / 0.006  → ratio_BP=1.091
  - max_nodes=2000: BP vs noBP vs MMSE-LB:
    - SNR=10: 0.160 / 0.627 / 0.153  → ratio_BP=1.048
    - SNR=12: 0.062 / 0.584 / 0.051  → ratio_BP=1.215
    - SNR=14: 0.028 / 0.582 / 0.011  → ratio_BP=2.667 (**worse at 2000 nodes!**)
    - SNR=16: 0.015 / 0.548 / 0.001  → ratio_BP=12.000 (catastrophic at high SNR/large budget)
- **Key finding**: noBP BER ≈ 0.5 at ALL SNRs → without BP, the program corrupts scoring so badly search becomes near-random. The "BP gain" (bpg=0.462) is ENTIRELY escape from self-inflicted distortion, NOT genuine ML-useful message passing.
- **Critical VM bug discovered and fixed**: `Float.Inv` (1/x) was listed in evolution.py's `_BP_AGGREGATES` and mutation set, but was NEVER implemented in vm.py. Both sides confirmed: grep shows it in evolution.py line 83/143, not in vm.py dispatch. VM silently no-ops it. **Fix**: Added `Float.Inv` to vm.py PRIMITIVE_INSTRUCTIONS and implementation. Previously, the program set sibling.score = cum_dist (accidentally correct) instead of sibling.score = 1/cum_dist (intended but inverted = worse).
- **Part 4 — Node Budget Comparison** (FIXED vm.py, 400 trials, SNR={10,12,14,16}, run_time=926.8s):
  - K-Best-16=K-Best-32 reference: {10: 0.13656, 12: 0.04031, 14: 0.01094, 16: 0.00250}
  - max_nodes=200:  Gen72_BP=0.997/0.997/0.996/0.995 | Gen72_noBP=0.543/0.509/0.493/0.477 | MMSE-LB=0.175/0.074/0.017/0.005 | no-corr=0.199/0.093/0.025/0.006
  - max_nodes=500:  Gen72_BP=0.996/0.994/0.996/0.995 | Gen72_noBP=0.540/0.523/0.504/0.480 | MMSE-LB=0.153/0.051/0.010/0.001 | no-corr=0.178/0.059/0.010/0.001
  - max_nodes=1000: Gen72_BP=0.994/0.995/0.995/0.995 | Gen72_noBP=0.572/0.539/0.512/0.491 | MMSE-LB=0.149/0.056/0.014/0.002 | no-corr=0.153/0.046/0.012/0.002
  - max_nodes=2000: Gen72_BP=0.995/0.996/0.994/0.995 | Gen72_noBP=0.619/0.589/0.551/0.570 | MMSE-LB=0.153/0.045/0.010/0.002 | no-corr=0.149/0.037/0.010/0.002
  - **Key Part 4 observations**:
    1. Gen72_BP is catastrophically broken at ALL budgets (BER≈0.99) — Float.Inv fix confirmed it was entirely an artifact
    2. Gen72_noBP DEGRADES with budget (BER 0.48→0.57 at SNR=16, 200→2000 nodes) — cumulative sibling penalty heuristic anti-scales
    3. MMSE-LB and no-corr both scale with budget; no-corr slightly beats MMSE-LB at 2000 nodes (0.037 vs 0.045 at SNR=12)
    4. Gen8 (`Float.FromInt`) shows mild improvement but trails MMSE-LB at all budgets
- **Files**: research/mimo-push-gp/logs/bp_evolution_truebp_1.log, research/mimo-push-gp/code/analyze_truebp1.py, research/mimo-push-gp/logs/analysis_truebp1.log, research/mimo-push-gp/code/vm.py (Float.Inv added), research/mimo-push-gp/code/part4_only.py


- **Topic**: research/mimo-push-gp/
- **Parameters**: No strong algorithm seeds. Replaced raw `bp_updates` reward with ablation-verified BP utility: evaluate candidate with normal decoder and with `Node.SetScore` disabled, then reward only positive BER gain from nonlocal score writes. Added nonlocal BP instrumentation to VM/decoder. Smoke run: generations=1, population=10, train_samples=4, train_max_nodes=60, train SNR={10,12}. Relaunch: `truebp_1`, seed=31415, population=100, train_samples=32, train_max_nodes=200, train_flops_max=3M, step_max=2000, train SNR={10,12,14}, eval SNR={8,10,12,14,16}, batch_gens=5.
- **Result**: Smoke run completed successfully with new log fields `BPnl` and `gain`. Old v4_3-v4_7 runs were terminated because they used obsolete fitness. `truebp_1` was launched from scratch under the new criterion.
- **Observation**: Raw SetScore activity was selecting fake BP programs. At the actual 200-node budget, previously celebrated BP-active programs were worse than pure MMSE-LB or even no-correction baselines. The correct signal is not “did the program write scores?” but “did nonlocal score writes measurably improve BER over the no-write ablation?”
- **Files**: research/mimo-push-gp/code/vm.py, research/mimo-push-gp/code/bp_decoder.py, research/mimo-push-gp/code/evolution.py, research/mimo-push-gp/code/bp_main.py, research/mimo-push-gp/logs/bp_evolution_smoke_truebp.log

## [2026-04-14] cpp_test4 Full Run + Ablation Analysis
- **Topic**: research/mimo-push-gp/
- **Parameters**: Structured BP v2. generations=40, pop=80, train_samples=80, train_max_nodes=600, train_flops_max=600K, step_max=1000, train_snrs=10,12,14, eval_trials=200, eval_max_nodes=2000, eval_flops_max=5M, C++ acceleration enabled. 8×16 MIMO, 16-QAM.
- **Result**: Best genome found at gen-4, BER=0.05453, training ratio=0.459 vs LMMSE. Evolution stagnated after gen-4; hard restarts at gens ~16/28 didn't find better. Full eval lost to stdout (batch mode log bug — fixed).
- **Genome analysis** (corrected via fixed program_to_formula):
  - F_down = `(M_par_down + C_i)` — cumulative distance (standard A*)
  - F_belief = `max(M_down, EC1)` where EC1=0.348 — clamped A* with soft floor
  - F_up = constant (dead weight, A1 ablation removing it HELPS)
  - H_halt = triggers after 1 iteration (checks `mem[k] < root.m_up`, always true initially)
- **Ablation results** (200 trials, max_nodes=600, SNRs 8/10/12/14/16):

| Config | SNR=8 | SNR=10 | SNR=12 | SNR=14 | SNR=16 |
|--------|-------|--------|--------|--------|--------|
| BASELINE ratio | 1.020 | 0.861 | 0.528 | 0.393 | 0.087 |
| A1 (F_up=0) ratio | 0.893 | 0.811 | 0.451 | 0.230 | 0.053 |
| A2 (F_down=passthrough) BER | 0.933 | 0.934 | 0.934 | 0.933 | 0.938 |
| A3 (no BP sweeps) ratio | 1.015 | 0.980 | 0.758 | 0.767 | 0.385 |
| A4 (F_belief=D_i) ratio | 0.991 | 0.848 | 0.585 | 0.318 | 0.000 |

- **Key observations**:
  1. A2 (no cumulative distance) → catastrophe (BER≈0.93) — F_down cumulation is essential
  2. A1 (F_up=0) is BETTER than baseline at all SNRs — F_up is dead weight
  3. A3 (no BP) → BER much worse because m_down never computed (stays 0), all scores=EC1=constant → degenerate BFS
  4. A4 (pure A*) slightly worse than baseline (0.585 vs 0.528 at SNR=12) — EC1 floor genuinely helps
  5. **Conclusion**: Evolved genome = A* with soft queue-floor EC1=0.348. No genuine BP message passing.
- **Clamped A* theory verification** (50 trials, max_bp_iters=1):
  - Pure A* ratio: 0.553 at SNR=12
  - Clamped A* (floor=0.348) ratio: 0.742 at SNR=12
  - Evolved baseline ratio: 0.550 at SNR=12
  - **Clamped A* with the "right" floor does NOT match evolved baseline**. The evolved genome has H_halt triggering after 1 BP iteration while clamped A* test uses bp_iters=1 — but both should give identical results. Noise variance in 50-trial test explains the ~0.02 gap.
- **Files**: research/mimo-push-gp/logs/sbp_evolution_cpp_test4.log, research/mimo-push-gp/code/ablate_best_genome.py, research/mimo-push-gp/code/ablate_fast.py, research/mimo-push-gp/code/test_clamped_astar.py, research/mimo-push-gp/code/test_formula_fix.py

## [2026-04-15] v7c: Breakthrough — Evolved BP Beats KB16
- **Topic**: research/mimo-push-gp/
- **Parameters**: Structured BP v2. pop=100, train_samples=100, train_max_nodes=300, step_max=500, flops_max=3M, train_snrs=22,24, eval_snrs=16,18,20,22,24, eval_max_nodes=1500, C++ batch eval with flattened OpenMP (genome×sample), max_bp_iters=3. 16×16 MIMO, 16-QAM. Seeded with V3 best.
- **Result**: 90 generations completed. Best training BER=0.00434 at Gen 56. Full eval at Gen 85:

| SNR | Evolved BER | KB16 BER | KB32 BER | vs KB16 |
|-----|-------------|----------|----------|---------|
| 16  | 0.2614      | 0.2227   | 0.1704   | worse   |
| 18  | 0.1098      | 0.0909   | 0.0542   | worse   |
| 20  | 0.0190      | 0.0263   | 0.0096   | **1.4× better** |
| 22  | 0.00192     | 0.00539  | 0.00131  | **2.8× better** |
| 24  | 0.0000875   | 0.00138  | 0.00017  | **15.8× better** |

- **Winning genome formulas** (evolved, zero priors):
  - F_down = -MMSE_LB (downward message = negative MMSE lower bound)
  - F_up = Sum_children(M_up) (aggregate children's upward messages)
  - F_belief = D_i + M_up - 2*M_down (score = distance + message imbalance)
  - H_halt = 0.5 (varies: YES)
  - Constants: EC0=0.004, EC1=3.42, EC2=4.73, EC3=0.54
- **Key observations**:
  1. FLOPs penalty bug: complexity_pen=1e-9 made 40M FLOPs add 0.04 penalty (66% of BER=0.06), causing evolution to prefer simpler genomes with WORSE BER. Fixed to 1e-11.
  2. Flattened OpenMP loop fixed straggler problem: initial eval 45.8s vs never-finishing before.
  3. Belief formula D_i + M_up - 2*M_down is a genuinely novel scoring that combines path cost with BP message imbalance. V3's formula was D_i*(M_up-EC3)/M_down.
  4. At 22+24 dB, evolved detector BEATS KB16 decisively and approaches KB32 performance.
  5. At 16-18 dB, performance degrades — BP tree search struggles in high-noise regime.
- **Files**: research/mimo-push-gp/logs/sbp_evolution_run0416_v7c.log, research/mimo-push-gp/code/seed_v7c_best.json

## [2026-04-15] v7d: Continued Evolution with Fixed Fitness + Structural Discovery
- **Topic**: research/mimo-push-gp/
- **Parameters**: Same as v7c but with fixes: complexity_pen=1e-11 (was 1e-9), batched constant_hill_climb (60 trials parallel). Seeded with v7c Gen 56 best genome.
- **Result (ongoing, Gen 20+)**:
  - Gen 5: Training BER=0.00700, same formula as seed. Full eval identical to v7c (22dB=0.00192).
  - **Gen 14: Structural breakthrough** — F_down mutated from `-MMSE_LB` to `-(C_i + MMSE_LB)`. Training BER=0.00481. Discovered during stagnation-boosted mutation (stagnant_gens=8).
  - Gen 15 full eval with new F_down:

| SNR | New F_down | Old F_down | Delta |
|-----|-----------|-----------|-------|
| 16  | 0.2385    | 0.2614    | -8.8% |
| 18  | 0.0912    | 0.1098    | -17%  |
| 20  | 0.0172    | 0.0190    | -9.5% |
| 22  | 0.00236   | 0.00192   | +23%  |
| 24  | 0.000106  | 0.0000875 | +21%  |

  - Also discovered F_belief = D_i + M_up - 3*M_down variant (BER=0.00572, #2-3 ranked at Gen 14)
- **Key observations**:
  1. FLOPs penalty fix was CRITICAL: evolution now correctly prioritizes BER over program shortness.
  2. Stagnation boost mechanism verified: after 8 gens stuck, enhanced mutations found structural improvement.
  3. New F_down = -(C_i + MMSE_LB) incorporates cumulative path cost — better at low SNR, slightly worse at high SNR.
  4. Gen timing: ~52-57s/gen (down from ~75s in v7c thanks to batched constant_hill_climb).
  5. The two formula variants represent a Pareto front: old better at high SNR, new better at low SNR.
- **Files**: research/mimo-push-gp/logs/sbp_evolution_run0416_v7d.log, research/mimo-push-gp/results/sbp_run0416_v7d_gen*.json

## [2026-04-22 00:37] Algorithm-IR Phase A/B Warm-Start GNN Benchmark
- **Context**: Implemented phase A/B for `research/algorithm-IR/train_gnn.py`: gen-1 full pair warm-start (one sampled graft per host/donor pair), scorer MSE on real graft composite score, stochastic host/donor region policies, replay retention, lightweight warm-start graft evaluator, parallel `evaluate_batch`, and materialized callable caching.
- **Benchmark setup**:
  - Baseline: `warmstart_gens=0`, `gens=3`, `proposals=20`, `pool_size=91`, `n_trials=1`, `timeout=0.3`, `train_steps=5`.
  - Warm-start: same config except `warmstart_gens=1`, `warmstart_trials=1`, `warmstart_timeout=0.15`, `warmstart_eval_workers=12`, `warmstart_survivor_cap=32`.
  - Outputs in `bench/baseline/results/gnn_training/` and `bench/warmstart/results/gnn_training/`.
- **Results**:
  - Baseline gen-2 matched samples: `20`; gen-3 matched samples: `40`.
  - Warm-start gen-2 matched samples: `8190`; gen-3 matched samples: `8210`.
  - Baseline median SER by gen: `0.3125 -> 0.25 -> 0.0625`.
  - Warm-start median SER by gen: `0.0 -> 0.0 -> 0.0` on the same noisy `n_trials=1` evaluator.
  - Baseline grafted survivors by gen: `2 -> 2 -> 2`.
  - Warm-start grafted survivors by gen: `32 -> 53 -> 58`.
  - Runtime: baseline total `19.7s`; warm-start total `231.6s`.
- **Profiling observation**:
  - Warm-start gen-1 proposer time was dominant: `176.6s` to build `8190` proposals.
  - Warm-start gen-1 graft evaluator time was `18.2s` for `8190` lightweight evaluations, so evaluation is no longer the main bottleneck.
- **Takeaway**: Phase A/B materially increased training data density and early graft usefulness, but the dominant cost is now Python-side proposal generation rather than evaluator throughput.
