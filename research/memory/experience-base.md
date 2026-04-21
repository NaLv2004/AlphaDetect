# Experience Base

Lessons learned, reusable patterns, and insights accumulated from research.
These guide future research decisions and help avoid repeating mistakes.

---

## [2026-04-11] Easy Bounded Curricula Collapse to the Shortest Valid Metric
- **Context**: A 30-generation PushGP run on 4x4 QPSK with train_max_nodes=160 and a mixed low/moderate-noise dataset.
- **Lesson**: If the bounded curriculum is easy enough that cumulative distance already solves every training case, lexicase plus code-length pressure will converge immediately to `Node.GetDistance` and stop exploring deeper structure.
- **Applicable to**: Any AlphaDetect search where a trivial primitive can already satisfy the current training budget.

## [2026-04-11] Zero-Prior Stack Search Can Beat LMMSE Under a Modest Budget
- **Context**: The implemented typed-stack VM plus QR-based stack decoder was evaluated on 16x8 QPSK with 1500 node budget and the best evolved score `Node.GetDistance`.
- **Lesson**: Even the simplest primitive score inside the zero-prior search framework can outperform LMMSE in BER at 6 dB and 10 dB while using substantially fewer counted detector FLOPs than the rough LMMSE solve estimate.
- **Applicable to**: Early-stage AlphaDetect experiments where the goal is to validate the value of bounded tree search before attempting richer emergent structures.

## [2026-04-11] Human-Readable Program Logging Is Essential
- **Context**: Every generation was written to a persistent log with both one-line and structured renderings of the current best program.
- **Lesson**: Continuous readable logging makes it obvious when evolution has collapsed to a trivial invariant program and provides a direct hook for future novelty metrics based on syntax and behavior traces.
- **Applicable to**: Any long-running program-synthesis loop where interpretability and resumable research state matter.

## [2026-04-11 03:33] Raw Score Motion Is a Fake Dynamic Novelty Metric
- **Context**: The first hard 32x16/16-QAM experiments rewarded absolute rescoring movement, and `Graph.GetOpenCount + Node.GetDistance` looked highly dynamic even though it was only adding a cheap global offset.
- **Lesson**: Dynamic novelty must be tied to frontier rank utility, not raw score movement. Otherwise trivial global offsets masquerade as message passing.
- **Applicable to**: Any AlphaDetect search that tries to reward dynamic graph behavior in a bounded search tree.

## [2026-04-11 03:33] Holdout and Tight-Budget Pressure Reveal Real Dynamic Structure
- **Context**: Small-sample 32x16/16-QAM training repeatedly found zero-BER programs that collapsed on evaluation until the fitness was changed to include an independent holdout set and a stricter node-budget decoder.
- **Lesson**: On hard discovery tasks, generalization pressure must be baked into the inner fitness loop. More samples alone are not enough if the search can still win by exploiting one generous budget.
- **Applicable to**: Program synthesis over bounded decoders, anytime search, and other settings where train-time and eval-time budgets can diverge.

## [2026-04-11 03:33] Dynamic Best-Open-Score State Can Beat LMMSE at Medium/High SNR
- **Context**: After adding ancestor writeback and holdout/tight-budget fitness, the search discovered a metric equivalent to `Node.GetDistance * Node.GetState1 + const`, where `State1` stores the subtree's best open descendant score.
- **Lesson**: A dynamic tree-state term tied to the evolving frontier can outperform both static distance heuristics and LMMSE in the moderate/high-SNR regime without injecting privileged detector formulas.
- **Applicable to**: Future AlphaDetect work on dynamic search-tree computation, residual writeback, and learned anytime ranking.

## [2026-04-12] Distance Metric Is a Strong Local Attractor in GP Search
- **Context**: Complete v2 rewrite with fine-grained operators (77 primitives, no coarse domain ops). 16x8 16QAM with varied training budgets (40-80 max_nodes). After gen 3 the search converges to `Node.GetCumDist` (len=1) and never escapes despite 16+ generations.
- **Lesson**: Single-instruction `Node.GetCumDist` is a powerful local optimum because (a) any random perturbation makes it worse, (b) mutations of short programs mostly produce other short programs, (c) building multi-instruction programs that access R matrix requires ~5-7 coordinated instructions.
- **Mitigation strategies tried**: (1) Lower SNR training (6/8/10 dB) to increase distance-ordering failure rate, (2) 20% fresh random injection per generation, (3) `mutate_grow` operator that adds 2-8 instructions, (4) env_bias=0.3 in random instruction generation to favor R-matrix/vector access ops, (5) Fixed training datasets per epoch for stable fitness landscape, (6) Per-sample lexicase selection.
- **Key numbers**: At SNR 6-10 dB, distance-only achieves BER≈0.28-0.31 (with 80 max_nodes), while K-Best-16 achieves 0.28 (barely better). The gap is too small for GP to exploit unless the program uses fundamentally different tree search behavior.
- **Applicable to**: Any program synthesis where a trivial baseline is a strong local optimum. Need architectural changes (structured programs, library learning) or very strong diversity pressure.

## [2026-04-12] Rescoring During Training Creates Huge Computational Overhead
- **Context**: With rescore_interval=5 and max_nodes=80, each sample requires ~1680 VM executions (vs ~1296 without rescoring). For 150 pop × 15 samples, initial evaluation takes >5 minutes instead of ~63 seconds.
- **Lesson**: Rescoring (frontier re-evaluation) should be DISABLED during training evaluation for speed. Enable only during full evaluation or challenge testing.
- **Applicable to**: Any iterative search where the fitness evaluation runs the decoder many times per generation.

## [2026-04-12] Residual Scoring Breaks the Distance Attractor — KEY BREAKTHROUGH
- **Context**: Changed decoder from `score = VM_output` to `score = cum_dist + VM_output` (residual scoring). Programs now compute a CORRECTION to distance ordering.
- **Lesson**: Residual scoring fundamentally changes the fitness landscape: (1) zero correction = distance-only baseline (always safe), (2) random programs that return inf default to distance rather than breaking ordering, (3) small corrections can improve ordering without replicating distance computation, (4) the search space for "useful corrections" is vastly smaller than "good metrics from scratch." IMMEDIATE effects: complex programs (len=18-53) survive, 100% population diversity, programs use ForEachAncestor/ForEachSibling/Mat.ElementAt.
- **Applicable to**: ANY program synthesis problem where there's a known-good baseline — always evolve corrections/residuals rather than full solutions.

## [2026-04-12] Composite Score Penalty Dominance Blocks Discovery
- **Context**: FLOPs penalty (2e-6 * 14K = 0.028) EXCEEDED the BER (0.02083) for the best evolved program. This means the search prefers programs that reduce FLOPs even at the cost of higher BER.
- **Lesson**: When auxiliary penalties (FLOPs, length) dominate the primary objective (BER), the search optimizes the wrong thing. Reduce penalties so BER dominates at the current performance level. Rule of thumb: auxiliary penalties should be <30% of primary objective value.
- **Resolution**: Reduced FLOPs penalty 2e-6→3e-7 (7x), length penalty 5e-4→1e-4 (5x).
- **Applicable to**: Any multi-objective optimization where regularization terms can dominate the loss.

## [2026-04-12] Discovered Correction: Double-Weighted Distance + Symbol Preference
- **Context**: The best evolved program (len=18, BER=0.02083) computes correction = min(2*num_siblings, local_dist + Re(symbol)). Effective score: cum_dist + local_dist + Re(symbol) when not capped.
- **Lesson**: The evolved correction discovers two principles: (1) double-weighting local distance penalizes recent poor decisions more heavily — this is a form of "myopic pessimism" that favors paths with good recent choices, (2) symbol preference based on Re(symbol) breaks ties by preferring certain constellation points — this may encode an SNR-dependent preference for inner vs outer constellation points. The min cap prevents extreme penalties at deep layers.
- **Performance**: Matches K-Best-16 at most SNRs (8-14 dB), BEATS it at SNR=14 (BER=0.01 vs 0.0125), uses 7x fewer FLOPs (14K vs 100K). Beats LMMSE by 2-5x at all SNRs.
- **Limitation**: Does NOT access R matrix at all — the graph traversal (ForEachAncestor, ForEachSibling) is mostly dead code or used for sibling counting. An R-matrix-aware correction could be substantially better.
- **Applicable to**: Understanding what the GP can discover vs what it theoretically could discover. The gap between current and potential performance indicates room for algorithmic innovation.

## [2026-04-12] Theoretical Ceiling of One-Step Look-Ahead Correction
- **Context**: Fixed test_lookahead.py to use complex-valued R and y_tilde (removal of incorrect np.real() calls). LookAheadDecoder implements h_{k-1} = min_{s∈Ω} |y'_{k-1} - R[k-1,k-1]*s|² with interference cancellation.
- **Lesson**: At max_nodes=1500 (eval budget), LookAheadDecoder EXACTLY matches K-Best-16 at SNR≥12. At max_nodes=60 (train budget), LookAhead gives BER≈0.063 vs Distance 0.064 (only ~2% gain). This means the ceiling improvement for one-step look-ahead at training budget is very small. However the look-ahead IS a valid lower bound for A*, meaning correct orderings improve with it.
- **Implication**: The gap between current GP correction (BER=0.038 at 60 nodes) and LookAhead (BER=0.063 at 60 nodes) is NEGATIVE — current GP already outperforms pure look-ahead! This means the GP has discovered something better than pure one-step look-ahead at limited node budgets.
- **Applicable to**: Calibrating expectations for what A*-inspired corrections can achieve and understanding how current GP results compare to theory.

## [2026-04-12] V4 Gen9 Discovered New Program via ForEachAncestor
- **Context**: V4 evolution (with Hall-of-Fame seeding) discovered a new program at gen 9: `Node.GetParent ; Float.FromInt ; Node.ForEachSibling([Float.Swap, Matrix.Pop, Int.LT]) ; Vec.Sub ; Node.ForEachAncestor([Int.LT, Int.Const1, Vec.ElementAt])`. BER=0.03750, FLOPs=14478.
- **Lesson**: Each new evolution run with different seed discovers DIFFERENT programs with similar BER, suggesting multiple valid local optima exist in the correction space. Both v2 and v4 programs converge to BER≈0.02-0.038, 10K-15K FLOPs. The programs use graph-traversal operations (ForEachAncestor) in non-obvious ways.
- **New seeds**: Both programs added to Hall-of-Fame. Future runs can mutate from either.
- **Applicable to**: Understanding the discovered solution landscape. Diverse HoF seeding may prevent convergence to the same local optimum each time.

## [2026-04-12] New Instructions: Im() Access and Const-symbol Iteration
- **Context**: Added Mat.PeekAtIm, Vec.PeekAtIm (non-destructive imaginary-part access), Exec.ForEachSymbol (sum over QAM symbols), Exec.MinOverSymbols (min over QAM symbols). Total: 89 instructions (81 primitive + 8 control).
- **Lesson**: Imaginary-part instruction completeness is critical for complex-valued MIMO. Before these additions, GP could not exploit the imaginary part of R or y_tilde. The constellation iteration allows the GP to express h_{k-1} = min_{s} |y'_k - R_kk*s|² directly (the theoretically optimal A* heuristic). This is the first time the GP can DISCOVER the look-ahead correction from scratch.
- **Applicable to**: Designing GP instruction sets — completeness of the algebra over the domain is necessary for solving the target problem. Complex MIMO requires both real and imaginary part access.
## [2026-04-13] C++ Evaluator Stack Overflow from Large On-Stack Arrays
- **Context**: C++ evaluator with `SearchTree` containing `TreeNode nodes[8192]` (~10MB) caused stack overflow on Windows (default 1MB stack).
- **Lesson**: Never allocate large arrays on the stack in C/C++. Use heap allocation (new/delete, std::vector, std::unique_ptr) for any structure over ~100KB. SearchTree nodes must be heap-allocated.
- **Resolution**: Changed to `TreeNode* nodes = new TreeNode[cap]` with destructor cleanup.
- **Applicable to**: Any C++ simulation code with large per-function data structures.

## [2026-04-13] Architecture v3: Per-Node Program Execution + Dirty Tracking
- **Context**: Rewrote bp_decoder.py to remove all hardcoded BP (no sweeps, no bp_interval). Program runs on every new node. dirty_nodes set tracks which nodes had scores modified by the program via Node.SetScore. After each expansion, dirty frontier nodes get re-inserted into PQ with updated queue_version.
- **Lesson**: The architecture is much simpler than hardcoded BP: detect() → for each child: _score_node() → _process_dirty() → continue. The program can now create ANY message passing pattern by traversing the tree and calling SetScore. The framework doesn't prescribe WHEN or WHERE BP happens — it just provides the mechanism.
- **Key concern**: Programs must discover that SetScore on other nodes + graph traversal = BP. This requires evolving at least ~5-8 coordinated instructions. The distance attractor may remain a problem. env_bias=0.45 and aggressive fresh injection (20%) should help.
- **Applicable to**: Any program synthesis where the framework should provide mechanisms, not policies.

## [2026-04-13 03:20] Raw SetScore Count Is Not True BP
- **Context**: v4 runs were selecting programs with large `bp_updates` but weak BER at the real 200-node training budget. Direct comparison showed that some BP-active programs were worse than pure MMSE-LB and even worse than no-correction search.
- **Lesson**: Never reward graph write activity by count alone. True BP utility must be measured by ablation: run the same program with `Node.SetScore` enabled and disabled, and reward only positive BER improvement caused by nonlocal score writes. Count-based bonuses optimize fake message passing.
- **Applicable to**: Any AlphaDetect search space that exposes write primitives, frontier rescoring, or message-memory updates.

## [2026-04-13] CRITICAL BUG: TreeNode Was Not Hashable — SetScore NEVER Executed
- **Context**: `dirty_nodes` is a Python `set()`. `g.dirty_nodes.add(nd)` inside Node.SetScore raised `TypeError: unhashable type: 'TreeNode'` because Python `@dataclass` without explicit `__hash__` is unhashable when `__eq__` is defined. This exception was silently swallowed by `vm.run()`'s try/except. Every call to `Node.SetScore` in ALL previous experiments (v3_1, v3_2, and history) silently failed — no BP updates were EVER recorded, EVER.
- **Symptoms**: bp_updates=0 for ALL programs including manually constructed BP patterns. Result=inf instead of expected finite value when program used SetScore. Children[1] score not updated even though loop appeared to run.
- **Root cause**: Python `@dataclass` generates `__eq__` from all fields, which makes instances unhashable. Needed explicit `__hash__ = lambda self: hash(self.node_id)`.
- **Fix**: Added `__hash__(self)` and `__eq__(self, other)` to `TreeNode` in stacks.py, using `node_id` as the hash key.
- **Impact**: ALL v3 experiments (v3_1 through v3_2, 25+ generations total) produced results WITHOUT any BP message passing. The v4 experiments are the FIRST to have functional SetScore.
- **Lesson**: ALWAYS test that set operations actually work on custom classes. `dirty_nodes = set()` + custom class = silent failure if class is unhashable. When debugging "why isn't feature X working", instrument the actual operation with try/except and print the exception.
- **Applicable to**: Any Python code that stores custom objects in sets or as dict keys. Python silently (via try/except) swallows type errors inside GP execution blocks.

## [2026-04-13] Float.Inv Was Unimplemented: Silent No-Op Corrupted Instruction Set
- **Context**: `Float.Inv` (1/x) appeared in both `evolution.py` `_BP_AGGREGATES` mutation set (line 83) and random BP pattern generator (line 143), but was never added to vm.py's dispatch table. EVERY program containing `Float.Inv` silently ignored it.
- **Impact**: The best evolved program Gen72 (truebp_1) was intended to set sibling.score = 1/cum_dist (inverted distance ordering = bad). Instead, due to Float.Inv no-op, it set sibling.score = cum_dist (pure A* = almost acceptable). The "good" behavior was accidental.
- **Fix**: Added `Float.Inv` to PRIMITIVE_INSTRUCTIONS and implemented as `1.0 / a if a != 0.0 else inf`.
- **Lesson**: When adding instructions to the mutation/grammar set, ALWAYS simultaneously implement them in the VM. Use a test that verifies every entry in ALL_INSTRUCTIONS executes at least one distinct value through the VM. Silent no-ops create deceptive fitness landscapes where "would-be useful" instructions evolve by accident and create confusing false positive behaviors.
- **Applicable to**: Any GP system with a typed instruction set — run a completeness test on the instruction dispatch table as part of the test suite.

## [2026-04-13] High bpg Score Can Be Self-Inflicted Distortion Recovery, Not Genuine BP
- **Context**: truebp_1 Gen72 had bpg=0.462 (46% BER reduction from enabling score writes). Ablation confirmed this is real. However, without score writes, the ForEachSibling correction returns sum_prior_sibling_scores which grows to ≈500 for later-layer nodes → all scores are astronomically large → near-random search (BER≈0.5). Score writes (ForEachChild SetScore → reset to cum_dist) fix this. The "gain" measures rescue from self-inflicted damage.
- **Lesson**: A high bpg only proves that score writes are necessary.  It does NOT prove that the scoring function is ML-useful. Multiple false positive cases are possible: (1) the program corrupts scores and BP repairs them, (2) the program discards all scores and BP restores a neutral baseline (cum_dist), (3) the program creates a systematic bias that BP partially corrects. A TRUE BP signal would show that: (a) the no-BP version still performs reasonably (not near-random), (b) the BP version improves further on top of the reasonable baseline.
- **Detection method**: Check BER_noBP relative to MMSE-LB, not just relative to BER_BP. If BER_noBP >> BER_MMSE by a large factor (e.g., 8x), the bpg is likely self-healing rather than genuine enhancement.
- **Applicable to**: All future truebp experiments. Update fitness function to penalize programs where BER_noBP > 5× BER_MMSE (they're corrupting the search and fixing it with BP, not genuinely improving it).

## [2026-04-13] Pure A* Does Not Scale With Node Budget at High SNR
- **Context**: Gen72 at max_nodes=200 achieves ratio≈1.1 vs MMSE-LB at SNR=16. But at max_nodes=2000, ratio≈12.0 (catastrophic). MMSE-LB improves dramatically with more nodes (BER=0.00550→0.00125 at SNR=16 for 200→2000 budget), while Gen72 improves only slightly (0.00600→0.01500 — actually WORSE!).
- **Lesson**: Pure A* (score = cum_dist, no informed heuristic) degrades relative to MMSE-LB as node budget increases. MMSE-LB benefits from more nodes because it correctly prioritizes the globally-optimal path, finding it reliably before budget runs out. Pure A* expands an exponentially larger fraction of wrong paths. At high SNR, the optimal tree path has a tiny BER contribution but requires precise ordering — an uninformed heuristic fails here even with unlimited budget.
- **Implication**: Any evolved program that reduces to pure A* (score reset to cum_dist) is NOT a viable long-term solution. The fitness function at max_nodes=200 masks this degradation. Eval should always include max_nodes=2000 in the fitness signal.
- **Applicable to**: Setting evaluation budgets for future truebp experiments. Consider multi-budget fitness: f = w1*BER@200 + w2*BER@2000.

## [2026-04-13] 32-Sample Training Set Overfits — Evolution Exploits Fixed Seeds
- **Context**: Gen72 training BER=0.04688 ratio=0.857 (reported as BETTER than LMMSE) on 32 fixed training samples. Full 500-trial evaluation gives ratio=1.094-1.575 (WORSE than MMSE-LB). This 72-170% gap between training and eval performance exists across all SNRs.
- **Lesson**: 32 fixed training samples are insufficient for 16×8 MIMO 16-QAM with noise variability. The GP finds programs that exploit specific channel realizations in the fixed training set. Mitigation: (a) use ≥200 training samples with fresh random samples each generation, (b) use diverse fixed training sets that cover extreme channel conditions (ill-conditioned H), (c) add an explicit generalization holdout. The "ratio < 1" illusion on training is particularly dangerous — it encourages false confidence.
- **Applicable to**: All future evolution runs. Increase train_samples to ≥200, use epoch rotation with new random seeds each generation.

## [2026-04-14] Structured BP v2: Three Root Causes of Silent Failure
- **Context**: Structured-BP GP v2 (4-program genome: F_down, F_up, F_belief, H_halt) suffered three silent failures in cpp_test1 run.
- **Root Cause 1 — F_up noop filter false positive**: The F_up filter perturbed children's (local_dist, m_up) and checked if output changed. A noop program (returns stack top = last m_up) PASSES this test because perturbing m_up changes the output. Fix: after sensitivity test, check if output = orig_floats[-1] (the last pushed m_up); reject if yes.
- **Root Cause 2 — Leaf m_up = local_dist makes all BP messages trivially equal**: When leaf nodes initialize m_up = local_dist, for the root's children (all leaves at first level): m_up = local_dist = cum_dist. So F_belief(cum_dist, ~0, cum_dist) computes something that gives the SAME ordering as pure distance for ALL programs. Fix: initialize leaf m_up = 0.0 (no information from below = neutral initialization).
- **Root Cause 3 — No stagnation detection**: Evolution stuck at BER plateau indefinitely with no mechanism to escape. Fix: track consecutive gens without BER improvement; at threshold=4, triple fresh injection rate, halve elitism, increase mutation intensity.
- **Impact of fixes**: cpp_test3 run (15 gens) shows BER=0.05→0.05 (reduced from 0.05888 in test1). More importantly: population shows genuine BER differentiation (0.05-0.134), evolved programs are non-trivial (e.g., F_up=Sum_children(M_up), F_belief=max(M_down-M_up, D_i)), different programs produce different BERs.
- **Applicable to**: Any push-GP system where filter functions need to distinguish real computation from passthrough behaviors.

## [2026-04-14] Training Sample Count Controls BER Resolution — 30 Samples Too Few
- **Context**: cpp_test3 with 30 training samples: BER=0.05 corresponds to 12 errors/240 symbols. ALL top programs hit the same quantized BER=0.05 floor, evolution has no gradient to differentiate them.
- **Lesson**: With N training samples × Nt symbols, BER resolution = 1/(N × Nt). For 8-tx 16-QAM: min resolution = 1/(N×8). Need resolution ≤ 0.005 for meaningful selection pressure, which requires N ≥ 25 samples minimum, but 60-100 samples for reliable selection.
- **Production configuration**: 80 training samples across SNRs 10/12/14 dB → resolution = 1/640 ≈ 0.0016, sufficient for meaningful evolutionary pressure. This gives Gen2 BER=0.05852, Gen4 BER=0.05453 — clear improvement gradient.
- **Applicable to**: All future structured BP runs. Never use <50 training samples with 8-tx MIMO.

## [2026-04-14] Full Evaluation Is the Bottleneck Without C++ Acceleration
- **Context**: cpp_test3 with 200 eval_trials × 5 SNRs = 1000 Python decoders. Each decoder uses max_nodes=1500 — total: ~8 minutes even after 15 gens of C++-accelerated training.
- **Fix**: Modified full_evaluation() to accept optional cpp_evaluator; when provided, uses C++ bridge's evaluate_genome() instead of Python decoder per sample. Only baselines (LMMSE, K-Best) still use Python.
- **Result**: With --use-cpp, full eval should run in <30 seconds instead of 8+ minutes.
- **Applicable to**: Any future experiment using full_evaluation(). Always pass cpp_evaluator when use_cpp=True.

## [2026-04-14] Multiple Training SNRs Prevent Single-SNR Specialization
- **Context**: cpp_test1/3 used single training SNR (12 dB). Programs can specialize for this specific noise level, performing poorly at other SNRs. cpp_test4 uses SNRs [10,12,14] simultaneously.
- **Lesson**: Multi-SNR training forces programs to discover SNR-robust principles (e.g., distance-based ordering) rather than SNR-specific tuning. This aligns with the AlphaDetect goal of discovering universally valid detection algorithms.
- **Applicable to**: All production runs. Use at least 3 training SNRs spanning the eval range.

## [2026-04-14] Evolved Structured BP = Clamped A* — Confirmed by Ablation (200 trials)
- **Context**: Best evolved genome from cpp_test4 (gen-4, BER=0.05453) was traced instruction-by-instruction using the CORRECTED `program_to_formula` (multi-stack aware) and confirmed by ablation experiments.
- **Correct formula analysis** (fixed from earlier wrong analysis caused by buggy `program_to_formula`):
  - **F_down** = `(M_par_down + C_i)` — pure cumulative distance (A*). `Int.Inc` and `Node.GetParent` operate on the INT/NODE stacks, NOT the float stack. `Float.Add` computes the sum.
  - **F_belief** = `max(M_down, EC1)` = `max(cum_dist, 0.348)` — clamped A*. `Int.Sub` and `Int.Inc` operate INT stack only; `Float.Max` takes max(M_down, EC1).
  - **F_up** = `log(sqrt(max(M_last_child_up, 0.5/(0.5-EC1))))` ≈ `log(sqrt(max(M_up, 3.29)))`. For leaf children M_up=0, gives `log(1.814)≈0.596`. Sets root.m_up > 0.
  - **H_halt** = `(mem[layer%16] < root.m_up)`. Initially mem[8]=0.0, root.m_up≈0.596 → **ALWAYS TRIGGERS AFTER FIRST ITERATION**. Only 1 BP sweep actually runs despite max_bp_iters=3.
- **Ablation results (200 trials, max_nodes=600)**:
  - A1 (F_up=0): ratio=0.451 at SNR=12 vs baseline 0.528 — **F_up is dead weight, removing it HELPS**
  - A2 (F_down=passthrough, no C_i): BER≈0.93 (near-random!) — **F_down MUST accumulate distance**
  - A3 (max_bp_iters=0): ratio=0.758 at SNR=12 — **BP sweeps mechanically necessary** to compute m_down=cum_dist; without them all scores are constant EC1 → degenerate BFS
  - A4 (F_belief=D_i, pure A*): ratio=0.585 at SNR=12 — **floor EC1=0.348 genuinely helps** (baseline 0.528 is better)
- **Conclusion**: The evolved genome is identical to "A* with soft queue-floor 0.348" on the 16-QAM 8×16 MIMO problem. BP infrastructure is used only mechanically (to run the DOWN sweep once), not for genuine message passing.
- **Why the floor helps**: Nodes with very small cum_dist (shallow, fortunate partial paths) would otherwise dominate the frontier queue. The floor EC1=0.348 treats all such nodes equally, creating exploration diversity near the search tree root. This is an evolved, SNR-tuned form of epsilon-greedy tree search.
- **Next step**: cpp_test5 with max_nodes=1200 forces true look-ahead pressure. To escape clamped-A*, need programs that use R-matrix context or multi-iteration BP with non-trivial F_up messages.
- **Applicable to**: Understanding what structured BP can discover vs degenerate solutions. Check A1/A2/A3/A4 ablations for any future evolved genome before claiming "real BP".

## [2026-04-14] program_to_formula Bug: Int/Node Stack Ops Corrupted Float Formula Output
- **Context**: `program_to_formula` in bp_main_v2.py simulated only a single float stack. Instructions operating on OTHER stacks (Int.Inc, Int.Sub, Node.GetParent, Mat.Row, Bool.Or, etc.) were treated as unknown → pushed spurious tokens onto the symbolic float stack → completely wrong formula output.
- **Symptom**: For the gen-4 best genome: F_down showed `[Node.GetParent]` (should be `M_par_down + C_i`); F_belief showed `max([Int.Sub], [Int.Inc])` (should be `max(M_down, EC1)`). This led to wrong initial hypothesis about genome behavior.
- **Root cause**: VM has 5 stacks (float, int, bool, node, vector/matrix). `Int.Inc` increments INT stack; `Node.GetParent` pops/pushes NODE stack; `Bool.Or` operates BOOL stack — none of these touch the float stack. But the old function pushed `[Int.Inc]` etc. as float tokens.
- **Fix**: Rewrote `program_to_formula` to track SEPARATE symbolic stacks for float, int, and bool. Added explicit cross-stack operations: `Int.Inc/Sub/Add/Dec` → int stack no-ops; `Node.GetParent/Pop/Dup` → node nops; `Bool.Or/And/Not` → bool stack; `Float.FromInt`/`Int.FromFloat` → proper cross-stack conversion.
- **Validation**: Fixed function correctly outputs `(M_par_down + C_i)` for F_down and `max(M_down, EC1)` for F_belief.
- **Lesson**: Any symbolic interpreter of a multi-stack VM must model ALL stacks. Never treat unrecognized instructions as float-stack pushes — they should be classified as: (a) int-stack ops (no float effect), (b) bool-stack ops (no float effect), (c) node/vector ops (no float effect), (d) cross-stack ops (pop one stack, push another), or (e) truly unknown (mark as `[?name]`).
- **Applicable to**: Any future debugging/analysis of evolved programs. Always verify program_to_formula output against manual stack trace before concluding "what the program computes".

## [2026-04-22 00:37] Algorithm-IR Warm-Start GNN: Cheap Eval Helps, Proposal Generation Dominates
- **Context**: Phase A/B benchmark on `research/algorithm-IR/train_gnn.py` with gen-1 full pair warm-start.
- **Lesson**:
  - Switching scorer supervision from sparse sign labels to dense MSE targets and giving the host/donor region policies real stochastic actions makes the training signal much denser.
  - Lightweight warm-start evaluation plus parallel batch evaluation is sufficient to make `8190` graft evaluations practical.
  - Once that is in place, the main bottleneck becomes proposal generation itself, not evaluation.
- **Evidence**:
  - Warm-start gen-2 matched samples: `8190` vs baseline `20`.
  - Warm-start gen-1 proposal generation: `~176.6s`.
  - Warm-start gen-1 graft evaluation: `~18.2s` for all `8190` grafts.
- **Applicable to**:
  - Any future optimization pass on algorithm-IR training.
  - Deciding whether to invest in a native backend: profile first, because the evaluator may no longer be the slowest stage.
