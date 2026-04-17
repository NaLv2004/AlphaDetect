# Decision History

Record of major research decisions, rationale, alternatives, and impact.

---

## [2026-04-11] Remove Privileged Fallback from Bounded Search
- **Decision**: Replaced the previous LMMSE fallback path with a bounded greedy QR-metric completion of the best partial node when the search budget expires.
- **Rationale**: A fallback detector outside the evolved search loop violates the zero-prior spirit of the theme and hides whether the evolved score is actually carrying the search.
- **Alternatives**: Keep the LMMSE fallback for convenience; return the best partial path without completion.
- **Impact**: The detector remains inside the same stack-decoder framework even under budget exhaustion, and measured performance can be attributed to the evolved metric plus bounded tree search rather than an injected baseline.

## [2026-04-11] Keep Primitive-Only Seed Programs
- **Decision**: Seeded the initial population with a small set of one-line primitive programs that produce valid Float outputs, including `Node.GetDistance` and `Node.GetData`.
- **Rationale**: Purely random typed programs frequently return no score or hit execution limits, which stalls evolution before any meaningful selection pressure can form.
- **Alternatives**: Start from fully random programs only; inject a handcrafted advanced metric.
- **Impact**: The search starts from valid zero-prior building blocks without introducing privileged detector structure, but the curriculum now needs to be strengthened to prevent collapse to the shortest trivial program.

## [2026-04-11] Prioritize Harder Curriculum Over More Handcrafted Logic
- **Decision**: After the first 16x8 success, the next step is to make the training environment more adversarial and relational instead of hardcoding richer detector logic.
- **Rationale**: The current result already shows that the framework works; the failure mode is not insufficient heuristic engineering but insufficient evolutionary pressure to discover structure beyond cumulative distance.
- **Alternatives**: Add handcrafted search heuristics; weaken the zero-prior principle to chase more immediate performance.
- **Impact**: Future changes will focus on state exposure, curriculum, and selection pressure while preserving the AlphaDetect requirement that advanced structure must emerge rather than be prescribed.

## [2026-04-11 03:33] Measure Dynamic Behavior by Frontier Rank Change
- **Decision**: Replaced raw rescoring-delta novelty with frontier rank-change novelty when scoring dynamic behavior.
- **Rationale**: Raw score motion was being gamed by trivial global offsets and did not indicate that later evidence actually changed which branches would be explored.
- **Alternatives**: Keep absolute score movement; remove the dynamic novelty term entirely.
- **Impact**: Fitness now rewards changes that matter to search order, making fake dynamic structure much harder to exploit.

## [2026-04-11 03:33] Add Holdout and Tight-Budget Challenge to Inner Fitness
- **Decision**: Evaluated each candidate program on both a primary dataset and an independent holdout dataset using a stricter node budget, and penalized generalization gap directly in the composite score.
- **Rationale**: The first 32x16/16-QAM hard runs were overfitting tiny training sets and producing zero-BER programs that collapsed on real evaluation.
- **Alternatives**: Only increase training-sample count; keep a single generous budget and hope lexicase selection generalizes.
- **Impact**: The search now favors metrics that survive both distribution shift and tighter anytime constraints, which was necessary to reveal the first useful dynamic tree-state metric.

## [2026-04-11 03:33] Promote the Simplified State1 Metric to a Seed
- **Decision**: Added the simplified discovered metric `Node.GetDistance ; Node.GetState1 ; Float.Mul ; Float.ConstHalf ; Float.Add` to the seed set for future evolution.
- **Rationale**: Cross-checking showed that the learned `Mat.VecMul` instruction was dead code, and the simplified form preserves BER while reducing FLOPs.
- **Alternatives**: Keep only the exact evolved syntax; rely on future mutation to rediscover the shorter equivalent.
- **Impact**: Subsequent searches can refine the genuinely useful dynamic structure directly instead of wasting budget on syntactically bloated equivalents.

## [2026-04-12] v2 Rewrite: Remove All Coarse Operators, True From-Scratch Discovery
- **Decision**: Complete rewrite of the PushGP system. Removed all coarse-grained operators (Node.GetState0-7, Node.GetData, Node.GetScore, Node.GetVisitCount, etc.) and pre-computed tree statistics. Replaced with fine-grained Node.ReadMem/WriteMem + raw physical accessors. Added graph traversal control flow (ForEachChild/Sibling/Ancestor). Removed ALL seed programs. Added K-Best-16/32 baselines with runtime FLOPs. Changed to 16×8 16QAM.
- **Rationale**: Existing operators were too coarse-grained, seeds biased search, system was not truly from-scratch.
- **Alternatives**: Incremental refinement; keep some seeds; keep pre-computed stats.
- **Impact**: Much harder search space but maximum freedom for true algorithm discovery.

## [2026-04-12] Tight Training Budget to Force Innovation Beyond Distance
- **Decision**: Use very tight node budget (30-40 nodes) during training instead of generous budgets (120+). Train on low SNR (8-12 dB).
- **Rationale**: With generous budgets, trivial distance ordering achieves BER=0 on training. Verified at max_nodes=120.
- **Alternatives**: More samples; explicit trivial-program penalty.
- **Impact**: Training BER now non-zero, creating genuine pressure for smarter ordering strategies.

## [2026-04-12] Residual Scoring Architecture
- **Decision**: Changed decoder from `score = VM_output` to `score = cum_dist + VM_output`. The evolved program now computes a CORRECTION to the distance metric rather than the full metric.
- **Rationale**: With full-metric scoring, the search converges to `Node.GetCumDist` (len=1) as a strong local optimum. All mutations add noise to a good metric. With residual scoring: (a) zero/inf correction = distance-only baseline (always reasonable), (b) random programs that return inf default to distance rather than breaking ordering, (c) programs that compute small adjustments based on R matrix can improve ordering without needing to replicate the distance computation, (d) the search space for "useful corrections" is much smaller than "good metrics from scratch."
- **Alternatives**: Keep full-metric scoring; add explicit penalty for programs identical to distance.
- **Impact**: DRAMATIC improvement. Immediately: (1) complex multi-instruction programs survive in population (uniq=30/30 vs 13/30), (2) programs use real operations (ForEachAncestor, Mat.ElementAt, Vec.Dot), (3) `Node.GetLocalDist` correction (score=cum_dist+local_dist) found and gives BER=0.03125 vs 0.04167 for pure distance — a genuine 25% improvement at moderate SNR.

## [2026-04-12] Environment Instruction Bias in Random Generation
- **Decision**: Added 30-40% bias towards environment-access instructions (Mat.ElementAt, Vec.Dot, Node.ReadMem, Node.WriteMem, etc.) in random program generation, mutation, and fresh injection.
- **Rationale**: With 77 uniform primitives, the probability of randomly generating matrix/vector operations that access R and y_tilde is very low (~25%). Biasing increases the chance of generating programs that interact with the channel model.
- **Alternatives**: No bias (fully uniform); structured program templates.
- **Impact**: More programs interact with environment data. Combined with residual scoring, enables discovery of non-trivial corrections.

## [2026-04-13] Remove Hardcoded BP — Fully Program-Controlled Architecture
- **Decision**: Complete architecture rewrite: removed all hardcoded BP phases (bottom-up/top-down sweeps, bp_interval, bp_sweeps), removed 4 macro operators (SumChildScores, MinChildScore, GetParentScore, AvgChildMem). Program now runs on every new node during search. dirty_nodes tracking enables program-triggered PQ rebuild.
- **Rationale**: User critique: "BP timing, node selection, stopping must be fully program-controlled. The framework provides NO hardcoded BP schedule. No macro operators that cheat."
- **Alternatives**: Keep hardcoded BP with evolved message update; keep macros as convenience.
- **Impact**: Much harder search space but fully zero-prior. Programs must discover message passing from scratch via Node.SetScore + graph traversal. ~99 instructions (down from 103).

## [2026-04-13] C++ Evaluator for Speed
- **Decision**: Built a complete C++ evaluator DLL (~830 lines) with all ~99 instructions, QR decomposition, stack decoder with dirty tracking. Heap-allocated search tree to avoid stack overflow. Compiled with MSVC /O2 /openmp.
- **Rationale**: Need speed for large populations and node budgets. Python decoder at ~3ms/sample limits scalability.
- **Alternatives**: Cython; numba JIT; pure Python optimization.
- **Impact**: 4x speedup at 100 nodes with simple programs. More for complex programs. Enables larger population/node budget experiments.

## [2026-04-13 03:20] Enforce Ablation-Verified True BP Without Strong Seeds
- **Decision**: Keep the GP strictly from scratch with no strong known detector seeds, and replace the raw `bp_updates` bonus with a no-write ablation test that rewards only BER gains from nonlocal `Node.SetScore` activity.
- **Rationale**: The user explicitly forbids injecting known strong algorithms as seeds. The previous bonus was selecting syntactic BP activity that did not actually improve tree-search quality.
- **Alternatives**: Seed `Float.GetMMSELB` or other strong heuristics; keep rewarding raw score-write counts; remove BP pressure entirely.
- **Impact**: Evolution remains aligned with the AlphaDetect zero-prior requirement, but the selection signal now points toward genuine tree-search-plus-BP behavior rather than fake message passing.

## [2026-04-13] Fix Float.Inv in vm.py — Implementation Was Missing
- **Decision**: Added `Float.Inv` (1/x) to the vm.py PRIMITIVE_INSTRUCTIONS list and dispatch implementation. Previously `Float.Inv` appeared in evolution.py's mutation set (`_BP_AGGREGATES`, `random_bp_pattern`) but was never implemented in vm.py — every call silently no-oped.
- **Rationale**: The bug was discovered during analysis of truebp_1 Gen72: `Float.Inv` inside ForEachChild should have set sibling.score = 1/cum_dist (inverted ordering, bad for ML detection), but instead set sibling.score = cum_dist (accidentally good). The program was optimized by evolution based on accidental behavior of a missing instruction. ANY future program containing Float.Inv would suffer from the same silent no-op deception.
- **Alternatives**: Remove Float.Inv from evolution.py's mutation set (preserve old behavior); implement Float.Inv as expected.
- **Impact**: Validated by Part 4 analysis: with Float.Inv fixed, Gen72_BP achieves BER≈0.996 (random noise) at all SNRs and node budgets, confirming that the accidental no-op was the ONLY reason for Gen72's measured "good" performance. Future evolution now has an honest instruction set — the deceptive local optimum at Gen72 is eliminated.

## [2026-04-13] Plan truebp_2: Harder Training, Larger Dataset, Multi-Budget Fitness
- **Decision**: After truebp_1 converges with a false positive due to (a) Float.Inv no-op bug and (b) 32-sample overfitting, plan truebp_2 with these fixes: (1) Fixed vm.py (Float.Inv works), (2) ≥200 training samples with fresh random seeds each generation, (3) Multi-budget fitness combining BER@200 + BER@2000 nodes, (4) Penalize programs where BER_noBP > 5× BER_MMSE (self-corruption false positives), (5) Consider population diversity injection with explicit restart mechanism.
- **Rationale**: truebp_1 Gen72 demonstrates that a 32-sample fixed training set + broken instruction = program that discovers apparent ratio=0.857 while achieving ratio=1.1-12 on real evaluation. The fitness gap was 4-14× across SNRs. With ≥200 samples and multi-budget eval, this overfitting cannot survive.
- **Alternatives**: Continue truebp_1 by injecting diversity; accept Gen72 as baseline and refine.
- **Impact**: truebp_2 should discover programs that genuinely improve search ordering using channel information, without the deceptive local optima created by the float.Inv bug.

## [2026-04-13] Full-Tree BP Sweeps — Fix Sibling Non-Update Bug
- **Decision**: Rewrote bp_decoder_v2.py with full-tree BP sweeps. Previous version only ran F_up along the ancestor chain of the expanded node (expanded→root). Sibling nodes and their descendants were NEVER updated. New version: after each expansion, runs (1) full up-sweep over ALL tree nodes (leaves→root), (2) full down-sweep over ALL nodes (root→leaves), (3) rescores ALL frontier nodes with F_belief.
- **Rationale**: User identified critical flaw: when node B is expanded (children D,E created), sibling C's score was never updated. Only ancestors (B→A→root) were updated. This violates the fundamental requirement that BP should propagate information to ALL explored nodes.
- **Alternatives**: Only update along ancestor chain + siblings; partial updates with dirty-node tracking.
- **Impact**: Correct BP semantics. Computationally more expensive (O(N_nodes) per expansion vs O(depth) before) but ensures all frontier nodes have consistent beliefs. Added human-readable formula translator for interpretability.

## [2026-04-14 18:00] Fix program_to_formula Multi-Stack Bug
- **Decision**: Rewrote `program_to_formula` in bp_main_v2.py to properly simulate SEPARATE symbolic stacks (float, int, bool) matching the real VM architecture.
- **Rationale**: The old version used a single symbolic float stack. Instructions operating on other stacks (Int.Inc, Int.Sub, Node.GetParent, Mat.Row, Bool.Or, etc.) were pushed as unknown tokens `[Int.Inc]` onto the float stack, corrupting all subsequent operations. This produced completely wrong analysis: e.g., F_down showed `[Node.GetParent]` instead of the correct `(M_par_down + C_i)`, and F_belief showed `max([Int.Sub],[Int.Inc])` instead of the correct `max(M_down, EC1)`.
- **Alternatives**: None — incorrect analysis tool was misleading all genome analysis.
- **Impact**: All previous formula analyses of evolved genomes may have been wrong. The corrected function confirms the gen-4 best genome computes clamped A* (score = max(cum_dist, 0.348)), not the garbled expressions previously reported. The experience-base.md entry was corrected accordingly.

## [2026-04-14 18:00] Launch cpp_test5 with Warm Start and Larger Budget
- **Decision**: cpp_test5 uses max_nodes=1200 (vs 600 in test4), 60 generations (vs 40), warm-started from gen-4 best genome via --seed-genome-json.
- **Rationale**: With max_nodes=1200, the clamped A* heuristic should begin to degrade relative to R-matrix-informed scoring, creating evolutionary pressure for programs that USE the R matrix or multi-iteration BP. The seed genome provides a strong starting point so evolution has more cycles to explore beyond the clamped-A* local optimum.
- **Alternatives**: (1) Start from scratch — but would likely rediscover clamped A* again; (2) Change the architecture — premature until we know if the current GP space can't escape this attractor.
- **Impact**: If test5 finds programs beyond clamped A* (using R matrix, real multi-pass BP, etc.), it validates the architecture. If it stagnates at clamped A* again, we need architectural changes (e.g., force min 2 BP iterations, add MMSE-LB instruction to F_belief).

## [2026-04-17 18:50] Algorithm IR: Make RewriteRegion Primary, Projection Optional
- **Decision**: Updated `research/algorithm-IR/ir_plan.md` so the core manipulation object is `RewriteRegion` with an inferred `BoundaryContract`, while `Projection` is explicitly optional and treated as an interpretation / matching layer rather than the primary rewrite target.
- **Rationale**: The intended workflow is not "IR automatically discovers the true structure" but "the IR remains structure-neutral while users or upper-layer analysis peel off a computation region, preserve its executable boundary, and override it with a donor skeleton." A projection-first design made the role of projection unclear and risked confusing "view" with "rewrite object."
- **Alternatives**: Keep projection as the central object for grafting; adopt a typed high-level skeleton IR that encodes search-tree / message-passing semantics directly in the core IR.
- **Impact**: The plan now centers on region slicing, boundary inference, override plans, and donor lowering back into structure-free IR. This better supports skeleton transplantation without baking in host-side semantic bias, while still allowing optional structural annotations for later matching or NN guidance.

## [2026-04-17 19:25] Algorithm-IR MVP Implemented End-to-End with Stack/BP Grafting Demo
- **Decision**: Implemented the `research/algorithm-IR/algorithm_ir/` package and tests as a structure-neutral, executable MiniIR stack with frontend, interpreter, runtime tracing, shadow store, factgraph, RewriteRegion selection, BoundaryContract inference, optional Projection annotations, and a minimal BP-summary donor grafting pipeline.
- **Rationale**: The plan was intentionally made region-first rather than projection-first. The implementation therefore prioritizes the rewrite closure: compile restricted Python to IR, execute IR, select a rewrite region on a real host algorithm, infer its boundary, graft a donor skeleton, regenerate IR, and execute the rewritten algorithm again.
- **Alternatives**: Stop at static IR / CFG; implement projection discovery before rewrite; postpone grafting until a larger typed semantic layer existed.
- **Impact**: The repository now contains a working `stack_decoder_host` IR, a `bp_summary_update` IR, and an integration test that grafts the BP-like donor into the stack-decoder score region to produce a new executable IR. This creates a concrete substrate for future NN-guided region ranking, richer donor skeletons, and more realistic MIMO detector examples.

## [2026-04-17 20:05] Algorithm-IR README and Demo Outputs Added
- **Decision**: Added a detailed architecture guide at `research/algorithm-IR/algorithm_ir/readme.md` and a reproducible demonstration script at `research/algorithm-IR/demo_outputs.py`.
- **Rationale**: The codebase had become executable, but the architecture was still difficult to understand for non-compiler readers. The README now explains the system in region-first terms, and the demo script prints concrete IR, region, contract, projection, override-plan, and graft-before/after evidence.
- **Alternatives**: Keep explanations in chat only; rely on tests without a human-readable demo script.
- **Impact**: Future work on algorithm transplantation can now point to a stable written explanation and a reproducible evidence script, reducing ambiguity about what the current MVP actually represents and how the BP-into-stack rewrite is carried out.

