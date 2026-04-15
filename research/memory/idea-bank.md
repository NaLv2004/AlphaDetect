# Idea Bank

Research ideas with status tracking. Ideas progress through:
`proposed` → `exploring` → `coding` → `testing` → `writing` → `completed` or `abandoned`

---

## [ID-001] Typed Stack PushGP Metric for QR Stack Decoder
- **Status**: testing
- **Created**: 2026-04-11
- **Updated**: 2026-04-11
- **Summary**: Evolve a typed PushGP program to score candidate nodes inside a QR-based stack decoder using only low-level algebra, graph, and control primitives under a FLOPs budget.
- **Related**: Experiment 2026-04-11 Full 30-Generation 16x8 Evaluation; Decision 2026-04-11 Remove privileged fallback from bounded search

## [ID-002] Sibling-Aware Relational Scoring
- **Status**: exploring
- **Created**: 2026-04-11
- **Updated**: 2026-04-11
- **Summary**: Let the evolved metric score a node relative to its siblings and the current frontier through generic tree primitives so it can invent ambiguity-aware branch selection rather than just reuse cumulative distance.
- **Related**: Experiment 2026-04-11 Full 30-Generation 16x8 Evaluation; Decision 2026-04-11 Keep zero-prior primitive seeding but harden curriculum

## [ID-003] Hard Anytime Curriculum for Budgeted Search
- **Status**: testing
- **Created**: 2026-04-11
- **Updated**: 2026-04-11 03:33
- **Summary**: Train on tighter node budgets, independent holdout data, and harder 32x16 16-QAM cases so evolution must discover rankings that matter before exhaustive completion becomes possible.
- **Related**: Experiment 2026-04-11 Full 30-Generation 16x8 Evaluation; Experiment 2026-04-11 03:33 Holdout/Tight-Budget 32x16 16-QAM Dynamic Metric Discovery; Insight 2026-04-11 03:33 Holdout and Tight-Budget Pressure Reveal Real Dynamic Structure

## [ID-004] Ancestor Writeback Residual Memory
- **Status**: abandoned
- **Created**: 2026-04-11 03:33
- **Updated**: 2026-04-12
- **Summary**: Propagate rescoring deltas up the ancestor chain into generic state slots so later descendant evidence can revise earlier branch priorities through explicit residual memory rather than only passive subtree summaries.
- **Reason abandoned**: Full v2 rewrite removed all coarse TreeNode statistics and ancestor writeback. The new approach uses fine-grained operators with N_MEM=8 writable memory slots per node, so memory-based belief propagation must emerge naturally from the GP.
- **Related**: Decision 2026-04-11 03:33

## [ID-005] Anti-Distance-Attractor Mechanisms for From-Scratch GP
- **Status**: testing
- **Created**: 2026-04-12
- **Updated**: 2026-04-12
- **Summary**: Overcome the Node.GetCumDist local optimum in from-scratch GP search via multiple diversity mechanisms: (1) 20% fresh random injection per generation with varied program sizes, (2) `mutate_grow` operator adding 2-8 instructions at once, (3) 30-40% env_bias favoring R-matrix/vector access instructions, (4) per-sample lexicase selection, (5) fixed training datasets per epoch for stable fitness landscape, (6) very low SNR training (6/8/10 dB) to maximize distance-ordering failure rate.
- **Key insight**: Single-instruction distance metric is a strong local optimum because mutations of 1-instruction programs can't grow them fast enough. The `mutate_grow` and `mutate_segment` operators for short programs enable larger structural jumps.
- **Related**: Experience 2026-04-12 Distance Metric Attractor

## [ID-006] Ablation-Verified Nonlocal BP Selection
- **Status**: testing
- **Created**: 2026-04-13 03:20
- **Updated**: 2026-04-13 03:20
- **Summary**: Evaluate each candidate both with normal score-write propagation and with `Node.SetScore` disabled, so evolution rewards only nonlocal message passing that actually improves BER under the same bounded tree search.
- **Related**: Decision 2026-04-13 03:20 Enforce Ablation-Verified True BP Without Strong Seeds; Experiment 2026-04-13 03:20 True-BP Fitness Refactor and From-Scratch Relaunch

## [ID-007] truebp_2: Hardened Fitness to Prevent Self-Corruption False Positives
- **Status**: proposed
- **Created**: 2026-04-13
- **Updated**: 2026-04-13
- **Summary**: Design truebp_2 with multiple improvements over truebp_1 to prevent the Gen72 false positive from recurring:
  1. **BER_noBP corruption guard**: If BER_noBP > 5× BER_MMSE, penalize fitness heavily (or disqualify). This prevents programs that corrupt the search space and then "fix" it with BP writes.
  2. **Larger training set**: ≥100 fresh random samples per generation (epoch rotation). Prevents 32-sample overfitting that gave Gen72 training ratio=0.857 vs eval ratio=1.1-1.6.
  3. **Multi-budget secondary eval**: Log BER at both 200 AND 2000 nodes when reporting fitness. Flag programs where ratio gets worse at 2000 nodes (scaling failure = not a genuine correction).
  4. **Instruction completeness test**: Add a unit test that verifies every instruction in evolution.py ALL_INSTRUCTIONS actually executes and produces a distinct result in the VM. Prevents Float.Inv-style silent no-ops.
  5. **BER_BP ≤ BER_MMSE requirement**: Only reward programs that are actually better than MMSE-LB, not just better than the self-corrupted noBP version.
- **Motivation**: truebp_1 Gen72 converged to a local optimum created by Float.Inv no-op bug + 32-sample overfitting. With these fixes, only genuine algorithm improvements survive.
- **Related**: Experience 2026-04-13 Float.Inv unimplemented; Experience 2026-04-13 High bpg = self-corruption; Experience 2026-04-13 pure-A* doesn't scale; Decision 2026-04-13 Fix Float.Inv in vm.py

## [ID-008] Instruction Set Sanity Tests
- **Status**: proposed
- **Created**: 2026-04-13
- **Updated**: 2026-04-13
- **Summary**: Add a Python unit test module that: (1) verifies every string in ALL_INSTRUCTIONS executes without no-op in the VM, (2) checks every instruction in evolution.py's mutation lists exists in ALL_INSTRUCTIONS, (3) verifies each instruction produces a stack change (not silent no-op) under controlled input. This prevents recurrence of the Float.Inv-style bugs where an instruction name exists in the grammar but is not implemented.
- **Related**: Experience 2026-04-13 Float.Inv unimplemented; Decision 2026-04-13 Fix Float.Inv

## [ID-009] Structured BP Ablation Studies to Identify True Contributions
- **Status**: completed
- **Created**: 2026-04-14
- **Updated**: 2026-04-14 18:00
- **Summary**: Ablation study completed (200 trials). Results:
  - A1 (F_up=0): BETTER than baseline → F_up is dead weight
  - A2 (F_down passthrough): catastrophe (BER≈0.93) → F_down cumulative distance is essential
  - A3 (no BP sweeps): much worse → but only because m_down never computed, not true BP value
  - A4 (F_belief=D_i): slightly worse → EC1 floor helps ~10%
  - **Conclusion**: Evolved genome = clamped A* (score=max(cum_dist,0.348)). No genuine BP.
- **Related**: Experiment 2026-04-14 cpp_test4 Full Run

## [ID-010] Curriculum Learning for Structured BP Discovery
- **Status**: proposed
- **Created**: 2026-04-14
- **Updated**: 2026-04-14
- **Summary**: The structured BP system stagnates at ratio≈0.459 after gen 4. The local optimum is a "distance + local BP aggregation" program that outperforms LMMSE but doesn't do true global BP. To escape this:
  1. **Phase 1** (current): Train at SNRs 10-14 with 600 max_nodes. Find initial good programs.
  2. **Phase 2**: After gen 30, increase to 1200 max_nodes and change LMMSE ratio pressure. Programs that fail to improve at larger budgets get penalized.
  3. **Phase 3**: Multi-budget fitness: f = W1*BER@600nodes + W2*BER@2000nodes. Programs must do true BP to win at 2000 nodes.
  4. **Progressive difficulty**: Add harder channels (ill-conditioned H) that specifically punish distance-only orderings.
- **Technical note**: The gap between snr=10 and snr=16 performance is key. A pure-distance program degrades at high SNR with large node budget (based on earlier truebp experiments). True BP should show consistent improvement with more nodes.

## [ID-011] Hybrid DSL: Typed Search + Structured BP
- **Status**: proposed
- **Created**: 2026-04-14
- **Updated**: 2026-04-14
- **Summary**: The structured BP framework (4 programs) is limited because it assumes a fixed architectural pattern. A more powerful AlphaDetect approach: define a DSL where programs can express ARBITRARY message flow over the partial search tree. This connects to the original AlphaDetect vision (DSL composition for MIMO detection).
  - Core DSL: nodes, edges, messages, aggregation primitives
  - Template: MAP(nodes, f(node, parent_msg, sibling_msgs, children_msgs)) → priority
  - This allows: (a) different aggregation at different layers, (b) different weighting based on channel quality, (c) adaptive number of BP iterations
  - Could be encoded as a higher-level program that generates the 4 sub-programs
- **Connection to AlphaDetect vision**: This is the "formal reasoning about message passing" that the research proposal envisions.
- **Risk**: Large search space, may need guided search or grammar constraints.

