# Session Plan — algorithm-IR refactor implementation

**Date**: 2026-04-23
**Goal**: Make GNN produce meaningful (non-loop-severing) grafts; make micro-evolution actually work.

## Scope decision

The full code_review.md plan (S0–S6, ≥240 pool entries, full type-aware GP) is multi-day work.
To deliver actual training-run evidence in this session, I'm implementing the high-leverage subset:

### IN SCOPE this session
- **P0 quick fix**: operators.py — fix insert/delete probability=0 bug
- **P1 pool fixes**: const-lifting (KBEST K, BP max_iters, EP/AMP it<20, damping=0.5),
  replace identity defaults (cavity, damping_none), AST-canonicalize 3× hard_decision
- **P2 pool expansion**: ~50 high-value primitives across categories B/C/D/F
  (not the full 240 — focus on those that materially expand graft target space)
- **P3 random_program rewrite**: tuple/list returns, exec validation
- **P4 micro-pop scale-up**: pop=32, gens=3, fitness EMA
- **P5 FII core**: inline_genome_to_full_ir + provenance + cache
- **P6 GNN feature extension**: provenance hash buckets in node features
- **P7 region enumeration on FII**: with fallback flag for A/B testing
- **P8 graft Case I + II dispatch** (Case III dissolution deferred)
- **P9 background training run + live observation**

### DEFERRED (out of scope this session)
- Full 240-entry pool (we add ~50 high-value primitives instead)
- Case III graft dissolution (rare in early training; defer until needed)
- Slot rediscovery cycle (orthogonal feature)
- Full GP operator set (subtree replace, grafted_callsite cx, etc.) — minimal version only
- Differentiable surrogates (Cat J), sketching (Cat H), calibration (Cat K) — not on critical path

### Gating tests
Each phase ships with a focused test. No phase advances until its test passes.
Final gate: 30-gen training run shows ≥1 effective Case I graft (slot-internal) AND
non-degenerate algorithm in best individual.

## Scope rationale

The user said: "GNN must produce meaningful algorithms; micro-evolution must work."
The minimum-viable change to attack both:
1. FII (P5) — fixes "GNN can only see slot boundaries"
2. operators+random_program+micro-pop (P0/P3/P4) — fixes "micro-evolution dormant"

Pool expansion (P2) provides graft target diversity but partial implementation is sufficient
for the GNN to demonstrate non-degenerate graft selection. Rest is breadth.
