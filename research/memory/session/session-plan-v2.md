# Session Plan v2 — focus on FII (the actual blocker)

**Date**: 2026-04-23 (revised partway through implementation)

## Why narrow scope

User's primary diagnosis: "slot internal structure is invisible to GNN, that's the
key prior". The FII change directly fixes that. Pool expansion (240 entries),
const-lifting, and full GP framework are all secondary — they expand the
search space *for an unblocked GNN*. If the GNN remains blocked, none of
them help.

## Compressed scope (this session)

| Phase | Status | Notes |
|---|---|---|
| P0 operators.py mutate fix | DONE | insert/delete probability now non-zero, smoke test passes |
| P4 micro-pop size bump | NEXT | config-only change in pool_types + train_gnn |
| P5 FII core | NEXT | the central change: inline_genome_to_full_ir + cache + provenance |
| P6 GNN feature extension | NEXT | provenance hash channel in node features |
| P7 region enum on FII | NEXT | switch call site in algorithm_engine + gnn_pattern_matcher |
| P8 graft Case I+II dispatch | NEXT | with Case III rejection (logged) |
| P9 30-gen background run | NEXT | observe non-trivial graft proposals |

## Deferred (next session)

- P1 const-lifting (KBEST K, BP max_iters, EP/AMP it<20, damping=0.5)
- P2 240-entry pool expansion
- P3 random_program rewrite
- P5 Case III dissolution
- Slot rediscovery cycle
- Full GP operator set

If P9 shows the GNN proposing slot-internal regions but performance is
still capped by the small donor library, P1-P3 will be the next session.
