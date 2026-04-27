# Major Refactor Plan — Annotation-Only Slot Architecture

**Status**: Plan only — no code changes yet.
**Authors / drivers**: User design directive after rejecting fallback patches in `slot_evolution.py`.
**Anchor commit**: `a05ebce` (S1–S7 IR remediation, pushed). The uncommitted patch in `evolution/slot_evolution.py` (`_passthrough_donor_arg`, `_NON_DATA_TYPE_HINTS`, modified `pick_real_exit_value`) MUST be reverted (`git checkout evolution/slot_evolution.py`) before M1 begins, or as the first action of M1.

---

## 0. TL;DR

The current framework reverse-engineers slot regions from FII (Fully-Inlined IR) provenance tags after the fact. SSA def-use heuristics (`pick_real_exit_value`, `_augment_variant_multi_exit`, `collect_slot_region`'s alias-snapshot adjustments) try to recover boundary information that was thrown away during inlining. This is the root cause of the 42 / 221 silent-`None` failures observed in the v7 default re-apply audit and of the chronic eval-stage runtime exceptions.

The refactor changes ONE thing — but it cascades:

> **Slot boundaries are declared at the source level via a `with slot(name, inputs=(...), outputs=(...)):` context manager. The IR builder records both per-op `slot_id` tags AND explicit `slot_meta` boundary metadata as it compiles. Slot mutation becomes a deterministic op-set swap with the boundary already known.**

Consequences:

- **DELETE**: `evolution/fii.py`, `evolution/gp/region_resolver.py`, `evolution/slot_discovery.py` (replaced by Track B GNN slot-stamping), the AlgSlot dialect op, and the boundary-recovery heuristics in `evolution/slot_evolution.py` that depend on FII tags (`pick_real_exit_value`, `_augment_variant_multi_exit`, etc.).
- **REWRITE**: `evolution/skeleton_library.py` (DSL change), `evolution/ir_pool.py` (drop helper compilation + AlgSlot conversion), `evolution/slot_evolution.py` (deterministic op-swap for Case I micro-evolution).
- **KEEP & REWORK** (NOT delete): `algorithm_ir/grafting/graft_general.py` (still needed for Case II/III general grafting), `evolution/slot_rediscovery.py` → renamed/folded into Track B (Case III dissolve still runs the dissolve half).
- **EXTEND**: `algorithm_ir/frontend/ir_builder.py` (`with` statement support, nested slots), `algorithm_ir/ir/model.py` (`FunctionIR.slot_meta` field with parent links), `evolution/gnn_pattern_matcher.py` (free-form region sampling + coupled host→donor type-mask + post-hoc Case I/II/III classification).
- **KEEP UNCHANGED**: GP operators, R6 behavior gate, R7 telemetry, subprocess evaluator, type lattice, validator, fitness, materialize, codegen (minor cleanup only).

---

## 1. Architectural Overview

### 1.1 Core invariants (post-refactor)

1. **Single flat IR.** `AlgorithmGenome.ir` is the only canonical IR. No inlined view, no AlgSlot view, no helper-decomposed view.
2. **Slot membership is a per-op tag, slots may nest.** `op.attrs["slot_id"]` carries the **innermost** enclosing slot's pop_key (set by `ir_builder` during `with slot(...)` compilation). Outer-slot membership is recovered by walking `slot_meta[k].parent` upward.
3. **Slot boundary is explicit metadata, with parent links.** `FunctionIR.slot_meta: dict[str, SlotMeta]` is populated at compile time. `SlotMeta` carries the declared inputs (value IDs visible at slot entry), outputs (value IDs the slot must produce), the slot population key, and the **parent slot pop_key (or None)**. Slots form a forest.
4. **No FII, no inlining, no helper functions in the genome IR.** Templates are written as a single function whose body contains `with slot(...)` blocks (which may be nested). They compile to one `FunctionIR` directly — no `_slot_xxx` helpers ever exist as IR.
5. **Slot micro-mutation (Case I) is an op-set swap.** Given `slot_meta[pop_key]` and a variant subgraph (op-list with boundary placeholders), apply = delete all ops with matching `slot_id` + insert variant ops + wire variant inputs to declared `slot_meta.inputs` + redirect downstream consumers. No graft, no contract inference, no fallback.
6. **GNN graft (Case I/II/III) keeps three-case routing.** GNN samples a region freely on host (NOT snapped to slot boundaries). The region is post-hoc classified into Case I (region attributable to a slot), Case II (region purely structural), or Case III (region half-cuts at least one slot — touched slots get dissolved before graft). Donor sampling is **coupled**: GNN samples donor entry/exit values under a type-mask derived from the host region's boundary signature, so only type-compatible donors can be proposed.
7. **Track B (slot stamping) is a separate emission mode.** Same `BoundaryHead` run on a single IR proposes "this op-set deserves to be a slot". Orchestrator stamps `slot_id` tags + `slot_meta` entry. Replaces both `slot_discovery.py` and the post-graft re-tagging half of `slot_rediscovery.py`.

### 1.2 Why the old design failed

| Failure mode | Root cause |
|---|---|
| 42 silent-`None` re-applies | `pick_real_exit_value` picked module callables (np.real, len) as exit values because FII inlining produced `load_var(np.real)` ops whose consumers crossed the slot boundary. |
| Multi-exit alias snapshots leaked | `_augment_variant_multi_exit` aborted all-or-nothing because FII produces alias assigns like `__fii_ep_t_1 = v_t` whose outputs feed downstream phis — tagging them as exits broke arity. |
| Const-None dangling-input repair | `graft_general` had no way to know which donor inputs corresponded to which host values, so it filled with `const None` when type-binding failed. Const-None feeds ran-time NameError/TypeError. |
| Region resolver tier proliferation | 3-tier resolver (provenance / binding / callee_name) exists because no single signal is reliable when slot identity has been smeared across inlining. |

The unifying lesson: **boundary information must travel with the slot from declaration time. Reverse engineering it from SSA is fundamentally lossy.**

### 1.3 The new authoring DSL — by example

Old (current) — slot is a function parameter, body calls it like a helper:

```python
def detector_ep(H, y, sigma2, constellation, n_iter,
                slot_cavity, slot_site_update, slot_final_decision):
    # ... init ...
    while it < n_iter:
        i = 0
        while i < Nt:
            cav_t, cav_h2 = slot_cavity(t, h2, gamma_i, alpha_i)
            new_gamma, new_alpha = slot_site_update(cav_t, cav_h2, ...)
            # ... etc
    return slot_final_decision(t, constellation)
```

New — slot is an inline `with` block declaring its boundary:

```python
def detector_ep(H, y, sigma2, constellation, n_iter):
    # ... init ...
    while it < n_iter:
        i = 0
        while i < Nt:
            with slot("ep.cavity",
                      inputs=(t, h2, gamma_i, alpha_i),
                      outputs=("cav_t", "cav_h2")):
                cav_t = t / (1.0 - h2 * alpha_i)   # default body
                cav_h2 = h2 / (1.0 - h2 * alpha_i)
            with slot("ep.site_update",
                      inputs=(cav_t, cav_h2, sigma2, constellation),
                      outputs=("new_gamma", "new_alpha")):
                # ... default site update ...
            i = i + 1
    with slot("ep.final_decision",
              inputs=(t, constellation),
              outputs=("x_out",)):
        x_out = hard_decision_inline(t, constellation)
    return x_out
```

The `with slot(...)` form is interpreted ONLY by `ir_builder` — at runtime `slot` is an inert context-manager stub (no-op enter/exit) so the source remains directly executable for tests / debugging.

---

## 2. Files to DELETE

| Path | Reason |
|---|---|
| `evolution/fii.py` (~600 LOC) | Entire FII pipeline (`build_fii_ir_with_provenance`, marker injection, AST inliner, provenance map, marker scrub) is obsolete: no inlining is needed. |
| `evolution/gp/region_resolver.py` | 3-tier resolver replaced by one-liner `[oid for oid, op in ir.ops.items() if op.attrs.get("slot_id") == pop_key]`. |
| `evolution/slot_discovery.py` | Static SESE backward-slice heuristic is superseded by Track B GNN slot-stamping (§6.2). |
| `algorithm_ir/ir/dialect.py::AlgSlot` | The xDSL `alg.slot` op disappears. Pool no longer converts slot calls to AlgSlot. (Other dialect ops kept; only `AlgSlot` removed from the bundle list.) |

Files **NOT** deleted (revised from earlier draft):

- `evolution/slot_rediscovery.py` — the **dissolve half** (Case III) is still needed; the **re-tagging half** is replaced by Track B. Rename to `evolution/slot_dissolve.py` and keep only the dissolve helper; the re-tag heuristics go away.
- `algorithm_ir/grafting/graft_general.py` — kept (Case II and Case III still need general grafting). Reworked, not deleted: see §3.7.
- `algorithm_ir/region/contract.py`, `region/selector.py` — kept; `RewriteRegion` / `BoundaryContract` are still the carrier types between GNN and graft. Drop only the heuristic SSA inference helpers that are no longer called.

Plus large excisions inside files (kept overall):

| File | Functions / sections to delete |
|---|---|
| `evolution/slot_evolution.py` | `map_pop_key_to_from_slot_ids`, `collect_slot_region`, `pick_real_exit_value`, `_value_label_candidates`, `_augment_variant_multi_exit`, `_passthrough_donor_arg`, `_NON_DATA_TYPE_HINTS`, `_is_data_exit`. Telemetry counters that refer to graft/validator failures simplify. |
| `algorithm_ir/grafting/graft_general.py` | The dangling-input **const-None fill** path (the chief source of runtime NameErrors). Typed-binder fallback to module-name binding. `from_slot_id`-aware site mapping. **KEEP** the donor↔host op-cloning + SSA renaming + consumer rewiring — that's the shared core for Case II / Case III. |
| `algorithm_ir/region/contract.py` | `infer_boundary_contract` SSA-scan heuristics — boundary is now always supplied (either by `slot_meta` for Case I, or by GNN-sampled cuts for Case II/III). Keep `BoundaryContract` data class. |
| `evolution/ir_pool.py` | `_template_globals` keeps only safe-math helpers; `compile_slot_default`, `convert_slot_calls_to_algslot`, `_DETECTOR_SPECS.slot_default_keys` go away (slot bodies live inline). |

---

## 3. Files to REWRITE (substantial)

### 3.1 `evolution/skeleton_library.py`

- **Action**: Rewrite every detector template source string in the new `with slot(...)` form.
- **Detail**: For each of LMMSE / ZF / OSIC / K-Best / Stack / BP / EP / AMP / and the ~80 extended skeletons, replace the `def detector_xxx(..., slot_a, slot_b)` signature with `def detector_xxx(...)` and convert each `result = slot_a(args)` (or `r1, r2 = slot_a(args)`) call site into a `with slot("algo.a", inputs=(args), outputs=("result" | "r1","r2")): <inline default body>` block.
- **`SkeletonSpec`**: remove `slot_arg_names` and `slot_default_keys` fields. Slot identity now comes purely from the `with slot(...)` declarations parsed by `ir_builder`. Keep `algo_id`, `source`, `func_name`, `slot_defs` (still needed for ProgramSpec / boundary-type metadata), `tags`, `level`, `extra_globals`.
- **`EXTENDED_SLOT_DEFAULTS`**: deleted — defaults now live inline in the `with` body of each template.
- **Risk**: bp.bp_sweep currently has nested loops over messages; the inline body must preserve that. Multi-block helpers (sa.accept, mh.accept) similarly must stay inside their `while`/`if` containers — `with slot(...)` may appear at any nesting level.

### 3.2 `algorithm_ir/frontend/ir_builder.py`

- **Action**: Add `ast.With` handling to `_compile_stmt` dispatch (currently raises `NotImplementedError`).
- **Detail**:
  1. Verify the With has exactly one `withitem` whose `context_expr` is an `ast.Call` to a name `slot`.
  2. Parse the call: `slot(<str_literal_name>, inputs=(<Name>,...), outputs=(<str>,...))`. Reject anything else with a clear compile error (no fallback).
  3. Resolve each `inputs` Name to its current SSA value-id via `state.name_env`. These are the slot's `entry_value_ids`.
  4. Push `(slot_name, entry_value_ids, declared_output_names)` onto a `state.slot_stack`.
  5. Compile body statements as normal. Inside the active slot, every emitted op gets `op.attrs["slot_id"] = slot_name`. (Achieve via a single tag pass at slot-exit, or via a stack-aware `_emit_op` wrapper — implementation choice; prefer the wrapper for locality.)
  6. After body compilation, look up each declared output name in `state.name_env` to obtain `exit_value_ids`.
  7. Pop the stack; record `func_ir.slot_meta[slot_name] = SlotMeta(pop_key=slot_name, op_ids=tuple(<tagged ops>), inputs=entry_value_ids, outputs=exit_value_ids, output_names=declared_output_names)`.
- **Nested slots**: support stack-style nesting; outer slot's `op_ids` includes inner slot's ops; inner slot is a finer-grained subset. (BP's `bp_sweep` outer / per-message inner is the canonical case.)
- **Validation**: if any declared output name is not bound at slot exit, raise — this is a template authoring bug and must surface immediately, not silently None-fill.

### 3.3 `algorithm_ir/ir/model.py`

- **Action**: Add `slot_meta: dict[str, SlotMeta]` field to `FunctionIR` (default factory `dict`) and define the `SlotMeta` dataclass with parent link for nesting support.

```python
@dataclass
class SlotMeta:
    pop_key: str
    op_ids: tuple[str, ...]          # all ops whose innermost-enclosing slot is this one;
                                     # nested slot's ops are NOT included here — recover
                                     # full membership via children-walk in slot_meta
    inputs: tuple[str, ...]          # value-ids visible at slot entry
    outputs: tuple[str, ...]         # value-ids the slot must produce
    output_names: tuple[str, ...]    # source-level names, length == len(outputs)
    parent: str | None = None        # parent slot pop_key (None at top level)
```

  Helper API on `FunctionIR`:
  ```python
  def slot_full_op_ids(self, pop_key: str) -> set[str]:
      """All ops belonging to this slot OR any descendant slot (transitive)."""
      result = set(self.slot_meta[pop_key].op_ids)
      for k, m in self.slot_meta.items():
          if self._is_descendant(k, pop_key):
              result.update(m.op_ids)
      return result

  def slot_children(self, pop_key: str | None) -> list[str]:
      return [k for k, m in self.slot_meta.items() if m.parent == pop_key]
  ```

- **clone()**: deep-copy `slot_meta` alongside ops/values.
- **Validator**: `validate_function_ir` learns to verify `slot_meta` consistency:
  - every op tagged `slot_id=K` is in `slot_meta[K].op_ids` OR in some descendant of K (innermost-tag invariant);
  - every input value-id in `slot_meta[K].inputs` is defined OUTSIDE `slot_full_op_ids(K)` (or is a func arg);
  - every output is defined INSIDE `slot_full_op_ids(K)`;
  - parent chain has no cycles.

### 3.4 `evolution/ir_pool.py`

- **Action**: Drop helper-compilation pipeline entirely.
- **New `build_ir_pool` flow** per detector spec:
  1. `compile_source_to_ir(spec.source, spec.func_name, _template_globals())` → flat `FunctionIR` whose `slot_meta` is already populated by `ir_builder`.
  2. For each `(pop_key, meta)` in `ir.slot_meta.items()`, build a `SlotPopulation`:
     - `slot_id = pop_key`
     - `spec = spec.slot_defs[pop_key]` (ProgramSpec for type info)
     - `variants = [SubgraphSnapshot.from_ir(ir, meta)]` — exactly one variant: the default body taken from the IR itself.
     - `fitness = [float("inf")]`, `best_idx = 0`.
- **Drop**: `compile_slot_default`, `convert_slot_calls_to_algslot`, all `_template_globals` slot-call shims, `_DETECTOR_SPECS.slot_default_keys`, `_DETECTOR_SPECS.slot_arg_names`.
- **Keep**: tag-set and detector spec tags, extra_globals plumbing, ID allocation helpers (`_next_prefixed_id`, `_value_payload`, `_append_xdsl_op`).

### 3.5 `evolution/pool_types.py`

- **`SlotPopulation.variants`**: change type from `list[FunctionIR]` to `list[SubgraphSnapshot]`. A `SubgraphSnapshot` is a self-contained, IR-agnostic representation of a slot body (no host-IR pointers). Definition:

```python
@dataclass(frozen=True)
class SubgraphSnapshot:
    ops: tuple[Op, ...]                     # deep-copied; placeholder input value-ids `__in_<i>`, output `__out_<j>`
    input_placeholders: tuple[str, ...]     # length == len(slot.inputs); ordered to match SlotMeta.inputs
    output_placeholders: tuple[str, ...]    # length == len(slot.outputs)
    output_names: tuple[str, ...]
    type_signature: tuple[tuple[str, ...], tuple[str, ...]]  # input type-hints, output type-hints
```

The snapshot's ops reference value-ids local to the snapshot only; concrete host value-ids are bound at apply time. Memory cost is bounded by the slot body, not the whole IR.

### 3.6 `evolution/slot_evolution.py`

The bulk is rewritten as a single deterministic algorithm:

```python
def apply_slot_variant(genome: AlgorithmGenome,
                       pop_key: str,
                       variant: SubgraphSnapshot) -> FunctionIR | None:
    """Splice variant into a deep copy of genome.ir. No fallback."""
    ir = deepcopy(genome.ir)
    meta = ir.slot_meta.get(pop_key)
    if meta is None:
        return None  # slot unknown — only legitimate failure
    # 1. Validate signature compatibility (arity + lattice subtype).
    if not _signatures_compatible(ir, meta, variant):
        return None
    # 2. Delete tagged ops (and their unique outputs) inside the slot.
    _delete_ops(ir, set(meta.op_ids))
    # 3. Insert variant ops with fresh op-/value-ids; build placeholder→host map.
    in_map = dict(zip(variant.input_placeholders, meta.inputs))
    out_map = _insert_variant(ir, variant, in_map, slot_id_tag=pop_key)
    # 4. Redirect downstream consumers: every old slot output value
    #    is replaced by the corresponding variant output value.
    _rewire_consumers(ir, old=meta.outputs, new=tuple(out_map[p] for p in variant.output_placeholders))
    # 5. Update slot_meta entry with the new op_ids / output value-ids.
    ir.slot_meta[pop_key] = SlotMeta(
        pop_key=pop_key,
        op_ids=tuple(out_map[op.id] for op in variant.ops if op.id.startswith("op_")),
        inputs=meta.inputs,           # entry values unchanged
        outputs=tuple(out_map[p] for p in variant.output_placeholders),
        output_names=meta.output_names,
    )
    if validate_function_ir(ir):  # cheap structural check only
        return None
    return ir
```

- `step_slot_population`, `evaluate_slot_variant` keep their public signatures but lose all heuristic branches. `SlotMicroStats` keeps the high-level counters; the boundary-recovery counters (`skipped_no_sids`, `n_apply_validator_failed`, etc.) are simplified — failure now means signature mismatch, structural validation, or eval-stage runtime failure only.
- `commit_best_variants_to_ir` becomes a no-op or trivial wrapper: `genome.ir = apply_slot_variant(genome, pop_key, pop.variants[pop.best_idx])` per slot in topological order. (Topological because nested slots must apply outer-then-inner or vice versa consistently — define and document the rule.)

### 3.7 `algorithm_ir/grafting/graft_general.py`

**Kept and reworked.** Used by Case II and Case III GNN grafts (Case I micro-evolution uses `apply_slot_variant` from §3.6; it never calls `graft_general`).

- **Drop**: dangling-input **const-None fill** (this was the chief source of runtime NameErrors). Drop typed-binder fallback to module-name binding. Drop `from_slot_id`-aware site mapping.
- **Keep**: the core surgery — clone donor ops into host with fresh SSA value-ids, rebind donor placeholder inputs to host-side entry values supplied by the proposal, redirect host downstream consumers from old exit values to new exit values, run `validate_function_ir`.
- **New contract**: caller passes an explicit `BoundaryContract` listing host entry values, host exit values, donor entry values (placeholders), donor exit values (placeholders). For Case I this contract comes from `slot_meta[pop_key]` (with donor side coming from a donor-side `slot_meta` entry or from a fresh GNN sample); for Case II/III it comes directly from GNN cut sampling on both sides. **Arity mismatch or type-lattice incompatibility is a hard rejection** — no fill, no fallback.

---

## 4. Files to MODIFY (small to medium)

| File | Change |
|---|---|
| `algorithm_ir/regeneration/codegen.py` | Drop any code generating `slot_xxx(args)` call expressions. With slot bodies inlined into the IR, codegen just emits the flat function. Ignore `slot_id` attrs (or emit `# slot: <name>` comments for debuggability). |
| `evolution/materialize.py` | `materialize` becomes "emit Python source from `genome.ir`" — no helper namespace, no override-with-slot-variants pathway. `materialize_with_override(genome, overrides)` becomes "for each `(pop_key, variant)` in overrides, call `apply_slot_variant`, then emit". `_default_exec_namespace` keeps only numpy / math / TreeNode. |
| `evolution/algorithm_engine.py` | `_micro_evolve` keeps its contract but operates on the new `SlotPopulation` whose variants are `SubgraphSnapshot`. Remove FII-related branches, `use_fii_view` plumbing in `to_entry`. |
| `evolution/gnn_pattern_matcher.py` | (a) Skeleton-graft path: same as before, just calls the simplified `graft_general`. (b) Slot-discovery extension: when proposing a new slot from a host IR, output `(op_ids, input_value_ids, output_value_ids, pop_key)`. The orchestrator stamps `slot_id` tags + adds `slot_meta` entry; no inference. |
| `train_gnn.py` | Telemetry log fields renamed/dropped: `n_apply_graft_failed` → `n_apply_signature_failed`, `n_validate_failed` covers structural validation only. |
| `algorithm_ir/ir/dialect.py` | Remove `AlgSlot` from imports & `AlgDialect` bundle. |
| `algorithm_ir/ir/validator.py` | Add slot_meta consistency check (see §3.3). Drop any `_provenance.from_slot_id` validations. |
| `evolution/gp/contract.py` | If `SlotContract` is referenced by the GP individual / lineage modules, simplify to a thin alias of `SlotMeta`. Otherwise delete. |
| `evolution/gp/individual.py`, `gp/lineage.py`, `gp/canonical_hash.py` | Adjust any references to `_provenance` / `from_slot_id` / `region_resolver` to use `slot_id` attr / `slot_meta` directly. Canonical hash should incorporate `slot_meta` so cross-slot mutations are distinguishable. |

---

## 5. Files to KEEP UNCHANGED

- `evolution/subprocess_evaluator.py`, `evolution/_eval_worker.py` — both operate on plain Python source via `evaluate_source_returning_xhat`; they are slot-agnostic.
- `evolution/fitness.py`, `evolution/mimo_evaluator.py` — score/eval logic unchanged.
- `evolution/gp/operators/{base.py, typed_mutations.py, structural.py}` — all 11 GP operators consume an op subgraph and produce an op subgraph. They are already snapshot-friendly. Minor: ensure they tolerate the new `SubgraphSnapshot` representation in/out.
- `evolution/gp/population.py` — `micro_population_step` and R6 behavior gate untouched. R6 still compares `xhat_parent` vs `xhat_child` from `evaluate_source_returning_xhat`.
- `algorithm_ir/ir/type_lattice.py`, `type_info.py`, `xdsl_bridge.py` — unchanged.
- `algorithm_ir/runtime/`, `analysis/`, `factgraph/`, `projection/`, `visualize/` — unchanged.
- `algorithm_ir/region/slicer.py` (backward/forward slice utilities) — kept; used by GNN slot discovery only.

---

## 6. GNN-Assisted Graft Pipeline (Case I / II / III preserved)

**Verdict: KEEP the GNN architecture (encoder, PairScorer, BoundaryHead, RL training loop). REWORK the proposer pipeline so that (a) host region is sampled FREELY without snapping to slot boundaries, (b) donor region is sampled UNDER A TYPE-MASK derived from the host region's boundary signature (coupled autoregressive proposal), (c) the resulting region is post-hoc CLASSIFIED into Case I / II / III, and (d) each case routes to its appropriate apply path. Track B (slot stamping) is a separate emission mode that operates on a single IR.**

### 6.1 The three preserved cases

Definitions (unchanged from the original `slot_dissolution.py` design, but re-anchored on the new `slot_meta` data model):

| Case | Topological condition on `R = host_region.op_ids` vs `host.slot_meta` | Apply path |
|---|---|---|
| **Case I** | `R` does not half-cut any slot, AND there exists a non-empty set `F = { S : slot_full_op_ids(S) ⊆ R }` (i.e. `R` fully contains at least one slot, possibly equal to `R`); OR `R` is fully contained in some slot's `slot_full_op_ids`. | Pick the **most specific attribution slot** per §6.4. The graft is applied via the slot-aware path, AND the donor subgraph is appended to that slot's `SlotPopulation.variants` so future micro-evolution can re-pick it. |
| **Case II** | `R` does not intersect any slot at all (purely structural). | Plain `graft_general`. `slot_meta` unchanged. Optionally: orchestrator may invoke Track B on the post-graft IR to discover whether the new structural region deserves to be stamped as a slot. |
| **Case III** | `R` half-cuts at least one slot S (i.e. `R ∩ slot_full_op_ids(S)` is non-empty AND not all of S). | **Dissolve** every half-cut slot S (clear `op.attrs["slot_id"]` for all of S's ops, drop `slot_meta[S.pop_key]`, archive `slot_populations[S.pop_key]` to lineage history). Then proceed as Case II. Dissolve is intentional and not reversible in the same generation; Track B may re-stamp later. |

**No "snap region to nearest slot boundary" path exists.** GNN's freedom to propose arbitrary regions is preserved.

### 6.2 Pipeline (sequential, host-conditioned donor)

```
┌─────────────────────────────────────────────────────────────────┐
│ Step 0  PairScorer samples (host_genome, donor_genome) pair.    │
├─────────────────────────────────────────────────────────────────┤
│ Step 1  Free-form host_region sampling.                         │
│         encode host_ir → BoundaryHead.cut_logits/output_logits  │
│         on ALL host values; sample entry_values + exit_values   │
│         (no slot_meta mask). Build RewriteRegion R from the     │
│         backward/forward slice intersection.                    │
├─────────────────────────────────────────────────────────────────┤
│ Step 2  Compute boundary signature of R.                        │
│         BoundarySignature(                                      │
│           entry_types : ordered tuple of normalized type-hints  │
│                         from R.entry_values,                    │
│           exit_types  : ordered tuple of normalized type-hints  │
│                         from R.exit_values,                     │
│         )                                                       │
├─────────────────────────────────────────────────────────────────┤
│ Step 3  Coupled donor sampling under type-mask.                 │
│         build_donor_mask(donor_ir, signature) → mask_entry,     │
│                                                  mask_exit      │
│         For each entry slot in signature.entry_types (ordered): │
│           logits = head.cut_logits(donor_emb, donor_value_feats,│
│                                    mask=mask_entry_step_i)      │
│           sample one donor value of the required type.          │
│         (ditto exits). Build donor RewriteRegion from the       │
│         sampled values. If any step has all-zero mask → ABORT   │
│         this proposal as a negative training sample (no         │
│         fallback, no near-miss).                                │
├─────────────────────────────────────────────────────────────────┤
│ Step 4  Classify host_region against host.slot_meta.            │
│         classify_region(R, host.slot_meta) → (case, attrib_key) │
├─────────────────────────────────────────────────────────────────┤
│ Step 5  Route by case (Case I/II/III, see §6.1 + §6.5).         │
├─────────────────────────────────────────────────────────────────┤
│ Step 6  validate_function_ir on the post-apply IR; if invalid,  │
│         reject + record negative reward.                        │
└─────────────────────────────────────────────────────────────────┘
```

Key contrast vs the old pipeline:
- Old: host and donor regions sampled independently, then `infer_boundary_contract` tries to align them post-hoc, with const-None fill on mismatches.
- New: donor is conditioned on host's boundary signature from the start; mismatches are impossible by construction (or the proposal is aborted).

### 6.3 Boundary type-mask (Step 3 detail)

```python
@dataclass(frozen=True)
class BoundarySignature:
    entry_types: tuple[str, ...]   # length = len(host_region.entry_values)
    exit_types:  tuple[str, ...]   # length = len(host_region.exit_values)

def build_donor_mask(donor_ir, sig, step, port_kind):
    """port_kind in {'entry','exit'}; step indexes into the corresponding
    type tuple. Returns a bool mask over donor.values of size Nv."""
    target_type = (sig.entry_types if port_kind == 'entry' else sig.exit_types)[step]
    mask = torch.zeros(len(donor_ir.values), dtype=torch.bool)
    for i, v in enumerate(donor_ir.values.values()):
        v_t = normalize_type_hint(v.type_hint)
        if is_subtype(v_t, target_type) or is_subtype(target_type, v_t):
            mask[i] = True
    return mask
```

`BoundaryHead.cut_logits` / `output_logits` get a new `mask` kwarg; when given, `logits[~mask] = -inf` before softmax. Existing call sites (Track B, training-time replay) pass `mask=None` to recover current behavior.

### 6.4 Region classification (Step 4 detail)

```python
def classify_region(R: RewriteRegion,
                    slot_meta: dict[str, SlotMeta],
                    ) -> tuple[Literal["I","II","III"], str | None]:
    R_ops = set(R.op_ids)
    half_cut = []         # slot.full_ops ∩ R is non-empty AND non-superset
    fully_contained = []  # slot.full_ops ⊆ R  → candidates for attribution
    for key, S in slot_meta.items():
        S_full = _full_ops(slot_meta, key)        # includes descendants
        inter = S_full & R_ops
        if not inter:
            continue
        if S_full <= R_ops:
            fully_contained.append((len(S_full), key))
        else:
            half_cut.append(key)
    if half_cut:
        return ("III", None)
    if fully_contained:
        # "R contains at least one whole slot." Attribute to the LARGEST
        # fully-contained slot — that's the user's "max contained slot" rule.
        fully_contained.sort(reverse=True)
        return ("I", fully_contained[0][1])
    # R contains no whole slot. Check if R sits inside some slot.
    enclosing = []
    for key, S in slot_meta.items():
        if R_ops <= _full_ops(slot_meta, key):
            enclosing.append((len(_full_ops(slot_meta, key)), key))
    if enclosing:
        # "R is inside a slot." Attribute to the SMALLEST enclosing slot
        # (most specific). Variant body becomes a sub-mutation of that slot.
        enclosing.sort()
        return ("I", enclosing[0][1])
    return ("II", None)
```

### 6.5 Apply path per case (Step 5 detail)

- **Case I (`attribution_key` set):**
  1. Run `graft_general(host_ir, host_region, donor_region, contract)` — produces post-graft IR.
  2. Compute the new op_ids for the attribution slot: re-scan ops with `slot_id == attribution_key` (the graft preserved tags on host ops it didn't touch; new donor ops are stamped with `attribution_key` during insertion).
  3. Update `slot_meta[attribution_key]` (op_ids may shrink/grow if R extended beyond the slot; inputs/outputs unchanged because R's boundary equals the slot's boundary by construction OR encloses it consistently).
  4. Append the donor subgraph (canonicalized as a `SubgraphSnapshot`) to `genome.slot_populations[attribution_key].variants` for future micro-evolution reuse.
- **Case II:**
  1. Run `graft_general(host_ir, host_region, donor_region, contract)`.
  2. `slot_meta` untouched.
  3. Optionally enqueue the post-graft IR for Track B inspection in the next generation.
- **Case III:**
  1. For each half-cut slot S: clear all `op.attrs["slot_id"]` matching S.pop_key (and its descendants), `slot_meta.pop(S.pop_key)`, archive `genome.slot_populations.pop(S.pop_key)` to `genome.metadata["dissolved_slots"]` for lineage telemetry.
  2. Then run the Case II path on the (now structural-only) IR.

### 6.6 Track B — slot stamping (separate emission mode)

Not part of the Case I/II/III graft pipeline. A dedicated entry point:

```python
class IRPatternMatcher:
    def propose_slot_stampings(self, entries: list[AlgorithmEntry], n: int = 4,
                               ) -> list[SlotStampingProposal]: ...
```

For each candidate IR it runs the same `BoundaryHead` (mask=None, single-IR mode), samples entry/exit values, builds an SESE region, validates SESE-ness, rejects regions that touch existing slots, and emits `SlotStampingProposal(host_algo_id, op_ids, inputs, outputs, suggested_pop_key, confidence)`. Orchestrator stamps tags + creates `slot_meta` entry + spawns a fresh `SlotPopulation` whose first variant is the stamped region's `SubgraphSnapshot`.

Reward: a stamped slot earns +1 if any subsequent micro-evolution variant of it passes the R6 behavior gate AND improves fitness; 0 otherwise. Fed back into the same PG loop.

### 6.7 File-level deltas in `evolution/gnn_pattern_matcher.py`

| Current code | New code |
|---|---|
| `_OPCODE_LIST = [..., "algslot", ...]` (line 53) | Remove `"algslot"`. Re-index → `_N_OPCODES -= 1` → `_NODE_DIM` shifts. Bump checkpoint version sentinel. |
| `_PROV_HASH_SALT = "fii-prov-v0"` (line 123); `_hash_provenance(slot_id, is_boundary)` (line 137); `_PROV_FEATURES = 17` | Rename to `_SLOT_HASH_SALT = "slot-id-v1"`; `_hash_slot_id(slot_id)` drops the `is_boundary` arg; `_SLOT_FEATURES = 16`. |
| `prov = op.attrs.get("_provenance") or {}; prov_feat = _hash_provenance(...)` (lines 174–177, 1721–1724) | `slot_id = op.attrs.get("slot_id"); slot_feat = _hash_slot_id(slot_id)`. Two call sites: `ir_to_graph` and `_get_op_feats`. |
| `BoundaryHead.cut_logits(emb, value_feats)` and `output_logits(...)` | Add `mask: torch.BoolTensor \| None = None` kwarg; when given, `logits = logits.masked_fill(~mask, -inf)`. |
| `_build_boundary_region` (line 1180) — synthesizes a region purely from cut-logits sampling on host | **No slot_meta snapping.** Keep the cut-logits sampling; just feed `mask=None` for host (free-form). For donor, call from new `_sample_donor_under_signature` helper. |
| `_make_boundary_proposal` (line 924) — emits `GraftProposal(region, contract, …)` | Two-stage: (a) build host region via `_build_boundary_region(host)`; (b) compute `BoundarySignature`; (c) sample donor via `_sample_donor_under_signature(donor, sig)`; (d) call `classify_region` to set `proposal.case` + `proposal.attribution_slot_pop_key`. Reject proposal if donor sampling aborts or classification rejects. |
| `_compute_return_slice_values` and basic `RewriteRegion` builders | Unchanged. |
| Encoder / PairScorer / training loop / experience buffer / sampling utilities | **Unchanged.** Only tensor-shape change is `_NODE_DIM` shrink (kwarg `mask` is optional). |

### 6.8 New small additions

- `evolution/pool_types.py::GraftProposal`: add `case: Literal["I","II","III"]`, `attribution_slot_pop_key: str | None = None`, and `boundary_signature: BoundarySignature` (for telemetry / replay). Drop the earlier-proposed plain `pop_key` field.
- `evolution/graft_classifier.py` (new, ~80 LOC): `BoundarySignature` dataclass + `classify_region` + `_full_ops` helper.
- `evolution/slot_dissolve.py` (renamed from `slot_rediscovery.py`, slimmed to ~50 LOC): `dissolve_slot(genome, pop_key)` — clears tags, removes meta + population, archives to lineage.
- `evolution/gnn_pattern_matcher.py`: new helpers `_sample_donor_under_signature`, `_propose_slot_stampings`, plus `SlotStampingProposal` dataclass.

### 6.9 Training data / checkpoint compatibility

- `_NODE_DIM` decreases by ~2 (one `algslot` opcode + 1 boundary flag scalar). Old GNN checkpoints become incompatible — bump version sentinel in `gnn_state.json`; refuse to load older versions; fresh-start training required.
- The mask kwarg defaults to `None`, so the architecture trains identically on legacy paths during the transition.
- Experience buffer gains `case` and `boundary_signature` fields; old buffers can be discarded since the RL is on-policy.

### 6.10 Backward-compat shim during M1–M5

While templates are being migrated (some still use old `slot_xxx(...)` calls, some already use `with slot(...)`), the GNN node featurizer reads BOTH `op.attrs.get("slot_id")` AND falls back to `op.attrs.get("_provenance", {}).get("from_slot_id")`. Removed in M7 alongside `fii.py` deletion.

---

## 7. Test Plan

### 7.1 Tests requiring full rewrite (DSL change + boundary semantics)

| Test | New focus |
|---|---|
| `tests/unit/test_slot_evolution.py` | Op-set swap correctness, signature-mismatch rejection, slot_meta update post-apply. |
| `tests/unit/test_behavior_gate.py` | R6 still rejects identity variants — re-fixture against new SlotPopulation. |
| `tests/unit/test_core_slots_evolvable.py` | Re-fixture each detector with the new template DSL; assert default re-apply succeeds 91/91 (was 179/221). |
| `tests/unit/test_ir_evolution.py` | End-to-end: load template, identify slot, apply variant, eval. |
| `tests/unit/test_no_source_roundtrip_mutation.py` | Target IR-level mutation (no source emit / re-parse) on the new SubgraphSnapshot. |
| `tests/unit/test_gp_micro_population.py` | Same story, plus stress with nested slots (BP). |
| `tests/unit/test_structural_operators.py` | Confirm 11 GP operators all consume/produce SubgraphSnapshot; smoke each. |

### 7.2 Tests requiring small touch-ups

- Any test importing `evolution.fii`, `gp.region_resolver`, `slot_discovery`, `slot_rediscovery`, or `algorithm_ir.ir.dialect.AlgSlot` — delete or migrate.
- `tests/unit/test_validator.py` — add slot_meta consistency cases.
- `tests/integration/test_grafting_demo.py` — the BP-summary graft demo uses `define_rewrite_region` + `infer_boundary_contract`; convert to explicit `slot_meta` lookup.

### 7.3 New tests to add

- `test_with_slot_compilation.py` — `ir_builder` correctly tags ops + populates slot_meta for nested / sequential / re-entrant `with slot(...)` blocks.
- `test_apply_slot_variant_invariants.py` — post-apply: every op tagged `slot_id=K` is in `slot_meta[K].op_ids`; downstream consumers wired correctly; validator passes.
- `test_default_reapply_audit.py` — for every `(genome, slot)` pair in the pool, default variant re-apply must succeed (target: 100%).

---

## 8. Migration Sequence

| Milestone | Work | Exit criterion |
|---|---|---|
| **M0** | Revert uncommitted patch in `slot_evolution.py` (`git checkout`). Branch off `a05ebce`. | Clean working tree, all 290 tests pass. |
| **M1** | Add `ast.With` to `ir_builder._compile_stmt`. Add `FunctionIR.slot_meta` + `SlotMeta`. Add `slot` no-op stub to `_template_globals`. New test: `test_with_slot_compilation.py`. | A toy template with one `with slot(...)` compiles to FunctionIR whose `slot_meta` has the expected entry. |
| **M2** | Rewrite `evolution/skeleton_library.py` templates (LMMSE first as smoke-test, then EP / BP / AMP, then the rest). Drop `slot_arg_names` + `slot_default_keys` from SkeletonSpec. | Every template parses; every detector produces the correct `slot_meta` (asserted by a parametrized test). |
| **M3** | Rewrite `evolution/ir_pool.py` to drop helper compilation + AlgSlot conversion. Add `SubgraphSnapshot` to `pool_types.py`. `build_ir_pool()` returns 91 genomes with populated `slot_populations`. | `build_ir_pool` runs; default re-apply for all 91 succeeds. |
| **M4** | Rewrite `evolution/slot_evolution.py` op-swap path. Update `algorithm_engine._micro_evolve` to use the new path. | New `test_apply_slot_variant_invariants.py` passes; `test_default_reapply_audit.py` reaches 100%. |
| **M5** | Cleanup `materialize.py`, `codegen.py`, `algorithm_engine.to_entry`, `train_gnn.py` log fields. | Full pipeline materialize → eval works on a synthetic genome. |
| **M6a** | Add `BoundarySignature` + `classify_region` (`evolution/graft_classifier.py`). Add `BoundaryHead.cut_logits(mask=...)` plumbing. Add `_sample_donor_under_signature`. Rewire `_make_boundary_proposal` to two-stage + classify. Add Case I/II/III dispatch in `algorithm_engine` graft consumer. Rename `slot_rediscovery.py` → `slot_dissolve.py` (slim to dissolve helper). | A synthetic graft of each type (I/II/III) round-trips through `train_gnn` for 1 generation without exceptions; classifier unit-tested on hand-crafted IRs. |
| **M6b** | Add `propose_slot_stampings` (Track B). Integrate stamping into post-Case-II hook. | One Track B proposal stamps a slot and a follow-up Case I micro-evolution variant survives R6. |
| **M7** | DELETE `fii.py`, `gp/region_resolver.py`, `slot_discovery.py`, `AlgSlot` op. Remove the `_provenance` fallback shim from §6.10. | All imports remain resolvable; full test suite passes. |
| **M8** | Test suite migration (see §7). Run `train_gnn` for 5 generations; compare stats against v7 baseline. | ≥179/221 default re-apply (target 221/221); slot-evo telemetry shows ≥ v7 mutation rate; no silent-None failures; Case I/II/III distribution recorded per generation. |

Each milestone is independently committable. M1–M3 establish the new compile path while old runtime still works (templates not yet migrated keep using old code paths gated by `slot_meta` non-empty). M4 is the cutover.

---

## 9. Risk Register

| Risk | Mitigation |
|---|---|
| Nested slots (BP outer sweep + inner per-message) confuse op-tagging stack. | Stack-aware `_emit_op` wrapper writes ALL stack frames' slot_ids as a tuple if needed; or finer-grained inner slot's `op_ids` are a subset of outer's. Decide in M1; document. |
| Template author forgets to bind a declared `outputs` name. | `ir_builder` raises at compile time with clear error — surfaced during M2 migration immediately, not at run-time. |
| Multi-block helper slots (sa.accept, mh.accept) span an `if` body. | The `with slot(...)` block can wrap an `ast.If`; tagging by stack frame correctly tags ops in both then/else branches and the resulting phis. Add specific test in M1. |
| `slot_meta` deep-copy cost on every genome clone. | `SlotMeta` is small (tuple of strings); negligible vs IR clone cost. |
| GP operators that delete or split ops break `slot_meta.op_ids` invariant. | Operators must re-tag emitted ops with the slot_id they came from (already do this for `_provenance`); `slot_meta` op_ids list is rebuilt at each commit by re-scanning `slot_id` attrs (cheap, robust). Document this contract in `gp/operators/base.py`. |
| GNN pattern matcher training data uses old `_provenance.from_slot_id` keys. | Add a one-shot translator script in M6 that maps old keys to new `slot_id`. Or retrain — slot identity is preserved by `pop_key` strings (unchanged). |
| Backward-incompat with persisted gene bank from v7 runs. | One-time migration: run `build_ir_pool` to regenerate. No need to load old pickles. |

---

## 10. Out of Scope

Explicitly NOT covered by this refactor:

- The fitness evaluator, R6/R7 telemetry, eval subprocess protocol — unchanged.
- The GP operator catalogue (11 operators) — unchanged behavior; only their input/output container type adjusts.
- The MIMO simulation framework, training loop top-level structure, telemetry CSV/JSONL schema (column names will change minimally — see M5).
- Performance optimization beyond what falls out naturally from removing FII (caching, compile time, eval throughput).
- Persistence/serialization of genomes (no on-disk format defined here).
