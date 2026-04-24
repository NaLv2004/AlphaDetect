# Algorithm-IR Code Review & Refactor Plan

**Status**: Planning document. Approved scope, awaiting implementation kickoff.
**Author**: Orchestrator (audit conducted 2026-04-23)
**Constraint (hard)**: No injection of algorithm priors. Every new piece of
code must either (a) remove an existing prior, (b) add a domain-agnostic
mechanism, or (c) replace a hand-written algorithm component with a
mechanically-derived equivalent.

---

## 0. Why this document exists

The 80-generation training run (`results/gnn_training/training_log.jsonl`)
exposed that the GNN converges to a **degenerate "loop-severing" graft
pattern** with effective rate plateauing at 1–5 %. The audit traced this
to a chain of structural defects in:

1. **GNN ↔ IR interface** — `algslot` is opaque to the GNN; cuts can only
   land on slot boundaries, so loop-severing is the only signal.
2. **Initial algorithm pool** — 14 hand-written slot defaults are either
   monolithic (`bp_sweep` ~80 lines), trivially degenerate (`cavity =
   identity`), or duplicated (3× `hard_decision`); 8 L3 templates hardcode
   K, max_iters, damping as Python literals; 200+ `skeleton_library`
   templates are largely structural duplicates.
3. **Micro-evolution** — `mutate_ir` samples insert/delete with probability
   0; `random_ir_program` only emits single-line return expressions, so
   7-of-8 initial variants per slot are unusable; population is 8 with
   1 generation and effective size ≈ 1.

This document is the **single source of truth** for the planned refactor.

---

## 1. Refactor objectives

| # | Objective | Verification metric |
|---|---|---|
| O1 | GNN sees the full IR including former slot internals; cuts can land anywhere | ≥ 30 % of GNN-proposed regions cross or enter slot boundaries within 30 generations |
| O2 | All initial algorithm-pool entries (MIMO + linear-algebra + probability) execute correctly and reach published baseline performance ranges | LMMSE/ZF/OSIC/KBest/BP/EP/AMP SER curves match reference within 0.5 dB at SER = 1e-2; non-MIMO ops pass property tests |
| O3 | No dead code, no fake/placeholder code, no degenerate-identity defaults remain in the initial pool | Static audit + runtime audit pass (Sec. 5.4) |
| O4 | Micro-evolution becomes a real, type-aware GP framework with population ≥ 32, ≥ 5 generations, multiple operator families | Effective slot-variant turnover ≥ 50 % per macro generation; mean micro-fitness improvement > 0 in ≥ 70 % of slots |
| O5 | Full-pipeline run shows non-degenerate grafts (slot-internal mutations, cross-slot grafts) make up ≥ 40 % of effective grafts | `top50_grafts.log` analysis with new provenance-tagging |

Every step below has a **gating test suite**. If the gate fails, no
subsequent step starts.

---

## 2. Architectural change: Fully-Inlined IR (FII)

### 2.1 Concept

Current state: GNN sees `genome.structural_ir` where every slot is a
single `algslot` op. Slot variants live in a separate `FunctionIR` field
and are only spliced in at materialization time via string substitution.

Target state: define a derived view

```
FII(genome) := inline_genome_to_full_ir(genome) → (FunctionIR, ProvenanceMap)
```

The FII is a **single FunctionIR** in which every `algslot` op is
replaced by the ops of its current best variant, with values
α-renamed. Each inlined op carries a `_provenance` attribute on
`Op.attrs` (the existing `attrs: dict[str, Any]` field, no schema change
needed):

```python
op.attrs["_provenance"] = {
    "from_slot_id": "bp_sweep" | None,        # None for originally structural ops
    "slot_pop_key": "bp.bp_sweep" | None,
    "variant_idx": int | None,
    "orig_op_id": str | None,                 # for back-mapping in Case I
    "is_slot_boundary": bool,                 # True for first/last op of inlined block
    "boundary_kind": "input_bind" | "output_bind" | None,
}
```

The FII is the GNN's only view of an algorithm. Slots become a **storage
& evolution unit** but cease to be an **ontological boundary**.

### 2.2 Module layout

| New file | Purpose |
|---|---|
| `evolution/fii.py` | `inline_genome_to_full_ir`, `ProvenanceMap`, `FIICache` |
| `evolution/graft_dispatch.py` | Decide Case I/II/III for a given graft, route to back-mapping or dissolution |
| `evolution/slot_dissolution.py` | Permanently fold slot ops into structural IR when a graft crosses a slot boundary |
| `evolution/slot_rediscovery.py` | Periodic structural-cohesion analysis to extract new region labels |
| `evolution/types_lattice.py` | Type lattice for type-aware GP (Sec. 4) |
| `evolution/gp_population.py` | Real micro-population (replaces `_micro_evolve`) |

Existing files modified: `gnn_pattern_matcher.py`, `algorithm_engine.py`,
`materialize.py`, `operators.py`, `random_program.py`, `pool_types.py`,
`ir_pool.py`, `skeleton_library.py`, `train_gnn.py`.

### 2.3 GNN feature change

Extend `_NODE_DIM` in `gnn_pattern_matcher.py` from
`_N_OPCODES + _CALLEE_FEATURES` (= 19 + 8 = 27) to
`27 + _PROVENANCE_DIM` where `_PROVENANCE_DIM = 17`:

* 16 floats: `hash(from_slot_id) → 16-bucket bag` (or all zeros for
  structural ops). Hash buckets, **not** one-hot — this prevents the
  feature from becoming a memorizable categorical prior.
* 1 float: `is_slot_boundary` flag.

### 2.4 Region enumeration on FII

`enumerate_cut_candidates` and `enumerate_observable_values` are pure
functions of a `FunctionIR`. The only required change is the call site:
in `gnn_pattern_matcher.py` and `algorithm_engine.py`, replace
`func_ir = entry.genome.structural_ir` with
`func_ir, prov = entry.fii_cache.get_or_build(entry.genome)`.

### 2.5 Graft application — three cases

After a graft region is selected on the FII, look up provenance of all
ops in the host region:

| Case | Provenance of host region ops | Action |
|---|---|---|
| **I — In-slot** | All `from_slot_id == X`, none are boundary | Compute the new variant: take slot X's current best variant IR, replace the orig ops listed in `orig_op_id` with the donor's region ops. Append result as a new variant in the slot's micro-population. |
| **II — Structural** | All `from_slot_id is None` | Apply graft directly to `genome.structural_ir`. No slot affected. |
| **III — Cross-boundary** | Mixed, or contains slot boundary ops | **Dissolution**: rebuild `structural_ir` as the FII (with all touched slots inlined and α-renamed), apply the graft, then **delete** those slots from `slot_populations`. Surviving slots are unaffected. |

Case III is the mechanism by which the slot prior is **gradually
unlearned**. Algorithms can flatten over generations.

### 2.6 Slot rediscovery

To prevent total dissolution and keep micro-evolution alive, run every
N generations (default N = 20) on `genome.structural_ir`:

```python
def rediscover_slots(structural_ir, *,
                     min_size=4, max_size=24, max_boundary=4,
                     max_new_per_pass=3) -> list[NewSlotProposal]
```

Selection criteria are purely structural:

* Connected subgraph in the dataflow DAG of size in `[min_size,
  max_size]`.
* Number of values entering the subgraph (live-in) + leaving it
  (live-out) ≤ `max_boundary`.
* Internal cohesion (intra/extra edge ratio) ≥ 1.5.
* Does not overlap an existing slot.
* Subgraph type signature (input types tuple, output types tuple) is
  inferable.

Each accepted region becomes a new slot with a fresh `slot_id`
(`auto_<hash>`), gets initialized as a slot population whose initial
variants are produced by the type-aware GP framework (Sec. 4).

---

## 3. Initial algorithm pool — fix and validate

### 3.0 Sizing principle and target totals

The previous draft proposed only ~30 entries — far too small to seed a
type-aware GP search and far too narrow to give the GNN diverse donor
regions. The pool serves **two distinct functions**:

1. **Macro hosts** — full L3 detectors (Category A) used as evolving
   genomes.
2. **Donor library** — every callable slot/primitive that can serve as
   a donor region for grafting. The donor library must densely cover
   the type lattice so that GNN graft proposals find compatible
   substitutions in most contexts.

Target total: **≥ 240 distinct, executable, non-degenerate entries**:

| Category | Count target | Role |
|---|---|---|
| A. MIMO detector templates | ≥ 24 | macro hosts |
| B. Linear-algebra primitives | ≥ 40 | donor library, lifted slots |
| C. Probability / statistics primitives | ≥ 30 | donor library |
| D. Numerical micro-kernels | ≥ 30 | donor library |
| E. Discrete / search primitives | ≥ 20 | donor library |
| F. Control-flow scaffolds (loops, branches) | ≥ 12 | structural slots |
| G. Tensor / reshape / index primitives | ≥ 25 | donor library |
| H. Sketching / projection / dimensionality | ≥ 15 | donor library |
| I. Robust-statistics / clipping primitives | ≥ 12 | donor library |
| J. Differentiable surrogate primitives | ≥ 12 | donor library, future-use |
| K. Calibration / decision primitives | ≥ 12 | donor library |
| L. Performance / utility primitives | ≥ 10 | donor library |

Every entry: stand-alone executable function, documented spec
(`ProgramSpec` in the type lattice from Sec. 4.1), passes the Pool
Health Test suite (Sec. 3.5). **No fake code, no dead code, no silent
underperformers.** Entries that fail PHT-6 (SER targets) or PHT-3
(property tests) are either fixed or removed with logged justification.

### 3.1 Category A — MIMO detector templates (≥ 24)

Target SER ranges measured at 16×16 MIMO, 16-QAM, Rayleigh flat fading,
perfect CSI, N_trials ≥ 5000.

#### A1. Linear detectors (4)

| ID | Description | Target SER @ 16 dB |
|---|---|---|
| `mf` | Matched filter only | ≤ 2e-1 |
| `zf` | Zero forcing | ≤ 1e-1 |
| `lmmse` | LMMSE | ≤ 2e-2 |
| `lmmse_diag` | Diagonal-load LMMSE (loading factor as const-slot) | ≤ 3e-2 |

#### A2. Iterative linear-system detectors (8)

Each iterates a linear system `Gx = H^H y` with a different splitting.
Iteration count, damping, and (where applicable) relaxation parameter
are all const-slots.

| ID | Splitting | Target SER @ 16 dB |
|---|---|---|
| `iter_jacobi` | Diagonal | ≤ 5e-2 |
| `iter_gs` | Gauss–Seidel | ≤ 3e-2 |
| `iter_sor` | SOR | ≤ 3e-2 |
| `iter_ssor` | Symmetric SOR | ≤ 3e-2 |
| `iter_richardson` | Richardson | ≤ 5e-2 |
| `iter_neumann` | Neumann series (≤ 4 terms, term count is a const-slot) | ≤ 5e-2 |
| `iter_chebyshev` | Chebyshev acceleration | ≤ 3e-2 |
| `iter_cg` | Conjugate gradients on `G` | ≤ 2e-2 |

#### A3. Sequential interference cancellation (3)

| ID | Description | Target SER @ 16 dB |
|---|---|---|
| `osic_snr` | OSIC ordered by post-detection SNR | ≤ 5e-3 |
| `osic_norm` | OSIC ordered by column norm | ≤ 1e-2 |
| `pic` | Parallel IC after LMMSE init (one pass) | ≤ 1e-2 |

#### A4. Tree / search detectors (3)

| ID | Description | Target SER @ 16 dB |
|---|---|---|
| `kbest` | K-best (K is a const-slot, default 16) | ≤ 8e-4 |
| `stack` | Stack decoder (max nodes is a const-slot) | ≤ 5e-3 |
| `sphere_fp` | Fincke–Pohst sphere with adaptive radius | ≤ 1e-3 |

#### A5. Message-passing / inference detectors (3)

| ID | Description | Target SER @ 16 dB |
|---|---|---|
| `bp` | Discrete BP on bipartite factor graph | ≤ 5e-3 |
| `ep` | Expectation propagation | ≤ 2e-3 |
| `amp` | AMP | ≤ 5e-3 |

#### A6. Hybrid / multi-stage detectors (3)

| ID | Description | Target SER @ 16 dB |
|---|---|---|
| `lmmse_pic_n` | LMMSE init + PIC, pass count const-slot | ≤ 1e-3 |
| `mmse_kbest` | LMMSE preprocessing + K-best refinement | ≤ 5e-4 |
| `bp_then_ep` | BP warm-start, then EP refine | ≤ 1e-3 |

> Any template that **cannot meet** its target after fix attempts is
> removed and logged in `code_review/pool_changes.md`. Pool size is
> backfilled by promoting structurally-different variants of remaining
> families (different const-slot defaults), never by adding fake
> entries.

### 3.2 Category B — Linear-algebra primitives (≥ 40)

Donor library for any algorithm that touches matrices/vectors. All are
pure functions; signatures registered in the type lattice.

#### B1. Matrix construction & manipulation (10)

`gram` (`H → H^H H`), `gram_regularized` (`H, σ² → H^H H + σ² I`),
`mf_project` (`H, y → H^H y`), `regularize_diag`,
`augment_matrix` (`A, B → [A; B]`), `block_diag2`,
`column_select` (`H, idx → H[:, idx]`), `column_drop`,
`row_normalize_rows`, `transpose_conj`.

#### B2. Decompositions (8)

`qr_decomp`, `qr_householder`, `cholesky_decomp`,
`cholesky_inv`, `lu_decomp`, `svd_decomp`,
`eigh_decomp` (Hermitian eigendecomposition),
`schur_decomp`.

Each returns the standard tuple of factors as a typed object in the
lattice (e.g. `mat_decomp_qr = (Q: mat_cx, R: mat_cx)`).

#### B3. Linear-system solvers (8)

`linsolve_direct` (numpy direct solve),
`linsolve_triangular_lower`, `linsolve_triangular_upper`,
`linsolve_via_qr`, `linsolve_via_chol`,
`linsolve_iterative_jacobi_step`, `linsolve_iterative_gs_step`,
`linsolve_iterative_sor_step` (each is a single iteration step,
designed to be wrapped in `slot_loop`).

#### B4. Inverses & pseudo-inverses (6)

`inverse_direct`, `inverse_via_chol`, `inverse_via_qr`,
`pseudo_inverse_left` (`(H^H H)⁻¹ H^H`),
`pseudo_inverse_right` (`H^H (H H^H)⁻¹`),
`inverse_neumann` (truncated Neumann series, term count const-slot).

#### B5. Norms, distances, projections (8)

`norm_l2_vec`, `norm_frob_mat`, `norm_l2_sq`,
`inner_product`, `column_norms`,
`project_onto_columnspace`, `gram_schmidt_orthogonalize`,
`distance_matrix` (pairwise `‖x_i − y_j‖²`).

### 3.3 Category C — Probability / statistics primitives (≥ 30)

#### C1. Soft-decision math (10)

`softmax_real` (numerically stable),
`softmax_complex_distance` (softmax of `−|x − support|²/τ`),
`log_softmax`, `log_sum_exp`, `entropy_disc`,
`kl_divergence_disc`, `js_divergence_disc`,
`renormalize_probs`, `clip_probs` (avoid 0/1 ill-conditioning),
`bernoulli_logit_to_prob`.

#### C2. Gaussian / mixture math (10)

`gaussian_pdf_log`, `gaussian_pdf`, `gaussian_cdf_real`,
`gaussian_complex_log_pdf`, `gaussian_mixture_logpdf`,
`mean_var_from_weights` (de-duplicated from `site_update`/`amp_iterate`),
`weighted_mean`, `weighted_variance`,
`covariance_from_samples`, `precision_from_covariance`.

#### C3. Damping / convex combinations (4)

`damping_convex` (3-arg with ω const-slot),
`damping_per_coord` (per-coordinate ω vector),
`damping_polyak` (heavy-ball-style two-term mix),
`identity_passthrough` (canonical neutral; only kept because the type
lattice requires a typed identity for some operators — never used as a
default for a real slot).

#### C4. Hard-decision / quantization (6)

`argmin_dist` (de-duplicated from 3 prior copies),
`argmax_logits`, `nearest_in_constellation_real`,
`nearest_in_constellation_complex`,
`top_k_indices`, `confidence_margin` (top1 − top2).

### 3.4 Category D — Numerical micro-kernels (≥ 30)

Small AST units the GP framework can compose. Each is ≤ 8 lines.

`safe_div`, `safe_log`, `safe_sqrt`, `safe_exp` (clip to avoid
overflow/underflow), `clip_real`, `clip_complex_magnitude`,
`abs_complex`, `phase_complex`, `real_part`, `imag_part`,
`conj_complex`,
`add_vec`, `sub_vec`, `mul_vec_elementwise`, `div_vec_elementwise`,
`scalar_times_vec`, `scalar_times_mat`,
`vec_dot`, `mat_vec_mul`, `mat_mat_mul`,
`fma_vec` (a + b·c fused), `axpy` (y ← α x + y),
`reduce_sum_vec`, `reduce_max_vec`, `reduce_min_vec`,
`cumulative_sum`, `running_mean`,
`sign_complex` (`x / |x|` with safe handling),
`sigmoid_real`, `tanh_real`.

### 3.5 Category E — Discrete / search primitives (≥ 20)

For tree, stack, beam, and combinatorial detectors.

`make_tree_node`, `extend_node` (append symbol + accumulate cost),
`copy_node`, `node_sort_by_cost`,
`heap_push`, `heap_pop`, `heap_top`,
`beam_select_top_k`, `beam_merge`,
`stack_push`, `stack_pop`, `stack_select_best`,
`sphere_radius_init`, `sphere_radius_update`,
`zigzag_enum_next`, `schnorr_euchner_next` (next symbol in the
SE order — pure index arithmetic, no detector logic),
`partial_metric_accumulate`, `partial_metric_combine`,
`pruning_threshold_update`,
`tree_node_compare`.

### 3.6 Category F — Control-flow scaffolds (≥ 12)

Generic structural slots (the loop-lifting target from Sec. 3.7 item 5).

`slot_loop_fixed_n` (default `for _ in range(n): state = body(state)`),
`slot_loop_while_threshold` (terminate when an L2-difference falls
below ε), `slot_loop_until_convergence` (track relative change),
`slot_loop_unroll2` (two body calls per iter), `slot_loop_unroll3`,
`slot_branch_predicate` (if-then-else dispatch),
`slot_switch_3way`, `slot_repeat_until_stable`,
`slot_pipeline2` (`f` then `g`),
`slot_pipeline3`, `slot_residual` (`x + f(x)`),
`slot_select_best_of_two_runs`.

These are **the only place** raw control-flow lives. Every detector
template is rewritten so its outer loops are calls into these scaffolds.

### 3.7 Category G — Tensor / reshape / index primitives (≥ 25)

`zeros_like`, `ones_like`, `eye_n`, `arange_n`,
`stack_two_vecs_to_mat`, `concat_vecs`, `reshape_to_real_imag`,
`reshape_to_complex`, `take_indices`, `put_indices`,
`mask_select`, `scatter_add`,
`fold_real_imag_to_double_dim`, `unfold_double_dim_to_complex`
(complex-to-real-augmented-system conversion — purely structural,
no algorithm prior),
`outer_product`, `kron2`, `khatri_rao2`,
`vec_op` (matrix→column-stacked vector), `unvec_op`,
`expand_dims`, `squeeze_dim`,
`broadcast_add_mat_vec`, `cyclic_shift_vec`,
`flip_vec`, `transpose_first_two`.

### 3.8 Category H — Sketching / projection / dimensionality (≥ 15)

Random / structured projections; useful for low-complexity preprocessing
and for the GP framework to discover dimensionality-trading variants.

`gaussian_random_projection`, `subsampled_columns`,
`hadamard_projection_pow2`, `dct_projection`, `dft_projection`,
`structured_diagonal_random`, `subspace_principal_directions`
(top-k from `eigh_decomp`), `subspace_residual`,
`leverage_scores`, `column_norm_weighted_sample`,
`coherence_metric`, `condition_number_estimate`,
`effective_rank`, `low_rank_approx` (truncated SVD wrapper),
`whiten_columns`.

Each must be reproducible with a fixed RNG seed to keep evaluation
deterministic.

### 3.9 Category I — Robust-statistics / clipping primitives (≥ 12)

`huber_norm`, `huber_grad`, `tukey_biweight_norm`,
`l1_shrinkage`, `soft_threshold` (proximal of `‖·‖_1`),
`hard_threshold`, `clip_value`, `clip_norm_l2`,
`winsorize`, `median_abs_dev`,
`outlier_mask_zscore`, `running_robust_mean`.

### 3.10 Category J — Differentiable surrogate primitives (≥ 12)

For future hybrid hard/soft pipelines and gradient-style refinements.
**Pure forward functions** — no autograd dependency. The forward output
is the only thing GP composes.

`smooth_min` (log-sum-exp minimum),
`smooth_max`, `smooth_argmin_softmax_weighted` (returns weighted
support combination), `smooth_argmax_softmax_weighted`,
`temperature_anneal_schedule` (returns scalar function of iter),
`gumbel_softmax_argmax_surrogate` (deterministic when seed fixed),
`straight_through_quantize` (forward = nearest, structurally pure),
`relu_complex_magnitude`, `prelu_real`,
`softplus_real`, `swish_real`,
`gradient_clip_l2` (output clipping helper, used by GP-evolved pipelines).

These are seed primitives, not currently used by any default detector;
they exist so the donor library is **larger than the host library**
(necessary condition for graft diversity).

### 3.11 Category K — Calibration / decision primitives (≥ 12)

`temperature_scaling`, `vector_scaling_calibration`,
`platt_scaling_logits`, `isotonic_step` (one PAVA pass),
`brier_score_disc`, `expected_calibration_error_buckets`,
`ll_ratio_to_prob`, `prob_to_ll_ratio`,
`majority_vote_disc`, `weighted_vote_disc`,
`tie_break_random`, `tie_break_lowest_index`.

### 3.12 Category L — Performance / utility primitives (≥ 10)

`is_finite_vec`, `nan_to_zero`, `nan_to_value`,
`replace_inf_with_value`, `assert_shape_or_pad`,
`copy_state`, `null_op` (no-op typed identity, **only** for use as
operator placeholder during type unification — flagged so it cannot be
chosen as a default; this is the lone exception to "no identity
defaults"), `deepcopy_typed`,
`frozen_state_get`, `frozen_state_set`.

### 3.13 Audit & cleanup of pre-existing code

Audit `evolution/ir_pool.py` and `evolution/skeleton_library.py`:

1. **Identity-only defaults** → remove or replace via construction
   above. Hits: `cavity`, `damping_none` (replaced by
   `damping_convex` with ω = 1.0 default).
2. **Single-constant defaults** → keep but the constant must be
   lifted into a 0-arg const-slot (Sec. 2 const-lifter). Hits:
   `step_size_fixed` (0.01), `regularizer` (σ² coefficient).
3. **Duplicated defaults** → AST-canonicalize and merge under a
   single canonical key with aliases. Hits: `hard_decision`,
   `final_decision`, `bp_final_decision` → `argmin_dist`;
   `mean_var_from_weights` extracted from `site_update` and
   `amp_iterate` (both contained an inline copy).
4. **Hardcoded literals in templates** → AST-lift to const-slots.
   Mandatory hits: `KBEST_TEMPLATE` K=16, `BP_TEMPLATE` max_iters=8,
   `EP_TEMPLATE` it<20 and damping=0.5, `AMP_TEMPLATE` it<20,
   `STACK_TEMPLATE` max_nodes=2000, all `while i < N` constants in
   `skeleton_library.py` templates.
5. **Outer-loop lifting** → templates whose body is exactly
   `while i<N: x = slot_y(...); i+=1` are rewritten so the loop becomes
   a call to one of Category F's loop scaffolds.
6. **Skeleton family deduplication** → AST hash all templates after
   const-lifting; templates with identical structural hash are merged
   under one family key with multiple (different) const-slot defaults
   acting as instances. Pool initializer samples by family with
   variant index.
7. **Dead / fake code scan** — fail PHT-4 if any of:
   * `pass` as the only body of a function in the pool;
   * `return None`, `return 0.0`, or `return args[0]` for non-void
     specs;
   * `try: ... except: pass` swallowing errors silently;
   * functions whose AST contains no use of any of their parameters
     (pure-constant returns disguised as functions);
   * functions that always return one of their inputs unchanged
     (identity-passthrough disguise) — except those in the
     explicitly-typed-identity allow-list (`identity_passthrough`,
     `null_op`).
8. **Cross-call audit** — every `slot_*` callee referenced in any
   template resolves to either a default in `SLOT_DEFAULTS` /
   `EXTENDED_SLOT_DEFAULTS` / new categories above, or a const-slot
   generated by lifting. No dangling references.

### 3.14 Rewriting standards for new pool entries

* Pure functions, no global state, no I/O.
* Signatures must match a registered `ProgramSpec`; the spec's types must
  exist in the type lattice (Sec. 4.1).
* Tolerate degenerate inputs (singular matrices, zero vectors,
  empty constellations) by returning a valid output of the declared
  type — never raising. Use the `safe_*` micro-kernels from Category D.
* Numerical constants used inside the body must be either (a) trivially
  domain-agnostic (e.g. `1e-30` as denominator floor), or (b) AST-lifted
  to const-slots so they can be evolved.
* Every entry has a 1-line docstring describing its mathematical intent
  (used by PHT-3 reference checks where applicable).
* Where a primitive is used by ≥ 2 detector templates, the primitive
  must be a separate slot — no copy-paste of the body across templates.

### 3.15 Validation — Pool Health Test Suite (PHT)

Lives at `tests/test_pool_health.py`. **Gating test** for Step S1.

```
PHT-1   Spec compliance        — every pool entry compiles to FunctionIR;
                                  FunctionIR signature matches its
                                  registered ProgramSpec.
PHT-2   Type validity          — all input/output types resolvable in
                                  the type lattice; no "object" or "Any"
                                  leakage.
PHT-3   Property tests         — random-input stress (N ≥ 1000) per
                                  entry: no NaN/Inf in output, output
                                  shape matches spec, output dtype
                                  matches spec. Where a closed-form
                                  reference exists (np.linalg /
                                  scipy.special / scipy.linalg), the
                                  primitive's output must agree to
                                  absolute tolerance 1e-6.
PHT-4   No-dead-code scan      — AST audit per Sec. 3.13 item 7.
PHT-5   No-trivial-identity    — no pool entry's output is provably
                                  equal to one of its inputs unless the
                                  spec is on the explicit
                                  typed-identity allow-list.
PHT-6   MIMO baseline SER      — each detector in Category A run for
                                  N_trials ≥ 5000 frames at SNR ∈
                                  {0,4,8,12,16,20,24} dB on 16×16
                                  16-QAM Rayleigh flat. SER curve must
                                  meet the targets in Sec. 3.1 within
                                  tolerance ±0.3 dB at the target
                                  operating SER.
PHT-7   Cross-call audit       — every `slot_*` callee referenced in
                                  any template resolves to either a
                                  default in `SLOT_DEFAULTS` /
                                  `EXTENDED_SLOT_DEFAULTS` / new
                                  categories, or a const-slot generated
                                  by lifting. No dangling references.
PHT-8   Category coverage      — each Category A–L meets its size
                                  target from Sec. 3.0; total
                                  ≥ 240 distinct entries.
PHT-9   Determinism            — every primitive that takes a seed
                                  produces identical output across
                                  runs with the same seed; primitives
                                  not declared stochastic produce
                                  identical output across two
                                  back-to-back invocations on the same
                                  input.
PHT-10  Performance budget     — each donor primitive's runtime on a
                                  reference input is bounded
                                  (e.g. ≤ 50 ms for vec/mat ops at
                                  N = 64). Outliers flagged but not
                                  necessarily failed; logged.
```

The suite must pass on a clean run with `pytest -x -q
tests/test_pool_health.py`. Run-time budget: ≤ 30 min on workstation
with `pytest -n auto`. PHT-6 is the dominant cost.

---

## 4. Type-aware genetic-programming micro-evolution framework

### 4.1 Type lattice

`evolution/types_lattice.py`. A finite set of types with a join operator
for type inference during random-program generation and crossover:

```
PRIMITIVE  ::= int | float | bool | cx
TENSOR     ::= vec_f | vec_cx | vec_i | mat_f | mat_cx | tensor3_*
COMPOSITE  ::= tuple<T1,...,Tn> | list<T> | dict<str,T>
OBJECT     ::= node | candidate_list | open_set | mat_decomp | prob_table
```

A `ProgramSpec` declares `param_types` and `return_type` from this set.
Every IR `Value` carries an inferred type (annotated post-hoc by a
single-pass type-inference walk; type info goes into `Value.attrs`,
no schema change needed).

The lattice provides:

* `is_subtype(a, b)` — for generalization
* `unify(a, b)` — for crossover compatibility
* `available_ops_for_type(t)` — which IR opcodes can produce/consume `t`
* `default_value(t)` — fallback for synthesis

This is **not a type system for users**; it is a search-space-pruning
mechanism for the GP operators.

### 4.2 GP operators (typed)

Reimplement and extend `evolution/operators.py` as
`evolution/gp_operators.py`:

| Operator | Description | Typed? | Replaces |
|---|---|---|---|
| `mut_point` | Swap an op's opcode with another opcode of the same type signature from the lattice | yes | `_mutate_point` |
| `mut_const` | Perturb a `const` value with N(0, σ²); σ proportional to past variance | yes (no-op for non-numeric) | `_mutate_constant_perturb` |
| `mut_insert` | Insert a new op consuming a randomly chosen subset of in-scope values whose joint type is acceptable to a randomly chosen target opcode | yes | `_mutate_via_recompile` (insert path) |
| `mut_delete` | Delete an op whose outputs are unused or can be type-safely substituted by an existing in-scope value | yes | `_mutate_via_recompile` (delete path) |
| `mut_swap` | Swap two adjacent same-block ops if dataflow allows | yes | new |
| `mut_subtree_replace` | Replace a sub-DAG (≤ k ops) rooted at a value with a freshly synthesised sub-DAG of the same return type | yes | new |
| `mut_loop_unroll` | If a `while` loop body is constant-bound, partially unroll it | structural | new |
| `mut_const_promote` | Promote a literal `const` to a 0-arg const-slot so it can be evolved cross-individual | yes | new |
| `cx_subtree` | Crossover by swapping sub-DAGs of compatible return type between two parents | yes | partial replacement of `crossover_ir` |
| `cx_block` | Swap entire basic blocks if their live-in/live-out type signatures match | yes | new |
| `cx_grafted_callsite` | Replace a callsite to a slot in parent A with the callsite (incl. arg adaptation) from parent B if callee return type matches | yes | new |

Operator selection per individual is a **multi-armed bandit** updated by
fitness improvement attribution (per-operator EMA of contribution).
Hyperparameters live in `EvolutionConfig`.

All operators preserve SSA validity. After every operator, run a
lightweight `validate_ssa(func_ir)` check; on failure, reject and
re-roll up to 3 times before falling back to the parent.

### 4.3 Population mechanics

`evolution/gp_population.py` defines `MicroPopulation` replacing the
ad-hoc `SlotPopulation` evolution loop in `algorithm_engine._micro_*`:

| Field | Default | Notes |
|---|---|---|
| `size` | 64 | per slot, was 8 |
| `elite_keep` | 4 | always carried over |
| `tournament_k` | 5 | parent selection |
| `mutation_rate` | 0.6 | was 0.3 |
| `crossover_rate` | 0.35 | was 0.15 |
| `generations_per_macro` | 5 | was 1 |
| `novelty_archive_size` | 256 | for diversity pressure |
| `fitness_eval_trials` | 20 | was 5 |
| `fitness_eval_snrs` | 2 (current SNR and current-2 dB) | was 1 |
| `fitness_ema_alpha` | 0.5 | EMA per individual |

Selection: combined fitness + novelty score (NSLC-style).

Novelty: behavior signature is the vector of slot outputs on a fixed
set of probe inputs (frozen seed). Novelty = mean Euclidean distance to
k-nearest archive members.

### 4.4 Initialization (replaces broken `random_ir_program`)

`evolution/random_program.py` rewrite:

* Generation strategy chosen by `spec.return_type`:
  * primitive scalar → expression-tree generator (current generator,
    but with depth ≥ 6 and 30 % chance of containing one local
    `if` ternary)
  * tensor → assignment-list generator producing a sequence of typed
    assignments terminating in the declared return type
  * tuple/list → generate each component with the appropriate
    sub-generator and pack at the end
  * object (e.g. `node`) → use a registered constructor; perturb only
    its numeric fields
  * loop required (when input types contain a length-bearing collection)
    → wrap body in a generated `while` with a generated termination
    condition
* Validation: every generated program is exec'd on a probe input set
  before being accepted. Failure → re-roll up to 5 times → fall back
  to spec's default. Track fallback rate; alert if > 20 %.
* Initial pool seeding for a slot of size 64:
  * 1 default
  * 4 mutations of the default (1–3 ops applied)
  * 1 crossover of the default with itself (sanity)
  * 58 freshly generated random programs

### 4.5 Macro ↔ micro coupling

Per macro generation, for each genome and each slot, run
`MicroPopulation.evolve(generations=5)`. Update slot's `best_idx` from
the micro-pop's best. The macro evolution and GNN graft mechanism are
unchanged in the loop structure, but they now act on a richer slot
population.

Cross-host slot variant exchange (light prior, but **structural**):
maintain a global LRU bank
`SlotVariantBank: dict[(slot_kind, return_type), list[FunctionIR]]`.
With probability 0.2 per slot per macro generation, inject the
top-fitness variant from the bank into the local micro-pop. After
every micro-evolve, the local elite is contributed back to the bank.

---

## 5. Implementation plan with gating tests

Each step has:

* **Inputs** — what code/files exist before the step
* **Outputs** — what code/files exist after
* **Gating test** — exact pytest selector + acceptance criterion
* **Rollback condition** — what to revert on failure

No step starts until the previous step's gating test is green.

---

### Step S0 — Type lattice + Pool Health baseline

**Inputs**: current code base, audit findings.

**Outputs**:
* `evolution/types_lattice.py` (read-only utilities)
* `tests/test_types_lattice.py`
* `tests/test_pool_health.py` (runs against the **current** pool —
  this is the baseline; many tests will fail. Document failures.)
* `code_review/baseline_pht_failures.md` — recorded list of failing
  current-pool entries.

**Gating test**:
```
pytest -q tests/test_types_lattice.py            # 100% pass
pytest -q tests/test_pool_health.py              # baseline; record output
```

`test_types_lattice.py` must achieve 100 % pass.
`test_pool_health.py` produces a baseline failure manifest; this is the
"known broken state" we will fix in S1.

**Rollback**: none — this step is purely additive.

---

### Step S1 — Initial pool fix

Fix every pool entry that fails PHT-1 through PHT-7 in the baseline.
Apply Sec. 3.2 audit actions. Add Category B (linear algebra) and
Category C (probability) primitives per Sec. 3.1. Const-lift hardcoded
literals and outer-loop literals via mechanical AST passes (Sec. 3.2
items 4–5). Deduplicate per item 3 and item 6.

**Outputs**:
* Cleaned `evolution/ir_pool.py`, `evolution/skeleton_library.py`
* New `evolution/const_lifter.py` (AST pass for const- and loop-lifting)
* Tests `tests/test_const_lifter.py`, `tests/test_pool_primitives.py`,
  `tests/test_mimo_baselines.py`

**Gating test**:
```
pytest -q tests/test_const_lifter.py             # 100% pass
pytest -q tests/test_pool_primitives.py          # 100% pass
pytest -q tests/test_mimo_baselines.py           # 100% pass (PHT-6)
pytest -q tests/test_pool_health.py              # 100% pass (was baseline-failing)
```

**Acceptance budgets**:
* `test_mimo_baselines.py` runtime ≤ 30 min on workstation (parallelisable
  with `-n auto`); SER curves saved to `results/baseline_pht/` as JSON.
* No silent removal of pool entries — every removal must be logged with
  reason in `code_review/pool_changes.md`.

**Rollback**: revert `ir_pool.py` and `skeleton_library.py` to git HEAD;
keep new test files for reuse.

---

### Step S2 — Type-aware GP framework (offline, no integration)

Implement `evolution/gp_operators.py` and `evolution/gp_population.py`
per Sec. 4.2–4.4. Rewrite `evolution/random_program.py` per Sec. 4.4.
**Do not yet wire into the macro evolution loop.**

**Outputs**:
* `evolution/gp_operators.py`, `evolution/gp_population.py`
* Rewritten `evolution/random_program.py`
* Tests `tests/test_gp_operators.py`, `tests/test_gp_population.py`,
  `tests/test_random_program_v2.py`

**Gating tests** (large-scale):

```
pytest -q tests/test_gp_operators.py             # 100% pass
pytest -q tests/test_random_program_v2.py        # 100% pass
pytest -q tests/test_gp_population.py            # 100% pass
```

`test_gp_population.py` includes a **convergence benchmark**:

* For each of 10 representative slots (covering every type in the
  lattice), run `MicroPopulation(size=64, generations=20)` from a
  randomized initial pop on a synthetic fitness function with known
  optimum.
* Acceptance: for ≥ 8/10 slots, the population's best fitness reaches
  ≥ 80 % of the optimum within 20 generations.

`test_random_program_v2.py`:

* For each ProgramSpec in the registry, generate 200 random programs.
* Acceptance: ≥ 95 % execute on a probe input without raising;
  ≥ 90 % return values whose dtype/shape matches the spec.

Runtime budget: ≤ 20 min total.

**Rollback**: leave new files; do not import them from existing modules
yet.

---

### Step S3 — FII construction + GNN feature extension

Implement `evolution/fii.py` per Sec. 2.1–2.2. Extend
`gnn_pattern_matcher.py` `_NODE_DIM` and `ir_to_graph` per Sec. 2.3.
Plumb `FIICache` into `AlgorithmEntry` (in `pool_types.py`).

GNN switch is **gated by a config flag** `use_fii_view: bool = False`
in `EvolutionConfig` so we can A/B test.

**Outputs**:
* `evolution/fii.py`
* Modified `gnn_pattern_matcher.py`, `pool_types.py`
* Tests `tests/test_fii.py`

**Gating tests**:

```
pytest -q tests/test_fii.py                      # 100% pass
pytest -q tests/test_gnn_graph_construction.py   # 100% pass (existing + new asserts)
```

`test_fii.py` requirements:

* For every detector template in the audited pool (≥ 8), inline the
  default genome and assert:
  * Resulting FII has zero `algslot` ops.
  * Materialized callable from FII produces **bit-identical**
    output to materialised callable from the original genome on a
    fixed probe input set (1000 random MIMO realizations).
  * Provenance map has 1 entry per op; back-mapping is consistent.
  * Roundtrip `inline → modify-nothing → reconstitute` reproduces the
    original genome.

* Cache invariants:
  * Cache hit returns identical `(ir, prov)` references.
  * Bumping a slot's `best_idx` invalidates the cache.

Runtime budget: ≤ 10 min.

**Rollback**: set `use_fii_view = False`; FII code remains but is dead.

---

### Step S4 — Region enumeration on FII; graft Case I + II only

Switch `gnn_pattern_matcher` and `algorithm_engine` to enumerate regions
on FII when `use_fii_view = True`. Implement `evolution/graft_dispatch.py`
that, after a graft is selected, classifies it as Case I, II, or III.
For S4, only Case I and Case II are accepted; Case III grafts are
**rejected and logged**.

Implement Case I back-mapping in `evolution/graft_dispatch.py`:

* Identify all ops in the host region; group by `from_slot_id`.
* For each affected slot, compute the new variant IR and append to the
  slot's micro-population (the GP framework from S2 takes ownership).

**Outputs**:
* `evolution/graft_dispatch.py`
* Modified `evolution/algorithm_engine.py`
* Tests `tests/test_graft_dispatch.py`,
  `tests/test_run_short_with_fii.py`

**Gating tests**:

```
pytest -q tests/test_graft_dispatch.py           # 100% pass
pytest -q tests/test_run_short_with_fii.py       # 100% pass (10-gen smoke)
```

`test_run_short_with_fii.py`:

* Run the full evolution loop for 10 macro generations with
  `use_fii_view=True`, micro-pop=32 (reduced for runtime), 4 hosts,
  GNN graft proposals enabled.
* Acceptance:
  * No crashes.
  * ≥ 1 effective Case I graft observed (i.e. GNN proposed a
    region strictly inside a slot, and it was accepted by the
    macro-fitness gate).
  * ≥ 1 effective Case II graft observed.
  * Case III rate logged but not accepted.

Runtime budget: ≤ 30 min.

---

### Step S5 — Case III dissolution + slot rediscovery

Implement Case III in `evolution/slot_dissolution.py`. When a graft
crosses slot boundaries, the affected slots are inlined into
`structural_ir` and removed from `slot_populations`. Implement
`evolution/slot_rediscovery.py` per Sec. 2.6 and run it every 20 macro
generations.

**Outputs**:
* `evolution/slot_dissolution.py`, `evolution/slot_rediscovery.py`
* Modified `evolution/algorithm_engine.py`
* Tests `tests/test_dissolution.py`, `tests/test_rediscovery.py`,
  `tests/test_run_30gen_with_fii.py`

**Gating tests**:

```
pytest -q tests/test_dissolution.py              # 100% pass
pytest -q tests/test_rediscovery.py              # 100% pass
pytest -q tests/test_run_30gen_with_fii.py       # 100% pass (30-gen run)
```

`test_dissolution.py`:

* Hand-construct a genome and a graft that triggers Case III.
* Assert: post-dissolution, materialised callable produces output
  equivalent to the FII-applied graft (within float tolerance);
  `slot_populations` no longer contains the dissolved slots; SSA is
  valid.

`test_rediscovery.py`:

* Take a flattened (post-dissolution) `structural_ir`.
* Run `rediscover_slots` and assert: ≥ 1 region found per template,
  every found region's signature is type-inferable, no overlap with
  existing slots, can be re-instantiated as a fresh `MicroPopulation`.

`test_run_30gen_with_fii.py` (objective verification — corresponds to
**O5**):

* 30 macro generations; pool size 8; GNN graft enabled; full FII path.
* Acceptance:
  * Effective graft rate ≥ baseline + 50 % relative.
  * Of effective grafts, ≥ 40 % are Case I or III (i.e. NOT
    pure structural / loop-severing).
  * Population mean SER at SNR = 16 dB strictly improves over the
    initial pool's mean SER by ≥ 0.5 dB.

Runtime budget: ≤ 90 min.

---

### Step S6 — Full integration & long-horizon validation

Set defaults: `use_fii_view = True`, micro-pop size = 64, micro
generations = 5. Re-enable cross-host SlotVariantBank.

Run the full 200-generation training and compare against the
80-generation baseline.

**Gating test**:

```
pytest -q tests/test_run_long_smoke.py           # 100% pass (60-gen sample)
python train_gnn.py --gens 200 --proposals 500 --pool-size 141 --n-trials 5 \
    --warmstart-gens 20 --snr-start 16 --snr-target 20 \
    --effective-margin 0.01 --effectiveness-snr 16 \
    --use-fii-view --micro-pop 64 --micro-gens 5 \
    > results/gnn_training/run_v2.log 2>&1
```

**Acceptance for the 200-gen run** (post-hoc analysis required, not
pytest):

* Effective graft rate plateau ≥ 8 % (vs prior 1–5 %).
* Best individual SER at SNR = 16 dB ≤ best of pool baseline by
  ≥ 1 dB.
* Distribution of effective graft cases: Case I ≥ 25 %, Case II ≥ 30 %,
  Case III ≥ 15 %, "loop-severing" pattern (provenance-tagged) ≤ 30 %.
* Slot rediscovery has emitted ≥ 5 new auto-slots over the run, with
  micro-evolution showing positive fitness improvement in ≥ half of them.

If any of the four numeric criteria fails, root-cause analysis is
required before declaring S6 complete; do not silently weaken the
acceptance bar.

---

## 6. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| FII inflates op count by 5–10×, slowing GNN training | High | Medium | Cap `max_region_size` and `max_region_ops` already in place; profile in S3, increase batch size if needed; FII is computed lazily and cached |
| Case III dissolution destabilises algorithms (SER regression) | Medium | High | Dissolution only proceeds if post-graft macro-eval ≥ pre-graft; otherwise reverted |
| Slot rediscovery picks meaningless cohesive regions | Medium | Low | Multiple structural filters (cohesion ratio, type inference); rediscovery is opportunistic, system functions without it |
| Type lattice is too restrictive, blocks valid programs | Medium | Medium | Lattice has an explicit `top` type; operators may fall back to it; track fallback rate as a metric |
| GP-pop-64 × micro-gens-5 × 5 SNRs × 20 trials per fitness eval makes per-generation runtime infeasible | High | High | Subprocess pool already in place; per-individual eval is parallel; if needed, drop trials to 10 and SNRs to 1 with EMA; profile in S2 |
| Pool health test (PHT-6) reveals existing skeleton templates that simply do not work and removing them shrinks pool below useful size | Medium | Medium | S1 explicitly allows removal with logging; if pool < 30 entries after cleanup, generate replacements via the type-aware random-program generator from S2 (deferred; flag in `pool_changes.md`) |
| GNN provenance feature degenerates to a one-hot prior over time as buckets stabilize | Low | Medium | Hash buckets are 16-wide; ≥ 14 slot kinds → collisions guaranteed; periodically re-hash with a new salt every 50 macro gens |

---

## 7. Out-of-scope (explicitly deferred)

* New detector families beyond Sec. 3.1 — only what is needed to
  validate baselines.
* GNN architecture changes (attention heads, depth, etc.) — keep
  current GAT until S6 results justify changes.
* Multi-task curriculum (MIMO sizes, modulation orders) — current
  16×16 16-QAM remains the only target.
* Distributed / GPU-accelerated micro-evaluation — single-workstation
  CPU is the deployment target.

---

## 8. Memory / state tracking

Per the workspace's research memory protocol
(`research/memory/state.json`, `experiment-log.md`, `decision-history.md`):

* Each step's start and completion is logged with timestamp in
  `research/memory/experiment-log.md`.
* Decision rationale (e.g. acceptance bar adjustments) goes into
  `research/memory/decision-history.md`.
* Any pool entry removal/replacement goes into
  `code_review/pool_changes.md` with full justification.
* The 200-gen run results go into `experiment-log.md` and the
  comparative analysis into `code_review/s6_results.md`.

---

## 9. Approval gates summary

```
S0  ──►  S1  ──►  S2  ──►  S3  ──►  S4  ──►  S5  ──►  S6
 │        │        │        │        │        │        │
 └ types  └ pool   └ GP     └ FII    └ Case   └ Case   └ 200-
   pass     PHT      conv     bit-     I+II     III      gen
            green    bench    exact    smoke    + redis  numeric
                                                          targets
```

No step starts until the prior gating test suite is **fully green**.
Failures are root-caused, fixed, and re-run; never bypassed.
