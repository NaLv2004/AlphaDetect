# Plan: IR-Based Evolution Framework with Dual Codegen

## TL;DR

Refactor mimo-push-gp's BP evolution into a **fully generic** `evolution/` framework where algorithm-IR is the sole representation medium. Each individual wraps N `FunctionIR`s + evolvable constants. **Dual codegen** (IR→Python + IR→C++) is compulsory from the start, backed by a **cross-language consistency test harness** that guarantees identical behavior. Skeletons may be provided with explicit slots OR fully concrete — the framework automatically discovers mutable regions via static/dynamic analysis. The MIMO BP stack decoder is a downstream **application** that plugs into the framework with zero framework-side awareness of BP specifics. Target: BER ≈ 2E-3 at 16×16 16QAM 24dB.

## Decisions

- **D1**: Evolve plain Python functions compiled to IR. Mutation/crossover on IR ops.
- **D2**: Codegen-only for per-generation evaluation (IR→callable). Grafting only for final artifact assembly.
- **D3 (revised)**: C++ codegen is **compulsory**. IR→C++ expression evaluator (serialized opcode interpreter, not per-program compilation). Cross-language consistency tests gate every commit.
- **D4**: 4 separate `FunctionIR`s per genome (one per BP program role).
- **D5**: Evolution framework is **100% generic** — zero BP/MIMO knowledge. All domain specifics live in `applications/mimo_bp/`.
- **D6**: `validation_fn` is **not hand-coded** — derived automatically from skeleton dependency analysis (entry/exit values, type constraints, data-flow invariants inferred by `region/` and `contract/`).
- **D7**: Unified static/dynamic analysis: static analysis (backward/forward slice, type inference from IR) and dynamic analysis (runtime trace, observed types from `infer_boundary_contract()`) are composable and share the same `BoundaryContract` output type.

## Module Dependency Graph

```
┌─────────────────────────────────────────────────────────────────┐
│ algorithm_ir/                                                   │
│   ir/model.py          ← FunctionIR, Op, Value, Block           │
│   ir/dialect.py        ← 22 IRDL typed ops                     │
│   frontend/ir_builder.py ← compile_function_to_ir()            │
│   regeneration/codegen.py ← emit_python_source(), emit_cpp_ops()│
│   region/selector.py   ← define_rewrite_region()               │
│   region/contract.py   ← infer_boundary_contract()             │
│   region/slicer.py     ← backward/forward_slice_by_values()    │
│   grafting/            ← match_skeleton(), graft_skeleton()     │
│   runtime/interpreter.py ← execute_ir()                        │
└────────────┬───────────────────────────────┬────────────────────┘
             │ IMPORTS                       │ IMPORTS
             ▼                               ▼
┌────────────────────────────────┐   ┌───────────────────────────────┐
│ evolution/  (GENERIC)          │   │ tests/cross_lang/             │
│   genome.py ─uses→ model      │   │   conftest.py                 │
│   operators.py ─uses→ model   │   │   test_consistency.py         │
│   random_program.py           │   │     ─uses→ codegen (Py+C++)   │
│     ─uses→ ir_builder         │   │     ─uses→ genome.to_callable │
│   skeleton_registry.py        │   │     ─uses→ genome.to_cpp_ops  │
│     ─uses→ region/contract    │   └───────────────────────────────┘
│   slot_discovery.py           │
│     ─uses→ region/selector    │
│     ─uses→ region/contract    │
│     ─uses→ runtime/interp     │
│   engine.py ─uses→ above      │
│   fitness.py (abstract)       │
│   config.py (dataclass)       │
└────────────┬──────────────────┘
             │ APPLICATION IMPLEMENTS fitness.py
             ▼
┌────────────────────────────────────────────┐
│ applications/mimo_bp/   (DOMAIN-SPECIFIC)  │
│   bp_skeleton.py ─uses→ skeleton_registry  │
│   bp_decoder.py  (pure algorithm, no IR)   │
│   mimo_simulation.py (channel model)       │
│   evaluator.py ─uses→ engine, genome,      │
│     codegen, cpp_evaluator                 │
│   cpp_evaluator.py ─uses→ C++ DLL         │
│   stacks.py (TreeNode, SearchTreeGraph)    │
│   run_evolution.py (CLI entry point)       │
└────────────────────────────────────────────┘
```

**Anti-pattern guards** (to prevent AI from reinventing wheels):
- `evolution/*.py` MUST import from `algorithm_ir.ir.model` for `FunctionIR`/`Op`/`Value`/`Block` — never redefine them
- `evolution/operators.py` MUST operate on `FunctionIR` objects — never convert to a different representation for mutation
- `evolution/random_program.py` MUST use `algorithm_ir.frontend.ir_builder.compile_function_to_ir()` — never hand-build IR
- `applications/mimo_bp/evaluator.py` MUST use `evolution.genome.IRGenome.to_callable()` and `to_cpp_ops()` — never bypass codegen
- `evolution/skeleton_registry.py` MUST use `algorithm_ir.region.contract.infer_boundary_contract()` for validation — never hand-code validation functions

## Directory Structure

```
research/algorithm-IR/
  algorithm_ir/                         # EXISTING — do not restructure
    ir/model.py, dialect.py
    frontend/ir_builder.py
    regeneration/codegen.py             # EXTEND: add emit_cpp_ops()
    region/selector.py, contract.py, slicer.py
    grafting/matcher.py, rewriter.py, skeletons.py
    runtime/interpreter.py
  evolution/                            # NEW — generic evolution framework
    __init__.py
    config.py
    genome.py
    operators.py
    engine.py
    fitness.py
    skeleton_registry.py
    slot_discovery.py
    random_program.py
  applications/
    mimo_bp/                            # NEW — MIMO BP application
      __init__.py
      bp_skeleton.py
      bp_decoder.py
      mimo_simulation.py
      evaluator.py
      cpp_evaluator.py
      stacks.py
      run_evolution.py
      cpp/                              # C++ IR evaluator + BP decoder
        ir_eval.h, ir_eval.cpp
        bp_ir_decoder.h, bp_ir_decoder.cpp
        CMakeLists.txt / build.bat
  tests/
    cross_lang/
      conftest.py
      test_consistency.py
    test_evolution/
      test_genome.py
      test_operators.py
      test_random_program.py
      test_skeleton_registry.py
      test_slot_discovery.py
      test_engine.py
    test_mimo_bp/
      test_bp_decoder.py
      test_mimo_simulation.py
      test_evaluator.py
```

---

## Phases

### Phase 1: Core Evolution Framework + Dual Codegen Infrastructure

*No external dependencies. Foundation for everything.*

#### Step 1.1 — `evolution/config.py`

`EvolutionConfig` dataclass: `population_size`, `n_generations`, `tournament_size`, `elite_count`, `mutation_rate`, `crossover_rate`, `constant_mutate_sigma`, `stagnation_threshold`, `hard_restart_after`, `hall_of_fame_size`, `niche_radius`, `program_roles: list[str]`, `n_constants`, `constant_range`, `use_cpp: bool`, `seed`.

**Tests:** `test_config_defaults()`, `test_config_override()`, `test_config_serialization()` (to_dict/from_dict round-trip)

#### Step 1.2 — `evolution/fitness.py`

`FitnessResult`: `metrics: dict[str, float]` (generic, no BER/flops hardcoding) + `composite_score() → float` (weighted sum) + `is_valid: bool` + `__lt__` for sorting.
`FitnessEvaluator(ABC)`: `evaluate(genome) → FitnessResult`, `evaluate_batch(genomes) → list[FitnessResult]`.

**Tests:** `test_fitness_result_comparison()`, `test_fitness_result_metrics()`, `test_abstract_evaluator()` (ABC guard)

#### Step 1.3 — `evolution/genome.py`

`IRGenome`: `programs: dict[str, FunctionIR]`, `constants: np.ndarray`, `generation`, `parent_ids`. Methods: `clone()`, `to_source(role) → str` (calls `emit_python_source`), `to_callable(role) → callable` (codegen→compile→exec, cached, exception fallback), `to_cpp_ops(role) → list[int]` (calls `emit_cpp_ops`), `serialize()/deserialize()`, `structural_hash()`.

**Tests:** `test_genome_create()`, `test_genome_clone()` (independent copy), `test_to_source()` (syntactically valid), `test_to_callable()` (produces output), `test_to_cpp_ops()` (non-empty, well-formed), `test_serialize_roundtrip()`, `test_structural_hash()` (same→same, diff→diff)

#### Step 1.4 — `algorithm_ir/regeneration/codegen.py` — extend with `emit_cpp_ops()`

Add `emit_cpp_ops(func_ir) → list[int]`: serialize IR to flat opcode array for C++ stack interpreter. ~27 opcodes: `CONST_F64, LOAD_ARG, ADD, SUB, MUL, DIV, SQRT, ABS, NEG, EXP, LOG, TANH, MIN, MAX, LT, GT, LE, GE, EQ, IF_START, ELSE, ENDIF, WHILE_START, WHILE_END, RETURN, SAFE_DIV, SAFE_LOG`. Constants inline as `[CONST_F64, lo_bits, hi_bits]`. Also extend `emit_python_source()` with safe math wrappers + default return.

**Tests:** `test_emit_cpp_ops_simple()` (a+b → opcodes), `test_emit_cpp_ops_conditional()`, `test_emit_cpp_ops_loop()`, `test_emit_cpp_ops_constants()` (float64 bit pattern), `test_emit_python_safe_math()` (div-by-zero → 0.0), `test_roundtrip_python_codegen()` (compile→IR→codegen→exec → same result)

#### Step 1.5 — `tests/cross_lang/test_consistency.py`

Cross-language consistency harness: for each test case, `emit_python_source()` → exec → result_py vs `emit_cpp_ops()` → C++ eval → result_cpp. Assert `|py - cpp| < 1e-12`.

**Tests:** `test_consistency_arithmetic()` (20 random exprs), `test_consistency_nested()`, `test_consistency_conditional()`, `test_consistency_loop()`, `test_consistency_edge_cases()` (div-0, log(0), sqrt(-1)), `test_consistency_fuzz(seed)` (100 random programs × random inputs)

**Git sync**: `git commit -m "Phase 1a: evolution core + dual codegen + cross-lang tests"`

#### Step 1.6 — `evolution/skeleton_registry.py`

`ProgramSpec`: `name, param_names, param_types, return_type, slot_regions|None, constraints`.
`SkeletonSpec`: `skeleton_id, host_ir|None, program_specs, mode: "explicit_slots"|"auto_discover"`.
`SkeletonRegistry`: `register()`, `validate_program(role, func_ir) → list[str]` — auto-validation via IR analysis (arg count, param names, return type, data-flow dependency check via backward slice, depth constraints, `infer_boundary_contract()` for runtime type checks). NO hand-coded `validation_fn`.

**Tests:** `test_register_explicit_slots()`, `test_validate_correct_program()`, `test_validate_wrong_arg_count()`, `test_validate_wrong_return_type()`, `test_validate_unused_arg()` (not in backward slice), `test_validate_depth_constraint()`, `test_validate_genome()`

#### Step 1.7 — `evolution/slot_discovery.py`

Automatic slot discovery for fully concrete skeletons. `discover_slots(func_ir, sample_inputs=None, mode="auto") → list[SlotCandidate]` where `SlotCandidate = (region, contract, score, program_spec)`. Static path: enumerate SESE regions → filter by output type/port count → rank. Dynamic path: `execute_ir()` → runtime trace → `infer_boundary_contract()` → identify hot/stable regions. Auto-generates `ProgramSpec` from region boundary.

**Tests:** `test_discover_on_simple_function()`, `test_discover_preserves_io()`, `test_discover_ranking()`, `test_discover_with_runtime_trace()`, `test_explicit_slots_bypass()`, `test_discover_on_concrete_bp_host()` (find 4 meaningful slots)

#### Step 1.8 — `evolution/random_program.py`

`random_ir_program(spec, rng, max_depth=5) → FunctionIR`: generate random Python function source → `compile_function_to_ir()`. Expression tree of `{binary, unary, const, arg_ref}` nodes. Special loop-body template for aggregation specs.

**Tests:** `test_random_program_compiles()` (50 programs → valid IR), `test_random_program_callable()`, `test_random_program_matches_spec()`, `test_random_program_diversity()` (50 → ≥30 unique hashes), `test_random_program_with_loop()`

#### Step 1.9 — `evolution/operators.py`

`mutate_ir()`: point (swap opcode/const/inputs), insert, delete, constant_perturb. All clone-first, validate after.
`crossover_ir()`: block-level splice with type compatibility check.
`mutate_genome()`, `crossover_genome()`: per-role delegation + constant blending.

**Tests:** `test_mutate_point()`, `test_mutate_insert()`, `test_mutate_delete()`, `test_mutate_constant_perturb()`, `test_mutate_preserves_signature()`, `test_crossover_produces_valid_ir()` (20 runs), `test_crossover_inherits_from_both()`, `test_mutate_genome()`, `test_crossover_genome()`, `test_operator_determinism()` (same seed → same result)

#### Step 1.10 — `evolution/engine.py`

`EvolutionEngine(config, evaluator, registry)`: `initialize_population()`, `run()`, `step()`. Tournament selection, niche diversity via `structural_hash()`, stagnation detection → hard restart (keep hall of fame), constant hill-climbing for top-k, checkpointing, per-gen logging.

**Tests:** `test_engine_initialization()`, `test_engine_step()`, `test_engine_selection()` (statistical), `test_engine_stagnation_restart()`, `test_engine_hall_of_fame()`, `test_engine_checkpoint_restore()`, `test_engine_niche_diversity()`, `test_engine_with_mock_evaluator()` (5 gens, no crash)

**Git sync**: `git commit -m "Phase 1 complete: evolution framework + dual codegen + cross-lang tests"`

---

### Phase 2: C++ Expression Evaluator DLL

*Depends on Phase 1 step 1.4 (opcode format).*

#### Step 2.1 — C++ IR Expression Evaluator

`cpp/ir_eval.h` + `ir_eval.cpp`: stack-based interpreter, switch-case ~27 opcodes, `double stack[64]`, safe math matching Python semantics exactly. IF/ELSE/ENDIF via skip counters, WHILE via loop-back markers.

#### Step 2.2 — C++ BP Decoder Integration

`cpp/bp_ir_decoder.h` + `bp_ir_decoder.cpp`: tree-search BP decoder calling `ir_eval()` for f_down/f_up/f_belief/h_halt. C API export: `bp_ir_eval_dataset()`.

#### Step 2.3 — DLL Build + Python Bridge

`build.bat` (MSVC) + `cpp_evaluator.py` (ctypes): `evaluate_genome_cpp(genome, dataset) → FitnessResult`.

**Tests:** `test_cpp_eval_arithmetic()`, `test_cpp_eval_conditional()`, `test_cpp_eval_loop()`, `test_cpp_eval_safe_math()`, `test_cpp_bp_decoder_baseline()` (LMMSE/K-Best known BER), `test_cpp_python_consistency_100_programs()` (`|py-cpp| < 1e-10`), `test_cpp_bridge_batch()` (20 genomes)

**Git sync**: `git commit -m "Phase 2: C++ IR evaluator DLL + cross-lang consistency verified"`

---

### Phase 3: MIMO Stack-BP Application (Must be implemented based on Algorithm IR and evoluton engine. Must not invent wheels.)

*Depends on Phase 1 + Phase 2.*
Implement a MIMO detector that joins 2 algorithm skeletons : BP decoder and the stack decoder. Must refer to mimo-push-gp code base for algorithm details. The skeletons must be joined using interfaces provided by the evolution engine or the algorithm IR. Algorithm IR must be the substrate for evolution.
#### Step 3.1 — Port `stacks.py`

`TreeNode`, `SearchTreeGraph` from mimo-push-gp. Strip Push VM deps.

**Tests:** `test_tree_node_creation()`, `test_search_tree_expand()`, `test_search_tree_frontier()`, `test_tree_structure_matches_original()`

#### Step 3.2 — `mimo_simulation.py`

`qam16_constellation()`, `generate_mimo_sample()`, `ber_calc()`, `lmmse_detect()`, `kbest_detect()`, `build_dataset()`.

**Tests:** `test_qam16_constellation()`, `test_generate_mimo_sample()`, `test_ber_calc_perfect()`, `test_ber_calc_worst()`, `test_lmmse_detect()` (BER < 0.01 at 30dB), `test_kbest_detect()`, `test_build_dataset()`

#### Step 3.3 — `bp_skeleton.py`

4 `ProgramSpec` definitions registered with `SkeletonRegistry`. f_down(2 floats→float), f_up(list+int→float, loop), f_belief(3 floats→float), h_halt(2 floats→bool).

**Tests:** `test_bp_specs_registered()`, `test_bp_specs_validate_known_good()`, `test_bp_specs_reject_bad()`

#### Step 3.4 — `bp_decoder.py`

Port `StructuredBPDecoder` using plain callables (no Push VM). `detect(H, y, noise_var, constellation, f_down, f_up, f_belief, h_halt, log_constants, ...)`.

**Tests:** `test_decoder_with_identity_functions()`, `test_decoder_with_known_good_functions()` (BER ≈ 2E-3 at 24dB), `test_decoder_fault_detection()`, `test_decoder_flops_counting()`, `test_decoder_matches_original()`

#### Step 3.5 — `evaluator.py`

`MIMOBPEvaluator(FitnessEvaluator)`: IR→codegen→decoder→BER. Configurable metric weights. `evaluate_batch()` with C++ path.

**Tests:** `test_evaluator_random_genome()`, `test_evaluator_known_good()` (BER ≈ 2E-3), `test_evaluator_cpp_matches_python()` (`|ber_py-ber_cpp| < 1e-6`), `test_evaluator_batch()`, `test_evaluator_caching()`

#### Step 3.6 — `run_evolution.py`

CLI entry point with argparse. Wires engine + evaluator + registry. Logging, checkpoint, resume.

**Tests:** `test_run_evolution_smoke()` (2 gens, pop=5), `test_run_evolution_checkpoint()`, `test_run_evolution_seed_genome()`

**Git sync**: `git commit -m "Phase 3: MIMO BP application + full integration"`

---

### Phase 4: Automatic Slot Discovery (can first do phase 5 and 6, then come back to this)
This phase means the user might provide the evolution engine with several concrete algorithms with no slots.
*Depends on Phase 1.7 skeleton + Phase 3 concrete decoder.*

#### Step 4.1 — Static SESE region enumeration + filtering + ranking
#### Step 4.2 — Dynamic discovery via `execute_ir()` traces
#### Step 4.3 — Unified interface + auto `ProgramSpec` generation

**Tests:** `test_sese_enumeration()`, `test_filter_by_output_type()`, `test_ranking_order()`, `test_dynamic_discovery_with_trace()`, `test_stable_contract_detection()`, `test_combined_static_dynamic()`, `test_auto_generates_program_spec()`, `test_end_to_end_on_bp_host()` (discovers 4 meaningful slots)

**Git sync**: `git commit -m "Phase 4: automatic slot discovery (static + dynamic)"`

---

### Phase 5: Integration + BER Target

*Depends on all previous phases.*

5.1 Smoke test (random pop → no crash, both Python/C++), 5.2 Seed genome validation (known-good → BER ≈ 0.00238, cross-lang match), 5.3 Small evolution (pop=30, 20 gens, verify BER improves, measure C++ speedup), 5.4 Full evolution (pop=100, 500+ gens), 5.5 Achieve BER ≈ 2E-3 at 16×16 16QAM 24dB

**Tests:** `test_smoke_random_pop()`, `test_seed_genome_ber()`, `test_seed_genome_cross_lang()`, `test_evolution_improves()`, `test_evolution_cpp_speedup()` (>5x), `test_final_ber_target()` (BER ≤ 2.5E-3 at 24dB — **acceptance criterion**)

**Git sync**: `git commit -m "Phase 5: BER target achieved"`

---

### Phase 6: Learnable Region Discovery (GNN+RL+Genetic)

*Post-target. 6.1 IR→graph, 6.2 GNN scorer, 6.3 RL env, 6.4 Meta-evolution.*

**Git sync**: `git commit -m "Phase 6: learnable region discovery framework"`

---

## Key Technical Challenges

- **C1**: IR mutation must preserve CFG validity (op-level safest, structure-level rarest)
- **C2**: C++ evaluator throughput: ~3μs per function call → ~6min/gen for pop=100 (acceptable)
- **C3**: Cross-lang consistency: strict IEEE 754, identical safe-math fallbacks, identical control flow
- **C4**: Auto slot discovery quality: fallback to explicit `ProgramSpec` if auto doesn't find good slots
- **C5**: Static/dynamic analysis unification: both produce `BoundaryContract`, composable via `infer_boundary_contract(runtime_trace=None)`
