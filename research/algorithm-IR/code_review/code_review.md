# Algorithm-IR 严格代码复审报告：Typed GP 与单一 IR 表示

**日期**: 2026-04-25  
**审查对象**: `research/algorithm-IR`  
**审查背景**: 另一 AI 已根据此前 `code_review.md` 和 `typed_gp_remediation_plan.md` 修改代码。本文重新审查当前代码是否满足“GP 只能直接作用在 `FunctionIR` 上，AST/源码不得参与 mutation、crossover、selection 或 slot variant 表示，源码只允许在执行阶段出现”的要求。  
**最终结论**: 当前代码有明显进步，但仍未达标。它删除了旧的源码往返 mutation，并新增了 `evolution/gp/*` 框架；但 `slot_evolution` 主链路仍保留 AST/source gate、legacy 常数扰动路径和 `source_variants` 语义路径。新的 typed GP 目前主要是 IR 点突变和值重连，还不是完整的 typed GP 框架。

---

## 1. 总体判断

当前实现应被视为“向纯 IR typed GP 迁移的中间版本”，不能视为完成。

已改善的部分：

- `evolution/operators.py` 已删除 `_mutate_via_recompile()`。
- `insert/delete/swap_lines` 这三条源码编辑再重新编译的 mutation 路径已从 `mutate_ir()` 中移除。
- 旧的 `swap_lines` 引用未定义变量 `deletable` 的 bug 已随该路径删除。
- 新增 `evolution/gp/`，包含 canonical IR hash、region resolver、contract、operator base、typed mutation、micro population。
- 新 GP 相关单测通过。
- `build_ir_pool()` 当前仍能生成 validator 通过的 pool。

仍未解决的核心问题：

- `slot_evolution.py` 仍在 evolution 层使用 `emit_python_source()` 和 `ast.parse()` 做合法性 gate。
- `source_variants` 仍是活跃语义路径，materialization 会优先使用源码 variant。
- 新 `region_resolver` 虽然声称覆盖 `221/221` slot，但 `apply_slot_variant()` 仍走旧 provenance-only 路径，14 个 legacy `slot_op` slot 实际全部无法应用。
- 新 typed GP operators 只有点突变和值替换，没有 typed insert/delete/subtree replacement/crossover/synthesis。
- typed contract 当前仍是 `any -> any` 占位，不是真正的 slot signature 类型约束。
- frontend loop/phi def-use 错误仍未修复，integration grafting demo 仍失败。

---

## 2. 本轮验证结果

### 2.1 新 GP 相关测试

命令：

```powershell
conda run -n AutoGenOld python -m pytest tests/unit/test_gp_foundation.py tests/unit/test_gp_typed_operators.py tests/unit/test_gp_micro_population.py tests/unit/test_slot_evolution.py -q
```

结果：

```text
43 passed, 1 skipped
```

说明：新增 GP 基础设施有基本测试覆盖，且当前测试集通过。但这些测试主要证明“新增路径能运行”，不能证明它已经满足完整 typed GP 要求。

### 2.2 frontend / regression 测试

命令：

```powershell
conda run -n AutoGenOld python -m pytest tests/unit/test_frontend.py tests/unit/test_regression_p0.py -q
```

结果：

```text
8 failed, 15 passed
```

仍失败的问题包括：

- loop / branch / for 编译后的 `use_ops mismatch`
- `for_sum`, `for_nested`, `for_with_branch`, `use_range` 等直接 frontend 编译路径仍不满足 `validate_function_ir()==[]`
- `list` annotation 仍回归为 `list<any>`，与既有测试期望不一致
- clone roundtrip 仍保留错误 def-use

典型错误：

```text
Value v_0 use_ops mismatch: stored=['op_5', 'op_5', 'op_5'] expected=['op_5', 'op_5']
Value v_2 use_ops mismatch: stored=['op_6', 'op_6'] expected=['op_6']
```

### 2.3 integration / cross-lang 测试

命令：

```powershell
conda run -n AutoGenOld python -m pytest tests/integration tests/cross_lang -q
```

结果：

```text
2 failed, 13 passed
```

两个失败均来自 grafting demo：

- `test_stack_decoder_bp_grafting_demo`
- `test_stack_decoder_runtime_tree_bp_grafting_demo`

失败原因仍是 `compile_function_to_ir()` 生成的 host IR 被 `validate_function_ir()` 拒绝，错误模式仍是重复 / 陈旧 `use_ops`。

### 2.4 pool health

探针结果：

```text
pool 91
invalid 0
```

说明：`build_ir_pool(np.random.default_rng(42))` 当前生成的 91 个 genome 的 canonical IR 通过 validator。这是此前问题的真实改善。

### 2.5 slot region resolver 覆盖率

探针结果：

```text
slots 221
resolved 221
missing 0
tiers {'binding': 0, 'provenance': 207, 'slot_op': 14}
```

表面看，slot 覆盖率已从此前 `207/221` 提升到 `221/221`。但这不是完整修复，因为下一节说明 `slot_op` tier 无法真正被 `apply_slot_variant()` 使用。

---

## 3. 问题状态矩阵

| 此前问题 | 当前状态 | 结论 |
|---|---:|---|
 `build_ir_pool()` 生成 invalid flat IR | 已改善 | 当前 91 个 pool genome validator 全绿 |
 `operators.py` 源码 round-trip mutation | 已删除 | `_mutate_via_recompile()` 已移除 |
 `swap_lines` 未定义 `deletable` | 已消失 | 随源码 round-trip 路径删除 |
 frontend loop/phi def-use 错误 | 仍存在 | 8 个 frontend/regression 失败 |
 integration grafting demo | 仍失败 | 2 个 integration/cross-lang 失败 |
 slot coverage 207/221 | 表面改善，实际未闭环 | resolver 221/221，但 apply 对 14 个 slot_op 全失败 |
 slot evolution 只扰动 float 常数 | 部分改善 | 新增点突变和值重连，但仍非完整 typed GP |
 单一 IR 表示原则 | 仍未满足 | `source_variants`、AST source gate、legacy path 仍存在 |
 typed contract | 未完成 | 当前为 `any -> any` 占位 |
 behavior / held-out gate | 未完成 | 有字段/注释，但未形成强 gate |

---

## 4. P0 问题

## P0-1. `slot_op` tier 只是“可解析”，不能真正应用

**文件**:

- `evolution/gp/region_resolver.py`
- `evolution/slot_evolution.py`
- `evolution/gp/population.py`

**现象**:

新 `region_resolver` 支持三层解析：

1. explicit binding
2. `_provenance.from_slot_id`
3. legacy `slot` op

代码位置：

```python
# evolution/gp/region_resolver.py
slot_op_ids = _scan_slot_ops(ir, short)
if slot_op_ids:
    return SlotRegionInfo(..., tier="slot_op")
```

这使审计结果显示：

```text
resolved 221 / 221
```

但真正执行 `apply_slot_variant()` 时，仍调用旧函数：

```python
sids = map_pop_key_to_from_slot_ids(genome, pop_key)
if not sids:
    return None
```

也就是说 `apply_slot_variant()` 只支持 provenance，不支持 `slot_op` tier。

**实测结果**:

```text
slotop_apply success 0 / 14
```

全部失败的 slot 包括：

```text
kbest.expand
kbest.prune
bp.bp_sweep
bp.final_decision
importance_sampling.score
importance_sampling.hard_decision
particle_filter.proposal
particle_filter.score
particle_filter.hard_decision
soft_sic.ordering
soft_sic.soft_estimate
soft_sic.hard_decision
turbo_linear.soft_estimate
turbo_linear.hard_decision
```

**影响**:

这些正是 tree search、BP/message passing、soft detection、particle filtering 等核心非线性算法族。当前 telemetry 如果只看 resolver 覆盖率，会误以为这些 slot 已可进化；实际它们仍被 `apply_slot_variant()` 阻断。

**必须修改**:

1. `apply_slot_variant()` 必须改为使用 `resolve_slot_region()`，而不是 `map_pop_key_to_from_slot_ids()`。
2. 对 `tier="slot_op"` 必须实现真实替换策略：
   - 要么强制先 FII inline 成 provenance region；
   - 要么把 `slot` op 本身包装为可替换 `RewriteRegion`；
   - 要么在 pool admission 阶段禁止保留 legacy `slot` op，确保所有 slot 都有 provenance region。
3. 新增测试：

```python
for genome in build_ir_pool(...):
    for slot_key, pop in genome.slot_populations.items():
        assert resolve_slot_region(genome, slot_key) is not None
        assert apply_slot_variant(genome, slot_key, pop.variants[0]) is not None
```

这条测试必须覆盖上述 14 个 legacy slot。

---

## P0-2. `slot_evolution.py` 仍在 evolution 层使用源码和 AST gate

**文件**:

- `evolution/slot_evolution.py`

**问题代码**:

```python
from algorithm_ir.regeneration.codegen import emit_python_source
```

```python
def _source_compiles_with_resolved_names(source: str) -> bool:
    import ast
    tree = ast.parse(source)
    ...
```

```python
source = emit_python_source(flat_ir)
if not _source_compiles_with_resolved_names(source):
    return float("inf"), source
```

**为什么不符合要求**:

用户明确要求：

> GP 只能直接作用在 IR 上。不要出现任何 AST/源码。源码只应该在执行阶段出现。

当前代码虽然没有用源码做 mutation，但仍在 `slot_evolution` 层通过 AST 检查源码 name binding。这使 evolution gate 依赖 Python 源码语义，而不是 IR def-use/type/contract。

**必须修改**:

1. 删除 `_source_compiles_with_resolved_names()`。
2. `slot_evolution.py` 不应 import `ast`。
3. `slot_evolution.py` 不应直接用 source 做合法性 gate。
4. 若需要 name/use 检查，必须实现 IR-level validator，例如：
   - all input values exist
   - all value uses are dominated by defs
   - all region exits are bound
   - no dangling values after graft
   - no unresolved global/call target values
5. `emit_python_source()` 只能在 evaluator 内部出现。`evaluate_slot_variant()` 可以把 `flat_ir` 交给 evaluator，由 evaluator 负责：
   ```text
   FunctionIR -> emit_python_source -> exec -> fitness
   ```
   而不是在 slot evolution 里先生成源码再做 AST gate。

---

## P0-3. `source_variants` 仍破坏单一 IR 表示

**文件**:

- `evolution/pool_types.py`
- `evolution/materialize.py`
- `evolution/ir_pool.py`
- `evolution/slot_evolution.py`
- `evolution/gp/population.py`

**问题代码**:

`SlotPopulation` 仍有：

```python
source_variants: list[str | None] = field(default_factory=list)
```

并且注释明确说：

```python
# When present, materialization prefers source_variants[i] over
# emit_python_source(variants[i]).
```

`materialize.py` 中仍存在语义优先级：

```python
if pop.source_variants and best_idx < len(pop.source_variants):
    variant_source = pop.source_variants[best_idx]
else:
    variant_source = emit_python_source(best_ir)
```

**为什么严重**:

这意味着同一个 slot variant 可以同时有：

- `FunctionIR`
- Python source string

且执行时可能优先使用 source string。这直接违反“同一程序不要有多种表示”的要求。即使 GP operators 本身只改 IR，只要 materialization 可能读 `source_variants`，系统语义仍不是单一 IR。

**必须修改**:

1. 从 `SlotPopulation` 中移除 `source_variants` 的语义作用。
2. 若短期不能删字段，可改为 debug-only，不允许 materialization 使用：
   ```python
   source_variants_debug: list[str | None]
   ```
3. `materialize.py` 必须只从 `FunctionIR` 生成 source。
4. pool admission 中无法编译为 IR 的 slot default 不应以 source variant 进入 population；应拒绝、降级为不可进化 slot，或修 frontend 直到可编译。
5. 新增静态测试：GP/evolution semantic path 禁止读取 `source_variants`。

---

## P0-4. 新 typed GP 不是完整 GP，只是 IR 点突变集合

**文件**:

- `evolution/gp/operators/typed_mutations.py`
- `evolution/gp/operators/base.py`
- `evolution/gp/population.py`

**当前注册 operators**:

```text
mut_argswap
mut_binary_swap
mut_compare_swap
mut_const
mut_const_to_var
mut_unary_flip
```

这些都是局部点突变或值重连。它们相对“只扰动 float 常数”有进步，但仍不具备 genetic programming 的结构搜索能力。

当前缺失：

- `mut_insert_typed`
- `mut_delete_typed`
- `mut_subtree_replace`
- `cx_subtree_typed`
- `cx_block_typed`
- typed random subgraph synthesis
- primitive injection
- loop/body mutation
- state-carrier aware mutation
- multi-output slot mutation

**影响**:

系统仍无法发现新的算法结构，例如：

- 新 message update 公式
- 新 damping pipeline
- 新 tree metric composition
- 新 soft estimate / hard decision pipeline
- 新 iterative refinement block
- 新 linear algebra primitive composition

**必须修改**:

实现真正 typed GP operator set，至少包括：

```text
mut_insert_typed:
  在支配关系合法的位置插入类型兼容 op/subgraph，并将下游 use 重连到新输出。

mut_delete_typed:
  删除 contract-safe 子图，用 live-in 或 passthrough value 重连 live-out。

mut_subtree_replace:
  选择一个单输出或多输出 region，用同 signature 的合成子图替换。

cx_subtree_typed:
  在同 slot population 内选取两个 type-compatible region 做 subtree crossover。

cx_block_typed:
  对 loop/branch block 做 contract-aware replacement。

mut_primitive_inject:
  从 domain primitive library 注入 MMSE step、projection、normalization、damping、metric update 等 IR 子图。
```

每个 operator 必须满足：

- 不读取 source
- 不读取 AST
- 不调用 `compile_source_to_ir()`
- 直接修改 `FunctionIR`
- 修改后重建 def-use
- 通过 `validate_function_ir()`
- child IR hash 必须不同
- slot signature 必须不变

---

## P0-5. typed contract 当前是 `any -> any` 占位，不是真正 typed GP

**文件**:

- `evolution/gp/population.py`
- `evolution/gp/contract.py`

**问题代码**:

```python
return SlotContract(
    slot_key=slot_key,
    short_name=short,
    input_ports=(TypedPort("in", "any"),),
    output_ports=(TypedPort("out", "any"),),
    complexity_cap=complexity_cap,
)
```

这不是严格类型系统，只是为了让框架能运行。

**影响**:

当前 operators 主要依赖：

- 原 op 自身输入输出形态
- `value.type_hint`
- `validate_function_ir()` 兜底

但它没有从 slot spec / region live-in live-out 推导真实 contract。因此不能保证：

- slot 输入参数顺序正确
- 输出 arity 正确
- 多输出 slot 正确绑定
- state carrier 正确处理
- matrix/vector/scalar 类型兼容
- shape 约束合理

**必须修改**:

`SlotContract` 必须从以下来源构建：

1. `SlotPopulation.spec`
2. `SlotDescriptor.spec`
3. resolved region 的 `entry_values` / `exit_values`
4. `FunctionIR.values[*].type_hint`
5. loop/state carrier 分析
6. optional runtime shape probe

禁止长期使用 `any -> any` 作为 production typed GP contract。

---

## P0-6. frontend def-use 错误仍未修复

**文件**:

- `algorithm_ir/frontend/ir_builder.py`
- `algorithm_ir/ir/validator.py`

**问题位置**:

```python
# ir_builder.py
self.state.values[backedge_value].use_ops.append(phi_op_id)
```

仍出现在：

- `_compile_while()`
- `_compile_for()`

而 `_register_op()` 已经维护 `use_ops`：

```python
for inp in inputs:
    self.state.values[inp].use_ops.append(op.id)
```

这会导致 duplicate `use_ops`。

**当前测试状态**:

```text
8 failed, 15 passed
```

**必须修改**:

1. 删除 loop backedge 更新中的手动 `use_ops.append()`。
2. 在 frontend build 结束前统一调用 `rebuild_def_use(func_ir)`。
3. `compile_function_to_ir()` 和 `compile_source_to_ir()` 必须保证对支持语法子集产出 valid IR。
4. 恢复以下测试全绿：
   - `test_frontend.py`
   - `test_regression_p0.py`
   - `test_grafting_demo.py`

---

## P0-7. behavior gate / held-out gate 尚未形成真实选择约束

**文件**:

- `evolution/gp/operators/base.py`
- `evolution/gp/population.py`

`OperatorStats` 有字段：

```python
n_probe_rejected
n_noop_behavior
n_evaluated
n_improved
```

但当前 micro population 主要逻辑仍是：

```python
ser = evaluate_slot_variant(...)
if finite and ser < ...
```

没有严格实现：

- behavior hash
- fixed probe output change
- train/val/held-out split
- commit 前 held-out 复核
- no-op behavior rejection

**必须修改**:

1. 每个 child 评估前后保存 behavior signature。
2. 若 IR hash 变了但 fixed probes 输出不变，应计入 `n_noop_behavior`，不能当作有效 structural novelty。
3. micro fitness 和 commit fitness 使用不同 seed。
4. commit 前必须在 held-out set 上验证。
5. telemetry 必须区分：
   - structural accepted
   - apply failed
   - validator rejected
   - runtime rejected
   - behavior unchanged
   - performance failed
   - improved

---

## 5. P1 问题

## P1-1. legacy constant-perturbation path 仍可通过参数启用

**文件**:

- `evolution/slot_evolution.py`
- `tests/unit/test_gp_micro_population.py`

当前：

```python
def step_slot_population(..., use_typed_gp: bool = True)
```

当传入 `use_typed_gp=False` 时，仍执行旧的 constant-perturbation-only path。测试中还专门保留：

```python
test_legacy_path_still_works_when_use_typed_gp_false
```

**问题**:

这给旧路径留下了回流入口。既然要求是严格 pure IR typed GP，应删除 legacy path，而不是保留可选参数。

**修改建议**:

- 删除 `use_typed_gp` 参数。
- 删除旧 constant-only step body。
- 若需要常数扰动，将其作为 `gp/operators/mut_const.py` 中的一个普通 typed operator。

---

## P1-2. `const_lifter.py` 是 AST 工具，当前只做 audit，但应限制边界

**文件**:

- `evolution/const_lifter.py`
- `evolution/skeleton_library.py`

`const_lifter.py` 使用 AST 重写 source。当前它在 `skeleton_library.get_extended_specs()` 中只用于 audit，不直接改写 GP population。这比在 GP 内使用 AST 安全。

但它仍位于 `evolution/` 下，容易被未来误用为 GP mutation 工具。

**修改建议**:

- 明确移动到 `frontend` / `preprocessing` / `audit` namespace。
- 文件头写明：不得在 GP runtime 中调用。
- 静态测试禁止 `slot_evolution.py`, `gp/*`, `algorithm_engine.py` 调用 `const_lifter.lift_source()`。

---

## P1-3. docs 与代码状态不一致

**文件**:

- `evolution/README.md`
- `code_review/typed_gp_remediation_plan.md`
- `evolution/gp/__init__.py`

部分文档仍提到旧的 `_mutate_via_recompile()` 或声明 typed GP 功能已经完整落地。当前实际仅实现了早期 operator set。

**修改建议**:

- README 应更新为：
  - legacy source-roundtrip mutation 已删除；
  - typed GP framework 尚处于 S0/S1/S2 部分落地；
  - structural GP operators 尚未完成。
- 避免文档让维护者误以为 typed GP 已完成。

---

## 6. 必须执行的整改清单

### S0. 删除表示混乱残留

- 删除 `slot_evolution.py` 中 `_source_compiles_with_resolved_names()`。
- `slot_evolution.py` 不再 `import ast`。
- `slot_evolution.py` 不再使用 source 做合法性 gate。
- 删除 `use_typed_gp=False` legacy path。
- `source_variants` 不再参与 materialization。

验收：

```text
rg "ast.parse|ast.unparse|compile_source_to_ir|source_variants|use_typed_gp=False" evolution/slot_evolution.py evolution/gp evolution/algorithm_engine.py
```

核心 GP/evolution 路径不应出现这些语义依赖。

### S1. 修复 slot application 与 coverage

- `apply_slot_variant()` 改用 `resolve_slot_region()`。
- 支持 `tier="slot_op"` 的真实替换。
- 或者在 pool admission 阶段强制所有 slot 变成 flat provenance region。

验收：

```text
resolve_slot_region: 221/221
apply_slot_variant(default): 221/221
```

尤其必须覆盖：

```text
kbest.*
bp.*
soft_sic.*
turbo_linear.*
particle_filter.*
importance_sampling.*
```

### S2. 修复 frontend def-use

- 删除 `_compile_while()` / `_compile_for()` 中重复 `use_ops.append()`。
- build 结束前统一 `rebuild_def_use()`。
- 恢复 frontend 和 integration 测试。

验收：

```text
tests/unit/test_frontend.py: pass
tests/unit/test_regression_p0.py: pass
tests/integration/test_grafting_demo.py: pass
```

### S3. 实现真实 typed contract

- 从 `SlotPopulation.spec` / `SlotDescriptor.spec` 构造 `SlotContract`。
- 从 region entry/exit value 推导类型。
- 支持多输入、多输出、state carrier。
- 移除 production 路径中的 `any -> any`。

验收：

```text
每个 slot 的 contract.input_ports / output_ports 与实际 variant arg/return 一致
多输出 slot 不再依赖 pick_real_exit_value() 启发式
```

### S4. 实现结构型 typed GP operators

最低要求：

- `mut_insert_typed`
- `mut_delete_typed`
- `mut_subtree_replace`
- `cx_subtree_typed`
- `mut_primitive_inject`

验收：

```text
至少 50% 以上 accepted children 的 structural hash 变化不是 const/operator attr-only change
每个 operator 有独立单测
每个 operator 直接操作 FunctionIR，不触碰源码/AST
```

### S5. 加入 behavior / held-out gate

- fixed probe behavior hash
- behavior unchanged rejection
- train/val/held-out split
- commit 前 held-out 复核

验收：

```text
telemetry 中能区分:
structural_changed
behavior_changed
performance_improved
heldout_accepted
heldout_rejected
```

---

## 7. 最终验收标准

只有同时满足以下条件，才能认为代码满足要求：

1. GP mutation/crossover/selection/slot variant 表示只使用 `FunctionIR`。
2. `slot_evolution.py` 和 `evolution/gp/*` 不使用 AST/source 做语义 gate。
3. `source_variants` 不参与 materialization。
4. `build_ir_pool()` 生成的所有 genome valid。
5. 所有 slot population 都可 `resolve_slot_region()` 且可 `apply_slot_variant()`。
6. frontend/regression/integration 测试全绿。
7. typed contract 不再是 `any -> any` 占位。
8. 至少实现 insert/delete/subtree/crossover 中的结构型 GP operators。
9. telemetry 能证明 child 不是 no-op，不只是重复评估父程序。
10. commit 前有 held-out gate。

当前代码尚未满足这些标准。

---

## 8. 简短结论

这次修改方向正确，但完成度不足。

可以认可的进展是：旧的源码 round-trip mutation 已从 `operators.py` 删除，新 GP 框架雏形已经建立，pool health 维持良好。

不能认可为完成的是：`slot_evolution` 仍没有完全摆脱源码/AST，`source_variants` 仍破坏单一表示，legacy slot 实际仍无法应用，typed contract 只是占位，GP operators 仍缺少结构搜索能力。

下一步不应继续堆训练或 GNN，而应先完成上述 S0-S5。否则系统会继续出现“看似 evolved，实际只在局部 attr/constant 上扰动，且大量核心 slot 无法真正替换”的问题。
