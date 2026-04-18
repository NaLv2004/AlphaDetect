# 重构计划状态审计与下一步修改方案

> **日期**: 2026-04-19
> **上下文**: 审计 `refactor_plan.md` 中每一项计划的完成状态，评估现有"双重进化引擎"与目标（模式匹配驱动的跨算法嫁接进化）之间的差距，给出详细修改路线图。

---

## 第一部分：refactor_plan.md 各项完成状态

### P0 任务（数据结构定义）

| 计划项 | 状态 | 详情 |
|--------|------|------|
| `AlgorithmEntry` 数据类 | ✅ 已定义 | `pool_types.py` L140，含 algo_id/ir/source/trace/runtime_values/factgraph/fitness/generation/provenance/tags/level/slot_tree/slot_fitness |
| `GraftProposal` 数据类 | ✅ 已定义 | `pool_types.py` L176，含 proposal_id/host_algo_id/region/contract/donor_algo_id/donor_ir/dependency_overrides/port_mapping/confidence/rationale |
| `DependencyOverride` 数据类 | ✅ 已定义 | `pool_types.py` L165，含 target_value/new_dependencies/reason |
| `PatternMatcherFn` 类型别名 | ✅ 已定义 | `pool_types.py` L241: `Callable[[list[AlgorithmEntry], int], list[GraftProposal]]` |
| `SlotPopulation` 数据类 | ✅ 已定义 | `pool_types.py`，含 slot_id/spec/variants/fitness/best_idx/source_variants/best_variant/best_fitness |
| `AlgorithmGenome` 数据类 | ✅ 已定义 | `pool_types.py`，含 structural_ir/slot_populations/constants/generation/parent_ids/graft_history/tags/metadata |
| `AlgorithmFitnessEvaluator` 抽象类 | ✅ 已定义 | `pool_types.py`，含 evaluate()/evaluate_batch()/evaluate_single_result() |
| `AlgorithmEvolutionConfig` 配置 | ✅ 已定义 | `pool_types.py`，含 pool_size/n_generations/micro_pop_size/micro_generations/micro_mutation_rate 等 |
| `GraftRecord` | ✅ 已定义 | `pool_types.py`，含 generation/host_algo_id/donor_algo_id/proposal_id/region_summary/new_slots_created |

**P0 小结：100% 完成。** 所有核心数据类型已在 `pool_types.py` 中定义。

---

### P1 任务（核心实现）

| 计划项 | 状态 | 详情 |
|--------|------|------|
| `materialize()` | ✅ 已实现 | `materialize.py`：source-level materialization，替换 `__slot_op_id__(args)` 占位符为具体 slot 实现 |
| `materialize_with_override()` | ❌ 未实现 | 计划中用于微层评估时"固定其他 slot、只变化目标 slot"的功能，当前不存在 |
| `ir_to_callable()` | ✅ 已实现 | `materialize.py`：source → exec → callable |
| `materialize_to_callable()` | ✅ 已实现 | `materialize.py`：materialize + ir_to_callable 一步完成 |
| `graft_general()` | ❌ 未实现 | 计划中 call-based 通用嫁接引擎（rewriter.py），当前只有硬编码的 `_graft_bp_summary()` 和 `_graft_bp_tree_runtime_update()` |
| `AlgorithmFitnessEvaluator` 具体实现 | ✅ 已实现 | `mimo_evaluator.py`: `MIMOFitnessEvaluator` — 16×16 16QAM 蒙特卡洛评估 |

**P1 小结：4/6 完成（67%）。** `materialize_with_override` 和 `graft_general` 未实现。

---

### P2 任务（引擎）

| 计划项 | 状态 | 详情 |
|--------|------|------|
| `AlgorithmEvolutionEngine` 宏层循环 | ⚠️ 部分实现 | `algorithm_engine.py` 存在，但**没有使用 `PatternMatcherFn`**，宏层仅做 tournament+crossover+mutation，不做嫁接提案 |
| `_evolve_slots()` 微层进化 | ✅ 已实现 | `algorithm_engine.py`: `_micro_evolve()` + `_micro_step()` — 对每个 genome 的每个 SlotPopulation 做变异/交叉 |
| `_graft_pass()` | ⚠️ 简化版 | 仅做"最佳 genome 的 slot → 最差 genome 的 slot"跨个体迁移，**不是** refactor_plan 中的 PatternMatcher → GraftProposal → graft_general 流程 |
| `apply_dependency_overrides()` | ❌ 未实现 | 依赖覆盖（改变数据流依赖）完全未实现 |

**P2 小结：1/4 完成（25%）。** 引擎框架存在但缺少核心功能。

---

### P3 任务（高级功能）

| 计划项 | 状态 | 详情 |
|--------|------|------|
| `discover_new_slots()` | ❌ 未实现 | 嫁接后发现新 slot。虽然有 `slot_discovery.py`（静态/动态 slot 发现），但未集成到嫁接流程 |
| `infer_slot_spec()` | ❌ 未实现 | 从 IR 推断 slot 类型签名。`slot_discovery.py` 中的 `_spec_from_contract()` 是最接近的，但未被使用 |
| 带 AlgSlot 的 donor 嫁接 | ❌ 未实现 | 依赖 `graft_general()` |

**P3 小结：0/3 完成（0%）。**

---

### P4 任务（优化）

| 计划项 | 状态 |
|--------|------|
| Inline pass（将 call-based 嫁接展开为 inline ops） | ❌ 未实现 |

---

### 基础设施（IR 层面）

| 模块 | 计划修改 | 实际状态 |
|------|----------|----------|
| `algorithm_ir/ir/model.py` | 不变 | ✅ 未修改 |
| `algorithm_ir/frontend/ir_builder.py` | 不变 | ✅ 未修改（但有严格的 AST 限制，已知约束） |
| `algorithm_ir/regeneration/codegen.py` | 不变 | ⚠️ 已修改：添加了 `MatMult: @` 和 slot opcode 的 call 渲染 |
| `algorithm_ir/grafting/rewriter.py` | 新增 `graft_general()` | ❌ 未修改，仍是硬编码的 BP-specific 嫁接 |
| `algorithm_ir/region/contract.py` | 扩展 `dependency_overrides` | ❌ 未修改 |
| `algorithm_ir/runtime/` | 不变 | ✅ 未修改 |
| `algorithm_ir/factgraph/` | 不变 | ✅ 未修改 |

---

### 测试状态

| 测试文件 | 数量 | 状态 |
|----------|------|------|
| 原有测试 (tests/) | 182 | ✅ 全部通过 |
| 新增 IR 进化测试 (test_ir_evolution.py) | 38 | ✅ 全部通过 |
| **总计** | **220** | **✅ 全部通过** |

| 测试覆盖 | 有测试 | 已通过 |
|----------|--------|--------|
| ir_pool 模板编译 (8 detector) | ✅ | ✅ |
| AlgSlot 转换 | ✅ | ✅ |
| Slot 默认实现编译 (9 个) | ✅ | ✅ |
| build_ir_pool() | ✅ | ✅ |
| materialize (source + callable) | ✅ | ✅ |
| MIMOFitnessEvaluator 构造 | ✅ | ✅ |
| AlgorithmEvolutionEngine init_population | ✅ | ✅ |
| **PatternMatcher 驱动嫁接** | ❌ 无测试 | — |
| **graft_general()** | ❌ 无测试 | — |
| **materialize_with_override()** | ❌ 无测试 | — |
| **apply_dependency_overrides()** | ❌ 无测试 | — |
| **完整进化循环 (E2E)** | ❌ 无测试 | — |
| **跨算法嫁接** | ❌ 无测试 | — |

---

## 第二部分：现有"双重进化引擎"的真实评估

### 现在存在一个真正的双重进化引擎吗？

**答：形式上存在，实质上不是 refactor_plan 描述的那个引擎。**

#### 现有引擎做了什么

`AlgorithmEvolutionEngine` 确实实现了二层结构：

1. **微层（Micro）**：✅ 存在。`_micro_evolve()` 对每个 genome 的每个 `SlotPopulation` 做 mutation + crossover，生成新 variant 并 trim。
2. **宏层（Macro）**：⚠️ 存在但极度简化。`_breed_macro()` 做 tournament select + 跨 genome 的 slot population 交叉 + 随机 slot mutation。**没有 PatternMatcher，没有 GraftProposal，没有 structural graft。**
3. **嫁接（Graft）**：⚠️ `_graft_pass()` 只是把最佳 genome 的 slot variant 复制给最差 genome——这是 **slot 级别的迁移**，不是 **结构级别的嫁接**。

#### 计划中的引擎应该做什么

```
每一代:
  1. PatternMatcher 分析池中所有算法（包括静态 IR、运行时轨迹、FactGraph）
  2. PatternMatcher 提出 GraftProposal：
     - "stack_decoder 的 score 计算区域可以被 BP 的消息传递替代"
     - "KBEST 的 prune 可以被 LMMSE 的线性滤波替代"
  3. graft_general() 执行嫁接：在 host IR 中插入 call donor_fn(...)
  4. 嫁接可能引入新 slot + DependencyOverride
  5. 微层为新 slot 初始化种群，继续进化
```

#### 差距总结

| 功能 | 计划 | 现状 | 差距 |
|------|------|------|------|
| 宏层结构进化 | PatternMatcher → GraftProposal → graft_general() 改变控制流骨架 | 仅做 slot 级别 tournament+crossover，骨架不变 | **本质差距** |
| PatternMatcher 集成 | `AlgorithmEvolutionEngine.__init__` 接收 `pattern_matcher: PatternMatcherFn` | 引擎不接收 PatternMatcherFn，没有这个参数 | **完全缺失** |
| 运行时轨迹驱动 | 每代执行算法→收集 trace→build FactGraph→传给 PatternMatcher | 引擎不执行算法，不收集 trace，不 build FactGraph | **完全缺失** |
| 跨算法结构嫁接 | graft_general() 将 donor 的完整函数 IR 以 call 方式插入 host | 只有 slot variant 的复制粘贴 | **完全缺失** |
| 依赖覆盖 | apply_dependency_overrides() 改变数据流依赖 | 不存在 | **完全缺失** |
| materialize_with_override | 微层评估时固定其他 slot 只变化目标 slot | 不存在，微层不做真正的评估 | **部分缺失** |

**结论：现有引擎是一个"多骨架 slot 进化器"，不是 refactor_plan 中描述的"模式匹配驱动的跨算法嫁接进化引擎"。差距巨大。**

---

## 第三部分：目标功能与现有代码的差距分析

用户的目标（原文摘要）：

> evolution engine 在一开始接收一大批已知算法（EP, BP, KBEST, stack decoder, AMP, LMMSE, GTA 等）。这些算法可能不带 slot，也可能带。每一轮进化中，有一个模式匹配器 P，对当前算法池的每个算法给出建议：某个算法 A_i 中存在一个区域 R(A_i) 可以被另一个算法结构 A' 替代。模式匹配器不但可以提出替代区域，还可以提出新的输入-输出依赖。例如从 stack decoder 的运行轨迹抽取动态搜索树，将 BP 框架放到树上替代 metric 计算。模式匹配器作为函数参数传入 engine。

### 差距 1：PatternMatcher 未集成到引擎

**现状**：`PatternMatcherFn` 类型已定义，但 `AlgorithmEvolutionEngine` 的构造函数没有 `pattern_matcher` 参数，进化循环中也从不调用它。

**所需修改**：
- 引擎构造函数增加 `pattern_matcher: PatternMatcherFn | None = None`
- 主循环中每 N 代调用 `self.pattern_matcher(entries, gen)` 获取 `list[GraftProposal]`
- 对每个 proposal 调用 `_execute_graft()` 产生新 genome

### 差距 2：graft_general() 完全缺失

**现状**：`algorithm_ir/grafting/rewriter.py` 只有 `_graft_bp_summary()` 和 `_graft_bp_tree_runtime_update()`——针对 BP 算法硬编码的嫁接函数。没有通用的 `graft_general()`.

**所需修改**：实现 call-based 通用嫁接引擎：
1. Clone host IR
2. 识别 region 中的 ops
3. 在 region 位置插入 `call donor_fn(mapped_args...)` op
4. 将 donor 的返回值重绑定到 region 原来的 exit_values
5. 删除 region 中被替换的 ops
6. Rebuild

### 差距 3：运行时轨迹收集与 FactGraph 构建未集成

**现状**：`algorithm_ir/runtime/interpreter.py` 有 `execute_ir()`，`algorithm_ir/factgraph/builder.py` 有 `build_factgraph()`。但这些**从未被进化引擎调用**。

**所需修改**：
- 每代评估后，对代表性个体执行 `execute_ir()` 收集 trace + runtime_values
- 构建 FactGraph
- 打包到 `AlgorithmEntry` 传给 PatternMatcher
- 注意：fitness 评估走 `materialize_to_callable()` exec path（快），trace 收集走 `execute_ir()` interpreter path（慢但提供运行时信息）。两条路径共存，各司其职。

### 差距 4：apply_dependency_overrides() 缺失

**现状**：`DependencyOverride` 数据类存在，但没有任何代码消费它。`BoundaryContract` 中也没有 `dependency_overrides` 字段。

**所需修改**：
- `BoundaryContract` 增加 `dependency_overrides` 字段
- 实现 `apply_dependency_overrides()` 修改 host IR 的数据流
- 集成到 `graft_general()` 的 Step 2

### 差距 5：materialize_with_override() 缺失

**现状**：`materialize()` 始终使用 best_variant。微层进化中需要"临时替换一个 slot 来评估候选 variant"的能力。

**所需修改**：
- 新增 `materialize_with_override(genome, override={slot_id: variant_ir})` → callable
- 在 `_micro_step()` 中使用它做单 variant 评估

### 差距 6：微层没有做真正的适应度评估

**现状**：`_micro_step()` 只做 mutation/crossover/trim，**不评估新 variant 的适应度**。`pop.fitness` 始终是 `float("inf")`。
- Trim 基于 `best_idx`（永远是 0）+ 随机保留，不是基于适应度。

**所需修改**：
- `_micro_step()` 需要 materialize + evaluate 每个新 variant
- 更新 `pop.fitness[i]` 和 `pop.best_idx`
- 这是引擎能否工作的**关键缺失**

### 差距 7：AlgorithmEntry 的构造未实现

**现状**：`AlgorithmEntry` 定义了丰富的字段（ir, trace, runtime_values, factgraph, slot_tree 等），但引擎中从不构造 `AlgorithmEntry` 对象。`AlgorithmGenome.to_entry()` 方法不存在。

**所需修改**：
- `AlgorithmGenome` 增加 `to_entry()` 方法
- 引擎在调用 PatternMatcher 前将 population 转换为 `list[AlgorithmEntry]`

### 差距 8：引擎不接受原始（无 slot）算法

**现状**：`build_ir_pool()` 生成的所有 genome 都已经有 slot。用户希望支持"不带任何显式 slot"的原始算法也能参与进化。

**所需修改**：
- 支持 `SlotPopulation == {}` 的 genome（纯粹的原始算法）
- PatternMatcher 负责在这些算法上发现可替代区域
- 嫁接后可能引入新 slot

---

## 第四部分：修改计划

### 整体架构概览

**核心原则：IR 是一切进化的唯一介质，Python 仅作为执行层。所有嫁接操作必须在 FunctionIR 层面直接操作 ops/values/blocks，不得经由 Python 源码做中间转换。**

**双重进化架构：结构嫁接（宏层）与 slot 子种群进化（微层）是两个不同层次但同步进行的进化过程，在每一代中并行执行。**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AlgorithmEvolutionEngine                         │
│                                                                     │
│  每一代同步执行两个进化轨道:                                          │
│                                                                     │
│  ┌─── 轨道 A: 结构嫁接 (宏层) ─────────────────────────────────┐   │
│  │  PatternMatcher 分析 list[AlgorithmEntry]                     │   │
│  │  (含 IR 结构 + 运行时 trace + FactGraph)                      │   │
│  │       ↓                                                        │   │
│  │  输出 list[GraftProposal]: 跨算法结构替换建议                  │   │
│  │       ↓                                                        │   │
│  │  graft_general() 在 FunctionIR 层面执行 op-level surgery       │   │
│  │  (region 定位 → ops 删除 → call op 插入 → 依赖覆盖 → rebuild) │   │
│  │       ↓                                                        │   │
│  │  产出: 结构上全新的 AlgorithmGenome (可能含新 slot)             │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─── 轨道 B: Slot 子种群进化 (微层) ─────────────────────────┐   │
│  │  对每个 AlgorithmGenome 的每个 SlotPopulation:              │   │
│  │    IR-level mutation/crossover → materialize_with_override   │   │
│  │    → evaluate → 更新 fitness → sub-population 选择存活       │   │
│  │  Slot 进化不改变 structural_ir 骨架，只优化可替换组件          │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  两个轨道产出合并 → 全员评估 → 淘汰 → 下一代                        │
└─────────────────────────────────────────────────────────────────────┘
```

### Phase 1：修复微层评估（使现有引擎真正可运行）

**优先级：P0  |  预计工作量：中**

目标：让 `_micro_step()` 真正评估 variant 的适应度，而不是全部标记为 `inf`。

1. **实现 `materialize_with_override(genome, override_map)`**
   - 文件：`evolution/materialize.py`
   - 功能：和 `materialize()` 类似，但对指定 slot 使用 override 的 variant
   - 返回 callable

2. **修改 `_micro_step()` 评估新 variant**
   - 文件：`evolution/algorithm_engine.py`
   - 当前：mutation/crossover → append → trim（无评估）
   - 修改后：mutation/crossover → `materialize_with_override` → `evaluator.evaluate_single_result()` → 更新 fitness → 基于 fitness 的 trim

3. **测试**
   - `test_micro_step_evaluates_fitness`：验证新 variant 的 fitness 不再是 inf
   - `test_micro_step_updates_best_idx`：验证 best_idx 被正确更新
   - `test_materialize_with_override`：验证 override 功能正确

### Phase 2：集成 PatternMatcher 到引擎

**优先级：P1  |  预计工作量：中**

目标：引擎接受 PatternMatcherFn，每 N 代调用它获取嫁接建议。

1. **修改 `AlgorithmEvolutionEngine.__init__`**
   - 新增参数 `pattern_matcher: PatternMatcherFn | None = None`
   - 新增参数 `test_inputs_fn: Callable[[], list] | None = None`（用于生成测试用例以收集 trace）

2. **新增 `AlgorithmGenome.to_entry(fitness)` 方法**
   - 文件：`evolution/pool_types.py`
   - 将 genome 的 structural_ir、slot_tree、cached trace/factgraph 打包为 AlgorithmEntry

3. **修改 `run()` 主循环**
   - 文件：`evolution/algorithm_engine.py`
   - 在宏层阶段，若 `self.pattern_matcher is not None`：
     ```python
     entries = [g.to_entry(f) for g, f in zip(self.population, self.fitness)]
     proposals = self.pattern_matcher(entries, self.generation)
     for proposal in proposals:
         child = self._execute_graft(proposal)
         new_genomes.append(child)
     ```

4. **新增 `_execute_graft(proposal)` 方法**
   - 调用 `graft_general()` 在 FunctionIR 层面执行 op-level surgery
   - 将 GraftProposal 中的 region 替换为 donor IR 的 call op
   - 处理 DependencyOverride 修改 IR 数据流
   - 发现嫁接后新增的 slot，初始化对应 SlotPopulation
   - 返回新 AlgorithmGenome
   - 注意：Phase 2 与 Phase 3 必须同步实现，`_execute_graft` 不可用桩实现

5. **测试**
   - `test_engine_accepts_pattern_matcher`
   - `test_pattern_matcher_called_each_gen`
   - `test_dummy_pattern_matcher_produces_offspring`
   - `test_null_pattern_matcher_engine_still_works`

### Phase 3：实现 graft_general()（纯 IR 层 op surgery）

**优先级：P1  |  预计工作量：高**

目标：通用的 call-based 嫁接引擎，**完全在 FunctionIR 层面操作 ops/values/blocks**，不经过任何 Python 源码中间步骤。

**核心原则：IR 是一切进化的唯一介质。graft_general() 操作的输入和输出都是 FunctionIR。嫁接 = IR 层的 op surgery。Python 源码仅在最终执行时由 materialize() 生成。**

1. **新增 `algorithm_ir/grafting/graft_general.py`**

   ```python
   @dataclass
   class GraftArtifact:
       """嫁接操作的完整产出。"""
       ir: FunctionIR              # 嫁接后的新 IR
       new_slot_ids: list[str]     # 嫁接引入的新 slot
       replaced_op_ids: list[str]  # 被删除的原始 ops
       call_op_id: str             # 插入的 call op

   def graft_general(
       host_ir: FunctionIR,
       proposal: GraftProposal,
   ) -> GraftArtifact:
       """
       纯 IR 层 call-based 嫁接（op-level surgery）：

       1. deepcopy(host_ir)
       2. 定位 proposal.region 中的 ops（通过 region.op_ids）
       3. 分析 region boundary：
          - entry_values: region 外部定义、region 内部使用的 values
          - exit_values:  region 内部定义、region 外部使用的 values
       4. 将 donor_ir 注册为可调用函数（存储在 host IR 的
          function table 或 constants 中）
       5. 创建新的 call op：
          - opcode = "call"
          - inputs = [donor_fn_ref] + [port_mapping[v] for v in entry_values]
          - outputs = exit_values 的替代 value ids
       6. 重绑定：将 region 外部对 exit_values 的所有引用
          指向 call op 的 outputs
       7. 删除 region 中的所有原始 ops
       8. 如果有 dependency_overrides：
          apply_dependency_overrides() 修改 IR 数据流
          — 在 call op 的 inputs 中追加新依赖
          — 必要时在 host IR 中添加新的 arg values
       9. 重建 op 拓扑排序 / block 结构
      10. validate_ir() 验证结果 IR 的合法性
       """
   ```

2. **IR 层 op surgery 的关键操作**

   需要实现以下 FunctionIR 层面的基本操作（部分可能已存在）：

   - `find_region_boundary(ir, op_ids) -> (entry_values, exit_values)`
     分析 region 的数据流边界：哪些 values 从外部流入，哪些流出到外部
   - `create_call_op(ir, callee_ref, args, result_types) -> OpInfo`
     在 IR 中创建一个 call op，返回其 op_id 和输出 value ids
   - `rebind_uses(ir, old_vid, new_vid)`
     将 IR 中所有对 old_vid 的使用替换为 new_vid
   - `remove_ops(ir, op_ids)`
     从 IR 中删除指定的 ops，同时清理其产生的 dead values
   - `topological_sort(ir)`
     重建 ops 的执行顺序
   - `validate_ir(ir) -> list[str]`
     验证 IR 的合法性（所有 uses 有对应 defs，无悬挂引用，类型一致）

3. **新增 `apply_dependency_overrides(ir, overrides, call_op_id)`**
   - 纯 IR 层操作
   - 对每个 DependencyOverride：
     - 在 host IR 中定位 `target_value`
     - 创建或查找 `new_dependencies` 对应的 values
     - 修改 call op 的 inputs 列表，追加新依赖
     - 如果新依赖来自 host 外部（需要新参数），则扩展 `ir.arg_values`
   - 所有操作直接修改 FunctionIR 的 ops/values 字典

4. **集成到 `_execute_graft()` 方法**
   - 调用 `graft_general()` 产生 `GraftArtifact`
   - 对 `artifact.new_slot_ids` 中每个新 slot：
     - 调用 `discover_new_slots()` + `infer_slot_spec()` 确定类型签名
     - 初始化对应的 `SlotPopulation`（default_impl = donor 的对应片段）
   - 构造新的 `AlgorithmGenome`（structural_ir = artifact.ir）
   - 记录 `GraftRecord` 到 graft_history

5. **测试**
   - `test_graft_general_lmmse_to_bp`：将 LMMSE 的 regularizer 区域替换为 BP 的消息传递
   - `test_graft_general_preserves_non_region_ops`：非 region ops 不变
   - `test_graft_general_exit_values_rebound`：region 的 exit values 被正确重绑定到 call op 输出
   - `test_graft_general_with_dependency_override`：新增依赖正确追加到 call op inputs
   - `test_graft_general_donor_with_slots`：donor 带 AlgSlot，嫁接后 host 继承这些 slot
   - `test_graft_general_validates_result`：嫁接后 IR 通过 validate_ir()
   - `test_graft_general_region_boundary_analysis`：entry/exit values 分析正确

### Phase 4：运行时轨迹集成

**优先级：P2  |  预计工作量：高**

目标：让 PatternMatcher 能看到算法的运行时行为（trace + FactGraph），以实现运行时驱动的跨算法嫁接。

1. **两条执行路径的协调**
   - `materialize_to_callable()` → exec path（快，用于 fitness 评估）
   - `execute_ir()` → interpreter path（慢，产生 RuntimeEvent trace）
   - 两条路径必须共存：fitness 评估走 exec path，trace 收集走 interpreter path

2. **Trace 收集策略**
   - 每代仅对代表性个体（如 top-K 和新嫁接产物）执行 interpreter path 收集 trace
   - 使用少量采样输入（不需要完整 Monte Carlo），目的是分析结构行为而非精确 BER
   - 收集的 trace 用于构建 FactGraph → 传给 PatternMatcher

3. **Instrumented codegen（高性能替代方案）**
   - 修改 `codegen.py` 的 `emit_python_source()`，支持可选的 tracing instrumentation
   - 在每个变量赋值后插入 `__trace_record__(var_name, value)` 调用
   - 生成的 Python 函数接受一个 `__tracer__` 上下文参数
   - 优点：与 exec path 完全一致，性能远好于 interpreter path
   - 作为 `execute_ir()` 的高性能替代

4. **AlgorithmGenome 增加 trace/factgraph 缓存字段**
   - `_cached_trace: list[RuntimeEvent] | None`
   - `_cached_runtime_values: dict[str, RuntimeValue] | None`
   - `_cached_factgraph: FactGraph | None`
   - 在 `to_entry()` 中填充
   - 缓存在 structural_ir 或 slot populations 变更后失效

### Phase 5：示例 PatternMatcher 实现

**优先级：P2  |  预计工作量：中**

目标：提供一个可工作的 PatternMatcher 示例，演示跨算法嫁接。

1. **`StaticStructurePatternMatcher`（基于静态 IR 分析）**
   - 分析每个算法的 IR 结构
   - 策略示例：
     - 识别"迭代计算"模式（while loop + 矩阵运算）→ 建议用另一个迭代算法的 body 替代
     - 识别"排序/选择"模式 → 建议用不同的排序/选择策略替代
     - 识别"距离计算"模式 → 建议用不同的 metric 替代

2. **`RandomGraftPatternMatcher`（随机嫁接，用于测试）**
   - 随机选择 host/donor 对
   - 随机选择 host 中的一个 region（一组连续的 ops）
   - 用 donor 的部分结构替换
   - 主要用于验证 graft_general() 的鲁棒性

3. **`ExpertPatternMatcher`（专家知识驱动）**
   - 硬编码一些已知有效的嫁接：
     - "Stack decoder + BP message passing"
     - "KBEST + MMSE initial estimate"
     - "EP + AMP denoiser"
   - 作为 baseline 验证嫁接框架的正确性

### Phase 6：端到端进化实验

**优先级：P3  |  预计工作量：中**

1. **E2E 测试**：
   - 16×16 16QAM 系统
   - 初始池：LMMSE, ZF, OSIC, KBEST, STACK, BP, EP, AMP
   - PatternMatcher: ExpertPatternMatcher
   - 运行 50 代
   - 验证：(a) 嫁接产生新结构, (b) 新结构可执行, (c) BER 有改善

2. **Benchmark**：
   - 对比：仅微层进化 vs. 微层+宏层嫁接进化
   - 指标：BER, 收敛速度, 结构多样性

---

## 第五部分：实施路线图

```
Phase 1 (P0): 修复微层评估
  ├── materialize_with_override()
  ├── _micro_step() 评估 + fitness 更新
  └── 测试: 3 个新测试

Phase 2+3 (P1): PatternMatcher 集成 + graft_general() [必须同步实现]
  ├── algorithm_ir/grafting/graft_general.py (纯 IR 层 op surgery)
  ├── IR 层基本操作: find_region_boundary, create_call_op, rebind_uses, remove_ops
  ├── apply_dependency_overrides() (纯 IR 层)
  ├── AlgorithmGenome.to_entry()
  ├── Engine 构造函数 + run() 循环修改
  ├── _execute_graft() 完整实现（调用 graft_general）
  └── 测试: 11 个新测试

Phase 4 (P2): 运行时轨迹
  ├── interpreter path trace 收集（代表性个体）
  ├── Instrumented codegen（高性能替代）
  ├── AlgorithmGenome 缓存字段
  └── to_entry() 填充 trace/factgraph

Phase 5 (P2): PatternMatcher 示例
  ├── RandomGraftPatternMatcher
  ├── ExpertPatternMatcher
  └── 测试: 跨算法嫁接 demo

Phase 6 (P3): E2E 实验
  ├── 16×16 16QAM benchmark
  ├── 对比实验
  └── 结果分析
```

---

## 第六部分：设计原则

### 原则 1：IR 是一切进化的唯一介质

**所有嫁接操作必须在 FunctionIR 层面直接操作 ops/values/blocks，不得引入 Python 源码作为中间转换步骤。**

分层职责：
- **IR 层**（FunctionIR）：所有进化操作的唯一介质 — 嫁接、变异、交叉、区域分析、依赖覆盖、slot 发现。所有结构变换直接操作 `ir.ops`、`ir.values`、`ir.blocks`。
- **Python 层**：纯执行层。`materialize()` 将 IR 翻译为 Python 源码仅用于 `exec()` 执行和 fitness 评估。Python 源码不参与任何进化操作。
- **codegen.py**：IR → Python 的单向翻译器。只在最终执行时调用，不在进化循环中反向使用（不做 Python → IR 回编译）。

`graft_general()` 的完整流程：
```
FunctionIR (host) + GraftProposal
  → IR-level op surgery (region分析 → ops删除 → call op插入 → rebind → validate)
  → FunctionIR (grafted)
```
不涉及任何 `emit_python_source` → 修改源码 → `compile_source_to_ir` 的往返。

### 原则 2：双重进化是同步的两个轨道

Graft proposal（Region 的替换）和 slot 的进化是两个不同层次但**同步进行**的进化过程：

| 轨道 | 操作对象 | 操作方式 | 进化介质 |
|------|----------|----------|----------|
| **宏层（结构嫁接）** | AlgorithmGenome 的 structural_ir 骨架 | PatternMatcher → GraftProposal → graft_general() IR op surgery | FunctionIR |
| **微层（Slot 子种群）** | 每个 SlotPopulation 的 variant IR 集合 | mutation/crossover on FunctionIR → materialize → evaluate | FunctionIR |

两个轨道在每一代中并行执行：
1. 微层为每个 genome 的每个 slot 优化实现（sub-population evolution）
2. 宏层通过 PatternMatcher 发现跨算法嫁接机会，改变控制流骨架
3. 嫁接产生新 genome 时，新 slot 以默认实现初始化 sub-population
4. 下一代中微层继续优化新 slot 的 sub-population

两个轨道的交互是单向的：宏层嫁接可能创建新 slot → 微层为新 slot 初始化种群。微层的结果不改变宏层骨架。

### 原则 3：PatternMatcher 支持任意粒度的区域操作

PatternMatcher 必须从一开始就支持在任意粒度上提出 GraftProposal：

- **完整 slot 替换**：将一个 slot 的实现替换为另一个算法的对应组件
- **函数内任意区域替换**：将函数内一组连续 ops 组成的 region 替换为另一个算法的片段
- **跨 block 区域替换**：region 可以跨越 if/while 的 block 边界
- **带依赖覆盖的替换**：PatternMatcher 可以在提出区域替换的同时，指定新的输入-输出依赖（DependencyOverride）

所有粒度由统一的 `graft_general()` 处理，区别仅在于 `GraftProposal.region` 的范围大小。

### 原则 4：IR builder 限制的处理

当前 `compile_source_to_ir` 对 Python 语法的限制（keyword args, lambda, list comprehension, tuple unpacking, BoolOp, Slice 等不支持）仅影响**初始算法模板的编译**（一次性操作），不影响进化过程。

- 初始模板编译时，使用 `_template_globals()` 中的 helper 函数绕过限制
- 进化过程中所有操作直接操作 FunctionIR，不经过 IR builder
- 长期方案：扩展 IR builder 支持更多 AST 节点，减少 helper 函数的需求

---

## 附录：文件修改清单

| 文件 | Phase | 修改类型 | 内容 |
|------|-------|----------|------|
| `evolution/materialize.py` | 1 | 新增函数 | `materialize_with_override()` |
| `evolution/algorithm_engine.py` | 1,2 | 修改 | `_micro_step()` 评估, `__init__` 加 pattern_matcher, `run()` 加嫁接阶段, `_execute_graft()` |
| `evolution/pool_types.py` | 2 | 新增方法 | `AlgorithmGenome.to_entry()` |
| `algorithm_ir/grafting/graft_general.py` | 3 | 新建文件 | `graft_general()`, `apply_dependency_overrides()` |
| `algorithm_ir/region/contract.py` | 3 | 修改 | `BoundaryContract` 增加 `dependency_overrides` 字段 |
| `evolution/pattern_matchers.py` | 5 | 新建文件 | `RandomGraftPatternMatcher`, `ExpertPatternMatcher` |
| `tests/unit/test_ir_evolution.py` | 1-5 | 扩展 | 新增测试覆盖所有新功能 |
| `evolution/run_evolution.py` | 2 | 修改 | 支持传入 PatternMatcher |
