# Algorithm-IR 代码审计与重构计划

> 审计范围：`research/algorithm-IR/` 下全部代码  
> 审计日期：2025-01  
> 审计维度：(1) 愿景一致性 (2) 死代码/冗余 (3) 补丁式实现 (4) 运行时IR嫁接/变异

---

## 一、愿景一致性审计

### 1.1 愿景回顾

期望架构：
- 进化引擎在一开始接收一大批**已知算法**（EP, BP, KBEST, Stack Decoder, AMP, LMMSE, GTA 等），这些算法以**通用图/系统求解器**（不限于 MIMO）的形式存在
- 算法可能不带任何显式 slot，也可能带有 slot
- **模式匹配器 P** 是一个抽象函数 `P(pool) → list[GraftProposal]`，在进化的每一代对算法池每个算法给出建议：某个区域 R(Aᵢ) 可以被另一个算法结构 A' 替代
- 模式匹配器不但可以提出替代区域，还可以提出**新的输入-输出依赖**（dependency_overrides）
- 支持**运行时 trace 嫁接**（例如：stack decoder 的搜索树节点 → BP 消息传递树）

### 1.2 现状评估

#### ✅ 已实现的部分

| 能力 | 实现位置 | 状态 |
|------|---------|------|
| 8 种已知算法模板 (LMMSE/ZF/OSIC/KBest/Stack/BP/EP/AMP) | `ir_pool.py` — `_DETECTOR_SPECS` + 模板字符串 | ✅ 可用 |
| AlgSlot 机制 (slot_* 参数自动转为 AlgSlot op) | `ir_pool.py::convert_slot_calls_to_algslot()` | ✅ 可用 |
| 两层进化 (macro 骨架 + micro slot population) | `algorithm_engine.py` | ✅ 可用 |
| PatternMatcher 抽象接口 | `pool_types.py::PatternMatcherFn` = `Callable[[list[AlgorithmEntry], int], list[GraftProposal]]` | ✅ 正确 |
| `GraftProposal` 包含 `dependency_overrides` 字段 | `pool_types.py` | ✅ 数据模型存在 |
| `graft_general()` 调用 `apply_dependency_overrides()` | `graft_general.py` line 497 | ✅ 已实现 |
| 3 种 PatternMatcher (Random/Expert/StaticStructure) + Composite | `pattern_matchers.py` | ✅ 可用 |
| Gene bank (永久保存原始算法供 donor 查找) | `algorithm_engine.py::gene_bank` | ✅ 可用 |
| Runtime trace 收集 (interpreter + factgraph) | `algorithm_engine.py::_collect_traces()` | ✅ 可用 |

#### ❌ 未实现或严重偏离愿景的部分

**问题 1：算法池硬编码为 MIMO 检测器，不是"通用图/系统求解器"**

`build_ir_pool()` 永远返回 8 个 MIMO 检测器。模板字符串中 hardcode 了 `(H, y, sigma2, constellation)` 参数签名。所有算法模板本质上是 `f(H, y, σ², C) → x̂` 的形式。

- `ir_pool.py` 所有模板：LMMSE, ZF, OSIC, KBest, Stack, BP, EP, AMP — 全部是 MIMO 特定的
- `mimo_evaluator.py` 的 `MIMOFitnessEvaluator` 硬编码了 Rayleigh fading MIMO 信道
- `_evaluate_slot_variant()` 在 `algorithm_engine.py` 中直接 import `qam16_constellation`, `generate_mimo_sample`

**影响**：系统无法用于非 MIMO 领域（如信道编码、均衡、波束成形等）。"通用图/系统求解器"的愿景完全未实现。

**问题 2：Pattern Matcher 的 Expert Rules 硬编码为 MIMO 算法名**

`pattern_matchers.py` 中 5 条 Expert Rules：
```python
ExpertGraftRule("mmse_init_for_kbest", host_pattern="kbest", donor_pattern="lmmse", ...)
ExpertGraftRule("bp_sweep_for_ep", host_pattern="ep", donor_pattern="bp", ...)
ExpertGraftRule("amp_denoise_for_ep", host_pattern="ep", donor_pattern="amp", ...)
ExpertGraftRule("osic_ordering_for_kbest", host_pattern="kbest", donor_pattern="osic", ...)
ExpertGraftRule("lmmse_regularizer_for_zf", host_pattern="zf", donor_pattern="lmmse", ...)
```

匹配逻辑：
```python
def _matches(self, entry, pattern):
    p = pattern.lower()
    if any(p in t.lower() for t in entry.tags): return True
    if p in entry.ir.name.lower(): return True
    if p in entry.algo_id.lower(): return True
```

这完全是字符串匹配，与算法的**计算结构/数据流图**无关。

**问题 3：StaticStructurePatternMatcher 过于粗糙**

`StaticStructurePatternMatcher._fingerprint()` 仅统计 opcode 计数 + 是否有循环 + 是否有分支 + call targets。匹配逻辑：
- 如果 host 和 donor 都有循环 → 替换 host 的循环体
- 如果 host 有 call op → 替换第一个非 grafted 的 call

这是非常朴素的启发式，与"模式匹配器分析算法的计算图结构"的愿景差距很大。

**问题 4：Runtime trace 收集了但未被 PatternMatcher 消费**

`_collect_traces()` 在每 3 代收集 top-K 基因组的 trace/factgraph，存入 `genome.metadata`，并通过 `to_entry()` 暴露给 PatternMatcher。但是：

- `RandomGraftPatternMatcher` — 不使用 trace
- `ExpertPatternMatcher` — 不使用 trace（只用 tags 和 opcode 匹配）
- `StaticStructurePatternMatcher` — 不使用 trace（只用静态 IR 结构）

**结论**：trace/factgraph 收集是一个"已建好但无人使用"的基础设施。

**问题 5：`dependency_overrides` 在 PatternMatcher 中从未被填充**

所有三个 PatternMatcher 创建 GraftProposal 时：
```python
dependency_overrides=[],  # 永远为空
```

`graft_general()` 中的 `apply_dependency_overrides()` 实现是完整的（能创建新值、扩展参数列表、连接到 call op），但**从未被实际调用**（因为 overrides 列表永远为空）。

**问题 6：`port_mapping` 在 PatternMatcher 中从未被填充**

所有 GraftProposal 的 `port_mapping` 字段默认为 `{}`（空 dict）。`graft_general()` 中的映射逻辑降级到按位置映射 host arg → donor arg。

### 1.3 两套嫁接系统并行存在

| 系统 | 位置 | 用途 | 状态 |
|------|------|------|------|
| **旧嫁接** `graft_skeleton()` | `algorithm_ir/grafting/rewriter.py` | xDSL 级别的精确 op 手术，只支持 `bp_summary_update` 和 `bp_tree_runtime_update` 两个 skeleton name | ⚠️ 仅测试中使用 |
| **新嫁接** `graft_general()` | `algorithm_ir/grafting/graft_general.py` | IR 级别的通用 call-based 嫁接，不依赖 xDSL | ✅ 进化引擎使用 |

旧嫁接系统 (`rewriter.py` + `skeletons.py` + `matcher.py`) 是 algorithm_ir 包最初的核心功能，但现在**只被测试代码引用**，进化引擎完全使用新系统 `graft_general()`。

---

## 二、死代码/冗余审计

### 2.1 BUG：`AlgorithmGenome.to_entry()` 重复定义

**文件**：`pool_types.py`  
**位置**：`to_entry()` 方法定义了两次（约 line 176-230 和 line 231-285），代码完全相同。第二个定义静默覆盖第一个。这是一个明显的 copy-paste 错误。

**修复**：删除重复的 `to_entry()` 方法。

### 2.2 旧进化引擎（完全死代码）

以下文件属于**第一代进化引擎**，已被 `algorithm_engine.py` + `pool_types.py` 完全取代：

| 文件 | 行数 | 关系 |
|------|------|------|
| `evolution/engine.py` | ~200 | 旧 `EvolutionEngine`，操作 `IRGenome`（不是 `AlgorithmGenome`） |
| `evolution/genome.py` | ~186 | 旧 `IRGenome` 类（programs dict + constants），被 `AlgorithmGenome` 取代 |
| `evolution/config.py` | ~63 | 旧 `EvolutionConfig`，被 `AlgorithmEvolutionConfig`（在 pool_types.py）取代 |

**但这些文件仍被测试代码引用**：
- `test_evolution.py` 大量使用 `engine.py`, `genome.py`, `config.py`, `operators.py`
- `operators.py::mutate_genome()`, `crossover_genome()` 操作旧 `IRGenome` — 这些是旧引擎的残留

**建议**：
- `engine.py`, `genome.py`, `config.py` 可以标记为 deprecated 但暂不删除（测试仍在用）
- 将来把 `test_evolution.py` 中对旧类型的测试迁移到 `test_ir_evolution.py`

### 2.3 旧嫁接系统（部分死代码）

| 文件 | 行数 | 状态 |
|------|------|------|
| `algorithm_ir/grafting/rewriter.py` | ~400+ | 仅被测试代码使用 (`test_grafting_demo.py`, `test_grafting_generality.py`)。进化引擎不使用。 |
| `algorithm_ir/grafting/skeletons.py` | ~64 | `Skeleton`, `OverridePlan`, `make_bp_summary_skeleton`, `make_bp_tree_runtime_skeleton` — 仅被测试使用 |
| `algorithm_ir/grafting/matcher.py` | ~54 | `match_skeleton()` — 仅被测试使用 |
| `algorithm_ir/grafting/__init__.py` | ~3 | 导出旧嫁接 API — 仅被测试使用 |

旧嫁接系统在概念验证阶段有价值（它演示了 stack decoder → BP 嫁接），但它的局限性已被 `test_grafting_generality.py::test_only_two_skeletons_supported` 明确记录：只支持 2 个 hardcoded skeleton name。

### 2.4 `SlotDescriptor` 中的冗余字段

`SlotDescriptor` 有 14 个字段，但在 `to_entry()` 中构造时：
```python
SlotDescriptor(
    slot_id=pop.slot_id,
    short_name=pop.slot_id.split(".")[-1],
    level=0,       # 永远是 0
    depth=0,       # 永远是 0
    parent_slot_id=None,  # 永远是 None
    spec=pop.spec,
)
```

以下字段**从未被赋予有意义的值**：
- `level`, `depth` — 始终为 0，层级概念未实现
- `parent_slot_id`, `child_slot_ids` — 始终为空，层级树未实现
- `default_impl`, `default_impl_level` — 从未设置
- `evolution_weight`, `max_complexity` — 从未被读取
- `description`, `domain_tags` — 从未设置

### 2.5 `materialize.py` 中的重复代码

`materialize()` 和 `_materialize_source_with_override()` 有大量重复逻辑（~60% 相同代码）：
- 相同的 slot op 遍历
- 相同的 fallback pass-through 生成
- 相同的 `_find_population_for_slot` / `_extract_func_name` 调用
- 相同的 placeholder 替换

应当提取公共的 `_resolve_slot_implementations()` 辅助函数。

### 2.6 `_evaluate_slot_variant()` 中内联的 MIMO 评估逻辑

`algorithm_engine.py::_evaluate_slot_variant()` （约 50 行）几乎是 `MIMOFitnessEvaluator.evaluate()` 的简化版复制品：
- 直接 import `generate_mimo_sample`, `qam16_constellation`
- 手动创建 `ThreadPoolExecutor` + timeout 逻辑
- 手动 nearest-symbol mapping

这违反了 DRY，并且把 MIMO 特定逻辑嵌入了本应通用的引擎。应委托给 evaluator 的 `evaluate_single_result()` 或新增 `evaluate_slot_variant()` 方法。

### 2.7 `_graft_pass()` 与 PatternMatcher 嫁接的冗余

主循环中有**两套嫁接机制**：

1. **PatternMatcher 嫁接**（每代执行）：基于结构分析，使用 `graft_general()` 进行 IR 级别手术
2. **`_graft_pass()`**（每 5 代执行）：简单的 "best genome donates slot implementations to bottom 25%"

`_graft_pass()` 本质上只是 slot 级别的克隆注入，不是真正的 IR 嫁接。它与 PatternMatcher 嫁接在概念上重叠，但机制完全不同（slot copying vs IR op surgery）。

---

## 三、补丁式实现审计

### 3.1 `_template_globals()` — 大量 workaround 函数

`ir_pool.py::_template_globals()` 返回约 30 个 helper 函数（`_czeros`, `_cones`, `_qr_Q`, `_col`, `_reverse_syms`, `_row`, `_row_set`, `_argmax_row`, `_row_normalize` 等），全部是因为 IR builder 不支持 keyword arguments (`dtype=complex`, `axis=0`, `keepdims=True`) 而创建的 workaround。

这些 helper 必须同时出现在：
1. `_template_globals()` — 模板编译时
2. `materialize.py::_default_exec_namespace()` — 运行时 exec 命名空间

如果两边不同步，会导致编译时成功但运行时 NameError。

### 3.2 `_find_population_for_slot()` — 模糊匹配逻辑

```python
def _find_population_for_slot(genome, slot_id):
    for pop_key, pop in genome.slot_populations.items():
        if pop.slot_id == slot_id: return pop
        if pop_key.endswith(f".{slot_id}"): return pop
        parts = pop_key.split("."); short = parts[-1]
        if short == slot_id: return pop
        if slot_id.endswith(short) or slot_id.endswith(f"_{short}"): return pop
    return None
```

这种"尝试各种后缀匹配"的策略说明 slot_id 命名不一致，需要多重 fallback 来解决。根源是 AlgSlot op 中的 `slot_id` 与 `slot_populations` 字典的 key 之间没有统一的命名约定。

### 3.3 `SLOT_DEFAULTS` 中使用 workaround 函数的默认实现

许多 slot 默认实现（如 `bp_sweep`, `site_update`, `amp_iterate`）因为使用了 3D 数组索引 (`Beta[a,j,m]`) 或复杂操作，无法被 IR builder 编译为 FunctionIR。这些实现通过 `source_variants` 字段作为**原始 Python 源码字符串**直接传递给 materialization，绕过了 IR 系统。

**测试确认**：`test_ir_evolution.py::test_compile_bp_sweep` 断言 `ir is None`，表明这是已知的限制而非 bug。

### 3.4 多处 try/except 静默吞异常

整个代码库中大量使用：
```python
try:
    child = mutate_ir(parent, rng)
    new_variants.append(child)
except Exception:
    pass  # 静默丢弃
```

这在以下位置出现：
- `_micro_step()` — mutation/crossover 失败静默跳过
- `_mutate_random_slot()` — mutation 失败静默跳过
- `_execute_graft()` — graft 失败静默跳过
- `_collect_traces()` — trace 收集失败静默跳过
- `materialize()` — source 生成失败降级为 pass-through

虽然在进化算法中这是常见做法（让无效个体自然淘汰），但没有**任何** failure rate 统计，使得无法诊断系统性问题（如 "99% 的 mutation 都失败了"）。

---

## 四、运行时 IR 嫁接/变异审计

### 4.1 `graft_general()` — 核心嫁接引擎评估

`graft_general()` 实现了 10 步 IR 级别 op 手术：
1. Deep-copy host IR
2. 定位 region ops
3. 分析 region 边界（entry/exit values）
4. 注册 donor callable 常量
5. 创建 call op
6. Rebind exit values → call outputs
7. 移除 region ops
8. 应用 dependency overrides
9. 拓扑排序受影响 block
10. 检测 donor 引入的新 slot

**评估**：这是一个**正确设计且功能完整**的通用嫁接引擎。它不依赖 xDSL，纯粹在 FunctionIR dict 层面操作。`apply_dependency_overrides()` 的实现也是完整的。

**但在实际使用中**，嫁接结果是一个 `call donor_fn(args...)` op，而 `donor_fn` 的实际代码通过 `_donor_sources` dict 在 `materialize()` 时注入 exec 命名空间。这意味着：
- 嫁接后的 IR 中没有 donor 的具体计算，只有一个 call op
- donor 的实际执行取决于 materialization 时 donor source 是否可用
- 如果 donor genome 被垃圾回收或 donor source materialize 失败，call op 会在运行时报 NameError

### 4.2 运行时 trace 嫁接 — 未实现

愿景中描述的 "stack decoder 搜索树 → BP 消息传递树" 式运行时 trace 嫁接**完全未实现**。

旧嫁接系统（`rewriter.py`）中有 `bp_tree_runtime_update` 的 demo，它在 stack decoder 的搜索树扩展节点处注入 BP 消息传递。但这是一个**硬编码的 demo**（只支持那一个 donor），且进化引擎从未使用它。

进化引擎中：
- `_collect_traces()` 收集 interpreter trace + factgraph
- 但没有任何代码**分析** trace 来发现嫁接机会
- 没有任何 PatternMatcher 使用 trace 数据

### 4.3 IR Mutation 操作

`operators.py` 提供 4 种 mutation：
- `point` — 交换 binary/compare opcode
- `constant_perturb` — 高斯扰动常量
- `insert` — 通过 source recompile 插入随机赋值语句
- `delete` — 通过 source recompile 删除随机行

`insert` 和 `delete` 操作采用**退化策略**：`emit_python_source() → 文本编辑 → recompile`。这绕过了 IR 的结构化表示，本质上是字符串操作而非图操作。

### 4.4 Slot 级别变异 vs 结构级别变异

当前系统中两个维度的变异：
| 维度 | 机制 | 颗粒度 |
|------|------|--------|
| Micro (slot 内) | `_micro_step()` → `mutate_ir()` + `crossover_ir()` | opcode 交换 / 常量扰动 |
| Macro (跨算法) | `_breed_macro()` → slot population 交换 | slot 级别移植 |
| 结构嫁接 | PatternMatcher → `_execute_graft()` → `graft_general()` | IR 区域替换为 call op |
| Slot 捐赠 | `_graft_pass()` | best genome slot → worst genome slot |

**缺失的维度**：
- 没有"添加新 slot"的 mutation（只有 graft 时可能发现新 slot）
- 没有"删除 slot"或"合并 slot"的操作
- 没有骨架结构本身的 mutation（如添加循环、删除分支）

---

## 五、重构优先级

### P0 — 必须修复的 Bug

1. **删除 `pool_types.py` 中重复的 `to_entry()` 方法**
   - 复杂度：< 5 min
   - 影响：消除 Python 静默覆盖导致的混乱

### P1 — 高优先级重构

2. **解耦 MIMO 特定逻辑**
   - 将 `_evaluate_slot_variant()` 中的 MIMO 逻辑移到 evaluator 接口
   - 在 `AlgorithmFitnessEvaluator` 中添加 `evaluate_slot_variant(genome, slot_id, variant_ir) → float`
   - `algorithm_engine.py` 不应直接 import `generate_mimo_sample` / `qam16_constellation`

3. **消除 `materialize()` 与 `_materialize_source_with_override()` 的重复**
   - 提取公共函数 `_resolve_all_slots(genome, override_map=None) → dict[op_id, (name, source)]`
   - 两个函数都委托给它

4. **让 PatternMatcher 实际消费 trace/factgraph 数据**
   - 至少让 `StaticStructurePatternMatcher` 使用 `AlgorithmEntry.trace` 来做更智能的匹配
   - 或者如果 trace 系统暂不可用，**移除 `_collect_traces()` 调用**以减少无用计算

### P2 — 中优先级

5. **统一嫁接系统**
   - `_graft_pass()` 应被重新定义为一种特殊的 PatternMatcher（"slot donation matcher"），统一到 CompositePatternMatcher 中
   - 或者至少明确文档说明两者的不同角色

6. **清理 SlotDescriptor**
   - 移除从未使用的字段，或者**实际实现**层级化 slot 树
   - 当前 `level`, `depth`, `parent_slot_id` 永远为默认值，给人以层级已实现的错觉

7. **添加变异失败率统计**
   - 在 `_micro_step()` 和 `_mutate_random_slot()` 中跟踪 success/failure count
   - 写入 `_history` 记录

8. **统一 slot_id 命名约定**
   - 消除 `_find_population_for_slot()` 中的多重 fallback 匹配
   - 在 `build_ir_pool()` 中确保 `slot_populations` key 与 AlgSlot op 的 `slot_id` 完全一致

### P3 — 低优先级 / 长期

9. **标记旧进化引擎为 deprecated**
   - 在 `engine.py`, `genome.py`, `config.py` 顶部添加 deprecation 注释
   - 将 `test_evolution.py` 中旧类型测试逐步迁移

10. **评估旧嫁接系统的去留**
    - `rewriter.py`, `skeletons.py`, `matcher.py` 仅被测试使用
    - 保留作为参考实现（demo 了 xDSL 级别的精确嫁接），但不应投入更多开发

11. **解决 IR builder 不支持 keyword arguments 的根本问题**
    - 当前 `_template_globals()` 中 30+ workaround 函数是不可持续的
    - 考虑在 `ast_parser.py` 中添加 `ast.keyword` 支持
    - 或在 IR model 中直接支持 keyword args

---

## 六、架构建议

### 6.1 通向"通用图求解器"的路径

当前系统的核心瓶颈是**算法模板与评估器都是 MIMO 特定的**。要实现通用化：

```
AlgorithmEvolutionEngine (通用)
    ├─ AlgorithmPool      → 由外部注入，不硬编码 MIMO
    ├─ FitnessEvaluator   → 抽象接口，MIMO 只是一个实例
    ├─ PatternMatcher      → 基于计算图结构匹配，不基于算法名字符串
    └─ GraftEngine         → graft_general() 已经是通用的 ✅
```

关键改动：
1. `build_ir_pool()` 接受 `list[DetectorSpec]` 参数而非 hardcode
2. `_evaluate_slot_variant()` 委托给 evaluator
3. Expert rules 不应匹配 algo_id 字符串，而应匹配 IR 结构模式

### 6.2 从"call-based graft"到"inline graft"

当前 `graft_general()` 将 donor 替换为一个 `call donor_fn(args)` op。这意味着：
- donor 的计算在 IR 中是不透明的
- 后续 mutation 无法修改 donor 内部
- 嫁接后的算法不是一个完整的 IR 图

未来可以考虑**inline graft**：将 donor 的 ops 直接嵌入 host IR，使得：
- 整个算法是一个单一的 IR 图
- 后续 mutation 可以跨越原始 host/donor 边界
- 不需要 `_donor_sources` 机制

### 6.3 Runtime Trace 利用路径

已有的 trace/factgraph 基础设施可以用于：
1. **Trace-guided pattern matching**: 分析运行时数据流，发现语义等价但结构不同的区域
2. **Fitness-guided slot discovery**: 在 concrete skeleton 上运行 trace，发现值波动最大的区域作为 slot 候选
3. **Cross-algorithm trace alignment**: 比较两个算法在相同输入上的 trace，发现可互换的子计算

这些都需要在 PatternMatcher 中实际读取 `entry.trace` 和 `entry.factgraph`。

---

## 七、文件清单与状态总结

### evolution/ 目录

| 文件 | 行数 | 角色 | 状态 |
|------|------|------|------|
| `algorithm_engine.py` | 734 | 两层进化引擎主体 | ✅ 活跃，有 MIMO 耦合问题 |
| `pool_types.py` | 362 | 核心数据类型 | ✅ 活跃，有重复 `to_entry()` bug |
| `pattern_matchers.py` | 495 | 3 种 PatternMatcher + Composite | ✅ 活跃，Expert 规则过于朴素 |
| `ir_pool.py` | 1063 | 8 种检测器模板 + slot 转换 | ✅ 活跃，MIMO 硬编码 |
| `materialize.py` | 387 | 物化管线 | ✅ 活跃，有代码重复 |
| `mimo_evaluator.py` | 242 | MIMO 适应度评估 | ✅ 活跃（应为可替换模块） |
| `algorithm_pool.py` | ~300 | L0-L3 pool builder + slot 定义 | ✅ 活跃（被 ir_pool.py 引用） |
| `operators.py` | ~200 | IR 变异/交叉 | ✅ 活跃 |
| `random_program.py` | ~189 | 随机程序生成 | ✅ 活跃 |
| `skeleton_registry.py` | ~190 | ProgramSpec + 验证 | ✅ 活跃 |
| `slot_discovery.py` | ~200 | 自动 slot 发现 | ⚠️ 已实现但未被引擎使用 |
| `run_evolution.py` | ~203 | CLI runner | ✅ 活跃 |
| `fitness.py` | ~59 | FitnessResult + 旧 FitnessEvaluator | ✅ 活跃 |
| `engine.py` | ~200 | 旧 EvolutionEngine | ⚠️ 仅被旧测试使用 |
| `genome.py` | ~186 | 旧 IRGenome | ⚠️ 仅被旧测试使用 |
| `config.py` | ~63 | 旧 EvolutionConfig | ⚠️ 仅被旧测试使用 |
| `pool_ops_l0.py` | ? | L0 原语 | ✅ 活跃 |
| `pool_ops_l1.py` | ? | L1 组合 | ✅ 活跃 |
| `pool_ops_l2.py` | ? | L2 模块 | ✅ 活跃 |
| `pool_ops_l3.py` | ? | L3 检测器 | ✅ 活跃 |

### algorithm_ir/ 目录

| 文件/目录 | 角色 | 状态 |
|-----------|------|------|
| `frontend/ir_builder.py` | Python → IR 编译器 | ✅ 核心，缺少 keyword arg 支持 |
| `ir/model.py` | FunctionIR 数据模型 | ✅ 核心 |
| `ir/dialect.py` | xDSL AlgDialect | ✅ 核心 |
| `ir/xdsl_bridge.py` | xDSL ↔ dict 桥接 | ✅ 核心 |
| `ir/validator.py` | IR 验证 | ✅ 核心 |
| `grafting/graft_general.py` | 通用 IR 嫁接 | ✅ 核心 |
| `grafting/rewriter.py` | 旧嫁接（xDSL 级别） | ⚠️ 仅被测试使用 |
| `grafting/skeletons.py` | 旧 skeleton 定义 | ⚠️ 仅被测试使用 |
| `grafting/matcher.py` | 旧 skeleton 匹配 | ⚠️ 仅被测试使用 |
| `runtime/interpreter.py` | IR 解释器 | ✅ 活跃 |
| `runtime/tracer.py` | 运行时 trace | ✅ 活跃（但 trace 未被消费） |
| `factgraph/builder.py` | FactGraph 构建 | ✅ 活跃（但 graph 未被消费） |
| `region/selector.py` | 区域选择 | ✅ 核心 |
| `region/contract.py` | 边界契约推断 | ✅ 核心 |
| `region/slicer.py` | 前向/后向切片 | ✅ 核心 |
| `regeneration/codegen.py` | IR → Python/C++ 代码生成 | ✅ 核心 |
| `projection/` | 投影注释 | ✅ 活跃（但投影分数是硬编码的） |
| `analysis/` | 静态/动态分析 | ✅ 活跃 |
