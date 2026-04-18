

## 修改方案：从固定程序进化到算法级嫁接进化

我将当前架构和你的目标做一个对比，然后给出分层的修改方案。

---

### 问题诊断

**当前架构**（我写的）做了一个关键简化：

```
当前: EvolutionEngine
  └─ 每个个体 = IRGenome {f_down: FunctionIR, f_up: FunctionIR, ...}
  └─ 变异/交叉在这些小函数内部做（换运算符、换常量）
  └─ 骨架(skeleton)是硬编码的参数签名——不是一个真正的host program
```

**你要的架构**：

```
目标: EvolutionEngine
  └─ 算法池 = [EP_ir, BP_ir, KBEST_ir, StackDecoder_ir, AMP_ir, ...]
  └─ 每代有一个 PatternMatcher(algo_pool) → [GraftProposal, ...]
  └─ 每个 GraftProposal = "把 A_i 的区域 R 用 A_j 的结构替换，可能修改依赖"
  └─ 执行 graft → 产生新算法 → 放入池中
  └─ 适应度驱动选择哪些新算法存活
```

本质区别：**当前框架进化的是「填空的内容」，目标框架进化的是「整个算法的结构」**。

---

### 分层修改方案

#### 层 1：PatternMatcher 的输入/输出 Spec

这是最核心的接口。PatternMatcher 是一个函数，签名为：

```python
@dataclass
class AlgorithmEntry:
    """算法池中的一个条目。"""
    algo_id: str                              # 唯一标识
    ir: FunctionIR                            # 编译后的 IR
    source: str | None                        # 原始源码（如有）
    trace: list[RuntimeEvent] | None          # 最近一次执行的运行轨迹
    runtime_values: dict[str, RuntimeValue] | None  # 运行时值
    factgraph: FactGraph | None               # 静态+动态统一图
    fitness: FitnessResult | None             # 当前适应度（如已评估）
    generation: int                           # 加入池的代数
    provenance: dict[str, Any]                # 谱系：来自哪些算法的哪些操作
    tags: set[str]                            # 标签：{"original", "evolved", "grafted"}


@dataclass
class DependencyOverride:
    """模式匹配器提出的依赖修改。
    
    示例：原本 v_6 只依赖 v_5，修改为 v_6 依赖 [v_0, v_1, v_5]。
    这会改变 region 的 entry_values / BoundaryContract 的 input_ports。
    """
    target_value: str              # 要修改其输入依赖的值 ID
    new_dependencies: list[str]    # 新的依赖值 ID 列表
    reason: str                    # 为什么做这个修改


@dataclass
class GraftProposal:
    """模式匹配器的一个嫁接建议。"""
    proposal_id: str
    
    # 宿主侧 (Host side)
    host_algo_id: str                           # 哪个算法被改写
    region: RewriteRegion                       # 要替换的区域（已定义好）
    contract: BoundaryContract                  # 该区域的边界契约
    
    # 供体侧 (Donor side)
    donor_algo_id: str | None                   # 用哪个算法的结构替换（可以是None=新生成的）
    donor_ir: FunctionIR | None                 # 供体的完整函数 IR（可能含 AlgSlot）
    donor_region: RewriteRegion | None          # 元信息：donor 中"有效区域"的标记（引擎不做子图提取）
    
    # 依赖覆盖 (Dependency overrides)
    dependency_overrides: list[DependencyOverride]  # 新的依赖关系
    
    # 参数绑定 (Port binding)
    port_mapping: dict[str, str]                # host_value_id → donor_value_id 的映射
    
    # 元信息
    confidence: float                           # 0~1，匹配器对这个建议的信心
    rationale: str                              # 人可读的理由
    

# PatternMatcher 的类型签名
PatternMatcherFn = Callable[
    [list[AlgorithmEntry], int],    # (当前算法池, 当前代数)
    list[GraftProposal]             # 返回嫁接建议列表
]
```

**关键设计决策**：

1. **PatternMatcher 输入的是完整的 `AlgorithmEntry`**，包含 IR（静态结构）、执行轨迹（动态结构）、FactGraph（统一视图）。这让匹配器既可以做纯静态分析（"发现 for 循环中有 metric 计算"），也可以做动态分析（"运行时展开的搜索树有 N 个节点"）。

2. **`GraftProposal` 包含已经构造好的 `RewriteRegion` 和 `BoundaryContract`**。这意味着 PatternMatcher 负责调用现有的 `define_rewrite_region()` 和 `infer_boundary_contract()`。这是合理的——匹配器需要理解区域才能提出建议。

3. **`DependencyOverride` 是关键创新点**。它让匹配器可以说"v_6 现在要依赖 v_0~v_5"，这在当前框架中没有对应物。实现上需要扩展 `BoundaryContract` 和 `graft_skeleton`/rewriter。

4. **`donor_ir` 可以带 slot**。供体结构本身可以包含 `AlgSlot` 操作，这些 slot 在后续进化中继续被填充。

---

#### 层 2：对 algorithm_ir 基础设施的修改

##### 2.1 扩展 BoundaryContract：支持 dependency override

```python
# contract.py 新增字段
@dataclass
class BoundaryContract:
    # ... 现有字段 ...
    
    # 新增：依赖覆盖
    dependency_overrides: list[DependencyOverride] = field(default_factory=list)
    # 某些 output port 的依赖被扩展了——rewriter 需要在嫁接时
    # 为供体注入额外的输入端口
```

##### 2.2 通用 rewriter：替代硬编码的 graft 分发

当前 `graft_skeleton()` 通过 `transform_rules[0]["kind"]` 硬分发到 `_graft_bp_summary()` 或 `_graft_bp_tree_runtime_update()`。这是最大的硬编码点。

需要一个通用的 rewriter：

```python
def graft_general(
    host_ir: FunctionIR,
    proposal: GraftProposal,
) -> AlgorithmArtifact:
    """通用嫁接：将 proposal.donor_ir 的结构注入 host_ir 的 proposal.region。
    
    步骤：
    1. clone host_ir
    2. 如有dependency_overrides：修改 region 的 entry_values
       - 对于每个 override：找到 target_value 的 def_op，
         添加新的 input values 到 region 的 entry_values
         在 xDSL 层面创建新的 block arguments 或 phi 节点
    3. 根据 port_mapping 将 donor_ir 的参数绑定到 host 的值
    4. 将 donor_ir 的所有 ops（重命名避免冲突）插入到 host 的 region 位置
    5. 用 donor 的 output 替换 region 原来的 output（通过 reconnect_map）
    6. 删除 region 中被替换的 ops
    7. rebuild from xDSL
    """
```

这是对现有 `_graft_bp_summary` 和 `_graft_bp_tree_runtime_update` 的泛化。核心挑战是 **op 重命名**（donor 的 op/value ID 可能和 host 冲突）和 **类型兼容性**。

##### 2.3 新增：DependencyInjector

处理 `DependencyOverride`——改变值的数据流依赖：

```python
def apply_dependency_overrides(
    func_ir: FunctionIR,
    region: RewriteRegion,
    overrides: list[DependencyOverride],
) -> tuple[FunctionIR, RewriteRegion]:
    """在嫁接前修改 host IR 的依赖关系。
    
    对每个 override：
    1. 找到 target_value
    2. 将 new_dependencies 中的值加入 region.entry_values
    3. 更新 BoundaryContract.input_ports
    4. 在 IR 层面：如果 target_value 的 def_op 需要更多输入，
       要么创建新的 binary/call op 来聚合新依赖，
       要么修改现有 op 的 inputs（危险）
    
    更安全的策略：不直接修改现有 op，而是在 region 入口处
    创建一个"聚合 op"（如 build_tuple），将所有新依赖打包，
    然后供体的代码可以通过 get_item 访问它们。
    """
```

---

#### 层 3：对 evolution 框架的修改

##### 3.1 新的 Genome 结构——二层进化架构

当前 `IRGenome` 是 `{role_name: FunctionIR}` 的字典。新架构中需要**二层结构**：

- **宏层（Macro）**：算法的整体结构（控制流骨架、嫁接拓扑）
- **微层（Micro）**：每个 slot/hole 的具体实现（小函数的子种群）

这两层**同步进化**：宏层通过 PatternMatcher 的嫁接建议改变结构，微层通过传统 GP 操作进化 slot 内容。

```
AlgorithmPool:
  Algorithm_A (stack_decoder 结构):
    ├── structural_ir: FunctionIR  (含 AlgSlot 占位符)
    ├── Slot "score_compute":
    │   ├── variant_0: basic metric       fitness=0.7
    │   ├── variant_1: bp_enhanced        fitness=0.3
    │   └── variant_2: mutated_bp         fitness=0.4
    ├── Slot "expansion_policy":
    │   ├── variant_0: greedy             fitness=0.7
    │   └── variant_1: beam_search        fitness=0.5
    ├── best_combination: (variant_1, variant_0)  → composite_fitness=0.25
    └── algo_fitness: 0.25  (最佳组合的适应度)

  Algorithm_B (bp_detector 结构):
    ├── structural_ir: FunctionIR
    ├── Slot "message_rule":
    │   ├── variant_0: sum_product        fitness=0.1
    │   └── variant_1: min_sum            fitness=0.15
    ├── best_combination: (variant_0,)
    └── algo_fitness: 0.1
```

```python
@dataclass
class SlotPopulation:
    """一个 slot 的变体子种群。"""
    slot_id: str                                 # 对应 ProgramSpec.role
    spec: ProgramSpec                            # 签名规范（输入/输出类型）
    variants: list[FunctionIR]                   # 变体列表
    fitness: list[float]                         # 每个变体的适应度
    best_idx: int = 0                            # 当前最佳变体索引

    @property
    def best_variant(self) -> FunctionIR:
        return self.variants[self.best_idx]

    @property
    def best_fitness(self) -> float:
        return self.fitness[self.best_idx]


@dataclass
class AlgorithmGenome:
    """进化个体 = 一个结构模板 + 每个 slot 的变体子种群。
    
    二层结构：
      - structural_ir: 包含 AlgSlot 占位符的完整算法 IR
      - slot_populations: 每个 slot 的多个变体实现
    
    执行时，用 best_combination 中的变体填充所有 slot，
    得到一个完整的可执行 FunctionIR。
    """
    algo_id: str
    structural_ir: FunctionIR                    # 含 AlgSlot 的结构模板
    slot_populations: dict[str, SlotPopulation]  # slot_id → 子种群
    constants: np.ndarray                        # 结构级可进化常量
    generation: int
    parent_ids: list[str]                        # 谱系
    graft_history: list[GraftRecord]             # 嫁接历史
    tags: set[str]                               # {"original", "evolved", "grafted"}
    
    # 缓存
    _materialized_ir: FunctionIR | None = None   # slot 填充后的完整 IR
    _trace: list[RuntimeEvent] | None = None
    _runtime_values: dict[str, RuntimeValue] | None = None
    _factgraph: FactGraph | None = None
    
    def materialize(self) -> FunctionIR:
        """用每个 slot 的最佳变体填充 structural_ir，得到可执行 IR。
        
        遍历 structural_ir 中的 AlgSlot ops：
          1. 找到对应的 SlotPopulation
          2. 取 best_variant
          3. inline 到 slot 位置（或替换为 call）
        """
        ...
    
    def execute(self, args: list) -> tuple[Any, list[RuntimeEvent], dict]:
        """执行算法（自动 materialize），缓存轨迹。"""
        if self._materialized_ir is None:
            self._materialized_ir = self.materialize()
        result, trace, rv = execute_ir(self._materialized_ir, args)
        self._trace = trace
        self._runtime_values = rv
        return result, trace, rv
    
    def build_factgraph(self) -> FactGraph:
        if self._trace is None:
            raise RuntimeError("Must execute() first")
        self._factgraph = build_factgraph(
            self._materialized_ir, self._trace, self._runtime_values
        )
        return self._factgraph
    
    def invalidate_cache(self) -> None:
        """当 slot variants 或 structural_ir 改变时，清除缓存。"""
        self._materialized_ir = None
        self._trace = None
        self._runtime_values = None
        self._factgraph = None
    
    def to_entry(self, fitness: FitnessResult | None = None) -> AlgorithmEntry:
        """转换为 PatternMatcher 需要的 AlgorithmEntry。
        
        注意：PatternMatcher 看到的是 structural_ir（含 AlgSlot），
        这样它可以识别哪些区域已经是 slot、哪些是固定结构。
        trace/factgraph 则来自 materialized 后的执行。
        """
        return AlgorithmEntry(
            algo_id=self.algo_id,
            ir=self.structural_ir,              # 结构模板
            source=None,
            trace=self._trace,                  # 来自 materialized 执行
            runtime_values=self._runtime_values,
            factgraph=self._factgraph,
            fitness=fitness,
            generation=self.generation,
            provenance={"graft_history": self.graft_history},
            tags=self.tags,
        )
    
    def clone(self) -> AlgorithmGenome:
        """深拷贝（不拷贝缓存）。"""
        return AlgorithmGenome(
            algo_id=self._make_id(),
            structural_ir=deepcopy(self.structural_ir),
            slot_populations={
                k: SlotPopulation(
                    slot_id=v.slot_id,
                    spec=v.spec,
                    variants=[deepcopy(ir) for ir in v.variants],
                    fitness=list(v.fitness),
                    best_idx=v.best_idx,
                )
                for k, v in self.slot_populations.items()
            },
            constants=self.constants.copy(),
            generation=self.generation,
            parent_ids=list(self.parent_ids),
            graft_history=list(self.graft_history),
            tags=set(self.tags),
        )


@dataclass
class GraftRecord:
    """记录一次嫁接操作。"""
    generation: int
    host_algo_id: str
    donor_algo_id: str | None
    proposal_id: str
    region_summary: str                 # 人可读摘要
    new_slots_created: list[str]        # 嫁接引入的新 slot ID 列表
```

##### 3.2 新的 Engine 循环——二层同步进化

进化引擎同时运行两个层级的搜索：

```
每一代（Generation）:
  ┌─────────────────────────────────────────────────────┐
  │ 宏层（Macro）: 结构进化                               │
  │   PatternMatcher → GraftProposal → graft_general()  │
  │   改变算法的控制流骨架，可能引入新 slot               │
  └─────────────────────────────────────────────────────┘
            ↓ 产生新的 AlgorithmGenome（含新 slot）
  ┌─────────────────────────────────────────────────────┐
  │ 微层（Micro）: Slot 子种群进化                        │
  │   对每个 AlgorithmGenome 的每个 SlotPopulation:      │
  │     - 变异/交叉现有 variants                         │
  │     - 评估新 variants 的适应度                       │
  │     - 选择存活 variants                              │
  └─────────────────────────────────────────────────────┘
            ↓ 每个 algo 的 slot 子种群更新
  ┌─────────────────────────────────────────────────────┐
  │ 组合评估                                             │
  │   用每个 algo 的 best_combination 执行完整算法        │
  │   → 计算 algo-level fitness                         │
  │   → 选择存活的 AlgorithmGenome                      │
  └─────────────────────────────────────────────────────┘
```

```python
class AlgorithmEvolutionEngine:
    def __init__(
        self,
        config: AlgorithmEvolutionConfig,
        evaluator: AlgorithmFitnessEvaluator,       # 新的评估器基类
        pattern_matcher: PatternMatcherFn,            # 核心新参数
        initial_algorithms: list[AlgorithmGenome],    # 算法池
        test_inputs_fn: Callable[[], list],           # 提供测试输入的工厂函数
        rng: np.random.Generator | None = None,
    ):
        self.pool: list[AlgorithmGenome] = initial_algorithms
        self.fitness: list[FitnessResult | None] = [None] * len(initial_algorithms)
        self.pattern_matcher = pattern_matcher
        self.evaluator = evaluator
        self.test_inputs_fn = test_inputs_fn
        self.config = config
        self.rng = rng or np.random.default_rng()
        ...
    
    def run(self, n_generations: int) -> AlgorithmGenome:
        # ── 初始化：执行所有初始算法，收集轨迹 ──
        for algo in self.pool:
            algo.execute(self.test_inputs_fn())
            algo.build_factgraph()
        self._evaluate_pool()
        
        for gen in range(n_generations):
            # ══════════════════════════════════════
            # 阶段 1: 宏层——结构进化（嫁接）
            # ══════════════════════════════════════
            entries = [a.to_entry(f) for a, f in zip(self.pool, self.fitness)]
            proposals = self.pattern_matcher(entries, gen)
            
            new_algos: list[AlgorithmGenome] = []
            for proposal in proposals:
                try:
                    child = self._execute_graft(proposal)
                    new_algos.append(child)
                except GraftError as e:
                    logger.warning("Graft failed: %s", e)
                    continue
            
            # ══════════════════════════════════════
            # 阶段 2: 微层——Slot 子种群进化
            # ══════════════════════════════════════
            # 对池中所有算法（包括新嫁接产生的）做 slot 进化
            all_algos = self.pool + new_algos
            for algo in all_algos:
                self._evolve_slots(algo, n_micro_gens=self.config.micro_generations)
            
            # ══════════════════════════════════════
            # 阶段 3: 组合评估 + 选择
            # ══════════════════════════════════════
            # 每个算法用其最佳 slot 组合 materialize 并评估
            for algo in all_algos:
                algo.invalidate_cache()
                algo.execute(self.test_inputs_fn())
                algo.build_factgraph()
            
            all_fitness = self.evaluator.evaluate_batch(all_algos)
            
            # 合并排序，保留最佳
            combined = list(zip(all_algos, all_fitness))
            combined.sort(key=lambda x: x[1].composite_score())
            combined = combined[:self.config.pool_size]
            self.pool = [a for a, f in combined]
            self.fitness = [f for a, f in combined]
            
            self._log_generation(gen)
    
    def _execute_graft(self, proposal: GraftProposal) -> AlgorithmGenome:
        """执行一个嫁接建议，返回新的 AlgorithmGenome。"""
        host = self._find_algo(proposal.host_algo_id)
        
        # 1. 应用依赖覆盖
        modified_ir, modified_region = apply_dependency_overrides(
            host.structural_ir, proposal.region, proposal.dependency_overrides
        )
        
        # 2. 执行嫁接（donor 是完整函数 IR，可能含 slot）
        artifact = graft_general(modified_ir, proposal)
        
        # 3. 发现嫁接后新增的 slot
        new_slot_ids = discover_new_slots(artifact.ir, host.structural_ir)
        
        # 4. 为新 slot 初始化子种群
        new_populations = dict(host.slot_populations)  # 继承已有 slot 的子种群
        for slot_id in new_slot_ids:
            spec = infer_slot_spec(artifact.ir, slot_id)
            initial_variants = [random_program(spec, self.rng) for _ in range(self.config.micro_pop_size)]
            new_populations[slot_id] = SlotPopulation(
                slot_id=slot_id,
                spec=spec,
                variants=initial_variants,
                fitness=[float("inf")] * len(initial_variants),
            )
        
        return AlgorithmGenome(
            algo_id=AlgorithmGenome._make_id(),
            structural_ir=artifact.ir,
            slot_populations=new_populations,
            constants=host.constants.copy(),
            generation=host.generation + 1,
            parent_ids=[proposal.host_algo_id, proposal.donor_algo_id],
            graft_history=host.graft_history + [GraftRecord(
                generation=host.generation + 1,
                host_algo_id=proposal.host_algo_id,
                donor_algo_id=proposal.donor_algo_id,
                proposal_id=proposal.proposal_id,
                region_summary=proposal.rationale,
                new_slots_created=new_slot_ids,
            )],
            tags={"grafted"},
        )
    
    def _evolve_slots(self, algo: AlgorithmGenome, n_micro_gens: int) -> None:
        """对一个算法的所有 slot 子种群做微层进化。
        
        对每个 slot：
        1. 变异/交叉现有 variants
        2. 将每个 variant 填入 structural_ir 的对应 slot
        3. 执行完整算法，评估 variant 的适应度
        4. 选择最佳 variants 保留
        
        注意：不同 slot 的评估不独立——variant_A 在 slot_1 的好坏
        可能取决于 slot_2 当前用的是哪个 variant。
        因此评估时固定其他 slot 用当前 best，只变化目标 slot。
        """
        for slot_id, pop in algo.slot_populations.items():
            for micro_gen in range(n_micro_gens):
                # 生成新 variant
                new_variants = []
                for v in pop.variants:
                    if self.rng.random() < self.config.micro_mutation_rate:
                        mutated = mutate_ir_program(deepcopy(v), pop.spec, self.rng)
                        new_variants.append(mutated)
                
                # 评估新 variant（固定其他 slot 用 best）
                for variant in new_variants:
                    test_ir = algo.structural_ir  # 含 AlgSlot
                    # 临时替换当前 slot 的 best 为此 variant
                    materialized = materialize_with_override(
                        test_ir, algo.slot_populations,
                        override={slot_id: variant}
                    )
                    try:
                        result, _, _ = execute_ir(materialized, self.test_inputs_fn())
                        fitness = self.evaluator.evaluate_single_result(result)
                    except Exception:
                        fitness = float("inf")
                    
                    pop.variants.append(variant)
                    pop.fitness.append(fitness)
                
                # 选择保留最佳 micro_pop_size 个
                ranked = sorted(range(len(pop.variants)), key=lambda i: pop.fitness[i])
                ranked = ranked[:self.config.micro_pop_size]
                pop.variants = [pop.variants[i] for i in ranked]
                pop.fitness = [pop.fitness[i] for i in ranked]
                pop.best_idx = 0  # ranked[0] 是最佳
    
    def _evaluate_pool(self) -> None:
        """评估当前池中所有算法。"""
        self.fitness = self.evaluator.evaluate_batch(self.pool)
```

##### 3.3 二层进化的配置

```python
@dataclass
class AlgorithmEvolutionConfig:
    """二层进化引擎配置。"""
    # 宏层参数
    pool_size: int = 20                  # 算法池最大容量
    n_generations: int = 100             # 宏层代数
    
    # 微层参数
    micro_pop_size: int = 8              # 每个 slot 的变体子种群大小
    micro_generations: int = 5           # 每宏代做多少微代 slot 进化
    micro_mutation_rate: float = 0.3     # slot variant 变异率
    
    # 传统微变异（常量扰动等）
    constant_perturb_rate: float = 0.1
    constant_perturb_scale: float = 0.05
```

---

#### 层 4：与现有代码的兼容性

##### 保留的部分

| 现有模块 | 状态 | 原因 |
|---|---|---|
| `algorithm_ir/ir/model.py` | **不变** | Value/Op/Block/FunctionIR 是基础 |
| `algorithm_ir/frontend/` | **不变** | compile_source_to_ir 仍然需要 |
| `algorithm_ir/regeneration/codegen.py` | **不变** | emit_python_source / emit_cpp_ops 仍然需要 |
| `algorithm_ir/runtime/` | **不变** | execute_ir + tracing 是匹配器的核心输入 |
| `algorithm_ir/factgraph/` | **不变** | FactGraph 是匹配器的核心输入 |
| `algorithm_ir/region/selector.py` | **不变** | define_rewrite_region 由匹配器调用 |
| `algorithm_ir/region/contract.py` | **扩展** | 新增 dependency_overrides 字段 |
| `algorithm_ir/region/slicer.py` | **不变** | 前/后向切片是匹配器的工具 |
| `algorithm_ir/projection/` | **不变** | annotate_region 是匹配器可选工具 |
| `algorithm_ir/analysis/` | **不变** | 静态/动态分析是匹配器的工具 |

##### 需要修改的部分

| 模块 | 修改 | 说明 |
|---|---|---|
| `algorithm_ir/grafting/rewriter.py` | **新增 `graft_general()`** | call-based 通用嫁接，不做 inline |
| `algorithm_ir/grafting/skeletons.py` | **保留但不再是唯一路径** | Skeleton 仍可用于手工定义的供体模板 |
| `algorithm_ir/grafting/matcher.py` | **扩展** | match_skeleton 可检查 GraftProposal 的兼容性 |
| `evolution/genome.py` | **新增 `AlgorithmGenome` + `SlotPopulation`** | 二层结构：结构模板 + slot 子种群 |
| `evolution/engine.py` | **新增 `AlgorithmEvolutionEngine`** | 二层同步进化引擎 |
| `evolution/operators.py` | **扩展** | 新增 micro-level slot 变异操作 |
| `evolution/fitness.py` | **新增 `AlgorithmFitnessEvaluator`** | 覆写基类，接收 AlgorithmGenome |

##### 不动的部分（向后兼容）

**现有的 `IRGenome` + `EvolutionEngine`** 保持不变——它们仍然适用于「固定骨架+进化填空」的场景（如 MIMO BP 检测器的4个小程序进化）。新的 `AlgorithmGenome` + `AlgorithmEvolutionEngine` 是平行新增的，不是替换。

---

#### 层 5：`graft_general()` 的详细设计

这是实现难度最大的部分。当前的 `_graft_bp_summary()` 本质上做了：

```
1. 找到 host 中名为 "score" 的变量
2. 把计算 score 的 ops 删掉
3. 插入 call donor_fn(frontier, costs, damping) 的 ops
4. 把 score 重绑定到新的计算结果
```

**关键简化**：donor 始终是**完整的函数 IR**（可能带 AlgSlot），不需要从 donor 中"切出子图"。
嫁接操作 = 在 host 的 region 位置插入一个 `call donor_fn(...)` 调用。

这大大简化了实现：
- 不需要 op 重命名（donor 的 ops 留在 donor 内部，不 inline 到 host）
- 不需要多 block 合并（donor 的控制流在 call 内部）
- 类型兼容只需检查 call 的参数/返回值

通用版本：

```python
def graft_general(host_ir: FunctionIR, proposal: GraftProposal) -> AlgorithmArtifact:
    """
    通用嫁接引擎。
    
    核心策略：donor 作为完整函数被 call，不做 inline。
    
    输入：
      host_ir:  宿主算法的 IR（会被 clone）
      proposal: 包含 region + donor_ir + port_mapping + dependency_overrides
    
    步骤：
    
    Step 1: Clone host
    ──────────────────
    new_ir = deepcopy(host_ir)
    
    Step 2: Apply dependency overrides
    ──────────────────────────────────
    对每个 DependencyOverride：
      - 找到 target_value 在 host IR 中的定义
      - 把 new_dependencies 加入 region.entry_values
      - 扩展 contract.input_ports
    
    Step 3: Build call op
    ────────────────────
    根据 port_mapping，构造一个 call op：
      result = donor_fn(host_v_3, host_v_7, ...)
    
    其中：
      - donor_fn 通过 const op 引入（指向 donor_ir 对应的 Python callable，
        或者如果 donor 本身含 AlgSlot，则指向一个 partial-materialized 版本）
      - 参数顺序由 port_mapping 决定
      - 返回值类型从 contract.output_ports 推断
    
    Step 4: Insert call + reconnect
    ──────────────────────────────
    将 call op 插入到 region 的合适位置。
    将 call 的返回值绑定到 region 原来的 exit_values。
    通过 contract.reconnect_points 找到下游消费者，重绑定。
    
    Step 5: Remove replaced ops
    ──────────────────────────
    删除 region 中被替换的 ops。
    
    Step 6: Discover new slots
    ────────────────────────
    如果 donor_ir 包含 AlgSlot ops：
      - 这些 AlgSlot 不会出现在 host 中（因为 donor 是 call 而非 inline）
      - 但 host 的 AlgorithmGenome 需要知道这些 slot 的存在
      - 返回的 AlgorithmArtifact.provenance 中记录 donor 的 slot 清单
      - 由 _execute_graft() 为每个新 slot 初始化 SlotPopulation
    
    Step 7: Rebuild & validate
    ────────────────────────
    从修改后的 xDSL module 重建 FunctionIR，验证结构完整性。
    """
```

**与当前 `_graft_bp_summary()` 的对比**：

| 特性 | 当前 `_graft_bp_summary` | 新 `graft_general` |
|------|--------------------------|---------------------|
| Donor 引入方式 | inline ops 到 host block | `call donor_fn(...)` |
| Donor 参数绑定 | 硬编码找 "frontier", "costs" | 通过 port_mapping |
| Donor 选择 | 通过 Skeleton.transform_rules 硬分发 | 通用，任意 FunctionIR |
| Op 冲突处理 | 手动分配新 op/value ID | 不需要（call 边界隔离） |
| Donor 控制流 | 不支持多 block donor | 天然支持（在 call 内部） |
| Slot 支持 | 无 | donor 可含 AlgSlot |

**call-based 嫁接的代价**：性能不如 inline（多一层函数调用），且 host 的 factgraph 无法"看透" donor 内部。
但好处是实现简单、正确性易保证。未来可以加 inline pass 作为优化。

---

#### 层 6：具体的嫁接例子（二层架构）

以用户提到的例子——"stack decoder + BP"：

```python
from algorithm_ir.frontend.compiler import compile_function_to_ir

# ── 初始算法池 ──
# 每个算法编译为 IR，发现其中的 slot，初始化子种群
def make_initial_genome(fn, algo_id: str) -> AlgorithmGenome:
    ir = compile_function_to_ir(fn)
    # 发现 IR 中的 AlgSlot 位置
    slot_ids = find_alg_slots(ir)
    populations = {}
    for sid in slot_ids:
        spec = infer_slot_spec(ir, sid)
        variants = [random_program(spec, rng) for _ in range(8)]
        populations[sid] = SlotPopulation(sid, spec, variants, [float("inf")]*8)
    return AlgorithmGenome(
        algo_id=algo_id,
        structural_ir=ir,
        slot_populations=populations,
        constants=np.zeros(0),
        generation=0,
        parent_ids=[],
        graft_history=[],
        tags={"original"},
    )

pool = [
    make_initial_genome(stack_decoder_host, "stack_decoder"),
    make_initial_genome(bp_summary_update, "bp_summary"),
    make_initial_genome(kbest_detector, "kbest"),
    make_initial_genome(lmmse_detector, "lmmse"),
]

# ── PatternMatcher 示例（由用户提供）──
def my_pattern_matcher(entries: list[AlgorithmEntry], gen: int) -> list[GraftProposal]:
    proposals = []
    
    for entry in entries:
        if entry.trace is None:
            continue
        
        # 分析运行轨迹，发现 metric 计算区域
        region = define_rewrite_region(
            entry.ir,
            exit_values=[find_metric_value(entry.ir)]
        )
        contract = infer_boundary_contract(
            entry.ir, region, entry.trace, entry.runtime_values
        )
        
        # 从 BP 算法中提取适合替换的结构
        bp_entry = find_entry_by_id(entries, "bp_summary")
        
        proposals.append(GraftProposal(
            proposal_id=f"graft_{gen}_{entry.algo_id}_bp",
            host_algo_id=entry.algo_id,
            region=region,
            contract=contract,
            donor_algo_id="bp_summary",
            donor_ir=bp_entry.ir,           # 完整 donor 函数 IR
            donor_region=None,              # 不做子图提取
            dependency_overrides=[
                DependencyOverride(
                    target_value="v_score",
                    new_dependencies=["v_frontier", "v_costs"],
                    reason="BP needs access to full frontier"
                )
            ],
            port_mapping={
                "v_0_donor": "v_frontier_host",
                "v_1_donor": "v_costs_host",
            },
            confidence=0.8,
            rationale="Stack decoder metric → BP message passing"
        ))
    
    return proposals

# ── 评估器示例 ──
class MIMOAlgorithmEvaluator(AlgorithmFitnessEvaluator):
    def evaluate(self, genome: AlgorithmGenome) -> FitnessResult:
        # materialize + 执行 + 计算 BER
        ...
    
    def evaluate_single_result(self, result: Any) -> float:
        # 快速评估：微层 slot 进化时使用
        return float(result)  # 或计算简单的 MSE

# ── 启动进化 ──
engine = AlgorithmEvolutionEngine(
    config=AlgorithmEvolutionConfig(
        pool_size=20,
        n_generations=100,
        micro_pop_size=8,
        micro_generations=5,
    ),
    evaluator=MIMOAlgorithmEvaluator(...),
    pattern_matcher=my_pattern_matcher,
    initial_algorithms=pool,
    test_inputs_fn=lambda: generate_mimo_test_case(),
)
best = engine.run(n_generations=100)
```

**进化过程示意**：
```
Gen 0: pool = [stack_decoder, bp_summary, kbest, lmmse]
  ├── 微层: 每个算法的 slot variants 进化 5 微代
  └── 评估: stack_decoder=0.7, bp=0.3, kbest=0.5, lmmse=0.4

Gen 1: PatternMatcher 建议 "stack_decoder.score → call bp_summary"
  ├── 嫁接: 产生 stack_decoder_bp (含新 slot "bp_damping")
  ├── 微层: stack_decoder_bp 的 slot variants 进化 5 微代
  └── 评估: ..., stack_decoder_bp=0.25  ← 新算法更好！

Gen 2: PatternMatcher 建议 "kbest.selection → call lmmse.filter"
  ├── 嫁接: 产生 kbest_lmmse
  ├── 微层: ...
  └── 选择: 淘汰适应度最差的算法，保留 pool_size=20 个
```

---

### 实施优先级

| 优先级 | 任务 | 复杂度 | 依赖 |
|---|---|---|---|
| **P0** | 定义 `AlgorithmEntry`, `GraftProposal`, `DependencyOverride`, `PatternMatcherFn` | 低 | 无 |
| **P0** | 定义 `SlotPopulation`, `AlgorithmGenome` 二层数据结构 | 中 | P0 数据类 |
| **P1** | 实现 `materialize()` 和 `materialize_with_override()` | 中 | AlgorithmGenome |
| **P1** | 实现 `graft_general()` (call-based，不 inline) | 中 | GraftProposal |
| **P1** | 实现 `AlgorithmFitnessEvaluator` 基类 | 低 | AlgorithmGenome |
| **P2** | 实现 `AlgorithmEvolutionEngine` 宏层循环 | 中 | P1 全部 |
| **P2** | 实现 `_evolve_slots()` 微层进化 | 中 | SlotPopulation |
| **P2** | 实现 `apply_dependency_overrides()` | 高 | BoundaryContract 扩展 |
| **P3** | 实现 `discover_new_slots()` / `infer_slot_spec()` | 中 | materialize |
| **P3** | 带 AlgSlot 的 donor 嫁接 | 中 | graft_general |
| **P4** | 可选：inline pass（将 call-based 嫁接展开为 inline ops） | 高 | graft_general 稳定后 |

---

### 设计决策记录

以下问题已在讨论中确定答案：

#### 决策 1：PatternMatcher 必须自己构造 RewriteRegion
**结论**：PatternMatcher 负责调用 `define_rewrite_region()` 和 `infer_boundary_contract()`，引擎不提供候选 region。
**理由**：匹配器需要完全控制区域选择策略——不同匹配器可能有完全不同的区域选择逻辑。

#### 决策 2：Donor 始终是完整的函数 IR
**结论**：`GraftProposal.donor_ir` 是完整的函数 IR（可能带 AlgSlot），不从 donor 中提取子图。
**理由**：大幅简化 `graft_general()` 的实现。嫁接 = 插入 `call donor_fn(...)` 而非 inline ops。避免了 op 重命名、多 block 合并、类型冲突等复杂问题。
**代价**：嫁接后 host 的 factgraph 无法"看透"donor 内部。可后续通过 inline pass 优化。

#### 决策 3：二层进化（结构 + Slot 子种群）
**结论**：`AlgorithmGenome` 采用二层结构：
- `structural_ir`：含 AlgSlot 的结构模板（宏层进化）
- `slot_populations`：每个 slot 的变体子种群（微层进化）
- 两层同步进化：每宏代做若干微代的 slot 进化

**理由**：高层结构特征（控制流骨架、嫁接拓扑）和低层实现细节（slot 中的具体计算）在不同时间尺度上变化。分层进化避免搜索空间爆炸。
**关键实现**：微层评估时固定其他 slot 用 `best_variant`，只变化目标 slot（坐标下降策略）。

#### 决策 4：暂时不允许多区域嫁接
**结论**：一代中对同一个 host 最多做一次嫁接。
**理由**：避免区域重叠的复杂性。未来可放开。

#### 决策 5：新的评估器基类
**结论**：新增 `AlgorithmFitnessEvaluator`，覆写现有 `FitnessEvaluator`。

```python
class AlgorithmFitnessEvaluator(ABC):
    """评估完整算法的适应度。
    
    与旧 FitnessEvaluator 的区别：
    - 输入是 AlgorithmGenome（完整算法），不是 IRGenome（多个小函数）
    - 可以访问 structural_ir 和 slot_populations 用于复杂度惩罚
    - evaluate_single_result() 用于微层 slot 快速评估
    """
    
    @abstractmethod
    def evaluate(self, genome: AlgorithmGenome) -> FitnessResult:
        """评估单个算法的适应度。"""
        ...
    
    def evaluate_batch(self, genomes: list[AlgorithmGenome]) -> list[FitnessResult]:
        """批量评估。默认逐个调用 evaluate()。"""
        return [self.evaluate(g) for g in genomes]
    
    @abstractmethod
    def evaluate_single_result(self, result: Any) -> float:
        """快速评估单次执行结果，用于微层 slot 进化。
        
        返回标量适应度（越小越好）。
        不做完整评估（省略复杂度惩罚等），只看输出质量。
        """
        ...
```

---

### GraftProposal 中 `donor_region` 字段的保留说明

虽然决策 2 确定 donor 是完整函数 IR，但 `GraftProposal.donor_region` 字段仍然保留：
- 当 `donor_region is None`：使用 donor 的完整函数（最常见情况）
- 当 `donor_region is not None`：指示 donor 中哪个区域是"有效内容"
  - 这用于 PatternMatcher 提供给引擎的**元信息**
  - 引擎本身不做子图提取，但可以用这个信息做 logging、可视化
  - 未来实现 inline pass 时可以利用此信息

