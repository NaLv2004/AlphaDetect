

## 修改方案：从固定程序进化到算法级嫁接进化

> **背景与摘要**
>
> AlphaDetect 项目旨在利用进化算法自动发现高性能 MIMO 检测算法。当前系统基于固定骨架 + 填空进化：预定义 4 个小程序的签名，在其内部做操作符/常量的随机搜索。这种架构无法产生**结构性新颖**的算法——进化被限制在骨架内部。
>
> 本重构计划定义了一套**算法级嫁接进化框架**，使系统能够：
> 1. 维护一个多粒度、分层的**算法池**（~30 个初始条目，从 L0 原子操作到 L3 完整检测器），所有条目统一表示为 `FunctionIR`；
> 2. 通过可插拔的 **PatternMatcher** 分析算法池，自动产生 **GraftProposal**（嫁接提案）——即"将算法 A 的某区域替换为算法 B 的结构"；
> 3. 由通用 **graft_general()** 引擎执行 call-based 嫁接，生成结构上全新的算法；
> 4. 采用**二层进化**：宏层做结构嫁接（跨算法组合），微层在每个 slot 子种群内做 GP 变异——两者在不同时间尺度上协同进化；
>
> 本计划分 6 层逐步展开（数据类型 → IR 修改 → 进化框架 → 嫁接引擎 → 嫁接示例 → 多粒度算法池设计），并附有实施优先级（P0~P4）、设计决策记录、兼容性分析、以及各阶段测试要求。
>
> **注意**：本框架是通用的算法进化系统——不以任何特定检测器（如 BP-Stack）为唯一目标。BP-Stack、K-Best、EP、AMP 等检测器是算法池中平等的初始条目，进化可以自由组合、替换、融合其中任何子结构。

---

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

本质区别：**当前框架进化的是「填空的内容」，目标框架进化的是「整个算法的结构」**(同时进化填空的内容)。

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

以 "stack decoder + BP 消息传递" 作为跨算法嫁接的**示例**（仅演示嫁接工作流，非唯一目标）：

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

---

### 依赖变更与 Slot 兼容性协议

#### 问题

当嫁接操作通过 `DependencyOverride` 引入新的依赖（例如 donor 需要 host 中原本不存在的变量），或当 donor 的输入/输出类型与 host 的 slot 规范不完全匹配时，slot 的定义（`ProgramSpec` / `SlotDescriptor`）是否需要改变？如何保证兼容性？

#### 兼容性规则

**规则 1：Slot 的类型签名是不可变契约**

slot 的 `ProgramSpec`（输入/输出类型签名）在创建后**不可修改**。如果 donor 的类型签名与目标 slot 的 spec 不兼容，嫁接被**拒绝**——而非修改 spec。

```python
def is_slot_compatible(slot_spec: ProgramSpec, donor_ir: FunctionIR) -> bool:
    """检查 donor 是否与 slot 的类型签名兼容。"""
    donor_spec = infer_spec_from_ir(donor_ir)
    return (
        donor_spec.input_types == slot_spec.input_types
        and donor_spec.output_types == slot_spec.output_types
    )
```

**规则 2：DependencyOverride 扩展的是 BoundaryContract，而非 SlotDescriptor**

`DependencyOverride` 的作用是告诉 `graft_general()` 在嫁接时将额外的值传入 donor 的 call 边界。这体现在 `BoundaryContract.input_ports` 的扩展上，而**不影响**已有 slot 的定义。

```
Before graft:
  host: ... → call slot_fn(a, b) → ...
  slot_spec: (TypeA, TypeB) → TypeC

After graft with DependencyOverride:
  host: ... → call donor_fn(a, b, NEW_DEP) → ...
  NEW slot_spec: (TypeA, TypeB, TypeD) → TypeC  ← 这是一个新的 slot！

关键: 旧的 slot 不被修改。嫁接产生的新结构中，对应位置是一个全新的 slot 
(新的 slot_id, 新的 ProgramSpec)。旧 slot 的 SlotPopulation 不被复用。
```

**规则 3：嫁接后新 slot 的创建协议**

```python
def create_new_slot_after_graft(
    old_slot: SlotDescriptor,
    dependency_overrides: list[DependencyOverride],
    donor_ir: FunctionIR,
) -> SlotDescriptor:
    """嫁接改变依赖后，创建新的 slot 描述符。"""
    new_input_types = list(old_slot.spec.input_types)
    for override in dependency_overrides:
        new_input_types.extend(override.new_dependency_types)
    
    new_spec = ProgramSpec(
        input_types=tuple(new_input_types),
        output_types=old_slot.spec.output_types,
    )
    
    return SlotDescriptor(
        slot_id=f"{old_slot.slot_id}__grafted_{hash(donor_ir)}",
        short_name=old_slot.short_name,
        level=old_slot.level,
        depth=old_slot.depth,
        parent_slot_id=old_slot.parent_slot_id,
        child_slot_ids=[],  # donor 的子 slot 会被单独发现
        spec=new_spec,
        default_impl=donor_ir,
        default_impl_level=infer_level(donor_ir),
        mutable=True,
        description=f"Grafted from {old_slot.slot_id}",
    )
```

**规则 4：版本化兼容性检查**

为防止进化过程中累积不兼容变更，每个 `AlgorithmGenome` 维护一个 `slot_version_map`：

```python
@dataclass
class AlgorithmGenome:
    # ... 已有字段 ...
    slot_version_map: dict[str, int] = field(default_factory=dict)
    # 每次嫁接改变 slot 时 version +1
    # 微层进化只能在 same-version 的 slot 间交叉
```

**规则 5：跨版本 SlotPopulation 迁移**

当嫁接导致 slot 的类型签名改变时，旧 SlotPopulation 中的变体**不可直接复用**。处理策略：

| 情况 | 处理方式 |
|------|----------|
| 新 spec 的输入是旧 spec 的超集（新增参数） | 旧变体通过包装函数（忽略新参数）迁移 |
| 新 spec 与旧 spec 完全不同 | 旧变体丢弃，从 donor 的默认实现 + 随机初始化重建 |
| 新 spec 与旧 spec 相同 | 旧变体直接复用 |

```python
def migrate_slot_population(
    old_pop: SlotPopulation, 
    old_spec: ProgramSpec,
    new_spec: ProgramSpec,
    rng: np.random.Generator,
) -> SlotPopulation:
    """将旧种群迁移到新的类型签名。"""
    if old_spec == new_spec:
        return old_pop  # 完全兼容
    
    if is_superset_inputs(new_spec, old_spec):
        # 包装: 旧变体忽略多余参数
        migrated = [wrap_ignore_extra_args(v, old_spec, new_spec) for v in old_pop.variants]
        return SlotPopulation(new_spec=new_spec, variants=migrated, ...)
    
    # 不兼容: 重新初始化
    return initialize_slot_population(new_spec, rng)
```

---

### 静态 IR 图与运行时动态图的鲁棒处理

#### 问题

系统中存在两种算法表示：
1. **静态 IR 图** (`FunctionIR`)：编译期的程序结构——ops、blocks、value 定义/使用关系。不包含运行时信息。
2. **运行时动态图**：执行期的信息——`RuntimeEvent`（跟踪哪些 op 被执行、执行顺序）、`RuntimeValue`（每个 value 的实际值/统计量）、`FactGraph`（从运行时数据推断的因果/依赖关系）。

两种表示在 `AlgorithmEntry` 中共存：
```python
@dataclass
class AlgorithmEntry:
    ir: FunctionIR                              # 静态结构
    trace: list[RuntimeEvent] | None            # 动态执行轨迹
    runtime_values: dict[str, RuntimeValue] | None  # 运行时值
    factgraph: FactGraph | None                 # 运行时因果图
```

嫁接和进化需要**同时利用两种表示**，且两者可能不一致（例如死代码在 IR 中存在但 trace 中没有）。本节明确处理策略。

#### 设计原则

**原则 1：静态 IR 是唯一的正确性依据**

所有代码变换（嫁接、slot 替换、materialize）仅基于静态 IR。运行时信息是**辅助性的**——用于分析和指导，但不参与代码变换的正确性验证。

```
代码变换正确性链:
  FunctionIR → graft_general() → new FunctionIR → validate_ir() ✓
  (不依赖 trace/factgraph)
```

**原则 2：运行时信息是 PatternMatcher 的输入，而非引擎的输入**

`AlgorithmEvolutionEngine` 的 `_execute_graft()` 不使用 trace/factgraph。它们的消费者是 PatternMatcher：

```python
# PatternMatcher 使用运行时信息做决策：
def my_pattern_matcher(entries: list[AlgorithmEntry], gen: int) -> list[GraftProposal]:
    for entry in entries:
        if entry.trace is None:
            continue
        # 分析 trace 发现热点区域 (哪些 op 被频繁执行)
        hot_ops = find_hot_ops(entry.trace)
        # 分析 factgraph 发现瓶颈 (哪些依赖链最长)
        bottlenecks = find_bottleneck_chains(entry.factgraph)
        # 分析 runtime_values 发现数值特征 (哪些值接近 0, 方差大)
        anomalies = find_value_anomalies(entry.runtime_values)
        # 基于以上分析构造 GraftProposal
        ...
```

**原则 3：嫁接后运行时信息失效并重建**

嫁接产生的新算法没有运行时信息（它还没被执行过）。引擎的处理：

```python
def _execute_graft(self, proposal: GraftProposal) -> AlgorithmGenome:
    # 1. 执行嫁接 (纯静态操作)
    new_ir = graft_general(host_ir, proposal)
    
    # 2. 创建新 genome (无运行时信息)
    new_genome = AlgorithmGenome(structural_ir=new_ir, ...)
    
    # 3. 在评估阶段, 新 genome 被执行后获得运行时信息
    #    → 下一代的 AlgorithmEntry 中 trace/factgraph 被填充
    return new_genome

def _build_entries(self, genomes, fitness_results) -> list[AlgorithmEntry]:
    entries = []
    for genome, result in zip(genomes, fitness_results):
        entries.append(AlgorithmEntry(
            ir=genome.structural_ir,
            trace=result.trace if result else None,         # 执行后才有
            runtime_values=result.values if result else None,
            factgraph=build_factgraph(result) if result else None,
            ...
        ))
    return entries
```

**原则 4：静态图与动态图的一致性校验**

当两种表示同时存在时，进行一致性校验作为健壮性保障（不阻塞进化，仅告警）：

```python
def check_static_dynamic_consistency(
    ir: FunctionIR, 
    trace: list[RuntimeEvent]
) -> list[ConsistencyWarning]:
    """检查静态 IR 与运行时 trace 的一致性。"""
    warnings = []
    
    # 1. 死代码检测: IR 中的 op 未出现在 trace 中
    traced_ops = {e.op_id for e in trace}
    for op in ir.all_ops():
        if op.id not in traced_ops:
            warnings.append(DeadCodeWarning(op_id=op.id))
    
    # 2. 类型一致性: runtime value 的实际类型与 IR 声明的类型匹配
    for event in trace:
        ir_type = ir.get_value_type(event.result_value)
        runtime_type = event.actual_type
        if ir_type != runtime_type:
            warnings.append(TypeMismatchWarning(event.result_value, ir_type, runtime_type))
    
    # 3. 控制流一致性: trace 中的 block 执行顺序与 IR 中的 CFG 兼容
    trace_blocks = extract_block_sequence(trace)
    if not is_valid_cfg_path(ir.cfg, trace_blocks):
        warnings.append(CFGInconsistencyWarning(trace_blocks))
    
    return warnings
```

**原则 5：FactGraph 的增量更新**

当 slot 内部的实现被微层进化替换时，只有被替换 slot 对应的 FactGraph 子图需要重建：

```python
def update_factgraph_after_slot_change(
    full_factgraph: FactGraph,
    changed_slot_id: str,
    new_trace: list[RuntimeEvent],  # 只重新执行变更的 slot 部分
) -> FactGraph:
    """增量更新 FactGraph: 只重建变更 slot 对应的子图。"""
    # 删除旧 slot 的因果边
    full_factgraph.remove_subgraph(changed_slot_id)
    # 从新 trace 构建 slot 的因果子图
    slot_subgraph = build_factgraph_from_trace(new_trace)
    # 合并回完整图
    full_factgraph.merge(slot_subgraph, boundary=changed_slot_id)
    return full_factgraph
```

#### 鲁棒性保障总结

| 场景 | 静态 IR | 运行时信息 | 处理策略 |
|------|---------|-----------|----------|
| 新编译的算法 | ✓ | ✗ | 正常：等待首次评估后获得 trace |
| 嫁接后的新算法 | ✓ | ✗ | 正常：graft_general 仅操作 IR |
| 微层 slot 变异后 | ✓ (局部变更) | ✓ (部分失效) | 增量重建 FactGraph 的变更子图 |
| 评估失败（崩溃/超时） | ✓ | ✗ | 赋予惩罚适应度，保留 IR 用于分析 |
| trace 与 IR 不一致 | ✓ | ✓ (不一致) | 告警但不阻塞；以 IR 为准 |
| FactGraph 缺失 | ✓ | 部分 | PatternMatcher 降级为仅用 IR 分析 |

---

### 各阶段测试计划与 Git 同步

> **核心原则**：每个实施阶段 (P0~P4) 必须有完整的单元测试和集成测试。所有测试通过后，方可提交 git 并进入下一阶段。

#### P0 阶段测试：数据类型定义

| 测试项 | 测试内容 | 验证标准 |
|--------|----------|----------|
| T0.1 AlgorithmEntry 创建 | 构造 AlgorithmEntry，验证所有字段可读写 | 所有字段有默认值，类型正确 |
| T0.2 GraftProposal 验证 | 创建 GraftProposal，测试 port_mapping 格式 | 非空 host/donor ID，port_mapping 值有效 |
| T0.3 DependencyOverride | 创建 DependencyOverride，验证序列化/反序列化 | round-trip 一致 |
| T0.4 PatternMatcherFn 签名 | 使用 dummy matcher 验证回调签名 | `(list[AlgorithmEntry], int) → list[GraftProposal]` |
| T0.5 SlotPopulation | 创建 SlotPopulation，测试 best_variant 属性 | 返回适应度最优的变体 |
| T0.6 AlgorithmGenome | 完整构造 AlgorithmGenome，含 slot_populations | 所有字段类型正确，slot_populations 与 structural_ir 一致 |
| T0.7 SlotDescriptor | 创建嵌套 slot 层次，验证 parent/child 关系 | slot 树的层次/深度/路径正确 |
| T0.8 向后兼容 | 现有 119 个测试全部通过 | 无回归 |

```
# P0 完成检查清单:
✅ 所有 T0.x 测试通过
✅ 现有 119 个测试无回归
→ git add + git commit -m "P0: data type definitions for algorithm-level evolution"
→ 进入 P1
```

#### P1 阶段测试：核心操作实现

| 测试项 | 测试内容 | 验证标准 |
|--------|----------|----------|
| T1.1 materialize 基础 | 单 slot 的 AlgorithmGenome → 完整 FunctionIR | 输出 IR 无 AlgSlot op，可执行 |
| T1.2 materialize 多 slot | 3+ 个 slot 的 genome → IR | 所有 slot 被填充 |
| T1.3 materialize_with_override | 用指定 variant 覆盖默认 | 输出 IR 使用指定的 variant |
| T1.4 graft_general 简单嫁接 | host + 无依赖 donor → 新 IR | 新 IR 包含 call donor_fn，类型正确 |
| T1.5 graft_general 带 DependencyOverride | 嫁接时扩展输入 | 新 call 的参数列表包含 override 的新依赖 |
| T1.6 graft_general 验证 | 对嫁接后的 IR 执行 validate_ir() | 结构完整性通过 |
| T1.7 graft_general donor 含 AlgSlot | donor 自带 slot | 新 genome 的 slot_populations 包含 donor 的 slot |
| T1.8 AlgorithmFitnessEvaluator | 实现一个 dummy evaluator，测试接口 | evaluate + evaluate_single_result 可调用 |
| T1.9 向后兼容 | 现有测试全部通过 | 无回归 |

```
# P1 完成检查清单:
✅ 所有 T1.x 测试通过
✅ 现有测试无回归
✅ materialize → 执行 → 得到正确输出 (端到端验证)
→ git add + git commit -m "P1: materialize, graft_general, evaluator base"
→ 进入 P2
```

#### P2 阶段测试：进化引擎

| 测试项 | 测试内容 | 验证标准 |
|--------|----------|----------|
| T2.1 宏层单代 | 2 个 genome，运行 1 宏代 | 种群大小正确，适应度被评估 |
| T2.2 宏层 + 嫁接 | 提供 PatternMatcher 返回 1 个 GraftProposal | 新 genome 出现在种群中 |
| T2.3 微层进化 | 单个 genome 的 slot 进化 5 微代 | slot_populations 中的 fitness 被更新 |
| T2.4 二层联合 | 宏层 + 微层联合运行 3 代 | 适应度单调下降（简单问题上） |
| T2.5 apply_dependency_overrides | 对 IR 应用多个 override | BoundaryContract 正确扩展 |
| T2.6 选择压力 | 种群淘汰低适应度个体 | 存活个体的平均适应度优于上一代 |
| T2.7 graft_history 追踪 | 嫁接后检查 provenance | graft_history 记录所有嫁接操作 |
| T2.8 向后兼容 | 现有测试全部通过 | 无回归 |

```
# P2 完成检查清单:
✅ 所有 T2.x 测试通过
✅ 现有测试无回归
✅ 二层进化在简单 benchmark 上收敛
→ git add + git commit -m "P2: AlgorithmEvolutionEngine macro+micro loop"
→ 进入 P3
```

#### P3 阶段测试：高级功能

| 测试项 | 测试内容 | 验证标准 |
|--------|----------|----------|
| T3.1 discover_new_slots | 从含 AlgSlot 的 IR 中发现 slot 树 | 所有 AlgSlot 被正确识别，层次正确 |
| T3.2 infer_slot_spec | 从 IR 中推断 slot 的类型签名 | 与手动标注一致 |
| T3.3 materialize_hierarchical | 4 层嵌套 slot 的递归填充 | 输出 IR 无 AlgSlot，可执行 |
| T3.4 带 AlgSlot 的 donor 嫁接 | donor 自带 slot → 嫁接后子种群初始化 | 新种群有正确的 ProgramSpec |
| T3.5 slot 兼容性检查 | 类型不匹配的 donor 被拒绝 | is_slot_compatible 正确返回 False |
| T3.6 SlotPopulation 迁移 | 嫁接后旧种群迁移到新 spec | 迁移后变体可执行 |
| T3.7 向后兼容 | 现有测试全部通过 | 无回归 |

```
# P3 完成检查清单:
✅ 所有 T3.x 测试通过
✅ 现有测试无回归
✅ 多层嵌套 slot 的完整进化流程可运行
→ git add + git commit -m "P3: slot discovery, hierarchical materialize, donor with slots"
→ 进入 P4 (可选) 或进入 E2E 集成测试
```

#### P4 阶段测试：优化 (可选)

| 测试项 | 测试内容 | 验证标准 |
|--------|----------|----------|
| T4.1 inline pass | call-based 嫁接 → inline 展开 | 展开后 IR 与 call 版本语义等价 |
| T4.2 inline 性能 | inline 后执行速度提升 | 对比 call 版本有可测量的加速 |
| T4.3 inline 正确性 | inline 后 op 不冲突 | 无 value ID 重复 |

---

### 端到端集成测试：完整进化流程验证

> **此测试是整个框架最重要的验证**。它必须走完从初始化到多代进化到输出最优算法的完整流程，覆盖所有核心子系统的交互。

#### 测试设计

```python
class TestEndToEndEvolution:
    """端到端进化集成测试。
    
    验证目标: 从初始算法池出发，经过多代二层进化（含嫁接），
    产生适应度持续改善的新算法。
    """

    def test_full_evolution_pipeline(self):
        """完整流水线测试。"""
        
        # ─── 1. 初始化 ───
        # 构造 3~5 个初始算法 (简化版，不需要完整 MIMO 检测器)
        # 使用简单的数学函数作为 host/donor，验证框架本身的正确性
        pool = build_test_initial_pool(rng)
        assert len(pool) >= 3
        
        # 验证每个 genome 结构完整
        for genome in pool:
            assert genome.structural_ir is not None
            assert len(genome.slot_populations) > 0
            ir = materialize(genome)
            validate_ir(ir)  # 可 materialize 且 IR 合法
        
        # ─── 2. 配置引擎 ───
        config = AlgorithmEvolutionConfig(
            pool_size=10,
            n_generations=5,         # 足够测试多代
            micro_pop_size=4,
            micro_generations=3,
        )
        evaluator = SimpleTestEvaluator()  # 使用简单目标函数
        matcher = SimpleTestPatternMatcher()  # 至少产生 1 个 GraftProposal
        
        engine = AlgorithmEvolutionEngine(
            config=config,
            evaluator=evaluator,
            pattern_matcher=matcher,
            initial_algorithms=pool,
            test_inputs_fn=lambda: generate_simple_test_case(),
        )
        
        # ─── 3. 运行进化 ───
        best = engine.run(n_generations=5)
        
        # ─── 4. 验证结果 ───
        # 4a. 最终适应度优于初始最佳
        initial_best_fitness = min(
            evaluator.evaluate(g).fitness for g in pool
        )
        final_fitness = evaluator.evaluate(best).fitness
        assert final_fitness <= initial_best_fitness, \
            f"Evolution should improve: {final_fitness} > {initial_best_fitness}"
        
        # 4b. 进化产生了新算法 (不全是初始个体)
        all_algo_ids = {g.algo_id for g in engine.current_population}
        original_ids = {g.algo_id for g in pool}
        assert all_algo_ids != original_ids, "Evolution should produce new algorithms"
        
        # 4c. 至少发生了一次嫁接
        grafted = [g for g in engine.current_population if len(g.graft_history) > 0]
        assert len(grafted) > 0, "At least one graft should have occurred"
        
        # 4d. 嫁接后的算法可以正确 materialize 和执行
        for genome in grafted:
            ir = materialize(genome)
            validate_ir(ir)
            result = evaluator.evaluate(genome)
            assert result.fitness < float("inf"), "Grafted algorithm should produce valid output"
        
        # 4e. 微层进化确实在工作
        for genome in engine.current_population:
            for slot_id, pop in genome.slot_populations.items():
                assert not all(f == float("inf") for f in pop.fitness), \
                    f"Slot {slot_id} should have been evaluated"

    def test_evolution_with_dependency_override(self):
        """测试含 DependencyOverride 的嫁接在完整进化流程中的表现。"""
        pool = build_test_pool_with_dependency_cases(rng)
        matcher = DependencyOverrideTestMatcher()
        engine = AlgorithmEvolutionEngine(
            config=AlgorithmEvolutionConfig(pool_size=6, n_generations=3, ...),
            evaluator=SimpleTestEvaluator(),
            pattern_matcher=matcher,
            initial_algorithms=pool,
            ...
        )
        best = engine.run(n_generations=3)
        # 验证依赖扩展后的 slot 被正确创建和进化
        for genome in engine.current_population:
            if any("dependency_override" in str(g) for g in genome.graft_history):
                assert genome.slot_version_map  # 版本号已更新
    
    def test_hierarchical_slot_evolution(self):
        """测试多层嵌套 slot 的进化。"""
        # 构造一个 3 层嵌套的 genome
        genome = build_deeply_nested_test_genome(rng)
        assert max(d.depth for d in genome.metadata["slot_tree"].values()) >= 2
        
        # 单独测试微层进化对深层 slot 的覆盖
        engine = AlgorithmEvolutionEngine(...)
        engine._evolve_slots(genome, evaluator, test_inputs)
        
        # 验证所有层次的 slot 都被进化触及
        for slot_id, pop in genome.slot_populations.items():
            desc = genome.metadata["slot_tree"][slot_id]
            if desc.mutable:
                assert any(f < float("inf") for f in pop.fitness), \
                    f"Slot {slot_id} (depth={desc.depth}) should be evaluated"

    def test_static_dynamic_consistency(self):
        """测试静态 IR 与运行时 trace 的一致性校验。"""
        pool = build_test_initial_pool(rng)
        engine = AlgorithmEvolutionEngine(...)
        engine.run(n_generations=2)
        
        entries = engine._build_entries(engine.current_population, ...)
        for entry in entries:
            if entry.trace is not None:
                warnings = check_static_dynamic_consistency(entry.ir, entry.trace)
                # 在正常情况下不应有严重不一致
                critical = [w for w in warnings if isinstance(w, TypeMismatchWarning)]
                assert len(critical) == 0, f"Type mismatches found: {critical}"
    
    def test_slot_compatibility_enforcement(self):
        """测试 slot 兼容性规则在整个流程中被正确执行。"""
        pool = build_test_initial_pool(rng)
        # 提供一个故意返回不兼容 donor 的 matcher
        bad_matcher = IncompatibleDonorTestMatcher()
        engine = AlgorithmEvolutionEngine(
            ..., pattern_matcher=bad_matcher, initial_algorithms=pool, ...
        )
        engine.run(n_generations=2)
        # 不兼容的嫁接应被拒绝，不产生无效个体
        for genome in engine.current_population:
            ir = materialize(genome)
            validate_ir(ir)  # 所有个体仍然是合法的
```

```
# 端到端测试完成检查清单:
✅ test_full_evolution_pipeline 通过
✅ test_evolution_with_dependency_override 通过
✅ test_hierarchical_slot_evolution 通过
✅ test_static_dynamic_consistency 通过
✅ test_slot_compatibility_enforcement 通过
✅ 所有 P0~P3 单元测试仍通过
✅ 现有 119 个旧测试无回归
→ git add + git commit -m "E2E: full evolution pipeline integration tests"
→ git tag v0.5-algorithm-evolution
```

---

## 初始算法池：多粒度分层设计

### 设计哲学

初始算法池**不应该仅包含完整的 MIMO 检测器**。它必须是一个多粒度、分层的操作库：从最基本的标量/向量/矩阵运算，到中等粒度的信号处理模式，再到完整的检测算法。这样做的原因：

1. **嫁接的供体不一定是完整算法**——进化可能想把一个"距离计算模块"嫁接到搜索树的评分函数中，这个供体只是一个 Level-1 的复合操作，不是完整检测器。
2. **多层次进化**——高层算法的 slot 可以被低层原语替换，低层原语可以被组合成新的中层模块。进化在所有层次上同时发生。
3. **与研究提案 DSL 的对齐**——`research_proposal.tex` 定义了四层类型体系（数学基础→信号处理→概率图→算法控制），算法池的分层与之对应。

### 分层架构总览

```
Level 0: 原子操作 (Atomic Ops)
  ├── 标量运算 (Scalar)
  ├── 向量运算 (Vector)
  ├── 矩阵运算 (Matrix)
  ├── 复数运算 (Complex)
  └── 统计/聚合 (Statistical)

Level 1: 复合操作 (Composite Ops)
  ├── 线性代数模式 (LinAlg Patterns)
  ├── 距离/度量 (Distance/Metric)
  ├── 滤波/均衡 (Filtering/Equalization)
  └── 分布操作 (Distribution Ops)

Level 2: 算法模块 (Algorithm Modules)
  ├── 树搜索操作 (Tree Search)
  ├── 图/消息传递操作 (Graph/Message Passing)
  ├── 推断操作 (Inference Ops)
  └── 迭代策略 (Iterative Strategies)

Level 3: 完整算法 (Full Algorithms)
  ├── 线性检测器 (Linear Detectors)
  ├── 树搜索检测器 (Tree Search Detectors)
  ├── 消息传递检测器 (Message Passing Detectors)
  └── 近似推断检测器 (Approximate Inference Detectors)
```

---

### Level 0：原子操作

这些是不可再分解的基本运算。它们**没有内部 slot**——是进化的"叶节点"。每个原子操作有明确的类型签名和复杂度标注。

#### 0.1 标量运算 (Scalar Ops)

| ID | 名称 | 签名 | 复杂度 | 说明 |
|----|------|-------|--------|------|
| `s.add` | 加法 | `(f64, f64) → f64` | O(1) | a + b |
| `s.sub` | 减法 | `(f64, f64) → f64` | O(1) | a - b |
| `s.mul` | 乘法 | `(f64, f64) → f64` | O(1) | a * b |
| `s.div` | 除法 | `(f64, f64) → f64` | O(1) | a / b (safe) |
| `s.abs` | 绝对值 | `f64 → f64` | O(1) | |a| |
| `s.neg` | 取负 | `f64 → f64` | O(1) | -a |
| `s.sqrt` | 平方根 | `f64 → f64` | O(1) | √a (safe) |
| `s.square` | 平方 | `f64 → f64` | O(1) | a² |
| `s.exp` | 指数 | `f64 → f64` | O(1) | eᵃ (clamped) |
| `s.log` | 对数 | `f64 → f64` | O(1) | ln(a) (safe) |
| `s.tanh` | 双曲正切 | `f64 → f64` | O(1) | tanh(a) |
| `s.sigmoid` | Sigmoid | `f64 → f64` | O(1) | 1/(1+e⁻ᵃ) |
| `s.relu` | ReLU | `f64 → f64` | O(1) | max(0, a) |
| `s.min` | 最小值 | `(f64, f64) → f64` | O(1) | min(a, b) |
| `s.max` | 最大值 | `(f64, f64) → f64` | O(1) | max(a, b) |
| `s.clamp` | 截断 | `(f64, f64, f64) → f64` | O(1) | clamp(x, lo, hi) |
| `s.sign` | 符号 | `f64 → f64` | O(1) | sign(a) |
| `s.inv` | 倒数 | `f64 → f64` | O(1) | 1/a (safe) |
| `s.lerp` | 线性插值 | `(f64, f64, f64) → f64` | O(1) | a + t*(b-a) |

#### 0.2 向量运算 (Vector Ops)

| ID | 名称 | 签名 | 复杂度 | 说明 |
|----|------|-------|--------|------|
| `v.add` | 向量加 | `(Vec[n], Vec[n]) → Vec[n]` | O(n) | 逐元素 |
| `v.sub` | 向量减 | `(Vec[n], Vec[n]) → Vec[n]` | O(n) | 逐元素 |
| `v.scale` | 标量乘 | `(Vec[n], f64) → Vec[n]` | O(n) | α·v |
| `v.dot` | 内积 | `(Vec[n], Vec[n]) → f64` | O(n) | v₁ᴴv₂ |
| `v.norm2` | 2-范数 | `Vec[n] → f64` | O(n) | ‖v‖₂ |
| `v.norm2sq` | 范数平方 | `Vec[n] → f64` | O(n) | ‖v‖₂² |
| `v.normalize` | 归一化 | `Vec[n] → Vec[n]` | O(n) | v/‖v‖ |
| `v.element_at` | 取元素 | `(Vec[n], int) → f64` | O(1) | v[i] |
| `v.set_element` | 设元素 | `(Vec[n], int, f64) → Vec[n]` | O(1) | v[i]=x |
| `v.concat` | 拼接 | `(Vec[m], Vec[n]) → Vec[m+n]` | O(m+n) | |
| `v.slice` | 切片 | `(Vec[n], int, int) → Vec[k]` | O(k) | v[i:j] |
| `v.hadamard` | 逐元素乘 | `(Vec[n], Vec[n]) → Vec[n]` | O(n) | v₁⊙v₂ |
| `v.softmax` | Softmax | `Vec[n] → Vec[n]` | O(n) | 概率归一化 |
| `v.log_softmax` | Log-Softmax | `Vec[n] → Vec[n]` | O(n) | 数值稳定版 |
| `v.cumsum` | 累积和 | `Vec[n] → Vec[n]` | O(n) | prefix sum |
| `v.reverse` | 翻转 | `Vec[n] → Vec[n]` | O(n) | |
| `v.abs` | 逐元素绝对值 | `Vec[n] → Vec[n]` | O(n) | |v_i| |
| `v.neg` | 取负 | `Vec[n] → Vec[n]` | O(n) | -v |

#### 0.3 矩阵运算 (Matrix Ops)

| ID | 名称 | 签名 | 复杂度 | 说明 |
|----|------|-------|--------|------|
| `m.matmul` | 矩阵乘法 | `(Mat[m,k], Mat[k,n]) → Mat[m,n]` | O(mkn) | AB |
| `m.matvec` | 矩阵向量乘 | `(Mat[m,n], Vec[n]) → Vec[m]` | O(mn) | Av |
| `m.transpose` | 转置 | `Mat[m,n] → Mat[n,m]` | O(mn) | Aᵀ |
| `m.conj_transpose` | 共轭转置 | `Mat[m,n] → Mat[n,m]` | O(mn) | Aᴴ |
| `m.add` | 矩阵加法 | `(Mat[m,n], Mat[m,n]) → Mat[m,n]` | O(mn) | A+B |
| `m.scale` | 标量乘 | `(Mat[m,n], f64) → Mat[m,n]` | O(mn) | αA |
| `m.inverse` | 求逆 | `Mat[n,n] → Mat[n,n]` | O(n³) | A⁻¹ |
| `m.solve` | 线性求解 | `(Mat[n,n], Vec[n]) → Vec[n]` | O(n³) | A⁻¹b |
| `m.qr` | QR 分解 | `Mat[m,n] → (Mat[m,n], Mat[n,n])` | O(mn²) | Q, R |
| `m.cholesky` | Cholesky 分解 | `Mat[n,n] → Mat[n,n]` | O(n³/3) | L: A=LLᴴ |
| `m.svd` | SVD 分解 | `Mat[m,n] → (Mat, Vec, Mat)` | O(mn²) | U,Σ,V |
| `m.diag` | 取/构造对角 | `Vec[n] → Mat[n,n]` 或 `Mat → Vec` | O(n) | |
| `m.trace` | 迹 | `Mat[n,n] → f64` | O(n) | tr(A) |
| `m.det` | 行列式 | `Mat[n,n] → f64` | O(n³) | det(A) |
| `m.eye` | 单位阵 | `int → Mat[n,n]` | O(n) | I_n |
| `m.row` | 取行 | `(Mat[m,n], int) → Vec[n]` | O(n) | A[i,:] |
| `m.col` | 取列 | `(Mat[m,n], int) → Vec[m]` | O(m) | A[:,j] |
| `m.element_at` | 取元素 | `(Mat, int, int) → f64` | O(1) | A[i,j] |
| `m.outer` | 外积 | `(Vec[m], Vec[n]) → Mat[m,n]` | O(mn) | uvᴴ |
| `m.gram` | Gram 矩阵 | `Mat[m,n] → Mat[n,n]` | O(mn²) | AᴴA |
| `m.regularize` | 正则化 | `(Mat[n,n], f64) → Mat[n,n]` | O(n) | A + σ²I |

#### 0.4 复数运算 (Complex Ops)

| ID | 名称 | 签名 | 复杂度 |
|----|------|-------|--------|
| `c.real` | 取实部 | `cx → f64` | O(1) |
| `c.imag` | 取虚部 | `cx → f64` | O(1) |
| `c.conj` | 共轭 | `cx → cx` | O(1) |
| `c.abs2` | 模的平方 | `cx → f64` | O(1) |
| `c.phase` | 相位角 | `cx → f64` | O(1) |
| `c.from_polar` | 极坐标构造 | `(f64, f64) → cx` | O(1) |
| `c.mul` | 复数乘法 | `(cx, cx) → cx` | O(1) |

#### 0.5 统计/聚合运算 (Statistical/Aggregation Ops)

| ID | 名称 | 签名 | 复杂度 |
|----|------|-------|--------|
| `st.mean` | 均值 | `Vec[n] → f64` | O(n) |
| `st.var` | 方差 | `Vec[n] → f64` | O(n) |
| `st.argmin` | 最小值索引 | `Vec[n] → int` | O(n) |
| `st.argmax` | 最大值索引 | `Vec[n] → int` | O(n) |
| `st.sort` | 排序 | `Vec[n] → Vec[n]` | O(n log n) |
| `st.argsort` | 排序索引 | `Vec[n] → Vec[int,n]` | O(n log n) |
| `st.top_k` | 取最大 k 个 | `(Vec[n], int) → (Vec[k], Vec[int,k])` | O(n log k) |
| `st.min_k` | 取最小 k 个 | `(Vec[n], int) → (Vec[k], Vec[int,k])` | O(n log k) |
| `st.cummin` | 累积最小 | `Vec[n] → Vec[n]` | O(n) |
| `st.logsumexp` | Log-Sum-Exp | `Vec[n] → f64` | O(n) |
| `st.weighted_sum` | 加权求和 | `(Vec[n], Vec[n]) → f64` | O(n) |

---

### Level 1：复合操作

由 Level-0 原子操作组合而成的常用模式。每个复合操作**内部有 1~3 个 slot**，表示可被进化替换的子步骤。

#### 1.1 线性代数模式 (LinAlg Patterns)

```python
# ---- 1.1.1 Regularized Solve (正则化求解) ----
# slot 标注: 两个可进化点
def regularized_solve(A: Mat, b: Vec, sigma2: f64) -> Vec:
    """(AᴴA + σ²I)⁻¹ Aᴴb"""
    G = m.gram(A)                          # 固定: Gram 矩阵
    G_reg = SLOT["regularizer"](G, sigma2) # ★ Slot L1: 正则化策略
    rhs = SLOT["rhs_transform"](A, b)      # ★ Slot L1: 右端项构造
    return m.solve(G_reg, rhs)             # 固定: 线性求解

# Slot 规范:
#   "regularizer": (Mat[n,n], f64) → Mat[n,n]
#       默认实现: lambda G, s: G + s * I
#       进化空间: G + f(s)*I, G + s*diag(G), Tikhonov 变体, ...
#   "rhs_transform": (Mat[m,n], Vec[m]) → Vec[n]
#       默认实现: lambda A, b: A.conj_transpose() @ b
#       进化空间: 加权版本, 预处理版本, ...


# ---- 1.1.2 Projection (投影) ----
def project_to_subspace(v: Vec, basis: Mat) -> Vec:
    """将 v 投影到 basis 列空间"""
    coeffs = SLOT["coeff_compute"](basis, v)  # ★ Slot L1: 系数计算
    return m.matvec(basis, coeffs)

# Slot:
#   "coeff_compute": (Mat, Vec) → Vec
#       默认: least squares solve
#       进化: 正则化、截断SVD、迭代求解...


# ---- 1.1.3 Whitening (白化) ----
def whiten(y: Vec, noise_cov: Mat) -> Vec:
    """噪声白化: L⁻¹y, 其中 noise_cov = LLᴴ"""
    L = SLOT["factorize"](noise_cov)         # ★ Slot L1: 分解方法
    return m.solve(L, y)                     # 前代

# Slot:
#   "factorize": Mat → Mat
#       默认: Cholesky
#       进化: LDL, 近似分解, 对角近似...
```

#### 1.2 距离/度量 (Distance/Metric)

```python
# ---- 1.2.1 Symbol Distance (符号距离) ----
def symbol_distance(y_k: cx, R_kk: cx, sym: cx, interference: cx) -> f64:
    """单层残差距离计算"""
    residual = SLOT["residual_compute"](y_k, R_kk, sym, interference)  # ★ Slot
    return SLOT["distance_metric"](residual)                            # ★ Slot

# Slot 规范:
#   "residual_compute": (cx, cx, cx, cx) → cx
#       默认: y_k - R_kk * sym - interference
#   "distance_metric": cx → f64
#       默认: |r|² (平方欧氏)
#       进化: |r|, |r|² + penalty, Mahalanobis, ...


# ---- 1.2.2 Cumulative Metric (累积度量) ----
def cumulative_metric(parent_metric: f64, local_dist: f64) -> f64:
    """从父节点继承的累积代价"""
    return SLOT["accumulate"](parent_metric, local_dist)  # ★ Slot

# Slot:
#   "accumulate": (f64, f64) → f64
#       默认: parent + local
#       进化: weighted sum, max, log-domain add, ...


# ---- 1.2.3 Likelihood (似然计算) ----
def log_likelihood(y: Vec, H: Mat, x: Vec, sigma2: f64) -> f64:
    """对数似然 -‖y - Hx‖²/σ²"""
    residual = v.sub(y, m.matvec(H, x))
    norm_sq = v.norm2sq(residual)
    return SLOT["ll_transform"](norm_sq, sigma2)  # ★ Slot

# Slot:
#   "ll_transform": (f64, f64) → f64
#       默认: -norm_sq / sigma2
#       进化: 加正则项, 修正项, log-domain 变换...
```

#### 1.3 滤波/均衡 (Filtering/Equalization)

```python
# ---- 1.3.1 Generic Linear Equalizer ----
def linear_equalize(H: Mat, y: Vec, sigma2: f64) -> Vec:
    """通用线性均衡器"""
    W = SLOT["weight_matrix"](H, sigma2)  # ★ Slot L1: 权重矩阵构造
    x_hat = m.matvec(W, y)                # 固定: 滤波
    return SLOT["post_process"](x_hat)    # ★ Slot L1: 后处理

# Slot 规范:
#   "weight_matrix": (Mat[m,n], f64) → Mat[n,m]
#       默认 (MMSE): (HᴴH + σ²I)⁻¹Hᴴ
#       变体 (ZF): (HᴴH)⁻¹Hᴴ
#       进化: 迭代近似, 对角loading, 截断SVD...
#   "post_process": Vec → Vec
#       默认: identity
#       进化: slicing to constellation, thresholding, SIC correction...


# ---- 1.3.2 Matched Filter ----
def matched_filter(H: Mat, y: Vec) -> Vec:
    """匹配滤波: Hᴴy"""
    return m.matvec(m.conj_transpose(H), y)

# 无 Slot (纯原子组合)


# ---- 1.3.3 MMSE Filter with Diagonal Loading ----
def mmse_filter_diag(H: Mat, y: Vec, sigma2: f64) -> Vec:
    """MMSE 均衡，对角加载可进化"""
    G = m.gram(H)
    loading = SLOT["loading_strategy"](G, sigma2)  # ★ Slot
    W_inv = m.add(G, loading)
    return m.solve(W_inv, m.matvec(m.conj_transpose(H), y))

# Slot:
#   "loading_strategy": (Mat, f64) → Mat
#       默认: sigma2 * I
#       进化: adaptive loading, per-column loading...
```

#### 1.4 分布操作 (Distribution Ops)

```python
# ---- 1.4.1 Gaussian Moment Matching (高斯矩匹配) ----
def moment_match_gaussian(samples: Vec, weights: Vec) -> (f64, f64):
    """从加权样本中匹配高斯参数"""
    mu = SLOT["mean_estimate"](samples, weights)     # ★ Slot
    var = SLOT["var_estimate"](samples, weights, mu)  # ★ Slot
    return mu, var

# Slot:
#   "mean_estimate": (Vec, Vec) → f64
#       默认: weighted mean
#   "var_estimate": (Vec, Vec, f64) → f64
#       默认: weighted variance


# ---- 1.4.2 KL Projection (KL 投影) ----
def project_to_gaussian(target_logpdf: Callable, mu_init: f64, var_init: f64) -> (f64, f64):
    """将任意分布投影到高斯族 (EP 的核心步骤)"""
    mu, var = mu_init, var_init
    for _ in range(SLOT["n_iters"]):                        # ★ Slot: 迭代次数
        mu, var = SLOT["update_rule"](target_logpdf, mu, var)  # ★ Slot: 更新规则
    return mu, var

# Slot:
#   "update_rule": (Callable, f64, f64) → (f64, f64)
#       默认: natural gradient on KL
#       进化: moment matching, Laplace approx, ...
#   "n_iters": int
#       默认: 5


# ---- 1.4.3 Cavity Computation (腔计算) ----
def cavity_distribution(prior_mu: f64, prior_var: f64,
                        site_mu: f64, site_var: f64) -> (f64, f64):
    """腔分布 = 先验 / 站点近似"""
    cav_var = SLOT["cavity_var"](prior_var, site_var)     # ★ Slot
    cav_mu = SLOT["cavity_mu"](prior_mu, site_mu, prior_var, site_var, cav_var)  # ★ Slot
    return cav_mu, cav_var

# Slot:
#   "cavity_var": (f64, f64) → f64
#       默认: 1/(1/prior_var - 1/site_var) (natural parameter subtraction)
#   "cavity_mu": (f64, f64, f64, f64, f64) → f64
#       默认: cav_var * (prior_mu/prior_var - site_mu/site_var)
```

---

### Level 2：算法模块

这些是**可独立调用的算法子单元**，内部有**多层嵌套 slot**。每个模块可以作为更高层算法的子组件，也可以独立进化。

#### 2.1 树搜索操作 (Tree Search Ops)

```python
# ---- 2.1.1 Node Expansion (节点展开) ----
def expand_node(tree: SearchTree, node: TreeNode,
                R: Mat, y_tilde: Vec, constellation: Vec[cx]) -> list[TreeNode]:
    """展开节点: 创建 |constellation| 个子节点"""
    children = []
    for sym in constellation:
        local_dist = SLOT["local_cost"](                      # ★ Slot L2-A
            y_tilde, R, node.layer, sym, node.partial_symbols
        )
        cum_metric = SLOT["cumulative_cost"](                  # ★ Slot L2-B
            node.cum_dist, local_dist
        )
        child = tree.add_child(node, sym, local_dist, cum_metric)
        children.append(child)
    return children

# Slot 层次:
#   "local_cost" (L2-A): 内部还可以包含 L1 slot
#       默认实现 = symbol_distance (Level 1, 自带 2 个 L1 slot)
#       → 总计 3 层: L2-A → L1.residual_compute → L0 原子
#   "cumulative_cost" (L2-B): 
#       默认实现 = cumulative_metric (Level 1, 自带 1 个 L1 slot)


# ---- 2.1.2 Frontier Scoring (前沿评分) ----
def score_frontier(tree: SearchTree, frontier: list[TreeNode]) -> list[f64]:
    """对所有前沿节点评分"""
    scores = []
    for node in frontier:
        features = SLOT["feature_extract"](tree, node)   # ★ Slot L2: 特征提取
        score = SLOT["score_function"](features)          # ★ Slot L2: 评分函数
        scores.append(score)
    return scores

# Slot:
#   "feature_extract": (SearchTree, TreeNode) → Vec
#       默认: [cum_dist, m_down, m_up, layer, n_siblings]
#       进化: 加入全局树统计量, 父节点特征, ...
#   "score_function": Vec → f64
#       默认: features[0] (cum_dist)
#       进化: 线性组合, 非线性变换, ...


# ---- 2.1.3 Pruning Strategy (剪枝策略) ----
def prune(candidates: list[(TreeNode, f64)], budget: int) -> list[TreeNode]:
    """从候选中选择保留的节点"""
    ranked = SLOT["ranking"](candidates)              # ★ Slot L2: 排序策略
    selected = SLOT["selection"](ranked, budget)      # ★ Slot L2: 选择策略
    return selected

# Slot:
#   "ranking": list[(TreeNode, f64)] → list[(TreeNode, f64)]
#       默认: sort by score ascending
#       进化: multi-criteria ranking, diversity-aware, ...
#   "selection": (list, int) → list
#       默认: take first `budget` elements (greedy)
#       进化: probabilistic selection, radius-based, ...


# ---- 2.1.4 Best-First Search Step (最佳优先搜索步) ----
def best_first_step(tree: SearchTree, pq: PriorityQueue,
                    R: Mat, y_tilde: Vec, constellation: Vec[cx]) -> TreeNode | None:
    """执行一步最佳优先搜索"""
    node = SLOT["node_select"](pq)                          # ★ Slot L2
    children = SLOT["expand"](tree, node, R, y_tilde, constellation)  # ★ Slot L2 → 嵌套 2.1.1
    scores = SLOT["score"](tree, children)                   # ★ Slot L2 → 嵌套 2.1.2
    for child, sc in zip(children, scores):
        pq.push(child, sc)
    return node

# 嵌套 slot 层次:
#   "expand" → expand_node → "local_cost" → symbol_distance → L0 原子
#   3~4 层嵌套！
```

#### 2.2 图/消息传递操作 (Graph/Message Passing Ops)

```python
# ---- 2.2.1 Message Up (自底向上消息) ----
def message_up(tree: SearchTree, node: TreeNode) -> f64:
    """计算一个节点的 m_up 值"""
    if not node.children:
        return SLOT["leaf_value"](node)                   # ★ Slot L2: 叶节点值
    child_vals = [
        SLOT["child_contribution"](child)                  # ★ Slot L2: 子节点贡献
        for child in node.children
    ]
    return SLOT["aggregate_up"](child_vals, node)          # ★ Slot L2: 聚合

# Slot:
#   "leaf_value": TreeNode → f64
#       默认: node.local_dist
#   "child_contribution": TreeNode → f64
#       默认: child.m_up + child.local_dist
#   "aggregate_up": (list[f64], TreeNode) → f64
#       默认: sum(child_vals) / len(child_vals) if len>0 else 0
#       进化: min, max, weighted sum, log-sum-exp, ...


# ---- 2.2.2 Message Down (自顶向下消息) ----
def message_down(node: TreeNode) -> f64:
    """计算一个节点的 m_down 值"""
    if node.parent is None:
        return 0.0
    return SLOT["down_rule"](                             # ★ Slot L2: 下行规则
        node.parent.m_down, node.local_dist, node
    )

# Slot:
#   "down_rule": (f64, f64, TreeNode) → f64
#       默认: parent_m_down + local_dist
#       进化: 加权版, 含 damping, 包含兄弟节点信息, ...


# ---- 2.2.3 Full BP Sweep (完整 BP 扫描) ----
def full_bp_sweep(tree: SearchTree) -> None:
    """执行一轮完整的 BP 消息传递"""
    bfs_order = tree.bfs_order()
    
    # 上行
    for node in reversed(bfs_order):
        node.m_up = SLOT["up_sweep"](tree, node)          # ★ Slot L2 → 嵌套 2.2.1
    
    # 下行
    for node in bfs_order:
        node.m_down = SLOT["down_sweep"](node)             # ★ Slot L2 → 嵌套 2.2.2
    
    # 评分
    for node in tree.frontier():
        node.score = SLOT["belief_compute"](node)          # ★ Slot L2: 置信度计算

# Slot:
#   "belief_compute": TreeNode → f64
#       默认: node.cum_dist + node.m_down + node.m_up
#       进化: 加权组合, 非线性变换, 只用子集...


# ---- 2.2.4 Factor-to-Variable Message (因子→变量消息, 用于 BP on Factor Graph) ----
def factor_to_variable_message(
    factor: Factor, incoming_messages: dict[Variable, Vec],
    target_var: Variable
) -> Vec:
    """因子图 BP 中因子到变量的消息"""
    # 对 target_var 以外的所有变量做边缘化
    excluded = [v for v in factor.variables if v != target_var]
    result = SLOT["marginalize"](                         # ★ Slot L2
        factor, incoming_messages, excluded, target_var
    )
    return SLOT["message_normalize"](result)               # ★ Slot L2

# Slot:
#   "marginalize": (...) → Vec
#       默认: 精确枚举 (指数复杂度)
#       进化: Gaussian approximation, min-sum, ...
#   "message_normalize": Vec → Vec
#       默认: normalize to sum=1
#       进化: log-domain, damped, ...


# ---- 2.2.5 Gaussian BP Message ----
def gaussian_bp_message(
    H_col: Vec, noise_var: f64,
    cavity_mean: f64, cavity_var: f64
) -> (f64, f64):
    """高斯 BP 消息 (用于 GTA 等算法)"""
    msg_var = SLOT["precision_compute"](H_col, noise_var, cavity_var)  # ★ Slot
    msg_mean = SLOT["mean_compute"](H_col, noise_var, cavity_mean, msg_var)  # ★ Slot
    return msg_mean, msg_var

# Slot:
#   "precision_compute": (Vec, f64, f64) → f64
#       默认: 1 / (‖h‖² / σ² + 1/cavity_var)
#   "mean_compute": (Vec, f64, f64, f64) → f64
#       默认: msg_var * (hᴴy/σ² + cavity_mean/cavity_var)
```

#### 2.3 推断操作 (Inference Ops)

```python
# ---- 2.3.1 EP Site Update (EP 站点更新) ----
def ep_site_update(
    cavity_mu: f64, cavity_var: f64,
    factor_fn: Callable, constellation: Vec[cx]
) -> (f64, f64):
    """EP 的单站点更新: cavity → tilted → moment match → new site"""
    # Tilted distribution
    tilted_weights = SLOT["tilted_compute"](                # ★ Slot L2
        cavity_mu, cavity_var, factor_fn, constellation
    )
    # Moment matching
    new_mu, new_var = SLOT["moment_match"](                 # ★ Slot L2 → 可嵌套 1.4.1
        constellation, tilted_weights
    )
    # Site parameter update  
    site_var = SLOT["site_precision_update"](                # ★ Slot L2
        new_var, cavity_var
    )
    site_mu = SLOT["site_mean_update"](                      # ★ Slot L2
        new_mu, new_var, cavity_mu, cavity_var
    )
    return site_mu, site_var

# 4 个 slot, 其中 "moment_match" 可嵌套 Level-1 的 moment_match_gaussian


# ---- 2.3.2 AMP Iteration Step ----
def amp_iteration_step(
    H: Mat, y: Vec, x_hat: Vec, s_hat: Vec, sigma2: f64
) -> (Vec, Vec):
    """AMP 的一步迭代"""
    z = SLOT["residual"](y, H, x_hat)                      # ★ Slot L2
    z_onsager = SLOT["onsager_correct"](z, H, s_hat)        # ★ Slot L2: Onsager 修正
    x_eff = SLOT["effective_obs"](H, z_onsager, x_hat)      # ★ Slot L2
    x_new = SLOT["denoiser"](x_eff, sigma2)                 # ★ Slot L2: 去噪器
    s_new = SLOT["divergence"](x_eff, x_new)                # ★ Slot L2
    return x_new, s_new

# Slot:
#   "residual": (Vec, Mat, Vec) → Vec
#       默认: y - Hx
#   "onsager_correct": (Vec, Mat, Vec) → Vec
#       默认: z + (Nt/Nr) * z_prev (Onsager correction)
#   "denoiser": (Vec, f64) → Vec
#       默认: MMSE denoiser for QAM constellation
#       进化: 不同非线性, 软硬混合, ...


# ---- 2.3.3 SIC Step (连续干扰消除步) ----
def sic_detect_one(
    H: Mat, y: Vec, sigma2: f64,
    detected: list[int], constellation: Vec[cx]
) -> (int, cx, Vec):
    """SIC: 检测一层, 消除干扰"""
    col_idx = SLOT["layer_select"](H, y, sigma2, detected)     # ★ Slot L2: 层选择
    x_est = SLOT["single_detect"](H, y, sigma2, col_idx)       # ★ Slot L2: 单符号检测
    symbol = SLOT["hard_decision"](x_est, constellation)        # ★ Slot L2: 硬判决
    y_cancelled = SLOT["interference_cancel"](y, H, col_idx, symbol)  # ★ Slot L2
    return col_idx, symbol, y_cancelled

# Slot:
#   "layer_select": (...) → int
#       默认: argmin of post-equalization SNR (OSIC ordering)
#       进化: 基于容量的排序, 随机排序, ...
#   "single_detect": (...) → cx
#       默认: MMSE equalization for column col_idx
#   "hard_decision": (cx, Vec[cx]) → cx
#       默认: nearest constellation point
#   "interference_cancel": (Vec, Mat, int, cx) → Vec
#       默认: y - H[:,col_idx] * symbol
```

#### 2.4 迭代策略 (Iterative Strategies)

```python
# ---- 2.4.1 Fixed-Point Iteration ----
def fixed_point_iterate(
    init_state: Any, update_fn: Callable,
    max_iters: int
) -> Any:
    """不动点迭代框架"""
    state = init_state
    for i in range(max_iters):
        old_state = state
        state = SLOT["update"](state, i)                     # ★ Slot L2
        state = SLOT["damping"](old_state, state, i)         # ★ Slot L2: 阻尼
        if SLOT["converged"](old_state, state, i):           # ★ Slot L2: 收敛判断
            break
    return state

# Slot:
#   "update": (State, int) → State
#       默认: update_fn(state)
#   "damping": (State, State, int) → State
#       默认: (1-α)*old + α*new, α=0.5
#       进化: 自适应步长, 动量, ...
#   "converged": (State, State, int) → bool
#       默认: ‖new - old‖ < ε
#       进化: 相对变化, 梯度范数, ...


# ---- 2.4.2 Coordinate Descent Step ----
def coordinate_descent_step(
    state: Vec, objective: Callable, idx: int
) -> Vec:
    """坐标下降: 优化第 idx 个坐标"""
    partial_obj = SLOT["partial_objective"](state, objective, idx)  # ★ Slot
    new_val = SLOT["minimize_1d"](partial_obj)                      # ★ Slot
    state[idx] = new_val
    return state

# Slot:
#   "partial_objective": (Vec, Callable, int) → Callable[f64→f64]
#       默认: fix all coords except idx
#   "minimize_1d": Callable → f64
#       默认: 解析解 (对高斯), 网格搜索 (对离散)
```

---

### Level 3：完整算法

由 Level-2 模块组合而成的端到端检测算法。这些算法**内部有深层嵌套的 slot**——从 L3 一直到 L0，形成多层可进化结构。

每个 Level-3 算法下面用**树形结构**标出所有 slot 及其嵌套关系。标记说明：
- `★ Slot Lk` = 第 k 层的可进化 slot
- `→` = 默认实现引用的子模块
- `[FIXED]` = 不可进化的固定步骤

#### 3.1 LMMSE 检测器

```
lmmse_detector(H, y, σ², constellation) → x_hat
├── [FIXED] G = HᴴH
├── ★ Slot L3-A "regularizer": (Mat, f64) → Mat
│   └── 默认: G + σ²I
│       进化: 对角loading, 自适应正则化, ...
├── [FIXED] rhs = Hᴴy
├── [FIXED] x_soft = solve(G_reg, rhs)
└── ★ Slot L3-B "hard_decision": (Vec[cx], Constellation) → Vec[cx]
    └── 默认: per-element nearest-point slicing
        进化: soft slicing, iterative refinement, ...
```

```python
def lmmse_detector(H, y, sigma2, constellation):
    G = m.gram(H)
    G_reg = SLOT["regularizer"](G, sigma2)      # ★ L3-A
    rhs = m.matvec(m.conj_transpose(H), y)
    x_soft = m.solve(G_reg, rhs)
    x_hat = SLOT["hard_decision"](x_soft, constellation)  # ★ L3-B
    return x_hat
```

**总 slot 数**: 2 (浅层)
**进化层次**: 仅 L3 → L0

---

#### 3.2 OSIC (Ordered SIC) 检测器

```
osic_detector(H, y, σ², constellation) → x_hat
├── ★ Slot L3-A "ordering": (Mat, Vec, f64) → list[int]
│   └── 默认: MMSE-based SNR ordering
│       进化: capacity-based, correlation-based, random, ...
├── Loop over layers (Nt iterations):
│   ├── ★ Slot L3-B "sic_step": → 嵌套 Level-2 SIC Step (2.3.3)
│   │   ├── ★ Slot L2-A "layer_select" — 被 ordering 覆盖
│   │   ├── ★ Slot L2-B "single_detect"
│   │   │   └── 默认: linear_equalize (Level 1) → "weight_matrix" (L1) → L0 原子
│   │   ├── ★ Slot L2-C "hard_decision"
│   │   └── ★ Slot L2-D "interference_cancel"
│   └── [FIXED] deflate H by removing detected column
└── [FIXED] assemble full x_hat
```

**总 slot 数**: 5 (多层嵌套)
**进化层次**: L3 → L2 → L1 → L0

---

#### 3.3 K-Best 检测器

```
kbest_detector(H, y, constellation, K) → x_hat
├── [FIXED] Q, R = qr(H)
├── [FIXED] y_tilde = Qᴴy
├── Loop over layers (Nt-1 down to 0):
│   ├── For each candidate in current_candidates:
│   │   ├── ★ Slot L3-A "expand": → 嵌套 expand_node (2.1.1)
│   │   │   ├── ★ Slot L2-A "local_cost": → symbol_distance (1.2.1)
│   │   │   │   ├── ★ Slot L1-A "residual_compute": (cx,cx,cx,cx) → cx
│   │   │   │   └── ★ Slot L1-B "distance_metric": cx → f64
│   │   │   └── ★ Slot L2-B "cumulative_cost": → cumulative_metric (1.2.2)
│   │   │       └── ★ Slot L1-C "accumulate": (f64,f64) → f64
│   │   └── ★ Slot L3-B "child_score": → 可选的额外评分修正
│   └── ★ Slot L3-C "prune": → 嵌套 prune (2.1.3)
│       ├── ★ Slot L2-C "ranking"
│       └── ★ Slot L2-D "selection"
└── [FIXED] return best candidate
```

**总 slot 数**: 8 (4 层嵌套)
**进化层次**: L3 → L2 → L1 → L0

---

#### 3.4 BP-Stack 检测器 (示例：组合树搜索与消息传递)

> **说明**：BP-Stack 仅作为"如何将树搜索与消息传递两类子结构组合"的**参考示例**，用于展示深层嵌套 slot 的设计模式。它**不是**本项目的主要或唯一目标。实际系统中，`bp_detector`（纯消息传递）和 `stack_decoder`（纯树搜索）作为独立条目加入算法池，进化框架可以自由决定是否/如何组合它们。

```
bp_stack_detector(H, y, σ², constellation, max_nodes, max_bp_iters) → x_hat
├── [FIXED] Q, R = qr(H); y_tilde = Qᴴy
├── [FIXED] 初始化搜索树, 展开根节点
├── Main loop (while nodes < max_nodes):
│   ├── ★ Slot L3-A "node_select": PriorityQueue → TreeNode
│   │   └── 默认: pop min-score
│   │       进化: UCB-style, diversity-aware, ...
│   ├── ★ Slot L3-B "expand": → 嵌套 expand_node (2.1.1)
│   │   ├── ★ Slot L2-A "local_cost": → symbol_distance (1.2.1)
│   │   │   ├── ★ Slot L1-A "residual_compute"
│   │   │   └── ★ Slot L1-B "distance_metric"
│   │   └── ★ Slot L2-B "cumulative_cost"
│   │       └── ★ Slot L1-C "accumulate"
│   ├── ★ Slot L3-C "bp_cycle": → 嵌套 full_bp_sweep (2.2.3)
│   │   ├── ★ Slot L2-C "up_sweep": → message_up (2.2.1)
│   │   │   ├── ★ Slot L2-C1 "leaf_value"
│   │   │   ├── ★ Slot L2-C2 "child_contribution"
│   │   │   └── ★ Slot L2-C3 "aggregate_up"
│   │   ├── ★ Slot L2-D "down_sweep": → message_down (2.2.2)
│   │   │   └── ★ Slot L2-D1 "down_rule"
│   │   ├── ★ Slot L2-E "belief_compute": TreeNode → f64
│   │   └── ★ Slot L2-F "halt_check": (f64, f64) → bool
│   └── [FIXED] 重建优先队列
└── [FIXED] return best complete path
```

**总 slot 数**: 14 (4 层嵌套)
**进化层次**: L3 → L2 → L1 → L0
**与当前 IRGenome 的对应关系**:
  - 当前的 `f_down` ≈ `Slot L2-D1 "down_rule"`
  - 当前的 `f_up` ≈ `Slot L2-C3 "aggregate_up"` + `Slot L2-C2 "child_contribution"`
  - 当前的 `f_belief` ≈ `Slot L2-E "belief_compute"`
  - 当前的 `h_halt` ≈ `Slot L2-F "halt_check"`

---

#### 3.5 EP (Expectation Propagation) 检测器

```
ep_detector(H, y, σ², constellation, max_iters) → x_hat
├── [FIXED] 初始化: site_mu=0, site_var=∞ for all Nt variables
├── ★ Slot L3-A "iterate": → 嵌套 fixed_point_iterate (2.4.1)
│   ├── ★ Slot L2-A "update": 一轮 EP sweep
│   │   └── For each variable i:
│   │       ├── ★ Slot L3-B "cavity_compute": → cavity_distribution (1.4.3)
│   │       │   ├── ★ Slot L1-A "cavity_var"
│   │       │   └── ★ Slot L1-B "cavity_mu"
│   │       ├── ★ Slot L3-C "site_update": → ep_site_update (2.3.1)
│   │       │   ├── ★ Slot L2-B "tilted_compute"
│   │       │   ├── ★ Slot L2-C "moment_match": → 嵌套 1.4.1
│   │       │   │   ├── ★ Slot L1-C "mean_estimate"
│   │       │   │   └── ★ Slot L1-D "var_estimate"
│   │       │   ├── ★ Slot L2-D "site_precision_update"
│   │       │   └── ★ Slot L2-E "site_mean_update"
│   │       └── ★ Slot L3-D "damping_strategy"
│   ├── ★ Slot L2-F "damping" (iteration-level)
│   └── ★ Slot L2-G "converged"
└── ★ Slot L3-E "final_decision": (Vec[cx], Constellation) → Vec[cx]
```

**总 slot 数**: 14 (4 层嵌套)

---

#### 3.6 GTA (Gaussian Tree Approximation) 检测器

```
gta_detector(H, y, σ², constellation) → x_hat
├── ★ Slot L3-A "build_graph": MIMOSystem → WeightedGraph
│   └── 默认: mutual information weights from LMMSE beliefs
│       进化: correlation-based, capacity-based, ...
├── ★ Slot L3-B "find_tree": WeightedGraph → Tree
│   └── 默认: max-weight spanning tree (Chow-Liu)
│       进化: min spanning tree, random tree, ...
├── ★ Slot L3-C "bp_on_tree": → 嵌套 fixed_point_iterate
│   └── Each iteration:
│       ├── ★ Slot L2-A "message_compute": → gaussian_bp_message (2.2.5)
│       │   ├── ★ Slot L1-A "precision_compute"
│       │   └── ★ Slot L1-B "mean_compute"
│       ├── ★ Slot L2-B "schedule": 消息更新顺序
│       └── ★ Slot L2-C "damping"
└── ★ Slot L3-D "final_decision"
```

**总 slot 数**: 9

---

#### 3.7 AMP 检测器

```
amp_detector(H, y, σ², constellation, max_iters) → x_hat
├── [FIXED] 初始化 x=0, s=ones, z=y
├── ★ Slot L3-A "iterate": → 嵌套 fixed_point_iterate (2.4.1)
│   ├── ★ Slot L2-A "update": → amp_iteration_step (2.3.2)
│   │   ├── ★ Slot L2-A1 "residual"
│   │   ├── ★ Slot L2-A2 "onsager_correct"
│   │   ├── ★ Slot L2-A3 "effective_obs"
│   │   ├── ★ Slot L2-A4 "denoiser"
│   │   └── ★ Slot L2-A5 "divergence"
│   ├── ★ Slot L2-B "damping"
│   └── ★ Slot L2-C "converged"
└── ★ Slot L3-B "final_decision"
```

**总 slot 数**: 9

---

### Slot 层次标注系统

#### Slot 数据结构

每个 slot 需要记录：

```python
@dataclass
class SlotDescriptor:
    """描述算法中一个可进化的 slot。"""
    
    # ── 标识 ──
    slot_id: str                              # 全局唯一 ID, e.g. "bp_stack.bp_cycle.up_sweep.aggregate_up"
    short_name: str                           # 短名, e.g. "aggregate_up"
    
    # ── 层次 ──
    level: int                                # 0=原子, 1=复合, 2=模块, 3=算法
    depth: int                                # 在所属算法中的嵌套深度 (0=顶层slot, 1=二级, ...)
    parent_slot_id: str | None                # 父 slot 的 ID (如果嵌套在另一个 slot 内)
    child_slot_ids: list[str]                 # 子 slot 的 ID 列表
    
    # ── 类型签名 ──
    spec: ProgramSpec                         # 输入/输出类型规范
    
    # ── 默认实现 ──
    default_impl: FunctionIR | str            # 默认的 IR 或源码
    default_impl_level: int                   # 默认实现所在的 Level
    
    # ── 进化控制 ──
    mutable: bool = True                      # 是否允许进化
    evolution_weight: float = 1.0             # 进化权重 (越高越容易被选中变异)
    max_complexity: int = 50                  # 此 slot 的最大 IR 深度
    
    # ── 元信息 ──
    description: str = ""                     # 人可读描述
    domain_tags: set[str] = field(default_factory=set)  
    # e.g. {"linear_algebra", "distance", "tree_search", "message_passing"}
```

#### 层次化 Slot 路径

每个 slot 有一个**路径式 ID**，反映其在算法中的嵌套位置。以 BP-Stack（仅作为嵌套层次最深的示例）为例：

```
bp_stack_detector
├── bp_stack.node_select                          depth=0, level=3
├── bp_stack.expand                               depth=0, level=3
│   ├── bp_stack.expand.local_cost                depth=1, level=2
│   │   ├── bp_stack.expand.local_cost.residual   depth=2, level=1
│   │   └── bp_stack.expand.local_cost.distance   depth=2, level=1
│   └── bp_stack.expand.cumulative_cost           depth=1, level=2
│       └── bp_stack.expand.cumulative_cost.accum depth=2, level=1
├── bp_stack.bp_cycle                             depth=0, level=3
│   ├── bp_stack.bp_cycle.up_sweep                depth=1, level=2
│   │   ├── ...leaf_value                         depth=2, level=2
│   │   ├── ...child_contribution                 depth=2, level=2
│   │   └── ...aggregate_up                       depth=2, level=2
│   ├── bp_stack.bp_cycle.down_sweep              depth=1, level=2
│   │   └── ...down_rule                          depth=2, level=2
│   ├── bp_stack.bp_cycle.belief_compute          depth=1, level=2
│   └── bp_stack.bp_cycle.halt_check              depth=1, level=2
└── bp_stack.node_select                          depth=0, level=3
```

#### 多层进化策略

进化不是对所有 slot 一视同仁，而是按层次区分进化策略：

```python
@dataclass
class HierarchicalEvolutionConfig:
    """多层进化策略配置。"""
    
    # ── 各层进化概率 ──
    # 当宏层 PatternMatcher 选择一个 slot 进行变异/嫁接时，
    # 实际操作发生在哪一层？
    level_mutation_probs: dict[int, float] = field(default_factory=lambda: {
        0: 0.05,    # Level 0: 很少直接换原子操作 (通常通过 L1 间接)
        1: 0.30,    # Level 1: 最常见的微调层
        2: 0.40,    # Level 2: 算法模块替换 (主力进化层)
        3: 0.25,    # Level 3: 结构性大改 (嫁接整个子算法)
    })
    
    # ── 深度衰减 ──
    # 嵌套越深的 slot，被选中变异的概率越低
    depth_decay: float = 0.7   # P(mutate at depth d) ∝ depth_decay^d
    
    # ── 各层变异类型分布 ──
    # Level 0: 只能换成另一个 L0 原子, 或常量扰动
    # Level 1: 可以换实现, 也可以改内部 L0 slot
    # Level 2: 可以换整个模块, 嫁接, 或改内部 L1/L0 slot
    # Level 3: 嫁接另一个完整算法的子结构
    
    # ── 跨层嫁接 ──
    # 允许用一个 Level-k 的操作替换 Level-j 的 slot (j≠k)?
    allow_cross_level_graft: bool = True
    # 例如: 把一个 L2 的 "aggregate_up" slot 替换为一个 L1 的简单操作,
    #       或者反过来，把一个 L1 slot 替换为一个包含子 slot 的 L2 模块。
```

#### 多层 MaterializeOps

`materialize()` 需要处理嵌套 slot:

```python
def materialize_hierarchical(
    structural_ir: FunctionIR,
    slot_populations: dict[str, SlotPopulation],
    slot_tree: dict[str, SlotDescriptor],
) -> FunctionIR:
    """递归填充所有层次的 slot。
    
    从最外层 (depth=0) 开始:
      1. 取 slot 的 best_variant (FunctionIR)
      2. 如果 variant 内部还有 AlgSlot (因为它自己是 L2+ 的模块):
         递归填充其子 slot
      3. inline 到 structural_ir
    
    复杂度: O(总 slot 数) 次 inline 操作
    """
    result = deepcopy(structural_ir)
    
    # 按 depth 排序: 先填外层，再填内层
    # (或反过来: 先填叶 slot，再填包含它们的父 slot)
    leaf_first = sorted(
        slot_tree.values(),
        key=lambda s: s.depth,
        reverse=True  # 叶子先
    )
    
    for desc in leaf_first:
        if desc.slot_id not in slot_populations:
            continue
        pop = slot_populations[desc.slot_id]
        variant_ir = pop.best_variant
        # inline variant_ir into the AlgSlot position
        result = inline_slot(result, desc.slot_id, variant_ir)
    
    return result
```

---

### 初始算法池构造

将上述各层操作实例化为 `AlgorithmGenome`：

```python
def build_initial_pool(rng: np.random.Generator) -> list[AlgorithmGenome]:
    """构造多粒度的初始算法池。"""
    pool = []
    
    # ══════════════════════════════════════════════════════
    # Group A: Level-3 完整检测算法 (5-7 个)
    # ══════════════════════════════════════════════════════
    pool.append(make_genome_from_source(
        algo_id="lmmse",
        source=LMMSE_SOURCE,       # regularizer + hard_decision
        level=3, tags={"original", "linear"},
    ))
    pool.append(make_genome_from_source(
        algo_id="zf",
        source=ZF_SOURCE,          # 同 LMMSE 但 σ²=0
        level=3, tags={"original", "linear"},
    ))
    pool.append(make_genome_from_source(
        algo_id="osic_mmse",
        source=OSIC_SOURCE,        # ordering + sic_step
        level=3, tags={"original", "sic"},
    ))
    pool.append(make_genome_from_source(
        algo_id="kbest_16",
        source=KBEST_SOURCE,       # expand + prune (K=16)
        level=3, tags={"original", "tree_search"},
    ))
    # 注意: bp_detector (消息传递) 和 stack_decoder (树搜索) 是独立的算法条目。
    # BP-Stack (两者的融合) 仅作为组合示例，不是本项目的唯一/主要目标。
    # 进化可以自由组合池中的任何算法子结构。
    pool.append(make_genome_from_source(
        algo_id="bp_detector",
        source=BP_DETECTOR_SOURCE,   # 纯消息传递检测器
        level=3, tags={"original", "message_passing"},
    ))
    pool.append(make_genome_from_source(
        algo_id="stack_decoder",
        source=STACK_DECODER_SOURCE, # 纯树搜索检测器
        level=3, tags={"original", "tree_search"},
    ))
    pool.append(make_genome_from_source(
        algo_id="ep_mimo",
        source=EP_SOURCE,          # cavity + site_update + iterate
        level=3, tags={"original", "approximate_inference"},
    ))
    pool.append(make_genome_from_source(
        algo_id="amp_mimo",
        source=AMP_SOURCE,         # residual + onsager + denoiser + iterate
        level=3, tags={"original", "approximate_inference"},
    ))
    
    # ══════════════════════════════════════════════════════
    # Group B: Level-2 算法模块 (8-12 个)
    # 这些可以作为嫁接供体
    # ══════════════════════════════════════════════════════
    
    # 树搜索模块
    pool.append(make_genome_from_source(
        algo_id="mod_expand_node",
        source=EXPAND_NODE_SOURCE,
        level=2, tags={"original", "tree_search"},
    ))
    pool.append(make_genome_from_source(
        algo_id="mod_frontier_score",
        source=FRONTIER_SCORE_SOURCE,
        level=2, tags={"original", "tree_search"},
    ))
    pool.append(make_genome_from_source(
        algo_id="mod_prune_kbest",
        source=PRUNE_KBEST_SOURCE,
        level=2, tags={"original", "tree_search"},
    ))
    pool.append(make_genome_from_source(
        algo_id="mod_best_first_step",
        source=BEST_FIRST_STEP_SOURCE,
        level=2, tags={"original", "tree_search"},
    ))
    
    # 消息传递模块
    pool.append(make_genome_from_source(
        algo_id="mod_bp_sweep",
        source=BP_SWEEP_SOURCE,
        level=2, tags={"original", "message_passing"},
    ))
    pool.append(make_genome_from_source(
        algo_id="mod_message_up",
        source=MESSAGE_UP_SOURCE,
        level=2, tags={"original", "message_passing"},
    ))
    pool.append(make_genome_from_source(
        algo_id="mod_message_down",
        source=MESSAGE_DOWN_SOURCE,
        level=2, tags={"original", "message_passing"},
    ))
    pool.append(make_genome_from_source(
        algo_id="mod_gaussian_bp_msg",
        source=GAUSSIAN_BP_MSG_SOURCE,
        level=2, tags={"original", "message_passing"},
    ))
    
    # 推断模块
    pool.append(make_genome_from_source(
        algo_id="mod_ep_site_update",
        source=EP_SITE_UPDATE_SOURCE,
        level=2, tags={"original", "inference"},
    ))
    pool.append(make_genome_from_source(
        algo_id="mod_amp_step",
        source=AMP_STEP_SOURCE,
        level=2, tags={"original", "inference"},
    ))
    pool.append(make_genome_from_source(
        algo_id="mod_sic_step",
        source=SIC_STEP_SOURCE,
        level=2, tags={"original", "sic"},
    ))
    
    # 迭代策略模块
    pool.append(make_genome_from_source(
        algo_id="mod_fixed_point",
        source=FIXED_POINT_SOURCE,
        level=2, tags={"original", "iterative"},
    ))
    
    # ══════════════════════════════════════════════════════
    # Group C: Level-1 复合操作 (6-10 个)
    # ══════════════════════════════════════════════════════
    pool.append(make_genome_from_source(
        algo_id="comp_regularized_solve",
        source=REGULARIZED_SOLVE_SOURCE,
        level=1, tags={"original", "linear_algebra"},
    ))
    pool.append(make_genome_from_source(
        algo_id="comp_symbol_distance",
        source=SYMBOL_DISTANCE_SOURCE,
        level=1, tags={"original", "distance"},
    ))
    pool.append(make_genome_from_source(
        algo_id="comp_cumulative_metric",
        source=CUMULATIVE_METRIC_SOURCE,
        level=1, tags={"original", "distance"},
    ))
    pool.append(make_genome_from_source(
        algo_id="comp_linear_equalize",
        source=LINEAR_EQUALIZE_SOURCE,
        level=1, tags={"original", "filtering"},
    ))
    pool.append(make_genome_from_source(
        algo_id="comp_matched_filter",
        source=MATCHED_FILTER_SOURCE,
        level=1, tags={"original", "filtering"},
    ))
    pool.append(make_genome_from_source(
        algo_id="comp_moment_match",
        source=MOMENT_MATCH_SOURCE,
        level=1, tags={"original", "distribution"},
    ))
    pool.append(make_genome_from_source(
        algo_id="comp_cavity_dist",
        source=CAVITY_DIST_SOURCE,
        level=1, tags={"original", "distribution"},
    ))
    pool.append(make_genome_from_source(
        algo_id="comp_kl_projection",
        source=KL_PROJECTION_SOURCE,
        level=1, tags={"original", "distribution"},
    ))
    pool.append(make_genome_from_source(
        algo_id="comp_log_likelihood",
        source=LOG_LIKELIHOOD_SOURCE,
        level=1, tags={"original", "distance"},
    ))
    pool.append(make_genome_from_source(
        algo_id="comp_whitening",
        source=WHITENING_SOURCE,
        level=1, tags={"original", "linear_algebra"},
    ))
    
    # ══════════════════════════════════════════════════════
    # Group D: Level-0 原子操作不作为 AlgorithmGenome
    # 它们通过 SlotPopulation 的随机初始化引入进化
    # ══════════════════════════════════════════════════════
    # Level-0 ops 被注册为 "primitive_registry"，
    # 当一个 slot 的类型签名匹配某些 L0 ops 时，
    # 随机初始化会从这些 ops 中采样构造小程序。
    
    return pool  # 总计 ~30 个初始个体


def make_genome_from_source(
    algo_id: str, source: str, level: int, tags: set[str]
) -> AlgorithmGenome:
    """从源码构造 AlgorithmGenome。
    
    步骤:
    1. compile_source_to_ir(source) → FunctionIR
    2. 发现 IR 中的 AlgSlot 占位符
    3. 为每个 slot 推断 SlotDescriptor (层次、类型签名)
    4. 为每个 slot 初始化 SlotPopulation (用默认实现 + 随机变体)
    """
    ir = compile_source_to_ir(source, algo_id)
    slot_tree = discover_slot_hierarchy(ir)  # 返回 dict[str, SlotDescriptor]
    
    populations = {}
    for sid, desc in slot_tree.items():
        variants = [desc.default_impl]  # 默认实现作为 variant #0
        # 加入随机变体
        for _ in range(7):
            v = random_program_for_slot(desc.spec, desc.level, rng)
            variants.append(v)
        populations[sid] = SlotPopulation(
            slot_id=sid,
            spec=desc.spec,
            variants=variants,
            fitness=[float("inf")] * len(variants),
        )
    
    return AlgorithmGenome(
        algo_id=algo_id,
        structural_ir=ir,
        slot_populations=populations,
        constants=np.zeros(0),
        generation=0,
        parent_ids=[],
        graft_history=[],
        tags=tags | {"original"},
        metadata={"level": level, "slot_tree": slot_tree},
    )
```

---

### 嫁接如何在不同层次上运作

#### 示例 1: L1 slot 替换 (微调)

将 K-Best 检测器的 `distance_metric` 从 `|r|²` 替换为 `|r|² + λ·layer_penalty`：

```
Host:  kbest_16
Target Slot: kbest.expand.local_cost.distance  (depth=2, level=1)
Donor: 一个新的 L1 复合操作 (可能来自 EP 的 site precision)
操作: 只替换叶 slot，不改变上层结构
```

#### 示例 2: L2 模块嫁接 (结构调整)

将 BP-Stack 的 `up_sweep` 模块替换为 AMP 的 `denoiser` 模块：

```
Host:  bp_stack
Target Slot: bp_stack.bp_cycle.up_sweep  (depth=1, level=2)
Donor: mod_amp_step 的 "denoiser" 子结构
操作: 替换整个 L2 模块，新模块自带新的 L1 slot
       → 新 slot 通过 SlotPopulation 初始化
```

#### 示例 3: L3 跨算法嫁接 (结构突变)

将 K-Best 的剪枝策略嫁接到 EP 的迭代循环中：

```
Host:  ep_mimo
Target Slot: ep_mimo.iterate.update  (depth=1, level=2)
Donor: kbest_16 的展开+剪枝结构
操作: 完全替换迭代更新步骤，产生一个"EP-KBest 混合体"
       → 大量新 slot 被引入
```

---

### AlgorithmEntry 扩展: 支持层次信息

`AlgorithmEntry` 需要新增字段以支持 PatternMatcher 的层次感知：

```python
@dataclass
class AlgorithmEntry:
    """算法池中的一个条目 (扩展版)。"""
    # ... 原有字段 ...
    algo_id: str
    ir: FunctionIR
    source: str | None
    trace: list[RuntimeEvent] | None
    runtime_values: dict[str, RuntimeValue] | None
    factgraph: FactGraph | None
    fitness: FitnessResult | None
    generation: int
    provenance: dict[str, Any]
    tags: set[str]
    
    # ── 新增: 层次信息 ──
    level: int = 3                                    # 此条目在层次中的级别
    slot_tree: dict[str, SlotDescriptor] | None = None  # 完整 slot 树
    slot_fitness: dict[str, float] | None = None       # 每个 slot 的当前最佳适应度
    compatible_slots: dict[str, list[str]] | None = None
    # compatible_slots: 每个 slot 可以接受哪些其他算法的子结构
    # e.g. {"bp_stack.bp_cycle.belief_compute": ["comp_log_likelihood", "comp_symbol_distance"]}
```

---

### 总结

| 维度 | 旧架构 | 新架构 |
|------|--------|--------|
| 算法池内容 | 仅 4 个固定骨架的小程序 | ~30 个多粒度操作 (L0~L3)，涵盖多种检测器 |
| slot 层次 | 1 层 (4 个固定 role) | 4 层嵌套 (每个算法 8~14 个 slot) |
| 进化粒度 | 仅 slot 内部 ops 变异 | L0 原子替换 → L3 跨算法嫁接 |
| 嫁接供体 | 无 (固定骨架) | 池中任何 L1~L3 条目 |
| 新算法产生 | 不可能 | 跨算法嫁接 → 新结构 + 新 slot |
| 与 DSL 对应 | 无 | 4 层与 DSL 类型层次对齐 |

**关键实现新增**:
- `SlotDescriptor`: slot 层次描述符
- `discover_slot_hierarchy()`: 从 IR 中发现嵌套 slot 树
- `materialize_hierarchical()`: 递归填充多层 slot
- `random_program_for_slot()`: 按 level 生成随机程序
- `HierarchicalEvolutionConfig`: 多层进化策略配置
- `AlgorithmEntry` 扩展: level, slot_tree, compatible_slots

