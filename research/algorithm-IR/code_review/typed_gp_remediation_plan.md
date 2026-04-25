# Typed-GP Slot-Evolution 整改方案 (响应 code_review.md)

**起草日期**: 2026-04-26
**作者**: Orchestrator
**对应审查**: [code_review.md](code_review.md)
**当前 HEAD**: `306d704` (Phase H+3.1 annotation hotfix)
**目标**: 把 `evolution/slot_evolution.py` 从「常数扰动器」升级为完整的 typed Genetic Programming 框架, 同时清理 P0/P1 级别基础设施缺陷.

---

## 0. 范围与不动声色

本方案 **只规划与设计**, 不动一行实现. 实施按 §11 的阶段顺序逐步落地, 每个阶段独立 commit + smoke + 单元测试.

不在本方案范围 (单独立项):
- GNN training-loop 的 reward shaping;
- subprocess evaluator 的 sandbox 隔离;
- Donor Library 的离线 mining.

---

## 1. 现状再确认 (与 code_review §2 一致, 加 H+3.1 修正)

| 维度 | code_review 判定 | H+3.1 后实际 | 本方案处理 |
|---|---|---|---|
| pool valid_errs | 0/91 已绿 | 仍 0/91 | 维持 |
| frontend def-use | 8 failed | 仍 8 failed (test_regression_p0 等未动) | **P0-S0 必修** |
| slot region 覆盖 | 207/221 | 207/221 (kbest/bp/soft_sic 仍空) | **P0-S1 必修** |
| slot 突变只扰常数 | 是 | 是 | **P0-S2/S3/S4 必做** |
| `swap_lines` 引用未定义 `deletable` | 是 | 是 (`evolution/operators.py:178`) | **P1 修一行即可** |
| graft 行为 gate | 仅 validate, 无 behavior gate | 同 | **S2 内同时引入** |
| commit 无 held-out | 是 | 是 | **S3 强制 held-out** |
| 单输出假设 | 是 (`pick_real_exit_value`) | 是 | **S2 引入多输出 contract** |
| skipped pop_keys 衰减 | — | gen-by-gen 增长 (12→39→78→114) | **S1 在 region resolver 层修, 同时清理 phantom pop_keys** |

**新增观察**: 实测 `default_float_consts = 54/221`, 即 75% 的默认 slot 在当前算子下零自由度. 这与 code_review 一致, 是本方案存在的最强理由.

---

## 2. 设计原则

1. **Type-safety first**: 每个 mutation/crossover 必须显式断言保持 slot signature, 失败则丢弃, 不靠 validator 兜底.
2. **Single canonical IR**: 所有 typed GP 操作的对象仍是 `FunctionIR`. 不引入新 IR 层. 不破坏 single-IR 不变量.
3. **Provenance preservation**: 所有新增/移动的 op 必须重新写入 `_provenance.from_slot_id` (H+3.1 已部分做到, S2 系统化).
4. **No silent failure**: 每条算子路径都必须在 `OperatorStats` 里留下 `attempted / succeeded / noop / type_rejected / validate_rejected / probe_rejected / behavior_unchanged` 七个计数.
5. **Held-out before commit**: micro-evolution 内部 fitness ≠ commit 决策 fitness; 后者用单独 seed.
6. **Reuse, not rewrite**: 复用 `algorithm_ir.ir.type_lattice` (已有 tuple/list/dict/数值 lattice 与 `available_ops_for_type`) 和 `algorithm_ir.region.contract.BoundaryContract` (已有 multi-port 字段). 不另起类型系统.

---

## 3. 模块布局

```
research/algorithm-IR/evolution/
  slot_evolution.py            (现存, 重构: 仅作为入口编排; 把算子/合成/选择全部抽走)
  gp/                          (NEW namespace package)
    __init__.py
    contract.py                (SlotContract: 复用 BoundaryContract, 增加 state/effect/shape)
    context.py                 (GPContext: in-scope values, dominance, live-in/out, rng)
    individual.py              (SlotIndividual: ir, fitness, behavior_hash, lineage)
    population.py              (MicroPopulation: μ+λ + tournament + novelty archive)
    operators/                 (one operator per file; each registers in OPERATOR_REGISTRY)
      __init__.py
      base.py                  (Operator ABC + decorator @register_operator)
      mut_const.py             (扩展自 perturb_constants_in_ir)
      mut_point_typed.py       (op-replace within type lattice)
      mut_insert_typed.py      (insert damping/clip/normalize/threshold)
      mut_delete_typed.py      (delete + passthrough rewrite)
      mut_subtree_replace.py   (synthesize new subgraph)
      mut_loop_body.py         (unroll/damp/conv-check loop bodies)
      mut_const_promote.py     (literal -> evolvable const slot)
      mut_primitive_inject.py  (donor primitive library)
      cx_subtree_typed.py      (intra-pop subtree crossover)
      cx_block_typed.py        (loop/branch block crossover)
    synthesis.py               (typed random subgraph generator)
    novelty.py                 (kNN behavior signature archive)
    fitness.py                 (train/val/heldout split + complexity penalty)
    lineage.py                 (MutationRecord + diff dump)
    region_resolver.py         (provenance OR slot-op OR explicit binding)
  operators.py                 (legacy; 仅修 swap_lines bug, 后续逐步迁出)

tests/unit/
  test_gp_contract.py
  test_gp_context.py
  test_gp_operators_const.py
  test_gp_operators_point.py
  test_gp_operators_insert.py
  test_gp_operators_delete.py
  test_gp_operators_subtree.py
  test_gp_operators_crossover.py
  test_gp_population.py
  test_gp_region_resolver.py
  test_gp_synthesis.py

tests/integration/
  test_slot_typed_gp_smoke.py
  test_slot_coverage_kbest.py
  test_slot_coverage_bp.py
  test_slot_coverage_amp.py
```

合计 ≈ 14 个新文件 + 2 个修改文件. 不删除现有文件, 只在 `slot_evolution.py` 内部把 `step_slot_population` 改写成调用 `gp.population.MicroPopulation.evolve()`.

---

## 4. 数据模型

### 4.1 `gp.contract.SlotContract`

```python
@dataclass(frozen=True)
class TypedPort:
    name: str            # 形参名, 用于源码可读
    type: str            # algorithm_ir.ir.type_lattice 中的字符串类型
    shape: tuple[int|None, ...] | None  # 已知则记下, 否则 None
    role: Literal["data", "state", "control"] = "data"

@dataclass(frozen=True)
class SlotContract:
    slot_key: str                       # e.g. "kbest.expand"
    short_name: str                     # e.g. "expand"
    input_ports: tuple[TypedPort, ...]
    output_ports: tuple[TypedPort, ...] # 多输出强制走 tuple<...>
    state_ports: tuple[TypedPort, ...]  # carry-in / carry-out 显式声明
    effects: frozenset[str]             # {"pure"} / {"mutates_state"} / {"random"}
    complexity_cap: int                 # 最大允许 op 数
    constants_budget: int               # 最大允许新增 const op 数
```

**构建途径**:
1. `lmmse/osic/stack/...` 已有 `SlotSpec` (在 `evolution/skeleton_registry.py`) — 直接读出 input/output 类型签名.
2. 缺失 shape 的字段允许 `None`; 后续 §5.3 通过 probe-driven inference 补全.

### 4.2 `gp.individual.SlotIndividual`

```python
@dataclass
class SlotIndividual:
    ir: FunctionIR
    contract: SlotContract              # 反向引用, 不深拷贝
    fitness_train: float = inf          # micro 评估
    fitness_val:   float = inf          # heldout, 仅 commit 前算
    novelty:       float = 0.0
    complexity:    int = 0
    source_hash:   str = ""             # sha1(emit_python_source(ir))
    behavior_hash: str = ""             # 见 §6.2
    lineage:       list[MutationRecord] = field(default_factory=list)
    operator_origin: str = "seed"       # 谁生的
    parents: tuple[str, ...] = ()       # 父代 source_hash
```

### 4.3 `gp.lineage.MutationRecord`

```python
@dataclass(frozen=True)
class MutationRecord:
    operator: str
    seed: int                           # rng 子种子, 可复现
    parent_hash: str
    parent2_hash: str | None
    diff_summary: str                   # human-readable, 例如 "replaced op_17 binary Add->Sub"
    accepted: bool                      # 是否进入 population (vs. 被 type/validate gate 拒)
    rejection_reason: str | None
```

---

## 5. Region Resolver (S1)

替换 `slot_evolution.map_pop_key_to_from_slot_ids`. 三段式 resolver, 命中即返回; 全部 miss 才报告 `skip_no_sids`.

### 5.1 算法

```
def resolve_slot_region(genome, slot_key) -> SlotRegion | None:
    # Tier 1: 显式 binding (新建于 ir_pool / fii / slot_rediscovery)
    if slot_key in genome.metadata.get("slot_bindings", {}):
        return _build_region_from_binding(genome, ...)

    # Tier 2: provenance.from_slot_id (现有 H+3 路径)
    sids = _provenance_match(genome.ir, slot_key)
    if sids:
        return collect_slot_region(genome.ir, sids)

    # Tier 3: legacy `slot` op (kbest/bp/soft_sic 等仍带 SlotOp 的 genome)
    slot_ops = _find_slot_ops(genome.ir, slot_key.split('.')[-1])
    if slot_ops:
        return _wrap_slot_op_as_region(slot_ops)

    return None
```

### 5.2 显式 binding 写入位置

在以下三处统一调用 `_register_slot_binding(genome, slot_key, sid_set)`:

| 位置 | 时机 |
|---|---|
| `ir_pool.build_ir_pool` (admission) | 每个 detector spec 初始化 SlotPopulation 时 |
| `fii.materialize` (inline 完成后) | provenance 标注完成的最后一步 |
| `algorithm_engine._build_child_from_graft` (post-graft) | `maybe_rediscover_slots` 之后 |

binding 字段:
```python
genome.metadata["slot_bindings"]: dict[str, SlotBinding]
SlotBinding(slot_key, from_slot_ids: frozenset[str], call_site_op_ids: frozenset[str])
```

### 5.3 接受标准 (§9.1 测试 1)

```
test_all_slot_populations_have_evolvable_region:
    for genome in build_ir_pool():
        for slot_key in genome.slot_populations:
            region = resolve_slot_region(genome, slot_key)
            assert region is not None, f"slot {slot_key} unresolvable on {genome.algo_id}"
```

实施完成后必须 221/221 全过, 并且 `kbest.expand`, `kbest.prune`, `bp.bp_sweep`, `bp.final_decision`, `soft_sic.*`, `turbo_linear.*`, `particle_filter.*`, `importance_sampling.*` **必须** 在覆盖之中.

### 5.4 Phantom pop_keys 清理

新增 `_prune_phantom_pops(genome)`: 在每次 `_build_child_from_graft` 末尾调用, 删除任何 `resolve_slot_region == None` 的 pop_key. 这吸收了 H+3.1 残留的 `skip_no_sids` 增长.

---

## 6. Typed GP 算子 (S2)

每个算子满足同一接口:

```python
class Operator(Protocol):
    name: str
    def propose(self, ctx: GPContext, parent: SlotIndividual,
                parent2: SlotIndividual | None = None) -> OperatorResult: ...

@dataclass
class OperatorResult:
    child_ir: FunctionIR | None         # None 表示 type/scope rejected
    diff_summary: str
    rejection_reason: str | None
```

注册由 `@register_operator(weight=...)` 控制, 默认权重见 §7.

### 6.1 `mut_const`

扩展自 `perturb_constants_in_ir`, 新增:

| 子类型 | 适用 literal type | 公式 |
|---|---|---|
| float_mul | float | `v * (1 + N(0, σ))` |
| float_add | float | `v + N(0, σ * |v|)` |
| float_log | float, v>0 | `exp(log v + N(0, σ))` |
| int_walk | int | `v + Geom(p)` (符号随机) |
| bool_flip | bool | `not v` |
| categorical | int (∈ enum) | uniform sample from enum minus current |
| complex | complex | 实部/虚部各自 float_mul |
| vector_broadcast | array(float) | 整体乘以 `(1 + N(0,σ))` |

接受标准: 必须改 `op.attrs["literal"]`, 否则计 `noop_mutation`.

### 6.2 `mut_point_typed`

只允许同 signature 替换. 替换表 (示例, 完整表写在 `gp/operators/mut_point_typed.py`):

```python
SAME_SIG_GROUPS = [
    {"add", "sub"},
    {"mul", "safe_div"},
    {"min", "max", "smooth_min", "smooth_max"},
    {"abs", "abs2"},                    # in (real|complex) -> real
    {"norm_l1", "norm_l2", "norm_inf"},
    {"softmax", "normalized_exp"},
    {"argmax", "argmin"},
    {"clip", "smooth_clip"},
    ...
]
```

实施: 对选中 op, 在所有同组算子里去掉自身, `rng.choice` 替换. 保持 input ids 不变. 用 `algorithm_ir.ir.type_lattice.combine_binary_type` / `combine_unary_type` 复算输出类型, 类型不兼容则 `type_rejected`.

### 6.3 `mut_insert_typed`

在选中的 use 之前插入新 op:

```
原: u = ... ; v = f(u) ; w = g(v)
新: u = ... ; v = f(u) ; v' = inject(v) ; w = g(v')
```

可注入算子按 v 的 type 查询:
```python
INSERT_LIBRARY: dict[str, list[InsertTemplate]] = {
    "vec_f":  [damping(α), clip(lo, hi), normalize_l2(), residual_add(δ), soft_threshold(τ)],
    "vec_cx": [damping(α), conjugate_pass(), phase_normalize()],
    "mat_f":  [add_jitter(γ * I), symmetrize(), spectral_clip(σ_max)],
    "float":  [clip(lo, hi), abs_clip(τ)],
    ...
}
```

每个 InsertTemplate 自带签名验证: `output_type == input_type`. 插入后用 `rebuild_def_use` 修复, 用 `validate_function_ir` 拦截.

### 6.4 `mut_delete_typed`

只对 `out-degree == 1` 的 op 删除, 用其唯一 use 的 input passthrough 替代; 或在 in-scope values 中找 type 兼容者直接 rewrite.

```
原: a -> f -> v -> g
删: a -> g (若 type(a) <: type(v))
否: 寻找 in-scope `a'` with type(a') <: type(v), rewrite g 的输入
```

类型/作用域不允许则 `type_rejected`.

### 6.5 `mut_subtree_replace`

1. 选 root value `v` (不能是 region 的 final exit, 否则改的是 contract 而非 body).
2. 计算 backward slice → 子 DAG `S`.
3. 调 `gp.synthesis.random_subgraph(target_type=type(v), in_scope=ctx.live_in_at(v), max_depth=4)` 生成 `S'`.
4. 替换 `S` 为 `S'`, 重连 `v` 的 uses.

`gp.synthesis.random_subgraph` 用 `available_ops_for_type` + 类型 lattice 自顶向下生成, 严格保证输出 type matches.

### 6.6 `cx_subtree_typed`

两个 individual `A, B`:
1. 在 `A` 中随机选 value `v_A` (type `T`).
2. 在 `B` 的同 contract DAG 中找出 type `T` 的 value `v_B`.
3. 把 `B` 中 `v_B` 的 backward slice clone 到 `A`, 替换 `v_A`.
4. 重映射 `B`-only 的 input ports 到 `A.live_in_at(v_A)` (按 type 一一对齐, 不能对齐则放弃).

### 6.7 `cx_block_typed`

针对 loop body / branch block: 把 `Block` 整体替换. 要求两个 block 的 live-in/live-out signature 完全相同. 不相同则放弃.

### 6.8 `mut_loop_body`

5 个子动作:
- unroll-1: 把循环体 inline 一次, 减一次迭代.
- damping-injection: 在 loop carry-out 上加 `(1-α)*old + α*new`.
- conv-check: 加 early-exit 条件 (residual norm 比阈值小).
- update-replace: 对 loop carry 的 update 表达式做 `mut_subtree_replace`.
- residual-connect: `new = update + scale * old`.

### 6.9 `mut_const_promote`

把某个 literal `c` 提升为新的 const-slot:
```
原:  v = mul(x, 0.5)
后:  d = const_slot_damping()  # 新建 SlotPopulation entry
     v = mul(x, d)
```
新建 slot 进入 `genome.slot_populations`, 默认值为原 literal. 这增加了未来的搜索维度.

### 6.10 `mut_primitive_inject`

从 donor library 注入预编译的 typed primitive (linear_solve, projection, hard_decision, soft_decision, message_update, tree_metric_update). Library 由 `evolution/donor_library.py` 维护, 每个 entry 是一段 valid `FunctionIR` + signature.

---

## 7. MicroPopulation (S3)

```python
class MicroPopulation:
    slot_key: str
    contract: SlotContract
    individuals: list[SlotIndividual]    # μ
    archive: NoveltyArchive              # 行为签名 kNN
    operator_stats: dict[str, OperatorStats]
    rng: np.random.Generator

    def step(self, λ: int) -> StepReport:
        # 1. 抽 λ 个 parent (tournament k=4, by combined score)
        # 2. 对每个 parent 选 operator (按权重 + UCB1 success-rate adjust)
        # 3. 调 operator.propose -> OperatorResult
        # 4. type/validate/probe gate
        # 5. 计算 source_hash, behavior_hash, novelty
        # 6. μ+λ 排序, 截到 μ
    def evolve(self, generations: int) -> MicroEvolutionResult: ...
```

**默认参数** (与 code_review §7 对齐):
```
μ = 32
λ = 32
elite_keep = 4
tournament_k = 4
generations_per_macro = 3
mutation_rate = 0.65
crossover_rate = 0.30
const_promote_rate = 0.05
novelty_weight = 0.05
complexity_weight = 0.001
```

**算子权重** (initial; UCB1 在线调):
```
mut_const                  0.20
mut_point_typed            0.15
mut_insert_typed           0.15
mut_delete_typed           0.10
mut_subtree_replace        0.15
mut_loop_body              0.05
mut_const_promote          0.05
mut_primitive_inject       0.05
cx_subtree_typed           0.07
cx_block_typed             0.03
```

---

## 8. Fitness / 行为签名 / Gates (S3 同期)

### 8.1 训练 / 验证 / held-out 切分

每个 contract 关联三组 probe seed:
```
SEEDS_TRAIN  = list of (Nr, Nt, mod, snr_db, seed) tuples       — μ-eval 用
SEEDS_VAL    = list, 不重叠                                       — 排序用
SEEDS_HOLDOUT = list, commit 前一次性使用                          — gate
```

数量初值: `|TRAIN|=2`, `|VAL|=2`, `|HOLDOUT|=4`. SNR 由 macro-engine 当前 SNR 决定.

### 8.2 行为签名 `behavior_hash`

```
probe = fixed_inputs_for_contract(contract)   # 同一 contract 复用
out = run_slot(individual.ir, probe)          # subprocess fallback in-process
behavior_hash = sha1(numpy_canonicalize(out))[:12]
novelty_signature = float[k] = flatten(out)[:k]    # k=64 默认
```

### 8.3 Gates (硬条件, 任一不过即 reject)

每个 OperatorResult 必须依次通过:

1. `child_ir is not None`
2. `validate_function_ir(child_ir) == []`
3. `_source_compiles_with_resolved_names(emit_python_source(child_ir))` (现有)
4. probe run 不 raise
5. probe 输出 type/shape 与 contract.output_ports 一致
6. `source_hash != parent.source_hash` (否则 noop)
7. `behavior_hash != parent.behavior_hash` (否则 noop_behavior)
8. `complexity <= contract.complexity_cap`

通过的 child 才进 μ+λ 池. 不通过的全部进 `OperatorStats` 计数, 不删除.

### 8.4 Commit gate

`commit_best_variants_to_ir` 改为:
```
candidate = pop.individuals[best_by_val]
gate:
  candidate.fitness_val + ε < current.fitness_val
  AND eval_holdout(candidate) + ε < eval_holdout(current)
  AND candidate.complexity <= contract.complexity_cap
```
不过则保留 current, 不写 `genome.ir`.

---

## 9. Telemetry 升级 (覆盖 code_review P0-4)

`SlotMicroStats` 字段重排:

```python
@dataclass
class SlotMicroStats:
    slot_pop_key: str
    n_attempted: int        = 0     # operator.propose 调用数
    n_type_rejected: int    = 0     # gate 1/5 失败
    n_validate_rejected: int = 0    # gate 2 失败
    n_compile_rejected: int  = 0    # gate 3 失败
    n_probe_rejected: int    = 0    # gate 4 失败
    n_noop_source: int       = 0    # gate 6 失败
    n_noop_behavior: int     = 0    # gate 7 失败
    n_complexity_rejected: int = 0  # gate 8 失败
    n_accepted: int          = 0    # 进入 μ+λ
    n_evaluated: int         = 0    # 真正算了 SER
    n_improved_train: int    = 0
    n_improved_val: int      = 0
    n_committed: int         = 0    # commit gate 通过
    skipped_no_sids: int     = 0
    skipped_no_variants: int = 0
    operator_breakdown: dict[str, OperatorStats] = field(default_factory=dict)
```

train_gnn 日志行随之扩展:
```
slot-evo: att=N typ=N val=N cmp=N prb=N noopS=N noopB=N cap=N acc=N eval=N
         imp_train=N imp_val=N committed=N | skip_no_sid=N skip_no_var=N
```

并按算子打印 top-3 by `acc/att`:
```
slot-evo ops: mut_subtree_replace 18/120=15% | mut_insert_typed 22/100=22% | ...
```

---

## 10. P0/P1 基础设施修复 (与 typed GP 并行)

### 10.1 `swap_lines` bug (P1, 立即可修)

文件: `evolution/operators.py:178`
变更: 删除以下两行
```python
                idx = rng.choice(deletable)
                lines.pop(idx)
```
swap 分支应该只 swap, 不 pop. 加 unit test `test_swap_lines_does_not_raise_nameerror`.

### 10.2 frontend def-use (P0-1, S0 必修)

`algorithm_ir/frontend/ir_builder.py`:
1. 移除 `_compile_while` 和 `_compile_for` 中所有显式 `target.use_ops.append(...)` 行 (审计后定位).
2. 在 `IRBuilder.build()` 末尾统一调用 `algorithm_ir.ir.utils.rebuild_def_use(func_ir)`.
3. 加单元测试 `test_frontend_loop_def_use_consistent`.
4. 跑 `tests/unit/test_frontend.py tests/unit/test_regression_p0.py tests/integration/test_grafting_demo.py` 必须全绿.

### 10.3 graft 行为 gate (P0-5, S2 内顺手补)

`evolution/algorithm_engine._build_child_from_graft` 末尾, 在 `return child` 前加:
```python
if not _passes_behavior_gate(child, host_genome, evaluator):
    self._dispatch_case_counts["graft_behavior_rejected"] += 1
    return None
```
`_passes_behavior_gate` 检查: 编译通过 + probe 不抛 + 输出 hash 与 host 不同.

---

## 11. 阶段化实施 (commit 单元)

| 阶段 | 文件改动 | 测试 gate | 预期 LOC |
|---|---|---|---|
| **S0** frontend def-use | `ir_builder.py` + util | `test_frontend.py` `test_regression_p0.py` `test_grafting_demo.py` 全绿 | ~80 |
| **S0.1** swap_lines hotfix | `operators.py` (-2 lines) + 1 test | 新 test 通过 | ~20 |
| **S1** region resolver | `gp/region_resolver.py` + `_register_slot_binding` 三处 + `_prune_phantom_pops` | `test_gp_region_resolver.py` (221/221) | ~250 |
| **S2.1** type lattice 强化 | `algorithm_ir/ir/type_lattice.py` 补 InsertTemplate signature | `test_gp_contract.py` | ~150 |
| **S2.2** OperatorResult + Gates | `gp/operators/base.py` `gp/individual.py` `gp/lineage.py` | `test_gp_contract.py` `test_gp_context.py` | ~300 |
| **S2.3** 三个最简算子 | `mut_const.py` (扩展) `mut_point_typed.py` `mut_insert_typed.py` | `test_gp_operators_const/point/insert.py` 100 次随机全过 | ~400 |
| **S2.4** 删除 + 子树 + 合成 | `mut_delete_typed.py` `mut_subtree_replace.py` `synthesis.py` | `test_gp_operators_delete/subtree.py` `test_gp_synthesis.py` | ~500 |
| **S2.5** crossover + loop_body + const_promote + primitive_inject | 4 文件 + library | `test_gp_operators_crossover.py` 等 | ~600 |
| **S3** MicroPopulation | `gp/population.py` `gp/novelty.py` `gp/fitness.py` | `test_gp_population.py` synthetic convergence | ~400 |
| **S3.1** slot_evolution 接线 | `slot_evolution.py` 重写 step (~150 LOC delta) `algorithm_engine._micro_evolve` 改调用 | `test_slot_evolution.py` 不回归 | ~150 |
| **S3.2** train_gnn telemetry | `train_gnn.py` 日志行 | 新 log 字段出现, 数值 sanity | ~30 |
| **S4** 真实 detector smoke | `tests/integration/test_slot_typed_gp_smoke.py` 等 4 个 | 4 个集成测试通过 | ~300 |
| **S5** health gate 脚本 | `code_review/preflight_health_gate.py` (在 train_gnn 启动前调) | gate fail 阻断长跑 | ~100 |

每阶段独立 commit. **S0+S0.1 必须先做完** — typed GP 的所有合成算子都依赖 frontend 编译正确.

---

## 12. 验收标准 (与 code_review §12 对齐)

完成 S0..S5 后, 必须满足:

```
[basic]
  test_frontend.py                pass 100%
  test_regression_p0.py           pass 100%
  test_grafting_demo.py           pass 100%
  pool valid_errs                 0 / 91
  swap_lines NameError            no longer raises

[coverage]
  resolve_slot_region(g, k)       returns non-None for 221/221 (g, k) pairs
  kbest.expand/prune              ∈ resolved set
  bp.bp_sweep/final_decision      ∈ resolved set
  amp.denoise                     ∈ resolved set
  soft_sic.*                      ∈ resolved set

[operators]
  每个 typed operator:            100/100 随机调用全过 type+validate+compile+probe
  no-op rate (source unchanged)   < 30%
  behavior-change child rate      > 10%
  type_rejected 率                < 50% (高于此说明 contract/lattice 描述不准)

[micro-evolution]
  synthetic clip(αx+β)           20 gens 内 80% case 收敛到 |Δ| < 0.01
  real lmmse.regularizer          gen 1 的 best_val 必须严格优于 baseline
  real kbest.expand               至少有 source_changed + behavior_changed child

[commit]
  commit_best_variants_to_ir      要求 holdout 严格优于 current 才写
  swap-to-position-0              保留, 但 only-on-success
  从 H+3.1 起的 phantom pop       已被 _prune_phantom_pops 清理为 0
```

未完全满足前, 不应在 paper / 报告中宣称 "Algorithm-IR 实现了 typed GP slot evolution".

---

## 13. 风险与回退

| 风险 | 触发 | 回退 |
|---|---|---|
| typed lattice 与现有 op 类型推断冲突 | S2.1 测试爆 | 在 `combine_*` 提供 `strict=False` 兜底; 类型未知时归 `TYPE_TOP`, 算子降级到只允许同 op 替换 |
| `random_subgraph` 生成无效 IR 比例 > 80% | S2.4 测试 | depth cap 调小, 黑名单一些必产 NaN 的 op (eg log on signed input) |
| MicroPopulation 评估时间膨胀 | S3 smoke | μ↓16, λ↓16, generations_per_macro↓1, 并以 `--micro-budget-sec` CLI 兜底 |
| held-out 评估抖动导致 commit 反复 | S3.1 smoke | held-out trial 数 ×2, 加 `+ε` margin 0.005 |
| frontend 修复破坏现有训练 | S0 完成后 | 跑 H+3.1 5-gen smoke, SER 不应回归 > 0.005 |

每阶段在合并前必须跑前一阶段的 smoke, 严禁累积式回归.

---

## 14. 与 H+3.1 的衔接

H+3.1 已经修了 "annotation 丢失" 这个最阻塞的问题. 本方案在此基础上:

- **保留** H+3.1 的 `apply_slot_variant` 末尾 re-annotate 逻辑 — 是 typed GP 的必要前置.
- **保留** `_source_compiles_with_resolved_names` AST gate — 升级为 §8.3 gate 3.
- **替换** `step_slot_population` 内部的 `perturb_constants_in_ir` 单一算子调用为 `MicroPopulation.evolve()`.
- **保留** `commit_best_variants_to_ir` 的 swap-to-position-0, 但加 §8.4 holdout gate.
- **删除** `_micro_evolve_legacy` (Phase G 残留死代码), 同时清理 `materialize_with_override` 调用点.

---

## 15. 立即下一步

按用户 "不要停止" 的方针, 在等待用户对本方案 sign-off 的同时, 从最低风险且必要前置的 **S0.1 (swap_lines hotfix)** 开始落地. S0 (frontend def-use) 因涉及 ir_builder 内部, 单独立 PR. typed GP 主体 (S2..S3) 在 S0/S1 通过后启动.

落地顺序: **S0.1 → S0 → S1 → S2.1 → S2.2 → S2.3 → S2.4 → S2.5 → S3 → S3.1 → S3.2 → S4 → S5**.
