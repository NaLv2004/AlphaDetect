# Evolution 模块实现原理（Algorithm-IR 版）

本文面向 `research/algorithm-IR/evolution/` 目录，解释三个核心问题：

1. `evolution` 框架整体是如何工作的。
2. 它和 `algorithm_ir` 的关系是什么（为什么必须绑定 IR）。
3. 进化引擎 `EvolutionEngine` 的接口与内部机制细节。

---

## 1. 模块定位：这是“IR 上的遗传编程引擎”

`evolution` 不是传统“字符串/AST 级”GP，而是基于 `FunctionIR` 的进化框架：

- 个体基因：`IRGenome`（多段 `FunctionIR` + 常量向量）
- 搜索空间：由 `SkeletonRegistry` 中的 `ProgramSpec` 定义函数签名
- 变异/交叉：直接作用在 `FunctionIR` 的 op/attr 层
- 评估：通过抽象接口 `FitnessEvaluator` 外挂领域任务
- 运行：`EvolutionEngine` 负责群体管理、选择、繁殖、重启、历史记录

一句话：`evolution` 负责“怎么搜”，`applications/*` 负责“搜什么、按什么标准打分”。

---

## 2. 目录内各文件职责

| 文件 | 作用 |
|---|---|
| `config.py` | 定义 `EvolutionConfig`（种群规模、交叉率、停滞阈值等超参数） |
| `fitness.py` | 定义 `FitnessResult` 与 `FitnessEvaluator` 抽象基类 |
| `genome.py` | 定义 `IRGenome`，提供 IR↔源码/可调用/C++ opcodes 的桥接 |
| `skeleton_registry.py` | 定义 `ProgramSpec`/`SkeletonSpec`，并做程序签名与结构校验 |
| `random_program.py` | 基于 `ProgramSpec` 随机生成可编译的 Python 函数，再编译成 IR |
| `operators.py` | IR 级变异与交叉（`mutate_ir`/`crossover_ir` + genome 级封装） |
| `slot_discovery.py` | 从完整 `FunctionIR` 自动发现可进化槽位（region/contract） |
| `engine.py` | 主进化循环：初始化、选择、繁殖、生存者选择、停滞重启、HoF |
| `__init__.py` | 包标识 |

---

## 3. 与 `algorithm_ir` 的关系（核心）

`evolution` 和 `algorithm_ir` 不是“调用关系这么简单”，而是“共用同一中间表示”的关系。

### 3.1 IR 是共同语言

- `algorithm_ir.ir.model.FunctionIR` 是进化个体程序的真实载体。
- `evolution` 不持有 Python AST，也不直接操作源码字符串作为主状态。
- 一切核心操作（突变、交叉、哈希、多样性）都以 `FunctionIR.ops/values/blocks` 为基础。

### 3.2 编译前端来自 `algorithm_ir.frontend`

- `random_program.random_ir_program()` 会先拼出随机 Python 函数字符串，再调用 `compile_source_to_ir()` 得到 `FunctionIR`。
- `IRGenome.deserialize()` 反序列化时，也通过 `compile_source_to_ir()` 把源码恢复成 IR。
- `operators._mutate_via_recompile()`（插入/删除突变）同样走“源码编辑→重新编译到 IR”的路径。

### 3.3 代码再生后端来自 `algorithm_ir.regeneration`

- `IRGenome.to_source()` -> `emit_python_source(func_ir)`
- `IRGenome.to_cpp_ops()` -> `emit_cpp_ops(func_ir)`
- `IRGenome.to_callable()` -> `emit_python_source` 后 `compile+exec` 缓存成 Python 函数

这使得同一份个体既能在 Python 中解释运行，也能转为 C++ 栈机 opcode 走高性能评估（如 MIMO BP 的 DLL）。

### 3.4 自动槽位发现绑定 `algorithm_ir.region/runtime`

`slot_discovery.py` 使用：

- `region.slicer` 做前/后向切片
- `region.selector.define_rewrite_region` 定义 rewrite 区域
- `region.contract.infer_boundary_contract` 推导边界契约
- `runtime.interpreter.execute_ir`（可选）做动态轨迹增强

这部分对应“自动从完整算法中挖可进化子程序”的能力。

---

## 4. 整体执行流程（从规范到最优个体）

```text
ProgramSpec/SkeletonSpec
        ↓ register
SkeletonRegistry
        ↓ (roles + 签名约束)
EvolutionEngine.init_population()
        ↓
random_ir_program(spec) -> compile_source_to_ir() -> FunctionIR
        ↓
IRGenome(programs + constants)
        ↓
FitnessEvaluator.evaluate(_batch)
        ↓
EvolutionEngine.run():
  tournament selection
  + crossover_genome / mutate_genome
  + parent/offspring 合并排序
  + niching 去重
  + stagnation/hard-restart
        ↓
best_ever + hall_of_fame + history
```

---

## 5. 核心数据结构

## 5.1 EvolutionConfig（`config.py`）

主要字段（按功能分组）：

- 种群规模：`population_size`、`n_generations`
- 选择：`tournament_size`、`elite_count`
- 变异/交叉：`mutation_rate`、`crossover_rate`、`constant_mutate_sigma`
- 停滞控制：`stagnation_threshold`、`hard_restart_after`
- 多样性：`hall_of_fame_size`、`niche_radius`
- 程序结构：`program_roles`、`n_constants`、`constant_range`
- 评估后端偏好：`use_cpp`
- 复合指标权重：`metric_weights`

注意：

- `to_dict()/from_dict()` 负责 JSON 友好序列化（如 tuple->list）。
- 当前 `engine.py` 实际用到了大部分字段，但 `stagnation_threshold` 与 `metric_weights` 本身未被引擎内部直接消费（权重通常在 evaluator 里设置）。

## 5.2 ProgramSpec / SkeletonSpec / SkeletonRegistry

`ProgramSpec` 定义单个可进化函数槽位：

- `name`: 角色名（如 `f_down`）
- `param_names` / `param_types`: 参数签名
- `return_type`: 返回类型
- `constraints`: 结构约束（如 `max_depth`）

`SkeletonSpec` 定义一个骨架：

- `skeleton_id`
- `program_specs`（显式槽位列表）
- `host_ir + mode="auto_discover"`（自动发现模式预留）

`SkeletonRegistry` 提供：

- `register()` 注册骨架
- `roles` 列出所有角色
- `validate_program(role, func_ir)` 返回违规列表
- `validate_genome_programs(programs)` 批量验证

校验逻辑重点：

- 参数个数/名称
- 返回类型（尤其 bool）
- 参数是否参与返回值依赖链（backward slice）
- `max_depth` 约束

注意：当前 `EvolutionEngine` 默认不会自动调用 `validate_program`，若你需要强约束，应在 evaluator 或初始化阶段主动校验。

## 5.3 IRGenome（`genome.py`）

一个 genome = 多角色程序 + 常量向量 + 元数据：

- `programs: dict[str, FunctionIR]`
- `constants: np.ndarray`
- `generation`, `parent_ids`, `genome_id`

关键接口：

- `to_source(role)`：IR -> Python 源码
- `to_callable(role)`：IR -> Python 可调用对象（带缓存）
- `to_cpp_ops(role)`：IR -> C++ opcode 数组
- `clone()`：深拷贝程序与常量
- `structural_hash()`：按 opcode 序列做结构哈希，用于 niching
- `serialize()/deserialize()`：JSON 往返（程序以源码存储）
- `invalidate_cache()`：变异后清空 callable 缓存

容错策略：

- `to_callable()` 编译失败时回退为恒等 0.0 函数，保证进化流程不中断。

## 5.4 FitnessResult / FitnessEvaluator（`fitness.py`）

`FitnessResult`：

- `metrics: dict[str, float]`
- `weights: dict[str, float]`
- `is_valid: bool`
- `composite_score() = Σ weights[k] * metrics[k]`（越小越好）
- 无效个体直接返回 `inf`

`FitnessEvaluator`：

- 必须实现 `evaluate(genome) -> FitnessResult`
- 可选覆写 `evaluate_batch(genomes)` 做并行/批处理优化

这是整个框架的“领域解耦点”。

---

## 6. 进化算子细节（`operators.py` + `random_program.py`）

## 6.1 初始化：random_ir_program

策略：

1. 依据 `ProgramSpec` 参数名生成随机表达式树（支持 safe ops）。
2. 拼成函数源码 `def role(...): return expr`。
3. `compile_source_to_ir()` 编译成 `FunctionIR`。
4. 编译失败回退 `return 0.0`。

额外提供 `random_loop_program()` 用于生成带 `while` 的聚合型程序（当前主引擎默认用 `random_ir_program`）。

## 6.2 mutate_ir

支持突变类型：

- `point`：替换 `binary` / `compare` 的运算符
- `constant_perturb`：扰动数值常量
- `insert`：源码层插入临时语句后重编译
- `delete`：源码层删除可删语句后重编译

默认采样偏向前两种（更稳定）。

## 6.3 crossover_ir

思路：保留 `ir1` 结构，局部用 `ir2` 的常量/运算符作为 donor 替换，降低 CFG 破坏风险。

## 6.4 genome 级封装

- `mutate_genome()`：随机挑一个角色程序突变 + 可选常量扰动 + 常量裁剪到 `constant_range`
- `crossover_genome()`：逐角色交叉 + 常量线性混合（`alpha`）

---

## 7. EvolutionEngine 详细接口（重点）

## 7.1 构造函数

```python
EvolutionEngine(
    config: EvolutionConfig,
    evaluator: FitnessEvaluator,
    registry: SkeletonRegistry,
    rng: np.random.Generator | None = None,
)
```

参数说明：

- `config`: 所有 EA 超参数来源
- `evaluator`: 领域适应度评估器（唯一必须由应用实现）
- `registry`: 提供 `program_roles` 对应的 `ProgramSpec`
- `rng`: 可注入随机源；否则用 `config.seed`

内部状态：

- `population`, `fitness`, `generation`
- `best_ever`, `best_fitness`
- `hall_of_fame: list[(IRGenome, FitnessResult)]`
- `stagnation_count`
- `_history`（每代统计）

## 7.2 `init_population(seed_genomes=None)`

职责：构建并评估初始种群。

流程：

1. 从 `config.program_roles` 向 `registry` 查 `ProgramSpec`。
2. 先注入 `seed_genomes`（clone 后放入种群）。
3. 不足部分用 `random_ir_program` 生成每个 role 的 IR。
4. 生成常量向量（均匀分布于 `constant_range`）。
5. 调 evaluator 批量评估。
6. 调 `_update_best()` 更新全局最优与 HoF。

输入输出：

- 输入：可选种子个体列表
- 输出：无返回值，更新 `self.population/self.fitness/...`

## 7.3 `run(n_generations=None, callback=None, seed_genomes=None) -> IRGenome`

### 参数

- `n_generations`: 覆盖 `config.n_generations`
- `callback`: 每代回调  
  `callback(generation: int, best_fitness: FitnessResult, population: list[IRGenome])`
- `seed_genomes`: 若尚未初始化种群，作为初始注入

### 主循环步骤

每代固定顺序：

1. `offspring = _breed_next_generation()`
2. `off_fitness = evaluator.evaluate_batch(offspring)`
3. `combined = parents + offspring` 合并
4. 按 `composite_score` 升序排序（越小越好）
5. `_apply_niching()` 结构哈希去同质化
6. 截断到 `population_size` 作为下一代
7. `_update_best()` 更新最优与 HoF
8. 根据最优是否改善更新 `stagnation_count`
9. 若超过 `hard_restart_after` 则 `_hard_restart()`
10. 写入 `history`（best/avg/elapsed/stagnation）
11. 调用 `callback`（如有）

返回值：

- `best_ever`（历史全局最优个体）

## 7.4 `_breed_next_generation()`

实现细节：

- 生成后代数量：`population_size - elite_count`
- 每个后代：
  - `parent1 = _tournament_select()`
  - 以 `crossover_rate` 决定是否再选 `parent2` 交叉
  - 以 `mutation_rate` 决定是否突变
  - 写 `child.generation = 当前代`

说明：这里的“精英保留”并非直接复制 top-k，而是通过“父代 + 子代合并再截断”间接保留优秀父代；`elite_count` 只影响子代数量。

## 7.5 `_tournament_select()`

- 从当前种群无放回抽 `tournament_size`
- 选其中 `composite_score` 最小者

## 7.6 `_apply_niching(combined)`

- 对排序后的个体按 `structural_hash` 分桶
- 每个 hash 最多保留 `niche_radius` 个
- 作用：防止结构完全一致的个体淹没种群

## 7.7 `_update_best()`

- 找当前代最佳
- 若优于全局最佳则更新 `best_ever/best_fitness`
- 将当前代最佳追加进 `hall_of_fame`，再按分数排序并裁到 `hall_of_fame_size`

## 7.8 `_hard_restart()`

触发条件：`stagnation_count >= hard_restart_after`。

动作：

1. 保留 HoF（clone）
2. 其余位置随机重建
3. 全量重评估
4. 重置停滞计数

注意：

- 若把 `hall_of_fame_size` 设得大于 `population_size`，重启后种群可能偏大（当前实现没有显式截断）。

## 7.9 `history` 属性

每代记录字段：

- `generation`
- `best_score`
- `avg_score`
- `elapsed`
- `stagnation`

用于后处理与画进化曲线。

---

## 8. `slot_discovery.py`：自动发现可进化槽位

这个模块是“从完整函数中自动切出可替换区域”的实验能力。

核心对象 `SlotCandidate`：

- `region: RewriteRegion`
- `contract: BoundaryContract`
- `score`
- `program_spec`（自动生成）

入口：

- `discover_slots(func_ir, sample_inputs=None, mode="auto")`
  - `static`: 纯静态切片+区域评分
  - `dynamic`: interpreter 运行后结合 runtime 信息
  - `auto`: 二者融合排序

静态发现思路：

1. 遍历可候选 op（排除 const/phi/branch/return 等）
2. 对输出值做 backward slice
3. 限制 region 大小（2~20 ops）
4. `define_rewrite_region` + `infer_boundary_contract`
5. 按“中等规模 + 少入口”打分
6. 从 contract 推导 `ProgramSpec`

这部分使未来可以不手写 skeleton，而是从宿主算法 IR 中自动挖槽。

---

## 9. 真实接入示例：`applications/mimo_bp`

以 `applications/mimo_bp/run_evolution.py` 为例：

1. `bp_skeleton()` 注册 4 个 role：`f_down/f_up/f_belief/h_halt`
2. 构造 `EvolutionConfig`（含 BER/FLOPs/泛化 gap 权重）
3. 实例化 `MIMOBPFitnessEvaluator`（C++ DLL + 数据集）
4. 创建 `EvolutionEngine(config, evaluator, skeleton)`
5. 可注入手工 seed genomes
6. `engine.run(callback=...)` 持续记录每代日志
7. 导出 best genome（序列化 JSON + 程序源码）

这体现了“框架层通用 + 任务层可插拔”的边界。

---

## 10. 最小可用接口模板

```python
import numpy as np
from evolution.config import EvolutionConfig
from evolution.engine import EvolutionEngine
from evolution.fitness import FitnessEvaluator, FitnessResult
from evolution.skeleton_registry import ProgramSpec, SkeletonSpec, SkeletonRegistry
from evolution.genome import IRGenome


class MyEvaluator(FitnessEvaluator):
    def evaluate(self, genome: IRGenome) -> FitnessResult:
        fn = genome.to_callable("f_score")
        err = abs(fn(1.0, 2.0) - 3.0)
        return FitnessResult(metrics={"err": err}, weights={"err": 1.0}, is_valid=True)


reg = SkeletonRegistry()
reg.register(SkeletonSpec(
    skeleton_id="demo",
    program_specs=[
        ProgramSpec(
            name="f_score",
            param_names=["x", "y"],
            param_types=["float", "float"],
            return_type="float",
            constraints={"max_depth": 6},
        )
    ],
))

cfg = EvolutionConfig(
    population_size=40,
    n_generations=30,
    seed=42,
    program_roles=["f_score"],
    n_constants=2,
)

engine = EvolutionEngine(cfg, MyEvaluator(), reg, rng=np.random.default_rng(42))
best = engine.run()
print(best.to_source("f_score"))
```

---

## 11. 关键设计结论

1. `evolution` 的本质是“FunctionIR 级搜索”，不是源码文本搜索。  
2. `algorithm_ir` 提供了前端编译、IR载体、后端再生与动态分析能力，`evolution` 直接复用这些能力形成闭环。  
3. `EvolutionEngine` 只负责优化流程，不包含领域知识；领域目标完全由 `FitnessEvaluator` 决定。  
4. 若希望更强约束（签名严格校验、类型限制、语义安全），应在 evaluator 或初始化环节显式接入 `SkeletonRegistry.validate_*`。  
5. `slot_discovery` 提供了从“固定算法”过渡到“自动可进化区域”的路径，是后续自动化程度提升的关键模块。

