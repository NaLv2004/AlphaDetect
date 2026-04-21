# Algorithm-IR：基于中间表示的 MIMO 检测器自动进化框架

> 将 MIMO 检测算法编译为 SSA 中间表示，通过**双层进化**（宏观骨架嫁接 + 微观槽位变异）自动发现新型检测器；进一步用 **GNN 在线学习**引导骨架嫁接提议。

---

## 项目概览

Algorithm-IR 是一个端到端的**算法自动发现**系统，面向通信领域的 MIMO 检测问题。它实现了：

1. **算法 IR 编译**：将 Python 检测器函数编译为结构中立的 SSA 中间表示 (`FunctionIR`)
2. **骨架嫁接**：跨算法族的结构迁移（如将 BP 消息传递注入 K-Best 树搜索）
3. **双层进化**：宏观层进化骨架结构 + 微观层进化槽位实现
4. **GNN 引导提议**：图注意力网络（GAT）通过 REINFORCE 在线学习最优宿主-供体配对
5. **Monte Carlo 评估**：在真实 MIMO 信道上评估 SER 和复杂度
6. **代码再生**：进化后的 IR 可重新生成可执行 Python 源码

---

## 目录结构

```
research/algorithm-IR/
│
├── algorithm_ir/                 # 核心 IR 包 (编译/分析/嫁接/再生)
│   ├── ir/                       #   数据模型: Value, Op, Block, FunctionIR
│   │   ├── model.py              #   FunctionIR, ModuleIR 定义
│   │   ├── dialect.py            #   22 个 IRDL 操作定义 (AlgConst, ..., AlgSlot)
│   │   ├── types.py              #   AlgType — 统一类型包装
│   │   ├── printer.py            #   render_function_ir() — 可读文本输出
│   │   ├── validator.py          #   validate_function_ir() — 结构验证
│   │   └── xdsl_bridge.py        #   xDSL ↔ dict-IR 双向转换
│   ├── frontend/                 #   Python → IR 编译前端
│   │   ├── ast_parser.py         #   parse_function() — AST 解析
│   │   ├── ir_builder.py         #   compile_source_to_ir(), compile_function_to_ir()
│   │   └── cfg_builder.py        #   CFGBlock, 控制流图构建
│   ├── regeneration/             #   IR → Python/C++ 代码再生
│   │   ├── codegen.py            #   emit_python_source(), emit_cpp_ops()
│   │   └── artifact.py           #   AlgorithmArtifact
│   ├── grafting/                 #   骨架嫁接（核心创新）
│   │   ├── graft_general.py      #   ★ graft_general() — 通用区域替换 + 调用注入
│   │   ├── rewriter.py           #   graft_skeleton() — xDSL 级嫁接
│   │   ├── matcher.py            #   match_skeleton() — 骨架模式匹配
│   │   └── skeletons.py          #   Skeleton, OverridePlan
│   ├── region/                   #   可重写区域
│   │   ├── selector.py           #   define_rewrite_region()
│   │   ├── contract.py           #   infer_boundary_contract()
│   │   └── slicer.py             #   backward_slice_by_values(), forward_slice_from_values()
│   ├── runtime/                  #   IR 解释执行
│   │   ├── interpreter.py        #   execute_ir()
│   │   ├── tracer.py             #   RuntimeValue, RuntimeEvent
│   │   ├── frames.py             #   RuntimeFrame
│   │   └── shadow_store.py       #   ShadowStore — 可变对象追踪
│   ├── analysis/                 #   静态/动态分析
│   │   ├── static_analysis.py    #   def_use_edges(), block_uses()
│   │   ├── dynamic_analysis.py   #   runtime_values_for_static()
│   │   └── fingerprints.py       #   fingerprint_runtime_value()
│   ├── factgraph/                #   因子图 (静态 IR + 动态执行对齐)
│   │   ├── model.py              #   FactGraph
│   │   ├── builder.py            #   build_factgraph()
│   │   └── aligner.py            #   静态-动态对齐
│   └── projection/               #   可选语义标注 (scheduling, local_interaction)
│       ├── base.py, scorer.py
│       ├── scheduling.py
│       └── local_interaction.py
│
├── evolution/                    # 进化引擎包
│   ├── algorithm_engine.py       #   ★ AlgorithmEvolutionEngine — 双层进化主循环
│   ├── pool_types.py             #   AlgorithmGenome, AlgorithmEntry, GraftProposal, ...
│   ├── mimo_evaluator.py         #   MIMOFitnessEvaluator — MIMO Monte Carlo 评估
│   ├── materialize.py            #   materialize(), materialize_to_callable()
│   ├── ir_pool.py                #   build_ir_pool() — 8 个基线检测器
│   ├── pattern_matchers.py       #   ★ ExpertPatternMatcher, CompositePatternMatcher, ...
│   ├── operators.py              #   mutate_ir(), crossover_ir()
│   ├── random_program.py         #   random_ir_program()
│   ├── slot_discovery.py         #   discover_slots()
│   ├── pool_ops_l0.py            #   L0: 标量/向量/矩阵原子操作 (~80)
│   ├── pool_ops_l1.py            #   L1: 信号处理模块 (10)
│   ├── pool_ops_l2.py            #   L2: 算法组件 (13)
│   ├── pool_ops_l3.py            #   L3: 完整检测器 (8)
│   ├── engine.py                 #   [旧版] EvolutionEngine (单层)
│   ├── genome.py                 #   [旧版] IRGenome
│   ├── config.py                 #   [旧版] EvolutionConfig
│   ├── fitness.py                #   FitnessResult, FitnessEvaluator
│   ├── skeleton_registry.py      #   ProgramSpec, SkeletonSpec, SkeletonRegistry
│   ├── algorithm_pool.py         #   build_initial_pool()
│   └── run_evolution.py          #   [旧版] 进化入口
│
├── tests/                        # 测试套件 (242 个测试)
│   ├── unit/                     #   单元测试
│   │   ├── test_frontend.py      #     IR 编译
│   │   ├── test_dialect.py       #     AlgDialect 22 操作
│   │   ├── test_regression_p0.py #     P0 级回归
│   │   ├── test_evolution.py     #     单层进化
│   │   ├── test_algorithm_pool.py#     双层基因组 + 池
│   │   ├── test_ir_evolution.py  #     IR 变异/交叉
│   │   ├── test_region_projection.py # 区域投影
│   │   ├── test_runtime_factgraph.py # 运行时 + 因子图
│   │   └── ...
│   ├── integration/              #   集成测试
│   │   ├── test_grafting_demo.py #     骨架嫁接端到端
│   │   ├── test_e2e_pipeline.py  #     完整进化管线
│   │   └── ...
│   └── cross_lang/               #   跨语言测试
│       └── test_cpp_codegen.py   #     C++ 操作码生成
│
├── e2e_experiment.py             # ★ 端到端进化实验入口（Expert+Static+GNN 匹配器）
├── train_gnn.py                  # ★ GNN 模式匹配器大规模训练入口
├── baseline_eval.py              #   评估 8 个基线检测器（16×16 16-QAM，24dB）
├── osic_sweep.py                 #   OSIC SNR 扫描（18–28dB，精细评估）
└── conftest.py                   #   pytest 配置
```

---

## 核心架构

### 双层进化流程

```
┌─────────────────────────────────────────────────────────┐
│                   E2E 实验入口                           │
│  e2e_experiment.py                                      │
│    → build_ir_pool() → 8 个 AlgorithmGenome (基线检测器) │
│    → AlgorithmEvolutionEngine.run()                     │
└───────────────────────┬─────────────────────────────────┘
                        │
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
    宏观层          微观层          评估层
   ────────        ────────        ────────
   PatternMatcher  slot 子种群     materialize()
   → GraftProposal  mutate_ir()   → exec Python
   → graft_general  crossover_ir  → MIMO Monte Carlo
   → 骨架嫁接子代   → 适应度排序   → SER + 复杂度

         │              │              │
         └──────────────┼──────────────┘
                        ▼
              合并选择 (按 composite_score)
              → 下一代种群
```

### 关键创新：骨架嫁接

传统 GP 只在同一骨架内进化子程序。Algorithm-IR 支持**跨算法族的结构迁移**：

- **ExpertPatternMatcher** 提供 5 条专家规则（如"将 LMMSE 初始化注入 K-Best"）
- **graft_general()** 在 IR 级执行手术：克隆宿主 → 定义替换区域 → 创建 `call` 操作调用供体 → 重绑定输出 → 拓扑排序
- **Gene Bank** 永久保存原始 8 个检测器，确保任何代都能找到嫁接供体

---

## 快速开始

### 环境准备

```bash
conda activate AutoGenOld    # Python 3.12+
cd research/algorithm-IR
```

### 1. 编译一个 Python 函数到 IR

```python
from algorithm_ir.frontend import compile_source_to_ir
from algorithm_ir.ir.printer import render_function_ir

source = """
def lmmse(H, y, sigma2, constellation):
    Nt = H.shape[1]
    W = np.linalg.solve(H.conj().T @ H + sigma2 * np.eye(Nt), H.conj().T)
    x_hat = W @ y
    return x_hat
"""
func_ir = compile_source_to_ir(source, "lmmse")
print(render_function_ir(func_ir))
```

### 2. 查看 8 个基线检测器

```python
from evolution.ir_pool import build_ir_pool

pool = build_ir_pool()
for g in pool:
    slots = list(g.slot_populations.keys())
    print(f"{g.algo_id:8s} tags={g.tags} slots={slots}")
# lmmse    tags={'linear', 'original'}          slots=['slot_regularizer', 'slot_hard_decision']
# zf       tags={'linear', 'original'}          slots=['slot_hard_decision']
# osic     tags={'sic', 'original'}             slots=['slot_ordering', 'slot_sic_step']
# kbest    tags={'tree_search', 'original'}     slots=['slot_expand', 'slot_prune']
# stack    tags={'tree_search', 'original'}     slots=['slot_expand', 'slot_prune']
# bp       tags={'message_passing', 'original'} slots=['slot_bp_sweep', 'slot_bp_final']
# ep       tags={'inference', 'original'}       slots=['slot_ep_site', 'slot_ep_final']
# amp      tags={'inference', 'original'}       slots=['slot_amp_denoise', 'slot_amp_final']
```

### 3. 物化并执行

```python
import numpy as np
from evolution.materialize import materialize_to_callable

fn = materialize_to_callable(pool[0])  # lmmse
H = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)) / np.sqrt(2)
x = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
y = H @ x + 0.1 * (np.random.randn(4) + 1j * np.random.randn(4))
x_hat = fn(H, y, 0.01, np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2))
```

### 4. 评估基线检测器

```bash
# 评估所有 8 个基线检测器 (16×16 16-QAM, SNR=24dB, 500 trials)
conda run --no-capture-output -n AutoGenOld python baseline_eval.py

# OSIC SNR 扫描 (18–28dB, 每点 2000 trials)
conda run --no-capture-output -n AutoGenOld python osic_sweep.py
```

### 5. 运行端到端进化实验

```bash
# 端到端进化 (50 代，Expert+Static+GNN 匹配器，16×16 16-QAM)
conda run --no-capture-output -n AutoGenOld python e2e_experiment.py
```

输出示例：
```
=== AlgorithmEvolutionEngine E2E Experiment ===
Population initialized: 8 genomes, best=0.000668
Structural grafts: 3/3 succeeded
Gen 1 in 46.7s: best=0.000668 (algo_6a98e5ca, gen=0, tags={'tree_search', 'original'})
Structural grafts: 3/3 succeeded
Gen 2 in 42.1s: best=0.000523 (algo_b3f17d21, gen=2, tags={'linear', 'grafted'})
...
```

结果保存到 `results/evolution_16x16_16qam_24dB/`（最佳检测器源码 + JSON 结果）。

### 6. GNN 大规模训练

`train_gnn.py` 使用 GAT 网络在线学习最优宿主-供体配对，每代产生 500+ 个嫁接提议，并通过 SNR 课程逐步提升训练难度。

```bash
# 默认配置（200 代，SNR 从 20dB 升到 24dB，500 提议/代）
conda run --no-capture-output -n AutoGenOld python train_gnn.py

# 自定义配置
conda run --no-capture-output -n AutoGenOld python train_gnn.py \
    --gens 200 \
    --snr-start 20 \
    --snr-target 24 \
    --proposals 500 \
    --pool-size 141 \
    --n-trials 5 \
    --timeout 1.5 \
    --warmstart-gens 1 \
    --warmstart-eval-workers 8 \
    --ckpt-interval 10 \
    --seed 42

# 从检查点恢复
conda run --no-capture-output -n AutoGenOld python train_gnn.py \
    --resume results/gnn_training/gnn_ckpt_gen050.pt
```

**`train_gnn.py` 参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--gens` | 200 | 总进化代数 |
| `--snr-start` | 20.0 | 初始训练 SNR（dB） |
| `--snr-target` | 24.0 | 目标 SNR（dB） |
| `--phase-thresh` | 0.5 | 触发 SNR 升级的嫁接成功率阈值 |
| `--proposals` | 500 | 每代 GNN 嫁接提议数 |
| `--pool-size` | 141 | 种群规模（91 原始 + 50 嫁接幸存者） |
| `--n-trials` | 5 | 每个基因组的评估试验数（320 bits/trial） |
| `--timeout` | 1.5 | 每个基因组的评估超时（秒） |
| `--eval-workers` | 1 | 主评估器并行工作者数 |
| `--warmstart-gens` | 1 | 热启动代数（穷举宿主-供体对扫描） |
| `--warmstart-trials` | 1 | 热启动阶段每次嫁接的试验数 |
| `--warmstart-timeout` | 0.5 | 热启动评估超时（秒） |
| `--warmstart-eval-workers` | 8 | 热启动并行工作者数 |
| `--warmstart-survivor-cap` | 48 | 进入种群选择的热启动嫁接子代上限 |
| `--train-steps` | 5 | 每代 GNN 梯度步数 |
| `--train-interval` | 1 | 每 N 代训练一次 GNN |
| `--buffer-size` | 20000 | 嫁接经验回放缓冲区大小 |
| `--pair-temp` | 0.7 | 对采样 softmax 温度 |
| `--pair-eps` | 0.10 | 对采样均匀探索混合比 |
| `--ckpt-interval` | 10 | 每 N 代保存 GNN 检查点 |
| `--seed` | 42 | 随机种子 |
| `--resume` | None | 从指定检查点路径恢复 |

输出写入 `results/gnn_training/`：
- `training_log.jsonl` — 每代的 JSONL 格式日志
- `top50_grafts.log` — 每代 Top-50 嫁接个体的人类可读报告
- `gnn_ckpt_gen{N:04d}.pt` — 每隔 `--ckpt-interval` 代保存的 GNN 权重

### 7. 运行全部测试

```bash
python -m pytest tests/ -v   # 242 tests, ~30s
```

---

## 系统详解

详细文档参见各子包的 README：

| 包 | 文档 | 主题 |
|---|------|------|
| `algorithm_ir/` | [algorithm_ir/readme.md](algorithm_ir/readme.md) | IR 编译、数据模型、骨架嫁接、代码再生 |
| `evolution/` | [evolution/README.md](evolution/README.md) | 双层进化引擎、PatternMatcher、GNN 匹配器、8 个基线检测器 |

---

## API 速查

### IR 编译与再生

```python
# 编译
from algorithm_ir.frontend import compile_source_to_ir, compile_function_to_ir
func_ir = compile_source_to_ir(source_code, func_name)
func_ir = compile_function_to_ir(python_function)

# 查看
from algorithm_ir.ir.printer import render_function_ir
print(render_function_ir(func_ir))

# 验证
from algorithm_ir.ir.validator import validate_function_ir
errors = validate_function_ir(func_ir)

# 执行
from algorithm_ir.runtime.interpreter import execute_ir
result, trace, values = execute_ir(func_ir, args=[...])

# Python 再生
from algorithm_ir.regeneration.codegen import emit_python_source
source = emit_python_source(func_ir)

# C++ 操作码
from algorithm_ir.regeneration.codegen import emit_cpp_ops
opcodes = emit_cpp_ops(func_ir)
```

### 区域与嫁接

```python
# 后向切片
from algorithm_ir.region.slicer import backward_slice_by_values
ops = backward_slice_by_values(func_ir, target_values=["v_5"])

# 区域定义 + 契约
from algorithm_ir.region.selector import define_rewrite_region
from algorithm_ir.region.contract import infer_boundary_contract
region = define_rewrite_region(func_ir, ops)
contract = infer_boundary_contract(func_ir, region)

# 通用嫁接
from algorithm_ir.grafting.graft_general import graft_general
artifact = graft_general(host_ir, proposal)
```

### 双层进化

```python
from evolution.ir_pool import build_ir_pool
from evolution.pool_types import AlgorithmEvolutionConfig
from evolution.algorithm_engine import AlgorithmEvolutionEngine
from evolution.mimo_evaluator import MIMOEvalConfig, MIMOFitnessEvaluator
from evolution.pattern_matchers import ExpertPatternMatcher, CompositePatternMatcher
from evolution.materialize import materialize, materialize_to_callable

# 构建池
pool = build_ir_pool()

# 物化
source = materialize(genome)
fn = materialize_to_callable(genome)

# 进化
engine = AlgorithmEvolutionEngine(evaluator, config, rng, pattern_matcher)
best = engine.run(callback=cb)
```

---

## 技术栈

- **Python 3.12+**（conda `AutoGenOld` 环境）
- **numpy / scipy** — 数值计算
- **torch / torch_geometric** — GNN 训练（GAT 网络 + REINFORCE）
- **xDSL** — IR 基础设施（IRDL 方言定义、类型系统）
- **pytest** — 测试框架（242 个测试）

---

## 设计原则

1. **结构中立**：IR 不预设算法类别，只记录可执行事实
2. **SSA 形式**：每个值只赋值一次，数据流显式表达
3. **双层进化**：宏观层负责跨算法族结构迁移，微观层负责局部优化
4. **可物化**：任何 `AlgorithmGenome` 都能生成可执行的 Python 代码
5. **Gene Bank**：原始检测器永久保留，确保嫁接供体始终可用
6. **领域解耦**：进化引擎不包含通信领域知识，领域目标通过 `FitnessEvaluator` 外挂