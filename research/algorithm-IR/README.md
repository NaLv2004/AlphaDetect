# Algorithm-IR：基于中间表示的 MIMO 检测器自动进化框架

> 将 MIMO 检测算法编译为 SSA 中间表示，通过**双层进化**（宏观骨架嫁接 + 微观槽位变异）自动发现新型检测器。

---

## 项目概览

Algorithm-IR 是一个端到端的**算法自动发现**系统，面向通信领域的 MIMO 检测问题。它实现了：

1. **算法 IR 编译**：将 Python 检测器函数编译为结构中立的 SSA 中间表示 (`FunctionIR`)
2. **骨架嫁接**：跨算法族的结构迁移（如将 BP 消息传递注入 K-Best 树搜索）
3. **双层进化**：宏观层进化骨架结构 + 微观层进化槽位实现
4. **Monte Carlo 评估**：在真实 MIMO 信道上评估 SER 和复杂度
5. **代码再生**：进化后的 IR 可重新生成可执行 Python 源码

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
├── e2e_experiment.py             # ★ 端到端进化实验入口
├── demo.py                       #   骨架嫁接演示
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
conda activate AutoGenOld    # Python 3.12.9
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
# lmmse    tags={'linear', 'original'}     slots=['slot_regularizer', 'slot_hard_decision']
# zf       tags={'linear', 'original'}     slots=['slot_hard_decision']
# osic     tags={'sic', 'original'}        slots=['slot_ordering', 'slot_sic_step']
# kbest    tags={'tree_search', 'original'} slots=['slot_expand', 'slot_prune']
# stack    tags={'tree_search', 'original'} slots=['slot_expand', 'slot_prune']
# bp       tags={'message_passing', 'original'} slots=['slot_bp_sweep', 'slot_bp_final']
# ep       tags={'inference', 'original'}  slots=['slot_ep_site', 'slot_ep_final']
# amp      tags={'inference', 'original'}  slots=['slot_amp_denoise', 'slot_amp_final']
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

### 4. 运行端到端进化实验

```bash
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

### 5. 运行全部测试

```bash
python -m pytest tests/ -v   # 242 tests, ~30s
```

---

## 系统详解

详细文档参见各子包的 README：

| 包 | 文档 | 主题 |
|---|------|------|
| `algorithm_ir/` | [algorithm_ir/readme.md](algorithm_ir/readme.md) | IR 编译、数据模型、骨架嫁接、代码再生 |
| `evolution/` | [evolution/README.md](evolution/README.md) | 双层进化引擎、PatternMatcher、8 个基线检测器 |

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
# Algorithm-IR：基于中间表示的算法进化框架

> 一个将 Python 函数编译为编译器中间表示 (IR)，在 IR 上执行遗传进化，再生成 Python / C++ 可执行代码的端到端框架。

---

## 目录

- [什么是 IR？为什么需要它？](#什么是-ir为什么需要它)
- [项目结构](#项目结构)
- [核心概念速览](#核心概念速览)
- [完整使用流程](#完整使用流程)
  - [第 1 步：Python → IR 编译](#第-1-步python--ir-编译)
  - [第 2 步：查看与理解 IR](#第-2-步查看与理解-ir)
  - [第 3 步：IR → Python 回生成](#第-3-步ir--python-回生成)
  - [第 4 步：IR → C++ 操作码](#第-4-步ir--c-操作码)
  - [第 5 步：定义骨架 (Skeleton) 与程序规格](#第-5-步定义骨架-skeleton-与程序规格)
  - [第 6 步：生成随机 IR 程序](#第-6-步生成随机-ir-程序)
  - [第 7 步：IR 变异与交叉](#第-7-步ir-变异与交叉)
  - [第 8 步：构建基因组 (Genome)](#第-8-步构建基因组-genome)
  - [第 9 步：定义适应度评估器](#第-9-步定义适应度评估器)
  - [第 10 步：运行进化引擎](#第-10-步运行进化引擎)
- [应用示例：MIMO BP 检测器进化](#应用示例mimo-bp-检测器进化)
- [C++ 后端](#c-后端)
- [API 参考](#api-参考)
- [运行测试](#运行测试)

---

## 什么是 IR？为什么需要它？

**IR（Intermediate Representation，中间表示）** 是编译器内部使用的一种数据结构，位于"源代码"和"机器代码"之间。你可以把它想成一种"标准化的伪代码"：

```
Python 源代码               IR（中间表示）              目标代码
  def f(a, b):      →    const 2.0            →    C++ opcodes [0,lo,hi,1,0,4,...]
    return a * 2.0        load_arg a                 或 Python source "def f(a,b):..."
                          binary Mult
                          return
```

**为什么需要 IR？** 因为直接操作 Python 文本字符串来做遗传编程既脆弱又低效：

| 直接操作文本 | 通过 IR 操作 |
|---|---|
| "把第3行的 `+` 换成 `-`" — 容易出错 | 修改 `Op(opcode="binary", attrs={"operator": "Add"})` 的 `operator` 为 `"Sub"` — 精确安全 |
| 无法验证语法正确性 | IR 自带结构化验证 |
| 无法生成 C++ | IR 可以同时生成 Python 和 C++ |
| 交叉操作会破坏语法 | 在 IR 上交叉只交换节点，结构完好 |

本项目的 IR 基于 [xDSL](https://xdsl.dev/) 框架构建，采用 SSA（Static Single Assignment，静态单赋值）形式。简单来说：**每个变量只被赋值一次**，如果要改变一个变量的值，就创建一个新版本。

---

## 项目结构

```
research/algorithm-IR/
│
├── algorithm_ir/                      # ★ 核心 IR 基础设施
│   ├── ir/                            # IR 数据模型
│   │   ├── model.py                   #   Value, Op, Block, FunctionIR — 核心数据类
│   │   ├── dialect.py                 #   21 个 xDSL IRDL 操作定义 (AlgConst, AlgBinary, ...)
│   │   ├── printer.py                 #   render_function_ir() — IR 的可读文本输出
│   │   ├── validator.py               #   validate_function_ir() — 结构验证
│   │   ├── types.py                   #   AlgType — 类型包装
│   │   ├── type_info.py               #   TypeInfo — 类型推导辅助
│   │   └── xdsl_bridge.py             #   xDSL ↔ dict-IR 桥接
│   │
│   ├── frontend/                      # ★ 编译前端：Python → IR
│   │   ├── ir_builder.py              #   compile_function_to_ir(), compile_source_to_ir()
│   │   ├── ast_parser.py              #   Python AST 解析
│   │   └── cfg_builder.py             #   控制流图构建
│   │
│   ├── regeneration/                  # ★ 代码生成后端：IR → Python/C++
│   │   ├── codegen.py                 #   emit_python_source(), emit_cpp_ops(), CppOp
│   │   └── artifact.py                #   AlgorithmArtifact
│   │
│   ├── analysis/                      # 静态分析
│   ├── factgraph/                     # 因子图构建
│   ├── grafting/                      # 骨架嫁接（替换区域）
│   ├── projection/                    # 区域投影分析
│   ├── region/                        # 可重写区域定义
│   └── runtime/                       # IR 解释执行
│
├── evolution/                         # ★ 通用进化引擎
│   ├── config.py                      #   EvolutionConfig — 所有超参数
│   ├── fitness.py                     #   FitnessResult, FitnessEvaluator (抽象基类)
│   ├── genome.py                      #   IRGenome — 进化个体（N 个 FunctionIR + 常量）
│   ├── skeleton_registry.py           #   ProgramSpec, SkeletonSpec, SkeletonRegistry
│   ├── slot_discovery.py              #   自动可变区域发现
│   ├── random_program.py              #   随机 IR 程序生成
│   ├── operators.py                   #   变异 (mutate_ir) / 交叉 (crossover_ir)
│   └── engine.py                      #   EvolutionEngine — 主进化循环
│
├── applications/                      # ★ 领域应用
│   └── mimo_bp/                       # MIMO BP 检测器进化
│       ├── bp_skeleton.py             #   4 个程序角色定义
│       ├── evaluator.py               #   MIMOBPFitnessEvaluator
│       ├── mimo_simulation.py         #   MIMO 数据集生成
│       ├── run_evolution.py           #   进化入口脚本
│       ├── validate_decoder.py        #   快速验证
│       ├── validate_large.py          #   多信噪比全面验证
│       ├── cpp_evaluator.py           #   Python ↔ C++ DLL 桥接
│       └── cpp/                       #   C++ 后端源码 & 编译产物
│           ├── ir_eval.h              #     33 操作码的栈式求值器
│           ├── bp_ir_decoder.cpp      #     完整 BP 树搜索解码器
│           ├── bp_ir_eval.dll         #     编译好的动态库
│           └── build.bat              #     MSVC 编译脚本
│
├── tests/                             # 测试
│   ├── unit/                          #   单元测试
│   │   ├── test_frontend.py           #     IR 编译测试
│   │   └── test_evolution.py          #     进化框架全面测试 (40+ 用例)
│   ├── cross_lang/
│   │   └── test_consistency.py        #     Python ↔ C++ 一致性测试 (13 用例)
│   └── integration/
│       └── test_grafting_demo.py      #     嫁接集成测试
│
└── ir_tutorial_demo.py                # ★ 交互式教程 (本 README 的配套脚本)
```

### 调用关系概览

```
用户代码
  │
  ├──→ compile_source_to_ir()  ──→  FunctionIR  (核心 IR 对象)
  │         (frontend)                   │
  │                                      ├──→ emit_python_source()    (regeneration)
  │                                      ├──→ emit_cpp_ops()          (regeneration)
  │                                      └──→ render_function_ir()    (printer)
  │
  ├──→ SkeletonRegistry.register()       定义程序骨架（参数签名）
  │         │
  │         └──→ random_ir_program()     按规格生成随机 IR
  │                    │
  │                    └──→ IRGenome(programs={"role": ir, ...})
  │
  ├──→ EvolutionConfig(...)              配置超参数
  │
  ├──→ FitnessEvaluator.evaluate()       评估适应度（用户实现）
  │
  └──→ EvolutionEngine(config, evaluator, registry)
              │
              └──→ engine.run()          主进化循环
                      │
                      ├── init_population()    生成/注入初始种群
                      ├── evaluate_batch()     适应度评估
                      ├── mutate_genome()      IR 级变异
                      ├── crossover_genome()   IR 级交叉
                      └── 返回 best: IRGenome
```

---

## 核心概念速览

### 1. FunctionIR — 函数的中间表示

一个 `FunctionIR` 对象表示**一个函数**的完整内部结构。它包含三种基本元素：

| 元素 | 类 | 含义 | 类比 |
|---|---|---|---|
| **Value** | `Value(id, name_hint, type_hint, def_op, use_ops, attrs)` | 一个数据值（变量、常量、中间结果） | 电路中的一根导线 |
| **Op** | `Op(id, opcode, inputs, outputs, block_id, attrs)` | 一个操作（加法、比较、返回...） | 电路中的一个门 |
| **Block** | `Block(id, op_ids, preds, succs)` | 一组顺序执行的操作 | 代码中的一个段落 |

**SSA 形式**：每个 `Value` 只被一个 `Op` 定义（`def_op` 字段），但可被多个 `Op` 使用（`use_ops` 字段）。

### 2. Opcodes — 操作码

IR 支持的操作码：

| 操作码 | 含义 | 示例 |
|---|---|---|
| `const` | 常量 | `3.14`, `True`, `"hello"` |
| `assign` | 赋值（SSA 重命名） | `x_1 = x_0` |
| `binary` | 二元运算 | `a + b`, `x * y` |
| `unary` | 一元运算 | `-x`, `not flag` |
| `compare` | 比较运算 | `a < b`, `x >= 0` |
| `call` | 函数调用 | `abs(x)`, `_safe_log(y)` |
| `branch` | 条件跳转 | `if cond: goto block_A else: goto block_B` |
| `jump` | 无条件跳转 | `goto block_C` |
| `return` | 函数返回 | `return result` |
| `phi` | φ 节点（合并分支） | 两个分支中 `x` 的值在此合并 |

### 3. IRGenome — 进化个体

一个 `IRGenome` 包含**多个**命名的 `FunctionIR`，每个对应一个"程序角色"（如 `f_down`, `f_up`），加上一组可进化的浮点常量。

### 4. SkeletonRegistry — 骨架注册

骨架定义了**需要进化的程序有哪些、每个程序接受什么参数、返回什么类型**。这是进化框架和具体应用之间的桥梁。

---

## 完整使用流程

> 以下所有示例都在 `research/algorithm-IR/` 目录下运行。
> 环境：`conda activate AutoGenOld`

### 第 1 步：Python → IR 编译

将一个 Python 函数编译为 `FunctionIR`：

```python
from algorithm_ir.frontend import compile_source_to_ir

source = """
def score(cum_dist, m_down, m_up):
    return cum_dist + m_down + m_up
"""

func_ir = compile_source_to_ir(source, "score")
print(f"函数名: {func_ir.name}")
print(f"参数:   {func_ir.arg_values}")
print(f"操作数: {len(func_ir.ops)}")
print(f"值数:   {len(func_ir.values)}")
print(f"块数:   {len(func_ir.blocks)}")
```

**输出：**
```
函数名: score
参数:   ['v_0', 'v_1', 'v_2']
操作数: 4
值数:   7
块数:   1
```

如果源码中使用了 `_safe_log` 等辅助函数，需要提供 `globals_dict`：

```python
import math

helpers = {
    "__builtins__": __builtins__,
    "_safe_log": lambda a: math.log(max(a, 1e-30)),
    "_safe_div": lambda a, b: a / b if abs(b) > 1e-30 else 0.0,
    "_safe_sqrt": lambda a: math.sqrt(max(a, 0.0)),
    "abs": abs,
    "math": math,
}

source = """
def f_up(sum_child_ld, sum_child_m_up, n_children):
    return _safe_log(sum_child_ld)
"""

func_ir = compile_source_to_ir(source, "f_up", globals_dict=helpers)
```

也可以直接传入 Python 函数对象（不需要字符串）：

```python
from algorithm_ir.frontend import compile_function_to_ir

def my_func(a, b):
    return a * b + a

func_ir = compile_function_to_ir(my_func)
```

---

### 第 2 步：查看与理解 IR

#### 方法 A：文本渲染

```python
from algorithm_ir.ir import render_function_ir

print(render_function_ir(func_ir))
```

**输出：**
```
FunctionIR(name=score, entry=b_entry)
  Block b_entry preds=[] succs=[]
    o_0: const in=[] out=[v_3] attrs={'literal': None, 'name': 'cum_dist'}
    o_1: binary in=[v_0, v_1] out=[v_4] attrs={'operator': 'Add'}
    o_2: binary in=[v_4, v_2] out=[v_5] attrs={'operator': 'Add'}
    o_3: return in=[v_5] out=[v_6]
```

#### 方法 B：遍历操作

```python
for op_id, op in func_ir.ops.items():
    print(f"{op_id}: {op.opcode:10s} inputs={op.inputs} → outputs={op.outputs}")
    if op.attrs:
        print(f"             attrs={op.attrs}")
```

#### 方法 C：查看值的定义-使用链

```python
for vid, val in func_ir.values.items():
    hint = val.name_hint or ""
    defined_by = val.def_op or "(arg)"
    used_by = val.use_ops
    print(f"  {vid} ({hint}): defined by {defined_by}, used by {used_by}")
```

**输出示例：**
```
  v_0 (cum_dist): defined by (arg), used by ['o_1']
  v_1 (m_down):   defined by (arg), used by ['o_1']
  v_2 (m_up):     defined by (arg), used by ['o_2']
  v_4 ():         defined by o_1,   used by ['o_2']
  v_5 ():         defined by o_2,   used by ['o_3']
```

#### 方法 D：结构验证

```python
from algorithm_ir.ir import validate_function_ir

errors = validate_function_ir(func_ir)
print("验证通过!" if not errors else f"错误: {errors}")
```

---

### 第 3 步：IR → Python 回生成

将 IR 重新转换为可读的 Python 源代码：

```python
from algorithm_ir.regeneration.codegen import emit_python_source

python_source = emit_python_source(func_ir)
print(python_source)
```

**输出：**
```python
def score(cum_dist, m_down, m_up):
    return (cum_dist + m_down) + m_up
```

---

### 第 4 步：IR → C++ 操作码

将 IR 编译为扁平的整数操作码数组，用于 C++ 栈式求值器：

```python
from algorithm_ir.regeneration.codegen import emit_cpp_ops, CppOp

cpp_ops = emit_cpp_ops(func_ir)
print(f"操作码数组 ({len(cpp_ops)} 个整数): {cpp_ops}")

# 反向解读
opcode_names = {v: k for k, v in vars(CppOp).items() if isinstance(v, int)}
i = 0
while i < len(cpp_ops):
    op = cpp_ops[i]
    name = opcode_names.get(op, f"UNKNOWN({op})")
    if op == CppOp.CONST_F64:
        # 后面跟两个 int32 编码一个 float64
        import struct
        lo, hi = cpp_ops[i+1], cpp_ops[i+2]
        raw = struct.pack("<ii", lo, hi)
        val = struct.unpack("<d", raw)[0]
        print(f"  [{i:3d}] {name} {val}")
        i += 3
    elif op == CppOp.LOAD_ARG:
        print(f"  [{i:3d}] {name} arg_index={cpp_ops[i+1]}")
        i += 2
    else:
        print(f"  [{i:3d}] {name}")
        i += 1
```

**输出：**
```
操作码数组 (6 个整数): [1, 0, 1, 1, 2, 1, 2, 2, 24]
  [  0] LOAD_ARG arg_index=0    # push cum_dist
  [  2] LOAD_ARG arg_index=1    # push m_down
  [  4] ADD                     # pop两个, push (cum_dist + m_down)
  [  5] LOAD_ARG arg_index=2    # push m_up
  [  7] ADD                     # pop两个, push (result + m_up)
  [  8] RETURN                  # 返回栈顶值
```

C++ 求值器 (`ir_eval.h`) 用一个栈来执行这些操作码。就像一个反波兰表达式计算器。

---

### 第 5 步：定义骨架 (Skeleton) 与程序规格

骨架定义了进化需要发现的**程序接口**——每个"角色"需要什么参数、返回什么类型：

```python
from evolution.skeleton_registry import ProgramSpec, SkeletonSpec, SkeletonRegistry

# 定义两个程序角色
spec = SkeletonSpec(
    skeleton_id="my_detector",
    program_specs=[
        ProgramSpec(
            name="scorer",
            param_names=["distance", "confidence"],
            param_types=["float", "float"],
            return_type="float",
            constraints={"min_depth": 1, "max_depth": 6},
        ),
        ProgramSpec(
            name="halter",
            param_names=["old_score", "new_score"],
            param_types=["float", "float"],
            return_type="float",
            constraints={"min_depth": 1, "max_depth": 4},
        ),
    ],
)

# 注册到 SkeletonRegistry
registry = SkeletonRegistry()
registry.register(spec)

print(f"已注册的角色: {registry.roles}")
# 输出: 已注册的角色: ['scorer', 'halter']

# 获取某个角色的规格
ps = registry.get_program_spec("scorer")
print(f"scorer 参数: {ps.param_names}, 类型: {ps.param_types}")
# 输出: scorer 参数: ['distance', 'confidence'], 类型: ['float', 'float']
```

**验证程序是否匹配骨架：**

```python
source = """
def scorer(distance, confidence):
    return distance + confidence
"""
ir = compile_source_to_ir(source, "scorer")
violations = registry.validate_program("scorer", ir)
print(f"验证结果: {'通过' if not violations else violations}")
# 输出: 验证结果: 通过
```

---

### 第 6 步：生成随机 IR 程序

按照 `ProgramSpec` 规格自动生成随机的合法 IR 程序：

```python
import numpy as np
from evolution.random_program import random_ir_program

rng = np.random.default_rng(42)
spec = registry.get_program_spec("scorer")

# 生成 3 个不同的随机程序
for i in range(3):
    ir = random_ir_program(spec, rng, max_depth=3)
    source = emit_python_source(ir)
    print(f"\n--- 随机程序 {i+1} ---")
    print(source)
```

**输出示例：**
```
--- 随机程序 1 ---
def scorer(distance, confidence):
    return (distance - confidence) + distance

--- 随机程序 2 ---
def scorer(distance, confidence):
    return abs(distance * 0.7432)

--- 随机程序 3 ---
def scorer(distance, confidence):
    return _safe_div(distance, confidence - 1.2)
```

---

### 第 7 步：IR 变异与交叉

#### 变异 (Mutation)

```python
from evolution.operators import mutate_ir

original_ir = compile_source_to_ir("""
def scorer(distance, confidence):
    return distance + confidence
""", "scorer")

print("原始:", emit_python_source(original_ir))

# 点变异：随机替换一个运算符
mutated = mutate_ir(original_ir, rng, mutation_type="point")
print("点变异:", emit_python_source(mutated))

# 常量扰动：微调一个常量值
ir_with_const = compile_source_to_ir("""
def scorer(distance, confidence):
    return distance * 2.5 + confidence
""", "scorer")
perturbed = mutate_ir(ir_with_const, rng, mutation_type="constant_perturb")
print("常量扰动:", emit_python_source(perturbed))
```

**输出示例：**
```
原始:     def scorer(distance, confidence): return distance + confidence
点变异:   def scorer(distance, confidence): return distance * confidence
常量扰动: def scorer(distance, confidence): return distance * 2.317 + confidence
```

#### 交叉 (Crossover)

```python
from evolution.operators import crossover_ir

parent1 = compile_source_to_ir("""
def scorer(distance, confidence):
    return distance + confidence * 3.0
""", "scorer")

parent2 = compile_source_to_ir("""
def scorer(distance, confidence):
    return distance - confidence * 1.5
""", "scorer")

child = crossover_ir(parent1, parent2, rng)
print("父代1:", emit_python_source(parent1))
print("父代2:", emit_python_source(parent2))
print("子代: ", emit_python_source(child))
```

**输出示例：**
```
父代1: def scorer(distance, confidence): return distance + confidence * 3.0
父代2: def scorer(distance, confidence): return distance - confidence * 1.5
子代:  def scorer(distance, confidence): return distance - confidence * 1.5
```

交叉策略：取父代1的结构（CFG 图），用父代2的运算符和常量随机替换。

---

### 第 8 步：构建基因组 (Genome)

`IRGenome` 是进化的基本单位，包含多个命名程序和可进化常量：

```python
from evolution.genome import IRGenome

# 手动构建一个基因组
programs = {
    "scorer": compile_source_to_ir("""
def scorer(distance, confidence):
    return distance + confidence
""", "scorer"),
    "halter": compile_source_to_ir("""
def halter(old_score, new_score):
    return new_score - old_score
""", "halter"),
}

genome = IRGenome(
    programs=programs,
    constants=np.array([1.0, 0.5, -0.3, 2.0]),  # 可进化常量
    generation=0,
)

# 生成 Python 源码
print(genome.to_source("scorer"))
# 输出: def scorer(distance, confidence): return distance + confidence

# 生成 C++ 操作码
cpp_ops = genome.to_cpp_ops("scorer")
print(f"C++ opcodes: {cpp_ops}")

# 编译为可直接调用的 Python 函数
fn = genome.to_callable("scorer")
result = fn(3.5, 1.2)
print(f"scorer(3.5, 1.2) = {result}")
# 输出: scorer(3.5, 1.2) = 4.7

# 克隆（深拷贝）
genome2 = genome.clone()

# 序列化为 JSON
data = genome.serialize()
print(f"序列化字段: {list(data.keys())}")
# 输出: 序列化字段: ['genome_id', 'generation', 'parent_ids', 'programs', 'constants']

# 结构哈希（用于多样性控制）
print(f"结构哈希: {genome.structural_hash()}")
```

---

### 第 9 步：定义适应度评估器

继承 `FitnessEvaluator` 抽象基类，实现 `evaluate` 方法：

```python
from evolution.fitness import FitnessResult, FitnessEvaluator
from evolution.genome import IRGenome

class MyEvaluator(FitnessEvaluator):
    """评估 scorer 程序的质量。"""

    def evaluate(self, genome: IRGenome) -> FitnessResult:
        try:
            scorer = genome.to_callable("scorer")

            # 在一些测试数据上评估
            test_data = [(1.0, 2.0, 3.0), (0.5, 0.5, 1.0), (3.0, 1.0, 4.0)]
            total_error = 0.0
            for d, c, expected in test_data:
                predicted = scorer(d, c)
                total_error += abs(predicted - expected)

            avg_error = total_error / len(test_data)

            return FitnessResult(
                metrics={"error": avg_error},
                weights={"error": 1.0},
                is_valid=True,
            )
        except Exception:
            return FitnessResult(metrics={"error": 999.0}, is_valid=False)

evaluator = MyEvaluator()
result = evaluator.evaluate(genome)
print(f"适应度分数: {result.composite_score():.4f}")
print(f"各项指标: {result.metrics}")
```

`composite_score()` 返回所有 `metrics[k] * weights[k]` 的加权和。**越低越好。**

---

### 第 10 步：运行进化引擎

```python
from evolution.config import EvolutionConfig
from evolution.engine import EvolutionEngine

config = EvolutionConfig(
    population_size=30,
    n_generations=50,
    seed=42,
    tournament_size=3,
    elite_count=2,
    mutation_rate=0.8,
    crossover_rate=0.3,
    program_roles=["scorer", "halter"],
    metric_weights={"error": 1.0},
)

engine = EvolutionEngine(config, evaluator, registry)

# 可以用回调追踪进度
def on_generation(gen, best_fit, pop):
    if gen % 10 == 0:
        print(f"  Gen {gen:3d}: best_score = {best_fit.composite_score():.6f}")

best_genome = engine.run(callback=on_generation)

# 查看最佳结果
print(f"\n最佳基因组:")
for role in config.program_roles:
    print(f"  {role}: {best_genome.to_source(role)}")
print(f"  适应度: {engine.best_fitness.composite_score():.6f}")
```

**可选：注入种子基因组**

```python
seed = IRGenome(programs={
    "scorer": compile_source_to_ir("def scorer(d, c): return d + c", "scorer"),
    "halter": compile_source_to_ir("def halter(o, n): return n - o", "halter"),
})
best = engine.run(seed_genomes=[seed])
```

---

## 应用示例：MIMO BP 检测器进化

`applications/mimo_bp/` 实现了一个完整的 MIMO 信号检测器进化应用。

### 问题描述

在 16×16 天线、16QAM 调制的 MIMO 系统中，进化4个控制 BP（Belief Propagation）树搜索行为的小程序：

| 角色 | 功能 | 参数 |
|---|---|---|
| `f_down` | 下行消息传递 | `parent_m_down`, `local_dist` |
| `f_up` | 上行消息聚合 | `sum_child_ld`, `sum_child_m_up`, `n_children` |
| `f_belief` | 信念评分 | `cum_dist`, `m_down`, `m_up` |
| `h_halt` | 早停判断 | `old_root_m_up`, `new_root_m_up` |

### 运行进化

```bash
cd research/algorithm-IR
python applications/mimo_bp/run_evolution.py \
    --pop-size 80 \
    --generations 300 \
    --n-train 300 \
    --n-test 100 \
    --snr-db 24 \
    --max-nodes 500 \
    --seed 42
```

### 验证结果

```bash
python applications/mimo_bp/validate_large.py
```

输出示例（2000样本/SNR点）：

```
=== Validation: Best Evolved ===
SNR(dB)  BER(evolved)  BER(LMMSE)  Gain(dB)
 16.0    0.013969      0.050344    +5.57
 18.0    0.004094      0.032063    +8.94
 20.0    0.001844      0.022594    +10.88
 22.0    0.002063      0.011938    +7.63
 24.0    0.001000      0.008438    +9.26   ← 目标: ≤0.002 ✓
 26.0    0.000000      0.005812    +∞
```

---

## C++ 后端

### 栈式求值器 (`ir_eval.h`)

33 个操作码的超轻量 C++ 执行器，用 `double` 栈执行 IR 操作码数组：

| 操作码 | 值 | 行为 |
|---|---|---|
| `CONST_F64` | 0 | 推入一个 float64 常量（后跟2个int32编码） |
| `LOAD_ARG` | 1 | 推入第 N 个函数参数 |
| `ADD` | 2 | 弹出两个值，推入它们的和 |
| `SUB` | 3 | 弹出两个值，推入差 |
| `MUL` | 4 | 弹出两个值，推入积 |
| `SAFE_DIV` | 25 | 安全除法（分母太小则返回0） |
| `SAFE_LOG` | 26 | 安全对数（参数≤0时取极小值） |
| `SAFE_SQRT` | 27 | 安全平方根（参数<0时取0） |
| `IF_START` | 19 | 弹出条件值，如果为假则跳到 `ELSE` |
| `ELSE` | 20 | 跳到 `ENDIF` |
| `ENDIF` | 21 | 分支结束标记 |
| `RETURN` | 24 | 返回栈顶值 |

### 编译 DLL

```bash
cd research/algorithm-IR/applications/mimo_bp/cpp
build.bat
```

需要 Visual Studio 2022 的 MSVC 编译器。生成 `bp_ir_eval.dll`。

### Python 调用 C++ DLL

```python
from applications.mimo_bp.cpp_evaluator import CppBPIREvaluator, ir_eval_expr_cpp

# 直接求值一个表达式
ops = [1, 0, 1, 1, 2, 24]  # LOAD_ARG 0, LOAD_ARG 1, ADD, RETURN
result = ir_eval_expr_cpp(ops, args=[3.0, 4.0])
print(f"3.0 + 4.0 = {result}")  # 输出: 7.0
```

---

## API 参考

### `algorithm_ir.frontend`

| 函数 | 签名 | 描述 |
|---|---|---|
| `compile_function_to_ir` | `(fn: function) → FunctionIR` | 编译 Python 函数对象 |
| `compile_source_to_ir` | `(source: str, func_name: str, globals_dict: dict = None) → FunctionIR` | 编译 Python 源码字符串 |

### `algorithm_ir.ir`

| 函数/类 | 描述 |
|---|---|
| `FunctionIR` | 核心 IR 对象，包含 `.values`, `.ops`, `.blocks`, `.arg_values` |
| `Value` | 数据值：`id`, `name_hint`, `type_hint`, `def_op`, `use_ops`, `attrs` |
| `Op` | 操作：`id`, `opcode`, `inputs`, `outputs`, `block_id`, `attrs` |
| `Block` | 基本块：`id`, `op_ids`, `preds`, `succs` |
| `render_function_ir(ir)` | 返回 IR 的可读文本 |
| `validate_function_ir(ir)` | 验证 IR 结构完整性，返回错误列表 |

### `algorithm_ir.regeneration.codegen`

| 函数/类 | 描述 |
|---|---|
| `emit_python_source(ir) → str` | IR → Python 源码 |
| `emit_cpp_ops(ir) → list[int]` | IR → C++ 操作码数组 |
| `CppOp` | 33 个 C++ 操作码常量 |

### `evolution`

| 类/函数 | 描述 |
|---|---|
| `EvolutionConfig` | 所有超参数：种群大小、变异率、锦标赛大小等 |
| `FitnessResult` | 适应度结果：`metrics`, `weights`, `is_valid`, `composite_score()` |
| `FitnessEvaluator` | 抽象基类，需实现 `evaluate(genome) → FitnessResult` |
| `IRGenome` | 进化个体：`programs`, `constants`, `to_source()`, `to_cpp_ops()`, `to_callable()` |
| `ProgramSpec` | 程序规格：`name`, `param_names`, `param_types`, `return_type` |
| `SkeletonSpec` | 骨架规格：`skeleton_id`, `program_specs` |
| `SkeletonRegistry` | 注册/查询/验证骨架 |
| `random_ir_program(spec, rng)` | 按规格生成随机 IR 程序 |
| `mutate_ir(ir, rng)` | IR 级变异 |
| `crossover_ir(ir1, ir2, rng)` | IR 级交叉 |
| `EvolutionEngine` | 主引擎：`run(callback=..., seed_genomes=...)` |

---

## 运行测试

```bash
cd research/algorithm-IR

# 全部测试 (106 个)
python -m pytest tests/ -v

# 仅单元测试
python -m pytest tests/unit/ -v

# 仅跨语言一致性测试
python -m pytest tests/cross_lang/ -v

# 运行交互式教程
python ir_tutorial_demo.py
```

---

## 端到端流程图

```
┌──────────────┐     compile_source_to_ir()      ┌──────────────┐
│ Python 源码  │ ──────────────────────────────→  │  FunctionIR  │
│ def f(a, b): │                                  │  (SSA 图)    │
│   return a+b │                                  └──────┬───────┘
└──────────────┘                                         │
                                                    ┌────┴────┐
                                    emit_python_    │         │  emit_cpp_
                                     source()       │         │   ops()
                                         │          │         │      │
                                         ▼          │         │      ▼
                                    ┌─────────┐     │         │ ┌──────────┐
                                    │ Python  │     │         │ │ C++ Ops  │
                                    │ source  │     │         │ │ [1,0,1,  │
                                    └─────────┘     │         │ │  1,2,24] │
                                                    │         │ └──────────┘
                                                    │         │      │
                              random_ir_program()───┘         │      │
                                                              │      │
                                    ┌─────────────────────────┘      │
                                    │                                │
                                    ▼                                ▼
                              ┌──────────┐                    ┌──────────┐
                              │ IRGenome │                    │ C++ DLL  │
                              │ programs │                    │ ir_eval  │
                              │ constants│                    │ (栈求值) │
                              └─────┬────┘                    └──────────┘
                                    │
                     ┌──────────────┤
          mutate_    │              │  crossover_
          genome()   │              │  genome()
                     ▼              ▼
              ┌────────────────────────┐
              │   EvolutionEngine      │
              │   ├─ init_population() │
              │   ├─ evaluate_batch()  │
              │   ├─ breed + select    │
              │   └─ run() → best      │
              └────────────────────────┘
```
