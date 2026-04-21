<p align="center">
  <h1 align="center">AlphaDetect</h1>
  <p align="center"><b>基于 IR 进化搜索的 MIMO 检测算法自动发现框架</b></p>
  <p align="center">
    <a href="README.md">English</a> | <a href="README_CN.md">中文</a>
  </p>
</p>

---

## 概述

**AlphaDetect** 是一个面向 **MIMO 检测算法自动发现**的研究项目，核心方法是将现有检测器实现（LMMSE、ZF、OSIC、K-Best、BP、EP、AMP 等）编译为结构中立的 SSA 中间表示（IR），再通过**双层进化引擎**进行跨算法族结构嫁接与槽位微变异，从而自动发现新型检测器。

> **当前状态**：活跃开发中。`research/algorithm-IR/` 文件夹包含完整框架——IR 编译器、带 GNN 引导嫁接的进化引擎、8 个基线检测器、Monte Carlo MIMO 评估器以及 242 个测试。

## 研究愿景

核心研究问题：

> *机器能否系统性地发现算法设计中的概念创新，而不仅仅是在固定算法空间内进行优化？*

当前最先进的 MIMO 检测器（ZF、MMSE、V-BLAST、球形解码、K-Best、BP、GTA、EP、AMP 等）是人类研究者经过数十年工作发现的。每一次突破不仅需要数学推导，还需要**发明新概念**：将检测问题表示为图上的概率推断、将搜索空间建模为树、或引入空穴分布（cavity distribution）等。

AlphaDetect 的 Algorithm-IR 框架通过以下方式实现**结构性自动创新**：

1. **IR 编译** — 将 Python 检测器函数编译为 SSA IR（`FunctionIR`），使计算结构可被显式操作。
2. **骨架嫁接** — 跨算法族结构迁移：如将 LMMSE 预滤波注入 K-Best 树搜索、将 EP 腔计算替换为 BP 消息扫描——均在 IR 层通过 `graft_general()` 执行。
3. **双层进化** — 宏观层进化骨架结构（嫁接），微观层进化每个槽位的实现（IR 变异/交叉）。
4. **GNN 引导提议** — 图注意力网络（GAT）通过 REINFORCE 在线学习哪些宿主-供体对能产生有效检测器，以学习到的结构直觉替代随机嫁接。
5. **代码物化** — 任何进化后的 `AlgorithmGenome` 都能被物化为可执行的 Python 源码。

完整技术细节请参阅 [research/research-proposal/research_proposal.tex](research/research-proposal/research_proposal.tex)。

## 项目结构

```
AlphaDetect/
├── .github/                        # 用于 vibe-coding 研究的 AI 智能体系统
│   ├── agents/                     # 8 个专业化智能体定义
│   │   ├── orchestrator.agent.md   # 中央研究协调者
│   │   ├── code-generation.agent.md
│   │   ├── experiment.agent.md
│   │   ├── ideator.agent.md
│   │   ├── literature-search.agent.md
│   │   ├── math-deduction.agent.md
│   │   ├── paper-writing.agent.md
│   │   └── review.agent.md
│   ├── instructions/               # 编码规范 & 记忆协议
│   └── skills/                     # 领域特定技能模块
├── AGENTS.md                       # 智能体系统概述 & 规范
├── research/
│   ├── memory/                     # 持久化研究记忆
│   │   ├── state.json              # 当前研究线程 & 阶段
│   │   ├── experiment-log.md       # 按时间排列的实验记录
│   │   ├── idea-bank.md            # 研究想法及状态追踪
│   │   ├── decision-history.md     # 重大决策及其理由
│   │   └── experience-base.md      # 经验教训 & 模式总结
│   ├── research-proposal/          # 核心研究提案（LaTeX）
│   │   ├── research_proposal.tex   # 完整 AlphaDetect 愿景（英文）
│   │   └── research_proposal_cn.tex  # 中文版本
│   └── algorithm-IR/              # ★ 主要研究实现
│       ├── algorithm_ir/           #   IR 编译器、嫁接、运行时
│       ├── evolution/              #   双层进化引擎
│       ├── tests/                  #   242 个测试（单元 + 集成）
│       ├── e2e_experiment.py       #   端到端进化实验入口
│       ├── train_gnn.py            #   GNN 模式匹配器训练
│       ├── baseline_eval.py        #   基线检测器评估
│       └── osic_sweep.py           #   OSIC 基线 SNR 扫描
└── code_for_reference/             # 外部参考代码（AutoML-Zero 等）
```

### AI 智能体系统（`.github/`）

本项目使用基于 VS Code Copilot 自定义智能体团队驱动的 **vibe-coding 研究**工作流。一个**协调者（Orchestrator）**智能体协调 7 个专业化子智能体：

| 智能体 | 职责 |
|--------|------|
| **Orchestrator（协调者）** | 中央研究主管——加载研究状态、做出战略决策、委派子智能体、维护记忆 |
| **Ideator（创意生成）** | 生成新颖的研究想法、进行差距分析、评估新颖性 |
| **Literature Search（文献搜索）** | 搜索 arXiv/IEEE/Scholar、下载 PDF、构建文献综述 |
| **Code Generation（代码生成）** | 编写仿真代码（Python/C++）、实现算法 |
| **Experiment（实验）** | 编译、运行和分析仿真；收集 BER/SER 结果 |
| **Math Deduction（数学推导）** | 执行严格的逐步数学推导 |
| **Paper Writing（论文撰写）** | 撰写 IEEE 格式 LaTeX 论文、创建 TikZ 图表 |
| **Review（评审）** | 模拟同行评审、评估新颖性和技术正确性 |

## Algorithm-IR：基于 IR 的 MIMO 检测器进化

`research/algorithm-IR/` 是主要实现。它将 Python MIMO 检测器编译为 SSA IR，进行结构进化，并通过 Monte Carlo 仿真评估适应度。

### 架构概览

```
Python 检测器 (LMMSE, ZF, OSIC, K-Best, BP, EP, AMP, Stack)
        ↓  compile_source_to_ir()
   FunctionIR  (SSA — Values, Ops, Blocks)
        ↓
   AlgorithmGenome = structural_ir + slot_populations
        ↓
   双层进化
   ├── 宏观层: graft_general()  ← GNN / 专家 / 静态模式匹配器
   └── 微观层: mutate_ir() / crossover_ir()  ← 槽位 IR 变异
        ↓
   MIMOFitnessEvaluator  (16×16 16-QAM Monte Carlo)
        ↓
   materialize()  →  可执行 Python 源码
```

### 8 个基线检测器

| 算法 | 标签 | 可进化槽位 |
|------|------|-----------|
| `lmmse` | `linear` | `slot_regularizer`, `slot_hard_decision` |
| `zf` | `linear` | `slot_hard_decision` |
| `osic` | `sic` | `slot_ordering`, `slot_sic_step` |
| `kbest` | `tree_search` | `slot_expand`, `slot_prune` |
| `stack` | `tree_search` | `slot_expand`, `slot_prune` |
| `bp` | `message_passing` | `slot_bp_sweep`, `slot_bp_final` |
| `ep` | `inference` | `slot_ep_site`, `slot_ep_final` |
| `amp` | `inference` | `slot_amp_denoise`, `slot_amp_final` |

### 快速开始

```bash
conda activate AutoGenOld
cd research/algorithm-IR

# 1. 评估所有 8 个基线检测器（16×16 16-QAM，SNR=24dB）
python baseline_eval.py

# 2. OSIC SNR 扫描（18–28 dB，每点 2000 次试验）
python osic_sweep.py

# 3. 端到端进化实验（50 代，Expert+Static+GNN 匹配器）
python e2e_experiment.py

# 4. GNN 引导训练器（大规模，每代 500 个嫁接提议）
python train_gnn.py [--gens 200] [--snr-start 20] [--snr-target 24] \
    [--proposals 500] [--pool-size 141] [--n-trials 5] [--timeout 1.5] \
    [--warmstart-gens 1] [--warmstart-eval-workers 8] [--seed 42] \
    [--resume path/to/checkpoint.pt]

# 5. 运行全部测试
python -m pytest tests/ -v   # 242 个测试，约 30 秒
```

### `train_gnn.py` 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--gens` | 200 | 总进化代数 |
| `--snr-start` | 20.0 | 初始训练 SNR（dB） |
| `--snr-target` | 24.0 | 目标 SNR（dB） |
| `--proposals` | 500 | 每代 GNN 嫁接提议数 |
| `--pool-size` | 141 | 种群规模（91 原始 + 50 嫁接幸存者） |
| `--n-trials` | 5 | 每个基因组的评估试验数（320 bits/trial） |
| `--timeout` | 1.5 | 每个基因组的评估超时（秒） |
| `--warmstart-gens` | 1 | 热启动代数（穷举宿主-供体对扫描） |
| `--warmstart-eval-workers` | 8 | 热启动评估的并行工作者数 |
| `--ckpt-interval` | 10 | 每 N 代保存 GNN 检查点 |
| `--seed` | 42 | 随机种子 |
| `--resume` | None | 从指定检查点路径恢复 |

输出写入 `results/gnn_training/`（JSONL 日志 + 检查点文件）。

### 环境要求

- Python 3.12+（`conda activate AutoGenOld`）
- `numpy`、`scipy`、`torch`、`torch_geometric`、`xdsl`
- `pytest` 用于运行测试套件

## 许可

本项目用于学术研究目的。

## 引用

如需引用本工作，请使用：

```
AlphaDetect: Explainable MIMO Detection Algorithm Discovery
via Neural-Symbolic Automated Reasoning. Research Proposal, April 2026.
```

---

## 概述

**AlphaDetect** 是一个研究项目，旨在构建一个统一框架，用于**自动发现可解释的MIMO检测算法**。与神经架构搜索或基于LLM的代码演化不同，AlphaDetect将算法发现建立在*形式推理*之上——发现的算法中每一步都可以追溯到形式推导、类型正确的原语组合或经过验证的代数变换。

最终愿景包含四个紧密集成的组件：

1. **类型化领域特定语言（DSL）**——使用精确的数学类型和算法组合子编码信号处理和概率推断原语。
2. **问题变换库**——将中间概念的发明（例如，将MIMO模型识别为因子图）转化为从类型化变换中生成见证（witness）的过程。
3. **计算机代数系统（CAS）**——在选定高层结构后，逐步执行算法更新方程的符号推导。
4. **LLM增强的结构映射引擎（SME）**——发现跨领域类比（例如，因子图推断与MIMO检测之间的映射），并提出新的概念假设。

这些组件共同解决算法发现的核心挑战——发明新的*中间概念*，这些概念无法简单归结为在固定概念空间内的搜索。

> **当前状态**：本项目处于活跃的早期研究阶段。`research/mimo-push-gp/` 文件夹包含一个概念验证性质的遗传编程系统，用于演化MIMO检测算法——这是迈向完整AlphaDetect愿景的一个基础步骤。

## 研究愿景

核心研究问题是：

> *机器能否系统性地发现算法设计中的概念创新，而不仅仅是在固定的算法空间内进行优化？*

当前最先进的MIMO检测器（ZF、MMSE、V-BLAST、球形解码、K-Best、BP、GTA、EP、AMP等）是人类研究者经过数十年工作发现的。每一次突破不仅需要数学推导，还需要**发明新概念**：将检测问题表示为图上的概率推断、将搜索空间建模为树、用Chow-Liu最小生成树近似含环因子图、或引入空穴分布（cavity distribution）等。

AlphaDetect的目标是**可解释的**算法发现。系统的工作流程为：

1. 通过LLM将MIMO问题和源域（如因子图推理）编码为谓词结构
2. 使用结构映射发现跨领域类比，生成*候选推断*
3. 在类型兼容性和复杂度约束的指导下，搜索DSL组合和问题变换的空间
4. 一旦找到高层结构，通过CAS推导精确的更新方程
5. 评估所发现算法的性能和复杂度，将成功的发现反馈回系统

完整技术细节请参阅 [research/research-proposal/research_proposal.tex](research/research-proposal/research_proposal.tex)。

## 项目结构

```
AlphaDetect/
├── .github/                        # 用于 vibe-coding 研究的AI智能体系统
│   ├── agents/                     # 8个专业化智能体定义
│   │   ├── orchestrator.agent.md   # 中央研究协调者
│   │   ├── code-generation.agent.md
│   │   ├── experiment.agent.md
│   │   ├── ideator.agent.md
│   │   ├── literature-search.agent.md
│   │   ├── math-deduction.agent.md
│   │   ├── paper-writing.agent.md
│   │   └── review.agent.md
│   ├── instructions/               # 编码规范 & 记忆协议
│   │   ├── cpp-simulation.instructions.md
│   │   ├── latex-writing.instructions.md
│   │   └── research-memory.instructions.md
│   ├── prompts/                    # 可复用的提示词模板
│   └── skills/                     # 领域特定技能模块
│       ├── data-analysis/          # 绘制BER曲线、生成表格
│       ├── literature-search/      # 搜索arXiv/IEEE、下载PDF
│       └── simulation-runner/      # 编译与运行C++/Python仿真
├── AGENTS.md                       # 智能体系统概述 & 规范
├── research/
│   ├── memory/                     # 持久化研究记忆
│   │   ├── state.json              # 当前研究线程 & 阶段
│   │   ├── experiment-log.md       # 按时间排列的实验记录
│   │   ├── idea-bank.md            # 研究想法及状态追踪
│   │   ├── decision-history.md     # 重大决策及其理由
│   │   └── experience-base.md      # 经验教训 & 模式总结
│   ├── research-proposal/          # 核心研究提案（LaTeX）
│   │   ├── research_proposal.tex   # 完整AlphaDetect愿景（英文）
│   │   └── research_proposal_cn.tex  # 中文版本
│   └── mimo-push-gp/              # 概念验证：GP演化MIMO检测
│       ├── code/                   # 实现代码（Python）
│       ├── results/                # 实验输出（JSON）
│       ├── papers/                 # 相关参考文献
│       └── logs/                   # 演化日志
└── code_for_reference/             # 外部参考代码（AutoML-Zero等）
```

### AI智能体系统（`.github/`）

本项目使用基于VS Code Copilot自定义智能体团队驱动的 **vibe-coding 研究** 工作流。一个**协调者（Orchestrator）**智能体协调7个专业化子智能体来进行端到端研究：

| 智能体 | 职责 |
|--------|------|
| **Orchestrator（协调者）** | 中央研究主管——加载研究状态、做出战略决策、委派子智能体、维护记忆 |
| **Ideator（创意生成）** | 生成新颖的研究想法、进行差距分析、评估新颖性 |
| **Literature Search（文献搜索）** | 搜索arXiv/IEEE/Scholar、下载PDF、构建文献综述 |
| **Code Generation（代码生成）** | 编写仿真代码（Python/C++）、实现算法 |
| **Experiment（实验）** | 编译、运行和分析仿真；收集BER/FER结果 |
| **Math Deduction（数学推导）** | 执行严格的逐步数学推导 |
| **Paper Writing（论文撰写）** | 撰写IEEE格式的LaTeX论文、创建TikZ图表 |
| **Review（评审）** | 模拟同行评审、评估新颖性和技术正确性 |

智能体共享一个持久化**记忆系统**（`research/memory/`），用于在会话之间跟踪研究状态、实验日志、想法、决策和经验教训。

## 概念验证：MIMO-Push-GP

`research/mimo-push-gp/` 文件夹包含一个可运行的概念验证系统，使用**遗传编程（Genetic Programming）**自动发现MIMO检测算法。虽然距离完整的AlphaDetect愿景（基于形式推理的发现）还很遥远，但这个GP系统证明了自动发现具有竞争力的检测算法的可行性。

### 工作原理

系统演化小程序来控制一个**信念传播（BP）增强的栈解码器**，用于MIMO信号检测：

```
MIMO信道 (y = Hx + n)
        ↓
   QR分解 (H = QR)
        ↓
   带有演化BP的树搜索
        ↓
   检测到的符号 (x̂)
```

**核心组件：**

- **Push风格虚拟机**（`vm.py`）：具有约80条指令的类型化栈式虚拟机，支持浮点数、整数、布尔值、向量、矩阵、节点和图类型
- **四程序基因组**：每个个体有四个演化程序，定义检测行为：

  | 程序 | 功能 |
  |------|------|
  | `F_down` | 下行扫描：将父节点消息传播给子节点 |
  | `F_up` | 上行扫描：聚合子节点消息给父节点 |
  | `F_belief` | 使用距离和BP消息为前沿节点评分 |
  | `H_halt` | 决定是否继续BP迭代 |

- **结构化BP解码器**（`bp_decoder_v2.py`）：带有全树BP消息传递的栈解码器
- **演化引擎**（`evolution.py`）：锦标赛选择、变异、交叉，具有停滞检测和硬重启机制

### 快速运行实验

```bash
conda activate AutoGenOld
cd research/mimo-push-gp/code

# 快速测试 (4×4 MIMO, QPSK, 5代演化)
python -u bp_main_v2.py --generations 5 --population 20 --train-samples 20 \
    --train-max-nodes 500 --train-flops-max 500000 --step-max 1000 \
    --train-snrs "24" --eval-snrs "20,24,28" --eval-trials 50 \
    --train-nt 4 --train-nr 4 --log-suffix quick_test --mod-order 4

# 完整实验 (16×16 MIMO, 16-QAM, 使用C++加速)
python -u bp_main_v2.py --generations 60 --population 100 --train-samples 80 \
    --train-max-nodes 1000 --train-flops-max 1200000 --step-max 1000 \
    --train-snrs "24,28" --eval-snrs "18,20,22,24,28" --eval-trials 500 \
    --eval-max-nodes 2000 --eval-flops-max 5000000 --eval-step-max 5000 \
    --train-nt 16 --train-nr 16 --log-suffix test_run --use-cpp
```

### 环境要求

- Python 3.8+，需安装 `numpy`、`scipy`、`matplotlib`、`tqdm`
- Conda环境：`AutoGenOld`
- （可选）C++编译器，用于加速适应度评估

## 许可

本项目用于学术研究目的。

## 引用

如需引用本工作，请使用：

```
AlphaDetect: Explainable MIMO Detection Algorithm Discovery
via Neural-Symbolic Automated Reasoning. Research Proposal, April 2026.
```
