<p align="center">
  <h1 align="center">AlphaDetect</h1>
  <p align="center"><b>基于神经符号自动推理的可解释MIMO检测算法发现</b></p>
  <p align="center">
    <a href="README.md">English</a> | <a href="README_CN.md">中文</a>
  </p>
</p>

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
