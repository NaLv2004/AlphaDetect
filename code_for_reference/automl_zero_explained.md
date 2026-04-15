# AutoML-Zero 代码架构与算法详解

> 基于 [google-research/automl_zero](https://github.com/google-research/google-research/tree/master/automl_zero) 源码分析
> 论文: *AutoML-Zero: Evolving Machine Learning Algorithms From Scratch* (Real et al., 2020)

---

## 目录

1. [项目概述](#1-项目概述)
2. [整体架构](#2-整体架构)
3. [核心数据结构](#3-核心数据结构)
4. [元操作系统 (Operations)](#4-元操作系统-operations)
5. [算法表示 (Algorithm Representation)](#5-算法表示-algorithm-representation)
6. [执行引擎 (Executor)](#6-执行引擎-executor)
7. [进化搜索框架 (Regularized Evolution)](#7-进化搜索框架-regularized-evolution)
8. [变异系统 (Mutation)](#8-变异系统-mutation)
9. [评估系统 (Evaluator)](#9-评估系统-evaluator)
10. [任务系统 (Task)](#10-任务系统-task)
11. [功能等价缓存 (FEC Cache)](#11-功能等价缓存-fec-cache)
12. [训练预算 (Train Budget)](#12-训练预算-train-budget)
13. [搜索实验入口 (run_search_experiment)](#13-搜索实验入口-run_search_experiment)
14. [非平凡设计选择总结](#14-非平凡设计选择总结)
15. [文件清单](#15-文件清单)

---

## 1. 项目概述

AutoML-Zero 的目标是**从零开始自动发现机器学习算法**。不同于传统 NAS（只搜索网络结构），AutoML-Zero 同时搜索：
- **模型结构**（如何从输入计算预测）
- **学习策略**（如何从标签更新模型参数）
- **初始化方法**（如何初始化模型参数）

搜索空间仅包含基础数学运算（加减乘除、线性代数、三角函数等），不预设任何 ML 先验知识（如梯度下降、反向传播）。

关键结果：进化搜索能自动发现线性回归+梯度下降、2层神经网络+反向传播等算法。

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     run_search_experiment.cc                     │
│                         (实验入口/main)                          │
├────────┬──────────┬──────────┬───────────┬──────────────────────┤
│        │          │          │           │                      │
│  Generator   Mutator   Evaluator  RegularizedEvolution  FECCache│
│  (生成初始   (变异     (评估      (进化搜索               (功能等 │
│   种群)      算子)     适应度)    主循环)                  价缓存)│
│        │          │          │           │                      │
├────────┴──────────┴──────────┴───────────┴──────────────────────┤
│                        Executor<F>                               │
│              (执行 Algorithm 的 Setup/Predict/Learn)             │
├─────────────────────────────────────────────────────────────────┤
│                          Algorithm                               │
│          (三个组件函数: setup_, predict_, learn_)                 │
│          每个组件函数 = vector<Instruction>                       │
├─────────────────────────────────────────────────────────────────┤
│                        Instruction                               │
│              (op_, in1_, in2_, out_, data fields)                │
├─────────────────────────────────────────────────────────────────┤
│                        Memory<F>                                 │
│     scalar_[20] + vector_<F>[20] + matrix_<F×F>[20]             │
├─────────────────────────────────────────────────────────────────┤
│                     Task<F> / TaskInterface                      │
│        (训练/验证数据集, 支持多 epoch 和 shuffling)               │
└─────────────────────────────────────────────────────────────────┘
```

### 数据流

```
1. Generator 生成初始 Algorithm 种群 (NoOp 或 Random)
2. RegularizedEvolution 循环:
   a. 锦标赛选择 → 选出 parent
   b. Mutator 对 parent 做变异 → child
   c. Evaluator 在多个 Task 上评估 child:
      i.  创建 Executor, 执行 Setup (初始化)
      ii. 循环: 执行 Predict (前向), 计算 error, 执行 Learn (更新)
      iii. 在验证集上执行 Predict, 计算 fitness
   d. 用 child 替换种群中最老的个体
3. 搜索结束后: T_select 任务上评估最佳算法, T_final 任务上做最终测试
```

---

## 3. 核心数据结构

### 3.1 Memory<F> — 寄存器式内存模型

**文件**: `memory.h`

这是 AutoML-Zero 最核心的设计之一。每个被搜索的算法操作的不是变量名，而是**固定地址的寄存器**。

```cpp
template<FeatureIndexT F>
class Memory {
  std::array<Scalar, kMaxScalarAddresses> scalar_;     // 默认 20 个标量寄存器
  std::array<Vector<F>, kMaxVectorAddresses> vector_;  // 默认 20 个向量寄存器
  std::array<Matrix<F>, kMaxMatrixAddresses> matrix_;  // 默认 20 个矩阵寄存器
};
```

**非平凡设计点**：
- **三种类型分离**：标量、向量、矩阵各有独立地址空间，而非统一内存。这简化了指令编码并保证类型安全。
- **地址数量可编译期配置**：通过 `MAX_SCALAR_ADDRESSES` 等宏在编译时确定，不同实验可使用不同大小（如 demo 用 4/3/1，完整实验用 20/20/20）。
- **特殊地址约定**：
  - `scalar_[0]` = 标签（labels），`scalar_[1]` = 预测值（predictions）
  - `vector_[0]` = 特征向量（features）
  - 这些是系统自动写入/读取的，搜索到的算法通过操作这些地址与训练框架交互。
- **模板参数 F**：特征向量维度在编译期确定（支持 2, 4, 8, 16, 32），向量和矩阵的维度随之确定。这是为了性能（Eigen 固定大小矩阵高度优化）。

### 3.2 Instruction — 指令

**文件**: `instruction.h`, `instruction.cc`

```cpp
class Instruction {
  Op op_;                    // 操作码 (枚举, 65种)
  AddressT in1_;             // 第一输入地址
  AddressT in2_;             // 第二输入地址
  AddressT out_;             // 输出地址
  double activation_data_;   // 标量常数数据
  float float_data_0_;       // 浮点数据0 (用于向量/矩阵索引或参数)
  float float_data_1_;       // 浮点数据1
  float float_data_2_;       // 浮点数据2
};
```

**非平凡设计点**：
- **immutable shared_ptr 设计**：指令一旦创建就不可修改，用 `shared_ptr<const Instruction>` 存储。变异时创建新指令而非修改原指令。这使得多个 Algorithm 可以安全共享相同的指令对象（浅拷贝），大幅减少内存分配。
- **float 索引编码**：向量/矩阵的坐标存储为 `[0,1)` 范围的 float，运行时按 `floor(size * float_value)` 转换为实际索引。这样同一个算法可以在不同维度 F 上运行而无需修改。
- **数据突变策略**：对数值参数使用**对数尺度高斯扰动**（`MutateActivationLogScale`），即 `x' = exp(log(x) + N(0,1))`，以及 10% 概率的符号翻转。这确保突变在数量级上均匀分布。

### 3.3 Algorithm — 被搜索的算法

**文件**: `algorithm.h`, `algorithm.cc`

```cpp
class Algorithm {
  std::vector<std::shared_ptr<const Instruction>> setup_;    // 初始化函数
  std::vector<std::shared_ptr<const Instruction>> predict_;  // 预测函数
  std::vector<std::shared_ptr<const Instruction>> learn_;    // 学习函数
};
```

每个 Algorithm 代表一个完整的 ML 算法，由三个**组件函数**（component functions）组成：
1. **Setup**: 执行一次，用于初始化参数（如权重矩阵）
2. **Predict**: 每个样本执行一次，从特征计算预测值
3. **Learn**: 每个训练样本执行一次（在 Predict 之后），用于更新参数

**非平凡设计点**：
- **三函数分解**：将 ML 算法分解为初始化/前向/学习三个独立函数，是关键的归纳偏置（inductive bias）。这不是从数据中学到的，而是人为指定的结构。然而这个偏置非常最小——它只假设算法有初始化阶段和训练循环，不假设具体的模型结构或学习规则。
- **浅拷贝**：Algorithm 的拷贝构造函数执行**浅拷贝**（只复制 shared_ptr，不复制 Instruction 内容），只有变异时才创建新的 Instruction 对象。这在大种群下节省大量内存和时间。

---

## 4. 元操作系统 (Operations)

**文件**: `instruction.proto`（Op 枚举定义），`executor.h`（执行实现）

AutoML-Zero 定义了 **65 种基本操作**（Op 0~64），涵盖三种数据类型（标量/向量/矩阵）的完整数学运算集合：

### 4.1 标量操作 (Scalar Ops, 19种)

| Op ID | 名称 | 语义 | 类别 |
|-------|------|------|------|
| 0 | NO_OP | 无操作 | 控制 |
| 1 | SCALAR_SUM_OP | s[out] = s[in1] + s[in2] | 算术 |
| 2 | SCALAR_DIFF_OP | s[out] = s[in1] - s[in2] | 算术 |
| 3 | SCALAR_PRODUCT_OP | s[out] = s[in1] × s[in2] | 算术 |
| 4 | SCALAR_DIVISION_OP | s[out] = s[in1] / s[in2] | 算术 |
| 5 | SCALAR_ABS_OP | s[out] = \|s[in1]\| | 算术 |
| 6 | SCALAR_RECIPROCAL_OP | s[out] = 1/s[in1] | 算术 |
| 7 | SCALAR_SIN_OP | s[out] = sin(s[in1]) | 三角函数 |
| 8 | SCALAR_COS_OP | s[out] = cos(s[in1]) | 三角函数 |
| 9 | SCALAR_TAN_OP | s[out] = tan(s[in1]) | 三角函数 |
| 10 | SCALAR_ARCSIN_OP | s[out] = arcsin(s[in1]) | 三角函数 |
| 11 | SCALAR_ARCCOS_OP | s[out] = arccos(s[in1]) | 三角函数 |
| 12 | SCALAR_ARCTAN_OP | s[out] = arctan(s[in1]) | 三角函数 |
| 13 | SCALAR_EXP_OP | s[out] = exp(s[in1]) | 微积分 |
| 14 | SCALAR_LOG_OP | s[out] = log(s[in1]) | 微积分 |
| 15 | SCALAR_HEAVYSIDE_OP | s[out] = (s[in1] ≥ 0) ? 1 : 0 | 非线性 |
| 44 | SCALAR_MIN_OP | s[out] = min(s[in1], s[in2]) | 比较 |
| 47 | SCALAR_MAX_OP | s[out] = max(s[in1], s[in2]) | 比较 |
| 56 | SCALAR_CONST_SET_OP | s[out] = activation_data | 常量设置 |
| 59 | SCALAR_UNIFORM_SET_OP | s[out] ~ U(float0, float1) | 随机初始化 |
| 62 | SCALAR_GAUSSIAN_SET_OP | s[out] ~ N(float0, float1) | 随机初始化 |

### 4.2 向量操作 (Vector Ops, 20种)

| Op ID | 名称 | 语义 |
|-------|------|------|
| 16 | VECTOR_HEAVYSIDE_OP | v[out] = (v[in1] > 0) ? 1 : 0 (逐元素) |
| 18 | SCALAR_VECTOR_PRODUCT_OP | v[out] = s[in1] × v[in2] |
| 19 | SCALAR_BROADCAST_OP | v[out] = s[in1] × ones |
| 20 | VECTOR_RECIPROCAL_OP | v[out] = 1/v[in1] (逐元素) |
| 21 | VECTOR_NORM_OP | s[out] = ‖v[in1]‖ |
| 22 | VECTOR_ABS_OP | v[out] = \|v[in1]\| (逐元素) |
| 23 | VECTOR_SUM_OP | v[out] = v[in1] + v[in2] |
| 24 | VECTOR_DIFF_OP | v[out] = v[in1] - v[in2] |
| 25 | VECTOR_PRODUCT_OP | v[out] = v[in1] ⊙ v[in2] (Hadamard) |
| 26 | VECTOR_DIVISION_OP | v[out] = v[in1] / v[in2] (逐元素) |
| 27 | VECTOR_INNER_PRODUCT_OP | s[out] = v[in1]ᵀ v[in2] |
| 28 | VECTOR_OUTER_PRODUCT_OP | M[out] = v[in1] v[in2]ᵀ |
| 45 | VECTOR_MIN_OP | v[out] = min(v[in1], v[in2]) (逐元素) |
| 48 | VECTOR_MAX_OP | v[out] = max(v[in1], v[in2]) (逐元素) |
| 50 | VECTOR_MEAN_OP | s[out] = mean(v[in1]) |
| 54 | VECTOR_ST_DEV_OP | s[out] = std(v[in1]) |
| 57 | VECTOR_CONST_SET_OP | v[out][idx] = float1 (idx由float0决定) |
| 60 | VECTOR_UNIFORM_SET_OP | v[out] ~ U(float0, float1) |
| 63 | VECTOR_GAUSSIAN_SET_OP | v[out] ~ N(float0, float1) |

### 4.3 矩阵操作 (Matrix Ops, 25种)

| Op ID | 名称 | 语义 |
|-------|------|------|
| 17 | MATRIX_HEAVYSIDE_OP | M[out] = (M[in1] > 0) ? 1 : 0 |
| 29 | SCALAR_MATRIX_PRODUCT_OP | M[out] = s[in1] × M[in2] |
| 30 | MATRIX_RECIPROCAL_OP | M[out] = 1/M[in1] (逐元素) |
| 31 | MATRIX_VECTOR_PRODUCT_OP | v[out] = M[in1] × v[in2] |
| 32 | VECTOR_COLUMN_BROADCAST_OP | M[out] = v[in1] 广播为列 |
| 33 | VECTOR_ROW_BROADCAST_OP | M[out] = v[in1]ᵀ 广播为行 |
| 34 | MATRIX_NORM_OP | s[out] = ‖M[in1]‖_F |
| 35 | MATRIX_COLUMN_NORM_OP | v[out] = 列范数 |
| 36 | MATRIX_ROW_NORM_OP | v[out] = 行范数 |
| 37 | MATRIX_TRANSPOSE_OP | M[out] = M[in1]ᵀ |
| 38 | MATRIX_ABS_OP | M[out] = \|M[in1]\| |
| 39 | MATRIX_SUM_OP | M[out] = M[in1] + M[in2] |
| 40 | MATRIX_DIFF_OP | M[out] = M[in1] - M[in2] |
| 41 | MATRIX_PRODUCT_OP | M[out] = M[in1] ⊙ M[in2] (Hadamard) |
| 42 | MATRIX_DIVISION_OP | M[out] = M[in1] / M[in2] (逐元素) |
| 43 | MATRIX_MATRIX_PRODUCT_OP | M[out] = M[in1] × M[in2] |
| 46 | MATRIX_MIN_OP | M[out] = min(M[in1], M[in2]) |
| 49 | MATRIX_MAX_OP | M[out] = max(M[in1], M[in2]) |
| 51 | MATRIX_MEAN_OP | s[out] = mean(M[in1]) |
| 52 | MATRIX_ROW_MEAN_OP | v[out] = row_mean(M[in1]) |
| 53 | MATRIX_ROW_ST_DEV_OP | v[out] = row_std(M[in1]) |
| 55 | MATRIX_ST_DEV_OP | s[out] = std(M[in1]) |
| 58 | MATRIX_CONST_SET_OP | M[out][i,j] = float2 |
| 61 | MATRIX_UNIFORM_SET_OP | M[out] ~ U(float0, float1) |
| 64 | MATRIX_GAUSSIAN_SET_OP | M[out] ~ N(float0, float1) |

### 4.4 操作的分类考量

**非平凡设计点**：
- **类型安全的跨类型操作**：如 `SCALAR_VECTOR_PRODUCT_OP` 从标量地址读输入，写入向量地址；`VECTOR_INNER_PRODUCT_OP` 从向量地址读，写入标量地址。操作的输入输出自然跨越不同类型空间。
- **随机操作**：`GAUSSIAN_SET_OP` 和 `UNIFORM_SET_OP` 类操作在**每次执行时产生不同的随机数**（使用与算法执行绑定的 `RandomGenerator`）。这意味着搜索到的算法可以包含随机性（如 dropout 或数据增强），但也增加了评估的随机性。
- **不包含控制流**：没有 if/else、循环等控制流指令。所有组件函数都是线性指令序列。这大大简化了搜索空间，但仍然通过 Heaviside 阶跃函数和 min/max 实现了有限的条件逻辑。
- **未做安全保护**：除法运算不检查除零，log 不检查非正输入。算法可能产生 NaN/Inf，由 Executor 的 early stopping 机制处理。这是有意为之——让搜索空间保持简单，由适应度过滤不良算法。

---

## 5. 算法表示 (Algorithm Representation)

### 5.1 组件函数的语义

```
Setup()    — 执行一次: 初始化权重、学习率等持久变量
             输入: 无 (memory 全零)
             输出: memory 中初始化好的变量

Predict()  — 每个样本执行一次: 从 features 计算 prediction
             输入: vector_[0] = features, scalar_[0] = 0 (labels清零)
             输出: scalar_[1] = prediction

Learn()    — 每个训练样本执行一次: 利用 label 更新 memory
             输入: vector_[0] = features, scalar_[0] = label
             输出: 对 memory 的任意修改 (权重更新)
```

### 5.2 执行模型

这个执行模型非常像一个**三段式虚拟机**:

```
Memory = new Memory()        // 全零初始化
Execute(Setup)               // 运行一次
for each training_example:
    memory.vector[0] = features
    memory.scalar[0] = 0     // ← 注意: Predict 看不到 label!
    Execute(Predict)
    error = |label - memory.scalar[1]|
    if error > max_error: early_stop
    memory.vector[0] = features
    memory.scalar[0] = label  // ← Learn 可以看到 label
    Execute(Learn)
// Validation:
for each valid_example:
    memory.vector[0] = features
    Execute(Predict)
    accumulate loss from memory.scalar[1]
fitness = FlipAndSquash(loss)
```

**非平凡设计点**：
- **Predict 看不到 label**: 标签地址在 Predict 前被清零，防止直接作弊（把 label 原样抄到 prediction）。
- **Learn 可读 Predict 的所有副作用**: Learn 函数执行时，memory 保留了 Predict 执行后的所有中间结果。这意味着 Learn 可以访问前向传播的中间激活值——这正是反向传播所需的。
- **memory 跨样本持久化**: Setup 写入的值（如权重矩阵）在所有训练样本间持久保留，Learn 对 memory 的修改也会保留到下一个样本。这使得累积式学习（如梯度下降）成为可能。

---

## 6. 执行引擎 (Executor)

**文件**: `executor.h`（~1350行，全在 header 中实现）

Executor 是性能关键路径——每个个体的评估都需要执行数千次指令。

### 6.1 指令分派机制

```cpp
// 函数指针查找表, 编译期构建, 128项
template<FeatureIndexT F>
static constexpr std::array<
    void(*)(const Instruction&, RandomGenerator*, Memory<F>*), 128>
    kOpIndexToExecuteFunction = { ... };

// 单条指令执行: 直接索引函数指针表
template<FeatureIndexT F>
inline void ExecuteInstruction(const Instruction& instruction,
                               RandomGenerator* rand_gen, Memory<F>* memory) {
  (*kOpIndexToExecuteFunction<F>[instruction.op_])(instruction, rand_gen, memory);
}
```

**非平凡设计点**：
- **编译期函数指针数组**: 将 Op 枚举值直接作为数组索引，避免了 switch-case（编译器可能生成跳转表，但不保证）。128 个槽位中只有 65 个有效，其余指向 `ExecuteUnsupportedOp`（LOG(FATAL)）。
- **全部内联**: 所有 Execute*Op 函数都是 `inline`，且 Executor 完全在 `.h` 文件中实现。这允许编译器在 `-O2`/`-Ofast` 下积极内联，最大化热循环性能。
- **模板特化 F**: 每种特征维度 F 生成独立的 Executor 实例（通过 `switch(task.FeaturesSize())` 分派）。这个编译期多态避免了运行时维度检查，并让 Eigen 使用固定大小矩阵的 SIMD 优化。

### 6.2 训练优化 (TrainOptImpl)

当训练步数 ≥ 1000 时，Executor 将指令**从 `vector<shared_ptr<const Instruction>>` 拷贝到 `std::array<Instruction, N>`**:

```cpp
template <size_t max_component_function_size>
bool TrainOptImpl(IntegerT max_steps, ...) {
    std::array<Instruction, max_component_function_size> optimized_predict;
    // 拷贝到栈上连续数组
    for (...) optimized_predict[i] = *algorithm_.predict_[i];
    // 训练循环中直接遍历连续数组
    for (const Instruction& instr : optimized_predict) { ... }
}
```

**非平凡设计点**：
- **消除间接寻址**: shared_ptr 需要两次间接访问（ptr→控制块→对象），栈上数组是连续存储，cache 友好。
- **编译期上界**: `max_component_function_size` 是模板参数（10/100/1000），编译器可以展开小循环。
- **仅用于长训练**: 短训练（<1000步）使用 `TrainNoOptImpl`，避免拷贝开销。

### 6.3 Fitness 计算

- **RMS Error 模式**: `fitness = FlipAndSquash(sqrt(sum_squared_error / n))`
  - `FlipAndSquash(x) = 2/π * arctan(1/x)`: 将 [0, ∞) 映射到 [0, 1]，使接近 0 的误差对应接近 1 的 fitness
- **Accuracy 模式**: 预测值通过 Sigmoid 转为概率，`fitness = 1 - error_rate`
- **Early Stopping**: 如果某训练/验证样本的误差超过 `max_abs_error`（默认100.0）或产生 NaN，整个执行返回 `kMinFitness = 0.0`

---

## 7. 进化搜索框架 (Regularized Evolution)

**文件**: `regularized_evolution.h`, `regularized_evolution.cc`

### 7.1 算法流程

AutoML-Zero 使用的是 **Regularized Evolution**（正则化进化），这是 Real et al. 2019 年提出的改进锦标赛选择算法。

```
Initialize:
  population = [Generator.TheInitModel()] × population_size  // 全用同一初始算法
  fitnesses = Evaluate(each individual)

Main Loop (until max_train_steps reached):
  for each position in population (轮流替换):
    parent = BestFitnessTournament(tournament_size)  // 锦标赛选择
    child = Mutate(parent)                           // 变异
    child_fitness = Evaluate(child)
    population[position] = child         // 直接替换当前位置!
    fitnesses[position] = child_fitness
```

### 7.2 种群管理策略

**非平凡设计点**：

- **顺序替换（非最老替换）**：开源代码中的实现与论文描述略有不同。论文描述的 Regularized Evolution 用 FIFO 队列移除最老个体。但这里的实现是**直接顺序遍历种群数组，逐个替换**（`for (shared_ptr<const Algorithm>& next_algorithm : algorithms_)`）。效果类似：每一代中整个种群全部被替换，类似于世代(generational)进化策略。

- **无精英保留 (No Elitism)**：每一代中所有个体（包括最优的）都会被替换。这是 Regularized Evolution 的关键特点——它通过"遗忘"最老个体来正则化搜索，避免过早收敛。

- **Init 时的空变异调用**: `Mutator.Mutate(0, algorithm)`——传入 `num_mutations=0`，不执行任何变异。代码注释指出这是为了"不影响随机数发生器"（保持与旧版本的可复现性）。

### 7.3 时间预算

搜索的终止条件是**总训练步数**（`max_train_steps`），而非个体数量或世代数。这是因为不同算法的训练步数可能不同（受 FEC cache 和 train budget 影响）。默认值为 80,000,000,000（800亿步），对应约 100万个个体各训练 8000步（10个任务×100样本×8 epoch）。

---

## 8. 变异系统 (Mutation)

**文件**: `mutator.h`, `mutator.cc`, `mutator.proto`

### 8.1 变异类型

系统定义了 **8 种变异操作**:

| MutationType | 说明 | 粒度 |
|---|---|---|
| ALTER_PARAM | 修改一条指令的一个参数（保持 op 不变） | 微调 |
| RANDOMIZE_INSTRUCTION | 随机化一条指令（包括 op） | 中等 |
| RANDOMIZE_COMPONENT_FUNCTION | 随机化整个组件函数（所有指令） | 大规模 |
| RANDOMIZE_ALGORITHM | 随机化所有三个组件函数 | 最大规模 |
| INSERT_INSTRUCTION | 在组件函数中插入一条新指令 | 结构变化 |
| REMOVE_INSTRUCTION | 从组件函数中删除一条指令 | 结构变化 |
| TRADE_INSTRUCTION | 删除一条+插入一条（保持长度不变） | 结构变化 |
| IDENTITY | 不做任何修改（用于调试） | 无 |

### 8.2 变异的执行逻辑

```
Mutate(algorithm):
  if random() < mutate_prob (默认 0.9):
    action = random_choice(allowed_mutation_types)
    component = random_choice(setup, predict, learn)  // 根据各自是否有允许的 ops 决定
    执行对应 action 在 component 上
```

### 8.3 ALTER_PARAM 变异细节

ALTER_PARAM 是最精细的变异——它只改变一个参数，保持操作码不变:

```cpp
Instruction::AlterParam(rand_gen):
  param_choice = random(0, 3)
  switch(param_choice):
    case 0: RandomizeIn1(rand_gen)     // 随机化输入地址1
    case 1: RandomizeIn2(rand_gen)     // 随机化输入地址2
    case 2: RandomizeOut(rand_gen)     // 随机化输出地址
    case 3: AlterData(rand_gen)        // 对数值数据做对数尺度扰动
```

**非平凡设计点**：
- **AlterData vs RandomizeData**: `AlterData` 在对数尺度上扰动现有值（小幅调整），而 `RandomizeData` 完全重新随机生成。ALTER_PARAM 使用前者，RANDOMIZE_INSTRUCTION 使用后者。
- **地址随机化是均匀的**: 从对应类型的全部地址中均匀采样，不考虑哪些地址更可能有意义。
- **每个组件函数可以有不同的允许操作**: `allowed_setup_ops`, `allowed_predict_ops`, `allowed_learn_ops` 可以分别设置。在 baseline 实验中，Setup 只允许初始化类 ops，Predict 允许前向计算 ops，Learn 允许梯度计算相关 ops。
- **变长组件函数**: INSERT/REMOVE 变异改变组件函数长度，但受 `min_size` 和 `max_size` 约束。

### 8.4 组件函数的选择

选择变异哪个组件函数时，从当前有允许 ops 的组件函数中**均匀随机选择**。如果某组件函数的 allowed_ops 为空，则永远不会被选中变异。

---

## 9. 评估系统 (Evaluator)

**文件**: `evaluator.h`, `evaluator.cc`

### 9.1 多任务评估

每个算法在**多个任务**上评估，fitness 是所有任务 fitness 的聚合:

```
Evaluate(algorithm):
  for each task in tasks:
    num_train = train_budget.TrainExamples(algorithm, task.max_train)
    fitness_i = Execute(task, num_train, algorithm)
  combined_fitness = CombineFitnesses(fitnesses, mode)
  // mode = MEAN 或 MEDIAN
```

**非平凡设计点**：
- **支持 MEAN 和 MEDIAN 聚合**: MEDIAN 聚合对离群任务（outlier tasks）更鲁棒。
- **特征维度分派**: 通过 `switch(task.FeaturesSize())` 跳转到对应的模板实例化版本（2/4/8/16/32），不同维度不能混用。

### 9.2 三阶段评估流程

实验使用三组不同的任务集:

```
T_search:  搜索过程中评估每个个体 (10个任务, 小数据)
T_select:  搜索结束后评估最佳候选 (100个任务, 大数据, 筛选最终算法)
T_final:   最终评估 (100个任务, 完全未见数据, 报告结果)
```

这种三阶段设计类似于 ML 中的 train/validation/test 分割，防止搜索过拟合到评估任务。

---

## 10. 任务系统 (Task)

**文件**: `task.h`, `task.proto`, `task_util.h`, `task_util.cc`

### 10.1 支持的任务类型

| 任务类型 | 说明 | 评估模式 |
|---------|------|---------|
| ScalarLinearRegression | 随机线性函数 y = wᵀx | RMS_ERROR |
| Scalar2LayerNNRegression | 随机2层神经网络生成的非线性函数 | RMS_ERROR |
| ProjectedBinaryClassification | MNIST/CIFAR10 降维到16维的二分类 | ACCURACY |

### 10.2 任务数据管理

**非平凡设计点**：
- **Epoch Shuffling**: 训练数据的遍历顺序在每个 epoch 随机打乱。打乱顺序在任务创建时就固定下来（通过 `GenerateEpochs`），而不是在训练时动态打乱。这保证了同一算法在同一任务上的评估是**确定性的**。
- **TaskIterator**: 封装了跨 epoch 的迭代逻辑，自动在 epoch 结束时切换到下一个 epoch 的打乱顺序。
- **Seed 控制**: 每个任务有独立的 `data_seed`（控制特征数据）和 `param_seed`（控制标签函数的参数）。`randomize_task_seeds` 标志可以在每次实验时重新随机化种子。
- **模板多态 + 运行时接口**: `Task<F>` 是模板类，`TaskInterface` 是基类接口。在需要统一容器存储不同 F 的任务时使用 `TaskInterface*`，执行时通过 `SafeDowncast<F>` 转换。

---

## 11. 功能等价缓存 (FEC Cache)

**文件**: `fec_cache.h`, `fec_cache.cc`, `fec_hashing.h`, `fec_hashing.cc`, `fec_cache.proto`

### 11.1 核心思想

FEC (Functional Equivalence Checking) 是论文的关键贡献之一。核心观察：**两个结构不同的算法可能在功能上完全等价**。例如，`v1 = v0 + v0` 和 `v1 = 2 * v0` 在数学上等价。

FEC 的做法是：先在**少量样本**（如10个训练+10个验证）上运行算法，收集产生的训练误差序列和验证误差序列，对其计算 hash。如果两个算法的 hash 相同，认为它们功能等价，可以复用之前的 fitness。

### 11.2 实现

```
Evaluator.Execute(algorithm):
  // 先在 FEC 子集上运行
  executor_fec = Executor(algorithm, task, fec_num_train, fec_num_valid)
  train_errors, valid_errors = executor_fec.Execute()
  hash = FECCache.Hash(train_errors, valid_errors, task_idx, num_train)
  
  if hash in cache:
    return cached_fitness  // 缓存命中, 跳过完整评估!
  else:
    // 缓存未命中, 做完整评估
    executor_full = Executor(algorithm, task, full_num_train, full_num_valid)
    fitness = executor_full.Execute()
    cache.Insert(hash, fitness)
    return fitness
```

**非平凡设计点**：
- **LRU 缓存**: 使用 LRU（最近最少使用）缓存实现，有固定大小上限。满时淘汰最久未访问的条目。
- **`forget_every` 参数**: 如果设置，当一个 hash 被命中 N 次后自动删除。这处理了一个微妙的问题：如果某个功能等价类在搜索中频繁出现但 fitness 很低，持续命中缓存会浪费锦标赛选择的"名额"。
- **Hash 设计**: 使用 `WellMixedHash` 将所有误差值和元数据混合为单个 `size_t`。误差值先量化为整数再混合，保证对微小数值差异不敏感。
- **确定性 RNG**: FEC 评估使用**固定种子的独立 RandomGenerator**（`kFunctionalCacheRandomSeed = 235732282`），确保对同一算法的 FEC hash 在不同时间点评估结果一致。

---

## 12. 训练预算 (Train Budget)

**文件**: `train_budget.h`, `train_budget.cc`, `train_budget.proto`

### 12.1 核心思想

不同算法的每步训练成本差异巨大（一条 `MATRIX_MATRIX_PRODUCT_OP` 比 `SCALAR_SUM_OP` 贵 3x）。TrainBudget 机制确保计算复杂度过高的算法不会消耗过多搜索预算。

### 12.2 实现

```
TrainBudget.TrainExamples(algorithm, budget):
  setup_cost = ComputeCost(algorithm.setup)
  train_cost = ComputeCost(algorithm.predict) + ComputeCost(algorithm.learn)
  if train_cost > threshold_factor * baseline_train_cost:
    return 0  // 不评估! fitness = 0
  else:
    return budget  // 正常评估
```

**非平凡设计点**：
- **基于基线的相对阈值**: 不是绝对成本上限，而是相对于 baseline 算法（如 `NEURAL_NET_ALGORITHM`）的倍数。这使得阈值对操作集的变化自适应。
- **ComputeCost 表**: 每种 Op 都有手动标定的成本值（基于实际计时），存储在 `compute_cost.cc` 的巨大 switch-case 中。成本值经过归一化校准。
- **二元决策**: 要么全额训练，要么完全跳过（返回0步 → fitness = 0）。不做部分训练。

---

## 13. 搜索实验入口 (run_search_experiment)

**文件**: `run_search_experiment.cc`

### 13.1 命令行参数

| 参数 | 说明 |
|------|------|
| `--search_experiment_spec` | SearchExperimentSpec protobuf 文本格式, 定义搜索的全部配置 |
| `--select_tasks` | T_select 任务集 (TaskCollection protobuf) |
| `--final_tasks` | T_final 任务集 |
| `--max_experiments` | 最大实验轮数 (每轮是一次完整进化搜索) |
| `--random_seed` | 随机种子 (0=自动生成) |
| `--sufficient_fitness` | 达到此 select fitness 时提前停止 |
| `--randomize_task_seeds` | 每次实验重新随机化任务数据 |

### 13.2 外循环: 多实验

```
while experiments < max_experiments:
  run evolution search on T_search → candidate_algorithm
  evaluate candidate on T_select → select_fitness
  track best_algorithm by select_fitness
  if best_select_fitness > sufficient_fitness: break

// 最终评估
evaluate best_algorithm on T_final → final_fitness
print final_fitness and best_algorithm.ToReadable()
```

### 13.3 SearchExperimentSpec 关键配置项

```protobuf
message SearchExperimentSpec {
  // 搜索任务
  optional TaskCollection search_tasks;
  
  // 搜索空间: 各组件函数允许的操作
  repeated Op setup_ops;
  repeated Op predict_ops;
  repeated Op learn_ops;
  
  // 组件函数初始大小
  optional int32 setup_size_init;
  optional int32 predict_size_init;
  optional int32 learn_size_init;
  
  // 变异参数
  optional MutationTypeList allowed_mutation_types;
  optional double mutate_prob = 0.9;
  optional int32 mutate_setup_size_min/max;
  optional int32 mutate_predict_size_min/max;
  optional int32 mutate_learn_size_min/max;
  
  // 搜索方法
  optional int64 population_size;
  optional int64 tournament_size;
  optional HardcodedAlgorithmID initial_population;
  optional int64 max_train_steps;
  
  // 加速技术
  optional FECSpec fec;
  optional TrainBudgetSpec train_budget;
  optional double max_abs_error = 100.0;
}
```

---

## 14. 非平凡设计选择总结

### 14.1 搜索空间设计

1. **三函数分解 (Setup/Predict/Learn)**: 唯一的结构性先验。不预设模型类型、层数、损失函数、优化器。
2. **寄存器式内存而非变量命名**: 固定数量的类型化寄存器，指令直接通过地址操作。消除了变量命名/作用域的复杂性。
3. **65种低级数学操作**: 覆盖标量/向量/矩阵三种类型的完整运算集。不包含 ML 特有操作（如 softmax、batch norm、attention）。
4. **无控制流**: 组件函数是线性指令序列，不支持分支或循环。通过 Heaviside + min/max 实现有限条件逻辑。
5. **float 编码的索引**: 向量/矩阵坐标用 [0,1) 浮点表示，与具体维度解耦。

### 14.2 进化策略设计

6. **Regularized Evolution（正则化进化）**: 持续替换最老个体，无精英保留。平衡探索与开发。
7. **锦标赛选择**: 从种群中随机抽样 `tournament_size` 个个体，选最好的作为 parent。选择压力可调。
8. **多粒度变异**: 从参数微调 (ALTER_PARAM) 到完全重随机化 (RANDOMIZE_ALGORITHM)，覆盖不同尺度的搜索步长。
9. **变长程序**: INSERT/REMOVE 变异允许程序长度变化（在 min/max 约束内），实现架构搜索。
10. **per-component 操作集**: 可以分别限制各组件函数的操作集，将搜索空间缩小为更有针对性的子空间。

### 14.3 评估效率设计

11. **FEC (功能等价缓存)**: 基于行为 hash 的去重，避免重复评估功能等价的算法。论文报告 30-60% 的缓存命中率，显著加速搜索。
12. **训练预算**: 基于操作成本的复杂度过滤，提前淘汰过于昂贵的算法。
13. **Early Stopping**: NaN/Inf/大误差触发提前终止，避免在注定失败的算法上浪费时间。
14. **三阶段评估 (search/select/final)**: 防止搜索过拟合到评估任务。
15. **编译期维度特化**: 模板参数 F 在编译期确定，结合 Eigen 固定大小矩阵实现零开销抽象。

### 14.4 实现效率设计

16. **浅拷贝 + immutable 指令**: shared_ptr<const Instruction> 使得算法拷贝近乎零成本。
17. **函数指针查找表**: O(1) 指令分派，比 switch-case 或虚函数调用更快。
18. **栈上优化训练循环**: TrainOptImpl 将指令拷贝到固定大小栈数组，消除指针间接访问。
19. **Protobuf 配置**: 通过 protobuf 定义配置和序列化格式，命令行传递 text-format proto。

### 14.5 搜索流程设计

20. **多实验外循环**: 可以运行多次独立搜索，每次搜索内部是一个完整的进化过程。
21. **充分 fitness 阈值**: `sufficient_fitness` 允许在发现足够好的算法时提前停止。
22. **任务种子随机化**: `randomize_task_seeds` 确保每次实验使用不同的随机任务数据。

---

## 15. 文件清单

### 核心源文件

| 文件 | 行数(约) | 说明 |
|------|----------|------|
| `algorithm.h` / `.cc` | 80 / 200 | 算法类定义 |
| `instruction.h` / `.cc` | 250 / 850+ | 指令类定义和突变逻辑 |
| `executor.h` | ~1350 | 执行引擎 (全部在 header 实现) |
| `memory.h` | 80 | 寄存器式内存 |
| `evaluator.h` / `.cc` | 110 / 210 | 多任务评估器 |
| `regularized_evolution.h` / `.cc` | 120 / 250 | 进化搜索主循环 |
| `mutator.h` / `.cc` | 140 / 350 | 8种变异操作 |
| `generator.h` / `.cc` | 120 / 500 | 初始算法生成（含手写 linear/NN） |
| `randomizer.h` / `.cc` | 80 / 80 | 算法全随机化 |
| `fec_cache.h` / `.cc` | 110 / 150 | 功能等价性缓存 |
| `fec_hashing.h` / `.cc` | 40 / 40 | 误差序列哈希 |
| `compute_cost.h` / `.cc` | 30 / 150 | 每种操作的计算成本表 |
| `train_budget.h` / `.cc` | 60 / 60 | 基于成本的训练预算 |
| `task.h` | 350 | 任务数据和迭代器 |
| `task_util.h` / `.cc` | 600 / 160 | 任务创建工具函数 |
| `random_generator.h` / `.cc` | 150 / 100 | RNG 封装 |
| `run_search_experiment.cc` | 300 | 实验入口 main 函数 |
| `definitions.h` | 300 | 全局类型、常量和工具函数 |

### Proto 定义

| 文件 | 说明 |
|------|------|
| `instruction.proto` | 65种 Op 枚举定义 |
| `algorithm.proto` | 算法序列化格式 |
| `mutator.proto` | 8种变异类型枚举 |
| `generator.proto` | 预定义算法类型枚举 |
| `experiment.proto` | 搜索实验完整配置 |
| `task.proto` | 任务规格定义 |
| `fec_cache.proto` | FEC 配置 |
| `train_budget.proto` | 训练预算配置 |

### 构建和脚本

| 文件 | 说明 |
|------|------|
| `BUILD` | Bazel 构建规则 |
| `WORKSPACE` | 外部依赖 (abseil, eigen, gtest, protobuf, glog, gflags) |
| `eigen.BUILD` | Eigen 库构建规则 |
| `setup.sh` | 安装 pip 依赖 |
| `run_demo.sh` | 演示: 发现线性回归 (<5min) |
| `run_baseline.sh` | 复现论文 baseline 实验 (12-18h) |
| `generate_datasets.py` | 生成 projected CIFAR10 数据集 |
