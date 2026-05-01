# Algorithm IR — 基于 GNN 引导双重进化的算法自动改进系统

> **读者对象**：本文档假设读者具备基本的 Python 编程知识，但**不需要**编译器原理、图神经网络（GNN）、强化学习（RL）或通信信号处理方面的专业背景。每个领域的概念在首次出现时都会给出详细解释。

---

## 目录

1. [算法的 IR 表示](#1-算法的-ir-表示)
   - 1.1 [什么是 IR？为什么需要它？](#11-什么是-ir为什么需要它)
   - 1.2 [IR 的体系结构](#12-ir-的体系结构)
   - 1.3 [类型格子（Type Lattice）](#13-类型格子type-lattice)
2. [双重进化框架概览](#2-双重进化框架概览)
   - 2.1 [核心思想](#21-核心思想宏观与微观双层进化)
   - 2.2 [核心数据结构](#22-核心数据结构)
   - 2.3 [算法引擎的运行循环](#23-算法引擎的运行循环)
3. [IR 嫁接机制详解](#3-ir-嫁接机制详解)
   - 3.1 [什么是嫁接？](#31-什么是嫁接grafting)
   - 3.2 [graft_general() 的 13 步管线](#32-graft_general-的-13-步管线)
   - 3.3 [嫁接前后对比示例](#33-嫁接前后对比示例)
   - 3.4 [嫁接的类型安全保证](#34-嫁接的类型安全保证)
4. [GNN 对 IR 的编码方式](#4-gnn-对-ir-的编码方式)
   - 4.1 [什么是图神经网络？](#41-什么是图神经网络gnn)
   - 4.2 [IR 到图的转换](#42-ir-到图的转换ir_to_graph)
   - 4.3 [GNN 编码器](#43-gnn-编码器irgraphencoder)
   - 4.4 [对打分器与策略网络](#44-对打分器与策略网络)
   - 4.5 [整条 GNN 管线](#45-整条-gnn-管线)
5. [通过 RL 训练 GNN](#5-通过-rl-训练-gnn)
   - 5.1 [什么是强化学习？](#51-什么是强化学习rl)
   - 5.2 [RL 公式化](#52-rl-公式化)
   - 5.3 [策略梯度](#53-策略梯度policy-gradient)
   - 5.4 [辅助损失](#54-辅助损失auxiliary-losses)
   - 5.5 [总训练损失](#55-总训练损失)
   - 5.6 [训练调度](#56-训练调度)

## 1. 算法的 IR 表示

### 1.1 什么是 IR？为什么需要它？

**IR**（Intermediate Representation，中间表示）是编译器技术中的一个核心概念。当我们写好一段 Python 代码（比如一个 MIMO 信号检测算法），计算机执行之前通常会把代码转换成某种"中间形式"——这种形式既保留了源代码的语义（"做了什么"），又比原始文本更结构化、更适合自动化分析和修改。

举个简单例子。源代码 `x = a + b * c` 在 IR 中可能被表示为：

```
op_mul:  binary(multiply)  inputs=[b, c]   → output v1
op_add:  binary(add)       inputs=[a, v1]  → output x
```

这种表示法的核心理念是 **SSA 形式**（Static Single Assignment，静态单赋值）。在 SSA 中，**每个变量只被赋值一次**（不能再给同一个变量名赋第二次值），所有数据流动都通过有向边（从定义它的操作指向使用它的操作）来显式表示。这样做的好处是：程序的数据依赖关系一目了然，自动化工可以安全地增删改代码片段而不会不小心破坏数据流。

> **为什么不用 Python AST？** Python 自带的 AST（抽象语法树）确实也是一种中间表示，但它与 Python 语法耦合太紧——`if`、`for`、`list comprehension` 等几十种语法结构使得跨语言分析和转换极其复杂。相比之下，本系统的 IR 只使用 **21 种操作码 + 3 种控制流指令** 就覆盖了所有必要的计算，极大地简化了下游的进化操作和 GNN 编码。

### 1.2 IR 的体系结构：FunctionIR 与它的组成部分

在本文档讨论的系统中，每个算法（一个 MIMO 检测器的 Python 函数）都被转换成一个 `FunctionIR` 对象。这个对象是算法在"IR 世界"中的**完整表示**。一个 `FunctionIR` 包含以下核心组成部分：

#### 1.2.1 `Value` — 数据流动的原子

```python
# algorithm_ir/ir/model.py, 第 41–49 行
@dataclass
class Value:
    id: str                      # 唯一标识符,如 "v_7", "v_out"
    name_hint: str | None = None # 源代码中的变量名（如果保留了下来）
    type_hint: str | None = None # 类型格子中的类型字符串,如 "mat_cx", "float"
    source_span: SourceSpan = None  # 源代码位置（行号）
    def_op: str | None = None    # 定义这个值的 Op 的 id（生产者）
    use_ops: list[str] = field(default_factory=list)  # 消费这个值的 Op 的 id 列表（消费者）
    attrs: dict[str, Any] = field(default_factory=dict) # 可扩展的元数据
```

**值的角色**：在 SSA 形式中，值（Value）是数据流动的"节点"。每一个值恰好由一个 Op 定义（`def_op` 指向那个 Op），但可以被多个 Op 消费（`use_ops` 列表）。这种"一产多消"的结构使我们可以精确追踪任何一个数据片段在算法中的流向。

**直观理解**：把算法想象成工厂流水线。Op 是加工站（机器），Value 是在传送带上流动的零件。每个零件（Value）只在一个加工站（`def_op`）被制造出来，但可能被多个下游加工站（`use_ops`）使用。

**`type_hint` 的含义**：每个值都携带一个类型标记，指示它"是什么类型的数据"。这里使用的不是 Python 原生类型（如 `numpy.ndarray`），而是**类型格子**（Type Lattice，见 1.3 节）中的类型字符串。例如：

| type_hint | 含义 | 近似的 Python/numpy 类型 |
|-----------|------|------------------------|
| `"float"` | 标量浮点数 | `float` / `np.float64` |
| `"int"` | 标量整数 | `int` / `np.int64` |
| `"bool"` | 布尔值 | `bool` |
| `"cx"` | 标量复数 | `complex` / `np.complex128` |
| `"vec_f"` | 浮点向量（一维数组） | `np.ndarray[float]` with `.ndim == 1` |
| `"vec_cx"` | 复数向量 | `np.ndarray[complex]` with `.ndim == 1` |
| `"mat_f"` | 浮点矩阵（二维数组） | `np.ndarray[float]` with `.ndim == 2` |
| `"mat_cx"` | 复数矩阵 | `np.ndarray[complex]` with `.ndim == 2` |
| `"tensor3_f"` | 浮点三维张量 | `np.ndarray[float]` with `.ndim == 3` |
| `"tuple<float, mat_cx>"` | 元组（浮点, 复数矩阵） | `(float, np.ndarray)` |

#### 1.2.2 `Op` — 计算操作

```python
# algorithm_ir/ir/model.py, 第 52–60 行
@dataclass
class Op:
    id: str                # 唯一标识符,如 "op_42"
    opcode: str            # 操作码,如 "binary", "call", "const"
    inputs: list[str]      # 输入 Value 的 id 列表（消耗的数据）
    outputs: list[str]     # 输出 Value 的 id 列表（产生的数据）
    block_id: str          # 属于哪个基本块
    source_span: SourceSpan = None
    attrs: dict[str, Any] = field(default_factory=dict)
```

**`opcode` 的完整类型清单**：系统中定义了 21 种不同的操作码（来自 `algorithm_ir/ir/dialect.py`），分为两大类：

**产生值的操作（18 种）**——这些 Op 将输入数据加工后产出一个或多个新值：

| opcode | 说明 | 数学/编程语义 | IR 示例 |
|--------|------|-------------|--------|
| `const` | 编译期常量 | 直接嵌入一个字面量值 | `op_c: const → v=3.14` |
| `assign` | SSA 重命名 | `target = source`（不做计算，只做名字绑定） | `op_a: assign(inputs=[v_x]) → v_y` |
| `binary` | 二元运算 | `lhs ⊗ rhs`，其中 ⊗ 可以是 `+`, `-`, `*`, `/`, `@`, `>`, `<`, `==`, `and`, `or` 等 | `op_b: binary(add, inputs=[v_a, v_b]) → v_c` |
| `unary` | 一元运算 | `⊗ operand`，其中 ⊗ 可以是 `-`（取负）, `not`（逻辑非）, `~`（按位取反）, `abs` | `op_u: unary(neg, inputs=[v_x]) → v_negx` |
| `compare` | 链式比较 | `a < b < c`（Python 语义，短路求值） | `op_cmp: compare(lt, inputs=[v_a, v_b, v_c]) → v_result` |
| `phi` | φ 节点 | 从不同控制流路径中选择值。形式化语义：若程序从 `pred_i` 到达本块，则 φ 选择第 i 个输入值 | `op_phi: phi(inputs=[v_from_then, v_from_else]) → v_merged` |
| `call` | 函数调用 | `callee(arg0, arg1, ...)`，被调用者可以是 numpy/scipy 函数或自定义函数 | `op_inv: call(np.linalg.inv, inputs=[v_H]) → v_Hinv` |
| `get_attr` | 属性读取 | `owner.attr`（如 `x.T`, `obj.shape`） | `op_ga: get_attr(T, inputs=[v_x]) → v_xt` |
| `set_attr` | 属性写入 | `owner.attr = value`（有副作用，无返回值） | `op_sa: set_attr(attr, inputs=[v_obj, v_val]) → []` |
| `get_item` | 下标读取 | `owner[index]` | `op_gi: get_item(inputs=[v_arr, v_idx]) → v_elem` |
| `set_item` | 下标写入 | `owner[index] = value`（有副作用） | `op_si: set_item(inputs=[v_arr, v_idx, v_val]) → []` |
| `build_list` | 列表构造 | `[a, b, c]` | `op_bl: build_list(inputs=[v_a, v_b, v_c]) → v_list` |
| `build_tuple` | 元组构造 | `(a, b)` | `op_bt: build_tuple(inputs=[v_a, v_b]) → v_tuple` |
| `build_dict` | 字典构造 | `{"key": value}` | `op_bd: build_dict(inputs=[v_k1, v_v1]) → v_dict` |
| `append` | 列表追加 | `lst.append(x)`（就地修改，无返回值） | `op_ap: append(inputs=[v_lst, v_x]) → []` |
| `pop` | 列表弹出 | `lst.pop()` | `op_pop: pop(inputs=[v_lst]) → v_item` |
| `iter_init` | 迭代器初始化 | `iter(seq)` | `op_ii: iter_init(inputs=[v_seq]) → v_iter` |
| `iter_next` | 迭代器前进 | `next(it)`，返回 `(next_value, has_next)` 两个值 | `op_in: iter_next(inputs=[v_iter]) → [v_next, v_has]` |

**控制流终结操作（3 种）**——这些 Op 不产生值，而是决定程序接下来去哪里：

| opcode | 说明 | 语法 |
|--------|------|------|
| `branch` | 条件分支 | `if condition: goto block_A else: goto block_B`。有两个后继块。 |
| `jump` | 无条件跳转 | `goto block_X`。有一个后继块。 |
| `return` | 函数返回 | `return value`。返回值 id 列在 `inputs` 中。 |

这三种终结操作在 `attrs` 中携带了后继块信息。在 xDSL 层面，每个控制流操作都有对应的 `successors` 属性。

#### 1.2.3 `Block` — 基本块与控制流图

```python
# algorithm_ir/ir/model.py, 第 63–69 行
@dataclass
class Block:
    id: str                                       # 唯一标识符
    op_ids: list[str] = field(default_factory=list)  # 有序的操作列表（保证按此顺序执行）
    preds: list[str] = field(default_factory=list)    # 前驱块 id（"我从哪里来"）
    succs: list[str] = field(default_factory=list)    # 后继块 id（"我去哪里"）
    attrs: dict[str, Any] = field(default_factory=dict)
```

**基本块（Basic Block）** 是编译器理论中的术语，指一段**完全直线**执行的代码：一旦进入，就按照 `op_ids` 中排好的顺序依次执行每个操作，**中间不会有任何跳转**。跳转只发生在块的末尾（通过 `branch`、`jump` 或 `return` 终结操作）。**块的最后一个操作必定是控制流终结操作**（terminator）。

多个基本块通过前驱/后继关系（`preds`/`succs`）连接成一个有向图，这个图就是**控制流图**（Control Flow Graph, CFG）：

```
         [entry_block]          ← 函数入口，接收参数值
              |
         [block_0]              ← 直线代码块
           /     \
    branch/       \jump         ← 条件/无条件分支
         /         \
   [block_1]    [block_2]
         \         /
          \       /
           [block_3]            ← 汇合块（可能包含 φ 节点）
              |
           [return]             ← 出口块
```

**什么是 φ 节点，为什么要用它？** —— 当一个变量在两条不同的控制流路径中分别被赋值后，在汇合点需要决定"到底用哪个值"。φ 节点就是解决这个问题的。例如：

```python
# 源代码：
if cond:
    x = 1           # 路径 A
else:
    x = 2           # 路径 B
y = x + 10          # ← 这里 x 是哪个值？

# SSA IR 等价表示：
# block_then:  op_const → v_1 = 1
# block_else:  op_const → v_2 = 2
# block_merge: op_phi(inputs=[v_1, v_2]) → v_x = φ(来自then则v_1, 来自else则v_2)
#              op_binary(add, inputs=[v_x, v_10]) → v_y
```

φ 节点的语义是：根据程序**实际从哪个前驱块到达**，选择对应的输入值。这是一个**纯函数式**的操作——没有副作用，只是"选择"。

#### 1.2.4 `FunctionIR` — 函数级 IR 容器

```python
# algorithm_ir/ir/model.py, 第 88–128 行
class FunctionIR:
    id: str                     # 函数的唯一 id（通常是算法名）
    name: str                   # 函数名
    arg_values: list[str]       # 参数值 id 列表（函数的输入参数在 IR 中的 Value id）
    return_values: list[str]    # 返回值 id 列表
    values: dict[str, Value]    # "v_7" → Value 对象的映射
    ops: dict[str, Op]          # "op_42" → Op 对象的映射
    blocks: dict[str, Block]    # "block_0" → Block 对象的映射
    entry_block: str            # 入口基本块的 id
    slot_meta: dict[str, SlotMeta]  # 插槽元数据（见 1.2.5）
    attrs: dict[str, Any]       # 可扩展属性
    # 底层 xDSL 引用（序列化/反序列化用）
    xdsl_module: ...            # 对应的 xDSL ModuleOp
    xdsl_func: FuncOp | None    # 对应的 xDSL 函数操作
```

一个 `FunctionIR` 提供了字典式的访问（`ir.values["v_7"]`、`ir.ops["op_42"]`）以方便遍历和修改。实际存储在硬盘上时，通过 **xDSL**（一个 SSA-based 编译器框架）进行序列化和反序列化。`FunctionIR` 内部维护了一个 `xdsl_module` 的引用，代表该 IR 在 xDSL 世界中的等价表示。

**两种构造路径**：`FunctionIR.from_xdsl()` 会根据 xDSL 模块中的元数据自动判断走哪条路：

```python
# algorithm_ir/ir/model.py, 第 157–178 行
@classmethod
def from_xdsl(cls, module) -> "FunctionIR":
    """从 xDSL ModuleOp 构建 FunctionIR。"""
    func = next(iter(module.ops))

    if func.attributes.get("alg_id") is not None:
        return cls._from_typed_xdsl(module, func)   # 新式：类型化方言
    else:
        return cls._from_legacy_xdsl(module, func)   # 旧式：UnregisteredOp
```

`_from_typed_xdsl` 遍历 xDSL 块和操作，为每个操作创建对应的 `Op`、`Value` 和 `Block` 对象，并提取 `type_hint`、`slot_meta` 等元数据。

#### 1.2.5 `SlotMeta` — 可替换的代码插槽

```python
# algorithm_ir/ir/model.py, 第 72–85 行
@dataclass
class SlotMeta:
    pop_key: str                      # Slot 的唯一标识名
    op_ids: tuple[str, ...]           # 直接属于本 Slot 的 op（不含子 Slot 的 op）
    inputs: tuple[str, ...]           # Slot 区域的外部输入值列表
    outputs: tuple[str, ...]          # Slot 区域的外部输出值列表
    output_names: tuple[str, ...]     # 源代码中输出变量的名字
    parent: str | None = None         # 父 Slot 的 pop_key（嵌套 Slot 时使用）
```

Slot 是源代码中的 `with slot("name"):` 代码块在 IR 中的对应物。在算法的 Python 源代码中，开发者可以手动标记某些代码块为"插槽"，语义上表示"这个区域是可替换的（可以被嫁接替换掉），也可以被微观进化填充新的内容"。插槽可以嵌套。

```python
# 源代码示例：
with slot("detection_core"):
    x_hat = some_detection(H, y)

# IR 中对应一个 SlotMeta:
# pop_key = "detection_core"
# op_ids = (定义 x_hat 的那个 op 的 id, ...)
# inputs  = (H, y)      ← 流入 slot 的值
# outputs = (v_x_hat,)  ← 流出 slot 的值
```

`FunctionIR` 提供了两个方法来遍历 Slot 层次：

```python
def slot_children(self, pop_key: str | None) -> list[str]:
    """返回 parent == pop_key 的子 Slot 列表。pop_key=None 返回根级 Slot。"""

def slot_full_op_ids(self, pop_key: str) -> set[str]:
    """返回该 Slot 及其所有子孙 Slot 的全部 op（BFS 遍历）。"""
```

---

### 1.3 类型格子（Type Lattice）

**类型格子**（Type Lattice）是本系统中最重要的基础设施之一。它的作用是用一个**有限的、可比较的类型集合**来近似描述 Python 运行时的无限多种数据类型。

> **什么是"格子"（Lattice）？** 在数学中，一个格（Lattice）是一个偏序集（partially ordered set），其中任意两个元素都有唯一的**最小上界**（join，记作 ∨）和**最大下界**（meet，记作 ∧）。在类型系统领域，"子类型关系" `<:` 定义了偏序：类型 A 比类型 B "更小"意味着 A 的值可以在任何需要 B 的地方安全使用。这个偏序天然形成一个格结构：
> - **最小上界** = 两个类型的"公共超类型"（unify），例如 `unify("int", "float") = "float"`
> - **最大下界** = 如果存在的话，是两个类型的"公共子类型"（本系统中未使用 meet）

#### 1.3.1 为什么需要类型格子？

在自动修改代码的时候（无论是 GP 变异还是嫁接），系统需要知道：如果我把操作 A 换成操作 B，类型是否还能"对得上"？例如，如果某个值在 IR 中标记为 `"mat_f"`（浮点矩阵），那么一个返回 `"int"` 的操作就不应该被放在这里——类型不匹配会导致运行时错误。

但是，如果每次都要完整地运行 Python 程序来推断类型，那就太慢了（每个候选变异都需要 ~1 秒的编译+执行时间）。类型格子提供了一种**静态近似**：用有限的符号（如 `"mat_cx"`、`"vec_f"`）来表示无限多种可能的 Python 运行时类型，并在这些符号上定义子类型关系。检查 `is_subtype("mat_f", "mat_cx")` 只需要 O(1) 的字典查找操作。

#### 1.3.2 类型的五类词汇

类型格子中定义了五大类原子类型（来自 `algorithm_ir/ir/type_lattice.py` 第 86–110 行）：

**第一类 — 标量（Primitive）**：

```python
PRIMITIVE_TYPES = ("int", "float", "bool", "cx")
```

| 类型 | 语义 | Python 等价 |
|------|------|-----------|
| `"int"` | 整数 | `int`, `np.int64` |
| `"float"` | 浮点数 | `float`, `np.float64` |
| `"bool"` | 布尔值 | `bool` |
| `"cx"` | 复数标量 | `complex`, `np.complex128` |

**第二类 — 张量（Tensor）**：

```python
TENSOR_TYPES = (
    "vec_f", "vec_cx", "vec_i",        # 一维数组（向量）
    "mat_f", "mat_cx",                  # 二维数组（矩阵）
    "tensor3_f", "tensor3_cx",         # 三维数组
)
# 命名规则: {形状前缀}_{dtype后缀}
#   vec_  = 向量（1维）, mat_ = 矩阵（2维）, tensor3_ = 三维张量
#   _f = 浮点, _cx = 复数, _i = 整数
```

**第三类 — 领域对象（Object）**：

```python
OBJECT_TYPES = (
    "node",           # MIMO 离散搜索候选节点（携带 cost + symbols）
    "candidate_list", # 候选节点列表
    "open_set",       # 未探索节点的优先队列
    "mat_decomp",     # 矩阵分解结果（例如 QR 分解的 (Q, R) 元组）
    "prob_table",     # 概率表（每符号的离散概率分布）
)
```

这些类型是通信信号处理领域特有的。它们对类型系统来说是"黑盒"——系统只知道它们的名字和层次关系（都是 `"object"` 的子类型），但不会深入它们的内部结构。

**第四类 — 组合类型（Composite）**：

通过字符串编码的泛型类型，格式为：

```
"tuple<T1,T2,...,Tn>"  — 固定长度的异构元组，n ≥ 0
"list<T>"              — 同构列表
"dict<T>"              — 字典（键固定为 str，值为类型 T）
```

例如：
- `"tuple<float, mat_cx>"` 表示一个元组，第一个元素是浮点数，第二个是复数矩阵
- `"list<vec_f>"` 表示一个浮点向量的列表
- `"tuple<>"` 表示空元组

组合类型在代码中通过 `parse_composite()` 函数（第 320 行）解析：

```python
def parse_composite(t: str):
    """
    解析组合类型字符串。
    "tuple<float, mat_cx>" → ("tuple", ["float", "mat_cx"])
    "list<vec_f>" → ("list", ["vec_f"])
    "dict<prob_table>" → ("dict", ["prob_table"])
    非组合类型 → (None, [])
    """
```

**第五类 — 通用类型（Universal）**：

| 类型 | 数学意义 | 语义 |
|------|---------|------|
| `"any"` | ⊤ （Top，顶类型） | 任何类型都是 `"any"` 的子类型。用作"兜底"类型。 |
| `"void"` | ⊥ （Bottom，底类型） | 用于不产生任何值的语句（如赋值语句、`set_item` 等副作用操作）。 |

#### 1.3.3 子类型层次结构

**子类型关系**（记为 `<:`）的直观含义是：如果类型 A 是类型 B 的子类型（写作 `A <: B`），那么在所有需要类型 B 的地方，类型 A 的值都可以安全使用而不会出错。

例如：`"int" <: "float"`（整数可以被当作浮点数使用，只是精度损失），但 `"float"` 并**不**是 `"int"` 的子类型（浮点数不能保证是整数，不能安全地用在只有整数才合法的地方）。

子类型关系在代码中通过 `_SUPER` 字典定义每个类型的直接父类型：

```python
# algorithm_ir/ir/type_lattice.py, 第 140–168 行
_SUPER = {
    # 标量链
    "bool":   {"int"},
    "int":    {"float"},
    "float":  {"cx"},

    # 张量链
    "vec_i":      {"vec_f"},
    "vec_f":      {"vec_cx"},
    "mat_f":      {"mat_cx"},
    "tensor3_f":  {"tensor3_cx"},

    # 跨维度提升（标量可广播为向量/矩阵）
    "int":   {"vec_i", "mat_f", "tensor3_f"},
    "float": {"vec_f", "mat_f", "tensor3_f"},
    "cx":    {"vec_cx", "mat_cx", "tensor3_cx"},

    # 领域对象 → object
    "node":           {"object"},
    "candidate_list": {"object"},
    "open_set":       {"object"},
    "mat_decomp":     {"object"},
    "prob_table":     {"vec_f"},   # prob_table 也是向量的一种特殊形式

    # 万物归于 any
    "object": {"any"},
    "cx":     {"any"},
    "vec_cx": {"any"},
    "mat_cx": {"any"},
    "tensor3_cx": {"any"},
}
```

`is_subtype(a, b)` 的实现使用了 BFS（广度优先搜索）来查找 b 是否在 a 的祖先集合中：

```python
def is_subtype(a: str, b: str) -> bool:
    # 1. 如果 a == b：平凡情况，直接 True
    # 2. 如果 b == "any"：任何类型都是 any 的子类型
    # 3. 否则：BFS 遍历 a 的所有祖先（通过 _SUPER），检查 b 是否在其中
    # 4. 组合类型：递归地对每个分量做子类型检查
```

对于组合类型，子类型关系是**协变**的：`"tuple<A1, A2>" <: "tuple<B1, B2>"` 当且仅当 `A1 <: B1` 且 `A2 <: B2`。`list<A> <: list<B>` 当且仅当 `A <: B`。

`unify(a, b)` 计算两个类型的**最小公共超类型**（Least Common Supertype）：

```python
def unify(a: str, b: str) -> str:
    # 1. 如果 a <: b：返回 b（b 已经是公共超类型）
    # 2. 如果 b <: a：返回 a
    # 3. 否则：从 a 开始 BFS，找到第一个也是 b 的超类型的祖先
    #    （这就是"最小的"公共超类型）
    # 4. 如果找不到（例如 "vec_f" 和 "mat_f" 没有非 any 的公共超类型）：
    #    返回 "any"
```

**数学性质**：对于任何满足 `a <: u` 且 `b <: u` 的类型 u，`unify(a, b) <: u` 恒成立。这保证了 `unify` 确实返回最小上界。

#### 1.3.4 类型推断机制

系统在 IR 构建阶段（从 Python AST 到 IR）和嫁接阶段（组合操作时）需要确定每个新 Value 的 `type_hint`。这是通过一系列规则实现的：

**二元操作的类型推断**（`combine_binary_type`，第 560 行）：

结果类型由操作符和两个操作数的类型共同决定：

```python
def combine_binary_type(op: str, lhs: str, rhs: str) -> str:
    """
    对于给定的二元操作符 op 和左右操作数类型，返回结果类型。
    使用 _TYPE_DECOMP 分解类型为 (rank, dtype_class)，然后通过 numpy 广播规则组合。
    """
    # 第一步：分解每种类型
    rank_l, dclass_l = _TYPE_DECOMP[lhs]  # 例如 "mat_cx" → (2, "c")
    rank_r, dclass_r = _TYPE_DECOMP[rhs]

    # 第二步：按操作符决定结果
    if op in ("+", "-", "*", "/", "**", "%", "//"):
        # 算术：dtype 取更"大"的，rank 取更大的，允许标量-张量广播
        dclass = promote_dtype(dclass_l, dclass_r)  # i < f < c
        rank = promote_rank(rank_l, rank_r)          # max
        return _TYPE_COMPOSE[(rank, dclass)]

    if op in ("@", "matmul"):
        # 矩阵乘法：(m,k) @ (k,n) → (m,n)
        return _matmul_result(lhs, rhs)

    if op in ("<", ">", "==", "!=", "<=", ">="):
        # 比较：如果是标量返回 bool，如果是张量返回逐元素比较结果
        if max(rank_l, rank_r) == 0:
            return "bool"
        else:
            return promote_to_vec(promote_to_real(_matmul_or_arith_result))

    if op in ("and", "or"):
        return "bool"

    if op in ("&", "|", "^", "<<", ">>"):
        # 按位运算：只对整数/布尔合法
        ...
```

**秩（rank）和 dtype 的底层表示**：系统将每个张量类型映射到一个 `(rank, dtype_class)` 对：

```python
# _TYPE_DECOMP 字典（类型 → (秩, dtype类别)），第 240 行
_TYPE_DECOMP = {
    "int":        (0, "i"),    # 秩=0（标量），dtype=整数
    "float":      (0, "f"),    # 秩=0，dtype=浮点
    "cx":         (0, "c"),    # 秩=0，dtype=复数
    "bool":       (0, "i"),
    "vec_i":      (1, "i"),    # 秩=1（向量），dtype=整数
    "vec_f":      (1, "f"),
    "vec_cx":     (1, "c"),
    "mat_f":      (2, "f"),    # 秩=2（矩阵），dtype=浮点
    "mat_cx":     (2, "c"),
    "tensor3_f":  (3, "f"),    # 秩=3（三维张量）
    "tensor3_cx": (3, "c"),
    "prob_table": (1, "f"),    # 特殊：概率表 ≈ 浮点向量
    "object":     (0, "?"),    # 黑盒对象
}

# dtype 提升规则
def promote_dtype(a: str, b: str) -> str:
    """i → f → c，取"更大"的方向。"""
    if a == "c" or b == "c": return "c"
    if a == "f" or b == "f": return "f"
    return "i"

# 秩的广播规则
def promote_rank(a: int, b: int) -> int:
    """取更大的秩（numpy 广播语义）。"""
    return max(a, b)

# 矩阵乘法的秩代数
def _matmul_result(lhs, rhs):
    """(m,k) @ (k,n) → (m,n)，秩 = 秩(lhs) + 秩(rhs) - 2"""
    rl, dl = _TYPE_DECOMP[lhs]
    rr, dr = _TYPE_DECOMP[rhs]
    # 如果都是标量 → 标量
    # 如果向量×矩阵 → 向量
    # 如果矩阵×矩阵 → 矩阵
    ...
```

**一元操作的类型推断**（`combine_unary_type`）：

```python
def combine_unary_type(op: str, operand: str) -> str:
    if op in ("neg", "pos", "abs"):   # -x, +x, abs(x)
        return operand                 # 不改变类型
    if op in ("not_",):                 # not x
        return "bool"                   # 总是返回布尔
    if op in ("invert",):              # ~x（按位取反）
        return promote_to_int(operand)  # 只对整数合法
```

**函数调用的类型推断**（`infer_call_return_type`，第 700 行）：系统维护了一个包含 ~90+ 个内置函数的类型注册表（`_CALLABLE_RETURNS`）。对于每个注册的函数，系统知道"给定这些输入类型，输出类型是什么"。

```python
# 注册表示例（第 470–540 行）
_CALLABLE_RETURNS = {
    # numpy 线性代数
    "np.linalg.inv":    lambda args: args[0],        # 输入矩阵，返回同类型矩阵
    "np.linalg.pinv":   lambda args: args[0],
    "np.linalg.cholesky": lambda args: args[0],
    "np.linalg.qr":     lambda args: "tuple<" + args[0] + ", " + args[0] + ">",
    "np.linalg.svd":    lambda args: "tuple<" + args[0] + ", vec_f, " + args[0] + ">",

    # 矩阵构造
    "np.eye":      lambda args: promote_to_mat(args[0]),   # int → mat_f
    "np.zeros":    lambda args: promote_to_mat_or_vec(args[0]),
    "np.diag":     lambda args: promote_to_mat(args[0]),

    # 数学函数
    "np.sqrt":     lambda args: args[0],
    "np.exp":      lambda args: args[0],
    "np.log":      lambda args: args[0],

    # Python 内置
    "len":         lambda args: "int",
    "range":       lambda args: "list<int>",
    "zip":         lambda args: f"list<tuple<{','.join(args)}>>",

    # ... 90+ 个注册项
}

def infer_call_return_type(name: str, arg_types: list[str]) -> str:
    """给定被调用函数名和参数类型列表，推断返回类型。"""
    entry = _CALLABLE_RETURNS.get(name)
    if entry is None:
        return "any"      # 未知函数 → 兜底为 any
    if callable(entry):
        return entry(arg_types)   # lambda/函数 → 动态计算
    return entry                   # 字符串 → 直接返回
```

**ndarray 属性/方法的类型推断**：对于 `H.conj()`、`x.T`、`x.shape` 这样的 numpy 数组属性访问：

```python
# ndarray 属性 → 类型（第 790–830 行）
_NDARRAY_PROPERTY_TYPES = {
    "T":    lambda t: t,              # 转置不改变类型
    "H":    lambda t: t,              # 共轭转置不改变类型
    "real": lambda t: promote_to_real(t),  # .real → 去掉复数部分
    "imag": lambda t: promote_to_real(t),  # .imag → 去掉复数部分
    "shape": lambda t: "tuple<int, int>",  # .shape → 元组
    "ndim": lambda t: "int",
}

def ndarray_property_type(owner_type: str, attr: str) -> str | None:
    """推断 ndarray 属性访问的结果类型。"""
    entry = _NDARRAY_PROPERTY_TYPES.get(attr)
    if entry is not None:
        return entry(owner_type)
    return None
```

#### 1.3.5 组合类型的解析与构造

组合类型使用轻量级的字符串编码，并通过正则表达式解析：

```python
# algorithm_ir/ir/type_lattice.py, 第 320–380 行
_COMPOSITE_RE = re.compile(
    r"^(tuple|list|dict)<(.+)>$"
)

def parse_composite(t: str):
    """解析组合类型字符串。返回 (kind, components)。"""
    m = _COMPOSITE_RE.match(t)
    if not m:
        return None, []   # 不是组合类型
    kind = m.group(1)
    inside = m.group(2)
    if kind == "tuple":
        return "tuple", [s.strip() for s in inside.split(",")]
    else:
        return kind, [inside.strip()]

def is_subtype_composite(a: str, b: str) -> bool:
    """组合类型的子类型检查。协变规则。"""
    ka, ca = parse_composite(a)
    kb, cb = parse_composite(b)
    if ka != kb:
        return False
    if ka == "tuple" and len(ca) != len(cb):
        return False  # 元组长度必须相同
    for ca_i, cb_i in zip(ca, cb):
        if not is_subtype(ca_i, cb_i):
            return False
    return True
```

#### 1.3.6 numpy 感知的类型代数

系统的深层设计原则是**与 numpy 的类型系统对齐**。类型格子不是凭空设计的，而是精确地反映了 numpy 在运行时的 dtype 提升规则和广播语义。

**标量与张量的交互**：`"int"` 同时是 `"vec_i"`、`"mat_f"` 和 `"tensor3_f"` 的父类型。这初看可能反直觉——标量"小于"向量？实际上是：`"int" <: "vec_i"` 意味着"在需要向量的地方，标量也可以安全使用"（numpy 会自动将标量广播为向量）。这是类型格子的"安全近似"语义：**只要运行时不报错，类型系统就应该认为它是合法的**。

**dtype 提升规则**：`i → f → c` 精确对应 numpy 的 `result_type(int, float) → float64`、`result_type(float, complex) → complex128`。

**矩阵乘法的秩代数**：矩阵乘法 `(m,k) @ (k,n) → (m,n)` 被编码为秩的代数规则：
- `rank(result) = rank(lhs) + rank(rhs) - 2`
- 标量×标量 → 标量：`0+0-2<0 → rank=0`
- 矩阵×矩阵 → 矩阵：`2+2-2=2 → rank=2`
- 矩阵×向量 → 向量：`2+1-2=1 → rank=1`
- 向量×向量 → 标量（内积）：`1+1-2=0 → rank=0`


3. [IR 嫁接机制详解](#3-ir-嫁接机制详解)
4. [GNN 对 IR 的编码方式](#4-gnn-对-ir-的编码方式)
5. [通过 RL 训练 GNN](#5-通过-rl-训练-gnn)

---

## 2. 双重进化框架概览

### 2.1 核心思想：宏观与微观双层进化

本系统的进化算法分为**两个层次**，它们以不同的时间尺度和不同的操作对象同时运行：

| 维度 | 宏观进化（Macro） | 微观进化（Micro） |
|------|-----------------|-----------------|
| **操作对象** | 算法骨架（完整的 FunctionIR） | 插槽内部的 IR 程序片段 |
| **变异方式** | GNN 引导的结构嫁接 + 基因组交叉 | 类型安全的 GP 算子（变异、交叉、子树替换） |
| **频率** | 每一代都进行 | 每代对前 20 个最优基因组运行 |
| **目标** | 探索不同的算法结构（骨架多样性） | 精细优化每个骨架内部的实现 |

**为什么要分两层？** 单层进化面临一个根本矛盾：如果你想改变算法的整体结构（比如把 MMSE 检测器改装成 BP 检测器），你需要做大规模的 IR 编辑（嫁接整个区域）。但如果你只是想微调（比如把 `x * 2 + y` 改成 `x * 1.5 + y / 2`），大规模编辑会引入太多不相关的改动。双层设计让"大改"和"微调"各司其职。

### 2.2 核心数据结构

#### 2.2.1 `AlgorithmGenome` — 进化个体

```python
# evolution/pool_types.py, 第 210 行
@dataclass
class AlgorithmGenome:
    algo_id: str                            # 唯一标识符（如 "algo_a1b2c3d4"）
    ir: FunctionIR                          # 唯一的、规范化的、扁平标注的 IR
    slot_populations: dict[str, SlotPopulation]  # 每个插槽的微观进化变体历史
    generation: int                         # 出生代
    parent_ids: list[str]                   # 父本 id 列表
    graft_history: list[GraftRecord]        # 嫁接历史记录
    tags: set[str]                          # 标签
    metadata: dict[str, Any]                # 可扩展元数据
```

每个 `AlgorithmGenome` 持有一个**单一的** `FunctionIR` 作为权威表示。之前系统有 "structural_ir" 和 "materialized_ir" 两个副本，后来统一为单一 IR（"single-representation principle"）以避免两者不同步的问题。

#### 2.2.2 `SlotPopulation` — 插槽子种群

```python
# evolution/pool_types.py, 第 164 行
@dataclass
class SlotPopulation:
    variants: list[FunctionIR]   # 该 Slot 的多个实现变体
    fitness: list[float]         # 每个变体的适应度（SER）
    best_idx: int                # 当前最优变体的索引
```

每个 Slot（`with slot("name"):` 标记的代码块）维护一个独立的子种群，其中包含了该 Slot 的多个不同 IR 实现。这使得系统可以在不改变算法整体结构的前提下，对每个插槽的代码进行局部优化。

#### 2.2.3 `GraftProposal` — 嫁接提议

```python
# evolution/pool_types.py, 第 389 行
@dataclass
class GraftProposal:
    proposal_id: str
    host_algo_id: str           # 宿主算法（被替换部分的来源）
    donor_algo_id: str | None   # 捐赠者算法（替换内容的来源）
    region: RewriteRegion       # 要被替换的 IR 区域
    donor_ir: FunctionIR | None # 捐赠者的 IR（要插入的内容）
    case: str                   # "I" | "II" | "III"
    attribution_slot_pop_key: str | None
    provenance: dict[str, Any]  # 来源跟踪
```

### 2.3 算法引擎的运行循环

`AlgorithmEvolutionEngine.run()`（`evolution/algorithm_engine.py` 第 134 行）是主循环。每一代执行以下步骤：

#### 步骤 1: 微观进化（Micro-evolve）

对种群中适应度最高的 20 个基因组，逐个对其插槽（Slot）运行类型安全 GP，进行精细优化。

微观进化的内部管道（来自 `evolution/slot_evolution.py` 第 689 行）：
1. 从 `genome.slot_populations` 中取出某个 Slot 的子种群
2. 调用 `micro_population_step()` 运行若干代微型 GP
3. 通过 `apply_slot_variant()` 将变异 IR 拼接到基因组中并评估 SER
4. 选择最佳变体写回 `genome.ir`

#### 步骤 2: 宏观繁殖

锦标赛选择 → 基因组交叉（交换 Slot 实现）→ 点变异。

#### 步骤 3: GNN 引导的结构嫁接（Track A）

GNN 编码所有算法 → 打分所有 (host, donor) 对 → 采样边界 → 匹配 donor → 构建 GraftProposal。

#### 步骤 4: Case I/II/III 分类嫁接

根据 donor 区域与 Slot 边界的关系，分三种情况处理。

#### 步骤 5-7: 评估 → 小生境选择 → 周期性操作

适应度函数为：$\text{fitness} = \text{SER} + \lambda \cdot \frac{\text{complexity}}{\text{max\_complexity}}$。

小生境选择保留前 50% 精英 + 每个算法族至少一个代表。

周期性：Graft pass（每 5 代）、Track B 自动发现 Slot（每 20 代）、GNN 训练（每 N 代）。

---

## 3. IR 嫁接机制详解

### 3.1 什么是嫁接（Grafting）？

**嫁接**这个词借用了园艺学的概念：将一棵树（捐赠者，donor）的枝条（一个 IR 代码片段）切下来，接到另一棵树（宿主，host）的树干上，让它们在接口处愈合。在 IR 层面，嫁接是将一个算法的**子图区域**替换为另一个算法的**对应子图区域**的操作。

与传统的函数调用（`call` 指令，产生一个不透明的黑盒）不同，本系统使用**内联嫁接**——捐赠者的每一个 op 都直接克隆到宿主 IR 中，成为宿主的一部分。这样后续的进化操作可以继续修改这些 op，使嫁接真正成为"进化"而非"组合"。

### 3.2 嫁接提议（Graft Proposal）的生成：完整的 8 层 Mask 管线

在 `graft_general()` 执行 IR 手术之前，GNN 必须先生成一个**嫁接提议**（`GraftProposal`），指定"替哪个算法（host）的哪个区域（region），用什么算法（donor）的什么代码来替换"。这个提议的生成过程本身就是一套复杂的采样+约束系统，涉及 **host 侧 4 层 mask** 和 **donor 侧 4 层 mask**（外加 1 层预过滤），共 9 个安全网。

#### 3.2.1 Host 侧：4 层 Mask 约束输出值和 Cut 值的采样

Host 侧的改造区域（`RewriteRegion`）由两个部分组成：**输出值**（`output_values`，区域的"出口"，即该区域产生哪些值）和 **cut 值**（`cut_values`，区域的"入口"，即从宿主 IR 的其余部分"切断"哪些数据流）。

**Layer 1（一次性预过滤）：死代码输出过滤**

`filter_dead_code_outputs()`（`evolution/host_region_mask.py` 第 55 行）在采样开始前运行一次，从候选 observable value 池中过滤掉那些"不在函数返回切片上"的值。如果一个值无论如何都不会影响函数的最终输出（即它处于死代码路径上），那么选取它为输出值是无意义的——嫁接后的区域也会是死代码。

实现方式：预先计算函数的返回切片（`_compute_return_slice_values(ir)`，从 `return`、`branch`、`store`、`set_item` 等副作用 op 出发做 backward slice）。只有在这个切片上的值才进入候选池。

**Layer 2（逐步输出 mask）：连通性不变式 + 可行性贪心前瞻**

`output_step_mask()`（第 188 行）是逐步调用的——每次 GNN 要选择下一个输出值时，调用此函数来 mask 掉不合法的候选。

对每个候选值 v（`remaining` 列表中的值），这个 mask 检查三项条件：

*条件 1 — 类型与签名兼容*：虽然 host 侧的输出类型约束不如 donor 侧严格（host 是 upper bound 而非 equality），但候选值的 `type_hint` 必须与 `BoundarySignature` 中对应位置的 `exit_types` 兼容（即 `is_subtype(cand_type, exit_type)`）。

*条件 2 — 连通性检查*：`selected ∪ {v}` 的 backward closure（通过 `op_closures` 中的 frozenset 并集计算）在 undirected dataflow 邻接中必须是连通的。这是**硬结构约束**：cut 只能**减少** op 集合（通过切断入口），不能**连接**两个孤岛。如果当前选择的输出值形成了一个不连通的 op 集合，以后无论加多少 cut 都无法挽回。

*条件 3 — 可行性贪心前瞻*：调用 `_combo_is_feasible(ir, prov_outs, ...)`，模拟"如果这些输出值被选中了，在预算内（max_cut_budget）能否通过加 cut 来使区域满足所有数值约束（op 数量、输出数量、输入数量）"。这个贪心前瞻使用 Plan 3 的 `op_closures` 快速路径（见 `evolution/host_region_mask.py` 第 290 行）。

- **Step 0**（尚未选择任何输出值）：只对每个候选值做可行性检查，不检查连通性（单个值的闭包自身必然连通）。
- **Step ≥1**：三个条件全部应用。此外，还维护一个模块级的 `_singleton_cut_cache`（第 48 行）来缓存每个值的单例 cut 候选列表，避免重复调用 `enumerate_cut_candidates`。

**Layer 3（cut 候选池生成）**：`enumerate_cut_candidates(ir, output_values, require_connected=True)`

在输出值选定后，枚举所有可以作为 cut 的候选值。cut 值必须满足：
- 在 output_values 的 backward closure 中
- 不在 output_values 中
- 实际能"截断"一部分 backward slice（`_actually_truncates` 检查）
- 加上这个 cut 后区域仍然连通（`require_connected=True` 时进行 BFS 验证）
- 不是平凡值（trivial value，如由 `const`、`get_attr`、`assign` 等产生的值）

**Layer 4（逐步 cut mask + STOP 门）：数值不变式**

`cut_step_mask()`（第 399 行）在每次 GNN 要选择下一个 cut 值时调用。

对每个候选 cut c，检查：

*条件 1 — 连通性*：`backward_slice_until_values(outputs, selected_cuts ∪ {c})` 必须产生非空且连通的 op 集合。

*条件 2 — 尺寸下界*：新的 `_nontrivial_op_count` 必须 ≥ `min_region_ops`（cut 只能减少 op，如果已经太少就无法挽回）。

*条件 3 — 贪心前瞻可行性*：调用 `_combo_is_feasible(initial_cuts=selected_cuts ∪ {c})`，模拟在剩余预算内能否通过更多 cut 使区域满足约束。这个前瞻调用还会收到一个**预缓存的 cut pool**（通过 Plan 1 从 GNN 侧传入），避免重新枚举 cut 候选。

**STOP 门逻辑**：

```python
# 如果当前状态（不增加更多 cut）已经满足所有约束：
#   - 连通
#   - min_region_ops ≤ size ≤ max_region_ops
#   - inputs ≤ max_region_inputs
#   - exits ≤ max_region_outputs
# 则 stop_allowed = True，采样器可以选择 STOP（不再加更多 cut）
```

**`_combo_is_feasible` 的贪心算法**（host 侧和 donor 侧共用同一模式，第 290 行）：

1. **下界检查**：将**所有**可能的 cut 同时应用（`backward_slice_until_values(outputs, all_cuts)`），检查最小区域是否满足 `max_region_ops` 和 `max_region_outputs`。如果不满足，任何 cut 子集也无法满足，直接返回 False。
2. **贪心迭代**（最多 `max_cut_budget` 次）：每一步选择能**最大程度减少 deficit** 的那个 cut：
   - 对每个候选 cut c，计算 `new_state = _state_arity(outputs, cur_cuts ∪ {c})`
   - `deficit = (exits超限) + (ops超限) + (ops不足) + (inputs超限)`
   - `score = cur_deficit - new_deficit`，选 score 最大的 cut
   - 如果某步后 `_feasible`，提前返回 True
3. **Plan 3 快速路径**：如果 `op_closures` 可用且 base_union（所有输出值的闭包并集）已连通、exit 上界 ≤ max_region_outputs、size 上界 ≤ max_region_ops，则跳过每候选的连通性/exit/size 检查（但仍做 BFS 用于评分）。

#### 3.2.2 Host 区域构建与验证

采样完成后，通过 `define_rewrite_region(output_values, cut_values)` 构建 `RewriteRegion`，然后通过 `validate_boundary_region()` 验证：

- 区域非空
- 输出值必须是 observable 的（在函数的 backward slice 上）
- 区域在 undirected dataflow 中必须连通
- `nontrivial_op_count` 在 [min_region_ops, max_region_ops] 之间
- 入口值数量 ≤ max_region_inputs
- 出口值数量 ≤ max_region_outputs

验证通过后，用 `infer_boundary_contract()` 提取边界的类型签名，生成 `BoundarySignature`——这是连接 host 和 donor 侧的桥梁。

#### 3.2.3 `BoundarySignature`：连接 Host 与 Donor 的类型桥梁

```python
# evolution/gnn_pattern_matcher.py, 第 ~1460 行
@dataclass
class BoundarySignature:
    entry_types: tuple[str, ...]  # host 区域入口值的类型（按位置排列）
    exit_types: tuple[str, ...]   # host 区域出口值的类型（按位置排列）
```

`BoundarySignature` 是从 host 区域推导出的类型契约。它告诉 donor 侧："你需要提供一个区域，它的**输出值**（对应 donor 的 exit）与 host 的**入口类型**在数量和类型上匹配，它的**入口值**（对应 donor 的 entry，即 cut）与 host 的**出口类型**匹配"。

注意这里的映射关系是**交叉的**：

| 对接关系 | Host 侧 | Donor 侧 | 约束 |
|---------|--------|---------|------|
| Host 出口 → Donor 入口 | `exit_types` | donor 的 entry（cut） | type 兼容 + 数量相等 |
| Host 入口 → Donor 出口 | `entry_types` | donor 的 exit（output） | type 兼容 + 数量相等 |

#### 3.2.4 Donor 侧：4 层 Mask + 1 层预过滤

Donor 侧的采样结构与 host 侧镜像对称，但多了一层签名预过滤，且约束更严格（等号约束，而非上界）。

**Layer D1（池级签名兼容性预过滤）**

`donor_pool_signature_compatible()`（`evolution/donor_region_mask.py` 第 105 行）在采样开始前检查：对给定的 `BoundarySignature`，该 donor IR 是否有**至少一个**类型兼容的输出值候选和 cut 值候选。

```python
def donor_pool_signature_compatible(donor_ir, observable_values, cut_pool_union,
                                     *, entry_types, exit_types):
    # 对于每个 host exit_types（→ donor output 需要匹配）：
    #   至少有一个 donor observable value 类型兼容
    for t in exit_types:
        if not any(_types_compatible(ot, t) for ot in obs_types):
            return False   # 没有兼容的输出值 → 这个 donor 不可能工作
    # 对于每个 host entry_types（→ donor cut 需要匹配）：
    #   至少有一个 donor cut 候选类型兼容
    for t in entry_types:
        if not any(_types_compatible(ct, t) for ct in cut_types):
            return False
    return True
```

这个预过滤器在 O(|values| × |types|) 时间内剔除 ~70-90% 的池级不匹配，避免了昂贵的逐步采样。

**Layer D2（逐步 donor 输出 mask）**

`donor_output_step_mask()`（第 351 行）与 host Layer 2 对应，但约束更强：donor 的 exit 数量必须**精确等于** `len(host.entry_types)`（等号约束），因为每个 host 入口需要一个对应的 donor 出口值。

对每个候选值 vid，检查：

1. **类型兼容**：`_types_compatible(cand_t, exit_types[next_step])`
2. **连通性**：`selected ∪ {vid}` 的 closure 是连通的（与 host Layer 2 相同）
3. **可行性贪心前瞻**：调用 `_donor_combo_is_feasible()`——这是一个与 host 侧 `_combo_is_feasible` 结构相同但约束不同的贪心算法（因为 donor 有 `target_entries` 和 `target_exits` 的精确等号约束，而非上界）。

**桥检测优化**（Plan 5，第 385-432 行）：在 step ≥1 时，连通性检查不再需要每个候选做完整 BFS。系统预计算 `base_ops` 的"边界 op 集合"（与 `base_ops` 有数据流边的外部 op），然后对每个候选只需检查其 `cand_ops`（一个 frozenset）与边界 op 集合是否有交集（`cand_ops.isdisjoint(_boundary_ops)`——O(|cand_ops|) 的短路检查）。这是因为 `base_ops` 已连通、`cand_ops`（单值闭包）先天连通，两者并集连通的充要条件是它们之间有至少一条数据流边。

**Layer D3（donor cut 候选池）**：与 host Layer 3 相同，通过 `_get_cut_context()` → `enumerate_cut_candidates()` 生成。

**Layer D4（逐步 donor cut mask + STOP 门）**

`donor_cut_step_mask()`（第 416 行）与 host Layer 4 对应，核心差异在于：

- **类型约束**：候选 cut 必须与 `BoundarySignature.entry_types[next_step]` 位置匹配
- **等号约束**：最终状态的 `exits == target_exits` 且 `entries == target_entries`（而非上界）
- **STOP 门**：允许 STOP 当且仅当当前区域满足所有约束且已类型匹配

对每个候选 cut c，还会调用 `_donor_combo_is_feasible(initial_cuts=selected_cuts ∪ {c})` 做贪心前瞻。

**`_donor_combo_is_feasible` 的特殊性**（第 198 行）：

- 与 host 侧版本相同的基础贪心框架
- 但使用了 Plan 4 的 `op_closures` 快速路径和合并遍历的 `_state_arity_fast`（第 181 行，将 `_entry_values`、`_exit_values`、`_nontrivial_op_count`、`is_op_set_connected` 的多次图遍历合并为一次）
- `_state_arity` 现在返回 5 元组 `(n_entries, n_exits, n_ops, n_inputs, connected)` 而非 4 元组，消除了冗余的 BFS 调用（Plan 4 双 BFS 修复）

#### 3.2.5 Whole-Pipeline：从一对 (host, donor) 到 GraftProposal

整个管线的完整流程（`gnn_pattern_matcher.py` 第 967 行 `_propose_pairs` 中）：

```
对每对 (host, donor) 入选对：

  ┌─ Host 侧 ──────────────────────────────────────────────┐
  │ Layer 1: filter_dead_code_outputs(ir)                   │
  │   → 候选 observable value 池                            │
  │                                                         │
  │ 迭代选择输出值（每次选一个）：                               │
  │   GNN BoundaryRegionPolicy.output_logits() → softmax 分数 │
  │   Layer 2: output_step_mask(selected, remaining, ...)    │
  │   →  mask 掉不合法的候选，剩下合法的候选中 softmax 采样     │
  │                                                         │
  │ Layer 3: enumerate_cut_candidates(ir, selected_outputs)  │
  │   → 候选 cut value 池                                    │
  │                                                         │
  │ 迭代选择 cut 值（每次选一个）：                              │
  │   GNN BoundaryRegionPolicy.cut_logits() → softmax 分数    │
  │   Layer 4: cut_step_mask(selected_cuts, remaining, ...)  │
  │   → mask + STOP 门                                       │
  │                                                         │
  │ define_rewrite_region(outputs, cuts) → RewriteRegion     │
  │ validate_boundary_region() → 验证                        │
  │ infer_boundary_contract() → BoundarySignature            │
  └─────────────────────────────────────────────────────────┘
                          ↓
        BoundarySignature = {entry_types, exit_types}
                          ↓
  ┌─ Donor 侧 ──────────────────────────────────────────────┐
  │ Layer D1: donor_pool_signature_compatible(ir, sig)      │
  │   → 快速剔除不兼容的 donor（~70-90% 对）                   │
  │                                                         │
  │ 迭代选择 donor 输出值（位置对应 sig.exit_types）：           │
  │   GNN BoundaryRegionPolicy.output_logits() → softmax 分数 │
  │   Layer D2: donor_output_step_mask(selected, ...)        │
  │   → 类型匹配 + 连通性 + 可行性贪心前瞻                      │
  │                                                         │
  │ Layer D3: enumerate_cut_candidates(donor_ir, outputs)    │
  │   → 候选 cut pool                                        │
  │                                                         │
  │ 迭代选择 donor cut 值（位置对应 sig.entry_types）：          │
  │   GNN BoundaryRegionPolicy.cut_logits() → softmax 分数    │
  │   Layer D4: donor_cut_step_mask(selected_cuts, ...)      │
  │   → 类型匹配 + 连通性 + 等号检查 + STOP 门                  │
  └─────────────────────────────────────────────────────────┘
                          ↓
                GraftProposal = {
                    host_algo_id, donor_algo_id,
                    region (RewriteRegion),
                    donor_ir (FunctionIR),
                    contract (BoundaryContract),
                    case (I/II/III)
                }
```

**重试机制**：如果采样过程中某个步骤遇到"所有候选都被 mask 掉"（`all-zero mask`），donor 采样器会**重新采样**（最多 `donor_sample_max_retries=10` 次，每次使用不同的随机种子来改变 GNN 的输出值采样和 cut 采样路径）。

### 3.3 `graft_general()` 的 13 步管线

GraftProposal 生成后，`graft_general()`（`algorithm_ir/grafting/graft_general.py` 第 769 行）执行实际的 IR 手术。

#### 步骤 1: 深度复制宿主 IR

```python
new_ir = deepcopy(host_ir)   # 或 _manual_clone_ir，如果 xDSL 不可 pickle
rebuild_def_use(new_ir)      # 确保 SSA def/use 元数据一致
```

#### 步骤 2: 定位要替换的区域

验证 `region.op_ids` 中的所有 op 在宿主 IR 中都存在。

#### 步骤 3: 边界分析

调用 `find_region_boundary()` 识别区域的入口值和出口值。入口值 = 由区域外部 op 定义、区域内部 op 消费的值。出口值 = 由区域内部 op 定义、区域外部 op 消费的值。

如果有 `host_contract`（来自 `BoundarySignature`），优先使用合同中的标准化端口。

#### 步骤 4: 克隆捐赠者 IR

`clone_donor_ir()` 为捐赠者的每个 Value/Op/Block 赋予新 UUID，维护 `id_map: 旧donor_id → 新host_id`。

#### 步骤 5-6: 多级端口绑定

四种回退策略，优先级从高到低：

1. **显式端口映射**（`proposal.port_mapping`）：GNN 明确指定了绑定关系
2. **类型化二分图绑定**（`bind_typed()`，使用匈牙利算法找最小成本匹配）：
   ```
   成本矩阵 C[i][j] = 将 donor_arg[i] 绑定到 host_val[j] 的代价
   C[i][j] = 0    如果 types_compatible  (完美)
   C[i][j] = 1    如果 unify ≠ "any"     (勉强)
   C[i][j] = ∞    如果类型互不兼容         (不可能)
   ```
3. **命名提示匹配**：通过 `name_hint` / `var_name` / `name` 属性匹配
4. **位置匹配**：按参数顺序直接对应

#### 步骤 7: 移除捐赠者的 return 操作

捐赠者的 return op 被删除，但其输入值（返回值）被保留为嫁接后的出口值。

#### 步骤 8: 内联捐赠者块

- **单块**：直接插入到 host 区域位置
- **多块**：将宿主块分裂为 pre_block + post_block，插入 donor 的 CFG 子图，通过 jump 指令连接

#### 步骤 9-10: 重定向 + 清理

将外部消费者重定向到捐赠者结果，删除区域 op 和死值。

#### 步骤 11-13: 后处理

拓扑排序、依赖覆盖、检测新 Slot。

### 3.4 嫁接的类型安全网

| 检查层 | 位置 | 检查内容 |
|--------|------|---------|
| GNN 提议 | `_propose_pairs` | Type lattice 过滤不兼容的 host-donor 对 |
| Host mask L2 | `output_step_mask` | 连通性 + 可行性贪心前瞻 |
| Host mask L4 | `cut_step_mask` | 连通性 + 数值约束 + STOP 门 |
| Donor mask D1 | `donor_pool_signature_compatible` | 签名类型预过滤 |
| Donor mask D2 | `donor_output_step_mask` | 类型 + 连通性 + 可行性 |
| Donor mask D4 | `donor_cut_step_mask` | 类型 + 连通性 + 等号 + STOP |
| 区域构建 | `define_rewrite_region` + `validate_boundary_region` | 结构一致性 |
| 端口绑定 | 匈牙利算法 | 类型化二分图最优匹配 |
| 嫁接执行 | `graft_general` | 多级端口绑定回退 |
| 嫁接后 | `validate_function_ir` | SSA 一致性、CFG 完整性、Slot 元数据 |



---

## 4. GNN 对 IR 的编码方式

### 4.1 什么是图神经网络（GNN）？

**图神经网络**（Graph Neural Network, GNN）是一种专门处理**图结构数据**的深度学习模型。与传统的神经网络不同（输入是固定大小的向量或网格，如图像），GNN 可以直接处理由节点和边组成的图。GNN 的每一层通过**消息传递**（Message Passing）来更新每个节点的表示：每个节点从它的邻居节点接收"消息"（邻居的特征），聚合这些消息，然后更新自己的特征。

在数学上，对于节点 $i$，第 $\ell+1$ 层的特征 $h_i^{\ell+1}$ 计算为：

$$h_i^{(\ell+1)} = \sigma\left(W^{(\ell)} \cdot \text{AGGREGATE}\left(\{h_j^{(\ell)} : j \in \mathcal{N}(i)\}\right)\right)$$

其中 $\mathcal{N}(i)$ 是节点 $i$ 的邻居集合，$\sigma$ 是激活函数（如 ReLU、ELU），$W^{(\ell)}$ 是可学习的权重矩阵。AGGREGATE 决定了如何合并邻居信息——可以是平均值（GCN）、加权注意力（GAT）等。

### 4.2 IR 到图的转换：`ir_to_graph()`

系统使用 PyTorch Geometric 库。`ir_to_graph(ir)` 函数（`evolution/gnn_pattern_matcher.py` 第 196 行）将一个 `FunctionIR` 转换为一个 `Data` 对象（PyG 的图数据结构）。

**节点**：IR 中的每个**非平凡 op**（non-trivial op）。平凡 op（如 `const`, `get_attr`, `assign` 和所有输入相同的 `phi`）被跳过——它们不携带算法结构性信息，只是"管道"或"常量"。GNN 通过 `visible_def_ops()` 函数直接"看穿"这些平凡 op。

**边**：数据流边。如果 op A 的输出被 op B 消费（经过 `visible_def_ops` 的传递闭包），则存在一条从 A 到 B 的**有向边**。这些边是**无向化**的（GAT 在 IR 的 undirected dataflow 邻接上操作，因为数据依赖天然是双向的——A 产生和 B 消费的数据）。

**节点特征**（每个节点是一个 $D = 53$ 维的实数向量）：

特征的构成（第 224-235 行）：

1. **操作码的 one-hot 编码**（21 维）：对应 2.2 节中列出的 21 种操作码。每个节点的一位是 1（表示该节点的 opcode），其余位是 0。这告诉 GNN 这个节点在做什么类型的计算。

2. **被调用函数的哈希**（`_CALLEE_FEATURES = 16` 维）：对于 `call` 类型的 op，被调用的函数名（如 `"np.linalg.inv"`、`"scipy.linalg.solve"`）被哈希为一个 16 维的浮点向量。相似名字的函数产生相似的哈希向量。对于非 `call` 类型的 op，这 16 维全为 0。

3. **来源/插槽归属的哈希**（`_PROV_FEATURES = 8` 维）：如果该 op 属于某个 Slot（通过 `_provenance` 元数据），则编码 Slot ID 的哈希。此外还有一个标志位指示该 op 是否位于 Slot 边界上。这告诉 GNN 哪些 op 是可替换的，哪些是结构性的骨架。

4. **图元特征**（8 维）：ir_to_graph 为图添加了以下全局元特征：
   - `op_count`：图中非平凡 op 的数量
   - `block_count`：基本块的数量
   - `has_branch` / `has_phi` / `has_loop`：是否存在条件分支 / φ 节点 / 循环
   - `slot_k` / `slot_k_boundary`：Slot 相关统计

**图的输出**：`ir_to_graph()` 返回一个 `Data` 对象，包含：
- `x`: 节点特征矩阵，形状为 `(num_nodes, 53)`
- `edge_index`: 边列表，形状为 `(2, num_edges)`——每列 `(src, dst)` 是一条边

### 4.3 GNN 编码器：`IRGraphEncoder`

```python
# evolution/gnn_pattern_matcher.py, 第 300-327 行
class IRGraphEncoder(nn.Module):
    def __init__(self):
        # 两层图注意力卷积（GATConv），4 个注意力头
        self.conv1 = GATConv(node_dim=53, hidden=64, heads=4)
        self.conv2 = GATConv(hidden=64, hidden=64, heads=4)
        self.fc = nn.Linear(64, 32)           # 图级投影（embedding）
        self.node_proj = nn.Linear(64, 16)    # 节点级投影（policy 用）
```

**图注意力卷积（GATConv）** 与标准的图卷积不同：它不是对所有邻居一视同仁，而是**自动学习**哪些邻居更重要。注意力权重 $\alpha_{ij}$ 表示"在处理节点 i 时，邻居 j 的信息有多重要"：

$$\alpha_{ij} = \frac{\exp(\text{LeakyReLU}(a^T[Wh_i \| Wh_j]))}{\sum_{k \in \mathcal{N}(i)} \exp(\text{LeakyReLU}(a^T[Wh_i \| Wh_k]))}$$

其中 $h_i, h_j$ 是节点特征，$W$ 是共享的线性变换，$a$ 是可学习的注意力向量，$\|$ 表示向量拼接。

**Forward 过程**（编码一个算法为固定长度的向量）：

1. 输入图通过两层 GATConv（ELU 激活）
2. `global_mean_pool` 对所有节点的最终特征取**平均值** → 得到一个 64 维的**图级向量**（graph-level embedding）。这个向量是整个算法 IR 的紧凑表示（32 维）
3. 同时，通过 `node_proj` 为每个节点生成 16 维的**节点级向量**

**编码向量 `h` 的含义**：32 维的图向量 `h = encoder(ir)` 是对该算法 IR 的"压缩摘要"。如果两个算法有相似的结构（相似的操作类型分布、相似的数据流模式），它们的编码向量 $h_1$ 和 $h_2$ 在欧几里得空间中会比较接近。

**批处理**：由于不同的 IR 有不同的节点数和边数，`ir_to_graph` 为每个算法生成独立的 `Data` 对象，然后通过 PyG 的 `Batch` 机制将它们合并为一个大图。`global_mean_pool` 根据批处理索引将大图中的节点分离回各自的图，然后对每个图的节点取平均。

### 4.4 对打分器与策略网络

#### 4.4.1 `GraftScorer` — 对兼容性打分

```python
# 第 330-366 行
class GraftScorer(nn.Module):
    def forward(self, host_emb, donor_emb):
        h = trunk(cat([host_emb, donor_emb]))   # 64 维
        return score_head(h)                     # 1 维标量
```

**输入**：两个 32 维向量 `host_emb` 和 `donor_emb`（分别是宿主和捐赠者的图级嵌入）
**处理**：拼接为 64 维向量 → 两层 MLP（ReLU 激活）
**输出**：一个标量分数（score）。**分数越低表示这对 host-donor 越适合嫁接**。

打分器同时输出三个辅助预测（用于训练监督）：
- `reasonable_logit`：预测该嫁接是否能通过 reasonable 检查
- `behavior`：预测嫁接后行为是否会改变
- `perf`：预测嫁接后的性能

#### 4.4.2 `BoundaryRegionPolicy` — 采样边界值

```python
# 第 369-433 行
class BoundaryRegionPolicy(nn.Module):
    def encode_values(self, value_feats):         # 编码候选值的特征
    def output_logits(self, ctx_emb, candidate_emb, output_summary):
        # 为每个候选输出值产生选择概率
    def cut_logits(self, ctx_emb, candidate_emb, output_summary):
        # 为每个候选 cut 值产生选择概率
```

**输出策略（output policy）**：给定上下文嵌入（donor 的图向量），对宿主 IR 中的每个 observable value 产生一个选择概率。选中概率高的值更可能成为区域的输出边界。

**切割策略（cut policy）**：类似地，对候选 cut 值产生选择概率。cut 值决定了区域的"入口"——在 backward slice 中，cut 值标记了停止搜索引擎进一步扩展的点。

#### 4.4.3 `CriticHead` — 状态值估计

```python
# 第 369-387 行
class CriticHead(nn.Module):
    def forward(self, host_emb, donor_emb):
        return net(cat([host_emb, donor_emb]))   # 1 维标量 V(s)
```

**状态值函数 $V(s)$**：估计在给定一对 (host, donor) 的状态下，期望能获得的总奖励。这是一个 Actor-Critic RL 架构中的 Critic 部分。

### 4.5 整条 GNN 管线

1. `_encode_entries(entries)` → 每个算法 IR → `ir_to_graph()` → `IRGraphEncoder` → 32 维向量
2. `_score_pairs(entries)` → 对每对 (host, donor) 计算 `GraftScorer(emb_h, emb_d)`
3. `_select_pair_candidates(scores)` → 按分数排序选 top-K 对
4. `_propose_pairs(pairs)` → 对每对：
   - `BoundaryRegionPolicy.output_logits()` → 采样 host 输出值
   - `BoundaryRegionPolicy.cut_logits()` → 采样 host cut 值
   - `define_rewrite_region(outputs, cuts)` → 构建 RewriteRegion
   - 匹配 donor 区域 → `GraftProposal`

---

## 5. 通过 RL 训练 GNN

### 5.1 什么是强化学习（RL）？

**强化学习**（Reinforcement Learning）是一类机器学习方法，其核心思想是：一个**智能体**（agent）在**环境**（environment）中采取**动作**（action），环境给予**奖励**（reward），智能体通过反复试错来学习"什么动作在什么状态下是好的"。

与监督学习（需要标注好的输入-输出对）不同，RL 不需要"正确答案"，只需要一个奖励信号来告诉你"做得好不好"。这对于算法自动发现非常关键——我们不知道"最优的嫁接是什么"，但我们能通过运行仿真来判断嫁接后的算法好不好。

### 5.2 RL 公式化

在我们的系统中：

- **状态 $s$**：由 `IRGraphEncoder` 编码的 (host, donor) 对嵌入向量 $s = [e_{host}, e_{donor}]$
- **动作 $a$**：由 `BoundaryRegionPolicy` 选择的边界值序列（输出值 + cut 值）。每个动作是序列式的——先选第 1 个输出，再选第 2 个，然后选 cut 值
- **奖励 $r$**：嫁接后的 SER 改进量。基准线是 host 算法本身的 SER

具体地，对每个 selected host output $v_i$，奖励定义为：

$$r_{out}(v_i) = \begin{cases} +0.3 & \text{如果该输出是 feasible 的} \\ 0 & \text{否则} \end{cases}$$

对每个 selected cut $c_j$，奖励定义为：

$$r_{cut}(c_j) = \begin{cases} +0.3 & \text{如果加了这个 cut 后区域仍然 feasible} \\ R_{graft} & \text{如果是最后一个 cut（reward 是 graft 的最终 reward）} \\ 0 & \text{否则} \end{cases}$$

最终嫁接奖励 $R_{graft}$ 的计算方式（`_compute_graft_reward` 函数）：

$$R_{graft} = \underbrace{(1.0 - 2.0 \times \text{SER}_{child})}_{\text{基础性能}} + \underbrace{0.2 \times \mathbb{1}[\text{is\_valid}]}_{\text{不崩溃奖励}} + \underbrace{0.3 \times \mathbb{1}[\text{SER}_{child} < 0.5]}_{\text{比随机好}} + \underbrace{0.5 \times \mathbb{1}[\text{SER}_{child} < \text{SER}_{host}]}_{\text{改进奖励}}$$

各项的含义：
- **基础性能**：SER 越低（越好），奖励越高。SER=0 时奖励为 1，SER=0.5 时奖励为 0，SER=1 时奖励为 -1
- **不崩溃奖励**：如果算法执行时没有报错（`is_valid = True`），加 0.2
- **比随机好**：如果 SER < 0.5（比随机猜测好），加 0.3
- **改进奖励**：如果子代 SER 比宿主 SER 更好，加 0.5（这是最关键的正信号）

**总的嫁接奖励范围**：最好的情况（子代完美且不崩溃、比宿主好得多）= $1.0 + 0.2 + 0.3 + 0.5 = 2.0$。最坏的情况（子代崩溃）= $> -1$。

### 5.3 策略梯度（Policy Gradient）

GNN 的边界策略使用 REINFORCE 算法（一种基础的策略梯度方法）来训练。核心思想是：如果一个动作序列导致了好的奖励，就增加它被选中的概率；如果导致了差的奖励，就减少概率。

对于选中的输出序列 $a_{out} = (v_1, v_2, ..., v_k)$ 和 cut 序列 $a_{cut} = (c_1, c_2, ..., c_m)$，策略损失的数学形式为：

$$\mathcal{L}_{RL} = -\sum_{v \in a_{out}} \log \pi_{out}(v | s) \cdot (R - V(s))_+ - \sum_{c \in a_{cut}} \log \pi_{cut}(c | s) \cdot R_{graft}$$

其中 $\pi_{out}(v | s)$ 是策略网络为值 $v$ 分配的 softmax 概率，$V(s)$ 是 Critic 网络估计的状态值，$(R - V(s))_+ = \max(0, R - V(s))$ 是**优势函数**（Advantage）——只对"比预期好"的动作给予正梯度。

**直观解释**：如果策略选了一个输出值 $v$ 并且最终奖励 $R$ 比 Critic 预测的 $V(s)$ 大，那么 $v$ 被选中的概率会增加。如果 $R < V(s)$，则该动作的概率不会增加（但也不会减少——这留给了 entropy regularization）。

### 5.4 辅助损失（Auxiliary Losses）

除了策略梯度，总训练损失还包含以下辅助项：

#### 5.4.1 对分数回归（Pair Score MSE）

`GraftScorer` 的 score 头通过监督学习来拟合实际的嫁接 SER：

$$\mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^N w_i \cdot (\text{score}_i - \text{SER}_i)^2$$

其中 $w_i$ 是经验权重（最近的经验权重更高）。这允许 GNN 在不运行完整仿真的情况下**预测**嫁接的效果。

#### 5.4.2 辅助分类损失

Multi-head 头通过二元交叉熵来预测嫁接质量：

$$\mathcal{L}_{reasonable} = \text{BCE}(\text{reasonable\_logit}, y_{reasonable})$$
$$\mathcal{L}_{behavior} = \text{BCE}(\text{behavior\_logit}, y_{behavior})$$
$$\mathcal{L}_{perf} = \text{BCE}(\text{perf\_logit}, y_{perf})$$

其中 $y_{reasonable} = 1$ 如果嫁接实际通过了 reasonable 检查，$y_{behavior} = 1$ 如果行为改变了，$y_{perf} = 1$ 如果性能确实提升。

#### 5.4.3 Critic 值损失

Critic 通过 MSE 来学习状态值：

$$\mathcal{L}_{value} = \frac{1}{N} \sum_i (V(s_i) - R_i)^2$$

#### 5.4.4 Mask 辅助损失

对于 host 输出和 cut 采样的每一步，如果所有候选都被 mask 掉了（`invalid_mass = 1`），意味着没有可行的选择。loss 惩罚这种情况：

$$\mathcal{L}_{mask} = -\log(1 - \text{invalid\_mass\_mean})$$

这个项推动策略学习产生至少有一些候选不被 mask 掉的采样分布。

#### 5.4.5 熵正则化（Entropy Regularization）

$$\mathcal{L}_{ent} = -\lambda_{ent} \cdot H(\pi) = \lambda_{ent} \cdot \sum_v \pi(v) \log \pi(v)$$

其中 $H(\pi)$ 是策略的**熵**（entropy）。负熵损失鼓励策略保持一定的随机性，防止过早"锁定"到某个特定选择。

#### 5.4.6 合法熵（Legal Entropy）

$$\mathcal{L}_{legal\_ent} = -\lambda_{legal} \cdot H(\pi_{legal})$$

其中 $\pi_{legal}$ 是只在合法（未被 mask 掉）的候选上重新归一化的概率分布。鼓励策略在合法候选之间均匀探索。

### 5.5 总训练损失

将所有损失项加总：

$$\mathcal{L}_{total} = \mathcal{L}_{MSE} + \lambda_{value} \cdot \mathcal{L}_{value} + \lambda_{RL} \cdot \mathcal{L}_{RL}$$

$$+ \lambda_{reasonable} \cdot \mathcal{L}_{reasonable} + \lambda_{behavior} \cdot \mathcal{L}_{behavior} + \lambda_{perf} \cdot \mathcal{L}_{perf}$$

$$+ \lambda_{mask} \cdot \mathcal{L}_{mask} + \lambda_{margin} \cdot \mathcal{L}_{mask\_margin}$$

$$- \lambda_{ent} \cdot H(\pi) - \lambda_{legal} \cdot H(\pi_{legal})$$

训练通过从经验回放缓冲区（replay buffer）中采样一个 mini-batch，计算上述损失，然后通过标准的 PyTorch 自动微分和 Adam 优化器来更新所有网络参数。

### 5.6 训练调度

- **Warmstart 代**（前 `warmstart_gens` 代）：GNN 不训练，只是收集经验（探索）。`_score_pairs` 使用随机权重，`_select_pair_candidates` 均匀采样
- **稳态代**：每 `train_interval` 代训练一次（做 `train_steps` 步梯度更新）。经验从 replay buffer 中采样，优先选择最近的、高奖励的经验

经验记录（`record_outcome`）会被写入 replay buffer，包含 host/donor embedding、选择的边界值、reward 和 mask 信息。这使得 GNN 能够"记住"过去的嫁接尝试并从中学习。

