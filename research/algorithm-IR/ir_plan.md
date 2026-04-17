# 算法发现 IR 实现规格文档

## 0. 文档目的

本文档定义一套**可执行、可追踪、可再生、可重写**的中间表示（IR）与配套分析框架，用于支持如下研究目标：

1. 不在算法表示中预设任何高层结构，例如树、图、frontier、message passing substrate、factor graph 等。
2. 让用户或上层分析模块可以在统一 IR 上**精确剥离某一计算区域**，保留其边界与可执行语义，而不必先把该区域强行命名为某种高层结构。
3. 让系统能够在该区域上执行**删除、保留、插入、覆写、重连**等操作，从而把 donor skeleton graft 到 host algorithm 的某个局部计算过程中。
4. 若有需要，可以在区域之上附加多种后验结构视角（projection / motif）作为解释层或匹配层，但这不是 IR 本身的职责。
5. 新算法必须仍然落回同一 IR 与事实层（fact layer），从而继续进行分析、重写、结构标注与演化。

本文档不是理论论文，而是**工程实现规格**。目标读者是另一个 AI / 工程代理，它需要严格按本文档实现最小可用系统（MVP），并逐步扩展。

---

# 1. 系统总目标

## 1.1 核心原则

### 原则 A：底层 IR 必须“无结构”

底层 IR 不允许出现以下任何高层语义：

* Tree / Graph / SearchTree / ExpandTree
* Frontier / Candidate / Heuristic
* FactorGraph / Belief / MessagePassing
* Decoder / Detector / StackDecoder / SphereDecoder
* MIMO / Constellation / Posterior 等领域语义

底层 IR 只允许表达：

* 值（value）
* 操作（operation）
* 控制块（block）
* 调用上下文（frame）
* 数据依赖
* 控制依赖
* 运行时实例化
* 读写与生命周期事实

### 原则 B：结构发现不是底层 IR 的职责

系统不得在 IR 中显式写入“该算法具有树结构”。
底层 IR 只负责表达事实、边界与可执行语义；如果上层需要把某段计算解释为树、局部更新系统或候选池流程，应当在 IR 之上额外附加 annotation / projection，而不是把这些视角回写成底层真类型。

### 原则 C：同一算法允许有多个结构视角

系统必须允许：

* 同一个算法同时具有 branching projection、scheduling projection、local interaction projection、constraint refinement projection 等。
* projection 之间彼此竞争、重叠、并存，而不是唯一真相。

### 原则 D：分析对象与生成对象同构

算法分析产物与算法生成产物必须共享统一中间表示。
也就是说，新算法不能仅输出源码；必须输出：

* 统一 IR
* 可执行体
* 与 IR 对齐的 tracing schema
* projection 注释

这样新算法才能继续进入下一轮分析。

### 原则 E：Region Rewrite 是第一等公民

系统必须把“可切除、可插入、可重连的局部计算区域”作为核心操作对象。
也就是说，底层 IR 不仅要支持 tracing，还要支持：

* 指定某组 op / block 组成的 RewriteRegion
* 推断该区域的输入、输出、读写集合与调度锚点
* 在不破坏外围程序的前提下删除区域内部计算
* 插入 donor skeleton 生成的新状态、新更新与新连接

---

# 2. 系统总体架构

整个系统分为九层：

1. **Source Frontend**：将受限 Python 算法代码编译到 MiniIR。
2. **Static IR Layer**：表示静态值、操作、基本块、函数、控制流、调用关系。
3. **Instrumented Executor**：执行 MiniIR，并在运行时产生事件。
4. **FactGraph Builder**：融合静态与动态信息，构造统一事实图。
5. **RewriteRegion Builder**：从显式 op 集、source span、切片结果或用户指定位置构造可重写区域。
6. **BoundaryContract Inference**：为 RewriteRegion 推断输入端口、输出端口、读写槽位、状态承载者与调度锚点。
7. **Optional Projection Layer**：在 RewriteRegion 或 FactGraph 上附加结构视角，作为解释层或 donor 匹配层。
8. **Skeleton Override / Grafting Engine**：将其他 skeleton 按 BoundaryContract graft 到目标区域，并完成重连。
9. **IR Regenerator**：将 graft / override 后算法重新编译回统一 IR，并可执行。

MVP 阶段只要求实现 1–6 与 8–9 的最小闭环；7 为可选增强，而不是前置条件。

---

# 3. 开发阶段规划

## Phase 0：约束范围

只支持受限 Python 子集：

* 标量赋值
* 算术表达式
* 比较表达式
* if / else
* for / while
* 函数调用
* list / dict 的受限操作
* heapq 的最小封装
* dataclass / namedtuple 状态对象
* 不支持异常、yield、async、反射、动态 import、metaclass

## Phase 1：Source Frontend -> MiniIR

目标：从 Python AST 生成无结构 MiniIR。

## Phase 2：MiniIR Interpreter + Tracing

目标：实现解释器，并在每个 op 执行时输出 RuntimeEvent。

## Phase 3：FactGraph

目标：将静态 IR 与动态 trace 对齐为统一事实图。

## Phase 4：RewriteRegion + BoundaryContract

目标：能在统一 IR 上手工或半自动地定义一个 RewriteRegion，并推断其边界契约：

* 区域输入 / 输出
* 区域读写集合
* 状态承载者与可新增状态
* 调度锚点与重连位置

## Phase 5：Minimal Override / Grafting Demo

目标：在某个 RewriteRegion 上删除原有中间计算，插入一个简单 skeleton（例如局部 summary update）并再生 IR。

## Phase 6：Optional Projection Annotation

目标：在同一个 RewriteRegion 或 FactGraph 上附加最小的结构视角，作为解释层或 donor 匹配辅助，而不是 grafting 的唯一入口。

---

# 4. MiniIR 设计要求

## 4.1 总要求

MiniIR 必须满足：

* 可解释执行
* 可附加唯一 ID
* 能恢复 def-use
* 能表达控制流
* 能承载运行时实例化
* 能稳定支撑 region 级别切片与 rewrite
* 不带高层结构语义

## 4.2 数据模型

### 4.2.1 Value

表示静态值位点，而非运行时具体对象。

必备字段：

* `id: str`
* `name_hint: str | None`
* `type_hint: str | None`
* `source_span: tuple[int, int, int, int] | None`
* `def_op: str | None`
* `use_ops: list[str]`
* `attrs: dict`

说明：

* `type_hint` 只允许是低层类型，例如 `int`, `float`, `bool`, `list`, `dict`, `object`，不允许高层语义类型。
* `attrs` 用于存放 shape / literal / mutability 等非结构语义信息。

### 4.2.2 Op

表示静态操作。

必备字段：

* `id: str`
* `opcode: str`
* `inputs: list[str]`
* `outputs: list[str]`
* `block_id: str`
* `source_span: tuple[int, int, int, int] | None`
* `attrs: dict`

`opcode` 允许的最小集合：

* `const`
* `assign`
* `binary`
* `unary`
* `compare`
* `phi`
* `call`
* `get_attr`
* `set_attr`
* `get_item`
* `set_item`
* `build_list`
* `build_dict`
* `append`
* `pop`
* `iter_init`
* `iter_next`
* `branch`
* `jump`
* `return`

禁止使用带解释性的 opcode，如：

* `expand_node`
* `message_pass`
* `frontier_select`
* `stack_push`

### 4.2.3 Block

表示基本块。

字段：

* `id: str`
* `op_ids: list[str]`
* `preds: list[str]`
* `succs: list[str]`
* `attrs: dict`

### 4.2.4 FunctionIR

字段：

* `id: str`
* `name: str`
* `arg_values: list[str]`
* `return_values: list[str]`
* `values: dict[str, Value]`
* `ops: dict[str, Op]`
* `blocks: dict[str, Block]`
* `entry_block: str`
* `attrs: dict`

### 4.2.5 ModuleIR

字段：

* `functions: dict[str, FunctionIR]`
* `global_values: dict[str, Value]`
* `attrs: dict`

---

# 5. Frontend 实现要求

## 5.1 输入

输入为单个 Python 源文件或一个函数对象。

## 5.2 实现策略

使用 Python `ast` 做 frontend。

要求：

1. 建立 AST visitor / transformer。
2. 为每个表达式与语句分配稳定 ID。
3. 显式构建 CFG。
4. 在 CFG 基础上生成 MiniIR。
5. 对循环和条件分支生成 block。
6. 在 merge 点生成 `phi` 或等价机制。

## 5.3 支持的 AST 节点

必须支持：

* `Module`
* `FunctionDef`
* `arguments`
* `Assign`
* `AugAssign`
* `Expr`
* `Return`
* `If`
* `While`
* `For`
* `Call`
* `Name`
* `Constant`
* `Attribute`
* `Subscript`
* `List`
* `Dict`
* `BinOp`
* `UnaryOp`
* `Compare`

MVP 阶段，遇到不支持节点必须报错并给出 source span。

## 5.4 CFG 构建

要求：

* 每个基本块唯一编号。
* 记录前驱 / 后继。
* 对 `if/else`、`while`、`for` 正确建边。
* 对 `return` 建立函数退出。
* 可选：生成 dominator tree（MVP 可后置）。

## 5.5 SSA-ish 变量处理

初版不需要完整 SSA，但必须做到：

* 每次赋值生成新的静态 value id。
* 使用点指向最近可达定义。
* 在控制流 merge 点用 `phi` 或 `merge_value` 解决多定义。

建议内部实现：

* 每个变量名维护当前 version。
* CFG 合流点时合并 version。

---

# 6. Interpreter / Executor 实现要求

## 6.1 总目标

实现一个 **MiniIR 解释器**，而不是依赖 Python 原始源码执行。

原因：

* 解释器可精确控制事件粒度。
* 可稳定跨 Python 版本。
* 可确保每个 op 都输出统一格式事件。

## 6.2 运行时对象

### 6.2.1 RuntimeValue

表示静态 value 的运行时实例。

字段：

* `rid: str`
* `static_value_id: str`
* `py_obj_id: int | None`
* `created_by_event: str`
* `last_writer_event: str | None`
* `frame_id: str`
* `version: int`
* `metadata: dict`

### 6.2.2 RuntimeEvent

表示一次 op 执行实例。

字段：

* `event_id: str`
* `static_op_id: str`
* `frame_id: str`
* `timestamp: int`
* `input_rids: list[str]`
* `output_rids: list[str]`
* `control_context: tuple[str, ...]`
* `attrs: dict`

### 6.2.3 RuntimeFrame

表示调用帧。

字段：

* `frame_id: str`
* `function_id: str`
* `parent_frame_id: str | None`
* `callsite_event_id: str | None`
* `locals: dict[str, str]`
* `attrs: dict`

## 6.3 Shadow Store

必须实现 Shadow Store，用于记录容器、对象字段、索引项的版本化写入。

字段建议：

* `object_versions: dict[int, list[str]]`
* `field_writers: dict[tuple[int, str], str]`
* `item_writers: dict[tuple[int, object], str]`
* `container_membership: dict[int, set[str]]`

说明：

* 如果某 list 中加入一个新元素，必须记录该元素的 runtime rid 与该 list 的关系。
* 如果某对象字段被更新，必须记录字段级别 writer。

## 6.4 解释器执行协议

每个 op 执行时，必须遵循：

1. 解析输入 runtime values。
2. 执行语义。
3. 创建输出 runtime values。
4. 记录 RuntimeEvent。
5. 更新 Shadow Store。
6. 推进 block / PC / frame。

## 6.5 控制上下文

必须维护 control context，例如：

* 当前位于哪个 if 分支
* 当前循环迭代编号
* 当前递归深度

建议将其编码为字符串元组，例如：

* `("if:block_3:true", "while:block_7:iter=2")`

---

# 7. 静态依赖分析实现要求

## 7.1 输出目标

静态分析必须至少导出以下关系：

* def-use edges
* use-def edges
* block control edges
* call graph edges
* variable version chain

## 7.2 Def-use 图

对每个静态 value：

* 记录 `def_op`
* 记录全部 `use_ops`

对每个静态 op：

* 记录 `inputs`
* 记录 `outputs`

## 7.3 控制依赖

至少记录：

* `branch op -> true successor block`
* `branch op -> false successor block`
* `loop header -> body`
* `loop backedge`

MVP 不要求完整 post-dominator 分析，但推荐预留接口。

## 7.4 别名与容器访问（近似即可）

静态阶段只做弱近似：

* 标记哪些 value 是容器
* 标记哪些 op 可能改变容器内容
* 标记哪些 op 是对象/字段访问

不要在静态阶段试图精确恢复高层结构。

---

# 8. 动态依赖分析实现要求

## 8.1 输出目标

动态分析必须至少恢复：

* event 输入输出关系
* event 时间顺序
* runtime value 创建链
* 容器成员变化
* frame 嵌套
* 控制上下文实例化

## 8.2 动态依赖种类

必须构造以下动态边：

### 8.2.1 数据边

* `runtime value -> runtime event`
* `runtime event -> runtime value`

### 8.2.2 时间边

* `event_t -> event_t+1`
* 可选：同 frame 内局部时间边

### 8.2.3 控制边

* `branch instance -> event under branch`
* `loop iteration header -> body event`

### 8.2.4 容器/字段边

* `container -> member rid`
* `field slot -> rid`
* `writer event -> slot`

### 8.2.5 调用边

* `call event -> callee frame`
* `callee return -> caller continuation`

## 8.3 动态实例归一

为了发现“状态族”，必须实现 runtime schema fingerprint。

建议最小实现：

* Python type name
* dataclass / dict keys / object fields
* 数值字段 shape
* 容器大小

输出为 hashable tuple，用于聚类 runtime values。

---

# 9. FactGraph 设计

## 9.1 总要求

FactGraph 是全系统核心数据结构。
它统一承载静态与动态事实，但不带高层结构解释。

## 9.2 数据结构

建议类：

```python
class FactGraph:
    static_functions: dict[str, FunctionIR]
    static_ops: dict[str, Op]
    static_values: dict[str, Value]

    runtime_events: dict[str, RuntimeEvent]
    runtime_values: dict[str, RuntimeValue]
    runtime_frames: dict[str, RuntimeFrame]

    static_edges: dict[str, set[tuple[str, str]]]
    dynamic_edges: dict[str, set[tuple[str, str]]]
    alignment_edges: dict[str, set[tuple[str, str]]]

    metadata: dict
```

## 9.3 静态边类型

至少包括：

* `def_use`
* `use_def`
* `cfg`
* `call_static`

## 9.4 动态边类型

至少包括：

* `event_input`
* `event_output`
* `temporal`
* `control_dynamic`
* `call_dynamic`
* `frame_nesting`
* `container_membership`
* `field_slot`

## 9.5 对齐边类型

至少包括：

* `instantiates_op: runtime_event -> static_op`
* `instantiates_value: runtime_value -> static_value`
* `runtime_in_frame: runtime_value -> runtime_frame`

---

# 10. RewriteRegion / Projection 支持层设计

## 10.1 原则

RewriteRegion 是**第一等的重写对象**；Projection 是**可选解释层**。

更具体地说：

* 如果用户或上层分析已经知道“要切掉哪段计算、保留哪些边界、在哪里接回”，系统应当可以**不依赖 projection**，直接在 RewriteRegion 上完成 override / grafting。
* 如果上层希望给 donor skeleton 一个更抽象的匹配视角，那么可以在 RewriteRegion 或 FactGraph 之上附加 Projection。
* Projection 不是程序本体，也不是底层真类型；它只是一种附加解释。

## 10.2 RewriteRegion 数据模型

```python
@dataclass
class RewriteRegion:
    region_id: str
    op_ids: list[str]
    block_ids: list[str]
    entry_values: list[str]
    exit_values: list[str]
    read_set: list[str]
    write_set: list[str]
    state_carriers: list[str]
    schedule_anchors: dict
    allows_new_state: bool
    attrs: dict
    provenance: dict
```

字段说明：

* `op_ids`：区域内部被替换 / 保留 / 观察的静态 op 集合
* `block_ids`：区域覆盖的基本块集合
* `entry_values`：区域外定义、区域内消费的输入值
* `exit_values`：区域内定义、区域外消费的输出值
* `read_set` / `write_set`：区域涉及的对象字段、容器成员、索引槽位等读写集合
* `state_carriers`：被上层视为“状态承载者”的 value / runtime schema / object family 标识
* `schedule_anchors`：可插入 donor skeleton 的时机锚点，例如循环头、select 后、expand 后、score 前
* `allows_new_state`：是否允许 graft 时引入额外状态槽位

## 10.3 BoundaryContract 数据模型

```python
@dataclass
class BoundaryContract:
    contract_id: str
    region_id: str
    input_ports: list[str]
    output_ports: list[str]
    readable_slots: list[str]
    writable_slots: list[str]
    new_state_policy: dict
    reconnect_points: dict
    invariants: dict
    evidence: dict
```

字段说明：

* `input_ports` / `output_ports`：grafting 时 donor skeleton 的显式输入输出端口
* `readable_slots` / `writable_slots`：donor 可访问 / 可修改的槽位
* `new_state_policy`：是否允许新增 summary state、message slot、scratch state 等
* `reconnect_points`：donor 输出重新接回 host 的位置
* `invariants`：必须保持的约束，例如输出仍需为标量、仍需可比较、仍需在有限迭代内结束

## 10.4 Projection 数据模型（可选）

```python
@dataclass
class Projection:
    proj_id: str
    region_id: str
    family: str
    node_set: list[str]
    edge_set: list[tuple[str, str]]
    evidence: dict
    interface: dict
    score: float
```

字段说明：

* `region_id`：Projection 依附的 RewriteRegion；如果没有指定 region，也可以指向整个 FactGraph 的某一子图
* `family`：结构视角家族名，例如 `branching`, `scheduling`, `local_interaction`, `refinement`
* `interface`：对 donor skeleton 仍然有帮助的补充性视角描述，而不是唯一的 grafting 契约

## 10.5 MVP 阶段必须实现的能力

### A. RewriteRegion 定义

至少支持以下几种方式：

* 用户显式给出 `op_ids`
* 通过 `source_span` 选择区域
* 从一个或多个 `exit_values` 做 backward slice
* 从一个或多个 `state_carriers` 做 forward / mixed slice

### B. BoundaryContract 推断

系统必须能从静态 IR、trace 与 ShadowStore 中推断：

* 区域外部输入端口
* 区域外部可见输出端口
* 区域内部对对象字段 / 容器成员的实际读写
* donor 可插入的调度锚点
* graft 后需要保持的最小不变量

### C. 可选的 Region Annotation / Projection

如果需要附加解释层，MVP 可先支持两类最小 annotation：

* `scheduling / candidate-pool`
* `local_interaction / local_update`

但这两类 annotation 不是 region rewrite 的前提条件。

## 10.6 Region 提取与 annotation 流程

推荐流程：

1. 由用户或上层系统选定一个 RewriteRegion。
2. 检查该区域是否形成可闭合的输入 / 输出边界。
3. 推断 BoundaryContract。
4. 如有需要，再为该区域附加 Projection。
5. 将 `RewriteRegion + BoundaryContract + optional Projection` 交给 grafting engine。

---

# 11. Skeleton Grafting 设计

## 11.1 原则

Skeleton 不匹配“算法名”，优先匹配 **BoundaryContract**；Projection 仅作为可选辅助信息。

也就是说：

* 主 grafting 路径是：`RewriteRegion -> BoundaryContract -> donor skeleton`
* 可选辅助路径是：`RewriteRegion -> Projection -> donor hint`

## 11.2 Skeleton 与 OverridePlan 数据模型

```python
@dataclass
class Skeleton:
    skel_id: str
    name: str
    required_contract: dict
    transform_rules: list[dict]
    lowering_template: dict
    optional_projection_hints: dict


@dataclass
class OverridePlan:
    plan_id: str
    target_region_id: str
    removed_op_ids: list[str]
    preserved_bindings: dict
    new_state_defs: list[dict]
    schedule_insertions: list[dict]
    reconnect_map: dict
    projection_id: str | None
```

说明：

* `required_contract`：skeleton 运行所需的最低边界契约
* `lowering_template`：把 donor skeleton 降到无结构 MiniIR 的模板或规则
* `removed_op_ids`：host 中被剥离的旧 op
* `preserved_bindings`：override 后仍保留的原有绑定
* `new_state_defs`：新增状态槽位、容器、对象字段等
* `schedule_insertions`：新逻辑插入到哪些调度锚点
* `reconnect_map`：新的 donor 输出接回哪些旧出口

## 11.3 MVP skeleton

MVP 只实现一个非常简单的 skeleton：

### local summary update skeleton

要求 contract 至少满足：

* 存在可识别的 `state_carriers`
* 存在至少一个可重连的输出端口
* 存在循环或重复调度锚点
* 允许新增局部 summary state

行为：

* 在某组状态上增加一个 summary state
* 在循环中对其进行局部更新
* 将 summary 输出接回原本的 scoring / selection 路径

注意：

* 不要在 MVP 阶段实现真正 BP；只实现 BP-like / summary-like donor 的最简版本。
* 重点验证 region override 闭环是否成立，而不是先追求复杂 donor。

## 11.4 graft 输出

Grafting 产物不能只是源代码。必须输出：

* 新 FunctionIR
* 新 / 旧 RewriteRegion 映射
* 可选的新 / 旧 Projection 映射
* 新代码
* tracing schema

---

# 12. IR Regenerator 设计

## 12.1 目标

将 graft / override 后结构重新生成统一 IR，并可导出 Python 源码。

## 12.2 要求

* 保持 op/value/block ID 可追踪
* 对新增操作给出稳定命名
* 保留 source mapping / provenance 信息

## 12.3 输出对象

```python
@dataclass
class AlgorithmArtifact:
    ir: FunctionIR
    source_code: str
    rewritten_regions: list[RewriteRegion]
    projections: list[Projection]
    provenance: dict
```

`provenance` 必须能回答：

* 哪些 op 是原算法已有
* 哪些 op 是 graft 新增
* 哪个 RewriteRegion 被重写
* 使用了哪个 BoundaryContract
* 哪个 projection 参与了 graft（如果有）
* 哪个 skeleton 被应用

---

# 13. 模块划分

建议项目目录：

```text
project_root/
  frontend/
    ast_parser.py
    cfg_builder.py
    ir_builder.py
  ir/
    model.py
    printer.py
    validator.py
  runtime/
    interpreter.py
    shadow_store.py
    tracer.py
    frames.py
  factgraph/
    model.py
    builder.py
    aligner.py
  region/
    selector.py
    slicer.py
    contract.py
  analysis/
    static_analysis.py
    dynamic_analysis.py
    fingerprints.py
  projection/
    base.py
    scheduling.py
    local_interaction.py
    scorer.py
  grafting/
    skeletons.py
    matcher.py
    rewriter.py
  regeneration/
    codegen.py
    artifact.py
  tests/
    examples/
    unit/
    integration/
```

---

# 14. 核心接口规范

## 14.1 Frontend

```python
def compile_function_to_ir(fn) -> FunctionIR:
    ...
```

## 14.2 Executor

```python
def execute_ir(func_ir: FunctionIR, args: list[object]) -> tuple[object, list[RuntimeEvent], dict[str, RuntimeValue]]:
    ...
```

## 14.3 FactGraph Builder

```python
def build_factgraph(func_ir: FunctionIR, runtime_trace: list[RuntimeEvent], runtime_values: dict[str, RuntimeValue]) -> FactGraph:
    ...
```

## 14.4 RewriteRegion / BoundaryContract

```python
def define_rewrite_region(
    func_ir: FunctionIR,
    *,
    op_ids: list[str] | None = None,
    source_span: tuple[int, int, int, int] | None = None,
    exit_values: list[str] | None = None,
    state_carriers: list[str] | None = None,
) -> RewriteRegion:
    ...
```

```python
def infer_boundary_contract(
    func_ir: FunctionIR,
    region: RewriteRegion,
    runtime_trace: list[RuntimeEvent] | None = None,
    runtime_values: dict[str, RuntimeValue] | None = None,
) -> BoundaryContract:
    ...
```

## 14.5 Optional Projection Annotation

```python
def annotate_region(
    region: RewriteRegion,
    fg: FactGraph | None = None,
) -> list[Projection]:
    ...
```

## 14.6 Grafting

```python
def graft_skeleton(
    func_ir: FunctionIR,
    region: RewriteRegion,
    contract: BoundaryContract,
    skeleton: Skeleton,
    projection: Projection | None = None,
) -> AlgorithmArtifact:
    ...
```

---

# 15. 验证与测试计划

## 15.1 单元测试

必须覆盖：

* AST -> IR 编译
* CFG 构建
* def-use 链
* interpreter 执行 correctness
* runtime event 生成
* shadow store 容器追踪
* factgraph 对齐
* RewriteRegion 选择 / 切片
* BoundaryContract 推断

## 15.2 集成测试

最小集成测试算法：

1. 简单优先队列循环
2. 简单递归/迭代状态扩展算法
3. 简单局部迭代更新算法
4. 指定局部区域后进行 override 的最小示例

## 15.3 MVP 成功标准

以下条件全部满足时，判定 MVP 成功：

1. 系统能将受限 Python 函数编译成 MiniIR。
2. 系统能执行 MiniIR 并生成 runtime trace。
3. 系统能构造统一 FactGraph。
4. 系统能在同一算法上定义至少一个 RewriteRegion，并推断其 BoundaryContract。
5. 系统能在一个 RewriteRegion 上 graft 一个 minimal local-summary-update skeleton。
6. graft 后算法能重新回到同一 IR，并再次被分析。
7. 可选增强：系统能为同一 RewriteRegion 附加至少一种 Projection annotation。

---

# 16. 非目标（MVP 阶段不要做）

以下内容暂时不要实现：

* 普通 Python 全语言支持
* LLVM/MLIR 后端
* 深度学习 projection classifier / region ranker
* 完整 BP / EP / search-tree grafting
* 自动性能优化
* 跨语言前端
* 并发 / 分布式执行

---

# 17. 推荐实现顺序

## 第 1 周

* 实现 `ir/model.py`
* 实现 AST frontend
* 实现最基础 CFG
* 将简单函数编译为 MiniIR

## 第 2 周

* 实现 MiniIR 解释器
* 实现 RuntimeEvent / RuntimeValue / RuntimeFrame
* 跑通简单算法 trace

## 第 3 周

* 实现 ShadowStore
* 实现 FactGraph Builder
* 实现静态 / 动态对齐

## 第 4 周

* 实现 RewriteRegion 选择 / 切片
* 实现 BoundaryContract 推断
* 在简单算法上测试区域闭合性

## 第 5 周

* 实现一个 minimal skeleton
* 做一次 region override / grafting demo
* 输出 AlgorithmArtifact

## 第 6 周（可选）

* 实现最小 Projection annotation
* 在同一 RewriteRegion 上测试多视角解释

---

# 18. 重要设计警告

1. **绝对不要**在 IR opcode 或 data model 中偷偷塞入高层结构语义。
2. **绝对不要**把 projection 当成唯一的重写对象。真正被切除 / 覆写的是 RewriteRegion。
3. **绝对不要**把 projection 结果回写成底层“真类型”。projection 只是视角。
4. **绝对不要**让 grafting 直接对源代码做字符串拼接。必须通过 IR rewrite 完成。
5. **绝对不要**省略 provenance。否则下一轮演化无法追踪来源。
6. **绝对不要**把动态分析退化成 line trace。必须是 op-level 事件。

---

# 19. 对实现 AI 的最终要求

实现 AI 必须：

* 严格按本文档的无结构原则实现系统。
* 在每一阶段交付可运行代码，而不是只写类定义。
* 每实现一个模块，补充对应单元测试。
* 遇到不支持的 Python 语法时明确报错，而不是 silently ignore。
* 优先保证 IR、tracing、region slicing 与 boundary inference 的稳定性，而不是追求复杂功能。

如果实现过程中遇到“是否应该把某种结构显式编码进 IR”的选择，一律选择：

**不编码结构，只编码事实。结构放在 projection 层。**

如果实现过程中遇到“是先做 projection discovery，还是先做可稳定的 RewriteRegion / BoundaryContract”这一选择，一律优先：

**先保证 region rewrite 闭环，再把 projection 作为可选增强。**

---

# 20. 一句话总结

本项目要实现的不是一个传统 typed DSL，而是一套：

**在无结构、可执行、可追踪的事实型 IR 上，支持局部计算区域的剥离、覆写与 donor skeleton grafting，并可选地为这些区域附加多种结构视角的自举式算法演化框架。**

这句话必须作为整个实现的最高约束。
