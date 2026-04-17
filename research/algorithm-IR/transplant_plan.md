# Algorithm-IR 迁移计划

本文档给出 `research/algorithm-IR/algorithm_ir/` 的详细迁移方案。  
迁移目标不是“推翻现有实现”，而是把当前的手写 IR/CFG/SSA/rewriter 基础设施迁移到一个更成熟的现有 IR 框架上，同时：

- 保留现有全部功能
- 保留现有全部对外 API
- 保证现有全部测试继续通过
- 在此基础上支持更丰富的数据类型、操作和语法

---

## 1. 迁移目标

### 1.1 必须满足的硬约束

迁移完成后，以下能力必须全部保留：

1. 仍然可以从受限 Python 函数构建结构中立 IR
2. 仍然可以打印和验证 IR
3. 仍然可以执行 IR，并得到运行时 trace / runtime values
4. 仍然可以构建 FactGraph
5. 仍然可以定义 `RewriteRegion`
6. 仍然可以推断 `BoundaryContract`
7. 仍然可以做可选 `Projection`
8. 仍然可以 graft donor skeleton，生成新的 IR
9. 当前所有测试必须继续通过
10. 当前对外函数签名尽量不变，至少外部调用方式不应破坏

### 1.2 本次迁移额外要达成的增强目标

1. 类型系统从当前的字符串 `type_hint` 升级为真正的 IR-level 类型
2. 支持更多基础数值类型
3. 支持 complex / vector / tensor / matrix 一类更接近通信算法的类型
4. 支持更稳健的 CFG / SSA / region / rewrite 基础设施
5. 为未来对接 MLIR / LLVM / CIRCT 留出下行路径

---

## 2. 选型结论

### 2.1 选型结果

本计划选择：

**`xDSL first, MLIR-compatible`**

即：

- 直接迁移到 `xDSL`
- 设计成与 MLIR 语义兼容
- 在需要时再下行到官方 MLIR / LLVM / CIRCT

### 2.2 为什么不直接选官方 MLIR Python bindings

官方 MLIR Python bindings 的确是终极底座，但它当前官方文档明确写着：

- `Current status: Under development and not enabled by default`

见官方文档：
- <https://mlir.llvm.org/docs/Bindings/Python/>

这意味着如果现在直接把整个项目压到官方 MLIR Python bindings 上，会有几个现实问题：

- Python 端开发摩擦更大
- 环境部署更重
- 自定义 dialect / 自定义工具链的开发成本更高
- 对一个仍在快速试验中的研究型代码库来说，迁移风险偏大

### 2.3 为什么选择 xDSL

`xDSL` 官方说明非常符合当前项目需求：

- 它是 `Python-native`
- 它是 `SSA-based`
- 它有 `regions` 和 `basic blocks`
- 它支持自定义 IR / dialect
- 它与 `MLIR` 兼容
- 它有 `pattern rewriter`

官方资料：
- <https://github.com/xdslproject/xdsl>
- <https://docs.xdsl.dev/reference/pattern_rewriter/>

这和当前 `algorithm_ir` 的约束几乎完全一致：

- 项目是 Python 实现
- 我们需要 SSA / Block / Region
- 我们需要自定义中立 dialect
- 我们需要 rewrite / replace / erase / insert
- 我们未来还想下接更强的 MLIR 生态

所以，从工程风险与研究灵活性来看，`xDSL` 是当前最合适的落地点。

---

## 3. 总体迁移原则

### 3.1 不做 big-bang 重写

不允许“一次性推倒重写”。  
迁移必须采用 **分层替换、双轨兼容、逐步切换** 的方法。

原因很简单：

- 当前系统已经有可执行闭环
- 已经有 7 个测试覆盖主路径
- 已经有可工作的 stack/BP graft demo

如果全量替换，一旦某一层出错，很难定位到底是前端、IR、执行器、region 还是 grafting 出问题。

### 3.2 先替换基础设施，再替换研究逻辑

本计划的核心思想是：

- 先把“手写编译器底座”迁到 xDSL
- 后保留“研究语义层”不变

因此：

- `IR / CFG / SSA / printer / validator / rewriter infrastructure` 是优先迁移对象
- `RewriteRegion / BoundaryContract / FactGraph / Projection / ShadowStore` 先保留语义和接口，再逐步改底层实现

### 3.3 对外 API 保持稳定

迁移期间，这些函数的对外调用方式必须保持不变：

- `compile_function_to_ir`
- `render_function_ir`
- `validate_function_ir`
- `execute_ir`
- `build_factgraph`
- `define_rewrite_region`
- `infer_boundary_contract`
- `annotate_region`
- `make_bp_summary_skeleton`
- `graft_skeleton`

内部实现可以逐渐切换到 xDSL，但外部测试和调用方不应感知破坏性变化。

---

## 4. 当前实现中要迁移的部分

### 4.1 需要被 xDSL 取代的部分

这些部分最适合迁移到 xDSL：

1. `algorithm_ir/ir/model.py`
2. `algorithm_ir/frontend/cfg_builder.py`
3. `algorithm_ir/frontend/ir_builder.py`
4. `algorithm_ir/ir/printer.py`
5. `algorithm_ir/ir/validator.py`
6. `algorithm_ir/grafting/rewriter.py` 中的通用 IR surgery 基础能力

原因：

- 当前这些层在手工维护 SSA value / op / block / use-def / CFG
- 这些正是标准 IR 框架最擅长的部分
- 继续手写会导致类型能力、鲁棒性、rewrite 安全性都受限

### 4.2 暂时不应被直接取代的部分

这些部分应先保留：

1. `runtime/interpreter.py`
2. `runtime/shadow_store.py`
3. `factgraph/*`
4. `region/*`
5. `projection/*`
6. donor skeleton 定义与匹配逻辑

原因：

- 这些层承载的是研究语义，而不是纯编译器基础设施
- xDSL/MLIR 不会自动提供你的 `RewriteRegion`、`BoundaryContract`、`FactGraph`
- 这些层只需要逐步改为“消费 xDSL IR”，不需要立刻重写成 xDSL 内部机制

---

## 5. 目标架构

迁移完成后的推荐架构如下：

```text
Python source
  -> frontend parser/lowering
  -> xDSL-based neutral dialect (alg dialect)
  -> xDSL standard dialects (builtin / func / cf / scf / arith / math / complex / tensor / memref ...)
  -> compatibility adapter
  -> runtime / factgraph / region / projection / grafting
```

### 5.1 两层 IR 设计

建议迁移后使用两层 IR：

#### 层 1：`alg` 中立自定义 dialect

这是你自己的 dialect，用来表达当前项目特有、但又不想提前硬编码成“树/BP”的东西。

推荐保留/新增的 op 家族：

- `alg.const`
- `alg.assign`
- `alg.get_attr`
- `alg.set_attr`
- `alg.get_item`
- `alg.set_item`
- `alg.build_list`
- `alg.build_dict`
- `alg.append`
- `alg.pop`
- `alg.call_py`
- `alg.iter_init`
- `alg.iter_next`
- `alg.object_ref`
- `alg.dict_ref`
- `alg.list_ref`

注意：

- 这里的 “alg” 不是高层算法语义
- 它只是“结构中立的动态对象 / 容器 / Python-like 引用层”

#### 层 2：标准 dialect

当某一部分已经可以 lower 到更标准、静态、数值化的层时，再下放到：

- `builtin`
- `func`
- `cf`
- `scf`
- `arith`
- `math`
- `complex`
- `tensor`
- `memref`
- `vector`

这样做的好处是：

- host/donor 的上层表示仍保持中立
- 但纯数值 kernel 已经能享受成熟 IR 生态

---

## 6. 类型系统升级方案

这是本次迁移最重要的增益之一。

### 6.1 当前类型问题

当前系统的 `type_hint` 还是这种风格：

- `int`
- `float`
- `bool`
- `list`
- `dict`
- `object`

这对于 demo 足够，但对于真实通信算法远远不够。

例如，未来你很可能需要表达：

- real scalar
- complex scalar
- integer index
- fixed-width integer
- vector
- matrix
- tensor
- probability table
- candidate record
- runtime object reference

### 6.2 迁移后的类型层次

建议采用三层类型设计：

#### A. 纯静态数值类型

- `i1`
- `i8`
- `i16`
- `i32`
- `i64`
- `index`
- `f16`
- `f32`
- `f64`
- `complex<f32>`
- `complex<f64>`

#### B. 形状化数据类型

- `vector<N x T>`
- `tensor<shape x T>`
- `memref<shape x T>`

用于未来表达：

- 符号向量
- 信道矩阵
- 软信息张量
- 消息表

#### C. 中立运行时引用类型

这些类型仍需要自定义，因为它们不完全是静态数值对象：

- `!alg.pyobj`
- `!alg.listref<T?>`
- `!alg.dictref<K,V>`
- `!alg.record<...>`
- `!alg.candidate`
- `!alg.frontier`

其中：

- `!alg.pyobj` 保底承接动态 Python 语义
- `!alg.candidate` 等只是中立记录类型，不意味着“预设搜索树”

### 6.3 通信算法优先支持的 richer types

迁移完成后，第一优先级应支持：

1. `complex<f32>` / `complex<f64>`
2. `tensor<?xcomplex<f64>>`
3. `tensor<?x?xcomplex<f64>>`
4. `vector<?xf64>`
5. `vector<?xindex>`
6. `!alg.dictref`
7. `!alg.listref`
8. `!alg.record`

因为这正对应：

- 信道
- 接收信号
- 候选状态
- 指标表
- 消息数组

---

## 7. 公共 API 兼容计划

迁移过程中，不能让测试全部改写后才能运行。  
因此必须引入 **compatibility layer**。

### 7.1 兼容策略

新增一个适配层，例如：

```text
algorithm_ir/compat/
  adapters.py
  legacy_ir_view.py
  legacy_runtime_bridge.py
```

### 7.2 兼容原则

`compile_function_to_ir(fn)` 在迁移后的行为应变为：

1. Python frontend 生成 xDSL module
2. 通过 adapter 生成“legacy-compatible view”
3. 对外仍返回一个满足当前测试期望的对象

换句话说：

- 内核已经切到 xDSL
- 外部 API 先看起来像旧版

### 7.3 最小兼容对象

建议保留一个兼容包装对象：

```python
class LegacyFunctionIRView:
    xdsl_module: ModuleOp
    xdsl_func: FuncOp
    ...
```

它暴露旧接口需要的字段：

- `name`
- `entry_block`
- `blocks`
- `ops`
- `values`
- `arg_values`
- `return_values`

初期它可以是只读视图。  
等所有上层逻辑逐步改为直接消费 xDSL 后，再考虑废弃。

---

## 8. 前端迁移计划

### 8.1 保留 `ast_parser.py`

这层先不换。

原因：

- 你当前的输入就是 Python 函数
- MLIR/xDSL 不负责 Python 解析

所以：

- `parse_function`
- `source_span`

这些接口先保留。

### 8.2 重写 `ir_builder.py`，但保留外部入口

当前 [frontend/ir_builder.py](./algorithm_ir/frontend/ir_builder.py) 里：

- 手工生成 values
- 手工生成 ops
- 手工生成 blocks
- 手工补 phi

迁移后，这层要改成：

- 把 AST lowering 到 xDSL `ModuleOp`
- 使用 xDSL 的 block / region / operation / value 体系
- block 参数替代大部分手工 phi

### 8.3 前端分阶段扩语法

在保证旧测试全绿的前提下，语法支持分三步增加：

#### 阶段 A：等价迁移

保持当前支持集不变：

- Assign
- AugAssign
- Return
- If
- While
- For
- Call
- Name
- Constant
- Attribute
- Subscript
- List
- Dict
- BinOp
- UnaryOp
- Compare

#### 阶段 B：低风险扩展

增加：

- Tuple
- BoolOp (`and/or`)
- IfExp
- Slice
- `break`
- `continue`
- `pass`

#### 阶段 C：中风险扩展

增加：

- 多返回值
- destructuring assignment
- comprehension 的受限形式
- 更复杂的 builtin 调用

所有新语法必须通过 feature flag 打开，不能直接影响旧测试路径。

---

## 9. 运行时与解释器迁移计划

### 9.1 不建议第一阶段完全替换 `execute_ir`

当前 `execute_ir` 除了解释执行，还承担：

- `RuntimeEvent`
- `RuntimeValue`
- ShadowStore 写入
- frame locals
- control context

这不是简单的“执行 MLIR/xDSL IR”问题。

所以第一阶段建议：

- 保留当前 `execute_ir` 对外接口
- 在内部改为解释 xDSL-based neutral dialect

### 9.2 两种实现路径

#### 路线 1：在现有解释器中适配 xDSL op

优点：

- 改动更可控
- 更容易保留现有 trace 结构

做法：

- 用 `Operation.name` / dialect class 替换当前 `opcode` 分派
- `RuntimeEvent.static_op_id` 改为引用 xDSL operation handle 或稳定 ID

#### 路线 2：使用 xDSL interpreter 机制做底层执行，再包 runtime trace

优点：

- 更贴近框架生态
- 长期可维护性更好

缺点：

- 迁移初期工作量更大

建议：

**先走路线 1，等稳定后再逐步吸收 xDSL interpreter 机制。**

---

## 10. RewriteRegion / BoundaryContract / Projection 迁移计划

这部分概念不变，底层证据来源变。

### 10.1 RewriteRegion 不变

`RewriteRegion` 仍然是第一等对象。

但它内部不再依赖手写 `FunctionIR`，而是依赖：

- xDSL operation id
- xDSL block id
- xDSL SSA value

### 10.2 BoundaryContract 不变

不改变其语义：

- input_ports
- output_ports
- readable_slots
- writable_slots
- reconnect_points
- invariants

只把底层分析换成：

- xDSL def-use
- block structure
- memory effect / side effect information

### 10.3 Projection 仍然可选

Projection 仍不是 rewrite 主体。

它继续作为：

- 可选解释层
- donor 匹配 hint
- NN 输入对象

这条设计原则不能在迁移中丢掉。

---

## 11. grafting 迁移计划

这是迁移中的重点难点之一。

### 11.1 当前问题

现在的 [rewriter.py](./algorithm_ir/grafting/rewriter.py) 里做了大量手工 surgery：

- 手工选 target output
- 手工删除 op
- 手工插入 op
- 手工维护 block.op_ids
- 手工重算 def-use

这正是现成 IR 框架最值得接管的地方。

### 11.2 迁移目标

迁移后：

- graft 语义仍由你的代码控制
- 但底层操作改由 xDSL pattern rewrite / IR mutation API 执行

### 11.3 donor graft 的实现分层

建议拆成两层：

#### A. 研究层

- 选择 target region
- 验证 contract
- 确定 donor skeleton
- 生成 override plan

#### B. IR surgery 层

改由 xDSL 完成：

- 插入 op
- 删除 op
- replace uses
- inline / move block
- type conversion

### 11.4 BP graft 的迁移目标

当前 `bp_summary_update` graft 是最小闭环。  
迁移后必须保证：

- 原有 graft demo 继续通过
- `OverridePlan` 仍然可用
- graft 前后 IR 仍然可打印和执行

---

## 12. 测试保真计划

这是整个迁移计划里最不能妥协的部分。

### 12.1 测试目标

现有这 7 个测试必须持续工作：

- `test_compile_simple_branch_loop`
- `test_compile_stack_decoder_host`
- `test_execute_simple_branch_loop`
- `test_execute_stack_decoder_and_build_factgraph`
- `test_define_region_and_infer_contract`
- `test_optional_projection_annotation`
- `test_stack_decoder_bp_grafting_demo`

### 12.2 测试策略

采用“三层测试护栏”：

#### 第一层：现有测试一个不删

现有测试文件全部保留，并在每个迁移阶段运行。

#### 第二层：新增等价性测试

每迁移一层，都加“旧 IR vs 新 IR”的等价性测试：

- CFG 结构等价
- 执行结果等价
- region 输出端口等价
- contract 等价
- graft 结果等价

#### 第三层：新增 richer type 测试

为新类型单独加测试：

- `complex<f64>` 运算
- vector 加法 / 比较
- tensor 索引
- dict/list/object 与 shaped type 混合存在

### 12.3 测试迁移顺序

建议每一阶段只允许新增测试，不允许删旧测试。

只有当新旧双轨完全等价后，才允许逐步废弃旧内部实现。

---

## 13. 分阶段实施计划

下面是推荐的实施顺序。

---

## 阶段 0：准备阶段

### 目标

- 引入 `xDSL` 依赖
- 不改变现有行为
- 建立双轨实验分支

### 任务

1. 在环境中增加 `xdsl`
2. 新建迁移目录：

```text
algorithm_ir_xdsl/
  dialects/
  lowering/
  compat/
  runtime/
  rewrite/
```

3. 新增 smoke test：
   - 能创建最小 xDSL module
   - 能打印 module

### 验收标准

- 原有 7 个测试仍然全绿
- 新增 xDSL smoke test 通过

---

## 阶段 1：引入 xDSL IR 核心，不接业务

### 目标

- 先把 IR 数据模型迁到 xDSL
- 不动前端、不动运行时

### 任务

1. 定义 `alg` dialect 草案
2. 定义最小类型：
   - `!alg.pyobj`
   - `!alg.listref`
   - `!alg.dictref`
3. 定义最小 op：
   - `alg.const`
   - `alg.assign`
   - `alg.get_item`
   - `alg.set_item`
   - `alg.get_attr`
   - `alg.set_attr`
   - `alg.build_list`
   - `alg.build_dict`
   - `alg.append`
   - `alg.pop`
   - `alg.call_py`
4. 实现 `LegacyFunctionIRView`
5. 让 `render_function_ir` 和 `validate_function_ir` 可以消费 compat view

### 验收标准

- 旧测试仍全绿
- 新增“从手写 IR 构造等价 xDSL module”的测试通过

---

## 阶段 2：前端 lowering 切到 xDSL

### 目标

- `compile_function_to_ir` 内部改为输出 xDSL
- 外部仍返回 legacy-compatible view

### 任务

1. 重写 `ir_builder.py`：
   - AST -> xDSL `ModuleOp`
   - `if` -> `cf.cond_br` 或 `scf.if`
   - `while` -> `scf.while` 或 `cf` 形式
   - block args 替换手工 `phi`
2. 保留 `source_span`
3. 保留变量版本追踪元信息
4. 编译结果通过 compat adapter 暴露旧视图

### 验收标准

- 前端两个测试仍全绿
- `demo_outputs.py` 中的 IR 打印仍然可读
- `simple_branch_loop` 与 `stack_decoder_host` 的执行结果不变

---

## 阶段 3：运行时桥接到 xDSL

### 目标

- `execute_ir` 消费 xDSL IR
- 保留原有 trace / runtime values / shadow store 输出格式

### 任务

1. 建立 `xdsl op -> runtime semantic` 分发器
2. 保留 `RuntimeEvent` / `RuntimeValue` 数据结构
3. 保留 ShadowStore 写入逻辑
4. 把 block / branch / loop 解释建立在 xDSL block/region 上

### 验收标准

- `test_execute_simple_branch_loop` 通过
- `test_execute_stack_decoder_and_build_factgraph` 通过
- runtime trace 规模与旧实现接近

---

## 阶段 4：Region / Contract / FactGraph 切到底层 xDSL

### 目标

- 不改语义，只改底层数据来源

### 任务

1. `RewriteRegion` 改用 xDSL operation/value 引用
2. `BoundaryContract` 改从 xDSL def-use / block 分析推导
3. `FactGraph` 静态层从 xDSL 抽取
4. Projection 逻辑继续保持可选

### 验收标准

- `test_define_region_and_infer_contract` 通过
- `test_optional_projection_annotation` 通过

---

## 阶段 5：grafting 迁到 xDSL rewrite

### 目标

- 彻底消除手工维护 `op_ids / block.op_ids / use_ops / def_op`

### 任务

1. donor graft 改为 xDSL rewrite pattern + builder API
2. `OverridePlan` 保留
3. graft 过程中的 `replace_all_uses_with`、`erase`、`insert` 改由 xDSL 完成
4. graft 后自动跑 verifier

### 验收标准

- `test_stack_decoder_bp_grafting_demo` 通过
- graft 前后 block 展示保持清晰
- 不再需要 `_recompute_use_def`

---

## 阶段 6：类型增强

### 目标

- 加入 richer types，而不破坏旧路径

### 任务

1. 引入：
   - `i1/i32/i64/index`
   - `f32/f64`
   - `complex<f32/f64>`
   - `vector`
   - `tensor`
2. 为通信算法新增：
   - complex vectors
   - complex matrices
   - score tables / belief tensors
3. 在 `alg` dialect 中保留动态对象引用类型
4. 新增类型转换 pass：
   - Python scalar -> arith scalar
   - Python complex -> complex dialect
   - Python list of float -> tensor/vector（当可静态化时）

### 验收标准

- 原有 7 测试全绿
- 新增 richer type 测试全绿

---

## 阶段 7：可选下行到官方 MLIR / CIRCT

### 目标

- 不是当前必须项
- 作为未来扩展口

### 任务

1. 增加 `export_to_mlir` 接口
2. 纯数值子图 lower 到标准 MLIR dialect
3. 如需硬件/RTL 路线，再考虑下接 CIRCT

### 验收标准

- 不影响现有 Python 运行闭环
- 只是新增导出能力

---

## 14. 文件级迁移建议

### 14.1 建议新增目录

```text
research/algorithm-IR/algorithm_ir_xdsl/
  dialects/
    alg_dialect.py
    alg_types.py
    alg_ops.py
  lowering/
    ast_to_alg.py
    alg_to_legacy_view.py
  compat/
    legacy_ir_view.py
    public_api.py
  runtime/
    execute_alg.py
  rewrite/
    graft_patterns.py
    donor_lowering.py
  tests/
    ...
```

### 14.2 对当前目录的建议

当前 `algorithm_ir/` 不要立即删除。  
建议分三步：

1. 保留旧目录，新增 `algorithm_ir_xdsl/`
2. public API 逐步重定向到新实现
3. 等所有测试与 demo 稳定后，再收敛目录结构

---

## 15. 风险分析

### 15.1 最大风险

1. xDSL 的类型/方言设计不当，导致后续又被迫重构
2. compatibility layer 不充分，导致旧测试大量破坏
3. 运行时 trace 与 ShadowStore 对 xDSL IR 的映射不稳定
4. 过早把动态 Python 语义强行静态化，反而损失结构中立性

### 15.2 风险控制策略

1. 先做 `alg` 中立 dialect，再做标准 dialect lowering
2. 保持旧测试和 demo 一直在线
3. 所有重大切换都走双轨：
   - 旧实现
   - 新 xDSL 实现
4. 每阶段只切一层，不同时切 frontend + runtime + rewriter

---

## 16. 成功标准

迁移完成后，必须同时满足：

1. `demo_outputs.py` 仍可运行
2. stack/BP graft demo 仍成立
3. 现有 7 个测试全部通过
4. `compile_function_to_ir` 仍可被外部直接调用
5. `graft_skeleton` 仍可直接返回新的 IR artifact
6. richer type 示例可以工作，至少包括：
   - complex scalar
   - complex vector
   - tensor / matrix
7. 生成的 IR 具备更强 verifier / printer / rewrite 基础设施

---

## 17. 推荐执行顺序

如果现在立刻开始做，我建议按以下顺序执行：

1. 阶段 0：引入 xDSL 与 smoke test
2. 阶段 1：定义 `alg` dialect 与 legacy view
3. 阶段 2：前端 lowering 到 xDSL
4. 阶段 3：运行时桥接
5. 阶段 4：region/contract/factgraph 切到底层 xDSL
6. 阶段 5：grafting 迁到 xDSL rewrite
7. 阶段 6：引入 richer types
8. 阶段 7：可选导出 MLIR / CIRCT

不要颠倒顺序。  
尤其不要在还没有稳定 compat layer 时，就急着重做 donor graft 或引入太多新类型。

---

## 18. 最终建议

这次迁移不应该被理解成“把 Python 研究原型替换成重型编译器项目”，而应该被理解成：

**把当前最脆弱、最手工、最难扩展的 IR 基础设施替换成 xDSL 提供的成熟 SSA/Block/Region/Rewrite 底座，同时保留你自己真正有研究价值的结构中立 region-rewrite 语义层。**

简短地说：

- **选 xDSL，不选直接 MLIR Python bindings**
- **先兼容，再替换**
- **先保测试，再扩类型**
- **先换底座，不换研究核心**

---

## 19. 参考资料

- xDSL GitHub README: <https://github.com/xdslproject/xdsl>
- xDSL Pattern Rewriter: <https://docs.xdsl.dev/reference/pattern_rewriter/>
- MLIR Python Bindings: <https://mlir.llvm.org/docs/Bindings/Python/>
- MLIR Python API docs: <https://mlir.llvm.org/python-bindings/>
- CIRCT Python Bindings: <https://circt.llvm.org/docs/PythonBindings/>
