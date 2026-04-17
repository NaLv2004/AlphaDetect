# Algorithm-IR Code Review Report

**日期**: 2026-04-18  
**审查范围**: `research/algorithm-IR/algorithm_ir/` 全部模块及 `tests/`  
**审查目标**: 评估代码是否真正实现了所声称的"结构中立算法IR + 灵活嫁接移植"能力

---

## 一、总体评价

### 1.1 整体架构评分

| 维度 | 评分 (1-5) | 说明 |
|------|-----------|------|
| 架构设计 | 4/5 | 九层架构清晰，关注点分离良好 |
| 代码实现质量 | 3/5 | 核心路径可用，但有多个实质性bug和遗漏 |
| 声称功能的真实性 | 2/5 | "灵活嫁接"严重受限，grafting实质上是硬编码的 |
| 测试覆盖 | 2/5 | 现有10个测试全部通过，但覆盖面窄，未暴露已知bug |
| 可扩展性 | 2/5 | 当前架构难以扩展到新类型的skeleton |

### 1.2 核心发现摘要

1. **Grafting引擎是硬编码的**：`graft_skeleton()` 只支持两种固定skeleton名称 (`bp_summary_update` 和 `bp_tree_runtime_update`)，任何其他名称直接 `raise NotImplementedError`。这意味着该框架**无法**灵活地将任意算法skeleton嫁接到任意算法中。
2. **for循环解释器有bug**：`_compile_for` 没有正确维护 `name_env`，导致 for-loop 中累加变量不能正确更新。
3. **类型推断对函数参数完全失效**：所有函数参数的 `type_hint` 固定为 `"object"`，即使 Python 类型注解为 `int`/`float`。
4. **Module-level属性访问不支持**：`math.sqrt(x)` 这样的表达式会崩溃，因为 `import math` 定义在外层作用域，局部函数的 `globals_dict` 中没有 `math`。
5. **Codegen是假的**：`emit_artifact_source()` 只是调用 `render_function_ir()`，输出的是IR文本dump，不是可执行Python代码。
6. **Projection评分是硬编码常数**：`local_interaction` 固定返回 0.7，`scheduling` 固定返回 0.6，不依赖任何实际数据分析。

---

## 二、逐模块详细审查

### 2.1 `frontend/` — Python → IR 前端

**已实现且工作正常的功能:**
- 标量赋值、算术表达式、比较表达式 ✅
- if/else（包括嵌套） ✅
- while循环（包括嵌套） ✅
- 函数调用（builtin如 `len`, `abs`） ✅
- list/dict/tuple 的构造与索引 ✅
- 复数字面量 ✅
- 早期return ✅
- 不支持的AST节点会正确拒绝（lambda, list comprehension, try/except, with, class, yield, starred） ✅

**Bug: for循环累加语义错误**

```python
def fn(items: list) -> int:
    s = 0
    for item in items:
        s = s + item
    return s
```

IR执行结果为 4（最后一个元素），Python 正确结果为 10（总和）。

**根因**: `_compile_for()` 在 for-body 结束后没有把 body 中更新过的 `name_env` 反馈到循环头部的 phi 节点。与 `_compile_while()` 的实现对比，while 有一个 `loop_phi_inputs` 的 backedge 更新逻辑，但 `_compile_for()` 完全没有这个逻辑。

**修复建议**: 参照 `_compile_while()` 中的 `loop_phi_inputs` 和 backedge 更新机制，在 `_compile_for()` 中也为所有在循环体中被修改的变量创建 phi 节点，并在循环体结束后更新 backedge 输入。

---

**Bug: 函数参数类型注解被忽略**

所有函数参数的 `type_hint` 一律设为 `"object"`：

```python
# ir_builder.py, line ~89
value_id = self._new_value(
    name_hint=arg.arg,
    type_hint="object",  # ← 固定为 object，忽略了 Python 类型注解
    ...
)
```

这导致 `int + int` 的二元运算结果类型也变成 `"object"` 而非 `"int"`，因为类型推断依赖输入类型。

**修复建议**: 解析 `ast.arg.annotation`，将 `int` / `float` / `bool` / `str` / `complex` 等简单注解映射为对应的 `type_hint`。

---

**Bug: Module-level 属性访问（如 `math.sqrt`）不支持**

`math.sqrt(x)` 解析为 `ast.Attribute(ast.Name("math"), "sqrt")`。`ast.Name("math")` 的编译路径走到 `_load_global("math")`，但如果 `math` 不在被编译函数的 `globals_dict` 中（例如 `import math` 在测试方法中而非模块级），就会抛出 `NameError`。

即使 `math` 在 `globals_dict` 中，`_load_global` 会把整个 `math` 模块作为常量存入 IR（`attrs={"literal": <module 'math'>}`），但模块对象无法被 xDSL payload 的 `_normalize_payload` 正确序列化。

**修复建议**: 
1. 对 `ast.Attribute` 形式的函数调用 `module.func(args)` 做特殊处理
2. 或者在 `_load_global` 中检查对象是否为模块，将其标记为特殊类型而非尝试序列化整个模块

---

### 2.2 `ir/` — IR 数据模型

**整体**: 设计合理，Value/Op/Block/FunctionIR 四个层次清晰。xDSL 作为真实底座的设计是正确的。

**问题: xDSL 类型映射未被实际使用**

`xdsl_bridge.py` 中 `_type_to_xdsl_attr()` 有 `int → i64`, `float → f64`, `bool → i1` 的映射，但由于参数类型始终是 `"object"`，这些映射形同虚设。实际 xDSL 输出中只有 `"object"` 字符串类型。

**问题: `_next_prefixed_id` 有潜在crash**

```python
def _next_prefixed_id(existing: dict[str, Any], prefix: str) -> str:
    next_id = max(
        (int(item_id.split("_", maxsplit=1)[1]) for item_id in existing if item_id.startswith(prefix)),
        default=-1,
    ) + 1
```

当 `prefix="op_"` 而 `existing` 中有 `"op_return_5"` 这样的 key 时，`int("return_5")` 会抛出 `ValueError`。当前代码碰巧不会产生这种 ID，但这是一个脆弱的假设。

**修复建议**: 用正则表达式 `re.match(rf"^{prefix}(\d+)$", item_id)` 来精确匹配。

---

### 2.3 `runtime/` — IR 解释器

**整体**: 解释器实现得相当完整，支持所有声称的 opcode。

**已验证正确的功能:**
- 整数/浮点/复数算术 ✅
- 分支和循环 ✅
- 列表/字典/元组操作 ✅
- 内置函数调用 ✅
- fibonacci、冒泡排序、GCD 等算法 ✅
- 运行时trace和ShadowStore追踪 ✅

**问题: phi 节点在 for 循环中的语义不正确**

由于 `_compile_for()` 没有正确生成 phi 节点，解释器中 phi 的选择逻辑虽然本身没问题，但前端给出的 IR 就已经错了。

**问题: `_build_control_context` 过于简单**

只记录 `block:id` 和 `loop:id:iter=N`，没有记录嵌套循环层次、if分支路径等信息。这限制了后续分析（如区分"在循环第3次迭代的if-true分支中"这样的精确定位）。

---

### 2.4 `region/` — 区域选择与边界契约

**已实现且工作正常的功能:**
- 通过 op_ids 显式定义区域 ✅
- 通过 source_span 定义区域 ✅
- 通过 exit_values 的 backward slice 定义区域 ✅
- 通过 state_carriers 的 forward slice 定义区域 ✅
- 推断 entry/exit values ✅
- 推断 read/write sets ✅
- 推断 reconnect points ✅

**问题: `state_carriers` 的自动推断过于粗糙**

当用户不提供 `state_carriers` 时，代码自动选择所有类型为 `list`/`dict`/`object` 的值作为 state carrier。这对于大型算法会产生大量虚假的 state carrier。

**问题: `schedule_anchors.loop_blocks` 依赖 `loop_backedge` attr**

只有当 op 的 attrs 中有 `loop_backedge` 标记时才会识别循环块。这个标记只在 `_compile_while` 和 `_compile_for` 的 jump 指令中设置，但如果区域是通过 source_span 选中的并且不包含 backedge 指令，则 `loop_blocks` 会为空。

---

### 2.5 `grafting/` — 嫁接引擎 ⚠️ 核心问题所在

**这是整个系统最关键的模块，也是问题最严重的模块。**

#### 问题 1: `graft_skeleton()` 是硬编码的 dispatch（严重）

```python
def graft_skeleton(func_ir, region, contract, skeleton, projection=None):
    if not match_skeleton(func_ir, region, contract, skeleton):
        raise ValueError(...)
    if skeleton.name == "bp_summary_update":
        return _graft_bp_summary(...)
    if skeleton.name == "bp_tree_runtime_update":
        return _graft_bp_tree_runtime_update(...)
    raise NotImplementedError(f"Unsupported skeleton {skeleton.name}")
```

这意味着:
- 框架**不支持**任何自定义skeleton
- 框架**不支持**任何非BP类的嫁接
- 要添加新的嫁接类型，必须在 `rewriter.py` 中手写一个新的 `_graft_xxx()` 函数
- 这与 ir_plan.md 中声称的"通用重写引擎"**严重不符**

#### 问题 2: `_graft_bp_summary()` 的逻辑是特定于场景的

该函数做了以下特定于 "BP summary injection into stack decoder score" 的事情:
1. 查找名为 `frontier`、`costs`、`candidate` 的值
2. 构建一个 `call(bp_fn, frontier, costs, damping)` 调用
3. 构建 `candidate["metric"] + bp_summary` 计算
4. 用新的 score 替代旧的 score

这是一个**完全手工编排的、针对特定场景的重写脚本**，不是一个通用的重写引擎。

#### 问题 3: `match_skeleton()` 的匹配逻辑过于宽松

```python
def match_skeleton(func_ir, region, contract, skeleton):
    # ... check scalar output ...
    required_names = set(skeleton.required_contract.get("needs_inputs", []))
    available_names = {
        func_ir.values[value_id].attrs.get("var_name")
        for value_id in region.entry_values + region.exit_values + region.state_carriers
    }
    available_names.update(
        value.attrs.get("var_name")
        for value in func_ir.values.values()
        if value.attrs.get("var_name") is not None
    )
    return required_names <= available_names
```

它搜索**整个 IR 中所有值的 var_name**，而不仅仅是区域内的值。这意味着只要函数中任何地方有一个变量名叫 `frontier`，匹配就会成功，即使该变量与选中区域毫无关系。

#### 问题 4: `Skeleton` 的 `transform_rules` 字段从未被使用

```python
@dataclass
class Skeleton:
    ...
    transform_rules: list[dict[str, Any]]  # 定义了但从未读取
    ...
```

`transform_rules` 本应驱动通用重写逻辑，但实际上 `graft_skeleton()` 完全不读它，直接根据 `skeleton.name` 跳到硬编码的函数。

#### 问题 5: `OverridePlan` 只是记录，不驱动行为

`OverridePlan` 看起来像是一个"重写计划"的数据结构，但它是在重写**完成之后**才被构造的。它只是一个事后日志，不是驱动重写的指令。

**修复建议**: 
1. 设计一个真正的 transform rule DSL，让 `transform_rules` 能描述"删除哪些ops、插入什么调用、如何重连"
2. 让 `graft_skeleton()` 解释执行 `transform_rules`，而不是根据名字分派到硬编码函数
3. 或者更实际地，将 `_graft_bp_summary` 和 `_graft_bp_tree_runtime_update` 中的共同逻辑提取为通用原语（如 `remove_score_slice_and_replace_with_call`）

---

### 2.6 `projection/` — 投影层

**问题 1: 评分是硬编码常数**

- `detect_local_interaction_projection` 始终返回 `score=0.7`
- `detect_scheduling_projection` 始终返回 `score=0.6`

这些分数不依赖任何数据分析，没有实际意义。

**问题 2: `edge_set` 始终为空**

两个投影检测器都返回 `edge_set=[]`。对于声称在检测"局部交互"或"调度"结构的模块，不产生任何边/关系信息，使得投影实际上只是一个标签。

**问题 3: 检测逻辑过于简单**

`detect_local_interaction_projection`: 只要区域有 entry_values 和 exit_values 就返回非None。这意味着**任何非空区域**都会被标记为"局部交互"。

`detect_scheduling_projection`: 只要 `write_set` 中有 `"container:"` 或者有 state_carriers 就返回非None。

**修复建议**: 如果投影层目前只是占位符，应在文档中明确说明是 placeholder，避免误导。

---

### 2.7 `analysis/` — 分析工具

**整体**: 三个分析工具都非常简单：

- `static_analysis.py`: `def_use_edges()` 和 `block_uses()` — 正确但trivial
- `dynamic_analysis.py`: `runtime_values_for_static()` — 一个简单的过滤函数
- `fingerprints.py`: `fingerprint_runtime_value()` — 提取4个metadata字段组成tuple

这些函数实现正确，但过于简单，不足以支撑有意义的算法分析。

---

### 2.8 `factgraph/` — 事实图

**已实现且工作正常**: 构建了包含以下边类型的统一图：
- 静态: `def_use`, `use_def`, `cfg`, `call_static`
- 动态: `event_input`, `event_output`, `temporal`, `control_dynamic`, `container_membership`, `field_slot`
- 对齐: `instantiates_op`, `instantiates_value`, `runtime_in_frame`

**问题**: FactGraph 被构建出来但**几乎不被使用**。唯一使用 FactGraph 的地方是 `annotate_region()` 中的一行代码，仅读取 `fg.metadata.get("function_name")` 写到 projection 的 evidence 中。FactGraph 中精心构建的所有边类型在整个系统中没有任何消费者。

---

### 2.9 `regeneration/` — 再生模块

**问题**: `codegen.py` 只有一个函数：

```python
def emit_artifact_source(artifact):
    return render_function_ir(artifact.ir)
```

这只是返回 IR 的文本dump（形如 `FunctionIR(name=...) Block b_entry_0 ...`），**不是可执行 Python 代码**。

ir_plan.md 第四原则要求"新算法不能仅输出源码；必须输出统一 IR、可执行体、与 IR 对齐的 tracing schema、projection 注释"。当前实现的确可以生成可执行 IR（通过 `execute_ir`），但没有可读的源码再生能力。

---

## 三、测试结果

### 3.1 现有测试 (10个) — 全部通过

```
tests/unit/test_frontend.py           3 passed
tests/unit/test_region_projection.py  2 passed
tests/unit/test_runtime_factgraph.py  3 passed
tests/integration/test_grafting_demo.py  2 passed
```

### 3.2 新增测试 (101个) — 96通过, 5失败

```
code_review/test_frontend_edge_cases.py    28 passed, 3 failed
code_review/test_interpreter_edge_cases.py 16 passed
code_review/test_region_boundary.py        12 passed
code_review/test_grafting_generality.py    10 passed
code_review/test_factgraph_analysis.py     12 passed
code_review/test_projection_quality.py      7 passed
code_review/test_xdsl_backend.py           10 passed, 2 failed
code_review/test_end_to_end.py              9 passed
```

### 3.3 失败测试详情

| 测试 | 失败原因 | 严重程度 |
|------|---------|---------|
| `test_for_loop` | for循环累加结果错误 (返回4而非10) | **高** — 真实bug |
| `test_external_function_call` | `math.sqrt(x)` 编译失败，`math` 不在 globals | **中** — 功能缺失 |
| `test_int_types_propagated` | 参数类型始终为 `object` | **中** — 类型系统未生效 |
| `test_int_maps_to_i64` | xDSL输出中无 i64，全是 `"object"` 字符串 | **中** — xDSL类型映射失效 |
| `test_float_maps_to_f64` | xDSL输出中无 f64，全是 `"object"` 字符串 | **中** — xDSL类型映射失效 |

### 3.4 关键通过测试（证实的能力边界）

| 能力 | 验证结果 |
|------|---------|
| 编译 fibonacci、冒泡排序、GCD | ✅ 正确 |
| while 嵌套循环 | ✅ 正确 |
| 早期return | ✅ 正确 |
| 嵌套 if/else | ✅ 正确 |
| 列表/字典/元组操作 | ✅ 正确 |
| clone产生独立等价副本 | ✅ 正确 |
| backward/forward slice | ✅ 正确 |
| 区域定义（op_ids/source_span/exit_values） | ✅ 正确 |
| 边界契约推断 | ✅ 正确 |
| BP summary skeleton嫁接 | ✅ 可用 |
| BP runtime skeleton嫁接 | ✅ 可用 |
| 嫁接后IR可再分析 | ✅ 可用 |
| 嫁接后IR可重新定义区域 | ✅ 可用 |
| 自定义skeleton嫁接 | ❌ NotImplementedError |

---

## 四、"灵活嫁接移植"能力评估

### 4.1 当前实际能力

该框架**可以**做到：
1. 将Python函数编译为结构中立的可执行IR（有bug但大部分工作）
2. 执行IR并收集运行时trace
3. 在IR上选择区域并推断边界
4. 将**特定的两种BP skeleton**嫁接到stack decoder的**特定区域**
5. 嫁接后的IR仍然可执行、可分析

该框架**不能**做到：
1. 将任意donor算法的skeleton嫁接到任意host算法中
2. 自动发现哪些区域适合哪些skeleton
3. 从IR再生可读的Python源代码
4. 利用FactGraph做有意义的算法结构分析
5. 基于projection做有意义的skeleton匹配

### 4.2 结论

当前代码是一个**概念验证 (PoC)**，成功演示了"编译→执行→追踪→切区域→嫁接→重执行"的完整pipeline。但距离"灵活的算法嫁接移植框架"还有很大距离。核心瓶颈是 grafting 引擎的硬编码本质。

---

## 五、优先修复建议

### P0 — 必须修复

1. **修复 `_compile_for()` 的循环变量更新bug**  
   参照 `_compile_while()` 的 `loop_phi_inputs` 机制，为 for-body 中修改的变量添加 phi 节点和 backedge 更新。

2. **修复函数参数类型推断**  
   解析 `ast.arg.annotation`，将简单类型注解映射为 `type_hint`。

3. **定义正式的 `alg` Dialect，替代 `UnregisteredOp`**  
   这是所有后续改进的前提。使用 xDSL IRDL 定义每个 op 的操作数和结果类型，让 xDSL SSA 系统接管 def-use 维护。详见第六节。

### P1 — 强烈建议

4. **引入 `AlgSlot` op，实现通用的骨架填充机制**  
   `AlgSlot` 是 IR 中的可替换占位符。GP 的交叉、变异、skeleton 填充都通过 `fill_slot()` 接口完成，底层使用 xDSL `PatternRewriter`。详见 6.4 节。

5. **用 xDSL Pattern Rewriting 替代硬编码的 `graft_skeleton()`**  
   每种嫁接逻辑写成一个 `RewritePattern` 子类，新 skeleton 不需要修改 dispatch 逻辑。

6. **实现 Python 源码再生**  
   让 `codegen.py` 能将 IR 转回可读的 Python 函数，这对调试和验证至关重要。

7. **修复 `match_skeleton()` 的匹配范围**  
   只在 `region.entry_values + region.exit_values` 中搜索 `var_name`，而非全局搜索。

### P2 — 建议改进

8. **实现 GP 原语操作** (`crossover()`, `mutate_op()`, `extract_subgraph()`)  
   基于 `AlgSlot` + xDSL PatternRewriter 实现，为后续遗传编程框架提供接口。详见 6.4 节。

9. **支持 module-level 函数调用 (`math.sqrt` 等)**  
   在前端中特殊处理 `ast.Attribute` 形式的函数调用。

10. **加强 `_next_prefixed_id` 的鲁棒性**  
    使用正则匹配而非简单 split，避免非数字 ID 导致 crash。

---

## 六、xDSL 使用现状分析与改进方案

### 6.1 xDSL 当前实际发挥的作用

当前代码使用 xDSL v0.62.0，但仅将其用作**不透明的存储容器**，核心编译器能力完全未被利用。

#### 6.1.1 当前使用方式一览

| 使用场景 | 如何使用 xDSL | 实际效果 |
|----------|-------------|---------|
| IR 存储 | `UnregisteredOp.with_name("alg.xxx")` + `payload` 字符串 | xDSL 看不懂 payload，只是透传 |
| 类型标注 | `_type_to_xdsl_attr()` 映射 int→i64, float→f64 | 由于 bug 参数全是 `"object"`，映射形同虚设 |
| Clone | `ModuleOp.clone()` → `FunctionIR.from_xdsl()` | 有效，但仅利用了通用对象克隆 |
| 嫁接变异 | `Block.insert_ops_before()` / `Block.erase_op()` | 有效，利用了 xDSL Block 操作 API |
| 文本输出 | `Printer` 渲染 xDSL 文本 | 输出中有意义的信息全在 payload 字符串里 |

#### 6.1.2 详细问题

**问题 1: 所有 op 使用 `UnregisteredOp`，xDSL 无法理解 IR 语义**

```python
# xdsl_bridge.py
op_cls = UnregisteredOp.with_name(f"alg.{op.opcode}")
xdsl_op = op_cls.create(
    attributes={"payload": StringAttr(repr(_normalize_payload(op_payload)))},
    successors=successors,
)
```

每个 op 的全部语义信息（inputs、outputs、opcode、attrs）被序列化为 Python `repr()` 字符串，塞进一个 `StringAttr("payload")` 中。xDSL 完全不知道这个 op 有几个操作数、结果类型是什么、def-use 关系如何。

**问题 2: Python 层维护了一套完整的冗余 IR 基础设施**

`ir/model.py` 中的 `Value.use_ops`、`Value.def_op`、`Op.inputs`、`Op.outputs` 构成了一套完整的 def-use 图，与 xDSL 的原生 SSA value 系统**完全重复**。代码在两套系统之间通过 `from_xdsl()` 同步，增加了复杂度和出错风险。

**问题 3: xDSL 的 SSA value 系统未被使用**

xDSL 的 `OpResult` 和 `BlockArgument` 是真正的 SSA value，它们自动追踪 def-use 关系。当前代码中：
- op 的 `inputs` 是**字符串 ID 列表** (`["v_0", "v_1"]`)，不是 xDSL SSA value 引用
- 要查找某个 value 的所有使用者，需要遍历 `value.use_ops` 列表（Python 层手动维护）
- xDSL 层面，所有 op 都是零操作数的 `UnregisteredOp`，没有 SSA 连接

**问题 4: xDSL 类型系统完全未生效**

`_type_to_xdsl_attr()` 存在 int→i64、float→f64、bool→i1 的映射，但由于前端 bug，所有函数参数类型都是 `"object"`。xDSL 的 result types 全是 `StringAttr("object")`，没有任何类型检查。

#### 6.1.3 核心能力未被利用

| xDSL 能力 | 当前状态 | 说明 |
|-----------|---------|------|
| **自定义 Dialect（方言）** | ❌ 未使用 | 用 `UnregisteredOp` 代替 |
| **SSA Value 系统** | ❌ 未使用 | def-use 关系在 Python dict 层维护 |
| **IRDL 类型约束** | ❌ 未使用 | 无类型验证 |
| **Pattern Rewriting** | ❌ 未使用 | 嫁接逻辑硬编码在 Python 函数中 |
| **PatternRewriteWalker** | ❌ 未使用 | 无自动 pattern 遍历 |
| **Pass 基础设施** | ❌ 未使用 | 无 xDSL pass pipeline |
| **Verification** | ❌ 未使用 | 自定义 `validate_function_ir()` 代替 |
| **Region 嵌套** | ❌ 未使用 | 平铺 Block + branch/jump |

---

### 6.2 xDSL 应该发挥什么作用

xDSL 是一个 Python-native 的 MLIR 实现，其核心价值不是"存储"，而是**编译器基础设施**。以下是 xDSL 应该为 algorithm-IR 提供的能力层次：

#### 层次 1: IR 表示层 — 自定义 `alg` Dialect

使用 `irdl_op_definition` 定义正式的操作类型，让 xDSL 理解每个 op 的结构：

```python
from xdsl.irdl import (
    irdl_op_definition, IRDLOperation, AnyAttr,
    operand_def, result_def, attr_def, var_operand_def,
    opt_attr_def, region_def, successor_def,
)
from xdsl.dialects.builtin import StringAttr, i64, f64
from xdsl.ir import Dialect

@irdl_op_definition
class AlgConst(IRDLOperation):
    """常量定义"""
    name = "alg.const"
    res = result_def(AnyAttr())
    value = attr_def(StringAttr)  # 常量值的字符串表示

@irdl_op_definition
class AlgBinary(IRDLOperation):
    """二元运算: Add, Sub, Mul, Div, ..."""
    name = "alg.binary"
    lhs = operand_def(AnyAttr())    # ← xDSL SSA operand
    rhs = operand_def(AnyAttr())    # ← xDSL SSA operand
    res = result_def(AnyAttr())     # ← xDSL SSA result
    operator = attr_def(StringAttr) # "Add", "Sub", "Mul", ...

@irdl_op_definition
class AlgCall(IRDLOperation):
    """函数调用"""
    name = "alg.call"
    callee = operand_def(AnyAttr())
    args = var_operand_def(AnyAttr())
    res = result_def(AnyAttr())
    func_name = attr_def(StringAttr)

@irdl_op_definition
class AlgBranch(IRDLOperation):
    """条件分支"""
    name = "alg.branch"
    cond = operand_def(AnyAttr())
    true_dest = successor_def()
    false_dest = successor_def()

@irdl_op_definition
class AlgPhi(IRDLOperation):
    """SSA phi 节点"""
    name = "alg.phi"
    incoming = var_operand_def(AnyAttr())
    res = result_def(AnyAttr())

AlgDialect = Dialect("alg", [AlgConst, AlgBinary, AlgCall, AlgBranch, AlgPhi, ...])
```

**效果**: xDSL 理解每个 op 有几个操作数、几个结果；def-use 图由 xDSL SSA 自动维护；不再需要 Python 层的 `values` / `use_ops` / `def_op`。

#### 层次 2: 变换层 — Pattern Rewriting

使用 xDSL 的 `RewritePattern` + `PatternRewriteWalker` 替代硬编码的嫁接逻辑。

#### 层次 3: 验证层 — IRDL 约束 + 自定义 Verifier

利用 xDSL 的类型系统对 IR 进行构造时验证，减少运行时错误。

---

### 6.3 xDSL Pattern Rewriter 能否替代当前的 skeleton 替换？— 实验验证

**回答: 能。** 以下是已验证的实验。

我们用 xDSL v0.62.0 构建了一个最小的 `alg` dialect，并使用 `PatternRewriteWalker` + `op_type_rewrite_pattern` 成功实现了 skeleton 注入：

```python
# 定义 RewritePattern: 找到 AlgBinary(Add)，注入 BP summary 调用
class InjectBPSummaryPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: AlgBinary, rewriter: PatternRewriter):
        if op.operator.data != "Add":
            return  # 不匹配
        # 构建 donor 常量
        bp_const = AlgConst.build(result_types=[f64],
                                  attributes={"value": StringAttr("bp_fn")})
        # 构建调用: bp_result = call bp_fn(lhs)
        bp_call = AlgCall.build(operands=[bp_const.res, [op.lhs]],
                                result_types=[f64],
                                attributes={"func_name": StringAttr("bp_summary")})
        # 构建新加法: new_score = lhs + bp_result
        new_add = AlgBinary.build(operands=[op.lhs, bp_call.res],
                                  result_types=[f64],
                                  attributes={"operator": StringAttr("Add")})
        rewriter.replace_op(op, [bp_const, bp_call, new_add])

# 执行
walker = PatternRewriteWalker(
    GreedyRewritePatternApplier([InjectBPSummaryPattern()]),
    apply_recursively=False,
)
walker.rewrite_module(module)
```

**实验结果** (已在 AutoGenOld 环境中运行验证):

```
BEFORE: %2 = "alg.binary"(%0, %1) {operator = "Add"} : (f64, f64) -> f64
AFTER:  %2 = "alg.const"() {value = "bp_fn"} : () -> f64
        %3 = "alg.call"(%2, %0) {func_name = "bp_summary"} : (f64, f64) -> f64
        %4 = "alg.binary"(%0, %3) {operator = "Add"} : (f64, f64) -> f64
```

xDSL 的 `rewriter.replace_op()` 自动完成了：
1. 在原 op 之前插入新 ops
2. 将原 op 结果的所有使用者重连到新 op 的结果
3. 安全删除原 op
4. SSA def-use 链自动更新

**结论**: `PatternRewriter` 完全能做到当前 `_graft_bp_summary()` 手工编排的所有操作，而且更安全（自动管理 SSA 连接）、更可扩展（新 pattern 只需写新类）。

#### 6.3.1 Pattern Rewriter 与当前硬编码方式的对比

| 维度 | 当前硬编码 | xDSL Pattern Rewriter |
|------|-----------|---------------------|
| 添加新 skeleton | 需在 `rewriter.py` 中写新的 `_graft_xxx()` 函数 + 修改 dispatch | 只需写新的 `RewritePattern` 子类 |
| SSA 重连接 | 手动调用 `_rebuild_from_xdsl()` 重建整个 IR | `rewriter.replace_op()` 自动重连 |
| 匹配逻辑 | 按 `skeleton.name` 字符串分派 | 按 op 类型 + 属性值匹配 |
| 可组合性 | 不可组合 | 多个 pattern 可组合为 `GreedyRewritePatternApplier` |
| 安全性 | 无安全检查，可能产生悬挂引用 | `safe_erase=True` 自动检查残留使用 |

---

### 6.4 面向遗传编程的 Skeleton/IR 操作接口设计

当前框架声称支持"灵活的嫁接移植"，但如果后续框架采用**遗传编程 (GP)** 来发现新算法，则 IR 操作接口需要支持以下核心操作：

1. **交叉 (Crossover)**: 从两个父代 IR 中各选一个子图，交换子图得到两个子代
2. **变异 (Mutation)**: 随机修改 IR 中的某个操作（替换 opcode、修改常量、添加/删除 op）
3. **骨架填充 (Skeleton Filling)**: 给定一个带"洞"的算法骨架，填入具体的计算子图
4. **子图提取 (Subgraph Extraction)**: 从完整 IR 中提取一个可独立运行的子图作为 skeleton

当前设计对以上操作的支持度：

| GP 操作 | 当前支持 | 需要的接口 |
|---------|---------|-----------|
| Crossover | ❌ 无接口 | 子图交换 + SSA 重连 |
| Mutation | ❌ 无接口 | 单 op 替换/属性修改 |
| Skeleton Filling | ⚠️ 仅两种硬编码 | 通用的"洞→子图"替换 |
| Subgraph Extraction | ⚠️ Region 选择可用 | Region → 独立 FunctionIR |

#### 6.4.1 推荐的接口设计

##### 核心概念: `Slot` — IR 中的可替换位置

在 GP 中，"skeleton" 不应该是一个固定名称的硬编码对象，而应该是一段**带有 Slot（槽位）标记的 IR 片段**。Slot 是 IR 中可以被替换的位置，类似于模板中的占位符。

```python
@irdl_op_definition
class AlgSlot(IRDLOperation):
    """
    Slot: IR 中可被 GP 操作替换的占位符。
    
    一个 Slot 代表"这里应该有一段计算，但具体是什么由 GP 决定"。
    Slot 声明了它接受的输入和产出的输出的类型签名，
    GP 操作只需要保证填充进去的子图满足这个类型签名即可。
    """
    name = "alg.slot"
    slot_inputs = var_operand_def(AnyAttr())   # Slot 接受的输入
    res = result_def(AnyAttr())                # Slot 的输出
    slot_id = attr_def(StringAttr)             # 唯一标识符
    slot_kind = opt_attr_def(StringAttr)       # "score", "update", "metric", ...
    # slot_kind 是语义标签，GP 可以用它来引导匹配，但不是硬性约束
```

**设计理念**: `AlgSlot` 是一个合法的 xDSL op，可以参与正常的 SSA 图。在解释执行时，slot 可以有默认行为（如直接返回第一个输入）；在 GP 操作时，slot 是可以被整体替换的单元。

##### 接口 1: `fill_slot()` — 骨架填充

```python
def fill_slot(
    module: ModuleOp,
    slot_id: str,
    filler: Region | list[Operation],
) -> ModuleOp:
    """
    将 slot_id 对应的 AlgSlot 替换为 filler 中的操作序列。
    
    filler 的最后一个 op 的结果将替代 AlgSlot 的结果（SSA 自动重连）。
    filler 中可以引用 AlgSlot 的 slot_inputs（通过 BlockArgument 映射）。
    
    底层使用 xDSL PatternRewriter 实现。
    """
```

**实现方式**: 写一个 `FillSlotPattern(RewritePattern)`，它匹配 `AlgSlot`，检查 `slot_id` 是否匹配，然后用 `rewriter.replace_op()` 替换。

##### 接口 2: `crossover()` — 子图交叉

```python
def crossover(
    parent_a: ModuleOp,
    parent_b: ModuleOp,
    region_a: Region,  # parent_a 中选定的子图区域
    region_b: Region,  # parent_b 中选定的子图区域
) -> tuple[ModuleOp, ModuleOp]:
    """
    交换两个父代 IR 中的选定子图区域，产生两个子代。
    
    要求:
    1. region_a 和 region_b 的边界签名兼容
       （输入数量和类型匹配，输出数量和类型匹配）
    2. 交换后的子图能正确连接到宿主的 SSA 图中
    
    实现步骤:
    1. clone 两个父代
    2. 从 clone_a 中提取 region_a 对应的 ops
    3. 从 clone_b 中提取 region_b 对应的 ops
    4. 用 PatternRewriter 将 clone_a 中的 region_a 替换为 region_b 的 ops
    5. 反向操作得到第二个子代
    """
```

**关键设计**: 交叉操作需要**类型兼容性检查**。如果 region_a 的边界签名是 `(f64, f64) → f64`，那么 region_b 也必须是 `(f64, f64) → f64`。这正是 xDSL 类型系统可以发挥作用的地方。

为了方便交叉，建议将可交叉的区域包装为 Slot：

```python
def wrap_region_as_slot(
    module: ModuleOp,
    op_ids: list[str],
    slot_id: str,
) -> ModuleOp:
    """
    将选定的 ops 替换为一个 AlgSlot，内部保存原始 ops 作为默认实现。
    这样交叉操作就变成了: 提取 slot 的内容 + 填充另一个 slot 的内容。
    """
```

##### 接口 3: `mutate_op()` — 单点变异

```python
def mutate_op(
    module: ModuleOp,
    target_op: Operation,
    mutation_kind: str,  # "replace_opcode", "modify_const", "swap_operands", ...
    **kwargs,
) -> ModuleOp:
    """
    对单个 op 进行变异。支持的变异类型:
    
    - "replace_opcode": 将 AlgBinary 的 operator 从 Add 改为 Mul 等
    - "modify_const": 修改 AlgConst 的 value
    - "swap_operands": 交换 AlgBinary 的 lhs 和 rhs
    - "insert_identity": 在某个 value 的 def-use 链中插入 identity op
    - "delete_op": 删除一个 op，将其输出直接连到其某个输入
    - "replace_with_slot": 将一个 op 替换为 AlgSlot
    
    底层每种变异对应一个 RewritePattern。
    """
```

##### 接口 4: `extract_subgraph()` — 子图提取

```python
def extract_subgraph(
    module: ModuleOp,
    op_ids: list[str],
) -> tuple[ModuleOp, list[Attribute], list[Attribute]]:
    """
    从 module 中提取指定 ops 组成的子图，返回:
    1. 包含子图的新 ModuleOp（子图被包装为一个函数）
    2. 子图的输入类型签名
    3. 子图的输出类型签名
    
    提取后的子图可以作为交叉或填充的素材。
    
    实现: 
    - 计算 op_ids 的闭包（所有需要的 ops）
    - 识别 entry values（子图需要但不在子图内定义的值）→ 函数参数
    - 识别 exit values（子图定义且在子图外使用的值）→ 函数返回值
    - 利用 xDSL Rewriter 将子图移动到新的 Region 中
    """
```

#### 6.4.2 完整的 GP 工作流示例

```python
# 1. 编译两个 Python 算法为 xDSL IR
ir_a = compile_to_xdsl(stack_decoder)
ir_b = compile_to_xdsl(bp_decoder)

# 2. 在 ir_a 中选择一个区域，包装为 slot
ir_a_slotted = wrap_region_as_slot(ir_a, 
    op_ids=find_score_computation(ir_a),
    slot_id="score_slot")

# 3. 从 ir_b 中提取一个子图
subgraph_b, input_sig, output_sig = extract_subgraph(ir_b,
    op_ids=find_message_passing(ir_b))

# 4. GP 交叉: 用 ir_b 的子图填充 ir_a 的 slot
child = fill_slot(ir_a_slotted, "score_slot", subgraph_b)

# 5. GP 变异: 随机修改一个常量
child = mutate_op(child, pick_random_const(child), "modify_const", 
    new_value=0.95)

# 6. 评估: 直接执行变异后的 IR
result = execute_xdsl_ir(child, test_input)
fitness = evaluate_ber(result)
```

#### 6.4.3 为什么 xDSL 原生功能对 GP 至关重要

| GP 需求 | 不用 xDSL 原生功能的问题 | 用 xDSL 原生功能的优势 |
|---------|----------------------|---------------------|
| SSA 重连接 | 每次操作后都要 `_rebuild_from_xdsl()` 全量重建 | `rewriter.replace_op()` 自动重连，O(1) |
| 类型检查 | 交叉后可能产生类型不匹配的 IR，要到解释执行时才会崩溃 | xDSL 在构造时即验证类型兼容性 |
| 子图提取 | 需要手动计算 entry/exit values 和维护 value ID 映射 | xDSL SSA 的 `uses` 属性直接可查 |
| 变异安全 | 删除 op 后可能留下悬挂引用 | `safe_erase=True` 自动检查 |
| 可组合性 | 每种操作是独立函数，无法链式执行 | 多个 RewritePattern 可组合为 pass pipeline |

#### 6.4.4 `AlgSlot` 与当前 `Skeleton` 的关系

当前的 `Skeleton` dataclass 试图描述"一段要被注入的计算"，但它不是 IR 的一部分——它是一个外部的 Python 对象，通过硬编码的 `_graft_xxx()` 函数被"翻译"为 IR ops。

改进后的设计中，skeleton 本身就是 IR：

```
当前设计:
  Skeleton (Python dataclass)  ──硬编码翻译──→  IR ops
  
改进设计:
  Skeleton = 包含 AlgSlot 的 IR fragment（本身就是 xDSL ops）
  fill_slot() 用 PatternRewriter 替换 AlgSlot → 完成注入
```

这意味着：
- Skeleton 不再是一个特殊的 Python 对象，而是一段普通的 IR
- `fill_slot()` 不需要知道 skeleton 的"名字"——它只需要知道 slot_id
- 新的 skeleton 类型不需要任何代码改动——只需要构造新的 IR fragment
- GP 的交叉操作自然等价于"从 parent_a 提取 slot 的内容，填入 parent_b 的 slot"

#### 6.4.5 实现路径建议

**Phase 1**: 定义 `alg` Dialect（最高优先级）
- 定义 `AlgConst`, `AlgBinary`, `AlgUnary`, `AlgCompare`, `AlgCall`, `AlgPhi`, `AlgBranch`, `AlgJump`, `AlgReturn`, `AlgGetItem`, `AlgSetItem`, `AlgSlot` 等所有 op 类型
- 修改 `ir_builder.py`，编译时直接生成 xDSL typed ops 而非 `UnregisteredOp`
- 删除 Python 层的 `Value.use_ops` / `Value.def_op` 等冗余数据结构，直接使用 xDSL 的 SSA
- 保留 `FunctionIR` 作为轻量包装器，但内部数据全部委托给 xDSL

**Phase 2**: 实现 GP 原语操作
- 实现 `fill_slot()`：基于 `RewritePattern` 的 slot 填充
- 实现 `mutate_op()`：基于 `PatternRewriter` 的单点变异
- 实现 `extract_subgraph()`：基于 xDSL Region/Block 操作的子图提取
- 实现 `crossover()`：基于 slot 的子图交换

**Phase 3**: 重构解释器
- 解释器直接遍历 xDSL ops（`for op in block.ops:`），读取 typed op 的属性
- 不再需要从 payload 字符串中反序列化

---

## 七、测试脚本清单

所有测试脚本位于 `research/algorithm-IR/code_review/` 目录：

| 文件 | 测试数 | 覆盖模块 |
|------|--------|---------|
| `test_frontend_edge_cases.py` | 31 | frontend, type tracking, unsupported constructs |
| `test_interpreter_edge_cases.py` | 16 | runtime interpreter, trace quality, edge cases |
| `test_region_boundary.py` | 12 | region selection, slicing, boundary contract |
| `test_grafting_generality.py` | 10 | grafting generality, skeleton matching, codegen |
| `test_factgraph_analysis.py` | 12 | factgraph construction, static/dynamic analysis |
| `test_projection_quality.py` | 7 | projection detection, scoring quality |
| `test_xdsl_backend.py` | 12 | xDSL integration, clone, type mapping |
| `test_end_to_end.py` | 9 | full pipeline stress, artifact reusability |

运行方式:
```powershell
cd research/algorithm-IR
conda run -n AutoGenOld python -m pytest code_review/ -v
```
