# `algorithm_ir` — 结构中立的算法中间表示

> 将 Python 算法函数编译为 SSA 形式的 IR，支持区域切片、边界推导、骨架嫁接与 Python/C++ 代码再生。

---

## 模块结构

```
algorithm_ir/
├── ir/                         # 核心 IR 数据模型
│   ├── model.py                #   Value, Op, Block, FunctionIR, ModuleIR
│   ├── dialect.py              #   22 个 IRDL 操作定义 (AlgConst, AlgBinary, ..., AlgSlot)
│   ├── types.py                #   AlgType — 统一类型包装
│   ├── type_info.py            #   TypeInfo — 类型推导辅助
│   ├── printer.py              #   render_function_ir() — IR 可读文本
│   ├── validator.py            #   validate_function_ir() — 结构验证
│   └── xdsl_bridge.py          #   xDSL ↔ dict-IR 双向桥接
│
├── frontend/                   # 编译前端：Python → IR
│   ├── ast_parser.py           #   parse_function() — AST 解析 + 源码位置
│   ├── ir_builder.py           #   compile_function_to_ir(), compile_source_to_ir()
│   └── cfg_builder.py          #   CFGBlock, link_blocks() — 控制流图构建
│
├── regeneration/               # 代码生成后端：IR → Python/C++
│   ├── codegen.py              #   emit_python_source(), emit_cpp_ops(), CppOp
│   └── artifact.py             #   AlgorithmArtifact — 嫁接产物
│
├── grafting/                   # 骨架嫁接（核心创新）
│   ├── graft_general.py        #   ★ graft_general() — 通用区域替换 + 调用注入
│   ├── rewriter.py             #   graft_skeleton() — xDSL 级嫁接 (BP 专用)
│   ├── matcher.py              #   match_skeleton() — 骨架模式匹配
│   └── skeletons.py            #   Skeleton, OverridePlan — 骨架描述
│
├── region/                     # 可重写区域
│   ├── selector.py             #   RewriteRegion, define_rewrite_region()
│   ├── contract.py             #   BoundaryContract, infer_boundary_contract()
│   └── slicer.py               #   backward_slice_by_values(), forward_slice_from_values()
│
├── runtime/                    # IR 解释执行 + 动态追踪
│   ├── interpreter.py          #   execute_ir() — 执行 FunctionIR
│   ├── tracer.py               #   RuntimeValue, RuntimeEvent
│   ├── frames.py               #   RuntimeFrame
│   └── shadow_store.py         #   ShadowStore — 可变对象追踪
│
├── analysis/                   # 静态/动态分析
│   ├── static_analysis.py      #   def_use_edges(), block_uses()
│   ├── dynamic_analysis.py     #   runtime_values_for_static()
│   └── fingerprints.py         #   fingerprint_runtime_value()
│
├── factgraph/                  # 因子图（静态 IR + 动态执行对齐）
│   ├── model.py                #   FactGraph
│   ├── builder.py              #   build_factgraph()
│   └── aligner.py              #   静态-动态对齐
│
├── projection/                 # 区域投影（可选语义标注）
│   ├── base.py                 #   Projection
│   ├── scorer.py               #   annotate_region()
│   ├── scheduling.py           #   detect_scheduling_projection()
│   └── local_interaction.py    #   detect_local_interaction_projection()
│
└── __init__.py
```

---

## 核心概念

### FunctionIR

`FunctionIR` 是一个函数的完整 SSA 中间表示，由三类对象组成：

| 对象 | 类 | 含义 |
|------|------|------|
| **Value** | `Value(id, name_hint, type_hint, def_op, use_ops, attrs)` | 数据值（变量/常量/中间结果） |
| **Op** | `Op(id, opcode, inputs, outputs, block_id, attrs)` | 操作（加法/调用/跳转/返回…） |
| **Block** | `Block(id, op_ids, preds, succs)` | 基本块（顺序执行的操作序列） |

SSA 规则：每个 Value 只被一个 Op 定义（`def_op`），可被多个 Op 使用（`use_ops`）。

关键字段：
- `values: dict[str, Value]` — 所有值
- `ops: dict[str, Op]` — 所有操作
- `blocks: dict[str, Block]` — 所有块
- `arg_values: list[str]` — 函数参数的 value ID
- `return_values: list[str]` — 返回值的 value ID
- `entry_block: str` — 入口块 ID
- `name: str` — 函数名

### 操作码 (Opcodes)

| 操作码 | 含义 | 示例 |
|--------|------|------|
| `const` | 常量 | `3.14`, `True` |
| `assign` | SSA 赋值 | `x_1 = x_0` |
| `binary` | 二元运算 | `a + b`, `x * y` |
| `unary` | 一元运算 | `-x`, `not flag` |
| `compare` | 比较 | `a < b` |
| `call` | 函数调用 | `np.linalg.solve(A, b)` |
| `get_item` / `set_item` | 索引 | `x[i]`, `x[i] = v` |
| `get_attr` / `set_attr` | 属性 | `x.real` |
| `build_list` / `build_tuple` / `build_dict` | 构造 | `[a, b]`, `(x, y)` |
| `append` / `pop` | 列表操作 | `lst.append(x)` |
| `iter_init` / `iter_next` | 迭代 | `for x in range(n)` |
| `phi` | φ 节点 | 控制流合并 |
| `branch` | 条件跳转 | `if cond: goto A else goto B` |
| `jump` | 无条件跳转 | `goto C` |
| `return` | 返回 | `return result` |
| `algslot` | 可进化槽位 | `slot_regularizer(...)` |

### RewriteRegion 与 BoundaryContract

**RewriteRegion** 定义"宿主算法中要被替换的局部区域"：
- `op_ids` — 区域内的操作 ID
- `entry_values` — 从外部流入的值
- `exit_values` — 流出到外部的值

**BoundaryContract** 定义"替换者必须满足的接口契约"：
- 输入/输出端口
- 读写集合
- 不变量

### graft_general() — 通用骨架嫁接

`graft_general()` 是骨架迁移的核心函数，执行 IR 级手术：

1. 克隆宿主 IR
2. 分析区域边界 (`find_region_boundary`)
3. 创建 `call` 操作调用供体函数 (`create_call_op`)
4. 重新绑定原区域输出的使用者 (`rebind_uses`)
5. 移除被替换的操作 (`remove_ops`)
6. 拓扑排序确保新操作位置正确 (`topological_sort_block`)
7. 返回 `GraftArtifact`（包含新 IR、新增槽位等元信息）

---

## 使用示例

### 1. Python → IR 编译

```python
from algorithm_ir.frontend import compile_source_to_ir

source = """
def detector(H, y, sigma2, constellation):
    x_hat = np.linalg.solve(H.conj().T @ H + sigma2 * np.eye(H.shape[1]), H.conj().T @ y)
    return x_hat
"""
func_ir = compile_source_to_ir(source, "detector")
print(f"函数: {func_ir.name}, 参数: {len(func_ir.arg_values)}, "
      f"操作: {len(func_ir.ops)}, 块: {len(func_ir.blocks)}")
```

也可编译函数对象：

```python
from algorithm_ir.frontend import compile_function_to_ir

def my_func(a, b):
    return a * b + a

func_ir = compile_function_to_ir(my_func)
```

### 2. 查看 IR

```python
from algorithm_ir.ir.printer import render_function_ir

print(render_function_ir(func_ir))
# FunctionIR(name=detector, entry=b_entry)
#   Block b_entry preds=[] succs=[]
#     o_0: call  in=[...] out=[v_5] attrs={'callee': 'np.linalg.solve'}
#     o_1: return in=[v_5] out=[v_6]
```

### 3. IR → Python 源码再生

```python
from algorithm_ir.regeneration.codegen import emit_python_source

source = emit_python_source(func_ir)
print(source)
# def detector(H, y, sigma2, constellation):
#     ...
#     return x_hat
```

### 4. IR → C++ 操作码

```python
from algorithm_ir.regeneration.codegen import emit_cpp_ops

cpp_ops = emit_cpp_ops(func_ir)  # list[int] — 栈式求值器操作码
```

### 5. 结构验证

```python
from algorithm_ir.ir.validator import validate_function_ir

errors = validate_function_ir(func_ir)
assert not errors, f"验证失败: {errors}"
```

### 6. 运行时解释执行

```python
from algorithm_ir.runtime.interpreter import execute_ir

result, trace, runtime_values = execute_ir(func_ir, args=[H, y, sigma2, constellation])
# result: 函数返回值
# trace: list[RuntimeEvent] — 执行轨迹
# runtime_values: dict[str, RuntimeValue] — 每个值的运行时快照
```

### 7. 区域切片与契约推导

```python
from algorithm_ir.region.selector import define_rewrite_region
from algorithm_ir.region.contract import infer_boundary_contract
from algorithm_ir.region.slicer import backward_slice_by_values

# 后向切片：找出生成某个值所需的所有操作
slice_ops = backward_slice_by_values(func_ir, target_values=["v_5"])

# 定义可重写区域
region = define_rewrite_region(func_ir, slice_ops)

# 推导边界契约
contract = infer_boundary_contract(func_ir, region)
print(f"输入: {contract.input_ports}, 输出: {contract.output_ports}")
```

### 8. 通用骨架嫁接

```python
from algorithm_ir.grafting.graft_general import graft_general

# proposal 由 PatternMatcher 生成，包含：
#   - host_algo_id, region, donor_ir, donor_algo_id 等
artifact = graft_general(host_ir, proposal)

# artifact.ir: 嫁接后的新 FunctionIR
# artifact.new_slot_ids: 新引入的槽位列表
new_source = emit_python_source(artifact.ir)
```

### 9. 因子图构建

```python
from algorithm_ir.factgraph.builder import build_factgraph

factgraph = build_factgraph(func_ir, trace, runtime_values)
# FactGraph 对齐静态 IR 和动态执行，用于后续的结构发现和 NN 引导
```

---

## 设计原则

1. **结构中立**：IR 不预设算法类别（不区分"树搜索""消息传递"），只记录可执行事实
2. **SSA 形式**：每个值只赋值一次，数据流显式表达
3. **可切片**：任意操作子集可定义为 RewriteRegion，自动推导边界
4. **可嫁接**：`graft_general()` 将供体函数注入宿主区域，通过 `call` 操作实现骨架迁移
5. **可再生**：修改后的 IR 可重新生成可执行的 Python 源码

---

## 测试

相关测试位于 `tests/` 目录：

```bash
cd research/algorithm-IR
conda activate AutoGenOld

# IR 编译、方言、回归测试
python -m pytest tests/unit/test_frontend.py tests/unit/test_dialect.py tests/unit/test_regression_p0.py -v

# 区域投影测试
python -m pytest tests/unit/test_region_projection.py -v

# 运行时 + 因子图测试
python -m pytest tests/unit/test_runtime_factgraph.py -v

# 嫁接集成测试
python -m pytest tests/integration/test_grafting_demo.py -v

# 全部 242 个测试
python -m pytest tests/ -v
```