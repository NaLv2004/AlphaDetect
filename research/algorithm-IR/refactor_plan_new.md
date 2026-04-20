# 重构计划：消除黑箱嫁接 + 消除 IR 前端语法限制

> 本文档仅针对两个核心问题：
> 1. **嫁接产生黑箱**——`graft_general()` 用 `call` op 替代被嫁接区域，donor 内部计算对 IR 不透明
> 2. **IR 前端不支持 keyword arguments**——导致 30+ workaround 函数散布在 `ir_pool.py` 和 `materialize.py` 中

---

## 问题一：嫁接产生黑箱

### 1.1 当前机制

`graft_general()` (`algorithm_ir/grafting/graft_general.py`) 执行嫁接时，将 host IR 中的一组 ops（region）替换为**一个 call op**：

```
# 嫁接前 (host IR)
op_5: order = argsort(|x_init|)    ← region 开始
op_6: H_ordered = H[:, order]      ← region 结束

# 嫁接后 (grafted IR)
op_99:  _donor_fn = const("osic_ordering_abc123")   ← 字符串常量
op_100: result = call(_donor_fn, H, x_init)          ← 黑箱 call op
```

donor 的**实际计算**（QR 分解、SNR 排序、循环等）不存在于 IR 中。它以 Python 源码字符串的形式存储在 `genome.metadata["_donor_sources"]` 字典中，在 `materialize()` 执行 `exec()` 时注入运行时命名空间。

### 1.2 问题所在

**代码位置**：

| 位置 | 代码 | 问题 |
|------|------|------|
| `graft_general.py` L425-435 | `create_call_op(new_ir, callee_name=donor_name, ...)` | 创建不透明的 call op，donor 内容不进入 host IR |
| `graft_general.py` L430-435 | `n_results = 1; call_op_id, out_vids = create_call_op(...)` | 永远只声明 1 个输出，多输出 donor 只取第一个 |
| `graft_general.py` L160-210 | `create_call_op()` 函数 | 注册 `const("donor_name")` + `call(const, args...)` 两个 op，donor 内部结构完全丢失 |
| `algorithm_engine.py` L546-553 | `donor_src = _materialize(donor_genome); donor_sources[donor_name] = donor_src` | 将 donor 物化为 Python 源码存入 metadata |
| `algorithm_engine.py` L577 | `metadata={"_donor_sources": donor_sources}` | 嫁接后的子代依赖一个 Python 源码字典来运行 |
| `materialize.py` L97-100 | `donor_sources = genome.metadata.get("_donor_sources", {}); donor_helpers = "\n".join(donor_sources.values())` | 物化时把 donor 源码 prepend 到最终源码中 |

**后果**：

1. **后续变异无法触及 donor 内部**——`mutate_ir()` 只能看到一个 call op，不能修改 donor 内的 opcode / 常量 / 控制流
2. **PatternMatcher 无法分析 donor 结构**——`StaticStructurePatternMatcher._fingerprint()` 只看到一个 call，不知道 donor 里面是循环还是矩阵分解
3. **嫁接不可组合**——如果对嫁接后的算法再次嫁接，新 donor 也是一个 call op。多次嫁接后 IR 变成一串 call 链，每个 call 指向一个源码字符串
4. **源码依赖脆弱**——如果 `_donor_sources` 中某个 key 的源码 materialize 失败（例如 donor 的 slot 缺失），运行时抛 `NameError`

### 1.3 重构目标

嫁接后的 IR 中**不应存在任何黑箱**。donor 的全部 ops 必须**内联（inline）**到 host IR 中，使得嫁接后的算法是一个单一的、完整的 IR 图。

### 1.4 重构方案：Inline Graft

#### 核心思路

将 `graft_general()` 从 "call-based graft" 改为 "inline graft"：

```
# 嫁接前 (host IR)
op_5: order = argsort(|x_init|)    ← region 开始
op_6: H_ordered = H[:, order]      ← region 结束

# 内联嫁接后 (grafted IR)
op_50: R = qr_R(H)                 ← donor ops 直接嵌入
op_51: snr_col = diag(R) ** 2
op_52: order = argsort(snr_col)
op_53: H_ordered = H[:, order]
```

#### 修改清单

**A. `graft_general.py` — 核心修改**

当前函数签名和返回类型不变，但内部逻辑从 "创建 call op" 变为 "复制 donor ops 到 host IR"：

```
graft_general(host_ir, proposal) → GraftArtifact
```

改写步骤（替换原来的 step 4-6）：

| 原步骤 | 新步骤 |
|--------|--------|
| 4. 注册 donor callable 常量 | 4. **克隆 donor IR 的所有 ops/values**，生成新的唯一 ID（避免与 host 冲突） |
| 5. 创建 call op | 5. **将克隆的 donor ops 插入 host IR 的目标 block**（在原 region 位置） |
| 6. Rebind exit values → call outputs | 6. **连接边界**：将 donor 的 arg values 绑定到 host 的 entry values（通过 port_mapping 或位置映射），将 donor 的 return values 绑定到 host 的 exit values |

新增辅助函数：

```python
def clone_ops_with_fresh_ids(
    donor_ir: FunctionIR,
    id_prefix: str,
) -> tuple[dict[str, Op], dict[str, Value], dict[str, str]]:
    """克隆 donor 的所有 ops 和 values，分配新的唯一 ID。
    
    Returns:
        new_ops: 新 op_id → Op（克隆后的）
        new_values: 新 value_id → Value（克隆后的）
        id_map: 旧 ID → 新 ID（用于 rebind）
    """
    ...

def inline_donor_ops(
    host_ir: FunctionIR,
    donor_ops: dict[str, Op],
    donor_values: dict[str, Value],
    insertion_block: str,
    insertion_index: int,
) -> None:
    """将 donor ops 和 values 插入 host IR 的指定位置。"""
    ...

def bind_donor_args_to_host_values(
    host_ir: FunctionIR,
    donor_arg_vids: list[str],   # donor 的 arg values（在 id_map 后的新 ID）
    host_entry_vids: list[str],  # host region 的 entry values
    port_mapping: dict[str, str],
    id_map: dict[str, str],
) -> None:
    """将 donor 的参数值绑定到 host 中已有的值。
    
    对每个 donor arg，找到对应的 host value，然后将 donor 内部
    对该 arg 的所有引用替换为 host value。
    """
    ...

def bind_donor_returns_to_host_exits(
    host_ir: FunctionIR,
    donor_return_vids: list[str],  # donor return op 的 inputs（在 id_map 后的新 ID）
    host_exit_vids: list[str],     # host region 的 exit values
) -> None:
    """将 host 中对 exit values 的引用重绑定到 donor 的输出值。"""
    ...
```

**B. `algorithm_engine.py::_execute_graft()` — 简化**

删除以下逻辑（约 15 行）：

```python
# 删除：不再需要 materialize donor 为源码字符串
from evolution.materialize import materialize as _materialize
donor_sources: dict[str, str] = {}
if donor_genome is not None:
    try:
        donor_src = _materialize(donor_genome)
        donor_name = proposal.donor_ir.name if proposal.donor_ir else (
            proposal.donor_algo_id or ""
        )
        donor_sources[donor_name] = donor_src
    except Exception as exc:
        logger.debug("Could not pre-materialize donor %s: %s",
                     proposal.donor_algo_id, exc)

# 删除：不再需要 _donor_sources metadata
metadata={"_donor_sources": donor_sources} if donor_sources else {},
```

嫁接后的子代不再需要 `_donor_sources`，因为 donor 的 ops 已经内联在 `structural_ir` 中。

**C. `materialize.py` — 简化**

删除 donor_sources 相关代码（约 5 行）：

```python
# 删除：不再有 donor 源码需要注入
donor_sources: dict[str, str] = genome.metadata.get("_donor_sources", {})
donor_helpers = "\n\n".join(donor_sources.values()) if donor_sources else ""
parts = [p for p in [donor_helpers, helpers, materialized] if p.strip()]
```

物化管线只需处理 skeleton IR + slot 实现，不再有外部源码依赖。

**D. `GraftArtifact` — 字段调整**

```python
@dataclass
class GraftArtifact:
    ir: FunctionIR                    # 嫁接后的完整 IR
    new_slot_ids: list[str]           # donor 引入的 AlgSlot op IDs
    replaced_op_ids: list[str]        # host 中被删除的原始 ops
    inlined_op_ids: list[str]         # donor 内联到 host 中的新 op IDs（替换原来的 call_op_id）
    id_map: dict[str, str]            # donor 旧 ID → host 中新 ID（调试用）
```

#### 需要处理的边界情况

**E1. Donor 包含多个 block（循环、分支）**

如果 donor 有 `while` 循环或 `if` 分支，它的 IR 不止一个 block。内联时需要：
- 克隆 donor 的所有 blocks 到 host IR（分配新 block ID）
- 如果 region 在 host 的单个 block 中间，需要将 host block 拆分为三段：
  - 前段：region 之前的 ops
  - 中段：donor 的 blocks（包括循环/分支结构）
  - 后段：region 之后的 ops
- 连接 CFG 边（preds/succs）

这是最复杂的部分，但 region/slicer.py 已经有 block 拆分的先例可以参考。

**E2. Donor 自身包含 AlgSlot ops**

如果 donor IR 中有 slot ops，内联后这些 slot 会出现在 host IR 中。当前 `graft_general()` 的 step 10 已经处理了这个情况（检测新 slot IDs），只需保留这一步。

**E3. 多输出 donor**

当前 call-based graft 硬编码 `n_results = 1`（`graft_general.py` L430）。inline graft 天然支持多输出：donor 的 return op 可能返回一个 tuple，这些值通过 `bind_donor_returns_to_host_exits()` 逐个绑定到 host 的多个 exit values。

**E4. Donor 的 arg_values 与 host entry values 不一一对应**

当前 call-based graft 通过 `port_mapping` 或位置映射来解决。inline graft 的解决方式相同：
- 如果有 `port_mapping`，按映射表绑定
- 否则按位置绑定（donor arg[0] → host entry[0], ...）
- 如果 donor 参数多于 host entry values，多出的参数从 host 的函数级 arg_values 中取（当前代码已经有这个逻辑）

#### 保留 `create_call_op()` 和 `rebind_uses()` 等辅助函数

这些函数本身设计正确，只是 `graft_general()` 不再使用 `create_call_op()` 创建黑箱 call。`rebind_uses()` 在 inline graft 中仍然需要（用于将 donor arg 引用替换为 host value 引用）。

---

## 问题二：IR 前端不支持 keyword arguments

### 2.1 当前机制

`ir_builder.py` 中的 `_compile_expr()` 处理 `ast.Call` 时：

```python
# ir_builder.py L449-451
if isinstance(expr, ast.Call):
    func_value = self._compile_expr(expr.func)
    args = [self._compile_expr(arg) for arg in expr.args]
    return self._emit_call(func_value, args, expr)
```

`expr.keywords`（keyword arguments）被**完全忽略**。并且 `SUPPORTED_AST`（L57-79）不包含 `ast.keyword`，所以 `_assert_supported()` 会对任何含有 keyword argument 的代码抛出 `NotImplementedError`。

### 2.2 问题所在

这导致以下合法 Python 代码无法编译为 IR：

```python
np.zeros(n, dtype=complex)           # → 必须用 _czeros(n)
np.sum(x, axis=0)                    # → 必须用 _sum0(x)
np.max(x, axis=1, keepdims=True)     # → 必须用 _max1_keepdims(x)
np.linalg.qr(H)[0]                  # → 必须用 _qr_Q(H)
np.delete(H, idx, axis=1)           # → 必须用 _delete_col(H, idx)
TreeNode(level=l, symbols=s, cost=c) # → 必须用 _make_tree_node(l, s, c)
```

**代码位置**：

| 位置 | 问题 |
|------|------|
| `ir_builder.py` L57-79 | `SUPPORTED_AST` 不包含 `ast.keyword` |
| `ir_builder.py` L449-451 | `_compile_expr(ast.Call)` 忽略 `expr.keywords` |
| `ir_pool.py` L57-165 | `_template_globals()` 定义 30+ 个 workaround 函数 |
| `ir_pool.py` L170 返回 dict | 30+ 个 workaround 函数必须导出到运行时命名空间 |
| `materialize.py::_default_exec_namespace()` | 必须导入同样的 workaround 函数，否则运行时 NameError |
| 所有模板字符串 (`ir_pool.py` L250+) | 所有检测器模板中充满 `_czeros()`, `_sum0()`, `_qr_R()` 等非标准调用 |
| `SLOT_DEFAULTS` (`ir_pool.py` L190+) | 所有 slot 默认实现中也使用这些 workaround 函数 |

**影响**：

1. 模板代码**可读性极差**——自然的 `np.zeros(n, dtype=complex)` 被替换为不直观的 `_czeros(n)`
2. **维护负担**——每次需要新的 keyword arg 组合（如 `np.argsort(x, kind='stable')`），都要在 `_template_globals()` 中新增一个 wrapper
3. **同步风险**——`_template_globals()` 和 `_default_exec_namespace()` 必须保持一致，否则编译能通过但运行时报错
4. 某些复杂 slot 实现（`bp_sweep`, `site_update`, `amp_iterate`）因 3D 数组索引等原因**完全无法编译为 IR**（`test_ir_evolution.py::test_compile_bp_sweep` 断言 `ir is None`），只能以源码字符串形式存储在 `source_variants` 中

### 2.3 重构方案：在 IR 中支持 keyword arguments

#### A. IR Model 扩展

在 `Op` 的 `attrs` 中添加 keyword 信息：

```python
# 对于 opcode="call" 的 Op：
# 现有：attrs = {"n_args": 3}
# 新增：attrs = {"n_args": 3, "kwarg_names": ["dtype", "axis"]}
#
# inputs 布局：[callee_vid, pos_arg_0, pos_arg_1, ..., kw_val_0, kw_val_1, ...]
# kwarg_names[i] 对应 inputs[1 + n_pos_args + i]
```

这不需要修改 `Op` dataclass 本身（attrs 已经是 `dict[str, Any]`），只需约定 call op 的 attrs 新增 `kwarg_names` 字段。

#### B. `ir_builder.py` — 编译 keyword arguments

**修改 1**：将 `ast.keyword` 加入 `SUPPORTED_AST`：

```python
# ir_builder.py L57
SUPPORTED_AST = (
    ...,
    ast.keyword,   # ← 新增
)
```

**修改 2**：修改 `_compile_expr(ast.Call)` 处理 keyword args：

```python
# ir_builder.py L449-451
if isinstance(expr, ast.Call):
    func_value = self._compile_expr(expr.func)
    pos_args = [self._compile_expr(arg) for arg in expr.args]
    # --- 新增：编译 keyword arguments ---
    kw_names: list[str] = []
    kw_values: list[str] = []
    for kw in expr.keywords:
        if kw.arg is None:
            # **kwargs 展开 — 暂不支持
            raise NotImplementedError("**kwargs expansion not supported")
        kw_names.append(kw.arg)
        kw_values.append(self._compile_expr(kw.value))
    return self._emit_call(func_value, pos_args, expr,
                           kwarg_names=kw_names, kwarg_values=kw_values)
```

**修改 3**：修改 `_emit_call()` 传递 keyword 信息到 Op attrs：

```python
def _emit_call(self, func_vid, arg_vids, node,
               kwarg_names=None, kwarg_values=None):
    ...
    all_inputs = [func_vid] + arg_vids + (kwarg_values or [])
    attrs = {
        "n_args": len(arg_vids),
    }
    if kwarg_names:
        attrs["kwarg_names"] = kwarg_names
    ...
```

#### C. `codegen.py` — 生成带 keyword arguments 的 Python 代码

**修改 `_emit_op()` 中 `opcode == "call"` 分支**：

```python
if op.opcode == "call":
    n_pos_args = op.attrs.get("n_args", 0)
    callable_vid = op.inputs[0]
    pos_args = [ctx.expr(v) for v in op.inputs[1:1 + n_pos_args]]
    
    # --- 新增：处理 keyword arguments ---
    kwarg_names = op.attrs.get("kwarg_names", [])
    kw_start = 1 + n_pos_args
    kw_args = [ctx.expr(v) for v in op.inputs[kw_start:kw_start + len(kwarg_names)]]
    
    kw_parts = [f"{name}={val}" for name, val in zip(kwarg_names, kw_args)]
    all_arg_parts = pos_args + kw_parts
    
    # ... 后续生成 func_name(all_arg_parts) 不变 ...
    expr = f"{func_name}({', '.join(all_arg_parts)})"
```

#### D. `ir/dialect.py` — xDSL AlgCall op 扩展

如果需要在 xDSL 层面也保留 keyword 信息，在 `AlgCall` op 中添加 `kwarg_names` 属性：

```python
@irdl_op_definition
class AlgCall(IRDLOperation):
    name = "alg.call"
    ...
    kwarg_names = opt_attr(ArrayAttr)  # 新增（可选）
```

但这不是必须的——keyword 信息也可以仅存在于 FunctionIR dict 层面的 `Op.attrs` 中。

#### E. 清理 workaround 函数

支持 keyword args 后，`_template_globals()` 中的以下函数可以**全部删除**：

| Workaround 函数 | 替换为 |
|-----------------|--------|
| `_czeros(n)` | `np.zeros(n, dtype=complex)` |
| `_czeros2(nr, nc)` | `np.zeros((nr, nc), dtype=complex)` |
| `_cones(n)` | `np.ones(n, dtype=complex)` |
| `_cempty(n)` | `np.empty(n, dtype=complex)` |
| `_cfull(n, val)` | `np.full(n, val, dtype=float)` |
| `_carray(lst)` | `np.array(lst, dtype=complex)` |
| `_sum0(x)` | `np.sum(x, axis=0)` |
| `_sum1(x)` | `np.sum(x, axis=1)` |
| `_max1_keepdims(x)` | `np.max(x, axis=1, keepdims=True)` |
| `_sum1_keepdims(x)` | `np.sum(x, axis=1, keepdims=True)` |
| `_qr_Q(H)` | `np.linalg.qr(H)[0]` |
| `_qr_R(H)` | `np.linalg.qr(H)[1]` |
| `_delete_col(H, idx)` | `np.delete(H, idx, axis=1)` |
| `_make_tree_node(l, s, c)` | `TreeNode(level=l, symbols=s, cost=c)` |

以下函数因涉及**多维切片语法**（`H[:, j]`、`mat[i, :]`）而与 keyword args 无关，需要单独评估是否能被 IR builder 支持（取决于 `ast.Subscript` + `ast.Tuple` 切片的编译）：

| 函数 | 等效 Python | 依赖 |
|------|------------|------|
| `_col(H, j)` | `H[:, j]` | 多维切片 |
| `_row(mat, i)` | `mat[i, :]` | 多维切片 |
| `_row_set(mat, i, v)` | `mat[i, :] = v` | 多维切片赋值 |
| `_argmax_row(mat, i)` | `int(np.argmax(mat[i, :]))` | 多维切片 |
| `_row_normalize(mat, i)` | 多行操作 | 多维切片 + 就地修改 |
| `_reverse_syms(symbols, Nt)` | 循环操作 | 可直接编译 |

这些需要 IR builder 同时支持 `ast.Slice` 编译（`H[:, j]` 在 AST 中是 `Subscript(value=H, slice=Tuple(elts=[Slice(), Constant(j)]))`）。如果暂不支持多维切片，这些 workaround 函数可以保留，但数量从 30+ 减少到 ~6 个。

#### F. 重写所有模板字符串

`ir_pool.py` 中所有检测器模板需要将 workaround 调用替换为标准 Python：

```python
# 修改前
"x_hat = _czeros(Nt)"
"G_inv = np.linalg.inv(G)"
"snr = _czeros(Nt)"
"H_new = _delete_col(H, idx)"

# 修改后
"x_hat = np.zeros(Nt, dtype=complex)"
"G_inv = np.linalg.inv(G)"
"snr = np.zeros(Nt, dtype=complex)"
"H_new = np.delete(H, idx, axis=1)"
```

`SLOT_DEFAULTS` 中的 slot 默认实现也需要同步修改。

#### G. 重写后 `_template_globals()` 的精简

修改后 `_template_globals()` 只需要保留：

```python
def _template_globals() -> dict[str, Any]:
    """Namespace for compiling detector templates to IR."""
    import math
    return {
        "np": np,
        "math": math,
        # 仅保留无法通过 keyword arg 支持解决的 workaround：
        "_col": lambda H, j: H[:, j],       # 多维切片（如果仍不支持）
        "_row": lambda mat, i: mat[i, :],    # 多维切片
        # ... 最多 ~6 个
    }
```

---

## 实施顺序

建议按以下顺序实施，每步可独立验证：

### Phase 1：IR 前端支持 keyword arguments

1. 在 `SUPPORTED_AST` 中添加 `ast.keyword`
2. 修改 `_compile_expr(ast.Call)` 编译 `expr.keywords`
3. 修改 `_emit_call()` 将 kwarg 信息写入 `Op.attrs`
4. 修改 `codegen.py::_emit_op()` 的 call 分支，生成 `f(a, b, key=val)` 格式
5. **验证**：编写测试，确认 `np.zeros(n, dtype=complex)` 能正确编译并 roundtrip（source → IR → source）
6. 将 `_template_globals()` 中的 workaround 函数逐个替换为标准调用
7. 更新所有模板字符串和 `SLOT_DEFAULTS`
8. **验证**：运行全部 242 个测试

### Phase 2：Inline Graft

1. 实现 `clone_ops_with_fresh_ids()` — 克隆 donor IR 并分配新 ID
2. 实现 `inline_donor_ops()` — 将 ops 插入 host block
3. 实现 `bind_donor_args_to_host_values()` — 参数绑定
4. 实现 `bind_donor_returns_to_host_exits()` — 输出绑定
5. 处理多 block donor（block 拆分 + CFG 连接）
6. 重写 `graft_general()` 调用上述函数，替换 `create_call_op()` 路径
7. 修改 `GraftArtifact` — `call_op_id` → `inlined_op_ids` + `id_map`
8. 从 `algorithm_engine.py::_execute_graft()` 中删除 `_donor_sources` 逻辑
9. 从 `materialize.py` 中删除 `donor_helpers` 注入逻辑
10. **验证**：运行全部测试 + E2E 进化运行，确认嫁接后的算法可被正确物化和评估
