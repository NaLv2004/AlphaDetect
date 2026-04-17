# Algorithm IR 详细架构说明

这份文档解释 `research/algorithm-IR/algorithm_ir/` 的整体架构。

它是写给“不懂编译器、但想看懂这个系统”的读者的，所以我会尽量不用编译器术语，或者在第一次出现时立刻解释它是什么意思。

这套系统想解决的核心问题很简单：

1. 把一个 Python 算法变成一种统一、可执行、可切片、可重写的中间表示。
2. 在这个表示里精确地切出一段局部计算。
3. 理清这段局部计算的输入、输出和可读写边界。
4. 把另一种算法 skeleton 嫁接进去。
5. 生成新的 IR，并再次执行它。

请先记住一句话：

**这个 IR 本身不负责说“这就是搜索树”或“这就是 BP”。**

它只负责保存“程序到底做了哪些可执行的低层事实”。

至于这些事实应该被解释成“树”“候选池”“局部消息传播”还是别的结构，那是上层分析或用户解释的事情，不是底层 IR 先验写死的事情。

---

## 1. 为什么不能直接拿 Python 源码做嫁接

如果直接改 Python 源码，会遇到几个问题：

- 源码层面太“粗”
- 很难准确说清楚“到底替换哪一段”
- 很难判断这段代码依赖哪些变量、写了哪些状态
- 很难在替换后验证程序仍然结构正确
- 很难把 host 算法和 donor 算法放在同一种统一表示里

所以我们需要中间层。

这个中间层不是为了“像编译器一样炫技”，而是为了得到 4 个关键能力：

- `execution`：它必须还能跑
- `slicing`：它必须能切出局部区域
- `rewriting`：它必须能删掉旧操作、插入新操作
- `regeneration`：它必须能把改写结果重新整理成新的工件

---

## 2. 一句话理解整套系统

你可以把整个系统想成“可编辑电路”。

- `Value`：一根线，承载某个中间结果
- `Op`：一个元件，吃几根输入线，吐出几根输出线
- `Block`：一段没有岔路的连续操作
- `FunctionIR`：整个算法的“电路图”
- `RuntimeEvent`：这张电路图某次真正运行时的一次元件触发
- `RewriteRegion`：你准备切掉并替换的一段子电路
- `BoundaryContract`：这段子电路边界的“插头形状”

所以整条流程就是：

1. 把 Python 函数翻译成电路图
2. 跑一次，看看哪些元件真的被触发了
3. 切出一个局部区域
4. 弄清楚这个区域的边界
5. 插入 donor skeleton
6. 得到新的电路图

---

## 3. 包结构总览

目录结构如下：

```text
algorithm_ir/
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
  analysis/
    static_analysis.py
    dynamic_analysis.py
    fingerprints.py
  region/
    selector.py
    slicer.py
    contract.py
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
    artifact.py
    codegen.py
```

每一层的职责可以用一句话概括：

- `frontend/`：把受限 Python 变成 IR
- `ir/`：定义 IR 结构，并提供打印和校验
- `runtime/`：执行 IR，记录运行时事实
- `factgraph/`：把静态 IR 和动态事实合并到同一个事实图里
- `region/`：定义要切掉的局部区域，并推断边界
- `projection/`：对 region 做可选解释，不是必须层
- `grafting/`：把 donor skeleton 插进去
- `regeneration/`：打包改写后的工件

---

## 4. 你必须先理解的 4 个核心对象

这 4 个对象是整个系统最重要的“骨架”。

### 4.1 `Value`

定义在 [model.py](./ir/model.py)。

`Value` 不是运行时对象，它是静态的“结果槽位”。

你可以把它理解成：

- 一个变量的某个版本
- 一根中间导线
- 一个操作的输出占位符

例如：

```python
score = candidate["metric"] + costs[candidate["depth"]]
```

这一行在 IR 里不会只有一个 `score`。
它会先产生很多中间 `Value`：

- `"metric"` 这个常量键
- `candidate["metric"]` 的结果
- `"depth"` 这个常量键
- `candidate["depth"]` 的结果
- `costs[...]` 的结果
- 两个数相加后的结果
- 最后 assign 给 `score`

所以 `Value` 更像“中间计算图中的节点”，不是源代码里肉眼看到的变量名本身。

关键字段：

- `id`：例如 `v_73`
- `name_hint`：例如 `score_0`
- `type_hint`：例如 `float`、`list`、`dict`
- `def_op`：谁定义了它
- `use_ops`：谁使用了它
- `attrs`：其他元信息，比如原始变量名、版本号

### 4.2 `Op`

`Op` 是一个操作。

你可以理解成：

- 一次取字典项
- 一次加法
- 一次比较
- 一次函数调用
- 一次赋值

例如刚才那行 `score = ...`，在 IR 里会被拆成多条 `Op`：

- `const`
- `get_item`
- `get_item`
- `get_item`
- `binary(Add)`
- `assign`

这样做的好处是：

- 每一步依赖关系都显式了
- 可以精确切片
- 可以只替换中间一部分

关键字段：

- `opcode`
- `inputs`
- `outputs`
- `block_id`
- `attrs`

### 4.3 `Block`

`Block` 是一段“连续执行、没有中途分叉”的操作序列。

为什么要有它？

因为真实程序有：

- `if / else`
- `while`
- `for`
- `return`

这些都会让控制流分叉、汇合、回跳。

如果没有 `Block`，整个程序结构会非常混乱。

所以你可以把 `Block` 理解成：

- 程序里一段直线型的路
- 走到尽头后，再决定跳去哪里

### 4.4 `FunctionIR`

这就是整个函数的完整 IR 表示。

它包含：

- 参数
- 返回值
- 所有 `Value`
- 所有 `Op`
- 所有 `Block`
- 入口块

它相当于“这个 Python 函数被翻译后的整张电路图”。

---

## 5. frontend：Python 是怎么变成 IR 的

这一层在 [frontend/](./frontend/)。

### 5.1 `ast_parser.py`

这一步做的事情很简单：

- 读取 Python 函数源码
- 用 Python 自带的 `ast` 模块解析
- 找到函数定义
- 记录每个语法节点的源码位置

这里的“AST”你可以先粗略理解为：

**Python 源码的语法树。**

例如：

```python
score = candidate["metric"] + costs[candidate["depth"]]
```

解析之后，系统知道这是：

- 一个赋值语句
- 右边是一个加法
- 加法左边是一次取字典项
- 加法右边又是一次嵌套索引

但 AST 还不是我们最终想要的 IR，因为 AST 太贴近源代码语法，不适合执行和重写。

### 5.2 `ir_builder.py`

这是最关键的 lowering（降级/翻译）阶段。

它把 Python AST 变成：

- `Value`
- `Op`
- `Block`
- 控制流边

它目前支持的受限 Python 子集包括：

- 赋值
- 增量赋值
- `if / else`
- `while`
- `for`
- 函数调用
- 列表/字典字面量
- 属性访问
- 下标访问
- 一元/二元运算
- 比较

### 5.3 为什么会出现 `score_0`、`frontier_2` 这种名字

这是因为系统使用了轻量级的“版本化变量”思想。

直觉上，你可以把它理解成：

- 同一个源代码变量，在程序不同位置其实是不同版本
- 如果不区分版本，后续切片和重写会混乱

例如：

```python
i = 0
i = i + 1
```

在 IR 里不会都叫一个 `i`，而会变成类似：

- `i_0`
- `i_1`

这使得“哪个值是哪个步骤产生的”变得完全明确。

### 5.4 什么是 `phi`

这是很多非编译器背景读者第一次会卡住的概念。

你可以这样理解：

如果一个变量可能来自多条控制路径，那么在路径汇合的地方，系统需要一个“合并器”。

例如：

```python
if cond:
    total = 1
else:
    total = 2
return total
```

在 `return total` 之前，`total` 到底是哪一个？

- 来自 then 分支？
- 来自 else 分支？

`phi` 就是这个“路径合并器”。

同理，在循环头部，变量也需要把“初始值”和“上一轮循环带回来的值”合并起来。

所以你不必把 `phi` 想得太神秘。

它就是：

**“如果值可能来自不同路径，那我在这里把这些来源统一起来。”**

---

## 6. 一个最小例子：`simple_branch_loop`

测试用例在 [algorithms.py](../tests/examples/algorithms.py)：

```python
def simple_branch_loop(x: int) -> int:
    total = 0
    i = 0
    while i < x:
        if i < 2:
            total = total + i
        else:
            total = total + 2
        i = i + 1
    return total
```

为什么要先编译它？

因为这是最小但又真实的控制流样本。它同时包含：

- 变量初始化
- 循环
- 分支
- 分支汇合
- 循环回边

如果这个例子都编不好，后面的 stack decoder 一定会出问题。

这个函数编译后会生成：

- 入口块
- while 判断块
- while body 块
- if true 块
- if false 块
- if merge 块
- 循环回跳
- return

也就是说，它已经不再是“源码长什么样”，而是“程序真正如何流动”。

---

## 7. 真正的 host 示例：`stack_decoder_host`

host 定义在 [algorithms.py](../tests/examples/algorithms.py)：

```python
def stack_decoder_host(costs: list[float], max_steps: int) -> float:
    frontier = [{"path": [0], "metric": 0.0, "depth": 0}]
    best_metric = 9999.0
    best_path = []
    steps = 0
    while steps < max_steps:
        if len(frontier) == 0:
            return best_metric
        best_index = 0
        scan = 1
        while scan < len(frontier):
            if frontier[scan]["metric"] < frontier[best_index]["metric"]:
                best_index = scan
            scan = scan + 1
        candidate = frontier.pop(best_index)
        score = candidate["metric"] + costs[candidate["depth"]]
        if score < best_metric:
            best_metric = score
            best_path = candidate["path"]
        next_depth = candidate["depth"] + 1
        if next_depth < len(costs):
            left = {"path": candidate["path"] + [0], "metric": score, "depth": next_depth}
            right = {"path": candidate["path"] + [1], "metric": score + 0.25, "depth": next_depth}
            frontier.append(left)
            frontier.append(right)
        steps = steps + 1
    return best_metric
```

这个例子非常重要，因为它已经具备了你真正关心的特征：

- 有候选池 `frontier`
- 有局部评分 `score`
- 有状态更新
- 有扩展新候选
- 有循环调度

而且它没有任何“搜索树”高层语义写进 IR。

这正符合你的要求：

**IR 不预设它一定是树，只保留它操作了哪些对象、做了哪些比较、哪些更新、哪些控制跳转。**

---

## 8. IR 到底长什么样

为了避免抽象，我直接给你一个真实风格的简化片段。

下面是 host 里局部 score 计算在 IR 里的样子：

```text
op_64: get_attr in=[frontier_2] out=[... ] attrs={'attr': 'pop'}
op_65: call in=[..., best_index_1] out=[candidate_0]
op_67: const in=[] out=[...] attrs={'literal': 'metric'}
op_68: get_item in=[candidate_0, ...] out=[...]
op_69: const in=[] out=[...] attrs={'literal': 'depth'}
op_70: get_item in=[candidate_0, ...] out=[...]
op_71: get_item in=[costs_2, ...] out=[...]
op_72: binary in=[..., ...] out=[... ] attrs={'operator': 'Add'}
op_73: assign in=[...] out=[score_0] attrs={'target': 'score'}
```

你需要注意的地方有两个：

第一，没有任何“stack decoder 原语”。

- 没有 `SearchTree`
- 没有 `MetricNode`
- 没有 `ExpandTree`

第二，也没有任何“BP 原语”。

IR 里只有非常基础的事实：

- 从容器里 pop 一个对象
- 从对象里取 `metric`
- 从对象里取 `depth`
- 根据 `depth` 去取 `costs`
- 做加法
- 赋给 `score`

这就是“结构中立”的真正含义。

---

## 9. 运行时层：为什么还要自己执行 IR

很多人会问：

“既然原始 Python 函数本来就能跑，为什么还要单独写一个 IR 解释器？”

原因是：我们不只是想拿到最后结果，而是想拿到**对齐到 IR 的、可操作的运行时事实**。

我们需要知道：

- 哪个静态 op 真正执行了几次
- 哪个静态 value 在运行时产生了哪些实例
- 哪些对象被写入了字段
- 哪些容器被 append/pop 了
- 哪些路径在这次执行里真的走到了

如果只是“运行原始 Python”，这些信息很难统一映射回 IR。

### 9.1 `RuntimeValue`

静态 `Value` 是“线槽位”。

运行时的 `RuntimeValue` 是“某次具体跑起来时，这根线上真的流过的那个值”。

所以：

- `Value` 是模板
- `RuntimeValue` 是实例

### 9.2 `RuntimeEvent`

每执行一条 `Op`，就会产生一个 `RuntimeEvent`。

它记录：

- 哪个静态 op 被触发了
- 输入 runtime values 是什么
- 输出 runtime values 是什么
- 当前控制上下文是什么

这相当于“动态执行轨迹”。

### 9.3 `ShadowStore`

这是运行时层最容易被忽略、但实际上非常关键的一部分。

因为 Python 里大量对象是可变的：

- list 会变
- dict 会变
- object 字段会变

如果没有额外记录，仅靠 def-use 是不够的。

比如：

```python
frontier.append(left)
candidate = frontier.pop(best_index)
```

你想知道的不只是“frontier 被用了”，而是：

- frontier 这个容器被写了
- 它什么时候多了成员
- 它什么时候少了成员

`ShadowStore` 就是用来记录这些“可变对象的影子历史”的。

没有它，你很难稳健地做 region rewrite。

---

## 10. FactGraph：为什么还要多一层事实图

`FactGraph` 定义在 [factgraph/model.py](./factgraph/model.py)。

它不是高层算法图，而是一个统一容器，把这些东西放在一起：

- 静态函数
- 静态 op
- 静态 value
- 运行时 event
- 运行时 value
- 对齐边

它的好处是：

后续分析时不必同时到处查静态 IR 和动态 trace。

你可以从一个统一对象里拿到：

- `def_use`
- `cfg`
- `event_input`
- `event_output`
- `temporal`
- `instantiates_op`
- `instantiates_value`

所以 `FactGraph` 更像一个“统一事实数据库”。

---

## 11. RewriteRegion：真正被替换的对象

这是整个系统当前最核心的设计。

`RewriteRegion` 定义在 [selector.py](./region/selector.py)。

它表示：

**“我要切出来并改写的那段局部计算。”**

这不是一个高层结构标签，而是一个由低层 IR 事实定义出来的可操作区域。

它包含：

- `op_ids`：区域里有哪些操作
- `block_ids`：涉及哪些块
- `entry_values`：区域外流入区域的值
- `exit_values`：区域内产生、但被区域外继续使用的值
- `read_set`：区域读了哪些槽位
- `write_set`：区域写了哪些槽位
- `state_carriers`：区域中承载状态的值
- `schedule_anchors`：它在控制流里大概挂在哪些点

你可以把它理解成：

- “待替换的精确切片”
- 而不是“某种解释视角”

### 11.1 Region 怎么选

当前实现支持 4 种方式：

- 直接给 `op_ids`
- 按 `source_span`（源码位置）
- 从 `exit_values` 做 backward slice
- 从 `state_carriers` 做 forward slice

这很重要，因为真实研究里，“我知道从哪里切”这个信息来源不一定一样。

---

## 12. BoundaryContract：region 的插头

`BoundaryContract` 定义在 [contract.py](./region/contract.py)。

如果说 `RewriteRegion` 是“我要切掉哪一段”，那么 `BoundaryContract` 就是：

**“这段切口的边界到底长什么样。”**

它回答的问题是：

- 外面哪些值喂进来？
- 里面哪些结果要吐出去？
- 它会读哪些槽位？
- 它会写哪些槽位？
- graft 后要接回哪里？
- 有哪些必须保留的不变量？

在当前 MVP 里，它会推断：

- `input_ports`
- `output_ports`
- `readable_slots`
- `writable_slots`
- `reconnect_points`
- `invariants`

例如 stack decoder 的 score region，最重要的一个不变量是：

- 最后必须还能给出一个可接回 `score` 的标量输出

这就是 donor 能否匹配的关键。

---

## 13. Projection：为什么它是“可选层”

这个地方最容易产生误解，我专门说明一下。

在当前实现里，`Projection` 不是主角。

它只是：

- 对 region 的一种解释
- 一个可选注释层
- donor 匹配时的辅助信息

它不是：

- IR 本体
- 真正被改写的对象
- 唯一合法的结构解释

为什么这样设计？

因为你的要求很明确：

**底层 IR 不应该先验地把 host 算法固定成“搜索树”或者“因子图”。**

所以真正核心的是：

- `RewriteRegion`
- `BoundaryContract`

而 `Projection` 只是一个可选说明：

- “这个 region 看起来像局部交互”
- “这个 region 看起来像某种调度片段”

当前最小实现里只有两个 projection family：

- `scheduling`
- `local_interaction`

这样做的目的不是“发现真实结构”，而是给未来的 donor 匹配、NN 排名留下接口。

---

## 14. donor skeleton 是怎么表示的

`Skeleton` 定义在 [skeletons.py](./grafting/skeletons.py)。

它并不是一个高层复杂语言，而是一个相对直接的 donor 模板对象。

它记录：

- donor 名字
- donor 所需的 contract
- 变换规则
- lowering 模板
- 可选 projection hints
- donor 原始 callable
- donor 自己的 IR

当前实现的 donor 是：

```python
def bp_summary_update(frontier: list[dict], costs: list[float], damping: float) -> float:
    idx = 0
    summary = 0.0
    while idx < len(frontier):
        item = frontier[idx]
        summary = summary + item["metric"] + costs[item["depth"]]
        idx = idx + 1
    return summary * damping
```

注意，这不是“完整 BP”，而是一个最小 BP-like donor。

它的作用不是证明“我们已经实现了真正的 BP detector”，而是证明：

**host 和 donor 可以先统一落在同一种 IR 里，然后发生一次真实的局部 skeleton 嫁接。**

---

## 15. grafting：真正发生替换的地方

这一层在 [rewriter.py](./grafting/rewriter.py)。

这是真正“动刀”的地方。

流程如下：

1. 匹配 donor 是否满足 region 的 contract
2. 选出要覆写的目标输出，目前优先选择 `score`
3. 计算生成这个 `score` 的 backward slice
4. 判断其中哪些 op 可以删，哪些必须保留
5. 插入 donor 调用和重连逻辑
6. 删掉被替换的旧 op
7. 重算 def-use
8. 验证新 IR 仍然结构正确

当前 BP graft 的实际替换逻辑是：

原来 host 做的是：

```python
candidate = frontier.pop(best_index)
score = candidate["metric"] + costs[candidate["depth"]]
```

graft 后等价于：

```python
summary = bp_summary_update(frontier, costs, damping)
candidate_metric = candidate["metric"]
score = candidate_metric + summary
```

也就是说，原本“candidate metric + 单点 cost”的局部评分，被改成了“candidate metric + donor 生成的 summary”。

---

## 16. 改写前后，哪些东西变了，哪些没变

### 16.1 变了的部分

- `score` 的计算路径被覆写
- 新增了 donor callable 常量
- 新增了 damping 常量
- 新增了 donor call
- 新增了 donor 输出和 `candidate["metric"]` 的相加

### 16.2 没变的部分

- host 整个函数仍然是同一种 IR
- donor 也仍然是同一种 IR
- 改写后的函数仍然能被解释器执行
- 改写后的函数仍然能继续被分析、继续被切片

这正是这套架构最重要的成功标准。

---

## 17. README 对应的真实 demo

除了单元测试外，我还提供了一个专门的演示脚本：

`research/algorithm-IR/demo_outputs.py`

它会打印：

- `simple_branch_loop` 的 IR 结构
- `stack_decoder_host` 的 IR 结构
- `bp_summary_update` 的 IR 结构
- host 的运行结果、trace 和 factgraph 规模
- `RewriteRegion`
- `BoundaryContract`
- 可选 `Projection`
- graft 前后的 IR 片段
- graft 前后的执行结果
- `OverridePlan`

运行方式：

```powershell
conda run -n AutoGenOld python demo_outputs.py
```

工作目录：

```text
research/algorithm-IR/
```

---

## 18. 每个测试为什么这么设计

测试目录在 [tests/](../tests/)。

### 18.1 `test_frontend.py`

目的：验证“能不能正确编译成 IR”。

#### `test_compile_simple_branch_loop`

检查点：

- 校验器无错误
- IR 里有 `branch`
- IR 里有 `return`
- block 数量足够，说明控制流真的展开了

为什么这样设计：

- 这是最小但真实的控制流例子
- 能同时测到 while、if、phi、return

#### `test_compile_stack_decoder_host`

检查点：

- 校验器无错误
- 渲染出的 IR 中出现 `build_dict`
- 出现 `call`
- 出现 `get_item`

为什么这样设计：

- 这是第一个“接近目标问题”的 host
- 它覆盖 list/dict、循环、比较、调用、可变状态

### 18.2 `test_runtime_factgraph.py`

目的：验证“IR 不只是长得像程序，而是真的能执行，并且能产生动态事实”。

#### `test_execute_simple_branch_loop`

检查点：

- IR 执行结果与原始 Python 完全一致
- trace 非空
- runtime values 非空

为什么这样设计：

- 如果编译正确但执行不一致，整个系统没有意义

#### `test_execute_stack_decoder_and_build_factgraph`

检查点：

- host IR 的执行结果和原始 Python 一致
- factgraph 中存在 `def_use`
- 存在 `event_input`
- 存在 `temporal`
- 存在 `instantiates_op`

为什么这样设计：

- 它验证了从“编译 -> 运行 -> 事实图”这一整条链

### 18.3 `test_region_projection.py`

目的：验证“能不能切出局部计算区域，并描述它的边界”。

#### `test_define_region_and_infer_contract`

流程：

1. 编译 host
2. 执行 host
3. 找到最后一个 `score`
4. 从这个 `score` 反向切片得到 region
5. 推断 boundary contract

检查点：

- region 非空
- contract 输出里有 `score`
- contract 至少有输入端口
- 不变量里存在 `scalar_outputs`

为什么这样设计：

- 这是第一次证明系统真的具备“可编辑局部计算”的能力

#### `test_optional_projection_annotation`

检查点：

- annotation 至少能返回 `local_interaction`

为什么这样设计：

- 只是验证 projection 层存在且可用
- 同时保持它不是 rewrite 的中心对象

### 18.4 `test_grafting_demo.py`

目的：验证完整闭环。

流程：

1. 编译 host
2. 编译 donor
3. 执行 host
4. 用源码范围切出 host region
5. 推断 contract
6. 构造 donor skeleton
7. graft
8. 校验新 IR
9. 执行新 IR

检查点：

- 改写后的 IR 仍然合法
- 改写后的 IR 仍然能执行
- provenance 中保留 donor 信息
- 新 IR 中确实存在被标记为 `grafted` 的操作

为什么这样设计：

- 这是最终的“系统真能做局部 skeleton transplantation”的证据

---

## 19. 当前实现的局限

必须诚实说明，当前实现是 MVP，不是终态。

局限包括：

- 只支持受限 Python 子集
- 目前还是单函数级别
- 没有完整强类型系统
- 没有自动 region 排名
- 没有更丰富的 donor 搜索
- 当前 donor graft 逻辑还偏示范性质
- 代码再生目前是“可读的 IR 文本”，不是完整高质量 Python 反编译

但这些局限是可以接受的，因为当前阶段的目标是先证明：

**region rewrite 闭环是通的。**

---

## 20. 推荐阅读顺序

如果你想最快看懂代码，建议按这个顺序：

1. [algorithms.py](../tests/examples/algorithms.py)
2. [ir_builder.py](./frontend/ir_builder.py)
3. [interpreter.py](./runtime/interpreter.py)
4. [selector.py](./region/selector.py)
5. [contract.py](./region/contract.py)
6. [rewriter.py](./grafting/rewriter.py)
7. [test_grafting_demo.py](../tests/integration/test_grafting_demo.py)
8. `demo_outputs.py`

这个顺序基本就是系统真实运行的顺序。

---

## 21. 最后用一句话总结

这套 `algorithm_ir` 系统实现的是：

**把 host 算法和 donor skeleton 统一表示成一种结构中立、可执行、可切片、可重写的 IR，然后在局部区域上完成真实的 graft，并把结果重新落回同一种 IR。**
