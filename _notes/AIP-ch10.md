---
layout: post
title: "人工智能中的编程 - 第10章: 计算图（Computational Graph）"
date: 2025-10-30 20:00:00
tags: notes AIP computational-graph DAG
categories: AIP
---

## 计算图在 AI 框架中的地位

计算图（Computational Graph）是现代 AI 框架的核心数据结构。它为高级编程语言（如 Python）和底层计算引擎（C/C++/CUDA）之间提供了统一的接口，使得用户可以用简洁的高级语言编写神经网络，同时框架可以在底层进行优化和加速。

### 计算图的四大任务

计算图需要支持以下关键功能：

1. **计算表示**（Computational Representation）

   - 统一的数据结构表示复杂的神经网络计算
   - 支持前向计算和反向求导

2. **自动求导**（Automatic Differentiation）

   - 自动计算神经网络中所有参数的梯度
   - 支持前向模式和反向模式自动微分

3. **变量生命周期分析**（Variable Lifecycle）

   - 精确追踪中间张量的生命周期
   - 辅助框架优化内存管理

4. **程序优化执行**（Program Optimization）
   - 对计算图进行优化和调度
   - 批处理、缓存、操作融合等优化

## 什么是计算图

计算图（Computational Graph, CG）是一种用**有向无环图**（DAG, Directed Acyclic Graph）表示神经网络和梯度计算的方式。

### 图的基本组成

计算图由以下基本元素组成：

- **节点（Nodes）**：代表操作符（Operators）
- **边（Edges）**：代表数据流（张量的流动）
- **特殊操作符**：控制流操作（if/else、for/while）
- **特殊边**：依赖边（表示操作之间的依赖关系）

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% raw %}{% include figure.liquid loading="eager" path="assets/img/notes_img/AIP-ch10/computational_graph_dag.png" title="计算图的前向和反向表示" class="img-fluid rounded z-depth-1" zoomable=true %}{% endraw %}
    </div>
</div>

上图展示了计算图的两个视图：

- **左图**：前向计算图，展示数据从输入流向损失函数的过程
- **右图**：前向和反向的完整计算图，包括反向传播的梯度流动

## 计算图中的操作符（DAG 节点）

### 张量操作（Tensor Operations）

用于张量的基本操作：

- Reshape、Concat、Matmul、Transpose、Slice 等
- 通常是元素级（element-wise）或矩阵级（matrix-level）操作

### 网络操作（Network Operations）

用于神经网络的操作：

- **损失函数**（Loss）：CrossEntropy、MSE 等
- **梯度计算**（Grads）：自动微分产生的反向操作
- **优化器**（Optimizers）：SGD、Adam 等参数更新操作

### 数据管理（Data Management）

用于数据处理的操作：

- Batch、Pre-fetch、Tile、Crop、Normalization 等
- 用于高效处理和预处理数据

### 控制流操作（Control Flow）

用于程序控制的操作：

- **条件分支**：if/else、switch
- **循环**：for/while、while_loop
- 提供程序的动态控制能力

## 计算图中的张量（DAG 边）

### 1. Ndarray（多维数组）

最常见的张量表示方式，适合 SIMT（Single-Instruction Multiple-Thread）并行计算。

**特点：**

- 密集存储，所有元素都占用内存
- GPU 友好，支持高效的并行计算
- 适合卷积、矩阵乘法等常见操作

**例子：**

```python
# 形状为 (batch, height, width, channels) 的图像数据
images = np.random.rand(32, 224, 224, 3)
```

### 2. Ragged Tensors（不规则张量）

用于表示长度不同的序列数据。

**特点：**

- 可变长度的行或列
- 适合处理文本、点云等不规则数据
- 节省内存，避免填充造成的浪费

**例子：**

```
句子1: [词1, 词2, 词3, 词4, 词5]        (长度5)
句子2: [词1, 词2, 词3]                  (长度3)
句子3: [词1, 词2, 词3, 词4]             (长度4)
```

### 3. Sparse Tensors（稀疏张量）

用于表示大量零元素的矩阵。

**存储格式：** 坐标列表（Coordinate List）

```
行索引：[0, 0, 0, 2, 2, 3, 4, 4, 4]
列索引：[0, 2, 4, 1, 4, 2, 0, 2, 4]
值列表：[1, 3, 5, 4, 8, 7, 6, 2, 9]
```

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% raw %}{% include figure.liquid loading="eager" path="assets/img/notes_img/AIP-ch10/tensor_types.png" title="三种张量表示方式" class="img-fluid rounded z-depth-1" zoomable=true %}{% endraw %}
    </div>
</div>

**应用场景：**

- 图神经网络中的邻接矩阵
- 推荐系统中的用户-物品交互矩阵
- 自然语言处理中的稀疏特征表示

## 计算图中的特殊边

### 依赖边（Dependency Edges）

除了数据边（tensor edges），计算图还包含特殊的依赖边，用于表示操作之间的执行依赖关系。

**分类：**

1. **直接数据依赖**

   - 操作 A 使用操作 B 的输出
   - A 和 B 之间有直接的数据边

2. **间接依赖**

   - 操作 A 依赖操作 B，但不直接使用 B 的输出
   - 例如：A 和 B 共享同一个 GPU 缓冲区

3. **独立操作**
   - A 和 B 之间没有任何边
   - 可以并行执行

**作用：**

- 帮助图调度器（Graph Dispatcher）确定操作的执行顺序
- 优化内存管理和资源利用

**具体例子：**

```python
tensor2 = opA(tensor1)
tensor3 = opB(tensor2)
tensor4 = opC(tensor2, tensor3)
```

在无依赖的情况下，opB 和 opC 可以并行执行（因为都依赖 tensor2）。但如果 opA 和 opB 共享 GPU 缓冲区，则需要添加依赖边，确保 opA 先完成再执行 opB。

## 计算图中的特殊操作符

### 控制流操作（Control-flow Operators）

现代 AI 框架支持三种方式来集成控制流操作：

#### 1. 后端原生支持（Native Backend Support）

AI 框架在底层直接提供控制流操作的支持。

**优点：**

- 控制流与数据流无缝集成
- 执行效率高

**例子：**

```python
# TensorFlow 原生支持
if_op = tf.cond(condition, lambda: output_true, lambda: output_false)
```

#### 2. 前端语言控制流（Frontend Language Control Flow）

框架直接利用前端语言（如 Python）的控制流逻辑。

**优点：**

- 编程灵活性高
- 用户可以使用熟悉的 Python 语法

**例子：**

```python
# PyTorch 动态图 - 使用 Python 的 if
if condition:
    x = model1(x)
else:
    x = model2(x)
```

**缺点：**

- 需要在前后端语言之间切换
- 可能运行在不同的硬件上

#### 3. 后端解析子图（Backend Parsing Subgraphs）

后端将前端的控制流逻辑解析为多个子图。

**优点：**

- 可以对控制流进行优化
- 支持图级优化

**例子：**

```python
# 解析为两个分支的子图
# Subgraph1: if 分支
# Subgraph2: else 分支
# Merge: 合并两个分支结果
```

## 编程范式与计算图的构建

### 声明式编程（Declarative Programming）vs 命令式编程（Imperative Programming）

构建计算图有两种主要的编程范式：

#### 声明式编程 - 静态图（Static Graph）

**代表框架：** TensorFlow 1.x

**工作流程：**

1. 先定义所有操作和数据流
2. 构建完整的计算图
3. 通过 Session 执行图

**特点：**

```python
# TensorFlow 1.x 示例
import tensorflow as tf

# 定义计算图
x = tf.placeholder(tf.float32, shape=(None, 10))
W = tf.Variable(tf.random_normal([10, 5]))
y = tf.matmul(x, W)

# 执行计算图
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run(y, feed_dict={x: data})
```

**优点：**

- ✓ 可以进行多种优化策略
- ✓ 执行效率高
- ✓ 易于部署

**缺点：**

- ✗ 编程不灵活
- ✗ 难以支持控制流（if/else、for/while）
- ✗ 调试困难

#### 命令式编程 - 动态图（Dynamic Graph）

**代表框架：** PyTorch

**工作流程：**

1. 动态执行代码
2. 在执行时记录操作和数据流
3. 构建计算图

**特点：**

```python
# PyTorch 示例
import torch

x = torch.randn(32, 10, requires_grad=True)
W = torch.randn(10, 5, requires_grad=True)

# 直接计算，自动构建图
y = torch.matmul(x, W)

# 反向传播
loss = y.sum()
loss.backward()
```

**优点：**

- ✓ 非常灵活，支持 Python 的所有语法
- ✓ 易于调试和理解
- ✓ 轻松支持控制流

**缺点：**

- ✗ 优化空间有限
- ✗ 执行性能相对较低
- ✗ 不易部署

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% raw %}{% include figure.liquid loading="eager" path="assets/img/notes_img/AIP-ch10/paradigms_comparison.png" title="编程范式对比" class="img-fluid rounded z-depth-1" zoomable=true %}{% endraw %}
    </div>
</div>

### 静态图与动态图的融合

现代 AI 框架（PyTorch 2.0、TensorFlow 2.0、JAX、MindSpore）采用**融合策略**：

**全局：动态图**

- 提供灵活的编程接口
- 支持 Python 的控制流和递归

**局部：静态子图**

- 通过函数式编程（Functional Programming）或图捕获
- 在特定函数中生成静态子图进行优化

**优点：**

- ✓ 编码和调试的灵活性（动态图）
- ✓ 执行和部署的效率（静态图）

## 图捕获（Graph Capture）

从动态图生成静态子图的关键技术是**图捕获**，主要分为两大类：

### 方法 1: 跟踪法（Trace-Based）

**原理：** 运行函数一次，记录所有操作

**过程：**

1. 用样本输入执行函数
2. 记录执行过程中的所有操作
3. 将记录转换为计算图

**代表实现：**

- `torch.jit.trace()`：记录执行轨迹
- `torch.fx.symbolic_trace()`：符号跟踪

**示例：**

```python
import torch

def f(x):
    return (x.relu() + 1) * x.pow(5)

# 跟踪方式1：JIT tracing
traced_f = torch.jit.trace(f, torch.randn(5, 5, 5))

# 跟踪方式2：Symbolic tracing
fx_f = torch.fx.symbolic_trace(f)

# 执行跟踪后的模型
result = traced_f(input_data)
```

**优点：**

- ✓ 易于实现
- ✓ 易于调试
- ✓ 与动态图执行方式相同

**缺点：**

- ✗ 只记录一次执行的路径
- ✗ 难以支持条件分支
- ✗ 难以处理依赖输入形状的操作

**局限性示例：**

```python
def conditional_func(x):
    if x.sum() < 0:
        return x + 1
    else:
        return x - 1

# 问题：跟踪只记录一个分支
traced = torch.jit.trace(conditional_func, torch.randn(10))
# 用其他输入执行时可能走另一个分支，但图里只有记录的分支
```

### 方法 2: 源码转换法（AST-Based Source Transformation）

**原理：** 解析 Python 源代码，转换为中间表示（IR）

**过程：**

1. 解析（Parse）：得到抽象语法树（AST）
2. 推断（Infer）：完成类型推断和代码规范化
3. 转换（Transform）：将 AST 转换为计算图 IR
4. 优化（Optimize）：对 IR 进行优化
5. 编译（Compile）：生成本地代码

**代表实现：**

- `@torch.jit.script`：JIT 脚本编译
- `torch.compile()`：TorchDynamo + 后端编译器
- `@tf.function`：TensorFlow 函数追踪

**示例：**

```python
# 方式1: JIT script（源码级转换）
@torch.jit.script
def conditional_func(x):
    if x.sum() < 0:
        return x + 1
    else:
        return x - 1

# 方式2: torch.compile（动态优化）
@torch.compile
def model(x):
    return (x + 1).relu() * 2
```

**优点：**

- ✓ 支持更广泛的控制流
- ✓ 支持高阶梯度计算
- ✓ 生成的图更完整

**缺点：**

- ✗ 实现复杂度高
- ✗ 生成的代码难以理解
- ✗ 需要强大的错误检查系统

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% raw %}{% include figure.liquid loading="eager" path="assets/img/notes_img/AIP-ch10/graph_capture_comparison.png" title="图捕获方法对比" class="img-fluid rounded z-depth-1" zoomable=true %}{% endraw %}
    </div>
</div>

### PyTorch 2.0 的混合方案

PyTorch 2.0 通过 `torch.compile()` 结合两种方法：

```python
import torch

# 用户代码保持完全动态
def model(x):
    if x.sum() > 0:
        return (x + 1).relu()
    else:
        return (x - 1).relu()

# 编译器自动优化
compiled_model = torch.compile(model)

# 第一次调用：捕获图
result1 = compiled_model(torch.randn(10))

# 后续调用：使用优化的静态图
result2 = compiled_model(torch.randn(10))
```

**执行过程：**

1. TorchDynamo 截获 Python 字节码
2. 识别可编译的子图
3. 对每个子图生成 IR
4. 后端编译器优化 IR
5. 执行编译后的代码

## AI 框架的三代发展

深度学习框架的演进经历了三个阶段，每个阶段都在编程灵活性和执行效率之间找到不同的平衡点。

### 第一代：库基础（Library-Based, pre-2010）

**代表框架：** NumPy、SciPy、MATLAB

**特点：**

- 提供基础数学库函数
- 用户手动组合库函数实现算法
- 基于表达式的自动微分实现

**优点：**

- 实现简单
- 通用性强

**缺点：**

- 编程复杂，需要大量库函数调用
- 无法使用高级语言的原生语法
- 新操作需要手动微分

### 第二代：DAG 基础（DAG-Based, 2010-present）

**代表框架：** TensorFlow 1.x、Caffe、Theano

**特点：**

- 使用有向无环图表示计算
- 明确的操作符（节点）和张量（边）
- 支持对计算图的全局优化

**分为两个方向：**

**方向 A：性能优先** - TensorFlow

- 静态图，编译优化能力强
- 执行效率高，但灵活性受限

**方向 B：灵活性优先** - PyTorch

- 动态图，支持即时开发
- 编程简单，但优化能力有限

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% raw %}{% include figure.liquid loading="eager" path="assets/img/notes_img/AIP-ch10/three_generations.png" title="AI 框架的三代发展" class="img-fluid rounded z-depth-1" zoomable=true %}{% endraw %}
    </div>
</div>

### 第三代：源码转换法（AST-Based，Present）

**代表框架：** PyTorch 2.0、TensorFlow 2.0、JAX、MindSpore

**特点：**

- 结合命令式编程的易用性和函数式编程的优化方式
- 动态图编程 + 静态子图优化
- 灵活性和效率的完美融合

**关键创新：**

1. **全局动态** → **局部静态**

   - 程序整体采用动态图，用户可自由使用 Python 特性
   - 在特定函数/模块处理为静态子图进行优化

2. **自动图捕获**

   - 框架自动捕获特定函数的执行
   - 转换为可优化的静态图

3. **智能优化**
   - 代码优化（死代码消除、公共子表达式）
   - 稀疏性优化
   - 硬件感知优化

**代表技术：**

```python
# PyTorch 2.0
@torch.compile
def forward(x):
    return model(x)  # 自动优化为静态子图

# JAX
jitted_fn = jax.jit(forward)

# TensorFlow 2.0
@tf.function
def forward(x):
    return model(x)
```

## 计算图的优化与执行

### 图优化（Graph Optimization）

计算图被构建后，在执行前进行多种优化：

1. **算子融合（Operator Fusion）**

   - 将多个小算子合并为一个大算子
   - 减少内存访问和数据传输

2. **批处理（Batching）**

   - 将多个独立计算合并为批处理
   - 提高硬件利用率

3. **缓存优化（Cache Optimization）**

   - 优化数据访问模式
   - 提高 CPU 缓存命中率

4. **内存管理（Memory Management）**
   - 合理分配和释放中间张量内存
   - 支持梯度检查点节省显存

### 图调度执行（Graph Dispatch & Execution）

优化后的图被调度到不同的硬件执行：

- **CPU 执行**：多线程并行
- **GPU 执行**：使用 CUDA 核函数
- **分布式执行**：跨机器、跨 GPU 通信

## 总结

**计算图**是现代 AI 框架的核心抽象：

1. **统一表示**：用 DAG 统一表示神经网络和梯度计算
2. **支持多种操作**：张量操作、网络操作、控制流操作
3. **支持多种张量**：密集、不规则、稀疏张量
4. **灵活性与效率的融合**
   - 全局：动态图编程的灵活性
   - 局部：静态子图的执行效率
5. **自动优化**：框架自动对图进行优化和并行化

现代 AI 框架（PyTorch、TensorFlow、JAX）都在朝着这个方向演进，为用户提供既灵活又高效的深度学习开发体验。

**下一步：** 图优化、自动求导实现、分布式执行等具体技术细节。
