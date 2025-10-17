---
layout: post
title: "人工智能中的编程 - 第7章: 卷积和池化（Convolution and Pooling）"
date: 2025-10-17 03:00:00
tags: notes AIP
categories: AIP
---

## AlexNet架构

AlexNet在2012年的ImageNet竞赛中取得了最佳性能，标志着深度学习在计算机视觉领域的重大突破。

### AlexNet的核心组件

AlexNet架构包含以下关键组件：

- **卷积层（Convolution）**：提取图像特征
- **最大池化（Max Pooling）**：降采样特征图
- **全连接层（Fully Connected Layer）**：高层特征整合
- **Softmax和损失函数（Softmax & Loss）**：分类和训练

## 全连接层的优缺点

### 全连接层的特点

**优点**：

- **表达能力强**：能够学习任意复杂的特征组合
- **易于实现**：可以通过GEMM高效实现

**缺点**：

- **参数量巨大**：200×200→1000的全连接层需要200M参数
- **缺乏平移不变性/等变性**：对于图像这样的输入，无法利用空间结构

$$y = xA^T$$

对于全连接层，输入的不同位置使用完全不同的权重矩阵列，无法共享参数。

## 卷积层

### 卷积层的核心思想

卷积层通过以下机制解决全连接层的问题：

**局部连接（Localized Connections）**：

- 在输入特征图上滑动一个$$k \times k$$的卷积核
- 大幅减少参数数量
- 典型卷积核大小：$$3 \times 3$$

**权重共享（Weight Sharing）**：

- 在所有空间位置共享同一组权重
- 实现平移等变性（Translation Equivariance）
- 输入特征的平移导致输出特征相应平移

### 卷积操作的数学定义

对于输入特征图$$X$$和卷积核$$W$$，输出特征图$$Y$$的计算为：

$$y_{i,j} = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} w_{m,n} \cdot x_{i+m, j+n}$$

## 卷积层的边界处理

### 常见的边界处理方法

**1. 忽略边界位置（Ignore These Locations）**：

- 不计算边界列的卷积结果
- 输出尺寸会缩小

**2. 零填充（Pad with Zeros）**：

- 在图像边界外填充零值
- 保持输出尺寸不变
- 最常用的方法

**3. 周期性假设（Assume Periodicity）**：

- 顶行环绕到底行
- 最左列环绕到最右列
- 适用于具有周期性的数据

**4. 反射边界（Reflect Border）**：

- 通过镜像边界复制行/列
- 保持边界的连续性

### PyTorch中的填充

```python
torch.nn.functional.pad(image, pad, mode='constant', value=None)
```

模式选项：`'constant'`、`'reflect'`、`'replicate'` 或 `'circular'`

## PyTorch中的卷积

### 多通道卷积

PyTorch中的卷积将多通道输入映射到多通道输出：

```python
torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0,
                           dilation=1, groups=1) → Tensor
```

**参数说明**：

- `input`：输入张量，形状为$$[N, C_{in}, H, W]$$
- `weight`：卷积核，形状为$$[C_{out}, C_{in}, K_H, K_W]$$
- `bias`：可选的偏置张量，形状为$$[C_{out}]$$
- `stride`：卷积核的步长
- `padding`：输入两侧的隐式填充

### 输出尺寸计算

对于输入尺寸$$H \times W$$，卷积核尺寸$$K \times K$$，填充$$P$$，步长$$S$$：

$$H_{out} = \left\lfloor \frac{H + 2P - K}{S} \right\rfloor + 1$$

$$W_{out} = \left\lfloor \frac{W + 2P - K}{S} \right\rfloor + 1$$

## 步长卷积和池化

### 生成不同分辨率的特征

有两种主要方法来降低特征图的空间分辨率：

**方法1：池化层**

- 使用最大池化或平均池化聚合信息
- 典型配置：$$2 \times 2$$窗口，步长2

**方法2：步长卷积**

- 卷积核以大于1的步长（stride）移动
- 同时实现下采样和特征提取

## 分组卷积

### 分组卷积的动机

对于大的输入/输出通道数，卷积核$$[C_{out}, C_{in}, K_H, K_W]$$仍然包含大量参数，导致：

- 过拟合风险
- 计算速度慢

### 分组卷积原理

**核心思想**：将通道分组，每组输出通道只依赖对应组的输入通道

**深度可分离卷积（Depthwise Convolution）**：

- 组数等于通道数
- 每个通道独立卷积
- 广泛应用于MobileNets等轻量级网络

**参数量对比**：

- 标准卷积：$$C_{out} \times C_{in} \times K \times K$$
- 深度可分离卷积：$$C \times K \times K$$

## 卷积的朴素实现

### 多重循环实现

对于输入形状$$[N, C_i, H, W]$$，输出形状$$[N, C_o, H, W]$$，卷积核形状$$[C_o, C_i, K, K]$$：

```python
for n in 1..N
    for w in 1..W
        for h in 1..H
            for k in 1..K
                for q in 1..K
                    for c0 in 1..Co
                        for c1 in 1..Ci
                            output(n, w, h, c0) +=
                                input(n, w+k, h+q, c1) * kernel(c0, c1, k, q)
```

**问题**：

- 七重循环，难以优化
- 比PyTorch慢1000倍以上

## 卷积作为矩阵乘法

### 版本1：稀疏矩阵-向量乘法（SpMV）

将卷积写成稀疏矩阵与向量的乘积：

$$Y = \hat{W}X$$

其中$$\hat{W}$$是从卷积核$$W$$构造的稀疏矩阵。

**前向传播**：

1. 从$$W$$构造$$\hat{W}$$
2. 执行SpMV：$$Y = \hat{W}X$$

**反向传播**：

- $$\frac{\partial L}{\partial X} = \hat{W}^T \times \frac{\partial L}{\partial Y}$$（转置卷积）
- $$\frac{\partial L}{\partial W} \leftarrow \frac{\partial L}{\partial \hat{W}}$$（需要排序和分段归约）

### 版本2：显式GEMM（Explicit GEMM）

通过im2col将卷积转换为密集矩阵乘法：

$$Y = \hat{X}W$$

**im2col操作**：

- 将输入的局部窗口展开成矩阵的列
- 构造的$$\hat{X}$$矩阵尺寸大，但可以利用高度优化的GEMM

**前向传播**：

1. 通过im2col从$$X$$构造$$\hat{X}$$
2. 执行GEMM：$$Y = \hat{X}W$$

**反向传播**：

- $$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial Y} \times \hat{X}^T$$（GEMM）
- $$\frac{\partial L}{\partial X} \leftarrow \frac{\partial L}{\partial \hat{X}}$$（col2im，使用原子操作）

**优点**：

- 实现简单
- 极其高效

**缺点**：

- 消耗大量内存
- 每个输入元素被复制$$K \times K$$次

### 版本3：隐式GEMM（Implicit GEMM）

**核心思想**：

- 在数据从全局内存加载到共享内存时，动态构造卷积矩阵的tile
- 利用现有的warp级GEMM组件累积卷积结果
- 更节省内存和计算

详见：[NVIDIA CUTLASS - Implicit GEMM Convolution](https://github.com/NVIDIA/cutlass/blob/main/media/docs/implicit_gemm_convolution.md)

## 使用CUDA实现卷积

### 实现步骤

1. **实现im2col和col2im**
2. **结合im2col和GEMM实现前向传播**：$$Y = \hat{X}W$$
3. **结合im2col和GEMM实现权重的反向传播**：$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial Y} \times \hat{X}^T$$
4. **结合col2im和GEMM实现输入的反向传播**：$$\frac{\partial L}{\partial \hat{X}} = W^T \times \frac{\partial L}{\partial Y}$$
5. **使用GEMM添加卷积偏置**

### 偏置项处理

偏置项也可以看作一个神经网络层，通过广播机制添加到每个空间位置。

## 深度可分离卷积

### 特点和应用

**应用场景**：

- MobileNets：广泛应用于移动和嵌入式视觉应用
- 显著更快且包含更少的可训练参数

**参数和计算量对比**：

对于输入形状$$[N, C, H, W]$$和卷积核大小$$[K, K]$$：

- **标准卷积**：每个输出元素需要$$K \times K \times C$$次乘法，卷积核形状$$[C, C, K, K]$$
- **深度可分离卷积**：每个输出元素只需$$K \times K$$次乘法，卷积核形状$$[C, K, K]$$

### 深度可分离卷积是内存受限的

**特点**：

- 工作复杂度比标准卷积低$$C$$倍（例如128倍）
- 对于输入形状$$[N, C_i, H, W]$$，输出形状$$[N, C_o, H, W]$$，卷积核形状$$[C, K, K]$$，只需要6重循环
- 应该优先改进内存复杂度

### CUDA实现

```cuda
for n in 1..N
    for c in 1..C
        for h in 1..H
            for w in 1..W
                for k in 1..K
                    for q in 1..K
                        output(n, w, h, c) +=
                            input(n, w+k, h+q, c) * kernel(c, k, q)
```

**实现策略**：

- 用CUDA kernel并行化前4个循环
- 每个线程负责一个输出元素
- 典型情况下$$K=3$$，每个线程执行9次乘法

**前向传播和输入梯度**：实现相似

**权重梯度计算**：相对困难

- 需要将梯度归约到$$C \times K \times K$$个参数
- 利用共享内存+归约操作

## 池化层

### 最大池化（Max Pooling）

最大池化在每个通道上独立进行空间降采样：

**常见配置**：

- 核大小：$$2 \times 2$$
- 步长：2
- 输出尺寸为输入的1/2

### PyTorch中的最大池化

```python
torch.nn.functional.max_pool2d(input, kernel_size, stride=None, padding=0,
                               dilation=1, ceil_mode=False, return_indices=False)
```

**参数**：

- `input`：输入张量 (minibatch, in_channels, $$iH$$, $$iW$$)
- `kernel_size`：池化区域的大小，可以是单个数字或元组 (kH, kW)
- `stride`：池化操作的步长，默认等于kernel_size
- `padding`：隐式负无穷填充
- `dilation`：滑动窗口内元素之间的步长

### 最大池化的前向传播

```cuda
__global__ void max_pool_forward(
    float* in_data, int nthreads, int num, int channels,
    int in_h, int in_w, int out_h, int out_w, int kernel_h,
    int kernel_w, int stride_h, int stride_w, int pad_h,
    int pad_w, float* out_data, float* out_mask) {

    CUDA_KERNEL_LOOP(index, nthreads) {
        int n = index / out_w / out_h / channels;
        int c = (index / out_w / out_h) % channels;
        int ph = (index / out_w) % out_h;
        int pw = index % out_w;

        // 对每个局部窗口实现最大池化
        // 将最大值和掩码存储到out_data[index]和out_mask[index]
    }
}
```

**实现要点**：

- 每个线程负责一个局部窗口
- 同时记录最大值位置（掩码）用于反向传播

### 最大池化的反向传播

**梯度传播规则**：

$$\max(x_1, x_2, \ldots, x_k)$$的梯度为：

- 对于最大值位置：梯度传递
- 对于其他位置：梯度为0

**反池化（Unpooling）**：

- 根据前向传播保存的掩码将梯度放回原位置
- 其他位置填充0
- 广泛用于图像分割任务

### Stencil并行模式

卷积和池化操作属于Stencil模式：

- **Gather（多对一）**：卷积/池化的前向传播
- **Scatter（一对多）**：卷积/池化的反向传播
- **Stencil（固定邻域的多对一）**：具有固定邻域模式的gather

**实现策略**：当邻域尺寸较小时，为每个输出元素启动一个线程。

## Softmax函数

### Softmax的定义

Softmax函数将网络预测转换为概率分布：

$$S(a) : \begin{bmatrix} a_1 \\ a_2 \\ \vdots \\ a_N \end{bmatrix} \rightarrow \begin{bmatrix} S_1 \\ S_2 \\ \vdots \\ S_N \end{bmatrix}, \quad S_j = \frac{e^{a_j}}{\sum_{k=1}^N e^{a_k}}, \quad \forall j \in 1..N$$

**示例**：

- $$[1.0, 2.0, 3.0] \rightarrow [0.09, 0.24, 0.67]$$
- $$[1.0, 2.0, 5.0] \rightarrow [0.02, 0.05, 0.93]$$

### 数值稳定性

**朴素实现**：

```python
def softmax(x):
    exps = np.exp(x)
    return exps / np.sum(exps)
```

**问题**：对于大的输入值会溢出

**稳定实现**：

```python
def stable_softmax(x):
    d = x - np.max(x)
    exps = np.exp(d)
    return exps / np.sum(exps)
```

**数学原理**：

$$S_j = \frac{e^{a_j}}{\sum_{k=1}^N e^{a_k}} = \frac{e^{a_j+D}}{\sum_{k=1}^N e^{a_k+D}}, \quad D = -\max(a_1, a_2, \cdots, a_N)$$

### 使用CUDA实现Softmax

对于输入$$[N, C]$$，输出$$[N, C]$$（$$C$$通常为10，用于MNIST分类）：

1. **计算每行的最大值**：使用max归约，或每行一个线程直接计算
2. **减去最大值**（Map操作）
3. **计算每个元素的指数**（Map操作）
4. **对每行求和计算归一化因子**：使用sum归约，或每行一个线程直接计算
5. **归一化结果**（Map操作）

### Softmax的梯度

Softmax的Jacobian矩阵：

$$\frac{\partial p_i}{\partial o_j} = \begin{cases} p_i(1-p_j) & \text{if } i = j \\ -p_j \cdot p_i & \text{if } i \neq j \end{cases}$$

其中$$p_i \triangleq S_i$$

## 交叉熵损失

### 交叉熵损失的定义

交叉熵损失衡量预测概率分布和真实概率分布之间的距离：

$$H(y, p) = -\sum_i y_i \log(p_i)$$

### Python实现

```python
def cross_entropy(X, y):
    p = softmax(X)
    m = y.shape[0]
    log_likelihood = -np.log(p[range(m), y])  # map
    loss = np.sum(log_likelihood) / m  # reduce mean
    return loss
```

### 交叉熵损失与Softmax的梯度

**组合梯度计算**：

$$L = -\sum_i y_i \log(p_i)$$

$$\frac{\partial L}{\partial o_i} = p_i - y_i$$

**推导过程**：

$$\frac{\partial L}{\partial o_i} = -\sum_k y_k \frac{\partial \log(p_k)}{\partial o_i} = -\sum_k y_k \frac{1}{p_k} \times \frac{\partial p_k}{\partial o_i}$$

利用Softmax的梯度公式，最终得到简洁的结果：$$p_i - y_i$$

**实现优势**：

- 直接计算$$p - y$$，无需显式计算Jacobian矩阵
- 数值稳定且计算高效

## 转置卷积（反卷积）

### 用途和原理

**应用场景**：

- 上采样特征图
- 广泛应用于图像分割和图像生成

**实现方法**：

- 交换卷积的前向传播和反向传播
- 前向传播使用：$$\frac{\partial L}{\partial \hat{X}} = W^T \times \frac{\partial L}{\partial Y}$$

**示例**：

步长卷积：
$$\text{Input}(6 \times 6) \xrightarrow[\text{stride 2}]{\text{conv } 3 \times 3} \text{Output}(2 \times 2)$$

转置卷积：
$$\text{Input}(2 \times 2) \xrightarrow[\text{stride 2}]{\text{transposed conv } 3 \times 3} \text{Output}(6 \times 6)$$

转置卷积通过在输入之间插入零值和应用常规卷积来实现上采样效果。

## 总结

### 卷积和池化的关键概念

1. **卷积层优势**：

   - 参数共享减少模型大小
   - 平移等变性适合图像处理
   - 局部连接捕获空间结构

2. **实现策略**：

   - im2col + GEMM：简单高效但内存密集
   - 隐式GEMM：内存高效的高性能实现
   - 深度可分离卷积：轻量级网络的选择

3. **池化层作用**：

   - 空间降采样
   - 增加感受野
   - 提供一定的平移不变性

4. **Softmax和损失**：

   - 数值稳定性很重要
   - Softmax + 交叉熵的组合梯度简洁高效

5. **并行模式**：
   - Stencil模式适用于固定邻域操作
   - Map和Reduce操作的组合

通过理解这些核心概念和实现技术，我们可以构建高效的深度学习系统，充分利用GPU的计算能力。
