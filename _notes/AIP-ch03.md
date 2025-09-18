---
layout: post
title: "人工智能中的编程 - 第3章: 并行通信（Parallel Communication）"
date: 2025-09-18 19:00:00
tags: notes AIP
categories: AIP
---

## 上节课回顾

- **并行编程的必要性**：时钟频率无法无限提升
- **第一个CUDA程序**：实现ReLU激活函数
- **GPU内存模型**：Local/Shared/Global内存层次结构
- **线程组织**：Thread/Block/Grid三级架构

## 线程块与编程人员

kernels: C/C++ functions (`__global__`)

**线程块 (Thread Blocks)**：协作解决子问题的线程组

- 程序员负责定义块结构
- GPU负责将线程块分配给硬件SM

### 线程块与GPU硬件

```
thread → GPU core
thread block → Streaming Multiprocessor (SM)
```

**关键特性**：

- 一个线程块必须在一个SM上运行；一个SM可以运行多个块
- 程序员负责定义块；GPU负责分配线程块到硬件SM
- 线程并行运行，CUDA对线程块何时何地运行提供很少保证
- 效率、简洁性、可扩展性
- 块之间无通信（避免死锁）

## CUDA执行保证与限制

### 线程块执行特点

**CUDA的少量保证**：

- 对线程块何时何地执行提供很少保证
- 不同块/线程并行运行
- 块之间不能通信（避免死锁）

**确定性保证**：

- 块内所有线程在同一SM上同时运行
- 当前kernel的所有块完成后，下一个kernel的块才开始

### 线程块编程示例

```c
#include <stdio.h>
__global__ void hello() {
    printf("Hello world! I'm a thread in block %d\n", blockIdx.x);
}

int main(int argc, char **argv) {
    hello<<<16, 1>>>();
    cudaDeviceSynchronize();
    printf("That's all!\n");
    return 0;
}
```

不同运行的结果会改变吗？这个程序在不同运行中可以产生多少种不同的结果？

答案：**16!** 种结果（让我们试试看）

没有 `cudaDeviceSynchronize()` 会发生什么？

### 另一个线程块编程示例

```c
#include <stdio.h>
__global__ void hello(float f) {
    printf("Hello world! I'm thread %d, f=%f\n", threadIdx.x, f);
}

int main(int argc, char **argv) {
    hello<<<1, 5>>>(1.2345f);
    cudaDeviceSynchronize();
    printf("That's all!\n");
    return 0;
}
```

输出示例：

```
Hello thread 2, f=1.2345
Hello thread 1, f=1.2345
Hello thread 4, f=1.2345
Hello thread 0, f=1.2345
Hello thread 3, f=1.2345
```

**注意**：在kernel中使用printf非常耗时

## SIMT：单指令多线程 (Single-Instruction, Multiple-Thread)

GPU多处理器以**32个并行线程**为组（称为**warp**）来创建、管理、调度和执行线程。

- **Warp执行**：一个warp同时执行一条共同指令，当所有32个线程在执行路径上一致时实现完全效率
- **分支分化**：如果warp内线程通过数据相关的条件分支发生分化，warp会执行每个分支路径，禁用不在该路径上的线程
- **分化范围**：分支分化只在warp内发生；不同warp独立执行，无论它们执行相同还是不同的代码路径

### 分支分化 (Thread Divergence)

当warp内线程执行不同的条件分支时发生分化：

```c
if (condition) {
    // 一些代码行
} else {
    // 其他代码行
}
```

**分化处理机制**：

- Warp串行执行每个分支路径
- 禁用不在当前路径的线程（等待状态）
- 分化只影响warp内部，不同warp独立执行

### 分支分化示例

```c
for(int i = 0; i <= threadIdx.x; ++i) {
    // 一些代码行
}
```

不同线程执行不同次数的循环，造成warp内分化，降低执行效率。

## 线程同步 (Thread Synchronization)

### 并行线程如何协作？

线程可以通过**全局内存和共享内存**访问彼此的结果

**注意事项**：

- 一个线程在另一个线程写入前读取结果
- 多个线程写入同一内存位置（例如求和）

**线程需要同步来协作**

**屏障 (Barrier)**：程序中线程停止并等待所有线程到达屏障的点；然后线程继续执行。

### 使用屏障避免数据竞争

错误的代码：

```c
const int N = 128;
__global__ void shift_sum(float* array) {
    int idx = threadIdx.x;
    if (idx < N-1) {
        array[idx] = array[idx] + array[idx+1];  // !!! 可能的BUG
    }
}
```

正确的代码：

```c
__global__ void shift_sum(float* array) {
    int idx = threadIdx.x;
    if (idx < N-1) {
        float tmp = array[idx] + array[idx+1];
        __syncthreads();
        array[idx] = tmp;
    }
}
```

### 使用屏障避免数据竞争 + 共享内存

使用共享内存减少全局内存访问：

```c
__global__ void shift_sum(float* array) {
    // 共享内存可以被块内所有线程访问
    __shared__ float shared[N];

    // 填充共享内存
    int idx = threadIdx.x;
    shared[idx] = array[idx];
    __syncthreads();

    // 执行"移位求和"
    if (idx < N-1) {
        array[idx] = shared[idx] + shared[idx+1];
    }

    // 以下代码无效果
    shared[idx] = 3.14;
}
```

### 线程协作的另一个示例

大量线程读写相同内存位置

- 例如，100万个线程增量10个数组元素

错误的方法：

```c
const int kNumThreads = 1000000;
const int kArraySize = 100;
const int kBlockWidth = 1000;

__global__ void increment_naive(int *g) {
    // 线程索引
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // 每个线程增量连续元素
    i = i % kArraySize;
    g[i] = g[i] + 1;
}

increment_naive<<<kNumThreads/kBlockWidth, kBlockWidth>>>(d_array);
```

这个程序能产生正确结果吗？ `__syncthreads()` 能修复这个bug吗？

### 原子内存操作

**原子内存操作**：在内存位置上以线程安全方式执行读-修改-写操作。

这些操作提供了确保在给定时间只有一个线程可以读或写内存位置的方法。

常用操作包括 `atomicAdd`、`atomicMin`、`atomicCAS`

**atomicCAS**: 比较并交换

- 我们可以使用 `atomicCAS` 构造通用原子操作

正确的方法：

```c
__global__ void increment_atomic(int *g) {
    // 线程索引
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // 每个线程增量连续元素
    i = i % kArraySize;
    atomicAdd(&g[i], 1);
}

increment_atomic<<<kNumThreads/kBlockWidth, kBlockWidth>>>(d_array);
```

我们必须使用 `atomicAdd()`。让我们运行代码试试看！

### 原子操作的局限性

**结果不完全可重现**

- 例如，$s = \sum_{i=1}^{N} x_i$ 的多次运行可能产生不同结果（为什么？）
- 即使随机种子固定，神经网络训练的多次运行产生的权重也会不同

**串行化内存访问**

- 极大地减慢程序速度

让我们运行代码并检查运行时间

### 原子操作测验

以下操作哪个是正确的？哪个是最快的？

- 10⁶个线程增量10⁶个元素
- 10⁶个线程原子增量10⁶个元素
- 10⁶个线程增量100个元素
- 10⁶个线程原子增量100个元素
- 10⁷个线程原子增量100个元素

## 测量速度

### CPU上测量速度

一个测量CPU时间的简单方法：

```c
// 测量CPU时间的简单方法
#include <ctime>

std::clock_t start = std::clock();
////////////////////////////
// 把你的C++代码放在这里 //
////////////////////////////
std::clock_t end = std::clock();
double time_elapsed = (end - start) / CLOCKS_PER_SEC;
```

我们能把CUDA kernel函数放在这里来测量时间吗？

**不行！**

CPU代码和GPU kernel异步运行。

### GPU上测量速度

```c
// 在GPU上测量时间
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, 0);
///////////////////////////////
// 把你的CUDA kernel放在这里 //
///////////////////////////////
cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);

float elapsed;
cudaEventElapsedTime(&elapsed, start, stop);
cudaEventDestroy(start);
cudaEventDestroy(stop);
```

**注意**：`cudaMemcpy` 不是kernel函数。

我们应该使用cuda事件API。

CPU代码和GPU kernel异步运行。

我们可以使用C++实现一个类来简化其用法。

### PyTorch封装的CUDA事件

```python
import torch

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
# 你要计时的任何操作放在这里
end.record()

# 等待所有操作完成运行
torch.cuda.synchronize()
print(start.elapsed_time(end))
```

## 并行通信模式 (Parallel Communication Patterns)

**并行计算**：许多线程协作解决问题

这些线程如何在内存上通信？

### Map：一对一

从特定内存位置读取并写入特定内存位置

### Gather：多对一

### Scatter：一对多

Scatter是Gather的逆操作

### Stencil

从数组中的固定邻域读取输入

- Stencil是gather的特殊情况
- Stencil的反向传播是scatter的特殊情况

### Transpose

#### 矩阵转置

#### 数据结构转置

```c
struct A {
    float f;
    int i;
};
```

数组结构（Array of Structures）转换为结构数组（Structure of Arrays）：

## 并行通信模式总结

### 通信模式概览

| 模式          | 映射关系 | 典型应用 | 示例                        |
| ------------- | -------- | -------- | --------------------------- |
| **Map**       | 一对一   | 激活函数 | 激活函数                    |
| **Transpose** | 一对一   | 数据重排 | `torch.transpose`/`permute` |
| **Gather**    | 多对一   | 数据收集 | 索引操作                    |
| **Scatter**   | 一对多   | 数据分发 | 反向索引                    |
| **Stencil**   | 多对一   | 卷积操作 | 卷积                        |
| **Reduce**    | 全对一   | 归约操作 | `torch.sum`/`mean`          |

## 本节课总结

### GPU内存

- 内存管理；Tensor
- Local / Shared / Global内存

### GPU硬件

- 并行化线程和块
- SIMT：单指令多线程

### 同步

- 屏障
- 原子操作

### 通信模式

- Map, Gather, Scatter, Stencil, Transpose

这些并行通信模式构成了现代深度学习框架的基础，理解它们对设计高效的GPU程序至关重要。
