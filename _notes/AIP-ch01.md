---
layout: post
title: "人工智能中的编程 - 第1章: 并行编程（Parallel Programming）"
date: 2025-09-11 01:00:00
tags: notes AIP
categories: AIP
---

## 并行编程（Parallel Programming）

如何加快计算的速度？

- 提高时钟频率
- 使用并行计算

使用并行计算的原因：时钟频率不能无限制地提高，存在功耗和热量问题。

CPU和GPU的区别：

- CPU（中央处理器）：适合处理复杂的任务，具有较少的核心，但每个核心都很强大。优化时延（Latency）。
- GPU（图形处理器）：适合处理大量简单的任务，具有大量的核心，但每个核心相对较弱。优化吞吐量（Throughput）。

## CUDA

在CUDA编程模型中，程序被分为主机代码（Host Code）和设备代码（Device Code）。

- 主机代码在CPU上运行，负责管理内存和启动设备代码。
- 设备代码在GPU上运行，执行并行计算任务。

> What CPU does
> • CPU allocates a block of memory on GPU
> • CPU copies data from CPU to GPU
> • CPU initiates launching kernels on GPU
> • CPU copies results back from GPU to CPU

> What GPU does
> • GPU efficiently launch a lot of kernels
> • GPU runs kernels in parallel
> • A kernel looks like a serial C program for a thread
> • The GPU will run the kernel for many threads in parallel

CPU和GPU之间的数据传输是一个瓶颈，应该尽量减少数据传输的次数和数据量。

## 第一个CUDA程序

我们以relu函数为例，展示如何编写一个简单的CUDA程序。

```c
//relu on CPU
float relu_cpu(float x) {
return x > 0 ? x : 0;
}

for (int i = 0; i < N; ++i) {
h_out[i] = relu_cpu(h_in[i]);
}

```

```c
//relu on GPU
__global__ void relu_gpu(float* in, float* out) {
int i = threadIdx.x; // get thread index
out[i] = in[i] > 0 ? in[i] : 0;
}

relu_gpu<<<1, N>>>(d_in, d_out);
```

两种模块的执行方式：

CPU代码：

```c
// CPU code
const int N = 64;
const int size = N * sizeof(float);
// allocate memory on CPU
float* h_in = (float*) malloc(size);
float* h_out = (float*) malloc(size);
// initialize input array
for (int i = 0; i < N; ++i) {
h_in[i] = (i - 32) * 0.1;
}
// relu on CPU
for (int i = 0; i < N; ++i) {
h_out[i] = relu_cpu(h_in[i]);
}
// free memory ...
```

GPU代码：

```c
// GPU code
// 1. allocate memory on GPU
float* d_in = nullptr;
float* d_out = nullptr;
cudaMalloc(&d_in, size);
cudaMalloc(&d_out, size);
// 2. copy data from CPU to GPU
cudaMemcpy(d_in, h_in, size,
cudaMemcpyHostToDevice);
// 3. launch the kernel
relu_gpu<<<1, N>>>(d_in, d_out);
// 4. copy data from GPU to CPU
cudaMemcpy(h_out, d_out, size,
cudaMemcpyDeviceToHost);
// free memory ...
```

CUDA函数的调用方式：

```c
kernel<<<numBlocks, blockSize>>>(args);
// numBlocks: number of blocks
// blockSize: number of threads per block， typically 256 512 or 1024
// 一般会固定blockSize，然后根据数据量计算numBlocks
```

numBlocks传入的参数实际上是 `dim3(x,y,z)` 结构体，可以指定三维的网格结构。
