---
layout: post
title: "人工智能中的编程 - 第8章: Pybind与单元测试（Pybind and Unit Test）"
date: 2025-10-18 04:00:00
tags: notes AIP
categories: AIP
---

## Python与CUDA/C++的互补性

### Python和C++各自的优势

现代深度学习框架普遍采用Python和CUDA/C++混合编程的架构：

**Python的特点**：
- 高级通用编程语言
- 专注于提高生产力和代码可读性
- 核心语法简洁，标准库功能丰富
- 提供快速开发和灵活性

**CUDA/C++的特点**：
- 提供高效的计算性能
- 底层硬件控制能力强
- 适合计算密集型任务

**最佳实践**：
- Python层：模型定义、数据处理、实验流程
- C++/CUDA层：高性能算子、底层优化

## Pybind11简介

### 什么是Pybind11

**pybind11**是一个轻量级的**仅头文件（header-only）库**，用于在Python和C++之间相互暴露类型。

**特点**：
- 类似于Boost.Python库，但更轻量
- Boost.Python几乎兼容所有C++编译器，但体积过大
- pybind11是Boost.Python的精简自包含版本
- PyTorch等主流深度学习框架广泛使用

**作者和应用**：
- 作者：Wenzel Jakob
- 代表性应用：Mitsuba、Instant Meshes（SGP软件奖获得者）、PBRT

### 安装Pybind11

**方法1：直接下载源码**

```bash
git clone https://github.com/pybind/pybind11.git
```

然后添加头文件目录到项目中。

**方法2：使用pip安装**

```bash
pip install pybind11
```

**注意**：如果已安装PyTorch，pybind11通常已经包含在内，可用于扩展PyTorch。

## Pybind11基础使用

### 第一个Pybind示例

**C++代码**：

```cpp
#include <pybind11/pybind11.h>

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(example, m) {
    m.def("add", &add, "A function that adds two numbers");
}
```

**关键要点**：
- 包含头文件`pybind11/pybind11.h`
- 使用`PYBIND11_MODULE`宏定义模块
- "example"是Python模块的名称
- `m.def`用于绑定C++函数到Python

### 使用CMake编译

**基本CMakeLists.txt配置**：

```cmake
cmake_minimum_required(VERSION 3.12)
project(cmake_example LANGUAGES CUDA CXX C)  # CUDA支持
set(CMAKE_CXX_STANDARD 11)  # C++11支持
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(pybind11)
pybind11_add_module(cmake_example src/main.cpp)  # 所有cpp文件
```

**构建步骤**：

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release --target check
```

**添加包含目录和源文件**：

```cmake
# 添加包含目录
set(your_library_path "${PROJECT_SOURCE_DIR}/external/library")
include_directories(${your_library_path})

# 查找所有源文件
file(GLOB src
    "${PROJECT_SOURCE_DIR}/*.cu"
    "${PROJECT_SOURCE_DIR}/*.cpp")

# 添加可执行文件
add_executable(example_exec ${src})
```

## 绑定C++类

### 类的绑定示例

**C++类定义**：

```cpp
#include <vector>

class Pet {
public:
    Pet(const std::string &name, const std::vector<int> &weights)
        : name(name), weights(weights) {}

    void setName(const std::string &name_) { name = name_; }
    const std::string &getName() const { return name; }
    const std::vector<int> &getWeights() const { return weights; }

protected:
    std::string name;
    std::vector<int> weights;
};
```

**Pybind绑定代码**：

```cpp
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(example, m) {
    py::class_<Pet>(m, "Pet")
        .def(py::init<const std::string &, const std::vector<int> &>())
        .def("setName", &Pet::setName)
        .def("getName", &Pet::getName)
        .def("getWeights", &Pet::getWeights);
}
```

**类型转换**：
- `std::string` ↔ Python `str`
- `std::vector<int>` ↔ Python `list`

### Python中使用绑定的类

```python
>>> import example
>>> p = example.Pet("Molly", [1, 2])
>>> print(p)
<example.Pet object at 0x10cd98060>
>>> p.getName()
'Molly'
>>> p.setName("Charly")
>>> p.getWeights()
[1, 2]
```

## 绑定NumPy数组

### NumPy与C++的交互

NumPy数组可以在C++中通过`py::array_t<Dtype>`访问，其中`Dtype`可以是`double`、`float`、`int`。

**示例：从NumPy创建Tensor**：

```cpp
Tensor tensor_from_numpy(py::array_t<float> data) {
    // 获取输入形状
    std::vector<int> shape(data.ndim());
    for (int i = 0; i < shape.size(); ++i) {
        shape[i] = data.shape(i);
    }

    // 创建张量
    Tensor tensor(shape);
    for (int i = 0; i < tensor.size(); ++i) {
        tensor[i] = data.data()[i];
    }

    return tensor;
}
```

**应用场景**：
- 使用NumPy加载图像数据
- 将NumPy数组转换为自定义Tensor
- 利用NumPy进行数据处理

**建议**：在Tensor中使用`std::shared_ptr`管理内存，便于在函数间传递和返回张量。

## 智能指针std::shared_ptr

### 共享指针的作用

**定义**：共享指针管理一个指针的存储，可能与其他对象共享该管理。

```cpp
std::shared_ptr<Pet> ptr(new Pet("Cat", {1, 2}));
```

**优点**：
- 可以将`ptr`赋值给其他智能指针
- 当没有引用指向初始对象时，内存自动释放
- 避免手动内存管理的错误

**在Tensor中的应用**：
- 实现一个`Blob`类管理内存
- 在Tensor中添加`Blob`的共享指针作为数据成员
- 可以自由地返回张量和接受张量作为参数

## 使用setup.py构建

### setup.py配置示例

以mesh2sdf项目为例：

```python
from pybind11.setup_helpers import Pybind11Extension, build_ext

__version__ = '1.1.0'

ext_modules = [
    Pybind11Extension(
        'mesh2sdf.core',
        ['csrc/pybind.cpp', 'csrc/makelevelset3.cpp'],
        include_dirs=['csrc'],
        define_macros=[('VERSION_INFO', __version__)],
    ),
]
```

**参考项目**：[mesh2sdf](https://github.com/wang-ps/mesh2sdf)

**注意**：使用setup.py构建CUDA代码相对复杂，需要特殊配置。

## 扩展PyTorch

### C++/CUDA扩展机制

C++扩展允许用户创建PyTorch外部定义的算子：

**应用场景**：
- 使用论文中发现的新激活函数
- 实现研究中开发的自定义操作
- 优化性能关键路径

**官方文档**：[PyTorch C++ Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)

### 实现前向传播函数

```cpp
#include <vector>
#include <torch/extension.h>

std::vector<torch::Tensor> cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias) {

    int64_t batch_size = data.size(0);
    int64_t channel = data.size(1);
    float* ptr_in = input.data_ptr<float>();

    torch::Tensor output = torch::zeros_like(input);

    // 在这里启动CUDA kernel

    return {output};
}
```

**关键API**：
- `torch::Tensor`：PyTorch张量类型
- `.size(dim)`：获取张量维度大小
- `.data_ptr<T>()`：获取底层数据指针
- `torch::zeros_like()`：创建相同形状的零张量

### 实现反向传播函数

```cpp
std::vector<torch::Tensor> cuda_backward(
    torch::Tensor grad_out,
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor bias) {

    torch::Tensor grad_in, grad_weights, grad_bias;

    // 实现反向传播逻辑

    return {grad_in, grad_weights, grad_bias};
}
```

### Pybind绑定

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cuda_forward", &cuda_forward, "forward (CUDA)");
    m.def("cuda_backward", &cuda_backward, "backward (CUDA)");
}
```

**注意**：`TORCH_EXTENSION_NAME`由setup.py提供。

### 使用CUDAExtension构建

```python
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

dir = './csrc'
sources = ['{}/{}'.format(dir, src) for src in os.listdir(dir)
           if src.endswith('.cpp') or src.endswith('.cu')]

setup(
    name='dwconv',
    version='1.1.0',
    packages=['dwconv'],
    include_package_data=True,
    zip_safe=False,
    install_requires=['torch', 'numpy', 'ocnn'],
    python_requires='>=3.8',
    ext_modules=[
        CUDAExtension(name='dwconv.core', sources=sources)
    ],
    cmdclass={'build_ext': BuildExtension},
)
```

**优势**：可以使用这个方法编译CUDA文件。

**参考项目**：[dwconv](https://github.com/octree-nn/dwconv)

## 包装前向和反向函数

### 使用torch.autograd.Function

```python
from torch.autograd import Function
from .core import forward_cuda, backward_cuda

class OurFunction(Function):
    @staticmethod
    def forward(ctx, data, weights):
        data = data.contiguous()
        out = forward_cuda(data, weights)
        ctx.save_for_backward(data, weights)
        return out

    @staticmethod
    def backward(ctx, grad):
        data, weights = ctx.saved_tensors
        grad_d, grad_w = backward_cuda(grad, data, weights)
        return grad_d, grad_w
```

**关键概念**：
- `@staticmethod`：静态方法装饰器
- `ctx.save_for_backward()`：保存前向传播中需要的张量用于反向传播
- `ctx.saved_tensors`：在反向传播中获取保存的张量

### Pybind总结

**核心工作流程**：

1. **绑定Tensor到Python**：
   - 使用`shared_ptr`智能管理指针
   - 将基于指针的函数包装为基于Tensor的函数
   - 使用pybind绑定函数到Python

2. **绑定NumPy到C++**：
   - 实现NumPy到Tensor的转换
   - 利用NumPy进行数据加载和处理

3. **实现Python包装类**：
   - 包装Tensor及其梯度的Python类
   - 包装前向和反向传播函数的Python类

4. **TODO**：
   - 实现计算图
   - 实现自动微分
   - 实现优化算法
   - 最终完成网络训练

## 单元测试

### 大型项目开发的挑战

**Bug的演化**：
- 项目开发过程中Bug数量先增后减
- 修复Bug时可能引入新Bug
- Bug修复成本随时间指数增长

**解决方案**：单元测试

### 单元测试的重要性

**核心理念**：
- 大型项目由众多小单元组成
- 确保每个单元的正确性
- 程序员必须对自己代码的质量负责
- 单元测试是对代码质量的基本承诺

### 单元测试的质量指标

**测试通过率（Test Pass Rate）**：
- 指测试过程中通过的测试用例比例
- 单元测试通常要求100%的测试用例通过率

**测试覆盖率（Test Coverage）**：
- 衡量测试完整性的手段
- 通过覆盖率数据了解测试是否充分
- 不能盲目追求覆盖率
- 包括：路径覆盖、if-else分支覆盖等

## 测试方法

### 黑盒测试（Black Box Testing）

**定义**：也称为功能测试，将测试对象视为黑盒，不考虑程序的内部逻辑结构和内部特性。

**特点**：
- 只检查程序功能是否满足需求规范
- 关注输入和输出的关系
- 不关心内部实现

**测试方法**：

**1. 等价类划分**：
- 将输入域划分为等价类
- 小于范围、在范围内、大于范围

**2. 边界值分析**：
- 大多数故障倾向于发生在输入或输出域的边界
- 测试边界值附近的情况

**3. 鲁棒性测试**：
- 添加略大于/小于最大值/最小值的值
- 检查超出极限值时系统的行为

### 白盒测试（White Box Testing）

**定义**：也称为结构测试，将测试对象视为透明盒，允许测试人员使用程序的内部逻辑结构和相关信息来设计或选择测试用例。

**目标**：
- 测试/覆盖所有路径、分支和逻辑
- 构造测试用例确保所有代码路径被执行

**示例**：

```cpp
double func1(int a, int b, double c) {
    if (a>0 && b>0) {     // ①
        c = c/a;          // ②
    }
    if (a>1 || c>1) {     // ③
        c = c+1;          // ④
    }
    c = b+c;              // ⑤
    return c;
}
```

**测试策略**：构造测试用例确保路径①-⑤都被覆盖，包括：
- 条件1为真和为假的情况
- 条件2为真和为假的情况
- 所有分支组合

## Python Unittest框架

### 基本使用

**继承unittest.TestCase**：

```python
import unittest
from mul import multiply

class MultiplyTestCase(unittest.TestCase):

    def test_multiplication_with_correct_values(self):
        self.assertEqual(multiply(5, 5), 25)

if __name__ == '__main__':
    unittest.main()
```

**关键要点**：
- 继承`unittest.TestCase`
- 测试方法名以`test_`开头
- 使用`self.assertEqual`判断测试是否通过

### 常用断言方法

```python
self.assertEqual(a, b)      # a == b
self.assertNotEqual(a, b)   # a != b
self.assertTrue(x)          # bool(x) is True
self.assertFalse(x)         # bool(x) is False
self.assertIs(a, b)         # a is b
```

### setUp和tearDown

```python
class MulTestCase(unittest.TestCase):

    def setUp(self):  # 在每个测试方法之前运行
        self.a = 10
        self.b = 20

    def test_mult_with_correct_values(self):
        self.assertEqual(multiply(self.a, self.b), 200)

    def tearDown(self):  # 在每个测试方法之后运行
        del self.a
        del self.b

if __name__ == '__main__':
    unittest.main()
```

**作用**：
- `setUp()`：初始化测试环境，每个测试方法前运行
- `tearDown()`：清理测试环境，每个测试方法后运行

## xUnit和Mock测试

### xUnit适用场景

xUnit通常适用于以下测试场景：
- 单个函数、类或几个功能相关类的测试
- 特别适合纯函数测试或接口级测试

### Mock测试

**定义**：使用虚拟对象（Mock对象）来模拟真实对象进行测试。

**使用场景**：
- 真实对象难以创建
- 真实对象具有用户界面
- 真实对象实际上不存在

**作用**：
- 隔离依赖
- 简化测试环境
- 提高测试可控性

## 课程前半部分总结

### 已学习的内容

**第1章 - 引言**：
- GPU并行计算的必要性
- 第一个PyTorch程序

**第2章 - 并行编程**：
- 线程、延迟和带宽
- ReLU、Sigmoid激活函数
- GPU内存模型：全局内存、共享内存等

**第3章 - 并行通信**：
- 并行线程交互：同步、原子操作等
- 内存一致性、GPU Stream

**第4章 - 并行算法I**：
- Reduce、Histogram、Scan、Compact

**第5章 - 并行算法II**：
- 分段扫描、转置、排序

**第6章 - 矩阵乘法**：
- 矩阵乘法和稀疏矩阵乘法
- 引入cuBLAS和Thrust
- 全连接层、GEMM

**第7章 - 卷积和池化**：
- 卷积、池化
- 损失函数、Softmax

**第8章 - 混合编程和单元测试**：
- Pybind、CMakeLists
- 扩展PyTorch
- 单元测试

## 深度学习框架的架构层次

### 三层架构

**底层：硬件特定后端**：
- CPU、GPU或移动处理器
- C++/CUDA或移动设备上的其他编程语言
- 并行编程思想的实现

**中层：脚本语言**：
- 计算图
- 自动微分
- 模型/数据并行

**系统层：分布式计算**：
- 跨机器/GPU训练大型模型
- 分布式训练策略

通过Pybind11，我们实现了底层高性能计算和上层灵活接口的无缝连接，这是现代深度学习框架的核心架构模式。

## 总结

### 混合编程的最佳实践

1. **使用Pybind11连接Python和C++**：
   - 简单的API设计
   - 自动类型转换
   - 高效的数据传递

2. **智能指针管理内存**：
   - 使用`std::shared_ptr`避免内存泄漏
   - 实现安全的对象生命周期管理

3. **扩展PyTorch**：
   - 实现自定义算子的前向和反向传播
   - 使用`torch.autograd.Function`集成到自动微分系统
   - 利用`CUDAExtension`简化构建流程

4. **单元测试保证质量**：
   - 100%的测试通过率
   - 合理的测试覆盖率
   - 黑盒测试和白盒测试结合

通过掌握这些技术，我们可以构建高性能、可维护的深度学习系统，充分发挥Python的灵活性和C++/CUDA的高效性。
