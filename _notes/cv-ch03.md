---
layout: post
title: "CV - 3: 图像处理 (Image Processing)"
date: 2025-09-18 03:00:00
tags: notes CV computer-vision image-processing
categories: CV
---

> **授课**: 王鹏帅 (Peng-Shuai Wang)
> **课程**: 北京大学计算机视觉2025秋季课程
> **内容**: 图像处理基础理论与实用技术

## 目录

1. [图像处理基础概念](#1-图像处理基础概念)
2. [线性滤波技术](#2-线性滤波技术)
3. [非线性滤波技术](#3-非线性滤波技术)
4. [图像金字塔与采样](#4-图像金字塔与采样)
5. [图像变换技术](#5-图像变换技术)

---

## 1. 图像处理基础概念

### 1.1 像素级操作 (Point Operations)

像素级操作是最基础的图像处理技术，对每个像素独立进行变换，不依赖于邻域信息。

#### 亮度和对比度调整

**线性变换**：

$$I'(x,y) = a \times I(x,y) + b$$

- **参数含义**：
  - `a`: 对比度系数 (a > 1增加对比度，0 < a < 1降低对比度)
  - `b`: 亮度偏移 (b > 0增加亮度，b < 0降低亮度)

**实际应用**：

- 照片曝光校正
- 医学图像增强
- 显示设备适配

#### Gamma 校正

**非线性亮度调整**：

$$I'(x,y) = c \times I(x,y)^\gamma$$

- **Gamma值影响**：
  - `γ > 1`: 暗部变更暗，亮部变化小 (增强对比度)
  - `γ < 1`: 暗部变亮，整体变亮 (适合暗环境显示)
  - `γ = 1`: 线性变换

**应用场景**：

- CRT显示器校正
- 图像打印预处理
- HDR图像处理

### 1.2 直方图处理

直方图反映了图像中各灰度级的分布情况，是图像统计特性的重要表示。

#### 直方图均衡化 (Histogram Equalization)

**目标**：将图像直方图变换为均匀分布，最大化图像对比度。

**变换函数**：

$$s = T(r) = (L-1) \times \sum_{i=0}^{k} p_r(r_i) = (L-1) \times CDF(r)$$

**算法步骤**：

1. 计算原图像直方图 `h(r_k)`
2. 计算概率密度 $p_r(r_k) = h(r_k)/(M \times N)$
3. 计算累积分布函数 `CDF(r_k)`
4. 应用变换函数得到新灰度值

**效果**：

- 增强全局对比度
- 可能产生伪影
- 适合低对比度图像

#### 直方图匹配 (Histogram Specification)

**目标**：将图像直方图变换为指定的目标分布。

**双步骤法**：

1. 对原图像进行直方图均衡化：`s = T(r)`
2. 对目标直方图进行均衡化：`v = G(z)`
3. 应用逆变换：`z = G^(-1)(s)`

**应用**：

- 图像风格转换
- 多图像一致性处理
- 特定效果模拟

---

## 2. 线性滤波技术

### 2.1 卷积与相关基础

线性滤波是通过卷积或相关运算实现的空间域处理技术。

#### 数学定义

**卷积 (Convolution)**：

$$(f * h)(x,y) = \sum_u \sum_v f(u,v) \times h(x-u, y-v)$$

**相关 (Correlation)**：

$$(f \otimes h)(x,y) = \sum_u \sum_v f(u,v) \times h(x+u, y+v)$$

**关键区别**：

- 卷积：滤波器需要旋转180°
- 相关：滤波器不旋转
- 对称滤波器：两者结果相同

#### 边界处理策略

处理图像边界的常用方法：

1. **零填充 (Zero Padding)**

   - 边界外像素值设为0
   - 简单但可能产生边界伪影

2. **复制填充 (Replicate)**

   - 复制最近的边界像素值
   - 适合自然图像

3. **反射填充 (Reflect)**

   - 以边界为轴镜像反射
   - 保持图像连续性

4. **循环填充 (Wrap)**

   - 将图像视为周期性延拓
   - 适合纹理图像

5. **裁剪 (Valid)**
   - 只计算完全在图像内的区域
   - 输出图像尺寸缩小

### 2.2 可分离滤波器

**定义**：如果2D滤波器可以表示为两个1D滤波器的外积，则称为可分离滤波器。

$$H(x,y) = h_x(x) \times h_y(y)$$

**计算优势**：

- 复杂度从 O(M²N²) 降至 O(MN(M+N))
- 内存需求大幅降低

**常见可分离滤波器**：

- 高斯滤波器
- Box滤波器
- Sobel算子

### 2.3 经典线性滤波器

#### 高斯滤波器

**2D高斯函数**：

$$G(x,y) = \frac{1}{2\pi\sigma^2} \times \exp\left(-\frac{x^2+y^2}{2\sigma^2}\right)$$

**离散化高斯核**：

$\sigma=1$时的5×5高斯核：

$$
\frac{1}{256} \begin{bmatrix}
1 & 4 & 6 & 4 & 1 \\
4 & 16 & 24 & 16 & 4 \\
6 & 24 & 36 & 24 & 6 \\
4 & 16 & 24 & 16 & 4 \\
1 & 4 & 6 & 4 & 1
\end{bmatrix}
$$

**特性**：

- 各向同性平滑
- 可分离性
- σ参数控制平滑程度
- 频域为低通特性

#### 导数滤波器

**Sobel算子**：

$$G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix} \quad G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$

**梯度计算**：

梯度幅值: $|\nabla I| = \sqrt{G_x^2 + G_y^2}$
梯度方向: $\theta = \arctan(G_y/G_x)$

**Prewitt算子**：

$$G_x = \begin{bmatrix} -1 & 0 & 1 \\ -1 & 0 & 1 \\ -1 & 0 & 1 \end{bmatrix} \quad G_y = \begin{bmatrix} -1 & -1 & -1 \\ 0 & 0 & 0 \\ 1 & 1 & 1 \end{bmatrix}$$

**Roberts算子**：

$$G_x = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix} \quad G_y = \begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix}$$

#### 拉普拉斯算子

**连续形式**：

$$\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}$$

**离散近似**：

4-连通拉普拉斯: $\begin{bmatrix} 0 & -1 & 0 \\\\ -1 & 4 & -1 \\\\ 0 & -1 & 0 \end{bmatrix}$

8-连通拉普拉斯: $\begin{bmatrix} -1 & -1 & -1 \\\\ -1 & 8 & -1 \\\\ -1 & -1 & -1 \end{bmatrix}$

**特点**：

- 二阶导数算子
- 对噪声敏感
- 产生双边缘效应
- 常用于边缘增强

---

## 3. 非线性滤波技术

### 3.1 中值滤波器

中值滤波是经典的非线性滤波技术，特别适合去除脉冲噪声。

**算法原理**：

1. 提取邻域内所有像素值
2. 按大小排序
3. 选取中位数作为输出

**伪代码**：

```python
def median_filter(image, kernel_size):
    for each pixel (x, y):
        neighborhood = extract_neighborhood(image, x, y, kernel_size)
        sorted_values = sort(neighborhood)
        output[x, y] = sorted_values[middle_index]
```

**优势**：

- 完全去除椒盐噪声
- 保持边缘信息
- 不引入新的像素值

**局限性**：

- 计算复杂度高
- 可能改变图像细节
- 对高斯噪声效果有限

### 3.2 双边滤波器 (Bilateral Filter)

双边滤波器同时考虑空间距离和像素值相似性，实现保边去噪。

**数学表达式**：

$$BF[I]_p = \frac{1}{W_p} \times \sum_{q \in S} G_{\sigma_s}(\lVert p-q \rVert) \times G_{\sigma_r}(\lvert I_p - I_q \rvert) \times I_q$$

**权重函数**：

空间权重: $w_s(p,q) = \exp\left(-\frac{||p-q||^2}{2\sigma_s^2}\right)$
值域权重: $w_r(p,q) = \exp\left(-\frac{|I_p-I_q|^2}{2\sigma_r^2}\right)$
综合权重: $w(p,q) = w_s(p,q) \times w_r(p,q)$

**参数调节**：

- `σ_s`：控制空间平滑范围 (越大平滑区域越大)
- `σ_r`：控制边缘保持程度 (越小边缘保持越好)

**应用场景**：

- 人像照片美化
- 医学图像去噪
- HDR图像处理

### 3.3 形态学滤波

形态学滤波基于集合论，专门处理二值图像和灰度图像的形状特征。

#### 基本操作

**膨胀 (Dilation)**：

$$(A \oplus B)(x,y) = \max\{f(x-s, y-t) + b(s,t) | (s,t) \in D_b\}$$

- 扩展亮区域
- 填充小孔洞

**腐蚀 (Erosion)**：

$$(A \ominus B)(x,y) = \min\{f(x+s, y+t) - b(s,t) | (s,t) \in D_b\}$$

- 缩小亮区域
- 去除小噪点

#### 复合操作

**开运算 (Opening)**：

$$A \circ B = (A \ominus B) \oplus B$$

- 先腐蚀后膨胀
- 去除小亮区域
- 平滑物体轮廓

**闭运算 (Closing)**：

$$A \bullet B = (A \oplus B) \ominus B$$

- 先膨胀后腐蚀
- 填充小暗区域
- 连接断裂部分

---

## 4. 图像金字塔与采样

### 4.1 图像金字塔概念

图像金字塔是多尺度图像表示的重要工具，在计算机视觉中有广泛应用。

#### 高斯金字塔 (Gaussian Pyramid)

**构建过程**：

1. 原图像 G₀ = I
2. 高斯平滑：G'*l = G_l \* G*σ
3. 下采样：G\_{l+1} = downsample(G'\_l, 2)

**数学表示**：

$$G_l(x,y) = \sum_m \sum_n w(m,n) \times G_{l-1}(2x+m, 2y+n)$$

**特点**：

- 每层图像尺寸减半
- 保持图像主要特征
- 计算效率高

#### 拉普拉斯金字塔 (Laplacian Pyramid)

**构建原理**：

$$L_l = G_l - \text{expand}(G_{l+1})$$

其中 expand 表示上采样操作。

**重建过程**：

$$G_l = L_l + \text{expand}(G_{l+1})$$

**应用优势**：

- 无损表示 (理论上)
- 适合图像压缩
- 支持多尺度分析

### 4.2 采样理论

#### Nyquist-Shannon采样定理

**定理内容**：为了无失真地重建连续信号，采样频率必须至少是信号最高频率的两倍。

$$f_s \geq 2 \times f_{max}$$

**在图像中的意义**：

- 图像包含各种频率成分
- 下采样可能导致混叠 (Aliasing)
- 需要预先低通滤波

#### 混叠现象

**产生原因**：

- 采样频率不足
- 高频信息折叠到低频

**视觉表现**：

- 摩尔纹 (Moiré patterns)
- 锯齿边缘
- 细节丢失

**防止方法**：

1. 增加采样率
2. 采样前低通滤波 (抗混叠滤波)
3. 使用更好的插值方法

### 4.3 插值方法

#### 最近邻插值 (Nearest Neighbor)

**原理**：选择最近的已知像素值。

**特点**：

- 计算最简单
- 保持原始像素值
- 产生锯齿效应

#### 双线性插值 (Bilinear)

**公式**：

$$f(x,y) = f(0,0)(1-x)(1-y) + f(1,0)x(1-y) + f(0,1)(1-x)y + f(1,1)xy$$

**特点**：

- 平滑的插值结果
- 计算适中
- 轻微模糊

#### 双三次插值 (Bicubic)

**使用4$\times$4邻域进行三次多项式插值**

**特点**：

- 最平滑的结果
- 计算复杂度最高
- 最佳视觉效果

---

## 5. 图像变换技术

### 5.1 几何变换

几何变换改变图像中像素的空间位置关系，包括平移、旋转、缩放等。

#### 基本变换

**平移 (Translation)**：

$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

**旋转 (Rotation)**：

$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \begin{bmatrix} \cos \theta & -\sin \theta & 0 \\ \sin \theta & \cos \theta & 0 \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

**缩放 (Scaling)**：

$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \begin{bmatrix} s_x & 0 & 0 \\ 0 & s_y & 0 \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

#### 复合变换

**仿射变换 (Affine)**：

$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \begin{bmatrix} a & b & c \\ d & e & f \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

**保持的性质**：

- 平行线保持平行
- 直线保持直线
- 平行线段长度比保持不变

**透视变换 (Perspective)**：

$$\begin{bmatrix} x' \\ y' \\ w' \end{bmatrix} = \begin{bmatrix} a & b & c \\ d & e & f \\ g & h & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

实际坐标：$x = x'/w', y = y'/w'$

### 5.2 图像配准与拼接

#### 特征点匹配

**SIFT特征**：

- 尺度不变
- 旋转不变
- 光照鲁棒

**匹配流程**：

1. 特征点检测
2. 特征描述符计算
3. 特征匹配
4. 变换估计

#### RANSAC算法

用于鲁棒地估计变换参数：

**算法步骤**：

1. 随机选择最小样本集
2. 估计模型参数
3. 计算内点数量
4. 保留最优模型
5. 重复直到收敛

### 5.3 图像融合

#### 拉普拉斯金字塔融合

**算法流程**：

1. 构建两幅图像的拉普拉斯金字塔
2. 构建掩模的高斯金字塔
3. 在每个金字塔层级进行加权融合
4. 重建最终图像

**优势**：

- 避免接缝伪影
- 保持多尺度细节
- 自然的过渡效果

#### 泊松融合 (Poisson Blending)

**目标函数**：

$$\min \iint_{\Omega} \lvert \nabla f - \nabla g \rvert^2 \, dx \, dy$$

**边界条件**：

$$f|_{\partial\Omega} = f^*|_{\partial\Omega}$$

**特点**：

- 基于梯度域处理
- 无缝融合效果
- 保持源图像细节

---

## 小结

本章介绍了图像处理的核心技术：

### 基础技术

1. **像素级操作**：亮度/对比度调整、Gamma校正
2. **直方图处理**：均衡化、匹配、分析

### 滤波技术

1. **线性滤波**：卷积、高斯、导数滤波器
2. **非线性滤波**：中值、双边、形态学滤波

### 多尺度技术

1. **图像金字塔**：高斯金字塔、拉普拉斯金字塔
2. **采样理论**：Nyquist定理、混叠、插值

### 变换技术

1. **几何变换**：仿射变换、透视变换
2. **图像配准**：特征匹配、RANSAC
3. **图像融合**：金字塔融合、泊松融合

这些技术构成了现代计算机视觉系统的基础，为目标检测、图像分割、立体视觉等高级应用提供了重要支撑。

---

_课程信息：北京大学计算机视觉 - 王鹏帅老师_
