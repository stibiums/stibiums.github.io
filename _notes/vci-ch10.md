---
layout: post
title: "VCI - 10: 几何重建（Geometry Reconstruction）"
date: 2025-10-25 00:00:00
tags: notes vci geometry-reconstruction 3d-modeling
categories: vci
---

## 10.1 概述

**几何重建**（Geometry Reconstruction）是将采集的三维点云数据转化为可用的三维模型的过程，是三维计算机视觉和计算机图形学的核心问题。

### 重建的数据来源

三维点云通常来自：

1. **RGB-D 相机**（深度相机）

   - Intel RealSense、Kinect 等
   - 直接获得深度信息

2. **LiDAR 扫描仪**

   - 激光测距扫描
   - 高精度，长距离

3. **计算机视觉算法**
   - 三角化
   - 光束调整
   - 深度学习方法
   - 结构运动（SfM）

### 重建的目标

从点云到可用的几何表示：

$$\text{点云} \rightarrow \text{配准} \rightarrow \text{曲面重建} \rightarrow \text{模型拟合} \rightarrow \text{可用模型}$$

## 10.2 坐标变换基础

### 二维变换

#### 平移

点 $$(x, y)$$ 平移向量 $$(t_x, t_y)$$ 后得到：

$$\begin{pmatrix} x' \\ y' \end{pmatrix} = \begin{pmatrix} t_x \\ t_y \end{pmatrix} + \begin{pmatrix} x \\ y \end{pmatrix}$$

#### 旋转

点绕原点旋转角度 $$\theta$$ 后得到：

$$\begin{pmatrix} x' \\ y' \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix}$$

旋转矩阵 $$\mathbf{R}$$ 是正交矩阵，满足 $$\mathbf{R}^T \mathbf{R} = \mathbf{I}$$

### 三维变换

#### 旋转矩阵

三维旋转矩阵 $$\mathbf{R} \in SO(3)$$ 是 $$3 \times 3$$ 的正交矩阵：

$$\mathbf{R}^T \mathbf{R} = \mathbf{I}, \quad \det(\mathbf{R}) = 1$$

常见表示方法：

- **欧拉角**（Euler Angles）：绕三个坐标轴的旋转角
- **轴角**（Axis-Angle）：旋转轴 + 旋转角
- **四元数**（Quaternion）：$$q = (w, x, y, z)$$，紧凑且无奇异性
- **旋转矩阵**：直接表示，但有 9 个约束

#### 刚体变换

点 $$\mathbf{X}$$ 经过旋转 $$\mathbf{R}$$ 和平移 $$\mathbf{t}$$ 后：

$$\mathbf{X}' = \mathbf{R}\mathbf{X} + \mathbf{t}$$

齐次坐标表示：

$$\begin{pmatrix} \mathbf{X}' \\ 1 \end{pmatrix} = \begin{pmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0}^T & 1 \end{pmatrix} \begin{pmatrix} \mathbf{X} \\ 1 \end{pmatrix}$$

这个 $$4 \times 4$$ 矩阵称为**变换矩阵**或**SE(3) 变换**。

## 10.3 点云配准（Registration）

### 问题定义

给定两个点云 $$\mathcal{P} = \{\mathbf{p}_i\}$$ 和 $$\mathcal{Q} = \{\mathbf{q}_i\}$$，找到变换 $$(\mathbf{R}, \mathbf{t})$ 使两个点云对齐。

**输入：** 两个点云及其点对应关系

**输出：** 最优的刚体变换

**优化目标：**

$$\mathbf{R}^*, \mathbf{t}^* = \arg\min_{\mathbf{R}, \mathbf{t}} \sum_i \|\mathbf{q}_i - (\mathbf{R}\mathbf{p}_i + \mathbf{t})\|^2$$

### 迭代最近点算法（ICP）

**Iterative Closest Point** 是点云配准的标准算法，虽然简单但非常有效。基本思想是反复建立对应关系和优化变换，直到收敛。

#### ICP 算法的完整流程

```
输入：源点云 P = {p₁, ..., pₙ}，目标点云 Q = {q₁, ..., qₘ}
初始化：变换矩阵 T = I（通常为单位矩阵或粗配准结果）
设定参数：最大迭代数 max_iter，收敛阈值 ε

for k = 1 to max_iter:
  1. 最近点查找：对 P 中的每一点 pᵢ 找 Q 中的最近点
     correspondence[i] = argmin_j ||T·pᵢ - qⱼ||

  2. 计算对应集合：M = {(T·pᵢ, correspondence[i])}

  3. 计算最优变换：使下式最小
     T_new = argmin_T Σᵢ ||T·pᵢ - qcorr[i]||²

  4. 更新点云：P ← T_new · P，T ← T_new · T

  5. 计算误差和收敛条件：
     error = Σᵢ ||T·pᵢ - qcorr[i]||²
     if |error_prev - error| < ε or error < ε_abs:
       break
     error_prev = error

输出：最终变换矩阵 T，对应关系 correspondence
```

#### 最近点查找的高效实现

**1. KD 树（推荐）**

- 构建时间：$$O(n \log n)$$
- 单次查询时间：$$O(\log n)$$（平均），$$O(n)$$（最坏）
- 空间复杂度：$$O(n)$$
- 适合：点数不超过百万的点云

**2. 八叉树（Octree）**

- 适合动态点云更新
- 支持空间相关查询
- 时间复杂度：$$O(\log n)$$（平均）

**3. GPU 加速的 Brute Force**

- 对于点数较多（百万级）时快速
- 时间复杂度：$$O(nm)$，但并行化程度高
- 适合 GPU 内存足够的情况

**距离度量的选择：**

- **欧几里得距离**：$$d(\mathbf{p}, \mathbf{q}) = \|\mathbf{p} - \mathbf{q}\|_2$$（标准）
- **加权距离**：考虑点的不确定性
- **鲁棒距离**：使用 Huber 损失或其他M-估计量，去除异常值的影响

#### 最优变换的计算（Point-to-Point）

给定对应点对集合 $$M = \{(\mathbf{p}_i, \mathbf{q}_i)\}$$，求最优的刚体变换 $$(\mathbf{R}, \mathbf{t})$ 使得：

$$\min_{\mathbf{R}, \mathbf{t}} \sum_i \|\mathbf{R}\mathbf{p}_i + \mathbf{t} - \mathbf{q}_i\|^2$$

**步骤 1：计算质心**

$$\bar{\mathbf{p}} = \frac{1}{n}\sum_i \mathbf{p}_i, \quad \bar{\mathbf{q}} = \frac{1}{n}\sum_i \mathbf{q}_i$$

**步骤 2：中心化点**

$$\mathbf{p}_i' = \mathbf{p}_i - \bar{\mathbf{p}}, \quad \mathbf{q}_i' = \mathbf{q}_i - \bar{\mathbf{q}}$$

优化问题化为：$$\min_{\mathbf{R}} \sum_i \|\mathbf{R}\mathbf{p}_i' - \mathbf{q}_i'\|^2$$

**步骤 3：构造 $$3 \times 3$$ 协方差矩阵**

$$\mathbf{H} = \sum_i \mathbf{p}_i'^T \mathbf{q}_i' = \begin{pmatrix} \sum p_x q_x & \sum p_x q_y & \sum p_x q_z \\ \sum p_y q_x & \sum p_y q_y & \sum p_y q_z \\ \sum p_z q_x & \sum p_z q_y & \sum p_z q_z \end{pmatrix}$$

**步骤 4：SVD 分解**

对 $$\mathbf{H}$$ 进行奇异值分解：$$\mathbf{H} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$

**步骤 5：计算最优旋转矩阵**

$$\mathbf{R}^* = \mathbf{V} \mathbf{U}^T$$

若 $$\det(\mathbf{R}^*) < 0$$（反射），则翻转 $$\mathbf{V}$$ 的最后一列重新计算。

**步骤 6：计算最优平移**

$$\mathbf{t}^* = \bar{\mathbf{q}} - \mathbf{R}^* \bar{\mathbf{p}}$$

**时间复杂度：** $$O(n)$$（SVD 分解最贵，但矩阵只有 $$3 \times 3$$）

#### 变换计算

给定对应的点对 $$\{(\mathbf{p}_i, \mathbf{q}_i)\}$$，求最优变换：

**平移计算：**

计算质心：$$\bar{\mathbf{p}} = \frac{1}{n}\sum_i \mathbf{p}_i, \quad \bar{\mathbf{q}} = \frac{1}{n}\sum_i \mathbf{q}_i$$

最优平移：$$\mathbf{t}^* = \bar{\mathbf{q}} - \mathbf{R}^*\bar{\mathbf{p}}$$

**旋转计算：**

中心化点：$$\mathbf{p}_i' = \mathbf{p}_i - \bar{\mathbf{p}}, \quad \mathbf{q}_i' = \mathbf{q}_i - \bar{\mathbf{q}}$$

构造协方差矩阵：$$\mathbf{H} = \sum_i \mathbf{p}_i'^T \mathbf{q}_i'$$

通过 SVD 分解 $$\mathbf{H} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$$，得最优旋转：

$$\mathbf{R}^* = \mathbf{V}\mathbf{U}^T$$

#### ICP 的性质

**优点：**

- 简单易实现
- 收敛速度快
- 精度高（局部）

**缺点：**

- 对初始对齐敏感
- 易陷入局部极值
- 需要粗对齐初始化

**改进方案：**

- Point-to-plane ICP：使用平面法向
- Generalized ICP：考虑点的不确定性
- 特征加权 ICP：加权重要特征

## 10.4 曲面重建

### 概述

从点云生成连续的曲面是几何重建的核心。目标是：

1. 通过稀疏点云拟合光滑曲面
2. 填补缺失区域
3. 降低噪声

### 10.4.1 Delaunay 三角剖分

#### 定义

**Delaunay 三角剖分**是平面点集的三角分割，满足：外接圆空性质——任何三角形的外接圆内部不包含其他点。

#### 性质

- **唯一性**（不考虑共圆情况）
- **最大化最小角**：所有三角形中最小角最大
- **对偶性**：与 Voronoi 图对偶

#### 三维推广

三维 Delaunay 三角剖分的单元是四面体，满足外接球空性质。

#### 应用

- 快速点云表面重建
- 局限性：仅连接已有的点，不能填补孔洞

### 10.4.2 Poisson 曲面重建

#### 核心思想

Poisson 曲面重建的基本想法是利用点云的法向信息，通过求解 Poisson 方程将分散的点云转化为连续的曲面。

**关键观察：** 闭合曲面的特征是内外点的**有向距离函数**（SDF）在表面处为 0，且法向指向表面。

将问题表述为：从点法向对 $$\{(\mathbf{p}_i, \mathbf{n}_i)\}$$ 重建曲面，使得：

1. 重建出的曲面通过点云
2. 曲面法向与点云法向一致
3. 曲面是闭合的

#### 数学推导

**定义向量场：** 基于点云法向定义一个向量场 $$\mathbf{F}$$

在每个点 $$\mathbf{p}_i$$ 处，设定 $$\mathbf{F}(\mathbf{p}_i) = \mathbf{n}_i$$

通过插值扩展到整个空间。

**求解 Poisson 方程：**

设 $$\phi(\mathbf{x})$$ 为符号距离函数（SDF），其梯度方向与法向一致：

$$\nabla \phi = \mathbf{F}$$（理想情况）

实际问题中，由于数据噪声，不存在完全的解。转化为最小二乘问题：

$$\min_{\phi} \int_\Omega \|\nabla \phi - \mathbf{F}\|^2 d\mathbf{x}$$

使用分部积分，等价于求解 Poisson 方程：

$$\nabla^2 \phi = \nabla \cdot \mathbf{F}$$

其中 $$\nabla^2 = \Delta$$ 为 Laplacian 算子。

#### 关键步骤

**1. 法向估计**

对于点 $$\mathbf{p}_i$$，使用其 $$k$$ 近邻点的 PCA 估计法向：

- 构造协方差矩阵：$$\mathbf{C} = \sum_{j \in N_k(i)} (\mathbf{p}_j - \bar{\mathbf{p}})(\mathbf{p}_j - \bar{\mathbf{p}})^T$$
- 法向为最小特征值对应的特征向量
- 方向一致性处理（选择面向摄像机的法向）

**2. 隐函数生成**

在体素网格上定义 $$\mathbf{F}$，通过逐线性（或高阶）插值：
$$\mathbf{F}(\mathbf{x}) = \sum_i w_i(\mathbf{x}) \mathbf{n}\_i$$

其中 $$w_i(\mathbf{x})$$ 为 RBF（径向基函数）或其他权重函数。

**3. 求解 Poisson 方程**

在体素网格上离散化 Laplacian 算子，得到线性系统：
$$\mathbf{L} \phi = \nabla \cdot \mathbf{F}$$

使用稀疏求解器（如共轭梯度法）求解。

时间复杂度：$$O(n \log n)$$（对数因子来自 KD 树和多重网格法）

**4. 等值面提取**

使用 **Marching Cubes** 算法在 $$\phi(\mathbf{x}) = 0$$ 处提取三角网格。

#### 优缺点分析

**优点：**

- **鲁棒性强**：处理噪声数据效果好，自动去除异常值
- **自动填孔**：通过全局优化自动填补缺失部分
- **高质量网格**：生成的网格拓扑正确，无自交
- **处理复杂几何**：可以处理高度复杂的几何结构

**缺点：**

- **计算复杂**：需要求解大规模线性系统，计算时间长
- **内存消耗**：体素网格可能占用大量内存（三维网格）
- **分辨率权衡**：网格分辨率越高，精度越好但内存和时间消耗指数增长
- **法向依赖**：依赖于准确的法向估计，坏的法向会导致重建失败

#### 与其他方法的对比

| 方法           | 优点           | 缺点               | 适用场景       |
| -------------- | -------------- | ------------------ | -------------- |
| **Delaunay**   | 快速，局部     | 无孔填补，参数多   | 高保真场景     |
| **Poisson**    | 鲁棒，自动填孔 | 慢，内存多         | 有噪声、有孔洞 |
| **Ball Pivot** | 快速，易理解   | 参数敏感，易失败   | 高质量点云     |
| **深度学习**   | 端到端，速度快 | 需训练数据，泛化差 | 特定物体类别   |

## 10.5 模型拟合

### 概述

从点云中提取几何原语（如平面、球体、圆柱体），用于简化表示。

### 10.5.1 平面检测（Plane Detection）

#### 方法一：RANSAC 算法

**RANSAC**（RANdom SAmple Consensus）是处理包含异常值数据的鲁棒拟合的标准方法。基本思想是通过随机采样和一致性投票来找到最佳的几何模型。

**算法流程：**

```
输入：点集 P = {p₁, ..., pₙ}，距离阈值 τ，内点比例下界 α
初始化：最优平面 plane_best = null，内点数 num_best = 0

for iter = 1 to N_iterations:
  1. 随机采样 3 个不共线的点 S = {p₁, p₂, p₃}

  2. 从 3 个点估计平面方程 ax + by + cz + d = 0
     两个向量：v₁ = p₂ - p₁，v₂ = p₃ - p₁
     法向：n = v₁ × v₂
     归一化：n = n / ||n||
     平面方程系数：(a, b, c, d) = (n, -n·p₁)

  3. 计算内点集 I：
     for each point pᵢ in P:
       distance = |a·pᵢ.x + b·pᵢ.y + c·pᵢ.z + d|
       if distance < τ:
         I ← I ∪ {pᵢ}

  4. 更新最优解：
     if |I| > num_best:
       plane_best = estimated_plane
       num_best = |I|

  5. 动态更新迭代次数（可选）：
     inlier_ratio = num_best / n
     if inlier_ratio > α:
       break  // 找到足够好的模型

输出：最优平面 plane_best，内点集 I_best
```

**RANSAC 的关键参数分析：**

**1. 迭代次数 N**

设外点（异常值）比例为 $$e$$，为了以概率 $$p$$ 找到所有点都是内点的样本，需要：

$$N = \frac{\log(1-p)}{\log(1-(1-e)^s)}$$

其中 $$s=3$$（平面需要 3 个点）

**例子：** 若 $$p = 0.99$$（99% 信度），$$e = 0.5$$（50% 异常值）：
$$N = \frac{\log(0.01)}{\log(1-0.5^3)} = \frac{4.605}{0.875} \approx 5.27$$

需要约 6 次迭代。

若 $$e = 0.9$$（90% 异常值）：
$$N = \frac{\log(0.01)}{\log(1-0.1^3)} = \frac{4.605}{0.001} \approx 4605$$

需要约 4605 次迭代！

**2. 距离阈值 τ**

- **太小**：大量真内点被排除，估计不准
- **太大**：异常值被误当成内点，模型不准
- **经验值**：通常选为噪声标准差的 2-3 倍

**3. 内点数下界 α·n**

- 若点集 50% 以上是异常值，可能无解
- 通常要求至少 50-70% 是内点

**4. 动态终止**

若已找到内点数占 85% 的模型，再继续迭代概率很低，可以提前终止。

更新后的迭代次数：
$$N_{remaining} = \log(1-p) / \log(1-(1-e_{current})^s)$$

#### 优化的 RANSAC 变种

**改进方向：**

1. **Preemptive RANSAC**：提前中断

   - 动态减少迭代次数，找到好解后快速终止

2. **Lo-RANSAC**：局部优化

   - 每找到一个模型，用所有内点重新拟合
   - 提高精度

3. **M-RANSAC**：多模型
   - 检测多个平面，分别处理不同的几何体

#### RANSAC 的评价

**优点：**

- 处理异常值鲁棒性极强
- 实现简单，易于理解
- 不需要初值，完全自动

**缺点：**

- 迭代次数随外点比例指数增长
- 随机性导致结果不稳定
- 对距离阈值参数敏感
- 高维问题时指数增长（$$s$$ 很大）

**应用场景：**

- 点数较少、异常值多的情况（如激光扫描噪声）
- 多物体分割（每个物体是一个模型）

#### 方法二：Hough 变换

将点到平面参数空间的映射，统计参数空间的峰值。

### 10.5.2 其他几何原语拟合

- **球面拟合**：最小二乘法或 RANSAC
- **圆柱面拟合**：非线性优化
- **二次曲面拟合**：代数方法

### 应用场景

- **建筑物识别**：提取平面墙面
- **工业零件**：识别标准几何体
- **点云压缩**：用几何原语替代点

## 10.6 坐标变换详解

### 向量旋转的几何意义

在 2D 中，旋转矩阵：

$$\mathbf{R}(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

将点 $$(1, 0)$$ 变换为 $$(\cos\theta, \sin\theta)$$，即旋转 $$\theta$ 角。

### 欧拉角表示

使用三个旋转角（通常为 ZYX 顺序）表示旋转：

$$\mathbf{R} = \mathbf{R}_z(\gamma) \mathbf{R}_y(\beta) \mathbf{R}_x(\alpha)$$

**优点：** 直观，易于理解

**缺点：** 存在奇异性（万向锁），非线性

### 变换的组合

多个刚体变换的组合：

$$\mathbf{T}_{13} = \mathbf{T}_{12} \mathbf{T}_{23}$$

表示从坐标系 3 到坐标系 1 的变换。

## 10.7 实际应用

### 三维扫描与重建

**工作流程：**

1. 采集多个视角的深度图或图像
2. 点云配准（全局对齐）
3. 点云融合
4. 曲面重建
5. 网格优化

**应用：**

- 文物数字化
- 建筑测量
- 工业检测

### 自动驾驶感知

- 实时点云处理
- 障碍物检测
- 道路分割

### 机器人视觉

- 物体识别和抓取
- 环境建图
- 路径规划

### 增强现实

- 场景重建
- 虚实融合
- 光照估计

## 10.8 常用开源工具

| 工具                      | 功能           | 语言       |
| ------------------------- | -------------- | ---------- |
| PCL (Point Cloud Library) | 全面的点云处理 | C++        |
| Open3D                    | 三维数据处理   | Python/C++ |
| CloudCompare              | 可视化和分析   | C++        |
| Meshlab                   | 网格处理       | C++        |
| COLMAP                    | SfM 重建       | C++        |

## 10.9 总结

几何重建是三维计算机视觉的重要环节，将原始点云转化为可用的几何表示。关键技术包括：

1. **坐标变换**：刚体变换的表示和组合
2. **点云配准**：多个点云的对齐（ICP 等）
3. **曲面重建**：从点到面的过渡（Delaunay、Poisson）
4. **模型拟合**：提取几何原语（RANSAC）
5. **应用**：三维重建、robotics、AR/VR 等

现代方法结合传统算法和深度学习，在精度和效率上都取得了显著进步。
