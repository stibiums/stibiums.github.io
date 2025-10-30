---
layout: post
title: "VCI - 11: 几何变换 (Geometric Transformations)"
date: 2025-10-30 01:00:00
tags: notes VCI graphics transformations matrices kinematics
categories: VCI
---

> **课程**: 北京大学视觉计算 (Visual Computing) 2025秋季
> **章节**: 第11章 几何变换
> **内容**: 2D/3D变换、齐次坐标、运动学、视图变换

## 目录

1. [基础变换](#1-基础变换)
2. [齐次坐标](#2-齐次坐标)
3. [3D变换](#3-3d变换)
4. [欧拉角与轴角旋转](#4-欧拉角与轴角旋转)
5. [变换层次结构](#5-变换层次结构)
6. [运动学](#6-运动学)
7. [视图变换](#7-视图变换)
8. [OpenGL中的变换](#8-opengl中的变换)

---

## 1. 基础变换

### 1.1 2D变换

2D几何变换是图形学的基础，主要包括以下几种：

#### 平移 (Translation)

将点 $(x, y)$ 沿向量 $(t_x, t_y)$ 平移：

$$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} x \\ y \end{bmatrix} + \begin{bmatrix} t_x \\ t_y \end{bmatrix}$$

平移不是线性变换（因为 $T(0) \neq 0$），需要使用齐次坐标转化为线性变换。

#### 旋转 (Rotation)

绕原点旋转 $\theta$ 角：

$$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix}$$

旋转矩阵是正交矩阵，其逆矩阵等于转置。

#### 缩放 (Scaling)

沿坐标轴缩放：

$$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix}$$

其中 $s_x, s_y$ 是缩放因子。

#### 反射 (Reflection)

关于x轴反射：

$$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix}$$

#### 剪切变换 (Shear)

沿x方向剪切：

$$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} 1 & sh_x \\ 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix}$$

---

## 2. 齐次坐标

### 2.1 齐次坐标概念

齐次坐标是计算机图形学中最重要的概念之一。通过在坐标系统中添加额外的维度，我们可以将所有变换（包括平移）统一为矩阵乘法。

**点与向量的区分**：

- 点 (Point): $\begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$ — 表示空间中的位置，齐次坐标最后一项为1
- 向量 (Vector): $\begin{bmatrix} x \\ y \\ 0 \end{bmatrix}$ — 表示方向和大小，齐次坐标最后一项为0

**基本性质**：

- 点 + 向量 = 点
- 点 - 点 = 向量
- 向量 + 向量 = 向量
- 点 + 点 = ？（一般不定义，除非使用加权形式）

### 2.2 2D齐次坐标变换矩阵

在齐次坐标下，所有2D变换都可以用 $3 \times 3$ 矩阵表示：

**平移** (齐次坐标形式)：

$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

**旋转** (齐次坐标形式)：

$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \begin{bmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

**缩放** (齐次坐标形式)：

$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \begin{bmatrix} s_x & 0 & 0 \\ 0 & s_y & 0 \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

### 2.3 齐次坐标的一般形式

一般的齐次坐标 $(x, y, w)$（$w \neq 0$）代表3D空间中的点 $(x/w, y/w)$：

$$\begin{bmatrix} x/w \\ y/w \end{bmatrix} = \begin{bmatrix} x \\ y \\ w \end{bmatrix}$$

这种表示允许不同的缩放因子，在投影变换中非常有用。

---

## 3. 3D变换

### 3.1 3D坐标系统

计算机图形学中常用**右手坐标系**：

- X轴向右
- Y轴向上
- Z轴指向观察者（或者相反，取决于约定）

Z轴可以通过叉积确定：$\vec{Z} = \vec{X} \times \vec{Y}$

### 3.2 3D基础变换

使用齐次坐标，3D变换用 $4 \times 4$ 矩阵表示。

**3D平移**：

$$\begin{bmatrix} x' \\ y' \\ z' \\ 1 \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 & t_x \\ 0 & 1 & 0 & t_y \\ 0 & 0 & 1 & t_z \\ 0 & 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}$$

**3D缩放**：

$$\begin{bmatrix} x' \\ y' \\ z' \\ 1 \end{bmatrix} = \begin{bmatrix} s_x & 0 & 0 & 0 \\ 0 & s_y & 0 & 0 \\ 0 & 0 & s_z & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}$$

### 3.3 3D旋转

#### 绕坐标轴旋转

**绕X轴旋转** (pitch)：

$$R_x(\theta) = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & \cos\theta & -\sin\theta & 0 \\ 0 & \sin\theta & \cos\theta & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

**绕Y轴旋转** (yaw)：

$$R_y(\theta) = \begin{bmatrix} \cos\theta & 0 & \sin\theta & 0 \\ 0 & 1 & 0 & 0 \\ -\sin\theta & 0 & \cos\theta & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

**绕Z轴旋转** (roll)：

$$R_z(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta & 0 & 0 \\ \sin\theta & \cos\theta & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

---

## 4. 欧拉角与轴角旋转

### 4.1 欧拉角 (Euler Angles)

欧拉角使用三个连续的旋转来表示任意3D旋转：

- **Heading/Yaw**: 绕Z轴旋转 ($R_z$)
- **Pitch**: 绕X轴旋转 ($R_x$)
- **Roll**: 绕Y轴旋转 ($R_y$)

复合旋转：$R = R_z(\gamma) R_y(\beta) R_x(\alpha)$

**欧拉角的问题**：

1. **角度插值不线性** — 两个欧拉角之间的插值会产生不自然的运动
2. **旋转顺序相关** — 不同的旋转顺序会产生不同的结果
3. **万向锁 (Gimbal Lock)** — 当中间轴旋转90度时，会失去一个自由度

### 4.2 轴角旋转 (Axis-Angle Rotation)

轴角旋转用单位向量 $\vec{a}$ 和旋转角 $\theta$ 表示，更适合插值和平滑旋转。

**罗德里格斯公式 (Rodrigues' Rotation Formula)**：

$$\vec{x}' = \cos\theta \cdot \vec{x} + (1-\cos\theta)(\vec{x} \cdot \vec{a})\vec{a} + \sin\theta(\vec{a} \times \vec{x})$$

**矩阵形式**：

$$\mathbf{R} = \cos\theta \mathbf{I} + (1-\cos\theta)\vec{a}\vec{a}^T + \sin\theta[\vec{a}]_\times$$

其中 $[\vec{a}]_\times$ 是反对称矩阵（叉积矩阵）：

$$[\vec{a}]_\times = \begin{bmatrix} 0 & -a_z & a_y \\ a_z & 0 & -a_x \\ -a_y & a_x & 0 \end{bmatrix}$$

这个公式的三个分量：

- $\cos\theta \mathbf{I}$ — 在垂直于旋转轴的平面内旋转
- $(1-\cos\theta)\vec{a}\vec{a}^T$ — 沿旋转轴方向的投影（保持不变）
- $\sin\theta[\vec{a}]_\times$ — 实现垂直于轴的旋转

---

## 5. 变换层次结构

### 5.1 仿射变换 (Affine Transformations)

仿射变换是一般形式的线性变换加平移：

$$\begin{bmatrix} x' \\ y' \\ z' \\ 1 \end{bmatrix} = \begin{bmatrix} \mathbf{A} & \vec{b} \\ 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}$$

其中 $\mathbf{A}$ 是 $3 \times 3$ 线性变换矩阵，$\vec{b}$ 是平移向量。

**性质**：

- 保持平行线平行
- 保持共线点仍然共线
- 不一定保持角度和长度

### 5.2 变换层次结构的应用

在建模复杂的对象（如人体骨架、机械臂）时，使用层次结构非常有效：

1. **根节点** — 代表整个对象的位置和方向
2. **内部节点** — 代表关节，包含相对于父节点的变换
3. **叶子节点** — 代表几何图元（形状）

通过这种方式，改变父节点的变换会自动影响所有子节点。

**有向无环图 (Directed Acyclic Graph, DAG)** — 允许多个父节点但避免循环引用。

---

## 6. 运动学

### 6.1 前向运动学 (Forward Kinematics)

给定关节角度和静止姿态，计算末端效应器的位置。

**递推关系**：

对于链式结构，第 $i$ 个关节的位置为：

$$\vec{P}_i = \vec{P}_{i-1} + \mathbf{R}(\vec{q}_{i-1}) \vec{v}_{i-1 \sim i}$$

其中：

- $\vec{P}_{i-1}$ — 上一个关节的位置
- $\mathbf{R}(\vec{q}_{i-1})$ — 上一个关节的旋转矩阵
- $\vec{v}_{i-1 \sim i}$ — 局部坐标系中两关节间的向量

**优点**：

- 计算快速
- 直观的控制参数（关节角度）

**缺点**：

- 无法直接指定末端位置（需要反向运动学）

### 6.2 逆向运动学 (Inverse Kinematics)

给定目标位置，求解关节角度。这是一个更复杂的问题，通常需要数值求解。

$$\vec{\theta} = f^{-1}(\vec{P}_{target}, \text{rest\_shape})$$

**挑战**：

- 解可能不存在（超出可达范围）
- 解可能不唯一（多个姿态达到同一目标）
- 计算复杂，需要迭代优化

---

## 7. 视图变换

### 7.1 相机坐标系统

将世界坐标系中的点变换到相机坐标系统中。

**相机参数**：

- **Eye** ($\vec{e}$) — 相机位置（世界坐标）
- **LookAt** ($\vec{a}$) — 相机指向的点（世界坐标）
- **Up** ($\vec{u}$) — 相机的上方向（世界坐标）

**相机坐标轴**（通过正交化得到）：

- $\vec{z} = \frac{\vec{e} - \vec{a}}{|\vec{e} - \vec{a}|}$ — 观察方向（指向相机）
- $\vec{x} = \frac{\vec{u} \times \vec{z}}{|\vec{u} \times \vec{z}|}$ — 右方向
- $\vec{y} = \vec{z} \times \vec{x}$ — 上方向

### 7.2 视图变换矩阵

将点从世界坐标变换到相机坐标的 $4 \times 4$ 矩阵：

$$\mathbf{V} = \begin{bmatrix} x_x & x_y & x_z & -\vec{e} \cdot \vec{x} \\ y_x & y_y & y_z & -\vec{e} \cdot \vec{y} \\ z_x & z_y & z_z & -\vec{e} \cdot \vec{z} \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

其中 $(x_x, x_y, x_z) = \vec{x}$，以此类推。

### 7.3 投影变换

#### 正交投影 (Orthographic Projection)

将立方体 $[l, r] \times [b, t] \times [n, f]$ 映射到标准立方体 $[-1, 1]^3$：

$$\mathbf{M}_{ortho} = \begin{bmatrix} \frac{2}{r-l} & 0 & 0 & -\frac{r+l}{r-l} \\ 0 & \frac{2}{t-b} & 0 & -\frac{t+b}{t-b} \\ 0 & 0 & \frac{2}{f-n} & -\frac{f+n}{f-n} \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

**性质**：

- 平行线保持平行
- 距离比例保持不变
- 不会产生透视效果

#### 透视投影 (Perspective Projection)

模拟人眼和相机的透视效果。视锥体被映射为标准立方体。

关键步骤：

1. 将视锥体变形，使得平行线收敛到相机点
2. 映射到标准立方体 $[-1, 1]^3$

对于点 $(x, y, z)$，投影后为 $(\frac{nx}{z}, \frac{ny}{z}, z')$

深度值的计算（为了保持线性的深度缓冲）：

$$z' = \frac{(n+f)z - nf}{(f-n)z}$$

当 $z = n$ 时，$z' = -1$；当 $z = f$ 时，$z' = 1$。

---

## 8. OpenGL中的变换

### 8.1 OpenGL变换管线

OpenGL使用两个主要的矩阵栈：

- **GL_MODELVIEW** — 模型变换 (世界坐标) + 视图变换 (相机坐标)
- **GL_PROJECTION** — 投影变换 (投影坐标)

```glsl
glMatrixMode(GL_PROJECTION);
glLoadIdentity();
gluPerspective(50.0, 1.0, 3.0, 7.0);  // 透视投影

glMatrixMode(GL_MODELVIEW);
glLoadIdentity();
gluLookAt(0.0, 0.0, 5.0,   // Eye
          0.0, 0.0, 0.0,   // LookAt
          0.0, 1.0, 0.0);  // Up
```

### 8.2 常用OpenGL变换函数

- `glTranslate(tx, ty, tz)` — 平移
- `glRotate(angle, vx, vy, vz)` — 绕任意轴旋转（轴角表示）
- `glScale(sx, sy, sz)` — 缩放
- `glPushMatrix()` / `glPopMatrix()` — 矩阵栈操作
- `glOrtho(l, r, b, t, n, f)` — 正交投影
- `gluPerspective(fov, aspect, near, far)` — 透视投影
- `gluLookAt(ex, ey, ez, ax, ay, az, ux, uy, uz)` — 设置相机视图

### 8.3 矩阵栈与层次结构

在绘制复杂对象时，矩阵栈用于管理变换层次：

```cpp
glLoadIdentity();

// 肩关节变换
glTranslatef(Tx, Ty, 0);
glRotatef(u, 0, 0, 1);  // 肩膀旋转
glTranslatef(-px, -py, 0);

glPushMatrix();

// 肘关节变换（只影响前臂）
glTranslatef(qx, qy, 0);
glRotatef(v, 0, 0, 1);  // 肘关节旋转
glTranslatef(-rx, -ry, 0);
Draw(A);  // 绘制前臂

glPopMatrix();

Draw(B);  // 绘制上臂
```

这种方式确保：

1. 下级关节的变换依赖于上级关节
2. 改变肩膀角度会自动移动整个手臂
3. 改变肘关节角度只影响前臂

---

## 总结

几何变换是计算机图形学的基础：

1. **齐次坐标** — 统一了线性变换和平移，是现代图形学的关键
2. **矩阵表示** — 所有变换可组合，支持高效计算
3. **层次结构** — 使复杂模型的建模和动画成为可能
4. **运动学** — 前向运动学用于骨架动画，反向运动学用于目标驱动的姿态
5. **视图变换** — 实现观察者相机的灵活控制
6. **投影变换** — 是3D到2D投影的数学基础

这些概念贯穿整个计算机图形学，是理解后续渲染、着色等高级主题的基础。
