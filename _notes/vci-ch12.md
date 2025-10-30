---
layout: post
title: "VCI - 12: 高级渲染 (Advanced Rendering)"
date: 2025-10-30 02:00:00
tags: notes VCI graphics rendering equation BRDF ray-tracing
categories: VCI
---

> **课程**: 北京大学视觉计算 (Visual Computing) 2025秋季
> **章节**: 第12章 高级渲染
> **内容**: 渲染方程、BRDF、微表面模型、纹理映射、光线追踪

## 目录

1. [光线追踪基础](#1-光线追踪基础)
2. [渲染方程](#2-渲染方程)
3. [BRDF - 双向反射分布函数](#3-brdf---双向反射分布函数)
4. [微表面BRDF模型](#4-微表面brdf模型)
5. [纹理映射](#5-纹理映射)
6. [环境和反射映射](#6-环境和反射映射)
7. [光线追踪在生产中的应用](#7-光线追踪在生产中的应用)

---

## 1. 光线追踪基础

### 1.1 光线追踪概述

光线追踪 (Ray Tracing) 是一种基于物理的渲染方法，通过模拟光线在场景中的传播来计算逼真的图像。与光栅化不同，光线追踪能够自然地处理：

- 软阴影 (Soft Shadows)
- 反射 (Reflections)
- 折射 (Refraction)
- 全局光照 (Global Illumination)

### 1.2 基本原理

光线追踪的核心思想：

1. **从相机发出光线** — 对每个像素发出一条主光线
2. **追踪光线** — 计算光线与场景的交点
3. **着色计算** — 在交点处计算光线的贡献
4. **递归追踪** — 沿反射/折射方向继续追踪

### 1.3 光线追踪的应用

**视觉特效**：

- Ex Machina (2014) — 机器人角色渲染
- Gravity (2013) — 空间场景光照
- The Lord of the Rings (2003) — 复杂场景渲染

**视频游戏**：

- Assassin's Creed Odyssey (2018)
- Black Myth: Wukong (2024)
- Elden Ring (2022)
- 实时光线追踪通过GPU加速实现

**科学可视化**：

- 物理模拟结果可视化
- 医学影像渲染
- 建筑光照分析

---

## 2. 渲染方程

### 2.1 光学基础

#### 辐射度量学 (Radiometry)

**辐射通量 (Radiant Flux)**：单位时间内的能量，单位为瓦特 (W)

$$\Phi = \frac{dQ}{dt}$$

**辐照度 (Irradiance)**：单位面积接收的辐射通量，单位为 W/m²

$$E = \frac{d\Phi}{dA}$$

**辐射强度 (Intensity)**：单位立体角的辐射通量，单位为 W/sr

$$I = \frac{d\Phi}{d\omega}$$

**辐射率 (Radiance)**：单位立体角、单位投影面积的辐射通量，单位为 W/(m²·sr)

$$L = \frac{d^2\Phi}{dA \cos\theta \, d\omega}$$

其中 $\theta$ 是光线与表面法线的夹角。

### 2.2 渲染方程

**渲染方程** (Rendering Equation) 是计算机图形学中最重要的方程，描述了点p处沿方向 $\omega_o$ 发出的辐射：

$$L_o(p, \omega_o) = L_e(p, \omega_o) + \int_{\Omega} f_r(p, \omega_i, \omega_o) L_i(p, \omega_i) \cos\theta_i \, d\omega_i$$

**各项含义**：

- $L_o(p, \omega_o)$ — 点p沿 $\omega_o$ 方向的出射辐射
- $L_e(p, \omega_o)$ — 点p的自发光
- $\int_{\Omega}$ — 对所有入射方向的积分
- $f_r(p, \omega_i, \omega_o)$ — **BRDF** (双向反射分布函数)
- $L_i(p, \omega_i)$ — 从 $\omega_i$ 方向入射到点p的辐射
- $\cos\theta_i$ — 余弦因子（表示入射角的影响）

### 2.3 渲染方程的含义

渲染方程表明：**出射光 = 自发光 + 入射光经过表面反射的贡献**

这是一个**积分方程**，因为：

- 入射光可能来自其他表面（需要递归求解）
- 需要对整个半球积分

### 2.4 蒙特卡罗积分求解

由于解析求解通常不可行，使用蒙特卡罗方法进行数值近似：

$$L_o \approx \frac{1}{N} \sum_{i=1}^{N} \frac{f_r(p, \omega_i, \omega_o) L_i(p, \omega_i) \cos\theta_i}{p(\omega_i)}$$

其中 $p(\omega_i)$ 是采样方向的概率分布。

---

## 3. BRDF - 双向反射分布函数

### 3.1 BRDF定义

**双向反射分布函数** (Bidirectional Reflectance Distribution Function, BRDF) 表示从入射方向 $\omega_i$ 来的光线有多少比例被反射到出射方向 $\omega_o$：

$$f_r(p, \omega_i, \omega_o) = \frac{dL_o(p, \omega_o)}{dE_i(p, \omega_i)}$$

**单位**：sr⁻¹ (每立体角)

**物理约束**：非负性

$$f_r(p, \omega_i, \omega_o) \geq 0$$

### 3.2 能量守恒

BRDF必须满足能量守恒定律——反射的能量不能超过入射的能量：

$$\int_{\Omega} f_r(p, \omega_i, \omega_o) \cos\theta_o \, d\omega_o \leq 1$$

### 3.3 光学可逆性

根据光学中的可逆原理，BRDF关于入射和出射方向对称：

$$f_r(p, \omega_i, \omega_o) = f_r(p, \omega_o, \omega_i)$$

### 3.4 标准反射模型

#### 漫反射材料 (Diffuse / Lambertian)

完全漫反射材料的BRDF为常数：

$$f_r = \frac{\rho}{\pi}$$

其中 $\rho$ 是反射率 ($0 \leq \rho \leq 1$)。

**特点**：

- 从任何方向看都有相同的亮度
- 服从Lambert余弦定律
- 常见于哑光表面

#### 镜面反射 (Perfect Specular)

完全镜面反射的BRDF：

$$f_r(p, \omega_i, \omega_o) = \delta(\omega_i - \omega_r) / \cos\theta_i$$

其中 $\omega_r$ 是根据反射定律计算的反射方向。

**特点**：

- 反射方向唯一确定
- 只在特定方向有反射
- 常见于镜子和抛光金属

#### 光泽材料 (Glossy Material)

介于完全漫反射和完全镜面反射之间的材料。

**特点**：

- 在镜面反射方向附近有峰值
- 但不完全限制在单一方向
- 常见于陶瓷、塑料等

### 3.5 BRDF的测量

**Stanford Gonioreflectometer** — 在实验室中精确测量真实材料的BRDF。

**过程**：

1. 固定入射光方向
2. 在各种出射方向测量反射光强度
3. 重复不同的入射方向
4. 建立BRDF数据库

---

## 4. 微表面BRDF模型

### 4.1 微表面理论

虽然宏观上表面看起来光滑，但在微观尺度上存在许多微小的凹凸。微表面理论将这些微观结构建模为**小的完全镜面反射器**。

**关键假设**：

- 每个微小表面都进行完全镜面反射
- 微表面法线分布造成了宏观的漫反射或光泽特性

### 4.2 微表面BRDF的三个组成部分

#### 1. 法线分布函数 (Normal Distribution Function, NDF)

描述微表面法线相对于宏观法线的分布。

**Beckmann分布**：

$$D(h) = \frac{1}{\pi \alpha^2 \cos^4\theta_h} \exp\left(-\tan^2\theta_h / \alpha^2\right)$$

其中 $h$ 是半方向（$h = \frac{\omega_i + \omega_o}{|\omega_i + \omega_o|}$），$\alpha$ 是表面粗糙度。

**GGX分布** (Trowbridge-Reitz)：

$$D(h) = \frac{\alpha^2}{\pi(\alpha^2 \cos^2\theta_h + \sin^2\theta_h)^2}$$

GGX在处理高光尾部时表现更好，更符合真实材料。

#### 2. 几何函数 (Geometry Function, G)

描述微表面遮挡和自遮挡的影响。当光线在微表面凹陷中被挡住时，不会对反射贡献。

**Smith's Shadowing-Masking 函数**：

$$G(\omega_i, \omega_o, h) = G_1(\omega_i) \cdot G_1(\omega_o)$$

其中：

$$G_1(\omega) = \frac{1}{1 + \Lambda(\omega)}$$

#### 3. Fresnel项

描述光线在不同角度的反射比例。在掠射角（接近平行于表面），反射率增加。

**Fresnel-Schlick 近似**：

$$F(\omega_i, h) = F_0 + (1 - F_0)(1 - \cos(\omega_i, h))^5$$

其中 $F_0$ 是垂直入射的反射率。

### 4.3 完整的微表面BRDF

**Cook-Torrance模型**：

$$f_r(\omega_i, \omega_o) = k_d \frac{\rho}{\pi} + k_s \frac{D(h) G(\omega_i, \omega_o, h) F(\omega_i, h)}{4(\omega_i \cdot n)(\omega_o \cdot n)}$$

**各项**：

- $k_d$ — 漫反射系数
- $k_s$ — 镜面反射系数
- $D$ — 法线分布
- $G$ — 几何函数
- $F$ — Fresnel项
- $(\omega_i \cdot n)$、$(\omega_o \cdot n)$ — 入射和出射方向的余弦

### 4.4 数据驱动的BRDF模型

对于复杂材料，直接使用测量数据拟合BRDF：

**过程**：

1. 在实验室测量真实材料的BRDF
2. 存储为高维查找表
3. 渲染时直接查表而不是计算

**优点**：

- 准确性高
- 对非标准材料有效

**缺点**：

- 存储空间大
- 难以进行实时渲染
- 难以在不同光照条件下插值

---

## 5. 纹理映射

### 5.1 纹理映射基础

**纹理映射** (Texture Mapping) 是将2D图像粘贴到3D模型表面的过程。

**参数化过程**：

1. **建立UV坐标** — 将3D模型表面映射到2D平面 $(u, v) \in [0,1]^2$
2. **查询纹理** — 根据点的UV坐标从纹理图像中查询颜色
3. **应用到模型** — 将纹理颜色用于着色计算

**数学表达**：

$$\text{Color}(p) = \text{Texture}(UV(p))$$

### 5.2 纹理坐标的创建

#### 展开法 (Unwrapping)

将3D模型表面"展开"成2D平面，最小化扭曲。

**挑战**：

- 如何在不过度拉伸或撕裂的情况下展开复杂表面
- 最小化面积和角度的扭曲

#### 投影法

直接将3D坐标投影到2D平面：

- **平面投影** — 垂直投影到平面
- **柱面投影** — 投影到圆柱面
- **球面投影** — 投影到球面

### 5.3 纹理反射率属性 (Texture Reflectance Properties)

使用纹理控制材料的各种属性：

#### 颜色纹理 (Diffuse/Albedo Map)

控制漫反射颜色 $\rho(u, v)$：

$$L_o = \int_{\Omega} f_r(u,v) L_i \cos\theta \, d\omega$$

其中 $f_r$ 依赖于纹理的颜色值。

#### 法线纹理 (Normal Map)

存储微小表面的法线信息，产生凹凸感而无需修改几何：

$$n(u, v) = \text{NormalMap}(u, v)$$

在着色计算中使用这个扰动法线而不是几何法线。

**优点**：

- 增加视觉细节而不增加模型复杂度
- 计算高效

#### 粗糙度纹理 (Roughness Map)

控制表面的微表面粗糙度参数 $\alpha(u, v)$：

$$\alpha(u, v) = \text{RoughnessMap}(u, v)$$

光滑的区域 ($\alpha$ 小) 显示尖锐的高光，粗糙的区域 ($\alpha$ 大) 显示扩散的高光。

#### 金属感纹理 (Metallic Map)

区分金属和非金属材料：

- 金属：反射率很高，反射颜色由金属类型决定
- 非金属：反射率低（通常 $F_0 \approx 0.04$），反射颜色为白色

#### 自发光纹理 (Emissive Map)

为某些区域添加自发光：

$$L_o = L_e(u, v) + \text{...反射项...}$$

### 5.4 纹理过滤和采样

#### 纹理缩小 (Minification)

当纹理像素小于屏幕像素时（较远的物体）：

- **最近邻** — 简单但有走样
- **双线性过滤** — 插值相邻4个像素
- **三线性过滤** — 在多个mipmap级别间插值
- **各向异性过滤** — 考虑观看方向

#### Mipmap

预先计算纹理的多个分辨率版本：

- Level 0 — 原始纹理 (1024×1024)
- Level 1 — 2×缩小 (512×512)
- Level 2 — 4×缩小 (256×256)
- ...

根据距离自动选择合适的mipmap级别。

### 5.5 几何细节纹理化 (Texture Geometric Detail)

#### 位移映射 (Displacement Mapping)

沿着表面法线移动顶点位置：

$$p'(u,v) = p(u,v) + h(u,v) \cdot n(u,v)$$

其中 $h(u,v)$ 是从高度纹理读取的位移值。

**优点**：

- 真正改变几何
- 可以看到剪影边缘的细节

**缺点**：

- 需要细分网格
- 计算成本高

#### 视差映射 (Parallax Mapping)

改进法线映射，考虑视点位置：

根据视线方向和高度纹理，调整UV坐标以创建深度错觉。

---

## 6. 环境和反射映射

### 6.1 环境映射 (Environment Mapping)

使用一张全景图像表示场景的光照环境。

#### 立方体映射 (Cube Mapping)

将环境表示为立方体的6个面：

- 正X、负X — 右、左
- 正Y、负Y — 上、下
- 正Z、负Z — 前、后

**优点**：

- 离散性强，便于存储和查询
- 支持硬件过滤

#### 球面映射 (Spherical Mapping)

环境存储在一张球形全景图上：

$$\text{dir} = (\sin\phi\cos\theta, \cos\phi, \sin\phi\sin\theta)$$

其中：

- $\phi$ — 极角 (0到π)
- $\theta$ — 方位角 (0到2π)

### 6.2 反射映射 (Reflection Mapping)

对于反射性强的物体，使用环境映射计算反射：

$$L_o = f_r L_e(\omega_r)$$

其中 $\omega_r$ 是根据反射定律计算的反射方向。

### 6.3 基于图像的光照 (Image-Based Lighting, IBL)

使用高动态范围 (HDR) 环境贴图作为场景的主要光源。

**过程**：

1. **采样环境光** — 从环境贴图采样多个方向的光
2. **计算着色** — 对每个方向求和贡献
3. **预计算优化** — 使用球谐函数或立方体贴图mipmap预计算光照

**优点**：

- 自然的全局光照效果
- 与任何环境兼容

---

## 7. 光线追踪在生产中的应用

### 7.1 视觉特效 (Visual Effects)

**静帧渲染** — 使用光线追踪计算高质量图像：

- **降噪** — 使用多个样本，然后应用降噪滤波器
- **自适应采样** — 在高频区域使用更多样本
- **缓存和重用** — 在序列帧间缓存计算结果

**性能指标** — 以**每像素样本数 (spp)** 衡量：

- 低质量：1-4 spp
- 中等质量：16-64 spp
- 高质量：256-1024 spp

### 7.2 实时光线追踪 (Real-Time Ray Tracing)

现代GPU支持硬件光线追踪：

**NVIDIA RTX** — 专用光线追踪核心
**AMD RDNA 2** — 光线加速器
**Intel Arc** — 光线追踪单位

**应用**：

- 实时反射和阴影
- 1080p 30-60 FPS（配合混合渲染）
- 游戏中的全局光照

### 7.3 混合渲染 (Hybrid Rendering)

结合光栅化和光线追踪：

1. **光栅化主要内容** — 快速渲染基础几何和着色
2. **光线追踪增强** — 仅对特定效果（如反射、阴影）使用光线追踪
3. **降噪** — 使用深度学习模型降低光线追踪的噪声

**优势**：

- 比纯光线追踪快得多
- 相比纯光栅化视觉效果显著提升

---

## 总结

高级渲染是从物理原理到图像生成的桥梁：

1. **渲染方程** — 统一的物理框架
2. **BRDF** — 材料的光学特性描述
3. **微表面模型** — 现代PBR的基础
4. **纹理映射** — 高效的细节表达
5. **环境映射** — 全局光照的实现
6. **光线追踪** — 物理准确但计算密集
7. **实时技术** — 平衡质量和性能

现代渲染通常采用**物理基渲染 (Physically Based Rendering, PBR)** 管线，确保材料在不同光照条件下的表现一致且可预测。
