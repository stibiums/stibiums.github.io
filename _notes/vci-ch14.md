---
layout: post
title: "VCI - 14: 渲染管线 (Graphics Pipeline)"
date: 2025-10-30 04:00:00
tags: notes VCI graphics pipeline GPU rendering API
categories: VCI
---

> **课程**: 北京大学视觉计算 (Visual Computing) 2025秋季
> **章节**: 第14章 渲染管线
> **内容**: 图形API、现代GPU架构、着色器、渲染算法、GPU历史

## 目录

1. [渲染管线概述](#1-渲染管线概述)
2. [图形API](#2-图形api)
3. [GPU架构](#3-gpu架构)
4. [着色器](#4-着色器)
5. [渲染策略](#5-渲染策略)
6. [现代渲染技术](#6-现代渲染技术)
7. [GPU发展历史](#7-gpu发展历史)

---

## 1. 渲染管线概述

### 1.1 什么是渲染管线

**渲染管线** (Graphics Pipeline / Rendering Pipeline) 是从3D模型数据到2D屏幕图像的一系列处理步骤。

现代渲染管线通常包括以下主要阶段：

1. **顶点处理** — 模型/视图变换、顶点着色
2. **几何处理** — 几何着色器、曲面细分
3. **光栅化** — 将图元转换为片元
4. **片元处理** — 纹理采样、片元着色
5. **帧缓冲** — 深度测试、混合、输出合并

### 1.2 光栅化渲染管线

光栅化（与光线追踪相反）是当今实时渲染的主要方法。

**特点**：

- 将3D基元投影到2D屏幕
- 按像素处理
- 并行性强，GPU友好

### 1.3 管线可编程性

现代GPU渲染管线的关键特性是**可编程性**：

| 阶段     | 是否可编程 | 工具                 |
| -------- | ---------- | -------------------- |
| 顶点处理 | ✓          | 顶点着色器           |
| 片元处理 | ✓          | 片元着色器           |
| 几何处理 | ✓          | 几何着色器、曲面细分 |
| 光栅化   | ✗          | 固定功能             |
| 深度测试 | ✗          | 固定功能             |
| 颜色混合 | ✗          | 固定功能             |

---

## 2. 图形API

### 2.1 主要图形API

#### OpenGL

**描述**：跨语言、跨平台的2D/3D图形API

**特点**：

- 开放标准
- 广泛支持（桌面、移动、Web）
- 学习资源丰富

**变体**：

- **OpenGL ES** — 嵌入式系统子集
- **WebGL** — 基于Web的JavaScript API，在浏览器中实现3D图形

#### DirectX 12 Ultimate

**描述**：微软的图形和游戏开发平台

**特点**：

- 与Windows深度集成
- NVIDIA GPU上功能最全
- 低级硬件控制

#### Metal

**描述**：苹果开发的图形API

**特点**：

- iOS、macOS、tvOS平台
- 与Apple硬件深度优化
- 低开销设计

#### Vulkan

**描述**：跨平台的现代图形API

**特点**：

- 基于AMD的Mantle API开发
- 低开销、高效的硬件访问
- 跨平台支持（包括移动和桌面）
- 显式驱动管理

### 2.2 API的选择

- **学习/教学** — OpenGL（简洁、概念清晰）
- **Web应用** — WebGL
- **Windows游戏** — DirectX 12
- **跨平台** — Vulkan 或 OpenGL
- **苹果平台** — Metal

---

## 3. GPU架构

### 3.1 现代GPU的特点

现代GPU是为**大规模并行计算**设计的：

- **数千个处理核心** — 比CPU的几个核心多得多
- **内存带宽高** — 优化了存储访问
- **专用硬件** — 纹理单元、光线追踪核心等

### 3.2 GPU核心设计

**流多处理器** (Streaming Multiprocessor, SM)：

- 包含多个小核心（通常32-128个）
- 共享缓存（L1、共享内存）
- 共享纹理单元

**GPU结构**：

```
GPU芯片
├── SM 0
│   ├── 核心 0-31
│   ├── L1缓存
│   └── 共享内存
├── SM 1
│   ├── 核心 0-31
│   └── ...
└── L2缓存 + HBM内存
```

### 3.3 专用硬件

#### 纹理单元

- 高效的纹理采样和过滤
- 支持多种采样模式（最近邻、双线性、三线性）
- 自动mipmap选择

#### 光线追踪核心 (RTX)

**NVIDIA RTX**：

- 专用BVH遍历
- 光线与几何体求交加速
- 显著提升光线追踪性能

#### 张量核心 (Tensor Cores)

- 专用矩阵运算单元
- 用于AI、深度学习
- 支持混合精度计算

---

## 4. 着色器

### 4.1 着色器概述

**着色器** (Shader) 是在GPU上运行的小程序，控制渲染管线的可编程部分。

### 4.2 顶点着色器 (Vertex Shader)

**运行时机**：对输入中的**每个顶点**并行运行

**典型输入**：

- 顶点位置 (position)
- 顶点法线 (normal)
- 纹理坐标 (texCoord)
- 顶点颜色 (color)

**典型输出**：

- 变换后的位置
- 法线（用于片元着色器）
- 颜色
- 深度值
- 其他顶点属性

**例子 — Phong着色顶点着色器**：

```glsl
#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texCoord;

out vec3 fragNormal;
out vec3 fragPosition;
out vec2 fragTexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    fragPosition = vec3(model * vec4(position, 1.0));
    fragNormal = mat3(transpose(inverse(model))) * normal;
    fragTexCoord = texCoord;
    gl_Position = projection * view * vec4(fragPosition, 1.0);
}
```

**关键点**：

- 输入顶点属性
- 输出给片元着色器的插值数据
- 执行模型/视图/投影变换

### 4.3 片元着色器 (Fragment Shader)

**运行时机**：对光栅化后的**每个片元**（大约每个像素）并行运行

**典型输入**：

- 来自顶点着色器的插值数据（法线、纹理坐标等）
- 统一变量（灯光参数、材料参数）
- 纹理采样结果

**典型输出**：

- 片元颜色 (RGBA)
- 深度值（如果需要）

**例子 — Phong着色片元着色器**：

```glsl
#version 330 core

in vec3 fragNormal;
in vec3 fragPosition;
in vec2 fragTexCoord;

out vec4 FragColor;

uniform sampler2D texture1;
uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 lightColor;

void main()
{
    // 采样纹理
    vec3 texColor = texture(texture1, fragTexCoord).rgb;

    // 环境光
    vec3 ambient = 0.1 * texColor;

    // 漫反射
    vec3 norm = normalize(fragNormal);
    vec3 lightDir = normalize(lightPos - fragPosition);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * texColor;

    // 镜面反射
    vec3 viewDir = normalize(viewPos - fragPosition);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    vec3 specular = spec * lightColor;

    vec3 result = (ambient + diffuse + specular);
    FragColor = vec4(result, 1.0);
}
```

### 4.4 其他着色器类型

#### 几何着色器 (Geometry Shader)

- 运行于顶点之后，光栅化之前
- 可以生成/删除顶点和图元
- 应用：点到球体的扩展、几何学爆炸等

#### 曲面细分着色器 (Tessellation Shader)

- **控制着色器** (Control Shader) — 定义细分参数
- **计算着色器** (Evaluation Shader) — 计算新顶点位置
- 应用：动态LOD、曲面细分

#### 计算着色器 (Compute Shader)

- 不与标准渲染管线关联
- 用于通用GPU计算（GPGPU）
- 应用：物理模拟、后处理、粒子系统

### 4.5 着色语言

**GLSL** (OpenGL Shading Language)：

- 基于C语言
- 内置向量和矩阵类型
- 广泛的数学函数库

**HLSL** (High Level Shading Language)：

- DirectX着色语言
- 语法与GLSL相似

---

## 5. 渲染策略

### 5.1 正向渲染 (Forward Rendering)

**原理**：对每个三角形，对所有光源进行着色计算

**伪代码**：

```cpp
for each triangle:
    for each pixel in triangle:
        for each light source:
            color += shading(triangle, light)
        test z-buffer and update
```

**复杂度**：$$O(N \times M)$$ — N个三角形，M个光源

**特点**：

- 直观、易于实现
- **多个光源时效率低** — 许多着色计算被后来的像素覆盖
- 优点：处理透明度容易，支持MSAA抗锯齿

### 5.2 早期Z测试 (Early-Z)

**优化思路**：先进行深度测试，只对未被遮挡的像素进行着色

**伪代码**：

```cpp
for each triangle (sorted by depth):
    for each pixel not occluded:
        for each light source:
            color += shading(triangle, light)
        test z-buffer and update
```

**效果**：

- 如果三角形按深度排序，可大幅减少着色计算
- 最坏情况（反向排序）无效果

### 5.3 延迟渲染 (Deferred Rendering)

**原理**：

1. 第一遍：将所有表面信息存储到屏幕空间缓冲（G-Buffer）
2. 第二遍：读取G-Buffer，对每个像素执行着色

**步骤**：

```cpp
// 第一遍：写G-Buffer
for each triangle:
    for each pixel:
        test depth and update G-Buffer
        store position, normal, albedo, depth...

// 第二遍：着色
for each pixel:
    read G-Buffer data
    for each light:
        color += shading(G-Buffer[pixel], light)
    output color
```

**复杂度**：$$O(N + M)$$ — 线性复杂度！

**G-Buffer典型内容**：

- **Position** (RGB) — 世界空间位置
- **Normal** (RGB) — 表面法线
- **Albedo** (RGB) — 漫反射颜色
- **Depth** (R) — 深度值
- **MaterialID** (R) — 材料标识
- 可能还有：高光颜色、粗糙度、金属感等

**优点**：

- **多光源效率高** — 与光源数量线性相关
- 支持海量点光源
- 计算着色与光源数无关

**缺点**：

- **高带宽开销** — G-Buffer可能很大（每像素32-64字节）
- 难以处理透明度
- MSAA抗锯齿开销大
- 着色质量可能降低（由于G-Buffer精度有限）

**G-Buffer优化**：

- 使用16位或10位纹理（压缩法线）
- 使用延迟照明缓冲
- 光源剔除（计算哪些光源影响哪些像素）

### 5.4 正向vs延迟

| 特性         | 正向渲染 | 延迟渲染 |
| ------------ | -------- | -------- |
| **少光源**   | 快       | 较慢     |
| **多光源**   | 很慢     | 快       |
| **透明度**   | 容易     | 困难     |
| **抗锯齿**   | 原生支持 | 高开销   |
| **内存带宽** | 低       | 高       |
| **着色质量** | 高       | 可能降低 |

**实践**：

- 现代引擎通常采用**混合**方法
- 不透明物体用延迟渲染
- 透明物体用正向渲染
- 或使用**前向+** (Forward+) 技术

---

## 6. 现代渲染技术

### 6.1 网格着色器 (Mesh Shader)

**背景**：传统管线中顶点→图元→片元的处理流程固定且低效

**网格着色器优势**：

- 从顶点着色器到网格着色器的演进
- 更灵活的几何处理
- 支持动态LOD（细节级别）
- 减少GPU-CPU通信开销

**应用**：

- **网格平滑** — 动态调整细节
- **剔除** — GPU端进行可见性剔除
- **对象着色** — 更高效的对象处理

### 6.2 动态全局光照 (Dynamic Global Illumination)

**传统做法**：

- 离线预计算全局光照（光照贴图、球谐函数）
- 缺乏动态性

**现代技术**：

- **实时GI** — 使用GPU计算实时全局光照
- **光线追踪** — NVIDIA RTX等支持硬件光线追踪
- **Voxel Cone Tracing** — 快速近似
- **Radiance Cascades** — 层次化光照

**应用示例** (Unreal Engine)：

- Lumen系统 — 完全动态的全局光照
- 支持动态物体和光源
- 实时反应场景变化

### 6.3 深度学习超采样 (DLSS)

**NVIDIA DLSS** (Deep Learning Super Sampling)：

- 使用深度学习网络进行超分辨率
- 从低分辨率渲染上采样到高分辨率
- 显著提升帧率同时保持质量

**工作流**：

1. 以1/4分辨率渲染内容
2. 使用深度学习网络上采样
3. 输出高质量高分辨率图像

**优势**：

- 性能提升 2-3 倍
- 通常质量好于TAA
- AI模型不断改进

---

## 7. GPU发展历史

### 7.1 早期阶段 (1976-1984)

- **2D硬件加速** — 仅支持2D绘制
- **3D在CPU** — 3D图形由CPU处理
- 主要产品：显示适配器、视频加速卡

### 7.2 SGI工作站时代 (1984-2005)

- **Silicon Graphics (SGI)** 推出高端图形工作站
- 支持3D加速
- 高价格，用于专业领域（电影、CAD）

### 7.3 PC 3D加速时代 (1995-2006)

**3Dfx VOODOO** (1995-1999)：

- 首个普及的3D加速卡
- 推动PC游戏3D化

**NVIDIA vs ATI** (2000-2006)：

- 可编程渲染管线出现
- **2002**: HLSL (DirectX 9) 和 GLSL (OpenGL) 发布
- **2006**: 几何着色器推出

**2006**: ATI被AMD收购

### 7.4 通用GPU时代 (2006-2013)

- **2007**: CUDA (Nvidia的GPU通用计算框架)
- GPU开始用于非图形计算：科学计算、模拟、AI
- **2009**: 曲面细分着色器、计算着色器

### 7.5 现代时代 (2013-至今)

**性能提升**：

- 核心数继续增加
- 内存带宽提升
- 功耗优化

**新功能**：

- **2018**: NVIDIA RTX — 硬件光线追踪
- **2020**: NVIDIA DLSS — 深度学习超采样
- **2021**: NVIDIA DLSS 2.0 — 更好的质量

**应用扩展**：

- **游戏** — 主要应用
- **AI/深度学习** — 快速成长的领域
- **科学计算** — HPC系统
- **加密货币挖矿** — 一度造成显卡短缺

---

## 总结

现代渲染管线是复杂的系统：

1. **API与硬件** — OpenGL/Vulkan/DirectX 映射到GPU硬件
2. **着色器** — 可编程处理为核心，GLSL/HLSL是标准语言
3. **管线阶段** — 顶点→几何→光栅化→片元→帧缓冲
4. **渲染策略** — 正向渲染适合少光源，延迟渲染适合多光源
5. **现代趋势** — 动态GI、硬件光线追踪、深度学习增强
6. **GPU演化** — 从专用显卡到通用计算加速器

**最佳实践**：

- 了解目标硬件的能力
- 选择合适的渲染策略（正向/延迟/混合）
- 充分利用GPU的并行性
- 使用profiling工具优化瓶颈
- 权衡视觉质量与性能

现代游戏引擎（Unreal、Unity等）封装了这些复杂性，但理解底层管线有助于优化和创意实现。
