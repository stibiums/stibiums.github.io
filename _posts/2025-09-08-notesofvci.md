---
layout: post
title: notes of VCI
date: 2025-09-08 02:00:00
description: 这是关于可视计算与交互概论(VCI)的学习笔记.
tags: notes
categories: my-notes
---

这是关于可视计算与交互概论的学习笔记

### 课程笔记目录

- **[VCI - 1: 颜色，颜色感知与可视化]({% link _notes/vci-ch01.md %})**

  - 人类视觉系统与颜色感知
  - 颜色空间与颜色模型
  - 色域与颜色管理
  - 可视化中的颜色应用

- **[VCI - 2: 显示]({% link _notes/vci-ch02.md %})**

  - 显示设备发展历程
  - 二维显示技术（CRT、LCD、OLED）
  - 立体显示与裸眼3D技术
  - 增强现实显示系统

- **[VCI - 3: 2D图形绘制]({% link _notes/vci-ch03.md %})**

  - 扫描转换与光栅化
  - 直线绘制算法（DDA、Bresenham）
  - 圆形绘制算法
  - 多边形填充技术
  - 颜色插值与图像变形

- **[VCI - 4: 抗锯齿]({% link _notes/vci-ch04.md %})**

  - 信号处理基础与采样定理
  - 锯齿现象的产生机制
  - 抗锯齿技术分类（预滤波、后处理）
  - 纹理抗锯齿与MIP映射
  - 现代抗锯齿技术（MLAA、FXAA、TAA）

- **[VCI - 5: 曲线]({% link _notes/vci-ch05.md %})**

  - 二维曲线表示方法（显式、隐式、参数）
  - 贝塞尔曲线理论与De Casteljau算法
  - 样条曲线（自然三次样条、Catmull-Rom、B样条）
  - NURBS曲线与有理贝塞尔曲线
  - 几何连续性与曲线质量评估

- **[VCI - 6: 图像表示与处理]({% link _notes/vci-ch06.md %})**

  - 图像的定义与连续表示
  - 矢量表示与栅格表示对比
  - 帧缓冲区与颜色存储（RGB、LUT）
  - Alpha通道与图像合成
  - 点处理与图像滤波（模糊、边缘检测）
  - 图像修复与泊松编辑
  - 图像抠图技术

- **[VCI - 7: 几何表示]({% link _notes/vci-ch07.md %})**

  - 几何的定义与经典模型（Utah Teapot, Stanford Bunny）
  - 计算机中的几何编码（显式表示vs隐式表示）
  - 点云表示与应用
  - 多边形网格（三角形网格、四边形网格）
  - 半边数据结构（Half-edge）
  - 细分曲面（Catmull-Clark、Loop）
  - 网格参数化与应用

- **[VCI - 8: 几何处理]({% link _notes/vci-ch08.md %})**

  - 基础几何操作（叉积、平面方程、距离计算）
  - 离散微分几何（重心坐标、三角形网格梯度）
  - Laplace-Beltrami算子（均匀与余切公式）
  - 网格光顺（扩散流、显式与隐式方法）
  - 保持细节的网格编辑
  - 网格简化（二次误差度量、边坍缩）

- **[VCI - 10: 几何重建]({% link _notes/vci-ch10.md %})**

  - 三维点云的数据来源
  - 坐标变换基础（旋转、平移、刚体变换）
  - 点云配准（ICP算法）
  - 曲面重建（Delaunay三角剖分、Poisson重建）
  - 模型拟合（RANSAC平面检测）
  - 实际应用与开源工具
