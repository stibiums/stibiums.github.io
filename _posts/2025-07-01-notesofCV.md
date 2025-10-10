---
layout: post
title: notes of CV
date: 2025-07-01 00:00:00
description: 这是关于计算机视觉(CV)的学习笔记.
tags: notes
categories: my-notes
---

这是我关于计算机视觉的学习笔记，课程为 CS231n 和 Introduction to Computer Vision

### 课程笔记目录

**CS231n Stanford**

- [第一章：Numpy]({% link _notes/cs231n-ch01.md %})

  - 创建 Arrays
  - 索引与切片
  - 数据类型
  - 数学运算
  - 广播机制
  - Matplotlib 绘图

- [第二章：图像分类]({% link _notes/cs231n-ch02.md %})

  - k-近邻分类器
  - 线性分类器
  - SVM 分类器
  - Softmax 分类器
  - 正则化
  - 梯度下降与优化

- [第三章：神经网络到CNN]({% link _notes/cs231n-ch03.md %})
  - 神经网络基础
  - 前向传播与反向传播
  - 矩阵求导
  - 卷积神经网络（CNN）介绍

**Introduction to Computer Vision (PKU)**

- [CV - 2: 图像形成]({% link _notes/cv-ch02.md %})

  - 相机模型与针孔相机
  - 镜头原理与薄镜头方程
  - 景深与弥散圈概念
  - 视野角度计算
  - 镜头畸变与校正
  - 颜色感知与数字成像

- [CV - 3: 图像处理]({% link _notes/cv-ch03.md %})

  - 图像处理基础概念
  - 线性滤波技术
  - 非线性滤波技术
  - 图像金字塔与采样
  - 图像变换技术

- [CV - 4: 特征检测]({% link _notes/cv-ch04.md %})

  - 边缘检测基础理论
  - Canny边缘检测算法
  - Harris角点检测器
  - 特征检测的数学原理
  - Blob检测方法

- [CV - 5: 图像拼接]({% link _notes/cv-ch05.md %})

  - 全景图像拼接概述
  - 图像变换与齐次坐标
  - 仿射变换与单应性变换
  - RANSAC外点检测算法
  - 图像融合技术（拉普拉斯金字塔与泊松编辑）

- [CV - 7: 对极几何]({% link _notes/cv-ch07.md %})
  - 双视图立体视觉基础
  - 对极几何的基本概念（基线、极点、对极平面、对极线）
  - 对极约束与对应点搜索
  - 本质矩阵（Essential Matrix）的性质与估计
  - 基础矩阵（Fundamental Matrix）的性质与估计
  - 八点算法与归一化八点算法
  - 从本质矩阵恢复相机运动
  - 对极几何在3D重建与视觉SLAM中的应用
