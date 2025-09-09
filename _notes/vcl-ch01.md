---
layout: post
title: "VCL - 1: 颜色，颜色感知与可视化"
date: 2025-09-09 01:00:00
tags: notes vcl
categories: vcl
---

## 颜色

颜色是光的属性，是人类视觉系统对不同波长光的感知。可见光的波长范围大约在400纳米到700纳米之间，不同波长对应不同的颜色。例如，波长约为400-450纳米的光被感知为紫色，450-495纳米为蓝色，495-570纳米为绿色，570-590纳米为黄色，590-620纳米为橙色，620-700纳米为红色。

### 相加混色和相减混色

颜色的混合可以通过两种主要方式实现：相加混色和相减混色。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/notes_img/vcl/p1.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/notes_img/vcl/p2.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

- **相加混色**：这是光的混合方式。当不同颜色的光叠加在一起时，会产生新的颜色。例如，红光、绿光和蓝光（RGB）混合可以产生白光。相加混色常用于显示器和投影仪等设备。

- **相减混色**：这是颜料或染料的混合方式。当不同颜色的颜料叠加在一起时，会吸收（减去）某些波长的光，反射出其他波长，从而产生新的颜色。例如，青色、品红色和黄色（CMY）混合可以产生黑色。相减混色常用于印刷和绘画等领域。

## 颜色的生理机制和感知

- 视杆细胞： 主要感知光线的明暗

- 视锥细胞：集中分布于视网膜中央凹，主要感知颜色

人类的视网膜中有三种类型的视锥细胞，分别对红光(L-cones)、绿光(M-cones)和蓝光(S-cones)敏感。

- **颜色的恒长性**：颜色恒长性是指在不同光照条件下，物体的颜色看起来相对稳定的现象。例如，一张白纸在阳光下和阴影中看起来仍然是白色，尽管光线的强度和色温有所不同。颜色恒长性是由大脑通过对比和记忆等机制实现的。

- **色诱导**：色诱导是指一种颜色会影响我们对另一种颜色的感知。例如，在一个红色背景上，白色看起来可能带有一点蓝色调，而在一个绿色背景上，白色看起来可能带有一点红色调。这种现象是由于大脑在处理颜色信息时，会受到周围颜色的影响。(体现其补色)

- **色觉缺陷**:
  - 单色视觉(完全色盲)：视锥细胞不可用，只能使用视杆细胞，只能感知一种颜色，通常是黑白灰。
  - 双色视觉(Protanopia, Deuteranopia, Tritanopia)：缺少一种类型的视锥细胞，导致对某些颜色的感知异常。例如，红绿色盲（Protanopia和Deuteranopia）使得红色和绿色难以区分，而蓝黄色盲（Tritanopia）使得蓝色和黄色难以区分。

## 颜色空间

### RGB颜色空间

RGB颜色空间是一种基于红、绿、蓝三种颜色通道的颜色表示方式。

- **优势**：与发光原理相同
- **劣势**：不符合人类视觉感知

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/notes_img/vcl/p3.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

### HSV颜色空间

HSV颜色空间是一种基于色调（Hue）、饱和度（Saturation）和明度（Value）的颜色表示方式。

- **优势**：更符合人类的颜色感知方式，便于理解和操作颜色
- **劣势**：不如RGB颜色空间直观，转换复杂

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/notes_img/vcl/p4.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

### 色域

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/notes_img/vcl/p5.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

HDR（高动态范围）图像可以表示比标准动态范围（SDR）图像更多的亮度级别和颜色细节。

### Gamma校正

Gamma校正是一种非线性操作，用于调整图像的亮度和对比度，以匹配人类视觉系统对亮度的感知。人眼对亮度的感知是非线性的，对暗部细节更敏感，而对亮部细节不太敏感。通过应用Gamma校正，可以使图像在显示设备上看起来更自然和真实。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/notes_img/vcl/p6.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

## 颜色在数据可视化中的应用

- 颜色直接叠加
- 加入噪声
