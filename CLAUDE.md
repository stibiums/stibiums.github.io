# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Jekyll-based academic website built using the **al-folio** theme. It's a personal academic portfolio site featuring blog posts, CV, publications, projects, and notes. The site is deployed on GitHub Pages and includes both Chinese and English content.

## Development Setup

### Primary Development Method (Recommended)

```bash
# Using Docker (most reliable across platforms)
docker compose pull
docker compose up
```

Access at `http://localhost:8080`

### Alternative Development Methods

```bash
# Local Jekyll development (requires Ruby/Bundler)
bundle install
bundle exec jekyll serve
# Access at http://localhost:4000

# Slim Docker image (under 100MB)
docker compose -f docker-compose-slim.yml up
```

### Code Quality Commands

```bash
# Format code with Prettier
npm run format

# Install Git hooks
npm run prepare
```

## Site Architecture

### Content Structure

- **`_posts/`** - Blog posts (Chinese travel logs, technical notes)
- **`_notes/`** - Study notes collection (CS231n, algorithms, Rust)
- **`_projects/`** - Project showcase
- **`_pages/`** - Static pages (about, CV, publications, etc.)
- **`_news/`** - News/announcements for homepage
- **`_data/`** - Structured data (CV, social links, repositories)
- **`_bibliography/`** - Publications in BibTeX format

### Key Configuration

- **`_config.yml`** - Main site configuration
- **`Gemfile`** - Ruby dependencies and Jekyll plugins
- **`package.json`** - Node.js tools (Prettier, Husky)

### Content Collections

The site uses several Jekyll collections:

- `posts` - Blog posts with date-based URLs
- `notes` - Study notes with custom permalinks
- `projects` - Project portfolio
- `books` - Book reviews/bookshelf
- `news` - Homepage announcements

### Multilingual Content

- Primary language: Chinese (with some English)
- Study notes cover: Computer Vision (CS231n), Algorithms, Rust programming
- Travel blogs document trips to various Chinese cities

### Asset Management

- **`assets/img/`** - Images with automatic WebP conversion
- **`assets/jupyter/`** - Jupyter notebooks for technical content
- **`assets/pdf/`** - PDF documents (CV, papers)
- **`assets/json/`** - Resume data in JSON format

## Content Creation

### Blog Posts

Create files in `_posts/` with format: `YYYY-MM-DD-title.md`

```yaml
---
layout: post
title: Post Title
date: 2025-07-01 00:00:00
description: Post description
tags: [tag1, tag2]
categories: [category]
---
```

### Study Notes

Create files in `_notes/` for technical content:

```yaml
---
layout: page
title: Note Title
permalink: /notes/note-path/
---
```

**重要格式要求：**

- 遵循现有笔记的统一格式和YAML前置元数据结构
- 图片统一存放在 `assets/img/` 目录下
- 使用Jekyll标准的图片插入方法：`{% raw %}{% include figure.liquid path="assets/img/filename.jpg" %}{% endraw %}`
- 保持笔记内容结构的一致性

**图片布局要求：**

- 所有图片必须使用Bootstrap响应式网格布局
- 单张图片使用单列布局包裹：
  ```html
  <div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
      {% raw %}{% include figure.liquid loading="eager" path="assets/img/filename.jpg" title="图片标题" class="img-fluid rounded z-depth-1"
      zoomable=true %}{% endraw %}
    </div>
  </div>
  ```
- 多张相关图片可以使用多列并列布局：
  ```html
  <div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
      {% raw %}{% include figure.liquid loading="eager" path="assets/img/image1.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}{% endraw %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
      {% raw %}{% include figure.liquid loading="eager" path="assets/img/image2.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}{% endraw %}
    </div>
  </div>
  ```
- 必须添加的属性：
  - `loading="eager"` - 启用快速加载
  - `zoomable=true` - 启用图片放大功能
  - `class="img-fluid rounded z-depth-1"` - 响应式、圆角、阴影效果
  - `title="..."` - 图片标题（可选但推荐）
- 每张图片后应有文字说明，解释图片内容和意义

### 图片处理工作流

**CRITICAL: 所有图片必须经过严格的科学性验证！**

整理学习笔记时，图片处理遵循以下流程：

#### 1. 图片来源选择

根据内容需求选择合适的图片来源：

**方式A：Python生成图片（推荐用于概念图解）**

- 适用场景：几何图形、数学概念、算法流程、数据可视化
- 优点：精确控制、可定制、无版权问题
- 要求：
  - 使用matplotlib/numpy等科学计算库
  - 图片分辨率≥300 DPI
  - 标注使用英文，避免中文
  - 颜色搭配清晰，适合学术展示
  - 保存为PNG格式

**方式B：从PDF提取图片（推荐用于课程原始材料）**

- 适用场景：课程幻灯片、教材插图、示例照片、硬件设备图
- 工具位置：`_tools/extract_pdf_images.py`
- 使用方法：

  ```bash
  # 提取所有嵌入图片
  python3 _tools/extract_pdf_images.py lecture.pdf assets/img/notes_img/chapter/

  # 提取特定页面
  python3 _tools/extract_pdf_images.py lecture.pdf output/ --pages 5,10,15

  # 交互模式
  python3 _tools/extract_pdf_images.py lecture.pdf output/ --interactive
  ```

- 提取后必须：
  - 重命名为有意义的文件名（如`epipolar_geometry.png`而非`page_08_img_2.jpeg`）
  - 检查图片清晰度和完整性
  - 删除提取的临时文件

**方式C：网络搜索（谨慎使用）**

- 仅在以上两种方式都不适用时考虑
- 必须注意版权问题
- 优先使用CC协议或公共领域图片

#### 2. 图片存储规范

- **存储路径**：`assets/img/notes_img/<chapter-name>/`
- **命名规范**：
  - 使用描述性英文名称，小写+下划线分隔
  - 例：`epipolar_constraint.png`、`stereo_camera_setup.png`
  - 避免：`图1.png`、`img001.png`、`screenshot.png`
- **目录结构**：
  ```
  assets/img/notes_img/
  ├── cv-ch02/          # Computer Vision Chapter 2
  ├── cv-ch03/
  ├── cv-ch07/
  │   ├── baseline.png
  │   ├── epipolar_lines.png
  │   └── README.md     # 列出本章所需图片
  └── cs231n-ch01/      # CS231n Chapter 1
  ```

#### 3. 科学性验证（必须执行！）

**对于Python生成的图片：**

- [ ] 几何关系是否正确（如对极线必过极点）
- [ ] 数学公式标注是否准确
- [ ] 坐标轴、单位、标签是否完整
- [ ] 颜色编码是否有图例说明
- [ ] 与课程/教材内容是否一致

**对于PDF提取的图片：**

- [ ] 图片是否清晰完整（无截断、模糊）
- [ ] 分辨率是否足够（建议≥800px宽度）
- [ ] 是否包含必要的标注和说明
- [ ] 来源是否可靠（课程官方材料优先）

**验证方法：**

1. 使用Read工具查看生成/提取的图片
2. 对照原始材料（PDF、教材）验证准确性
3. 检查关键概念的可视化是否符合定义
4. 如有疑问，查阅权威参考资料确认

#### 4. 图片插入与说明

在笔记中插入图片时：

```markdown
## 7.3 对极几何基础

### 7.3.1 基线（Baseline）

两个相机中心$$O$$和$$O'$$之间的连线称为**基线**（Baseline）。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% raw %}{% include figure.liquid loading="eager" path="assets/img/notes_img/cv-ch07/baseline.png" title="立体相机基线示意图" class="img-fluid rounded z-depth-1" zoomable=true %}{% endraw %}
    </div>
</div>

上图展示了双目立体视觉系统的基本配置，其中两个相机中心$$O$$和$$O'$$之间的橙色线段即为基线。基线长度影响深度测量的精度：基线越长，三角测量的精度越高，但对应点匹配难度也会增加。
```

**图片说明要求：**

- 必须在图片后添加文字说明（1-3句）
- 说明应包括：
  - 图片展示的主要内容
  - 关键元素的解释（如颜色、符号含义）
  - 与理论知识的联系
  - 重要性或应用场景（可选）
- 使用准确的学术术语
- 与正文内容紧密结合

#### 5. 图片管理最佳实践

**创建README记录：**
在每章图片目录下创建`README.md`，列出：

- 所需图片清单
- 每张图片的用途说明
- 图片来源（生成/提取/其他）
- 生成图片的脚本（如有）

**示例：**

```markdown
# CV Chapter 7 - Epipolar Geometry Images

## Required Images

1. `baseline.png` - Baseline between two cameras ✅ **Generated**
2. `stereo_setup.png` - Two-camera setup ✅ **Extracted from PDF**
3. `epipolar_lines.png` - Epipolar lines visualization ✅ **Generated**

## Generation Scripts

- `generate_images.py` - Generates geometric diagrams (items 1, 3)
- Extracted from `07 Epipolar Geometry.pdf` pages 2, 8

## Notes

- All diagrams use English labels only
- Color scheme: blue for cameras, orange for lines, purple for points
```

**清理临时文件：**

- 提取图片后，删除中间文件夹（如`extracted/`）
- 删除临时脚本（除非是通用工具）
- 仅保留最终使用的图片

### Projects

Add to `_projects/` directory for portfolio items.

## Deployment

### GitHub Pages (Current)

- Automatic deployment on push to `main` branch
- Built site deployed to `gh-pages` branch
- No manual deployment needed

### Build Process

```bash
# Manual build for other hosting
bundle exec jekyll build
# Output in _site/ directory

# CSS optimization
purgecss -c purgecss.config.js
```

## Important Notes

- **Never edit the `gh-pages` branch** - it's automatically generated
- All content changes should be made on the `main` branch
- The site includes extensive Jekyll plugins for academic features
- Jupyter notebooks are supported and automatically converted
- Math typesetting via MathJax is enabled
- Image optimization and lazy loading are configured
- The theme supports both light and dark modes
- 我希望你在为我生成笔记时可以插入一些合适的配图，你可以在网络上寻找，也可以使用python脚本生成（你无法运行时可以让我提供帮助），注意存贮照片文件夹里的层级结构，每章一个目录,图片中避免中文
- 注意命名规范，参考已有命名
- 在\_notes中完成笔记之后，需要在\_posts的相应文件中添加链接
- 单独的笔记文件中不要出现 # 一级标题
- 特殊符号必须使用latex的公式表示，确保LaTeX公式在Jekyll/Markdown环境中能够正确渲染
