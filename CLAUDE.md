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
- 我希望你在为我生成笔记时可以插入一些合适的配图，你可以在网络上寻找，也可以使用python脚本生成（你无法运行时可以让我提供帮助），注意存贮照片文件夹里的层级结构,图片中避免中文
- 注意命名规范，参考已有命名
- 在\_notes中完成笔记之后，需要在\_posts的相应文件中添加链接
