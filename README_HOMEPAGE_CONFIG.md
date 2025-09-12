# 主页配置说明 / Homepage Configuration Guide

## 概述 / Overview

现在你可以通过编辑 `_data/homepage.yml` 文件来轻松管理主页的技能标签和统计数据，无需修改HTML或Markdown代码。

## 配置文件位置 / Configuration File Location

```
_data/homepage.yml
```

## 技能标签配置 / Skills Configuration

### 添加新技能标签 / Adding New Skills

在 `skills` 部分添加新项目：

```yaml
skills:
  - name_zh: "技能中文名"
    name_en: "Skill English Name"
    category: "skill_category" # 影响颜色主题
    url: "/target/url/" # 点击跳转的链接
    tooltip_zh: "中文提示信息"
    tooltip_en: "English Tooltip"
```

### 可用的技能分类 / Available Skill Categories

- `ai` - 人工智能相关 (紫色主题)
- `vision` - 计算机视觉 (蓝色主题)
- `nlp` - 自然语言处理 (绿色主题)
- `embodied` - 具身智能 (粉色主题)
- `python` - Python语言 (金色主题)
- `rust` - Rust语言 (棕色主题)
- `javascript` - JavaScript语言 (黄色主题)
- `framework` - 框架工具 (橙色主题)
- `web` - Web技术 (紫色主题)
- `tool` - 开发工具 (灰色主题)

### 自定义颜色 / Custom Colors

在 `skill_colors` 部分修改或添加颜色配置：

```yaml
skill_colors:
  new_category:
    bg: "rgba(255, 0, 0, 0.1)" # 背景色
    border: "#ff0000" # 边框色
    color: "#cc0000" # 文字色
```

## 统计数据配置 / Statistics Configuration

### 修改统计卡片 / Modifying Stat Cards

在 `statistics` 部分编辑：

```yaml
statistics:
  - name_zh: "统计项目中文名"
    name_en: "Stat Item English Name"
    id: "unique-element-id" # JavaScript中使用的ID
    url: "/target/page/" # 点击跳转链接
    tooltip_zh: "中文提示"
    tooltip_en: "English Tooltip"
```

**重要：** `id` 字段必须与JavaScript代码中的统计逻辑匹配。当前支持的ID：

- `github-repos` - GitHub仓库数量
- `github-stars` - GitHub星数总计
- `blog-posts` - 博客文章数量
- `study-notes` - 学习笔记数量

## 示例修改 / Example Modifications

### 添加新的编程语言标签

```yaml
skills:
  # ... 现有技能
  - name_zh: "Go语言"
    name_en: "Go Language"
    category: "go"
    url: "https://github.com/stibiums?tab=repositories&q=&type=&language=go"
    tooltip_zh: "查看Go语言项目"
    tooltip_en: "View Go Language Projects"
```

然后在 `skill_colors` 中添加颜色：

```yaml
skill_colors:
  # ... 现有颜色
  go:
    bg: "rgba(0, 173, 216, 0.1)"
    border: "#00add8"
    color: "#007d9c"
```

### 修改统计项目链接

只需在 `statistics` 部分修改对应的 `url` 字段即可。

## 注意事项 / Important Notes

1. **修改后重新构建** - 更改配置文件后需要重新构建Jekyll站点才能生效
2. **保持ID一致** - 统计数据的 `id` 字段需要与JavaScript逻辑匹配
3. **URL格式** - 外部链接使用完整URL，内部链接使用相对路径
4. **颜色格式** - 使用标准CSS颜色格式（hex、rgba等）

## 文件结构 / File Structure

```
_data/
├── homepage.yml          # 主页配置（新文件）
├── cv.yml               # 简历配置
└── ...                  # 其他配置文件

_pages/
├── about.md             # 主页（已更新为使用配置文件）
├── notes.md             # 学习笔记页面（新文件）
└── ...
```

现在你可以轻松地通过编辑 `_data/homepage.yml` 文件来管理主页内容，而无需直接修改HTML代码！
