---
layout: default
permalink: /notes/
title: notes
nav: true
nav_order: 2
pagination:
  enabled: true
  collection: notes # 使用 notes 集合
  permalink: /notes/page/:num/ # 修改分页 URL 模板
  per_page: 5
  sort_field: date
  sort_reverse: true
  trail:
    before: 1 # 当前页前显示 1 个链接
    after: 3 # 当前页后显示 3 个链接
---

<div class="post">
  {% comment %}根据 blog.md 中的实现，你可以复用显示笔记列表的代码{% endcomment %}

{% for note in paginator.posts %}
<article class="note">
<h2><a href="{{ note.url }}">{{ note.title }}</a></h2>
<p>{{ note.excerpt | strip_html | truncate: 150 }}</p>
</article>
{% endfor %}

{% include pagination.liquid %}

</div>
