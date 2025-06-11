---
layout: default
permalink: /notes/
title: notes
description: A collection of my personal notes.
nav: true
nav_order: 6
pagination:
  enabled: true
  collection: notes
  permalink: /page/:num/
  per_page: 5
  sort_field: date
  sort_reverse: true
  trail:
    before: 1 # The number of links before the current page
    after: 3 # The number of links after the current page
---

{% for note in paginator.posts %}

  <h2><a href="{{ note.url }}">{{ note.title }}</a></h2>
  <p>{{ note.excerpt }}</p>
{% endfor %}

这里存储着我的个人笔记，内容包括但不限于：

- 学习笔记
- 读书笔记

{% if paginator.total_pages > 1 %}
{% include pagination.html %}
{% endif %}
