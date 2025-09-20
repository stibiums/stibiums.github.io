---
layout: default
permalink: /notes/
title: notes
nav: true
nav_order: 2
description: 学习笔记与技术分享 / Study Notes & Technical Sharing
---

<div class="post">

<div class="header-bar">
  <h1>学习笔记</h1>
  <h2>Study Notes & Technical Sharing</h2>
</div>

<p class="text-center">
  这里收录了我的学习笔记、技术探索和知识总结。内容涵盖机器学习、计算机视觉、编程语言等多个领域。笔记内容使用 claude code 生成，仅供参考，本网站不对任何笔记内容负责。
  <br>
  <em>Here are my study notes, technical explorations, and knowledge summaries covering machine learning, computer vision, programming languages, and more.</em>
</p>

<!-- Filter by categories -->
<div class="tag-category-list">
  <ul class="p-0 m-0">
    <li><i class="fa-solid fa-tag fa-sm"></i> <a href="#ai-ml">人工智能 & 机器学习</a></li>
    <p>&bull;</p>
    <li><i class="fa-solid fa-tag fa-sm"></i> <a href="#programming">编程语言</a></li>
    <p>&bull;</p>
    <li><i class="fa-solid fa-tag fa-sm"></i> <a href="#algorithms">算法与数据结构</a></li>
    <p>&bull;</p>
    <li><i class="fa-solid fa-tag fa-sm"></i> <a href="#systems">系统与工具</a></li>
  </ul>
</div>

<div class="notes-sections">

<!-- AI & Machine Learning Section -->
<section id="ai-ml" class="notes-category">
  <h3><i class="fa-solid fa-brain"></i> 人工智能 & 机器学习</h3>
  <div class="row">
    {% assign ai_posts = site.posts | where_exp: "post", "post.tags contains 'machine-learning' or post.tags contains 'computer-vision' or post.tags contains 'AI' or post.title contains 'ML' or post.title contains 'CV' or post.title contains '机器学习' or post.title contains '计算机视觉'" %}
    {% for post in ai_posts limit: 6 %}
    <div class="col-md-6 mb-3">
      <div class="card h-100">
        <div class="card-body">
          <h5 class="card-title">
            <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
          </h5>
          <p class="card-text">{{ post.description | truncate: 100 }}</p>
          <small class="text-muted">{{ post.date | date: '%Y-%m-%d' }}</small>
        </div>
      </div>
    </div>
    {% endfor %}
  </div>
</section>

<!-- Programming Languages Section -->
<section id="programming" class="notes-category">
  <h3><i class="fa-solid fa-code"></i> 编程语言</h3>
  <div class="row">
    {% assign prog_posts = site.posts | where_exp: "post", "post.tags contains 'rust' or post.tags contains 'python' or post.tags contains 'programming' or post.title contains 'Rust' or post.title contains 'Python'" %}
    {% for post in prog_posts limit: 6 %}
    <div class="col-md-6 mb-3">
      <div class="card h-100">
        <div class="card-body">
          <h5 class="card-title">
            <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
          </h5>
          <p class="card-text">{{ post.description | truncate: 100 }}</p>
          <small class="text-muted">{{ post.date | date: '%Y-%m-%d' }}</small>
        </div>
      </div>
    </div>
    {% endfor %}
  </div>
</section>

<!-- Algorithms & Data Structures Section -->
<section id="algorithms" class="notes-category">
  <h3><i class="fa-solid fa-project-diagram"></i> 算法与数据结构</h3>
  <div class="row">
    {% assign algo_posts = site.posts | where_exp: "post", "post.tags contains 'algorithms' or post.tags contains 'data-structures' or post.title contains 'algorithm' or post.title contains '算法'" %}
    {% for post in algo_posts limit: 6 %}
    <div class="col-md-6 mb-3">
      <div class="card h-100">
        <div class="card-body">
          <h5 class="card-title">
            <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
          </h5>
          <p class="card-text">{{ post.description | truncate: 100 }}</p>
          <small class="text-muted">{{ post.date | date: '%Y-%m-%d' }}</small>
        </div>
      </div>
    </div>
    {% endfor %}
  </div>
</section>

<!-- Systems & Tools Section -->
<section id="systems" class="notes-category">
  <h3><i class="fa-solid fa-cogs"></i> 系统与工具</h3>
  <div class="row">
    {% assign sys_posts = site.posts | where_exp: "post", "post.tags contains 'linux' or post.tags contains 'git' or post.tags contains 'tools' or post.title contains 'ICS' or post.title contains '系统'" %}
    {% for post in sys_posts limit: 6 %}
    <div class="col-md-6 mb-3">
      <div class="card h-100">
        <div class="card-body">
          <h5 class="card-title">
            <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
          </h5>
          <p class="card-text">{{ post.description | truncate: 100 }}</p>
          <small class="text-muted">{{ post.date | date: '%Y-%m-%d' }}</small>
        </div>
      </div>
    </div>
    {% endfor %}
  </div>
</section>

</div>

<!-- All Notes List -->
<hr>
<h3>所有学习笔记 / All Study Notes</h3>
<ul class="post-list">
  {% assign note_posts = site.posts | where_exp: "post", "post.title contains 'note' or post.title contains '笔记' or post.tags contains 'notes'" %}
  {% for post in note_posts %}
  <li>
    <h4>
      <a class="post-title" href="{{ post.url | relative_url }}">{{ post.title }}</a>
    </h4>
    <p>{{ post.description }}</p>
    <p class="post-meta">
      {{ post.date | date: '%Y年%m月%d日' }} &nbsp; &middot; &nbsp;
      {% if post.tags.size > 0 %}
        {% for tag in post.tags %}
        <span class="badge badge-secondary">{{ tag }}</span>
        {% endfor %}
      {% endif %}
    </p>
  </li>
  {% endfor %}
</ul>

</div>
