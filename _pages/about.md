---
layout: about
title: about
permalink: /
subtitle: <button id="bio-language-btn" class="theme-language-btn">EN</button>
profile:
  align: right
  image: touxiang.jpg
  image_circular: true # crops the image to make it circular
  more_info: >
    <p>Peking University, China</p>
    <p>Location: Beijing, China</p>

selected_papers: false # includes a list of papers marked as "selected={true}"
social: true # includes social icons at the bottom of the page

announcements:
  enabled: false # includes a list of news items
  scrollable: true # adds a vertical scroll bar if there are more than 3 news items
  limit: 2 # leave blank to include all the news in the `_news` folder

latest_posts:
  enabled: true
  scrollable: true # adds a vertical scroll bar if there are more than 3 new posts items
  limit: 3 # leave blank to include all the blog posts
---

<!-- Chinese Bio -->
<div class="bio-zh">
嗨！我是北京大学智能科学与技术专业的一名本科生。我的兴趣包括具身智能、机器学习、计算机视觉和自然语言处理。我希望可以运用人工智能解决现实世界中的问题。目前我的水平有限，还在不断学习和探索中。欢迎与我交流和分享经验！如果你对我的学习与生活感兴趣，或者有任何问题，请随时联系我。我的github账号是 <a href="https://github.com/stibiums">stibiums</a>，其中包含了我的一些学习笔记和项目代码。我也会在本网站上分享我的学习经历和心得体会，希望能与大家共同进步。
</div>

<!-- English Bio -->
<div class="bio-en" style="display: none;">
Hi! I am an undergraduate student majoring in Intelligent Science and Technology at Peking University, China. My interests include embodied intelligence, machine learning, computer vision, and natural language processing. I hope to apply artificial intelligence to solve real-world problems. I am still learning and exploring, and my skills are limited for now. Feel free to reach out and share your experiences! If you are interested in my studies and life, or have any questions, please contact me anytime. My GitHub account is <a href="https://github.com/stibiums">stibiums</a>, where you can find some of my study notes and project code. I will also share my learning experiences and insights on this website, hoping to make progress together with everyone.
</div>

<!-- Skills Cloud Section -->
<div class="skills-section">
  <h3>技能标签 / Skills</h3>
  <div class="skills-cloud">
    {% for skill in site.data.homepage.skills %}
      <span class="skill-tag {{ skill.category }} clickable" 
            data-url="{{ skill.url }}" 
            title="{{ skill.tooltip_zh }}">
        {{ skill.name_zh }}
      </span>
    {% endfor %}
  </div>
</div>

<!-- Statistics Section -->
<div class="stats-section">
  <h3>统计信息 / Statistics</h3>
  <div class="stats-grid">
    {% for stat in site.data.homepage.statistics %}
      <div class="stat-card clickable" 
           data-url="{{ stat.url }}" 
           title="{{ stat.tooltip_zh }}">
        <div class="stat-number" id="{{ stat.id }}">-</div>
        <div class="stat-label">{{ stat.name_zh }}</div>
      </div>
    {% endfor %}
  </div>
  
  <!-- GitHub Stats Card -->
  <div class="github-stats">
    <img src="https://github-readme-stats.vercel.app/api?username=stibiums&show_icons=true&theme=transparent&hide_border=true&title_color=b509ac&icon_color=b509ac&text_color=333" alt="GitHub Stats" class="github-stats-img">
    <img src="https://github-readme-stats.vercel.app/api/top-langs/?username=stibiums&layout=compact&theme=transparent&hide_border=true&title_color=b509ac&text_color=333" alt="Top Languages" class="github-langs-img">
  </div>
</div>

<!-- Dynamic Skill Tag Colors from YAML -->
<style>
{% for color_key in site.data.homepage.skill_colors %}
  {% assign color = color_key[1] %}
  .skill-tag.{{ color_key[0] }} {
    background: {{ color.bg }} !important;
    border-color: {{ color.border }} !important;
    color: {{ color.color }} !important;
  }
  
  html[data-theme="dark"] .skill-tag.{{ color_key[0] }} {
    background: {{ color.bg | replace: '0.1', '0.15' }} !important;
  }
{% endfor %}
</style>

<!-- Load Language Switcher Script -->
<script src="/assets/js/bio-language-switcher.js"></script>

<!-- Load Stats Script -->
<script src="/assets/js/homepage-stats.js"></script>
