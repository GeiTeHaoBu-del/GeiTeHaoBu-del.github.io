---
title: Hexo+GitHub Pages博客搭建指南：从零到自动部署
date: 2026-06-10
tags:
  - Hexo
  - GitHub Pages
  - CI/CD
  - NexT
categories:
  - Hexo建站
summary: 记录从零搭建Hexo博客并部署到GitHub Pages的全过程，涵盖环境准备、主题配置、深浅色切换、自定义组件注入、GitHub Actions自动部署等关键步骤。
description: 记录从零搭建Hexo博客并部署到GitHub Pages的全过程，涵盖环境准备、主题配置、深浅色切换、自定义组件注入、GitHub Actions自动部署等关键步骤。
---

## 问题引入：为什么选择Hexo？

作为AI应用开发方向的学习者，我需要一个技术博客来沉淀项目经历、展示工程能力。Hexo的优势：

- **Markdown写作**：技术文章天然适合Markdown，代码块、公式、表格无缝支持
- **静态站点**：GitHub Pages免费托管，零服务器成本
- **主题生态**：NexT主题成熟稳定，高度可定制
- **自动部署**：push代码即可触发GitHub Actions构建发布，全流程自动化

最终效果：**push一篇Markdown → GitHub Actions自动构建 → 1分钟内线上更新**。

## 原理讲解

### Hexo的工作流程

Hexo是一个基于Node.js的静态站点生成器，核心流程：

```
Markdown源文件 → Hexo渲染引擎 → HTML静态页面 → 部署到Web服务器
```

关键概念：

| 概念 | 说明 |
|------|------|
| `source/_posts/` | 文章Markdown存放目录 |
| `_config.yml` | 站点配置（标题、URL、部署方式等） |
| `_config.next.yml` | 主题配置（NexT专用，覆盖默认配置） |
| `source/_data/` | 自定义数据文件（样式、脚本、模板注入） |
| `themes/` | 主题目录（npm安装的主题在node_modules中） |

### GitHub Pages + Actions部署原理

```
本地开发(push main) → GitHub Actions触发 → npm install → hexo generate → 产物推送到gh-pages分支 → GitHub Pages提供服务
```

- `main`分支：存放源码（Markdown、配置、样式）
- `gh-pages`分支：存放构建产物（纯HTML/CSS/JS），由Actions自动推送

## 代码实现

### 1. 环境准备

```bash
# 安装Node.js（推荐18+LTS，我用的是20）
# Windows: https://nodejs.org/ 下载安装
# 验证
node -v   # v20.x.x
npm -v    # 10.x.x

# 安装Hexo CLI
npm install -g hexo-cli

# 初始化博客
hexo init my-blog
cd my-blog
npm install
```

### 2. 安装NexT主题

```bash
npm install hexo-theme-next
```

然后在`_config.yml`中设置：

```yaml
theme: next
```

NexT的主题配置不要改`node_modules/hexo-theme-next/_config.yml`，而是在项目根目录创建`_config.next.yml`，Hexo会自动合并。这样主题升级时不会丢失自定义配置。

### 3. 站点配置`_config.yml`关键项

```yaml
title: 你的博客标题
subtitle: '副标题'
description: 站点描述，用于SEO
keywords:
  - 关键词1
  - 关键词2
author: 你的名字
language: zh-CN
timezone: 'Asia/Shanghai'

url: https://username.github.io
permalink: posts/:title/

theme: next

# 每篇文章必须有description字段，否则首页会展示全文
```

### 4. 主题配置`_config.next.yml`关键项

```yaml
scheme: Pisces           # 双栏布局，适合技术博客
theme_color:
  light: "#7c3aed"       # 浅色主色
  dark: "#a855f7"        # 深色主色

font:
  enable: false          # 关闭Google Fonts，国内加载慢

local_search:
  enable: true           # 本地搜索功能

mermaid:
  enable: true           # Mermaid图表支持

darkmode: false          # 关闭系统跟随，改用手动切换按钮
```

### 5. 自定义样式和组件注入

NexT支持通过`source/_data/`目录注入自定义文件，这是最强大的定制方式：

```yaml
# _config.next.yml
custom_file_path:
  sidebar: source/_data/sidebar.njk    # 自定义左侧sidebar
  bodyEnd: source/_data/body-end.njk   # body末尾注入HTML/JS
  variable: source/_data/variables.styl # 覆盖CSS变量
  style: source/_data/styles.styl      # 自定义样式
```

**深浅色切换**的实现思路：

1. 关闭NexT自带的`darkmode`（它基于`prefers-color-scheme`媒体查询，无切换按钮）
2. 在`body-end.njk`中添加切换按钮和JS，通过给`<html>`添加`data-theme="dark"`属性控制
3. 在`styles.styl`中用`html[data-theme="dark"]`选择器覆盖NexT的CSS变量和自定义组件样式
4. 主题偏好保存在`localStorage`，刷新后保持

```javascript
// 核心切换逻辑
function applyTheme(dark) {
  if (dark) {
    html.setAttribute('data-theme', 'dark');
  } else {
    html.removeAttribute('data-theme');
  }
}

// 读取用户偏好
var saved = localStorage.getItem('theme');
applyTheme(saved === 'dark');

// 点击切换
toggle.addEventListener('click', function() {
  var isDark = html.getAttribute('data-theme') === 'dark';
  applyTheme(!isDark);
  localStorage.setItem('theme', isDark ? 'light' : 'dark');
});
```

```styl
// styles.styl 中的暗色覆盖
html[data-theme="dark"]
  --body-bg-color: #0a0a0f
  --content-bg-color: #1a1a2e
  --text-color: #e2e8f0
  --link-color: #c084fc
  // ...更多变量
```

### 6. 文章Front Matter模板

```yaml
---
title: 文章标题
date: 2026-06-10
tags:
  - 标签1
  - 标签2
categories:
  - 分类名
summary: 一句话摘要（列表页展示）
description: 一句话描述（首页展示，必须有，否则首页会展示全文）
---
```

**重要**：`description`字段是首页展示简介的关键。没有`description`的文章会在首页展示全文。

### 7. GitHub Actions自动部署

在`.github/workflows/deploy.yml`中配置：

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches:
      - main

jobs:
  deploy:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          submodules: recursive

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'

      - name: Install dependencies
        run: npm install

      - name: Build Hexo
        run: npx hexo clean && npx hexo generate

      - name: Deploy
        uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./public
          publish_branch: gh-pages
```

部署流程：`git push origin main` → Actions自动构建 → `gh-pages`分支更新 → 线上生效。

### 8. 头像本地化

GitHub头像URL在国内访问不稳定，下载到本地解决：

```bash
mkdir -p source/images
curl -L -o source/images/avatar.png "你的头像URL"
```

配置中使用本地路径：

```yaml
avatar:
  url: /images/avatar.png
```

## 实验结论与反思

### 踩过的坑

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 首页展示全文而非简介 | 文章缺少`description`字段 | Front Matter中必须加`description` |
| Google Fonts加载极慢 | 国内无法访问fonts.googleapis.com | `font.enable: false`，使用系统字体 |
| 头像加载失败 | GitHub avatars域名国内不稳定 | 头像下载到本地`source/images/` |
| NexT darkmode无切换按钮 | 只支持`prefers-color-scheme`媒体查询 | 自己实现JS切换+CSS变量覆盖 |
| 自定义组件暗色模式下变浅 | CSS变量覆盖不完整 | `html[data-theme="dark"]`中覆盖`.sidebar-inner`等所有容器 |
| `_data/body-end.njk`中使用Hexo变量报错 | 某些上下文中`site`变量不可用 | 改用`{{ site.posts.length }}`等Nunjucks模板语法 |

### 项目文件结构

```
├── _config.yml              # 站点配置
├── _config.next.yml         # 主题配置
├── .github/workflows/       # CI/CD
│   └── deploy.yml
├── source/
│   ├── _posts/              # 文章
│   ├── _data/               # 自定义注入
│   │   ├── sidebar.njk      # 左侧sidebar
│   │   ├── body-end.njk     # JS/HTML注入
│   │   ├── variables.styl   # CSS变量覆盖
│   │   └── styles.styl      # 自定义样式
│   ├── images/              # 本地图片
│   │   └── avatar.png
│   ├── about/               # 关于页
│   └── categories/          # 分类页
└── package.json
```

## 面试关联

1. **CI/CD流程设计？** → GitHub Actions自动构建部署，push触发、多步骤编排、权限控制
2. **前端主题切换实现？** → CSS变量 + data属性 + localStorage持久化，不需要框架
3. **静态站点生成原理？** → 模板引擎渲染Markdown为HTML，部署到CDN/静态托管
4. **性能优化实践？** → 字体本地化/禁用、头像本地化、关闭不必要的外部CDN
