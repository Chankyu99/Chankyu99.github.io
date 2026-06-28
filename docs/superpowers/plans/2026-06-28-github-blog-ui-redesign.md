# GitHub Blog UI Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the default Chirpy home feed and navigation sidebar with the approved profile home, thumbnail feed, project gallery, and Korean category disclosures.

**Architecture:** Preserve Chirpy's default shell and JavaScript hooks, but override the home, sidebar, topbar, and default layouts locally. Split the home into focused Liquid includes, drive editable copy from `_config.yml`, and drive categories and gallery cards from post front matter. Add a generated-site assertion script because this static Jekyll project has no unit-test suite.

**Tech Stack:** Jekyll 4, Liquid, Chirpy 7.5, Sass, Bootstrap 5, Ruby/Nokogiri, html-proofer

---

## File Map

- Create `_data/sidebar_categories.yml`: approved Korean category hierarchy.
- Create `_includes/sidebar-categories.html`: semantic sidebar disclosure tree.
- Create `_includes/home-profile.html`: editable profile introduction.
- Create `_includes/home-recent-posts.html`: latest three Chirpy-style cards.
- Create `_includes/home-project-gallery.html`: up to four selected project cards.
- Create `_sass/pages/_profile-home.scss`: home-only responsive styling.
- Create `tools/test-home-structure.rb`: generated HTML and front-matter assertions.
- Modify `_config.yml`: editable identity/profile and home limits.
- Modify `_includes/sidebar.html`: replace tab navigation with the category include.
- Modify `_includes/topbar.html`: replace the desktop Home breadcrumb with identity copy.
- Modify `_layouts/default.html`: remove the right panel and widen content only on home.
- Modify `_layouts/home.html`: compose the new home includes and remove pagination.
- Modify `_sass/pages/_index.scss`: load the profile-home styles.
- Modify `_sass/layout/_sidebar.scss`: category-tree and compact-sidebar styling.
- Modify `_sass/layout/_topbar.scss`: identity/search alignment.
- Modify `.gitignore`: ignore `.superpowers/` brainstorming artifacts.
- Modify post front matter under `_posts/`: migrate categories and add selected images/project metadata.
- Delete `_tabs/about.md`: remove the obsolete About Me tab page.

### Task 1: Add Generated-Site Contract Tests

**Files:**
- Create: `tools/test-home-structure.rb`
- Modify: `.gitignore`

- [ ] **Step 1: Ignore brainstorming artifacts**

Append this entry to `.gitignore`:

```gitignore
# Superpowers visual brainstorming artifacts
.superpowers/
```

- [ ] **Step 2: Create the failing home contract test**

Create `tools/test-home-structure.rb`:

```ruby
#!/usr/bin/env ruby

require 'date'
require 'nokogiri'
require 'yaml'

ALLOWED_TOP_CATEGORIES = ['공부 기록', '프로젝트', '취준 기록', '일상'].freeze

def assert(condition, message)
  raise message unless condition
end

Dir['_posts/**/*.md'].each do |path|
  source = File.read(path)
  front_matter = source[/\A---\s*\n(.*?)\n---\s*\n/m, 1]
  data = YAML.safe_load(front_matter, permitted_classes: [Date, Time], aliases: true)
  next if data['hidden'] == true

  top_category = Array(data['categories']).first
  assert(ALLOWED_TOP_CATEGORIES.include?(top_category), "#{path} has invalid top category #{top_category.inspect}")
end

home = Nokogiri::HTML(File.read('_site/index.html'))

assert(home.at_css('#panel-wrapper').nil?, 'home right panel must not render')
assert(home.at_css('#home-profile'), 'home profile is missing')
assert(home.css('#home-recent-posts .home-post-card').size == 3, 'home must show three recent posts')
assert(home.css('#home-projects .home-project-card').size <= 4, 'home must show at most four projects')
assert(home.css('#sidebar .sidebar-category-group').size == 4, 'sidebar must show four category groups')
assert(home.at_css('#topbar .home-identity'), 'home identity is missing from topbar')
assert(home.at_css('#search-input'), 'search input is missing')

puts 'home structure checks passed'
```

- [ ] **Step 3: Build and verify the contract fails**

Run:

```bash
bundle exec jekyll build
ruby tools/test-home-structure.rb
```

Expected: Jekyll build succeeds, then the Ruby script fails on the first post with an old top-level category.

- [ ] **Step 4: Unstage visual mockups without deleting them**

Run:

```bash
git restore --staged .superpowers
git status --short
```

Expected: `.superpowers/` no longer appears because it is ignored; no mockup is deleted.

- [ ] **Step 5: Commit the test contract**

```bash
git add .gitignore tools/test-home-structure.rb
git commit -m "test: define blog home contracts"
```

### Task 2: Configure Profile Data And Migrate Categories

**Files:**
- Create: `_data/sidebar_categories.yml`
- Modify: `_config.yml`
- Modify: `_posts/**/*.md`
- Delete: `_tabs/about.md`

- [ ] **Step 1: Add editable home configuration**

Add below the existing `avatar` setting in `_config.yml`:

```yaml
home_identity:
  title: "페이지 정체성 문구"
  subtitle: "AI 프로젝트와 학습 기록"

home_profile:
  image: "/assets/img/profile.jpg"
  image_alt: "이찬규 프로필 사진"
  role: "AI Engineer · Applied Data"
  headline: "사용자가 느끼는 단순함을 최고의 성능이라 믿는 AI 엔지니어입니다."
  description: >-
    복잡한 데이터와 AI 파이프라인을 실제 의사결정과 편리한 사용자 경험으로 연결합니다.
  recent_posts_limit: 3
  projects_limit: 4
```

- [ ] **Step 2: Define the sidebar category groups**

Create `_data/sidebar_categories.yml`:

```yaml
- title: 공부 기록
  children:
    - 알고리즘
    - CV
    - 데이터 사이언스
    - 선형대수
    - 머신러닝·딥러닝
    - SQL
    - 통계
- title: 프로젝트
  children:
    - 개인 프로젝트
    - 팀 프로젝트
- title: 취준 기록
  children:
    - 모두연 DS 7기
    - 인턴
    - 자격증·면접
- title: 일상
  children:
    - 취미
    - 여행
    - 독서
```

- [ ] **Step 3: Migrate every post category**

Apply this exact mapping to the front matter of all posts:

```ruby
CATEGORY_MAP = {
  ['Development', 'CS & Algorithm'] => ['공부 기록', '알고리즘'],
  ['Development', 'CV'] => ['공부 기록', 'CV'],
  ['Development', 'Data Science'] => ['공부 기록', '데이터 사이언스'],
  ['Development', 'Linear Algebra'] => ['공부 기록', '선형대수'],
  ['Development', 'ML&DL'] => ['공부 기록', '머신러닝·딥러닝'],
  ['Development', 'SQL'] => ['공부 기록', 'SQL'],
  ['Development', 'Statistics'] => ['공부 기록', '통계'],
  ['Career', 'ModuLABS DS 7th'] => ['취준 기록', '모두연 DS 7기'],
  ['Career', 'UOS Intern'] => ['취준 기록', '인턴'],
  ['Project', 'Personal'] => ['프로젝트', '개인 프로젝트'],
  ['Project', 'RAG'] => ['프로젝트', '팀 프로젝트']
}.freeze
```

Preserve every other front-matter key and every post body byte-for-byte.

- [ ] **Step 4: Add explicit thumbnails and project metadata**

Add these values to the selected posts:

```yaml
# _posts/Development/CV/2026-06-23-CV-2.md
image:
  path: /assets/img/cv/chapter3/chapter3-01.png
  alt: RGB 색 공간을 설명하는 도식

# _posts/Project/Personal/2026-06-16-RAG.md
image:
  path: /assets/img/langchainton4.png
  alt: 기내뭐돼 RAG 파이프라인
home_project: true
project_meta:
  topic: RAG
  team: 팀 프로젝트
  year: 2026

# _posts/Development/CV/2026-06-15-CV.md
image:
  path: /assets/img/cv/chapter2/chapter2-01.png
  alt: OpenCV 영상 처리 실습 화면

# _posts/Project/Personal/2025-12-03-project_data_transformation.md
image:
  path: /assets/img/data_cleaning.png
  alt: 데이터 전처리 과정
home_project: true
project_meta:
  topic: Data
  team: 개인 프로젝트
  year: 2025
```

- [ ] **Step 5: Remove the obsolete About tab**

Delete `_tabs/about.md`. The profile content now lives on the home page.

- [ ] **Step 6: Build and run the category assertion**

Run:

```bash
bundle exec jekyll build
ruby tools/test-home-structure.rb
```

Expected: category validation passes, then the test fails with `home right panel must not render`.

- [ ] **Step 7: Commit content configuration**

```bash
git add _config.yml _data/sidebar_categories.yml _posts _tabs/about.md
git commit -m "refactor: organize blog content categories"
```

### Task 3: Replace Sidebar Navigation With Category Disclosures

**Files:**
- Create: `_includes/sidebar-categories.html`
- Modify: `_includes/sidebar.html:24-42`

- [ ] **Step 1: Create the sidebar category include**

Create `_includes/sidebar-categories.html`:

```liquid
<nav class="sidebar-categories" aria-label="글 카테고리">
  {% for group in site.data.sidebar_categories %}
    {% assign group_open = false %}
    {% if page.layout == 'home' and group.title == '공부 기록' %}
      {% assign group_open = true %}
    {% elsif page.categories contains group.title or page.title == group.title or group.children contains page.title %}
      {% assign group_open = true %}
    {% endif %}

    <details class="sidebar-category-group"{% if group_open %} open{% endif %}>
      <summary>
        <span class="category-summary-label"><i class="far fa-folder fa-fw"></i>{{ group.title }}</span>
        <i class="fas fa-angle-down category-chevron" aria-hidden="true"></i>
      </summary>
      <ul>
        {% for child in group.children %}
          {% assign post_count = site.categories[child] | size %}
          <li{% if post_count == 0 %} class="empty"{% endif %}>
            {% if post_count > 0 %}
              {% capture category_url %}/categories/{{ child | slugify | url_encode }}/{% endcapture %}
              <a href="{{ category_url | relative_url }}">{{ child }}</a>
            {% else %}
              <span>{{ child }}</span>
            {% endif %}
            <small>{{ post_count }}</small>
          </li>
        {% endfor %}
      </ul>
    </details>
  {% endfor %}
</nav>
```

- [ ] **Step 2: Replace the old tab navigation**

In `_includes/sidebar.html`, replace the `<nav>` containing HOME and `site.tabs` with:

```liquid
{% include sidebar-categories.html %}
```

Keep `#sidebar`, `.profile-wrapper`, `#avatar`, `.site-title`, `.sidebar-bottom`, and all social links unchanged.

- [ ] **Step 3: Build and inspect generated sidebar markup**

Run:

```bash
bundle exec jekyll build
rg -n "sidebar-category-group|공부 기록|프로젝트|취준 기록|일상" _site/index.html
```

Expected: four disclosure groups appear in `_site/index.html`.

- [ ] **Step 4: Commit the sidebar structure**

```bash
git add _includes/sidebar.html _includes/sidebar-categories.html
git commit -m "feat: add sidebar category navigation"
```

### Task 4: Widen Home And Move Identity Into The Topbar

**Files:**
- Modify: `_includes/topbar.html:8-40`
- Modify: `_layouts/default.html:26-58`

- [ ] **Step 1: Render editable identity instead of the Home breadcrumb**

Replace the opening `<nav id="breadcrumb">` and its closing tag with this conditional wrapper while keeping the current breadcrumb assignments and loop between the `{% else %}` and `{% endif %}` branches:

```liquid
{% if page.layout == 'home' %}
  <div class="home-identity">
    {% if site.home_identity.title %}<strong>{{ site.home_identity.title }}</strong>{% endif %}
    {% if site.home_identity.subtitle %}<small>{{ site.home_identity.subtitle }}</small>{% endif %}
  </div>
{% else %}
  <nav id="breadcrumb" aria-label="Breadcrumb">
    {% assign paths = page.url | split: '/' %}
    {% for item in paths %}
      {% if forloop.first %}
        <span><a href="{{ '/' | relative_url }}">{{ site.data.locales[include.lang].tabs.home | capitalize }}</a></span>
      {% elsif forloop.last %}
        {% if page.collection == 'tabs' %}
          <span>{{ site.data.locales[include.lang].tabs[item] | default: page.title }}</span>
        {% else %}
          <span>{{ page.title }}</span>
        {% endif %}
      {% elsif page.layout == 'category' or page.layout == 'tag' %}
        <span><a href="{{ item | append: '/' | relative_url }}">{{ site.data.locales[include.lang].tabs[item] | default: page.title }}</a></span>
      {% endif %}
    {% endfor %}
  </nav>
{% endif %}
```

Keep `#sidebar-trigger`, `#topbar-title`, `#search-trigger`, `#search`, `#search-input`, and `#search-cancel` unchanged.

- [ ] **Step 2: Remove the right panel only on home**

In `_layouts/default.html`, assign home-aware column classes:

```liquid
{% assign home_layout = false %}
{% if page.layout == 'home' %}{% assign home_layout = true %}{% endif %}

<main aria-label="Main Content" class="col-12 col-lg-11 {% if home_layout %}col-xl-11 home-main{% else %}col-xl-9{% endif %} px-md-4">
  {% if layout.layout == 'default' %}
    {% include refactor-content.html content=content lang=lang %}
  {% else %}
    {{ content }}
  {% endif %}
</main>

{% unless home_layout %}
  <aside aria-label="Panel" id="panel-wrapper" class="col-xl-3 ps-2 text-muted">
    <div class="access">
      {% include_cached update-list.html lang=lang %}
      {% include_cached trending-tags.html lang=lang %}
    </div>
    {% for _include in layout.panel_includes %}
      {% assign _include_path = _include | append: '.html' %}
      {% include {{ _include_path }} lang=lang %}
    {% endfor %}
  </aside>
{% endunless %}
```

Apply the same `col-xl-11`/`col-xl-9` branch to `#tail-wrapper` so the footer aligns with the home content.

- [ ] **Step 3: Build and verify the panel contract**

Run:

```bash
bundle exec jekyll build
ruby tools/test-home-structure.rb
```

Expected: the test passes the `home right panel must not render` assertion and still fails on missing home sections.

- [ ] **Step 4: Commit the shell changes**

```bash
git add _includes/topbar.html _layouts/default.html
git commit -m "feat: adapt chirpy shell for profile home"
```

### Task 5: Build The Profile Home Components

**Files:**
- Create: `_includes/home-profile.html`
- Create: `_includes/home-recent-posts.html`
- Create: `_includes/home-project-gallery.html`
- Modify: `_layouts/home.html`

- [ ] **Step 1: Create the profile include**

Create `_includes/home-profile.html`:

```liquid
<section id="home-profile" class="home-profile">
  {% capture profile_image_url %}{% include media-url.html src=site.home_profile.image %}{% endcapture %}
  <img src="{{ profile_image_url }}" alt="{{ site.home_profile.image_alt | escape }}">
  <div class="home-profile-copy">
    <p class="home-profile-role">{{ site.home_profile.role }}</p>
    <h1>{{ site.home_profile.headline }}</h1>
    <p>{{ site.home_profile.description }}</p>
    <div class="home-profile-links">
      <a href="https://github.com/{{ site.github.username }}" rel="noopener noreferrer">GitHub</a>
      <a href="mailto:{{ site.social.email }}">Email</a>
    </div>
  </div>
</section>
```

- [ ] **Step 2: Create the recent-post cards**

Create `_includes/home-recent-posts.html`:

```liquid
<section id="home-recent-posts" class="home-section">
  <h2>최근 기록</h2>
  <div class="home-post-list">
    {% assign recent_posts = site.posts | where_exp: 'post', 'post.hidden != true' %}
    {% for post in recent_posts limit: site.home_profile.recent_posts_limit %}
      <article class="home-post-card">
        <a href="{{ post.url | relative_url }}" class="home-post-body">
          <h3>{{ post.title }}</h3>
          <p>{% include post-summary.html max_length=150 %}</p>
          <div class="home-post-meta">
            <span>{{ post.categories | last }}</span>
            {% include datetime.html date=post.date lang=lang %}
          </div>
        </a>
        {% if post.image %}
          {% assign image_src = post.image.path | default: post.image %}
          <a href="{{ post.url | relative_url }}" class="home-post-thumbnail" aria-label="{{ post.title | escape }}">
            <img src="{{ image_src | relative_url }}" alt="{{ post.image.alt | default: post.title | escape }}">
          </a>
        {% endif %}
      </article>
    {% endfor %}
  </div>
</section>
```

- [ ] **Step 3: Create the project gallery**

Create `_includes/home-project-gallery.html`:

```liquid
{% assign home_projects = site.posts | where: 'home_project', true %}
{% if home_projects.size > 0 %}
  <section id="home-projects" class="home-section">
    <h2>프로젝트 모음</h2>
    <div class="home-project-grid">
      {% for post in home_projects limit: site.home_profile.projects_limit %}
        <article class="home-project-card">
          <a href="{{ post.url | relative_url }}">
            {% if post.image %}
              {% assign image_src = post.image.path | default: post.image %}
              <div class="home-project-cover">
                <img src="{{ image_src | relative_url }}" alt="{{ post.image.alt | default: post.title | escape }}">
              </div>
            {% endif %}
            <div class="home-project-body">
              <h3>{{ post.title }}</h3>
              <p>{% include post-summary.html max_length=110 %}</p>
              <div class="home-project-properties">
                {% if post.project_meta.topic %}<span class="primary">{{ post.project_meta.topic }}</span>{% endif %}
                {% if post.project_meta.team %}<span>{{ post.project_meta.team }}</span>{% endif %}
                {% if post.project_meta.year %}<span>{{ post.project_meta.year }}</span>{% endif %}
              </div>
            </div>
          </a>
        </article>
      {% endfor %}
    </div>
  </section>
{% endif %}
```

- [ ] **Step 4: Replace the paginated home layout**

Replace the body of `_layouts/home.html` after `{% include lang.html %}` with:

```liquid
<div class="profile-home px-xl-1">
  {% include home-profile.html %}
  {% include home-recent-posts.html %}
  {% include home-project-gallery.html %}
</div>
```

Remove pinned-post pagination and `post-paginator.html`; category pages and search remain the complete archive surfaces.

- [ ] **Step 5: Build and run the generated-site contract**

Run:

```bash
bundle exec jekyll build
ruby tools/test-home-structure.rb
```

Expected: `home structure checks passed`.

- [ ] **Step 6: Commit the home structure**

```bash
git add _layouts/home.html _includes/home-profile.html _includes/home-recent-posts.html _includes/home-project-gallery.html
git commit -m "feat: build profile-led blog home"
```

### Task 6: Implement Responsive Visual Styling

**Files:**
- Create: `_sass/pages/_profile-home.scss`
- Modify: `_sass/pages/_index.scss`
- Modify: `_sass/layout/_sidebar.scss`
- Modify: `_sass/layout/_topbar.scss`

- [ ] **Step 1: Load the new page stylesheet**

Add to `_sass/pages/_index.scss`:

```scss
@forward 'profile-home';
```

- [ ] **Step 2: Add profile-home styles**

Create `_sass/pages/_profile-home.scss` with these component contracts:

```scss
@use '../abstracts/breakpoints' as bp;

.profile-home {
  --home-accent: #2563eb;
  padding-top: 1.5rem;
}

.home-profile {
  display: grid;
  grid-template-columns: 10rem minmax(0, 1fr);
  gap: 2rem;
  align-items: center;
  padding: 2rem 0 2.5rem;
  border-bottom: 1px solid var(--main-border-color);

  > img {
    width: 10rem;
    aspect-ratio: 4 / 5;
    object-fit: cover;
    object-position: top;
    border-radius: 0.375rem;
    box-shadow: 0.75rem 0.75rem 0 #e8efff;
  }
}

.home-profile-role,
.home-post-meta > :first-child {
  color: var(--home-accent);
  font-weight: 700;
}

.home-section {
  padding-top: 2rem;

  + .home-section {
    margin-top: 2rem;
    border-top: 1px solid var(--main-border-color);
  }
}

.home-post-list {
  display: grid;
  gap: 0.75rem;
}

.home-post-card {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 11rem;
  min-height: 9rem;
  overflow: hidden;
  border: 1px solid var(--card-border-color);
  border-radius: 0.5rem;
  background: var(--card-bg);
}

.home-post-body,
.home-project-body {
  padding: 1.25rem;
}

.home-post-thumbnail img,
.home-project-cover img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.home-project-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 1rem;
}

.home-project-card {
  overflow: hidden;
  border: 1px solid var(--card-border-color);
  border-radius: 0.5rem;
  background: var(--card-bg);
}

.home-profile-copy h1,
.home-section h2,
.home-post-card h3,
.home-project-card h3 {
  color: var(--heading-color);
  letter-spacing: 0;
}

.home-post-body,
.home-project-card > a {
  color: inherit;
  text-decoration: none;
}

.home-post-meta,
.home-project-properties {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  margin-top: auto;
  color: var(--text-muted-color);
  font-size: 0.8rem;
}

.home-project-properties span {
  padding: 0.25rem 0.5rem;
  border-radius: 999px;
  background: var(--tag-bg);
}

.home-project-properties .primary {
  color: #1d4ed8;
  background: #e8efff;
}

.home-post-card:focus-within,
.home-project-card:focus-within {
  outline: 2px solid var(--home-accent);
  outline-offset: 2px;
}

.home-post-card h3,
.home-project-card h3 {
  display: -webkit-box;
  overflow: hidden;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
}

.home-project-cover {
  aspect-ratio: 16 / 8;
  overflow: hidden;
  border-bottom: 1px solid var(--card-border-color);
}

@include bp.lt(bp.get(md)) {
  .home-profile,
  .home-project-grid {
    grid-template-columns: 1fr;
  }

  .home-post-card {
    grid-template-columns: 1fr;
  }

  .home-post-thumbnail {
    grid-row: 1;
    aspect-ratio: 16 / 7;
  }
}
```

- [ ] **Step 3: Style the sidebar disclosures**

Add nested rules under `#sidebar` in `_sass/layout/_sidebar.scss` for `.sidebar-categories`, `.sidebar-category-group`, `summary`, `ul`, `li`, links, counts, `[open]`, `.empty`, hover, and `:focus-visible`. Use native disclosure state and rotate `.category-chevron` only when open.

- [ ] **Step 4: Style the identity/search row**

Add `.home-identity` rules under `#topbar` in `_sass/layout/_topbar.scss`. It occupies the breadcrumb position on desktop, clamps long text, and remains hidden below the existing `lg` breakpoint where `#topbar-title` takes over.

- [ ] **Step 5: Run lint and build checks**

Run:

```bash
npm test
bundle exec jekyll build
ruby tools/test-home-structure.rb
```

Expected: all commands exit 0.

- [ ] **Step 6: Commit styling**

```bash
git add _sass/pages/_profile-home.scss _sass/pages/_index.scss _sass/layout/_sidebar.scss _sass/layout/_topbar.scss
git commit -m "style: polish responsive blog home"
```

### Task 7: Browser Verification And Final Integration

**Files:**
- Modify only files required by verification findings.

- [ ] **Step 1: Start the local site**

Run:

```bash
bundle exec jekyll serve --host 127.0.0.1 --port 4001
```

Expected: site is available at `http://127.0.0.1:4001/`.

- [ ] **Step 2: Verify desktop behavior**

At 1440x900, verify:

- Sidebar shows four Korean category disclosures.
- Avatar and site title return home.
- Home identity and search share the top row.
- Profile uses the full home content width and the right panel is absent.
- Exactly three recent cards render with right-side thumbnails.
- Project cards render as a two-column gallery with front-matter covers.

- [ ] **Step 3: Verify mobile behavior**

At 390x844, verify:

- Sidebar trigger opens the existing off-canvas sidebar.
- Category disclosures remain keyboard/click operable.
- Profile, recent cards, and project cards stack in one column.
- No text, image, or button overlaps or causes horizontal scrolling.

- [ ] **Step 4: Verify search and category navigation**

Search for `RAG`, open a result, return home through the site identity, expand `프로젝트`, and open both project subcategories. Verify generated category links resolve without html-proofer errors.

- [ ] **Step 5: Run the full production check**

Run:

```bash
bash tools/test.sh
```

Expected: production build and html-proofer both exit 0.

- [ ] **Step 6: Review scope and commit any verification fixes**

Run:

```bash
git status --short
git diff --check
git diff
```

Ensure `.superpowers/` is absent and no unrelated files are included. Commit only genuine verification fixes with a conventional commit message.

---

## Follow-Up Plans

After this UI plan is shipped and verified, create separate plans for:

1. SEO metadata, Search Console, and structured-data validation.
2. Giscus comments and analytics configuration.
3. Disabled-by-default AdSense slots, privacy policy, consent flow, and `ads.txt` support.
