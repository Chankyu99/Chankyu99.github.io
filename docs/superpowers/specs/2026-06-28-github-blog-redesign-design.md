# GitHub Blog Redesign Design

## Context

The site is a Jekyll blog built on Chirpy. It currently presents the default post feed as the home page, uses a fixed left sidebar with navigation buttons, and shows recent updates and popular tags in a right panel on wide screens.

The redesign turns the home page into a profile-led entry point while preserving Chirpy's recognizable shell, post-reading experience, search behavior, and mobile off-canvas sidebar. The blog remains both a job-facing AI engineer portfolio and a structured learning archive.

## Goals

- Make the home page immediately identify Chankyu Lee as an AI engineer.
- Keep recent writing and representative projects visible without turning the home page into a marketing landing page.
- Replace generic sidebar navigation with a compact, responsive category tree.
- Preserve Chirpy's post pages, search, category archives, responsive shell, and accessibility conventions.
- Make all profile copy, project metadata, and thumbnail images editable through Jekyll configuration or post front matter.
- Establish an SEO, comments, analytics, and future monetization foundation.

## Non-Goals

- Replacing Chirpy with another theme or frontend framework.
- Rewriting existing post bodies.
- Activating advertisements before an AdSense account and site are approved.
- Building a custom backend, account system, or comments database.
- Automatically selecting a "best" image from arbitrary post content.

## Information Architecture

### Sidebar

The desktop sidebar keeps the current avatar, site name, tagline, background image, and social links. The avatar and site name link to the home page.

The existing `HOME`, `BLOG`, `CATEGORIES`, `TAGS`, `ARCHIVES`, and `ABOUT ME` navigation buttons are removed. They are replaced by four top-level native disclosure controls:

1. `공부 기록`
2. `프로젝트`
3. `취준 기록`
4. `일상`

Each control expands to category archive links. The initial hierarchy is:

- `공부 기록`: 알고리즘, CV, 데이터 사이언스, 선형대수, 머신러닝·딥러닝, SQL, 통계
- `프로젝트`: 개인 프로젝트, 팀 프로젝트
- `취준 기록`: 모두연 DS 7기, 인턴, 자격증·면접
- `일상`: 취미, 여행, 독서

The implementation uses semantic `details` and `summary` elements so expansion works without custom JavaScript. Category names and post counts come from Jekyll collections rather than hardcoded counts. The active category is visually highlighted and its parent group opens automatically.

On the home page, `공부 기록` is open by default because it contains most of the existing archive. On post and category pages, the group containing the current post or category opens instead. The other groups remain collapsed to keep the sidebar compact.

On mobile, the existing Chirpy sidebar trigger, mask, and off-canvas behavior remain unchanged. The category tree appears inside that off-canvas sidebar.

### Home Page

The home page removes Chirpy's right-side recent-update and popular-tag panel. Its main area uses the available width and follows this order:

1. Identity and search row
2. Profile introduction
3. Recent posts
4. Project gallery

No `BLOG에서 모두 보기` or `프로젝트 글 모아보기` links appear. Search and category archives are the primary content-navigation mechanisms.

#### Identity And Search Row

The left side displays an editable site identity title and optional subtitle. The right side contains the existing Chirpy search control. The row does not display the default `Home` breadcrumb.

The copy is configured in `_config.yml`:

```yaml
home_identity:
  title: "페이지 정체성 문구"
  subtitle: "AI 프로젝트와 학습 기록"
```

If either value is empty, its element is omitted without leaving an empty gap.

#### Profile Introduction

The profile section is unframed and visually integrated with the page. It contains:

- A professional profile photograph
- A short role label
- One prominent identity statement
- A supporting paragraph
- GitHub and email links

The initial copy may be derived from the supplied resume, but all text remains editable in `_config.yml`. The professional image uses `assets/img/profile.jpg`; the sidebar avatar remains independent.

#### Recent Posts

The section shows the three latest visible posts in Chirpy-style horizontal cards. Each card contains:

- Title
- Short description or generated summary
- Category and publication date
- A right-aligned thumbnail

The thumbnail uses the post's existing `image.path` or string-valued `image` front matter. If no image is configured, the card becomes text-only and the text column expands. The layout does not scrape the first image from Markdown because that behavior would be brittle and difficult to control.

#### Project Gallery

The section shows up to four representative project posts in a responsive two-column gallery inspired by a Notion database gallery. Each card contains:

- Cover image
- Project title
- One-sentence description
- Compact property badges such as project type, team type, and year

Projects are selected with explicit front matter rather than inferred from title text:

```yaml
categories: [프로젝트, 팀 프로젝트]
home_project: true
image:
  path: /assets/img/projects/example-cover.png
  alt: 프로젝트 핵심 화면
project_meta:
  topic: RAG
  team: Team
  year: 2026
```

Only posts with `home_project: true` appear in the home gallery. They are ordered by post date and limited to four. If fewer than four are configured, the gallery renders only the available cards. If a project has no image, its card uses a restrained text-only cover treatment.

## Category Migration

Existing posts are migrated from the current category names to the four approved Korean top-level groups. Post permalinks remain unchanged because permalink generation is title-based, but category archive URLs will change.

Technology names such as `RAG`, `LangChain`, `ChromaDB`, `OpenCV`, and `Scikit-learn` remain tags. Categories describe the purpose or content type of a post; tags describe technologies and topics.

The migration must preserve every post in one of the four top-level groups. A build-time or test script verifies that no visible post is uncategorized.

## Visual System

- Preserve Chirpy's light canvas and sidebar composition.
- Replace the green mockup accent with a restrained blue accent compatible with the current theme.
- Use thin neutral borders, small radii, and minimal shadows.
- Keep the profile section unframed; cards are reserved for recent posts and project gallery items.
- Use existing typography unless a font change is separately approved.
- Avoid decorative gradients and unrelated visual effects.

## Responsive Behavior

- At desktop widths, the fixed sidebar remains visible and the home content occupies the remaining width.
- At Chirpy's existing mobile breakpoint, the sidebar becomes off-canvas.
- Profile content becomes one column with the image before the text.
- Recent post cards keep their thumbnail when enough width exists; on narrow phones the thumbnail moves above or below the text rather than becoming unreadably small.
- Project gallery changes from two columns to one.
- Long titles clamp without changing card dimensions or causing horizontal overflow.

## SEO And Discoverability

The implementation preserves `jekyll-seo-tag` and adds or verifies:

- Accurate site title, description, canonical URL, and social profile data
- Per-post `description`, representative `image`, and meaningful image `alt`
- Sitemap and crawlable robots configuration
- Google Search Console verification support
- JSON-LD for the home profile page and blog posts where it adds accurate, visible information
- Breadcrumb structured data on post and archive pages
- Open Graph and social preview images

Structured data must describe visible page content and pass Google's Rich Results Test. It improves machine understanding but does not guarantee rich-result placement.

Existing JSON-LD emitted by `jekyll-seo-tag` is inspected before adding custom markup. Custom markup must extend missing profile or article details without emitting conflicting duplicate entities.

## Comments

Post pages use Giscus backed by GitHub Discussions. Configuration stays in `_config.yml` using Chirpy's existing comment provider fields. The mapping uses the post pathname so title edits do not create a second discussion.

Comments are enabled only on posts. Home, category, tag, and archive pages do not load the Giscus script. If Giscus is not configured, post pages render normally without an empty comments container.

## Analytics

Analytics uses one of Chirpy's supported providers and is configured entirely through `_config.yml`. No analytics identifier is committed until the user supplies it. Development builds do not emit production analytics scripts.

## Monetization Readiness

The site includes two optional ad-slot locations: one below the article body and before post navigation/comments, and one below the desktop post-side panel content. The home page, profile introduction, category sidebar, and first screen of post content contain no advertising. The desktop side-panel slot is omitted at breakpoints where that panel is hidden. Ad slots are disabled by default.

Activation requires:

- An approved AdSense account and site
- A publisher identifier supplied outside the design document
- A privacy policy and required disclosures
- An `ads.txt` file when supplied by the advertising provider
- A compliant consent-management flow where required, including a Google-certified CMP for relevant EEA, UK, and Swiss traffic

When disabled or unconfigured, ad includes emit no script, markup, or reserved whitespace.

The privacy policy is linked from the site footer so it remains reachable after the sidebar navigation buttons are removed.

## Accessibility

- Category toggles use semantic disclosure elements and remain keyboard operable.
- Focus states are visible and meet contrast requirements.
- Profile, post, and project images have meaningful alt text.
- Search retains its accessible label.
- Links are distinguishable without relying only on color.
- Reduced-motion preferences are respected; no essential interaction depends on animation.

## Error And Fallback Behavior

- Missing home identity values collapse cleanly.
- Missing post or project images produce text-only cards.
- Missing project metadata omits only the affected property badge.
- Empty categories are not shown in the sidebar.
- If no projects are marked for the home gallery, the section is omitted.
- If comments, analytics, or advertising identifiers are absent, those integrations remain disabled without build errors.

## Implementation Boundaries

Expected local changes are limited to:

- Home and shared layout overrides under `_layouts/`
- Sidebar and metadata includes under `_includes/`
- Focused Sass files under `_sass/` and the existing custom stylesheet entrypoint
- `_config.yml` for editable identity and integration settings
- Post front matter for categories, images, and selected project metadata
- Small verification tests or scripts under the existing test/tool structure

Existing Chirpy IDs used by compiled JavaScript, including sidebar, mask, search, and main-wrapper hooks, must not be renamed.

## Verification

The implementation is complete when:

- Jekyll builds successfully in development and production modes.
- Existing lint checks pass.
- Home, post, category, search, and mobile sidebar flows are verified in a browser.
- Desktop and mobile screenshots show no overlap, clipping, horizontal overflow, or blank images.
- Category disclosures work with mouse and keyboard.
- Every visible post belongs to an approved top-level category.
- Recent and project thumbnails use explicit front matter and fall back cleanly.
- SEO metadata and structured data validate.
- Giscus loads only when configured and only on post pages.
- Analytics and ad scripts are absent in development and when disabled.

## External References

- Google Search structured data: https://developers.google.com/search/docs/appearance/structured-data/intro-structured-data
- Google AdSense eligibility: https://support.google.com/adsense/answer/9724
- Google EU user consent policy: https://support.google.com/adsense/answer/7670013
- Giscus: https://github.com/giscus/giscus
