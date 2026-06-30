# Home Profile Photo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the home-only profile photo and increase its responsive display size without changing the sidebar avatar.

**Architecture:** Keep the existing `home_profile.image` path and replace its JPEG asset in place. Update only the profile grid and image widths in the dedicated home stylesheet, with a generated-site contract covering the agreed desktop and mobile dimensions.

**Tech Stack:** Jekyll, Liquid, SCSS, Ruby/Nokogiri contract test

---

### Task 1: Replace And Resize The Home Profile Photo

**Files:**
- Modify: `tools/test-home-structure.rb`
- Modify: `assets/img/profile.jpg`
- Modify: `_sass/pages/_profile-home.scss`

- [ ] **Step 1: Write the failing size contract**

Add assertions that `_sass/pages/_profile-home.scss` contains a `12.5rem` desktop profile column and width, plus an `8.5rem` mobile width.

- [ ] **Step 2: Run the contract to verify it fails**

Run: `ruby tools/test-home-structure.rb`

Expected: FAIL because the current stylesheet still uses `10rem` and `7.5rem`.

- [ ] **Step 3: Apply the approved asset and dimensions**

Copy the supplied JPEG to `assets/img/profile.jpg`. Change the desktop grid column and image width to `12.5rem`, and the mobile image width to `8.5rem`. Keep `aspect-ratio: 4 / 5`, `object-fit: cover`, and `object-position: top` unchanged.

- [ ] **Step 4: Run focused and full verification**

Run:

```bash
ruby tools/test-home-structure.rb
npm test
bash tools/test.sh
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 5: Commit and push only related files**

```bash
git add assets/img/profile.jpg _sass/pages/_profile-home.scss tools/test-home-structure.rb docs/superpowers/plans/2026-07-01-home-profile-photo.md
git commit -m "feat: update home profile photo"
git push origin main
```
