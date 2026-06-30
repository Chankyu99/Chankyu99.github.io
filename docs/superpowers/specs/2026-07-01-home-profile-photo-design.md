# Home Profile Photo Update Design

## Goal

Replace only the home profile photo with the user-provided `IMG_0794.jpeg` while preserving the existing sidebar avatar and Chirpy-based layout.

## Scope

- Replace `assets/img/profile.jpg`, which is used only by `home_profile.image`.
- Keep `_config.yml` and the sidebar `avatar` setting unchanged.
- Increase the home profile photo from 160 x 200 px to 200 x 250 px on desktop.
- Increase the mobile photo from 120 x 150 px to 136 x 170 px.
- Preserve the existing 4:5 frame, `object-fit: cover`, and top-aligned crop.

## Implementation

1. Copy the provided JPEG to `assets/img/profile.jpg`.
2. Update the profile grid column and image width in `_sass/pages/_profile-home.scss`.
3. Update the mobile image width in the same stylesheet.
4. Keep all profile text, links, and sidebar presentation unchanged.

## Verification

- Build the Jekyll site successfully.
- Confirm the generated home profile references `/assets/img/profile.jpg`.
- Confirm the rendered image dimensions remain 4:5 on desktop and mobile.
- Run the existing lint and home structure checks.
