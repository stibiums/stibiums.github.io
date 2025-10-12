# Repository Guidelines

## Project Structure & Module Organization

- `_pages/` holds evergreen pages; `_posts/` and `_news/` use `YYYY-MM-DD-title.md` slugs for dated content, while `_projects/` showcases portfolio entries.
- `_layouts/`, `_includes/`, and `_sass/` define shared structure and styles; reusable assets live in `assets/` (images under `assets/img`, scripts under `assets/js`).
- `_data/` aggregates YAML/JSON config for menus, people, and publications; update these when adding new list-driven sections.
- Custom Liquid helpers reside in `_plugins/`; auxiliary scripts live in `_tools/`.
- `_site/` is regenerated output—clean or ignore it when committing.

## Build, Test, and Development Commands

- `bundle exec jekyll serve --livereload` spins up the site at `http://localhost:4000` for iterative work.
- `docker compose up jekyll` mirrors the GitHub Pages environment, exposing the build on `http://localhost:8080`.
- `bin/cibuild` (bundled) runs a production `jekyll build` to catch front-matter or Liquid errors.
- `npm run format` applies Prettier with the Liquid plugin across Markdown, HTML, SCSS, and JSON.

## Coding Style & Naming Conventions

- Favor two-space indentation for Liquid, Markdown, YAML, and SCSS to match existing files.
- Keep front matter minimal: include `layout`, `title`, `permalink`, plus collection-specific keys; order them consistently.
- Name assets descriptively with hyphenated lowercase (`assets/img/research-lidar-array.jpg`), and group custom components inside `_includes/sections/`.
- Run Prettier before committing; Husky/`lint-staged` will reformat staged files automatically.

## Testing Guidelines

- Run `bin/cibuild` (or `bundle exec jekyll build --trace`) before opening a PR to ensure the site renders without Liquid or collection errors.
- For content-heavy updates, review `_site/` or the served site to validate navigation, image paths, and data-driven lists.
- `bundle exec jekyll doctor` flags deprecated config; `npm run format -- --check` keeps formatting CI-ready.

## Commit & Pull Request Guidelines

- Follow the existing Conventional Commit style (`feat: ...`, `docs: ...`, `style: ...`) with concise, imperative summaries.
- Group related changes per commit; avoid mixing content edits, styling tweaks, and tooling updates in a single diff.
- PRs should describe motivation, key changes, and any new configuration flags; link related issues and add screenshots or screen recordings for visual tweaks.
- Mention deployment expectations (e.g., requires `bin/deploy`) when relevant so maintainers can plan releases.

## Deployment Tips

- `bin/deploy` promotes the latest build to `gh-pages`, purges unused CSS via PurgeCSS, and force-pushes the branch—run only on a clean working tree.
- GitHub Actions (`.github/workflows/deploy.yml`) will rebuild on pushes to `main`; keep `JEKYLL_ENV` set appropriately in custom workflows.
