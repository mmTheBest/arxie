# Arxie Workspace UI Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current operator-style `/app` dashboard with a workspace-first UI that matches the intended Arxie user workflow: choose a collection, search a field, inspect evidence, compare papers, save a workspace, and monitor jobs separately.

**Architecture:** Keep the backend routes unchanged and redesign the frontend as a single-page shell with module tabs. Reuse the existing Paperbase APIs and current static asset serving, but reorganize the HTML, JS state machine, and CSS around `Library`, `Workspace`, `Compare`, `Jobs`, and `Settings` modules.

**Tech Stack:** FastAPI static assets, plain HTML/CSS/vanilla JS, pytest `TestClient`.

### Task 1: Lock the new shell structure with failing tests

**Files:**
- Modify: `tests/paperbase/test_ui_shell.py`
- Reference: `services/paperbase_api/static/index.html`

**Step 1: Write the failing test**

- Assert `/app` includes the new app shell and module ids:
  - `app-shell`
  - `app-nav`
  - `library-view`
  - `workspace-view`
  - `compare-view`
  - `jobs-view`
  - `settings-view`
- Assert old operator-first panel ids like `summary-panel` and `search-surfaces-panel` are no longer required by the shell contract.

**Step 2: Run test to verify it fails**

Run:

```bash
/Users/mm/Projects/academic-research-assistant/.venv/bin/python -m pytest tests/paperbase/test_ui_shell.py -q
```

Expected: FAIL because the current shell still uses the old panel layout.

**Step 3: Commit after implementation**

```bash
git add tests/paperbase/test_ui_shell.py services/paperbase_api/static/index.html
git commit -m 'update on "2026-04-30"'
```

### Task 2: Replace the HTML shell with module-first layout

**Files:**
- Modify: `services/paperbase_api/static/index.html`

**Step 1: Implement the new shell**

- Add top navigation buttons for:
  - `Library`
  - `Workspace`
  - `Compare`
  - `Jobs`
  - `Settings`
- Add a left rail for collections and workspaces.
- Add view containers for:
  - library module
  - workspace module
  - compare module
  - jobs module
  - settings module

**Step 2: Keep existing action affordances available**

- Preserve import, parse, extract, reindex, search, save workspace, and job rendering hooks with stable element ids for JS.

### Task 3: Rewrite the client-side UI flow around modules

**Files:**
- Modify: `services/paperbase_api/static/paperbase-ui.js`

**Step 1: Add module-aware client state**

- Add `activeView`.
- Default to `workspace` when collections exist, otherwise `library`.
- Keep current collection/workspace/paper/job state.

**Step 2: Split rendering into module functions**

- `renderAppChrome`
- `renderLibraryView`
- `renderWorkspaceView`
- `renderCompareView`
- `renderJobsView`
- `renderSettingsView`

**Step 3: Fix current UI behavior bugs while moving logic**

- Correct the crossed collection-title/description wiring between upload import and host-path import.
- Make workspace save/pin behavior explicit in the workspace module.
- Ensure search results, chunk hits, and artifact hits live inside the workspace module rather than a detached panel.

**Step 4: Use existing compare APIs in a real compare module**

- Add lightweight compare controls:
  - result comparison by dataset + metric
  - method comparison by optional dataset/metric
  - engineering tricks by optional method
  - figure/table summaries for current collection

### Task 4: Restyle the UI for clearer hierarchy

**Files:**
- Modify: `services/paperbase_api/static/paperbase-ui.css`

**Step 1: Introduce module layout styling**

- top nav
- left rail
- central content surface
- right-side detail panel for workspace/paper context

**Step 2: Make the user flow visually clear**

- Library = import/manage collections
- Workspace = default research screen
- Compare = structured cross-paper analysis
- Jobs = operational monitoring
- Settings = advanced/local runtime info

### Task 5: Verify the redesign end to end

**Files:**
- Verify: `tests/paperbase/test_ui_shell.py`
- Verify: `tests/paperbase/`
- Verify: `tests/ -q --ignore=tests/integration`

**Step 1: Run focused UI shell test**

```bash
/Users/mm/Projects/academic-research-assistant/.venv/bin/python -m pytest tests/paperbase/test_ui_shell.py -q
```

Expected: PASS

**Step 2: Run Paperbase test slice**

```bash
/Users/mm/Projects/academic-research-assistant/.venv/bin/python -m pytest tests/paperbase -q
```

Expected: PASS

**Step 3: Run broader repo verification**

```bash
/Users/mm/Projects/academic-research-assistant/.venv/bin/python -m pytest tests/ -q --ignore=tests/integration
```

Expected: PASS

**Step 4: Manual smoke**

- Open `http://localhost:8080/app`
- Verify:
  - nav modules render
  - Library import forms still submit
  - Workspace search/paper detail render
  - Compare module renders collection compare slices
  - Jobs module still updates

