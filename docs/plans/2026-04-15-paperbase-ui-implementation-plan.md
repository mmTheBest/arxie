# Paperbase UI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add the first usable Paperbase web UI for browsing curated collections, inspecting papers and structured summaries, and operating queued extraction/reindex jobs.

**Architecture:** Serve a static local-first UI directly from the FastAPI Paperbase API service. Reuse the current JSON APIs, add only the missing recent-jobs listing endpoint for polling, and keep the frontend build-free for now so the product can ship a real interface without introducing a new JS toolchain.

**Tech Stack:** FastAPI, StaticFiles/FileResponse, vanilla JavaScript, CSS, pytest, existing Paperbase API routes

### Task 1: Add failing tests for the UI shell and job listing

**Files:**
- Create: `tests/paperbase/test_ui_shell.py`
- Modify: `tests/paperbase/test_background_jobs_api.py`

**Step 1: Write the failing tests**

- Assert `GET /app` returns the Paperbase UI shell as HTML.
- Assert the HTML references the shipped CSS and JS assets.
- Assert `GET /ui/paperbase-ui.js` and `GET /ui/paperbase-ui.css` return static assets.
- Assert `GET /api/v1/jobs` returns recent queued jobs for the UI jobs panel.

**Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/paperbase/test_ui_shell.py tests/paperbase/test_background_jobs_api.py -q
```

Expected: FAIL because the UI shell, static assets, and recent-jobs API route do not exist yet.

### Task 2: Add the recent-jobs API support

**Files:**
- Modify: `src/paperbase/db/repositories.py`
- Modify: `services/paperbase_api/routes/jobs.py`
- Modify: `services/paperbase_api/models.py`

**Step 1: Write the failing API test**

- Assert the jobs listing route returns jobs ordered newest-first with payload, result, and error fields.

**Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/paperbase/test_background_jobs_api.py -q
```

**Step 3: Write minimal implementation**

- Add `BackgroundJobRepository.list_recent(limit=...)`.
- Add `GET /api/v1/jobs?limit=...`.
- Reuse the existing job serializer so list and detail stay consistent.

**Step 4: Run tests to verify they pass**

Run:

```bash
.venv/bin/python -m pytest tests/paperbase/test_background_jobs_api.py -q
```

### Task 3: Serve the UI shell and static assets

**Files:**
- Modify: `services/paperbase_api/app.py`
- Create: `services/paperbase_api/routes/ui.py`
- Create: `services/paperbase_api/static/index.html`
- Create: `services/paperbase_api/static/paperbase-ui.css`
- Create: `services/paperbase_api/static/paperbase-ui.js`

**Step 1: Write the failing shell tests**

- Assert the UI shell contains:
  - collections panel
  - papers panel
  - structured summary panel
  - jobs panel
  - queue-action buttons for extraction and reindex

**Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/paperbase/test_ui_shell.py -q
```

**Step 3: Write minimal implementation**

- Mount the static assets under `/ui`.
- Add `GET /app` returning the UI shell.
- Build a static UI that:
  - loads collections
  - loads collection papers and structured summary for the selected collection
  - loads recent jobs and polls active jobs
  - queues extraction and reindex jobs
  - supports paper search and paper structured-data drilldown

**Step 4: Run tests to verify they pass**

Run:

```bash
.venv/bin/python -m pytest tests/paperbase/test_ui_shell.py tests/paperbase/test_background_jobs_api.py -q
```

### Task 4: Verify the UI milestone

**Files:**
- Modify: `docs/architecture/README.md`
- Modify: `README.md`

**Step 1: Update docs**

- Record that the first product UI is now served from the Paperbase API service.
- Describe the supported workflows and the local-first build-free frontend choice.

**Step 2: Run verification**

Run:

```bash
.venv/bin/python -m pytest tests/paperbase -q
make test-clean-baseline
```

Expected: passing Paperbase suite and passing clean repo baseline.

**Step 3: Commit**

```bash
git add docs/plans/2026-04-15-paperbase-ui-implementation-plan.md docs/architecture/README.md README.md services/paperbase_api/app.py services/paperbase_api/routes/ui.py services/paperbase_api/routes/jobs.py services/paperbase_api/models.py services/paperbase_api/static/index.html services/paperbase_api/static/paperbase-ui.css services/paperbase_api/static/paperbase-ui.js src/paperbase/db/repositories.py tests/paperbase/test_background_jobs_api.py tests/paperbase/test_ui_shell.py
git commit -m 'update on "2026-04-15"'
```
