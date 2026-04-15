# Arxie Workspace UI And Saved Workspaces Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the remaining April 14 product-surface gap by adding saved research workspaces and upgrading the static UI from an operator console into a user-facing Arxie homepage plus workspace app.

**Architecture:** Add a lightweight `workspaces` persistence layer to Paperbase so a workspace can store durable research context on top of an existing collection: title, description, saved query, pinned papers, filters, and a focus note. Keep the UI build-free and local-first by serving static HTML/CSS/JS from the Paperbase API, but split it into a public landing page at `/` and a research workspace app at `/app`.

**Tech Stack:** Python, FastAPI, SQLAlchemy, pytest, vanilla JS, static HTML/CSS

### Task 1: Add failing tests for saved workspaces

**Files:**
- Create: `tests/paperbase/test_workspaces_api.py`
- Modify: `tests/paperbase/test_schema_contract.py`

**Step 1: Write the failing tests**

- Assert `workspaces` becomes part of the canonical metadata contract.
- Assert `POST /api/v1/workspaces` creates a saved workspace tied to a collection.
- Assert `GET /api/v1/workspaces` lists saved workspaces.
- Assert `GET /api/v1/workspaces/{workspace_id}` returns the saved query, focus note, filters, and pinned paper summaries.
- Assert `PATCH /api/v1/workspaces/{workspace_id}` updates the research context.

**Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/paperbase/test_workspaces_api.py tests/paperbase/test_schema_contract.py -q
```

Expected: FAIL because the workspace schema and routes do not exist yet.

### Task 2: Implement workspace persistence and API routes

**Files:**
- Modify: `src/paperbase/db/models.py`
- Modify: `src/paperbase/db/repositories.py`
- Modify: `services/paperbase_api/models.py`
- Modify: `services/paperbase_api/app.py`
- Create: `services/paperbase_api/routes/workspaces.py`

**Step 1: Write minimal implementation**

- Add a `Workspace` model with `owner_id`, `title`, `description`, `collection_id`, `saved_query`, `focus_note`, `active_filters_json`, and `pinned_paper_ids_json`.
- Add a repository for create/list/get/update operations and a helper to resolve pinned paper summaries.
- Add request and response models for workspace CRUD.
- Register `workspaces` routes in the API app.

**Step 2: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/paperbase/test_workspaces_api.py tests/paperbase/test_schema_contract.py -q
```

Expected: PASS.

### Task 3: Add failing tests for the public homepage and workspace UI shell

**Files:**
- Modify: `tests/paperbase/test_ui_shell.py`

**Step 1: Write the failing tests**

- Assert `/` serves an Arxie homepage rather than the console shell.
- Assert `/app` references workspace-specific copy and controls rather than “Paperbase Console”.
- Assert the shell exposes workspace list/detail elements and a save/update workspace control.

**Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/paperbase/test_ui_shell.py -q
```

Expected: FAIL.

### Task 4: Implement the homepage and workspace UI

**Files:**
- Modify: `services/paperbase_api/routes/ui.py`
- Create: `services/paperbase_api/static/landing.html`
- Modify: `services/paperbase_api/static/index.html`
- Modify: `services/paperbase_api/static/paperbase-ui.css`
- Modify: `services/paperbase_api/static/paperbase-ui.js`
- Modify: `README.md`
- Modify: `docs/architecture/README.md`
- Modify: `docs/architecture/05-assistant-integration.md`

**Step 1: Write minimal implementation**

- Serve a public landing page at `/` using the April 14 homepage framing.
- Rename and restyle `/app` into an Arxie workspace app.
- Add workspace cards, workspace detail surface, and save/update workspace actions.
- Wire the app to list, load, create, and update workspaces using the new API.
- Keep the existing collection, search, artifact, and job surfaces, but subordinate them to the selected workspace context.

**Step 2: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/paperbase/test_ui_shell.py -q
```

Expected: PASS.

### Task 5: Run full verification and document the milestone

**Files:**
- Modify: `docs/architecture/06-collections-and-annotations.md`
- Modify: `docs/architecture/README.md`
- Modify: `README.md`

**Step 1: Run verification**

Run:

```bash
.venv/bin/python -m pytest tests/paperbase/test_workspaces_api.py tests/paperbase/test_ui_shell.py -q
.venv/bin/python -m pytest tests/paperbase -q
make test-clean-baseline
```

Expected: all pass.

**Step 2: Update docs**

- Record that Arxie now supports saved workspaces as persistent research contexts layered over collections.
- Update the README and architecture docs so collaborators see the difference between the public homepage, the workspace app, and the underlying Paperbase API.
