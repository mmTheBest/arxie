# Paperbase Search Ops Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expose the existing Paperbase search-runtime capabilities through the API so the local-first platform can trigger and observe reindexing without direct library calls.

**Architecture:** Keep the current `PaperbaseSearchReindexer` and injected `search_backend` as the execution engine. Add thin Paperbase API routes for search backend status and reindex actions instead of inventing a new service layer. The API should stay usable even when no backend is configured by returning explicit, structured status and errors.

**Tech Stack:** FastAPI, SQLAlchemy, Paperbase search runtime, pytest

### Task 1: Define the search-ops API contract

**Files:**
- Modify: `services/paperbase_api/models.py`
- Test: `tests/paperbase/test_search_ops_api.py`

**Step 1: Write the failing test**

Cover:

- `GET /api/v1/search/status` reports whether a search backend is configured
- `POST /api/v1/search/reindex` returns document counts after a successful reindex
- `POST /api/v1/search/reindex` returns a structured `503` when no search backend is configured

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/paperbase/test_search_ops_api.py -q`
Expected: FAIL.

**Step 3: Write minimal implementation**

Add response models for search status and reindex summaries.

**Step 4: Run test to verify it still fails for missing routes**

Run: `.venv/bin/python -m pytest tests/paperbase/test_search_ops_api.py -q`
Expected: FAIL with missing endpoint behavior.

### Task 2: Expose search status and reindex routes

**Files:**
- Modify: `services/paperbase_api/routes/search.py`
- Modify: `services/paperbase_api/app.py`
- Test: `tests/paperbase/test_search_ops_api.py`

**Step 1: Write the failing route expectations**

Use the same red test from Task 1 to confirm:

- backend status route exists
- reindex route exists
- reindex route uses `PaperbaseSearchReindexer`

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/paperbase/test_search_ops_api.py -q`
Expected: FAIL.

**Step 3: Write minimal implementation**

- add `GET /api/v1/search/status`
- add `POST /api/v1/search/reindex`
- return `503` with a structured error when `search_backend` is missing
- use the configured session factory plus `PaperbaseSearchReindexer` for the reindex action

**Step 4: Run focused verification**

Run: `.venv/bin/python -m pytest tests/paperbase/test_search_ops_api.py -q`
Expected: PASS.

### Task 3: Verify the broader search/runtime surface and document it

**Files:**
- Modify: `docs/architecture/04-search-compare.md`
- Create: `docs/plans/2026-04-15-paperbase-search-ops-implementation-plan.md`
- Test: `tests/paperbase/test_search_runtime_backend.py`

**Step 1: Run broader search/runtime verification**

Run: `.venv/bin/python -m pytest tests/paperbase/test_search_runtime_backend.py tests/paperbase/test_search_ops_api.py -q`
Expected: PASS.

**Step 2: Update docs**

Document that the Paperbase API now exposes the operational path for search indexing, not just the runtime classes.

**Step 3: Run broader Paperbase verification**

Run: `.venv/bin/python -m pytest tests/paperbase -q`
Expected: PASS.

Run: `make test-clean-baseline`
Expected: PASS.

**Step 4: Commit**

```bash
git add services/paperbase_api docs/architecture/04-search-compare.md tests/paperbase/test_search_ops_api.py docs/plans/2026-04-15-paperbase-search-ops-implementation-plan.md
git commit -m 'update on "2026-04-15"'
```
