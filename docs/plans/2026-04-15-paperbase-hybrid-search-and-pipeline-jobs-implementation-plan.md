# Paperbase Hybrid Search And Pipeline Jobs Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the next Paperbase gap by making local-first ingest and parse worker-dispatchable, then exposing richer hybrid search over papers, chunks, and artifacts.

**Architecture:** Keep the current DB-backed queue contract and extend it with `local_library_ingest` and `collection_parse` jobs. Reuse the existing local PDF importer and parse pipeline behind collection-level runners, then widen the search runtime to index tables and expose chunk/artifact search routes with SQL fallback and backend-assisted semantic queries when a search backend is configured.

**Tech Stack:** Python, FastAPI, SQLAlchemy, pytest, vanilla JS, deterministic local embeddings

### Task 1: Add failing tests for queued local-library ingest and collection parse

**Files:**
- Create: `tests/paperbase/test_pipeline_jobs_api.py`
- Modify: `tests/paperbase/test_worker_runtime.py`

**Step 1: Write the failing tests**

- Assert `POST /api/v1/ingest/local-library` enqueues a `local_library_ingest` job.
- Assert `POST /api/v1/collections/{collection_id}/parse` enqueues a `collection_parse` job.
- Assert the worker can execute both jobs and persist imported papers or parsed sections/chunks.

**Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/paperbase/test_pipeline_jobs_api.py tests/paperbase/test_worker_runtime.py -q
```

Expected: FAIL because the routes, job types, and parse runner do not exist yet.

### Task 2: Add failing tests for hybrid chunk and artifact search

**Files:**
- Create: `tests/paperbase/test_search_surfaces_api.py`
- Modify: `tests/paperbase/test_search_runtime_backend.py`
- Modify: `tests/paperbase/test_search_indexing.py`

**Step 1: Write the failing tests**

- Assert `/api/v1/search/chunks` returns chunk hits with paper metadata.
- Assert `/api/v1/search/artifacts` returns figure/table hits and supports `kind=figure|table|all`.
- Assert the backend-aware routes include `knn` query data when a backend is configured.
- Assert the reindexer now builds and pushes table documents in addition to papers/chunks/figures.

**Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/paperbase/test_search_surfaces_api.py tests/paperbase/test_search_runtime_backend.py tests/paperbase/test_search_indexing.py -q
```

Expected: FAIL because the routes and table search/index contracts do not exist yet.

### Task 3: Implement pipeline jobs and collection parse runner

**Files:**
- Create: `src/paperbase/parsing/runner.py`
- Create: `services/paperbase_api/routes/ingest.py`
- Modify: `services/paperbase_api/app.py`
- Modify: `services/paperbase_api/models.py`
- Modify: `services/paperbase_api/routes/collections.py`
- Modify: `services/paperbase_worker/runtime.py`

**Step 1: Write minimal implementation**

- Add a collection parse runner that iterates collection memberships through `PaperParsePipeline`.
- Add `local_library_ingest` and `collection_parse` job handlers to the worker.
- Add API endpoints to enqueue those jobs without running inline.

**Step 2: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/paperbase/test_pipeline_jobs_api.py tests/paperbase/test_worker_runtime.py -q
```

Expected: PASS.

### Task 4: Implement hybrid search surfaces and table indexing

**Files:**
- Modify: `src/paperbase/search/index_templates.py`
- Modify: `src/paperbase/search/indexer.py`
- Create: `src/paperbase/search/embeddings.py`
- Modify: `src/paperbase/search/runtime.py`
- Modify: `services/paperbase_api/models.py`
- Modify: `services/paperbase_api/routes/search.py`

**Step 1: Write minimal implementation**

- Add a table index template/document builder and include it in the reindexer summary.
- Add deterministic local embedding generation for backend-assisted semantic search.
- Add chunk and artifact search routes with SQL fallback and backend support when configured.

**Step 2: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/paperbase/test_search_surfaces_api.py tests/paperbase/test_search_runtime_backend.py tests/paperbase/test_search_indexing.py -q
```

Expected: PASS.

### Task 5: Update the Paperbase Console and docs

**Files:**
- Modify: `services/paperbase_api/static/index.html`
- Modify: `services/paperbase_api/static/paperbase-ui.js`
- Modify: `tests/paperbase/test_ui_shell.py`
- Modify: `docs/architecture/03-ingest-and-extraction.md`
- Modify: `docs/architecture/04-search-compare.md`
- Modify: `docs/architecture/README.md`
- Modify: `README.md`

**Step 1: Write the failing test**

- Assert the UI references the new ingest/parse/chunk/artifact search surfaces.

**Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/paperbase/test_ui_shell.py -q
```

Expected: FAIL.

**Step 3: Implement the minimal console updates**

- Add queue controls for parse jobs.
- Add chunk/artifact search surface wiring for the existing console.
- Update collaborator docs and README to reflect the broader pipeline/search state.

**Step 4: Run verification**

Run:

```bash
.venv/bin/python -m pytest tests/paperbase -q
make test-clean-baseline
```

Expected: passing Paperbase suite and clean repo baseline.
