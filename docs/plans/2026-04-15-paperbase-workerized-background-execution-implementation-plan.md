# Paperbase Workerized Background Execution Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace synchronous API-triggered reindex and collection extraction with worker-backed jobs that can be polled by the API and later consumed by the UI.

**Architecture:** Keep the first worker model local-first and database-backed. Add a canonical `background_jobs` table to Paperbase, enqueue jobs from the API, and execute them through a dedicated worker runner in `services/paperbase_worker`. Preserve the existing extraction and reindex execution code as the worker's execution engine instead of reimplementing those workflows.

**Tech Stack:** FastAPI, SQLAlchemy ORM, Pydantic, pytest, existing Paperbase extraction/search runtimes

### Task 1: Add failing tests for the async job contract

**Files:**
- Modify: `tests/paperbase/test_search_ops_api.py`
- Modify: `tests/paperbase/test_extraction_api.py`
- Create: `tests/paperbase/test_background_jobs_api.py`
- Create: `tests/paperbase/test_worker_runtime.py`

**Step 1: Write the failing tests**

- Change search reindex API expectations from synchronous `200` indexing results to `202` accepted job responses.
- Change collection extraction API expectations from immediate extraction summaries to accepted background job responses.
- Add a job status API test for `GET /api/v1/jobs/{job_id}`.
- Add worker runtime tests that:
  - claim and run a pending `search_reindex` job
  - claim and run a pending `collection_extract` job
  - mark unsupported or misconfigured jobs as failed

**Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/paperbase/test_search_ops_api.py tests/paperbase/test_extraction_api.py tests/paperbase/test_background_jobs_api.py tests/paperbase/test_worker_runtime.py -q
```

Expected: failures showing missing background job models, missing job routes, and synchronous API behavior still present.

### Task 2: Add canonical background job persistence

**Files:**
- Modify: `src/paperbase/db/models.py`
- Modify: `src/paperbase/db/repositories.py`
- Modify: `tests/paperbase/test_repositories.py`

**Step 1: Write the failing repository tests**

- Assert a job can be created with `pending` status.
- Assert the next pending job can be claimed and moved to `running`.
- Assert `completed` and `failed` transitions persist `result_json` and `error_message`.

**Step 2: Run repository tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/paperbase/test_repositories.py -q
```

**Step 3: Write minimal implementation**

- Add `BackgroundJob` model with:
  - `id`, `job_type`, `status`
  - `payload_json`, `result_json`
  - `error_message`
  - `attempt_count`
  - `started_at`, `finished_at`
- Add `BackgroundJobRepository` with:
  - `create`
  - `get_by_id`
  - `claim_next`
  - `mark_completed`
  - `mark_failed`

**Step 4: Run tests to verify they pass**

Run:

```bash
.venv/bin/python -m pytest tests/paperbase/test_repositories.py tests/paperbase/test_worker_runtime.py -q
```

### Task 3: Convert API routes to queued execution

**Files:**
- Modify: `services/paperbase_api/models.py`
- Modify: `services/paperbase_api/app.py`
- Modify: `services/paperbase_api/routes/search.py`
- Modify: `services/paperbase_api/routes/extraction.py`
- Create: `services/paperbase_api/routes/jobs.py`

**Step 1: Write the failing API tests**

- Assert `POST /api/v1/search/reindex` returns `202` with a job summary and does not index immediately.
- Assert `POST /api/v1/collections/{collection_id}/extract` returns `202` with a queued job summary and does not persist extraction artifacts yet.
- Assert `GET /api/v1/jobs/{job_id}` returns job state, timestamps, result, and error fields.

**Step 2: Run API tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/paperbase/test_search_ops_api.py tests/paperbase/test_extraction_api.py tests/paperbase/test_background_jobs_api.py -q
```

**Step 3: Write minimal implementation**

- Add job response models and a shared serializer.
- Register the jobs router in the API app.
- Enqueue `search_reindex` and `collection_extract` jobs from the API routes.
- Return `202 Accepted` responses with stable job payloads.

**Step 4: Run tests to verify they pass**

Run:

```bash
.venv/bin/python -m pytest tests/paperbase/test_search_ops_api.py tests/paperbase/test_extraction_api.py tests/paperbase/test_background_jobs_api.py -q
```

### Task 4: Implement the worker runtime

**Files:**
- Modify: `services/paperbase_worker/__init__.py`
- Create: `services/paperbase_worker/runtime.py`
- Create: `services/paperbase_worker/main.py`
- Modify: `services/paperbase_api/app.py`
- Modify: `docs/architecture/03-ingest-and-extraction.md`
- Modify: `docs/architecture/04-search-compare.md`

**Step 1: Write the failing worker runtime tests**

- Assert the worker dispatches `search_reindex` to `PaperbaseSearchReindexer`.
- Assert the worker dispatches `collection_extract` to `CollectionExtractionRunner`.
- Assert a job is marked `failed` when the worker lacks the required dependency, such as no search backend.

**Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/paperbase/test_worker_runtime.py -q
```

**Step 3: Write minimal implementation**

- Build a `PaperbaseWorker` that:
  - claims one pending job
  - executes the matching workflow
  - records `completed` or `failed`
- Add a simple `main.py` entrypoint for manual local worker execution.
- Keep dependency injection explicit:
  - `session_factory`
  - `search_backend`
  - `extraction_client_factory`

**Step 4: Run tests to verify they pass**

Run:

```bash
.venv/bin/python -m pytest tests/paperbase/test_worker_runtime.py tests/paperbase/test_search_ops_api.py tests/paperbase/test_extraction_api.py tests/paperbase/test_background_jobs_api.py -q
```

### Task 5: Verify the milestone and document the new contract

**Files:**
- Modify: `docs/architecture/README.md`
- Modify: `docs/plans/2026-04-14-arxie-platform-implementation-plan.md`
- Modify: `README.md`

**Step 1: Update docs**

- Document that Paperbase API routes enqueue jobs and the worker executes parse/extract/index tasks.
- Note the local-first implementation choice and the future path to Redis or a broker-backed queue.

**Step 2: Run verification**

Run:

```bash
.venv/bin/python -m pytest tests/paperbase -q
make test-clean-baseline
```

Expected: passing Paperbase suite and passing clean repo baseline.

**Step 3: Commit**

```bash
git add docs/plans/2026-04-15-paperbase-workerized-background-execution-implementation-plan.md docs/architecture/README.md docs/architecture/03-ingest-and-extraction.md docs/architecture/04-search-compare.md docs/plans/2026-04-14-arxie-platform-implementation-plan.md README.md services/paperbase_api/app.py services/paperbase_api/models.py services/paperbase_api/routes/search.py services/paperbase_api/routes/extraction.py services/paperbase_api/routes/jobs.py services/paperbase_worker/__init__.py services/paperbase_worker/runtime.py services/paperbase_worker/main.py src/paperbase/db/models.py src/paperbase/db/repositories.py tests/paperbase/test_repositories.py tests/paperbase/test_search_ops_api.py tests/paperbase/test_extraction_api.py tests/paperbase/test_background_jobs_api.py tests/paperbase/test_worker_runtime.py
git commit -m 'update on "2026-04-15"'
```
