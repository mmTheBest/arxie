# Paperbase Figure And Table Artifacts Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the current figure/table gap by making tables first-class Paperbase artifacts and exposing figures and tables through paper browse, collection summaries, comparison APIs, and the Paperbase Console UI.

**Architecture:** Reuse the existing figure-artifact pattern as the contract for tables. Add a canonical `tables` table plus a placeholder `src/paperbase/tables/` pipeline, then extend the API and UI to treat figures and tables as comparable, browseable artifacts alongside result rows. Keep extraction placeholder-based for now so the storage and comparison contracts land before parser hardening.

**Tech Stack:** SQLAlchemy ORM, FastAPI, Pydantic, vanilla JavaScript/CSS, pytest

### Task 1: Add failing tests for canonical table storage

**Files:**
- Modify: `tests/paperbase/test_schema_contract.py`
- Create: `tests/paperbase/test_table_pipeline.py`

**Step 1: Write the failing tests**

- Assert `tables` exists in the canonical metadata.
- Assert a placeholder table pipeline can persist one table artifact for a parsed paper.

**Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/paperbase/test_schema_contract.py tests/paperbase/test_table_pipeline.py -q
```

Expected: FAIL because the `tables` table and table pipeline do not exist yet.

### Task 2: Add failing tests for paper and collection artifact browse

**Files:**
- Modify: `tests/paperbase/test_paperbase_api.py`
- Modify: `tests/paperbase/test_collection_summary_api.py`

**Step 1: Write the failing tests**

- Assert `GET /api/v1/papers/{paper_id}/tables` returns persisted table artifacts.
- Assert `GET /api/v1/papers/{paper_id}/structured-data` now includes `figures` and `tables`.
- Assert `GET /api/v1/collections/{collection_id}/structured-summary` includes figure/table counts or sample artifacts.

**Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/paperbase/test_paperbase_api.py tests/paperbase/test_collection_summary_api.py -q
```

### Task 3: Add failing tests for figure/table comparison APIs

**Files:**
- Modify: `tests/paperbase/test_compare_api.py`

**Step 1: Write the failing tests**

- Assert `POST /api/v1/compare/figures` can return figures filtered by collection and optional method.
- Assert `POST /api/v1/compare/tables` can return table artifacts filtered by collection and optional method.

**Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/paperbase/test_compare_api.py -q
```

### Task 4: Add failing tests for the console artifact panels

**Files:**
- Modify: `tests/paperbase/test_ui_shell.py`

**Step 1: Write the failing tests**

- Assert the UI shell exposes figure and table panel anchors.
- Assert the shipped JS references the new figure/table endpoints and renders those surfaces.

**Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/paperbase/test_ui_shell.py -q
```

### Task 5: Implement the minimal artifact layer

**Files:**
- Modify: `src/paperbase/db/models.py`
- Create: `src/paperbase/tables/__init__.py`
- Create: `src/paperbase/tables/models.py`
- Create: `src/paperbase/tables/pipeline.py`
- Modify: `services/paperbase_api/models.py`
- Modify: `services/paperbase_api/routes/papers.py`
- Modify: `services/paperbase_api/routes/collections.py`
- Modify: `services/paperbase_api/routes/compare.py`
- Modify: `services/paperbase_api/static/index.html`
- Modify: `services/paperbase_api/static/paperbase-ui.js`
- Modify: `services/paperbase_api/static/paperbase-ui.css`

**Step 1: Write minimal implementation**

- Add `TableArtifact` canonical model with label, caption/title, page number, storage URI, and structured payload JSON.
- Add placeholder table extraction contracts mirroring the figure pipeline shape.
- Add paper-level figures/tables browse and structured-data exposure.
- Add collection structured summary fields for figures and tables.
- Add `compare/figures` and `compare/tables`.
- Extend the console with figure/table artifact panels for the selected paper and collection.

**Step 2: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/paperbase/test_schema_contract.py tests/paperbase/test_table_pipeline.py tests/paperbase/test_paperbase_api.py tests/paperbase/test_collection_summary_api.py tests/paperbase/test_compare_api.py tests/paperbase/test_ui_shell.py -q
```

Expected: PASS.

### Task 6: Verify the milestone and document it

**Files:**
- Modify: `docs/architecture/03-ingest-and-extraction.md`
- Modify: `docs/architecture/04-search-compare.md`
- Modify: `docs/architecture/README.md`
- Modify: `README.md`

**Step 1: Update docs**

- Record that figures and tables are now first-class artifacts in the local-first product.
- Note that extraction is still placeholder-based and ready for later parser hardening.

**Step 2: Run verification**

Run:

```bash
.venv/bin/python -m pytest tests/paperbase -q
make test-clean-baseline
```

Expected: passing Paperbase suite and clean repo baseline.

**Step 3: Commit**

```bash
git add docs/plans/2026-04-15-paperbase-figure-table-artifacts-implementation-plan.md docs/architecture/03-ingest-and-extraction.md docs/architecture/04-search-compare.md docs/architecture/README.md README.md src/paperbase/db/models.py src/paperbase/tables/__init__.py src/paperbase/tables/models.py src/paperbase/tables/pipeline.py services/paperbase_api/models.py services/paperbase_api/routes/papers.py services/paperbase_api/routes/collections.py services/paperbase_api/routes/compare.py services/paperbase_api/static/index.html services/paperbase_api/static/paperbase-ui.css services/paperbase_api/static/paperbase-ui.js tests/paperbase/test_schema_contract.py tests/paperbase/test_table_pipeline.py tests/paperbase/test_paperbase_api.py tests/paperbase/test_collection_summary_api.py tests/paperbase/test_compare_api.py tests/paperbase/test_ui_shell.py
git commit -m 'update on "2026-04-15"'
```
