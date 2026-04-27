# Paperbase Local Persistence Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Turn the Paperbase schema scaffold into a usable local-first persistence layer for canonical papers, curated collections, extraction profiles, and user annotations.

**Architecture:** Keep the canonical SQLAlchemy models as the storage schema, add a thin database bootstrap layer that initializes a local SQLite database for development, and expose repository classes for the first product-critical write paths. This keeps v1 local-first while preserving ownership and schema boundaries needed for future multi-user and larger-corpus expansion.

**Tech Stack:** Python, SQLAlchemy 2.x, SQLite for local-first development, pytest.

### Task 1: Track the persistence plan in the clean worktree

**Files:**
- Modify: `.gitignore`
- Create: `docs/plans/2026-04-14-paperbase-local-persistence-implementation-plan.md`

**Step 1: Make `docs/plans` trackable**

Update `.gitignore` so `docs/plans/*.md` can be committed in the clean worktree while other dev-only docs remain excluded by default.

**Step 2: Record the persistence slice**

Document the scope: database bootstrap, repository APIs, and focused tests for papers, collections, extraction profiles, and annotations.

### Task 2: Write the failing bootstrap test

**Files:**
- Create: `tests/paperbase/test_bootstrap.py`
- Create: `src/paperbase/db/bootstrap.py`

**Step 1: Write the failing test**

Create a test that initializes a SQLite database inside a nested temporary directory and asserts that:
- parent directories are created
- schema tables are created
- the database file exists on disk

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/paperbase/test_bootstrap.py -q`

Expected: FAIL because `paperbase.db.bootstrap` or `initialize_database` does not exist yet.

### Task 3: Write the failing repository tests

**Files:**
- Create: `tests/paperbase/test_repositories.py`
- Create: `src/paperbase/db/repositories.py`

**Step 1: Write the failing tests**

Cover these behaviors:
- `PaperRepository.upsert` deduplicates on `(provider, external_id)` and updates mutable fields
- `ExtractionProfileRepository.create` persists a collection-specific extraction schema
- `CollectionRepository.create` and `add_paper` persist curated membership without duplicates
- `AnnotationRepository.create` and `list_for_target` persist user notes separately from canonical facts

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/paperbase/test_repositories.py -q`

Expected: FAIL because repository classes and/or constraints are missing.

### Task 4: Implement minimal bootstrap and repositories

**Files:**
- Modify: `src/paperbase/db/models.py`
- Modify: `src/paperbase/db/session.py`
- Create: `src/paperbase/db/bootstrap.py`
- Create: `src/paperbase/db/repositories.py`
- Modify: `src/paperbase/db/__init__.py`

**Step 1: Add the minimal schema constraints**

Add uniqueness guarantees required for local correctness:
- `papers(provider, external_id)`
- `paper_sources(provider, provider_record_id)`
- `collection_papers(collection_id, paper_id)`

**Step 2: Add bootstrap support**

Implement `initialize_database(database_url)` that:
- creates parent directories for SQLite database paths
- creates an engine
- runs `Base.metadata.create_all`
- returns the engine

**Step 3: Add repository APIs**

Implement repository classes with simple methods:
- `PaperRepository.upsert(...)`
- `PaperRepository.get_by_provider_id(...)`
- `ExtractionProfileRepository.create(...)`
- `CollectionRepository.create(...)`
- `CollectionRepository.add_paper(...)`
- `CollectionRepository.list_papers(...)`
- `AnnotationRepository.create(...)`
- `AnnotationRepository.list_for_target(...)`

### Task 5: Verify and stabilize

**Files:**
- Test: `tests/paperbase/test_bootstrap.py`
- Test: `tests/paperbase/test_repositories.py`
- Test: `tests/paperbase/test_repo_layout.py`
- Test: `tests/paperbase/test_schema_contract.py`

**Step 1: Run focused Paperbase tests**

Run:
- `.venv/bin/python -m pytest tests/paperbase/test_bootstrap.py -q`
- `.venv/bin/python -m pytest tests/paperbase/test_repositories.py -q`
- `.venv/bin/python -m pytest tests/paperbase -q`

**Step 2: Run the clean baseline**

Run:
- `make test-clean-baseline`

Expected: all Paperbase tests pass and the clean baseline stays green.

**Step 3: Commit**

```bash
git add .gitignore docs/plans/2026-04-14-paperbase-local-persistence-implementation-plan.md src/paperbase/db tests/paperbase Makefile
git commit -m 'update on "2026-04-14"'
git push
```
