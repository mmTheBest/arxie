# Paperbase Local Library Import Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Allow a local directory of PDFs to become a seeded Paperbase collection by importing filesystem papers into canonical paper, file, and collection records.

**Architecture:** Add a thin local-library import module that scans a PDF directory, creates or reuses a collection, upserts canonical `local_filesystem` paper records keyed by absolute path, records `paper_files` metadata, and attaches papers to the collection. Keep the importer deterministic and idempotent so re-running it on the same folder does not duplicate papers, files, or collection membership.

**Tech Stack:** Python, SQLAlchemy 2.x, SQLite, pytest.

### Task 1: Record the import slice

**Files:**
- Create: `docs/plans/2026-04-15-paperbase-local-library-import-implementation-plan.md`
- Modify: `docs/architecture/03-ingest-and-extraction.md`

**Step 1: Capture the local-library requirement**

Document that v1 corpus seeding can start from a user-owned PDF directory, not only remote providers.

**Step 2: Document the concrete module boundary**

Describe the importer as the bridge from a filesystem folder into canonical `papers`, `paper_files`, and `collection_papers`.

### Task 2: Write the failing importer tests

**Files:**
- Create: `tests/paperbase/test_local_library_import.py`
- Create: `src/paperbase/ingest/__init__.py`
- Create: `src/paperbase/ingest/local_library.py`

**Step 1: Write the failing tests**

Cover these behaviors:
- importing a directory with PDFs creates one collection, one paper record per PDF, and one `paper_files` record per PDF
- non-PDF files are ignored
- re-running the import is idempotent and reuses existing collection/paper/file membership

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/paperbase/test_local_library_import.py -q`

Expected: FAIL because `paperbase.ingest.local_library` does not exist yet.

### Task 3: Implement the importer and repository support

**Files:**
- Modify: `src/paperbase/db/models.py`
- Modify: `src/paperbase/db/repositories.py`
- Create: `src/paperbase/ingest/__init__.py`
- Create: `src/paperbase/ingest/local_library.py`

**Step 1: Add minimal file-level dedupe support**

Add the uniqueness guarantees and repository methods required to avoid duplicate `paper_files` rows and duplicate collections by title/owner.

**Step 2: Implement the importer**

Implement a local import entrypoint that:
- scans a directory recursively for PDFs
- derives a deterministic paper identity from the absolute path
- creates or reuses a collection
- upserts paper and file records
- attaches imported papers to the collection in deterministic order

### Task 4: Seed the real SamplePapers corpus and verify

**Files:**
- Runtime input: `/Users/mm/school/scRegNet/SamplePapers`

**Step 1: Run focused tests**

Run:
- `.venv/bin/python -m pytest tests/paperbase/test_local_library_import.py -q`
- `.venv/bin/python -m pytest tests/paperbase -q`

**Step 2: Seed the real collection**

Run the importer against `/Users/mm/school/scRegNet/SamplePapers` and confirm the created collection plus imported paper count.

**Step 3: Run the clean baseline**

Run:
- `make test-clean-baseline`

Expected: importer tests pass, the real corpus is seeded, and the clean baseline remains green.
