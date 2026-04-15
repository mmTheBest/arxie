# Arxie Platform Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Evolve the existing Arxie repo into a layered product where the current assistant remains the user-facing surface and a new Paperbase platform provides persistent corpus storage, extraction, indexing, and comparison.

**Architecture:** Keep `src/ra` focused on assistant workflows. Add sibling Paperbase packages and services at the repo root for canonical data, workers, search indexing, and corpus APIs. Integrate Arxie with Paperbase through explicit gateway interfaces rather than folding the platform into the existing retrieval-agent package tree.

**Tech Stack:** Python, FastAPI, PostgreSQL, Alembic, Elasticsearch, Redis, object storage, OpenAI API, PyMuPDF/pdfplumber initially, GROBID and PDFFigures2 when parser hardening begins, pytest, Docker Compose

## Target Repo

All implementation planning and future code work should move to:

`/Users/mm/Projects/academic-research-assistant`

Reason:

- it already has the active git remote
- it already contains the user-facing product
- the merged product decision makes it the source-of-truth repo

## Desired Repo Layout

```text
src/
  ra/                       # existing Arxie assistant app
  paperbase/                # new corpus domain package
services/
  paperbase_api/
  paperbase_worker/
infra/
docs/
  plans/
tests/
  unit/
  integration/
  paperbase/
skills/
  paperbase-postgres/
  paperbase-elasticsearch/
  paperbase-corpus-ops/
```

## Execution Principles

- do not put Paperbase under `src/ra/paperbase`
- preserve current Arxie behavior while adding platform capabilities
- use current Arxie retrieval and parsing code as migration scaffolding where practical
- prefer adapters and gateways over broad rewrites

### Task 1: Land the merged product docs in Arxie

**Files:**
- Create: `docs/plans/2026-04-14-arxie-homepage.md`
- Create: `docs/plans/2026-04-14-arxie-platform-prd.md`
- Create: `docs/plans/2026-04-14-arxie-platform-implementation-plan.md`

**Step 1: Write the failing smoke check**

Create `tests/paperbase/test_planning_docs_present.py` and assert the three planning files exist.

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/paperbase/test_planning_docs_present.py -q`
Expected: FAIL because the files do not exist yet.

**Step 3: Write minimal implementation**

Add the three planning docs and confirm they reflect the merged product direction.

**Step 4: Run verification**

Run: `.venv/bin/python -m pytest tests/paperbase/test_planning_docs_present.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add docs/plans tests/paperbase/test_planning_docs_present.py
git commit -m 'docs: add merged arxie platform planning docs'
```

### Task 2: Prepare the repo for layered platform work

**Files:**
- Modify: `pyproject.toml`
- Create: `src/paperbase/__init__.py`
- Create: `services/paperbase_api/__init__.py`
- Create: `services/paperbase_worker/__init__.py`
- Create: `infra/docker-compose.paperbase.yml`
- Test: `tests/paperbase/test_repo_layout.py`

**Step 1: Write the failing test**

Assert the new root-level package and service directories exist.

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/paperbase/test_repo_layout.py -q`
Expected: FAIL.

**Step 3: Write minimal implementation**

Add the directories and any necessary packaging/config entries without changing existing Arxie entry points yet.

**Step 4: Run verification**

Run: `.venv/bin/python -m pytest tests/paperbase/test_repo_layout.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add pyproject.toml src/paperbase services infra tests/paperbase
git commit -m 'chore: prepare layered platform repo structure'
```

### Task 3: Bootstrap platform infrastructure

**Files:**
- Create: `infra/docker-compose.paperbase.yml`
- Create: `infra/env/postgres.env`
- Create: `infra/env/elasticsearch.env`
- Create: `infra/env/minio.env`
- Create: `infra/env/redis.env`
- Test: `tests/paperbase/test_compose_config.py`

**Step 1: Write the failing test**

Assert compose config includes `postgres`, `elasticsearch`, `minio`, and `redis`.

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/paperbase/test_compose_config.py -q`
Expected: FAIL.

**Step 3: Write minimal implementation**

Add the local platform stack. Keep it separate from any existing lightweight Arxie Docker configuration.

**Step 4: Run verification**

Run: `docker compose -f infra/docker-compose.paperbase.yml config`
Expected: valid compose output.

**Step 5: Commit**

```bash
git add infra tests/paperbase
git commit -m 'chore: add paperbase infrastructure stack'
```

### Task 4: Define the canonical Paperbase schema

**Files:**
- Create: `src/paperbase/config.py`
- Create: `src/paperbase/db/models.py`
- Create: `src/paperbase/db/session.py`
- Create: `src/paperbase/db/migrations/`
- Create: `src/paperbase/schemas/extraction.py`
- Test: `tests/paperbase/test_schema_contract.py`

**Step 1: Write the failing test**

Assert the core tables and schema models exist for `papers`, `chunks`, `figures`, `methods`, `datasets`, `metrics`, `results`, `evidence_spans`, and `extraction_runs`.

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/paperbase/test_schema_contract.py -q`
Expected: FAIL.

**Step 3: Write minimal implementation**

Add the canonical relational schema plus JSONB-backed payload/version fields. Keep the schema independent from `src/ra/proposal/*`.

**Step 4: Run verification**

Run: `.venv/bin/python -m pytest tests/paperbase/test_schema_contract.py -q`
Expected: PASS.

Run: `alembic upgrade head`
Expected: migration succeeds.

**Step 5: Commit**

```bash
git add src/paperbase tests/paperbase
git commit -m 'feat: add paperbase canonical schema'
```

### Task 5: Reuse current source adapters for platform ingest

**Files:**
- Create: `src/paperbase/ingest/openalex.py`
- Create: `src/paperbase/ingest/crossref.py`
- Create: `src/paperbase/ingest/arxiv_seed.py`
- Modify: `src/ra/retrieval/arxiv.py`
- Modify: `src/ra/retrieval/semantic_scholar.py`
- Test: `tests/paperbase/test_ingest_adapters.py`

**Step 1: Write the failing test**

Define tests that map external source payloads into a canonical paper seed object.

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/paperbase/test_ingest_adapters.py -q`
Expected: FAIL.

**Step 3: Write minimal implementation**

Reuse logic from current Arxie retrieval clients where possible, but move canonical ingest transforms into `src/paperbase`.

**Step 4: Run verification**

Run: `.venv/bin/python -m pytest tests/paperbase/test_ingest_adapters.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/paperbase src/ra/retrieval tests/paperbase
git commit -m 'feat: add paperbase ingest adapters'
```

### Task 6: Add PDF persistence and parser pipeline

**Files:**
- Create: `src/paperbase/parsing/store.py`
- Create: `src/paperbase/parsing/pipeline.py`
- Create: `src/paperbase/parsing/chunker.py`
- Modify: `src/ra/parsing/pdf_parser.py`
- Test: `tests/paperbase/test_parse_pipeline.py`

**Step 1: Write the failing test**

Assert a paper PDF can be stored, parsed, chunked, and persisted as canonical sections/chunks.

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/paperbase/test_parse_pipeline.py -q`
Expected: FAIL.

**Step 3: Write minimal implementation**

Use current Arxie parser heuristics as the initial parser adapter. Design the interface so GROBID can replace or augment it later without breaking downstream storage contracts.

**Step 4: Run verification**

Run: `.venv/bin/python -m pytest tests/paperbase/test_parse_pipeline.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/paperbase src/ra/parsing tests/paperbase
git commit -m 'feat: add paperbase parse pipeline'
```

### Task 7: Add figure extraction and artifact storage

**Files:**
- Create: `src/paperbase/figures/pipeline.py`
- Create: `src/paperbase/figures/models.py`
- Test: `tests/paperbase/test_figure_pipeline.py`

**Step 1: Write the failing test**

Assert figure metadata can be created for a parsed paper, even if the first implementation is a placeholder adapter.

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/paperbase/test_figure_pipeline.py -q`
Expected: FAIL.

**Step 3: Write minimal implementation**

Add the storage contract and placeholder extraction interface first. Wire in PDFFigures2 after the contract is stable.

**Step 4: Run verification**

Run: `.venv/bin/python -m pytest tests/paperbase/test_figure_pipeline.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/paperbase tests/paperbase
git commit -m 'feat: add figure extraction contracts'
```

### Task 8: Build schema-constrained extraction

**Files:**
- Create: `src/paperbase/extract/contracts.py`
- Create: `src/paperbase/extract/prompts.py`
- Create: `src/paperbase/extract/client.py`
- Create: `src/paperbase/extract/pipeline.py`
- Test: `tests/paperbase/test_extraction_contracts.py`

**Step 1: Write the failing test**

Assert methods, datasets, metrics, results, findings, glossary terms, and engineering tricks validate against explicit schemas and attach evidence spans.

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/paperbase/test_extraction_contracts.py -q`
Expected: FAIL.

**Step 3: Write minimal implementation**

Persist extraction runs with model, prompt version, schema version, and validation outcomes.

**Step 4: Run verification**

Run: `.venv/bin/python -m pytest tests/paperbase/test_extraction_contracts.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/paperbase tests/paperbase
git commit -m 'feat: add schema-constrained paper extraction'
```

### Task 9: Add Elasticsearch-backed search models

**Files:**
- Create: `src/paperbase/search/index_templates.py`
- Create: `src/paperbase/search/indexer.py`
- Create: `src/paperbase/search/query_builder.py`
- Test: `tests/paperbase/test_search_indexing.py`

**Step 1: Write the failing test**

Define tests for paper, chunk, and figure documents, including filters and vector fields.

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/paperbase/test_search_indexing.py -q`
Expected: FAIL.

**Step 3: Write minimal implementation**

Add `papers_v1`, `chunks_v1`, and `figures_v1` mappings and builders.

**Step 4: Run verification**

Run: `.venv/bin/python -m pytest tests/paperbase/test_search_indexing.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/paperbase tests/paperbase
git commit -m 'feat: add elasticsearch read models'
```

### Task 10: Expose the Paperbase API service

**Files:**
- Create: `services/paperbase_api/app.py`
- Create: `services/paperbase_api/routes/papers.py`
- Create: `services/paperbase_api/routes/search.py`
- Create: `services/paperbase_api/routes/compare.py`
- Test: `tests/paperbase/test_paperbase_api.py`

**Step 1: Write the failing test**

Cover:

- paper search
- paper fetch
- fulltext fetch
- figures fetch
- result comparison

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/paperbase/test_paperbase_api.py -q`
Expected: FAIL.

**Step 3: Write minimal implementation**

Expose platform endpoints independent of the existing `src/ra/api/app.py` API.

**Step 4: Run verification**

Run: `.venv/bin/python -m pytest tests/paperbase/test_paperbase_api.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add services/paperbase_api tests/paperbase
git commit -m 'feat: add paperbase api service'
```

### Task 11: Integrate Arxie with Paperbase via gateway

**Files:**
- Create: `src/ra/retrieval/paperbase_gateway.py`
- Modify: `src/ra/retrieval/unified.py`
- Modify: `src/ra/tools/retrieval_tools.py`
- Modify: `src/ra/agents/research_agent.py`
- Modify: `src/ra/agents/lit_review_agent.py`
- Test: `tests/unit/test_unified_retriever_paperbase.py`

**Step 1: Write the failing test**

Assert Arxie can query Paperbase first and fall back to external providers second.

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_unified_retriever_paperbase.py -q`
Expected: FAIL.

**Step 3: Write minimal implementation**

Introduce `PaperbaseRetriever` or `PaperCorpusGateway` and keep the existing external clients for fallback.

**Step 4: Run verification**

Run: `.venv/bin/python -m pytest tests/unit/test_unified_retriever_paperbase.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/ra/retrieval src/ra/tools src/ra/agents tests/unit
git commit -m 'feat: integrate arxie retrieval with paperbase'
```

### Task 12: Connect Paperbase to proposal and lit-review workflows

**Files:**
- Modify: `src/ra/proposal/evidence_mapper.py`
- Modify: `src/ra/proposal/assembler.py`
- Modify: `src/ra/api/app.py`
- Test: `tests/unit/test_api_proposal_paperbase.py`
- Test: `tests/unit/test_lit_review_agent.py`

**Step 1: Write the failing tests**

Assert proposal evidence queries and lit-review generation can draw from Paperbase-backed evidence.

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_api_proposal_paperbase.py tests/unit/test_lit_review_agent.py -q`
Expected: FAIL.

**Step 3: Write minimal implementation**

Use Paperbase-derived evidence for comparison tables, evidence buckets, and literature review synthesis where available.

**Step 4: Run verification**

Run: `.venv/bin/python -m pytest tests/unit/test_api_proposal_paperbase.py tests/unit/test_lit_review_agent.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/ra/proposal src/ra/api tests/unit
git commit -m 'feat: connect proposal and lit-review workflows to paperbase'
```

### Task 13: Add Codex skills and operator docs

**Files:**
- Create: `skills/paperbase-postgres/SKILL.md`
- Create: `skills/paperbase-elasticsearch/SKILL.md`
- Create: `skills/paperbase-corpus-ops/SKILL.md`
- Create: `docs/runbooks/paperbase-ingest.md`
- Create: `docs/runbooks/paperbase-reindex.md`
- Test: `tests/paperbase/test_skill_assets.py`

**Step 1: Write the failing test**

Assert the new skills and runbooks exist and validate.

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/paperbase/test_skill_assets.py -q`
Expected: FAIL.

**Step 3: Write minimal implementation**

Document safe query patterns, operational rules, and corpus workflows for Codex.

**Step 4: Run verification**

Run: `.venv/bin/python -m pytest tests/paperbase/test_skill_assets.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add skills docs/runbooks tests/paperbase
git commit -m 'feat: add paperbase skills and runbooks'
```

## Recommended Phase Order

1. Tasks 1-4
2. Tasks 5-8
3. Tasks 9-11
4. Tasks 12-13

## MVP Exit Criteria

- Arxie can search Paperbase-backed papers through a stable gateway
- Paperbase can ingest and store at least one public corpus slice
- extracted methods, datasets, metrics, and results are queryable
- at least one comparison workflow is evidence-backed and repeatable
- proposal or lit-review workflows use Paperbase evidence successfully

## Notes

- This plan supersedes the earlier greenfield PaperSource-only plan.
- The implementation home is the Arxie repo, but the code layout must remain layered.
- No separate memory files were found in PaperSource beyond the planning markdown artifacts.
