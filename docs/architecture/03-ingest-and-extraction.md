# Ingest And Extraction

## Pipeline

The ingest pipeline should turn a paper identifier or PDF into persistent structured records.

Stages:

1. seed metadata from providers
2. fetch/store PDF
3. parse sections and chunks
4. extract schema-constrained entities
5. attach provenance spans
6. index searchable read models

For local-first operation, the first seed path can be a user-owned PDF directory. That importer should register each filesystem paper in the canonical schema, attach file records, and group them into a curated collection before any deeper parsing or extraction runs happen.

## Parser Strategy

Use the current `src/ra/parsing/pdf_parser.py` heuristics as the phase-1 parser adapter.

Later parser upgrades such as GROBID or PDFFigures2 should plug into the same pipeline contract rather than forcing a redesign.

## Current Module Map

The current ingest and parse modules are:

- `src/paperbase/ingest/models.py` — canonical seed object for provider-normalized paper imports
- `src/paperbase/ingest/arxiv_seed.py` — adapters from current arXiv and Semantic Scholar client models
- `src/paperbase/ingest/openalex.py` and `src/paperbase/ingest/crossref.py` — raw-provider payload normalization
- `src/paperbase/ingest/local_library.py` — local PDF directory import into canonical paper/file/collection records
- `src/paperbase/parsing/pipeline.py` — stored-PDF parse orchestration
- `src/paperbase/parsing/store.py` — persistence of sections and chunks
- `src/paperbase/parsing/chunker.py` — deterministic section chunking
- `src/paperbase/extract/contracts.py` — typed bundle contract for structured LLM extraction outputs
- `src/paperbase/extract/prompts.py` — prompt builder for field-specific extraction profiles
- `src/paperbase/extract/client.py` — OpenAI-backed structured extraction client
- `src/paperbase/extract/pipeline.py` — persistence of datasets, methods, metrics, result rows, findings, glossary terms, engineering tricks, and evidence spans
- `src/paperbase/extract/runner.py` — collection-level orchestration for local-first extraction runs

These modules are intentionally local-first. They should keep the import and parse contracts stable while the worker service and richer extraction stack are still being built.

## Field-Specific Extraction

Extraction should support collection-specific profiles so different research fields can ask for different structured outputs.

The first extraction pipeline should remain schema-constrained and replaceable:

- parsed sections are the default extraction input for local-first corpora
- extraction clients return a typed bundle rather than directly mutating the database
- persistence is handled inside Paperbase so Arxie and future workers reuse the same storage contract
- collection runners should execute profile-specific extraction over user-curated corpora without requiring the future worker stack
- glossary terms are stored as first-class canonical entities because field-specific databases often need shared vocabulary and benchmark definitions alongside result rows
