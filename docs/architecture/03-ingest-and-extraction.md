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
- `src/paperbase/figures/pipeline.py` — placeholder figure candidate extraction and persistence contracts
- `src/paperbase/tables/pipeline.py` — placeholder table candidate extraction and persistence contracts
- `services/paperbase_api/routes/extraction.py` — API surface for extraction profiles and queued collection extraction jobs
- `services/paperbase_worker/runtime.py` — worker dispatcher for queued parse/extract/index jobs
- `src/paperbase/profiles/` — built-in field-specific extraction profile presets such as `sc_regnet`

These modules are intentionally local-first. They should keep the import and parse contracts stable while the worker service and richer extraction stack are still being built.

## Field-Specific Extraction

Extraction should support collection-specific profiles so different research fields can ask for different structured outputs.

The first extraction pipeline should remain schema-constrained and replaceable:

- parsed sections are the default extraction input for local-first corpora
- extraction clients return a typed bundle rather than directly mutating the database
- persistence is handled inside Paperbase so Arxie and future workers reuse the same storage contract
- collection runners are the execution engine behind worker-dispatched extraction jobs
- glossary terms are stored as first-class canonical entities because field-specific databases often need shared vocabulary and benchmark definitions alongside result rows

## Current API Surface

Paperbase now exposes extraction profile management plus queued collection extraction:

- `GET /api/v1/extraction-profiles`
- `POST /api/v1/extraction-profiles`
- `POST /api/v1/collections/{collection_id}/extract`
- `GET /api/v1/jobs/{job_id}`

The contract is now intentionally asynchronous:

- the API validates the request and enqueues a `collection_extract` job
- the worker claims that job and executes `CollectionExtractionRunner`
- clients poll `GET /api/v1/jobs/{job_id}` to observe `pending`, `running`,
  `completed`, or `failed`

This keeps the current local-first extraction stack operational for curated
collections without keeping long-running extraction work inside the API process.

## Current Local Corpus Status

The `SamplePapers` scRegNet collection is now fully operational as a local-first
field database:

- all `11` imported papers have a latest `completed` extraction run
- the extraction contract now tolerates malformed scalar fragments inside entity
  lists from function-calling outputs, instead of failing the entire paper run
- the built-in `sc_regnet` preset is attached to the collection and can be
  rerun locally without the future worker stack

This is an important local-first milestone because it proves the collection can
already behave like a persistent field database rather than a PDF folder plus ad
hoc prompts.

## Artifact Contracts

Paperbase now also stabilizes figure and table artifact storage even before
heavier OCR or diagram extraction is added:

- figure records preserve page number, figure label, caption, optional bounding
  boxes, and storage URIs
- table records preserve page number, table label, caption, optional bounding
  boxes, storage URIs, and a lightweight JSON payload for structured table shape
- paper and collection browse surfaces can expose those artifacts immediately,
  while richer extraction continues to evolve behind the same storage contract

This keeps figure and table work on the same architectural path as datasets,
methods, metrics, and result rows: typed candidates first, canonical storage
second, and browse/compare surfaces built on top.

## Built-In Field Presets

Paperbase now includes a built-in `sc_regnet` preset derived from the local
`SamplePapers` corpus. It encodes the current field model for single-cell gene
regulatory network inference:

- task families around GRN link prediction and regulatory graph reconstruction
- modalities such as `scRNA-seq`, `scATAC-seq`, and `multiome`
- benchmarks such as `BEELINE` and related single-cell evaluation settings
- prior-knowledge sources, relation types, and split strategies that materially
  change how papers in this field should be compared

This keeps the extraction pipeline generic while still giving local-first users a
real domain-specific schema for their curated collection.
