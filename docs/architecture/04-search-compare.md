# Search And Compare

## Search Modes

Paperbase should support:

- keyword search
- semantic retrieval
- structured filtering

Search scope should be explicit:

- full corpus
- one collection
- one field-specific curated database

The current DB-backed Paperbase API now supports practical filtering on the
existing paper search route for:

- collection membership
- publication year bounds
- venue
- author
- tag
- dataset
- method
- metric
- extraction state

Paper fetch responses now also expose `authors` and `tags`, so paper records can
behave like structured corpus entries rather than title-only shells.

## Current Module Map

The first search/read-model layer should stay intentionally thin:

- `src/paperbase/search/index_templates.py` — Elasticsearch-style mappings for paper, chunk, and figure documents
- `src/paperbase/search/indexer.py` — deterministic builders for paper, chunk, and figure documents
- `src/paperbase/search/query_builder.py` — text, filter, and vector query composition

This layer is a read-model scaffold, not a full search service. It should be reusable by the future Paperbase API, worker indexing jobs, and Arxie retrieval gateway.

Even before live Elasticsearch-backed indexing is added, the Paperbase API should
still expose enough structured filters for curated local-first databases to be
usable through SQL-backed search.

## Comparison

Comparison is a database feature, not just an agent prompt.

The database should be able to return normalized comparison slices such as:

- method families
- dataset/metric result rows
- engineering tricks across a collection
- figures or captions tied to evidence

## Current Comparison Surface

The Paperbase API now exposes three comparison workflows directly on top of the
local-first database:

- `POST /api/v1/compare/results`
- `POST /api/v1/compare/methods`
- `POST /api/v1/compare/engineering-tricks`

These routes make comparison behave like a database query surface rather than a
single agent prompt:

- result comparison can filter within a collection and optionally return evidence
  spans for each result row
- method comparison summarizes result slices by method across a paper set and
  returns the best observed row per method
- engineering-trick comparison can summarize recurring tricks across a corpus or
  within a method family

This is still DB-backed, not search-index-backed. It is the current local-first
comparison layer while the richer read-model and Elasticsearch workflow are
still open.

## Structured Browse Surface

Search and comparison are not enough on their own. After a collection extraction
run, users also need to inspect the extracted artifacts for a specific paper.

The current API now supports that browse layer through:

- `GET /api/v1/papers/{paper_id}/structured-data`
- `GET /api/v1/collections/{collection_id}/structured-summary`

That endpoint returns the paper-level extracted datasets, methods, metrics,
result rows, glossary terms, findings, engineering tricks, extraction runs, and
evidence spans. It gives the local-first database a usable inspection surface
before the future UI and richer search indexing layers are finished.

The collection structured summary endpoint now also normalizes obvious metric
aliases such as long-form `Area Under the Receiver Operating Characteristic
Curve` into `AUROC` and collapses duplicate summary entries. That keeps the
local-first browse layer readable even before richer metric normalization and
dedicated comparison tables are added.

## Search Operations Surface

The Paperbase search runtime is now exposed operationally through the API, not
just as internal library code:

- `GET /api/v1/search/status`
- `POST /api/v1/search/reindex`

This closes the gap between “the runtime classes exist” and “the platform can
actually trigger and observe reindexing.” The current behavior is intentionally
thin:

- report whether a backend is configured
- run a full rebuild of paper, chunk, and figure read models through the
  configured backend
- return explicit `503` errors when the search backend is unavailable

This is still a local-first operator surface. It is not yet a full workerized
index-management system.
