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

- `src/paperbase/search/index_templates.py` — Elasticsearch-style mappings for paper, chunk, figure, and table documents
- `src/paperbase/search/indexer.py` — deterministic builders for paper, chunk, figure, and table documents
- `src/paperbase/search/query_builder.py` — text, filter, and vector query composition
- `src/paperbase/search/embeddings.py` — deterministic local embedding generation for backend-assisted semantic search
- `src/paperbase/search/runtime.py` — backend integration plus reindex orchestration across all read models

This layer is a read-model scaffold, not a full search service. It should be reusable by the future Paperbase API, worker indexing jobs, and Arxie retrieval gateway.

Even before live Elasticsearch-backed indexing is added, the Paperbase API should
still expose enough structured filters for curated local-first databases to be
usable through SQL-backed search.

The current implementation now exposes three search surfaces:

- `GET /api/v1/search/papers`
- `GET /api/v1/search/chunks`
- `GET /api/v1/search/artifacts`

Those routes fall back to SQL when no backend is configured, and switch to
backend-assisted queries with deterministic local embeddings when a search
backend is available.

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
- `POST /api/v1/compare/figures`
- `POST /api/v1/compare/tables`

These routes make comparison behave like a database query surface rather than a
single agent prompt:

- result comparison can filter within a collection and optionally return evidence
  spans for each result row
- method comparison summarizes result slices by method across a paper set and
  returns the best observed row per method
- engineering-trick comparison can summarize recurring tricks across a corpus or
  within a method family
- figure comparison can filter artifact slices by collection and method family
  so visual evidence is queryable instead of buried inside PDFs
- table comparison can filter tabular artifact slices by collection and method
  family so benchmark tables become database browse objects

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
result rows, glossary terms, findings, engineering tricks, extraction runs,
evidence spans, figures, and tables. It gives the local-first database a usable
inspection surface before the future UI and richer search indexing layers are
finished.

The collection structured summary endpoint now also normalizes obvious metric
aliases such as long-form `Area Under the Receiver Operating Characteristic
Curve` into `AUROC` and collapses duplicate summary entries. That keeps the
local-first browse layer readable even before richer metric normalization and
dedicated comparison tables are added.

The same summary surface now exposes collection-level figure and table slices,
which the local Paperbase Console uses to render an artifact panel alongside the
existing structured entity summaries.

## Search Operations Surface

The Paperbase search runtime is now exposed operationally through the API, not
just as internal library code:

- `GET /api/v1/search/status`
- `POST /api/v1/search/reindex`
- `GET /api/v1/jobs/{job_id}`

This closes the gap between “the runtime classes exist” and “the platform can
actually trigger and observe reindexing.” The current behavior is now
worker-backed rather than API-blocking:

- report whether a backend is configured
- enqueue a `search_reindex` background job from the API
- execute the actual rebuild of paper, chunk, figure, and future table read
  models inside the Paperbase worker
- expose job state through the jobs API for polling clients and the future UI

The local Paperbase Console now uses these wider search surfaces directly:

- paper search for the main browse list
- chunk search for section-level query hits
- artifact search for figure/table query hits

This is still a local-first operator surface. The queue is DB-backed today so it
can grow later into Redis or a larger broker-backed worker topology without
changing the product-facing job contract.
