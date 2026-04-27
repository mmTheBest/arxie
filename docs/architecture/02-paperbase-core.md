# Paperbase Core

## Purpose

Paperbase is the persistent source of truth for research data.

It is not just a cache and not just a vector store. It is the structured database for:

- papers
- canonical venues
- canonical author and tag metadata
- source provenance
- PDFs and derived artifacts
- extracted research entities, including limitations
- evidence links
- user collections and annotations

## Storage Layers

### Canonical store

Use PostgreSQL for normalized entities and durable workflow data.

For local-first development, the same schema must run cleanly on SQLite so a single user can build and curate a corpus without external services.

### Object storage

Use object storage for PDFs, parser artifacts, and figure assets.

In the shipped self-hosted runtime, provider-downloaded PDFs and imported local
library PDFs are copied into canonical object storage and referenced by stable
`s3://bucket/key` URIs.

Local filesystem storage remains only as a deliberate dev/test fallback for
single-user runs that do not want MinIO.

### Search/read models

Use Elasticsearch for keyword search, semantic retrieval, filtering, and comparison-oriented read models.

## Expandability Constraint

Even though v1 is local-first, the schema should preserve future support for:

- ownership fields
- additional providers
- larger corpora
- background reindex and re-extraction at scale

## Current Module Map

The first concrete Paperbase modules are:

- `src/paperbase/config.py` — environment-driven platform configuration
- `src/paperbase/version.py` — shared release version surface for API responses and packaging
- `src/paperbase/db/models.py` — canonical relational schema, including normalized `venues`, `authors`, `paper_authors`, `tags`, and `paper_tags`
- `src/paperbase/db/session.py` — engine and session factory helpers
- `src/paperbase/db/bootstrap.py` — local schema initialization
- `src/paperbase/db/cli.py` — packaged migration commands over Alembic
- `src/paperbase/db/repositories.py` — write paths for papers, venue/author/tag metadata, extraction profiles, collections, and annotations
- `src/paperbase/object_store.py` — filesystem and S3-compatible object-store adapters for canonical paper assets

These modules are intentionally thin. They should encode durable boundaries now without prematurely building API, worker, or ingest orchestration around them.

The first service layer now exists at:

- `services/paperbase_api/app.py` — FastAPI entrypoint for corpus search, fetch, fulltext, figures, and comparison routes
- `services/paperbase_api/main.py` — packaged API process entrypoint
- `services/paperbase_api/health.py` — readiness checks for DB, search, Redis, and object-store dependencies
- `services/paperbase_api/routes/` — route groups for `search`, `papers`, `compare`, `collections`, `ingest`, `extraction`, `workspaces`, and `jobs`
- `services/paperbase_worker/main.py` — packaged worker process entrypoint

This service should stay independent from `src/ra/api/app.py`. Arxie will consume it through a gateway layer rather than sharing endpoint code directly.
