# Paperbase Core

## Purpose

Paperbase is the persistent source of truth for research data.

It is not just a cache and not just a vector store. It is the structured database for:

- papers
- source provenance
- PDFs and derived artifacts
- extracted research entities
- evidence links
- user collections and annotations

## Storage Layers

### Canonical store

Use PostgreSQL for normalized entities and durable workflow data.

For local-first development, the same schema must run cleanly on SQLite so a single user can build and curate a corpus without external services.

### Object storage

Use object storage for PDFs, parser artifacts, and figure assets.

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
- `src/paperbase/db/models.py` — canonical relational schema
- `src/paperbase/db/session.py` — engine and session factory helpers
- `src/paperbase/db/bootstrap.py` — local schema initialization
- `src/paperbase/db/repositories.py` — first write paths for papers, extraction profiles, collections, and annotations

These modules are intentionally thin. They should encode durable boundaries now without prematurely building API, worker, or ingest orchestration around them.
