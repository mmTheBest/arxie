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

