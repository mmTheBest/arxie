# System Overview

## Product Shape

Arxie is a research database plus an agent that operates on that database.

The system has two top-level layers:

- **Paperbase** — persistent corpus storage, extraction, indexing, and comparison
- **Arxie Assistant** — search, chat, synthesis, literature review, and proposal workflows

## Initial Deployment Mode

V1 is single-user and local-first.

That means:

- one local operator
- one local Paperbase instance
- one local set of collections and annotations

But the architecture should leave room for:

- multi-user ownership later
- hosted deployments later
- larger scholarly databases later

## Planned Runtime Modules

- `src/ra/` — current assistant application
- `src/paperbase/` — database/platform package
- `services/paperbase_api/` — platform API service
- `services/paperbase_worker/` — async ingest/extract/index worker
- `infra/` — local stack and environment setup

## Current V1 Flow

The finished local-first flow is now:

1. import papers from a local library or provider identifiers
2. persist canonical paper, provenance, venue, author, and tag records
3. parse stored PDFs into sections, chunks, figures, and tables
4. extract structured entities such as datasets, methods, metrics, result rows,
   findings, limitations, glossary terms, and engineering tricks
5. enqueue background jobs for parse, extraction, provider refresh, and reindex
6. search the corpus by paper, chunk, or artifact, with backend-first hybrid
   retrieval when a search backend is configured
7. save a workspace and reuse it across browse, compare, answer, chat, lit
   review, and proposal evidence workflows

This is now a real product path, not just a planned decomposition.
