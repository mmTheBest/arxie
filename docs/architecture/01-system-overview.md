# System Overview

## Product Shape

Arxie is a research database plus an agent that operates on that database.

The system has two top-level layers:

- **Paperbase** — persistent corpus storage, extraction, indexing, and comparison
- **Arxie Assistant** — search, chat, synthesis, literature review, and proposal workflows

## Initial Deployment Mode

V1 is single-user and self-hosted, with local-first development support.

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
- `Dockerfile` — shared production image for migrate, API, and worker commands

## Current V1 Flow

The finished local-first flow is now:

1. import papers from a local library or provider identifiers
2. persist canonical paper, provenance, venue, author, and tag records
3. parse stored PDFs into sections, chunks, figures, and tables
4. extract structured entities such as datasets, methods, metrics, result rows,
   findings, limitations, glossary terms, engineering tricks, and research-design
   elements
5. enqueue background jobs for ingest, parse, extraction, provider refresh, and reindex
   into the Redis-backed worker queue while keeping the DB as the audit/status surface
6. search the corpus by paper, chunk, or artifact, with backend-first hybrid
   retrieval when a search backend is configured
7. save a workspace and reuse it across browse, compare, answer, chat, lit
   review, and proposal evidence workflows
8. run collection-grounded research-agent jobs that persist threads, messages,
   artifacts, evidence payloads, and paper research labels

This is now a real product path, not just a planned decomposition.

For the single-user local path, the intended front door is `/app`. The shipped
UI is now module-based rather than one operator console:

- `Library` for import, collection selection, readiness, and paper-level preparation controls
- `Workspace` for collection-scoped search, reading, evidence inspection, and saving research context
- `Research` for collection-grounded experiment plans, hypotheses, field patterns, critiques, and evidence review
- `Compare` for structured evidence comparison across an extraction-ready collection
- `Jobs` for monitoring and diagnosing ingest, parse, extraction, and reindex activity
- `Settings` for local runtime, search backend, object-store, worker, and selected context state

Collection readiness is exposed through the collection API and structured
summary responses so the browser does not infer preparation state from partial
UI state. The readiness inputs include paper count, parsed paper count,
extracted paper count, latest parse/extraction job status, and failed job count.
That means a user can import a local PDF directory, process all papers or a
selected subset, move from imported to text-ready to evidence-ready, search
within it, ask the Research agent to design studies against the collected
evidence, compare extracted evidence, and save the investigation without
dropping to scripts.

## Self-Hosted Runtime

The production stack now expects:

- PostgreSQL for canonical storage
- Elasticsearch for backend search and reindex
- Redis for the production worker queue and dispatch path
- MinIO for canonical PDF and artifact storage
- one Paperbase API service
- one Paperbase worker service
- one migration entrypoint before API and worker startup

The code still supports SQLite fallback for local development and tests, but the
intended deployed surface is the full self-hosted stack.

For the single-user launcher path, search readiness is intentionally
non-blocking. If the local Elasticsearch container is unavailable, Arxie still
comes up over the canonical DB/object-store/runtime surface and uses its
database-backed fallback paths until backend search is restored.
