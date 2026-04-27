# Architecture Docs

This folder is the collaborator-facing system map for Arxie.

The goal is simple: a new contributor should be able to read this folder and understand:

1. what the major modules are
2. what each module owns
3. where a new change should live

## Document Map

- [01-system-overview.md](01-system-overview.md)
- [02-paperbase-core.md](02-paperbase-core.md)
- [03-ingest-and-extraction.md](03-ingest-and-extraction.md)
- [04-search-compare.md](04-search-compare.md)
- [05-assistant-integration.md](05-assistant-integration.md)
- [06-collections-and-annotations.md](06-collections-and-annotations.md)
- [07-design-principles.md](07-design-principles.md)

## Operations Docs

Contributor-facing runbooks live in `docs/runbooks/`.

- [paperbase-ingest.md](../runbooks/paperbase-ingest.md)
- [paperbase-reindex.md](../runbooks/paperbase-reindex.md)

## Current Product State

Arxie is now shipped as:

- a persistent research database layer named Paperbase
- an assistant layer that runs on top of that database

The user experiences one product, but the code should keep those responsibilities separate.

Operationally, the runtime is split three ways:

- the canonical DB and domain logic live under `src/paperbase/`
- the query/CRUD API contract lives under `services/paperbase_api/`
- long-running ingest, parse, extraction, and reindex work is queued by the API
  and executed by `services/paperbase_worker/` through a Redis-backed runtime queue

The first product UI is served from `services/paperbase_api/` as a build-free
web surface:

- `/` is the public Arxie homepage
- `/app` is the saved-workspace UI over the local Paperbase APIs

That UI is still intentionally thin: it sits directly on the workspaces,
collections, papers, chunk/artifact search, and jobs APIs so the product can
ship a real research workspace before a larger frontend stack is introduced.

For the supported single-user local workflow, `/app` is now enough to:

- import a local PDF folder into a new collection
- queue parse and extraction jobs for that collection
- monitor background job completion
- browse the resulting papers, structured evidence, and saved workspace state

The workspace app also includes figure and table browse/comparison surfaces, so
collaborators can inspect visual evidence without dropping into raw PDFs or ad
hoc scripts.

The current release surface also exposes:

- first-class `limitations` alongside findings, result rows, figures, and tables
- provider-backed ingest by DOI, arXiv ID, and OpenAlex identifier
- backend-first collection-aware search when a search backend is configured
- workspace-aware `/answer` and `/api/chat` execution paths
- a Paperbase structured-evidence tool that the Arxie agent can call directly
- dependency-aware `/health`, `/livez`, and `/readyz` service probes
- packaged `paperbase-api`, `paperbase-worker`, and `paperbase-db` entrypoints
- a self-hostable Compose stack with PostgreSQL, Elasticsearch, MinIO, Redis,
  API, worker, and migration service wiring
- canonical object-store-backed PDF persistence in the shipped runtime, with
  filesystem fallback only for local development
