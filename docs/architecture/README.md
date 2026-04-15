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

## Current Direction

Arxie is being evolved into:

- a persistent research database layer named Paperbase
- an assistant layer that runs on top of that database

The user experiences one product, but the code should keep those responsibilities separate.

Operationally, the current Paperbase branch is now split three ways:

- the canonical DB and domain logic live under `src/paperbase/`
- the query/CRUD API contract lives under `services/paperbase_api/`
- long-running extraction and reindex work is queued by the API and executed by
  `services/paperbase_worker/`

The first product UI is now also served from `services/paperbase_api/` as a
build-free local console. That UI is intentionally thin: it sits directly on the
collections, papers, search, and jobs APIs so the product can ship a real
operator surface before a larger frontend stack is introduced.
