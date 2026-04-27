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
- long-running ingest, parse, extraction, and reindex work is queued by the API
  and executed by `services/paperbase_worker/`

The first product UI is now also served from `services/paperbase_api/` as a
build-free web surface:

- `/` is the public Arxie homepage
- `/app` is the saved-workspace UI over the local Paperbase APIs

That UI is still intentionally thin: it sits directly on the workspaces,
collections, papers, chunk/artifact search, and jobs APIs so the product can
ship a real research workspace before a larger frontend stack is introduced.

The workspace app also includes figure and table browse/comparison surfaces, so
collaborators can inspect visual evidence without dropping into raw PDFs or ad
hoc scripts.

The current V1 surface now also exposes:

- first-class `limitations` alongside findings, result rows, figures, and tables
- provider-backed ingest by DOI, arXiv ID, and OpenAlex identifier
- backend-first collection-aware search when a search backend is configured
- workspace-aware `/answer` and `/api/chat` execution paths
- a Paperbase structured-evidence tool that the Arxie agent can call directly
