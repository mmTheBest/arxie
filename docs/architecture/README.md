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
- `/app` is the study-first UI over the local Paperbase APIs

That UI is still intentionally thin, but it is now organized around the two
surfaces a local researcher needs most:

- `Study` owns search, paper reading, explicit user work sources, research-agent
  threads, artifacts, paper labels, evidence payloads, structured comparison,
  activity, and local runtime context
- `Library` owns import, collection selection, readiness labels, paper-level
  parse/extract controls, and processing logs

It still sits directly on the studies/workspaces, collections, papers,
chunk/artifact search, compare, jobs, and study-source APIs so the product can
ship a real research workspace before a larger frontend stack is introduced.

For the supported single-user local workflow, `/app` is now enough to:

- upload a local PDF folder into a new collection
- process all unprocessed papers or a selected subset from imported to text-ready to evidence-ready
- monitor background job completion in Library or the Study activity panel
- browse the resulting papers and evidence in Study
- add explicit text, code, draft, or result sources to the saved Study
- generate evidence-grounded experiment plans, benchmark plans, revision plans,
  assumption maps, hypotheses, field patterns, and critiques in Study
- inspect structured comparison slices in the Study compare panel
- save the investigation as a reusable study state

The study app also includes figure and table browse/comparison surfaces, so
collaborators can inspect visual evidence without dropping into raw PDFs or ad
hoc scripts.

The current release surface also exposes:

- first-class `limitations` and research-design elements alongside findings,
  result rows, figures, and tables
- durable Research threads, messages, artifacts, paper labels, explicit study
  sources, and `research_agent_run` worker jobs
- provider-backed ingest by DOI, arXiv ID, and OpenAlex identifier
- an `arxie-local` launcher command for single-user local boot/open/stop flows,
  with mounted local source, a lighter default stack, optional `--rebuild`, and
  optional `--with-search` backend search
- backend-first collection-aware search when a search backend is configured
- workspace-aware `/answer` and `/api/chat` execution paths
- a Paperbase structured-evidence tool that the Arxie agent can call directly
- dependency-aware `/health`, `/livez`, and `/readyz` service probes
- packaged `paperbase-api`, `paperbase-worker`, and `paperbase-db` entrypoints
- a self-hostable Compose stack with PostgreSQL, Elasticsearch, MinIO, Redis,
  API, worker, and migration service wiring
- canonical object-store-backed PDF persistence in the shipped runtime, with
  filesystem fallback only for local development
