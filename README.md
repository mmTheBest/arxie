# Arxie

*A self-hostable AI research agent that learns from your paper corpus.*

Arxie helps researchers turn a curated folder of papers into a field-aware
working context. It parses and extracts evidence from full papers, learns
recurring methods, datasets, metrics, result patterns, limitations, and
validation norms, then uses that corpus model to support literature comparison,
benchmark planning, experiment design, draft critique, proposal refinement, and
idea review.

The system combines:

- a canonical paper database named Paperbase
- structured extraction over full papers
- derived research memory and field-graph records
- hybrid search and comparison surfaces
- a study-aware research assistant that reasons over paper evidence plus user-provided work context

This repository now ships the `v0.2.0` product surface described in the April 14 PRD:
persistent corpora, saved studies, structured evidence, provider-backed ingest, and a
browser study workspace at `/app`.

## What You Can Do

- build a curated paper collection from local PDFs, DOI, arXiv, and OpenAlex identifiers
- parse papers into sections, chunks, figures, and tables
- extract datasets, methods, metrics, result rows, findings, limitations, glossary terms,
  engineering tricks, and research-design elements such as baselines, ablations,
  protocols, validity threats, and reproducibility signals
- search papers, chunks, and artifacts with Elasticsearch-backed hybrid retrieval
  in the self-hosted stack, plus explicit local fallbacks for development
- ask Arxie to compare results, methods, tricks, figures, and tables through the Study chat
- save studies with a collection, query, focus note, filters, pinned papers, and explicit work sources
- generate collection-grounded research artifacts: experiment plans, hypotheses,
  benchmark plans, revision plans, assumption maps, field patterns, and critiques
  with evidence payloads
- add explicit code, draft, result, or text-note sources so Arxie can reason from
  what you have already built or written without auto-scanning your filesystem
- see each collection's readiness for search and structured comparison before choosing the next action
- run Arxie answer, chat, literature review, and proposal evidence flows against that saved context

## Current Scope

`v0.2.0` is production-ready for a **single-user, self-hosted** deployment.

The current release is optimized for local and self-hosted research workflows:
import a corpus, process it, add explicit study context, and ask Arxie to
generate evidence-grounded research artifacts. Advanced field modeling, richer
source parsing, stronger model-output contracts, and broader evaluation gates
are active development areas.

The code keeps ownership boundaries so the system can grow toward hosted use
later, but the supported deployment model today is one operator running one
server stack.

## Quick Start

### 1. Install

```bash
git clone https://github.com/mmTheBest/arxie.git
cd arxie

python -m venv .venv
source .venv/bin/activate
pip install -e .
```

If you specifically want local embedding-model dependencies on the host, install:

```bash
pip install -e .[local-embeddings]
```

### 2. Configure

```bash
cp .env.example .env
```

Set at least:

- `OPENAI_API_KEY`

For the full self-hosted stack, `.env.example` also includes the Paperbase runtime variables for PostgreSQL, Elasticsearch, Redis, MinIO-compatible object storage, queue dispatch, cache lifecycle, and semantic search configuration.

Important runtime defaults:

- `PAPERBASE_WORKER_QUEUE_BACKEND=redis` in the shipped server stack
- `PAPERBASE_OBJECT_STORE_BACKEND=s3` in the shipped server stack
- `PAPERBASE_EMBEDDING_PROVIDER=openai` for production semantic retrieval
- `PAPERBASE_WORKER_PROJECT_ID` can bind a worker to one opened project; leave
  it empty for the default database worker
- `PAPERBASE_UPLOAD_MAX_FILE_COUNT`, `PAPERBASE_UPLOAD_MAX_SINGLE_FILE_BYTES`,
  and `PAPERBASE_UPLOAD_MAX_TOTAL_BYTES` bound browser PDF-folder uploads while
  raw multipart request bytes are streamed
- `PAPERBASE_HOSTED_MODE=true` requires project-open and local-library import
  paths to live under `PAPERBASE_LOCAL_PATH_IMPORT_ALLOWED_ROOTS`

If you intentionally want a lighter local process mode, you can switch to:

- `PAPERBASE_WORKER_QUEUE_BACKEND=db`
- `PAPERBASE_OBJECT_STORE_BACKEND=filesystem`
- `PAPERBASE_EMBEDDING_PROVIDER=deterministic`

### 3. Start The Self-Hosted Stack

Start infrastructure:

```bash
docker compose -f infra/docker-compose.paperbase.yml up -d postgres elasticsearch minio redis
```

Apply schema migrations:

```bash
docker compose -f infra/docker-compose.paperbase.yml run --rm paperbase-migrate
```

Start the API and worker:

```bash
docker compose -f infra/docker-compose.paperbase.yml up -d paperbase-api paperbase-worker
```

### 4. Open Arxie

- Homepage: `http://localhost:8080/`
- Study app: `http://localhost:8080/app`
- Liveness: `http://localhost:8080/livez`
- Readiness: `http://localhost:8080/readyz`

### Shortcut Launch

If you want a more app-like local workflow, use the launcher command:

```bash
arxie-local run
```

That boots the lighter single-user local stack, waits for readiness, and opens
`/app`. By default it starts PostgreSQL, MinIO, Redis, the API, and the worker.
If you explicitly want the heavier backend-search service too, use:

```bash
arxie-local run --with-search
```

Other useful shortcuts:

```bash
arxie-local open
arxie-local down
arxie-local run --rebuild
arxie-local install-shortcut
```

`arxie-local install-shortcut` writes a double-clickable `Arxie.command` launcher
to your Desktop by default.

On the first launch, Arxie may need a few minutes to start Colima, build the
application images, and boot the local stack. The shipped local Compose profile
is tuned for a single-user machine, including a smaller Elasticsearch heap so
the default stack can run on modest laptop memory.

For the single-user local path, Arxie does not hard-block readiness on the
search backend. The default launcher path skips Elasticsearch entirely so parse,
extraction, and the dashboard remain reliable on a modest laptop. The local
Compose services mount the current `src/` and `services/` directories, so code
updates are picked up on restart. Use `arxie-local run --rebuild` only after
dependency or Dockerfile changes. If you later start Arxie with `--with-search`,
the workspace can use the backend search surface when Elasticsearch is healthy.

### 5. Use Your Own Local Paper Collection

The browser study app is now enough for the single-user local workflow:

1. open `http://localhost:8080/app`
2. start in **Library** and use **Upload PDF Folder**
3. select a local folder containing PDFs and optionally set a collection title
4. stay in **Library** and watch the processing log until the ingest job finishes
5. use the Library paper list to parse all unprocessed papers
   or only selected papers
6. extract all unextracted papers, or select specific text-ready papers and extract
   just those
7. switch to **Study** to search the collection, inspect evidence, and label
   papers as exemplars or baselines
8. save the study, then add explicit sources such as a draft path, code file path,
   results file path, or text note
9. ask Arxie for experiment ideas, benchmark design, revision priorities,
   assumption checks, hypotheses, field patterns, or critiques grounded in the
   paper collection plus those explicit sources
10. ask comparison questions in the Study chat once extraction is ready; save or
    export the generated outputs you want to keep

If you are running the API and worker directly on the host instead of in Docker,
the Library module also exposes an advanced absolute-path import form.

For scripted or operator-driven ingestion, see [docs/runbooks/paperbase-ingest.md](docs/runbooks/paperbase-ingest.md).

## Local Process Mode

If you prefer running the services without Compose:

```bash
paperbase-db upgrade
paperbase-api
paperbase-worker
```

In this mode, the default `.env.example` still points at the self-hosted stack.
If you want a no-MinIO/no-Redis local run, switch the queue, object-store, and
embedding settings as described above before launching the processes.

Useful make targets:

```bash
make paperbase-db-upgrade
make paperbase-api
make paperbase-worker
make paperbase-compose-config
```

## Adjacent CLI And RA API

The `src/ra` assistant and proposal workflows still ship with the repo as
developer-facing and scripted interfaces. The browser product path is `/app`,
served by `services/paperbase_api/`.

Examples:

```bash
ra query "What are recent approaches to long-context LLMs?"
ra lit-review "attention mechanisms in computer vision"
ra trace "Attention Is All You Need"
ra chat
```

The RA FastAPI surface is still available too:

```bash
uvicorn ra.api.app:app --host 0.0.0.0 --port 8000
```

## Product Architecture

```text
src/ra/                 Assistant workflows, CLI, and legacy REST API
src/paperbase/          Canonical schema, ingest, parse, extract, search
services/paperbase_api/ Browser-facing corpus API and UI
services/paperbase_worker/ Background job execution
infra/                  Self-hosting stack and environment files
```

Public runtime docs live in `docs/API.md`, `docs/MIGRATION.md`, and
`docs/USAGE_EXAMPLES.md`. Contributor architecture notes are maintained on
development branches.

## Known Limits

- Deployment is single-user and self-hosted, not collaborative or multi-tenant.
- Figure and table extraction is phase-1 caption-driven extraction, not full OCR or chart digitization.
- The legacy RA API and the Paperbase product API coexist; the browser product surface is the Paperbase API at port `8080`.

## License

MIT
