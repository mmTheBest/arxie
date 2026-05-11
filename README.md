# Arxie

*An AI research workspace backed by a persistent paper database.*

Arxie is a self-hostable research system for serious literature work. It combines:

- a canonical paper database named Paperbase
- structured extraction over full papers
- hybrid search and comparison surfaces
- a study-aware research assistant that reasons over paper evidence plus user-provided work context

This release branch ships the `v0.2.0` product surface: persistent corpora, saved studies, structured evidence, comparison workflows, provider-backed ingest, and a browser study workspace at `/app`.

## What You Can Do

- build a curated paper collection from local PDFs, DOI, arXiv, and OpenAlex identifiers
- parse papers into sections, chunks, figures, and tables
- extract datasets, methods, metrics, result rows, findings, limitations, glossary terms,
  engineering tricks, and research-design elements such as baselines, ablations,
  protocols, validity threats, and reproducibility signals
- search papers, chunks, and artifacts with Elasticsearch-backed hybrid retrieval
  in the self-hosted stack, plus explicit local fallbacks for development
- compare results, methods, tricks, figures, and tables across a corpus slice
- save studies with a collection, query, focus note, filters, pinned papers, and explicit work sources
- generate collection-grounded research artifacts: experiment plans, hypotheses,
  benchmark plans, revision plans, assumption maps, field patterns, and critiques
  with evidence payloads
- add explicit code, draft, result, or text-note sources so Arxie can reason from
  what you have already built or written without auto-scanning your filesystem
- see each collection's readiness for search and structured comparison before choosing the next action
- run Arxie answer, chat, literature review, and proposal evidence flows against that saved context

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

## Product Architecture

```text
src/ra/                 Assistant workflows, CLI, and legacy REST API
src/paperbase/          Canonical schema, ingest, parse, extract, search
services/paperbase_api/ Browser-facing corpus API and UI
services/paperbase_worker/ Background job execution
infra/                  Self-hosting stack and environment files
```

Detailed architecture, planning notes, tests, and internal developer process files live on the private development branch, not on this release branch.

## License

MIT
