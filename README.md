# Arxie

*An AI research workspace backed by a persistent paper database.*

Arxie is a self-hostable research system for serious literature work. It combines:

- a canonical paper database named Paperbase
- structured extraction over full papers
- hybrid search and comparison surfaces
- a workspace-aware research assistant that runs on top of that database

This repository now ships the `v0.2.0` product surface described in the April 14 PRD: persistent corpora, saved workspaces, structured evidence, comparison workflows, provider-backed ingest, and a browser workspace at `/app`.

## What You Can Do

- build a curated paper collection from local PDFs, DOI, arXiv, and OpenAlex identifiers
- parse papers into sections, chunks, figures, and tables
- extract datasets, methods, metrics, result rows, findings, limitations, glossary terms, and engineering tricks
- search papers, chunks, and artifacts with SQL fallback or Elasticsearch-backed retrieval
- compare results, methods, tricks, figures, and tables across a corpus slice
- save workspaces with a collection, query, focus note, filters, and pinned papers
- run Arxie answer, chat, literature review, and proposal evidence flows against that saved context

## Current Scope

`v0.2.0` is production-ready for a **single-user, self-hosted** deployment.

It is not a multi-tenant SaaS product. The code keeps ownership boundaries so the system can grow later, but the supported deployment model today is one operator running one server stack.

## Quick Start

### 1. Install

```bash
git clone https://github.com/mmTheBest/arxie.git
cd arxie

python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Configure

```bash
cp .env.example .env
```

Set at least:

- `OPENAI_API_KEY`

For the full self-hosted stack, `.env.example` also includes the Paperbase runtime variables for PostgreSQL, Elasticsearch, Redis, and object storage.

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
- Workspace app: `http://localhost:8080/app`
- Liveness: `http://localhost:8080/livez`
- Readiness: `http://localhost:8080/readyz`

## Local Process Mode

If you prefer running the services without Compose:

```bash
paperbase-db upgrade
paperbase-api
paperbase-worker
```

Useful make targets:

```bash
make paperbase-db-upgrade
make paperbase-api
make paperbase-worker
make paperbase-compose-config
```

## Legacy CLI And RA API

The original `src/ra` assistant still ships with the repo.

Examples:

```bash
ra query "What are recent approaches to long-context LLMs?"
ra lit-review "attention mechanisms in computer vision"
ra trace "Attention Is All You Need"
ra chat
```

The legacy FastAPI surface is still available too:

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

Contributor-facing system docs live in [docs/architecture](docs/architecture/README.md).

## Known Limits

- Deployment is single-user and self-hosted, not collaborative or multi-tenant.
- Figure and table extraction is phase-1 caption-driven extraction, not full OCR or chart digitization.
- The legacy RA API and the Paperbase product API coexist; the browser product surface is the Paperbase API at port `8080`.

## License

MIT
