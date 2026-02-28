# Academic Research Assistant

AI-powered research workflow for paper discovery, metadata retrieval, and citation-grounded answers.

## Overview

Academic Research Assistant combines:
- Multi-source retrieval from Semantic Scholar and arXiv
- A LangChain-based research agent for synthesized answers with references
- A CLI for retrieval/evaluation workflows
- A FastAPI service for REST integrations

The project is designed for local development first, with optional Docker deployment.

## Features

- Unified retrieval layer with source deduplication (`semantic_scholar`, `arxiv`, or both)
- Citation formatting and reference-oriented answer generation
- Batch API endpoints for concurrent search and retrieve operations
- Evaluation harness for dataset-based benchmarking
- Local Chroma cache support for retrieval acceleration
- Structured JSON logging and retry/rate-limiting around upstream calls
- Unit and integration test suites

## Prerequisites

- Python `3.10+` (Docker image uses Python `3.11`)
- `pip` and virtual environment support
- Network access for Semantic Scholar/arXiv queries
- OpenAI API key for answer generation and evaluation flows

## Installation

```bash
cd /Users/mm/Projects/academic-research-assistant
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

Install API server runtime (if running FastAPI locally):

```bash
pip install "uvicorn[standard]>=0.30.0"
```

## Configuration

### Required and Supported Environment Variables

The runtime reads these variables directly:

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `OPENAI_API_KEY` | Required for agent-based answer/eval flows | None | OpenAI credential used by `ResearchAgent` |
| `RA_MODEL` | No | `gpt-4o-mini` | OpenAI model for answer generation |
| `SEMANTIC_SCHOLAR_API_KEY` | No | None | Optional key for higher Semantic Scholar limits |
| `RA_LOG_LEVEL` | No | `INFO` | Logging level (`DEBUG`, `INFO`, etc.) |
| `RA_LOG_DIR` | No | `data/logs` | Directory for structured log file output |

Example shell setup:

```bash
export OPENAI_API_KEY="sk-..."
export RA_MODEL="gpt-4o-mini"
export SEMANTIC_SCHOLAR_API_KEY=""
export RA_LOG_LEVEL="INFO"
export RA_LOG_DIR="data/logs"
```

You can also create a local `.env` file for development/testing tools that load dotenv.

## Usage

### CLI

The project installs a `ra` command.

Search papers:

```bash
ra search --query "retrieval augmented generation benchmark survey" --limit 5 --source both
```

Get one paper by DOI/arXiv/Semantic Scholar ID:

```bash
ra get --id "10.5555/3295222.3295349"
```

Run evaluation harness:

```bash
ra eval --dataset tests/eval/dataset.json --output results/
```

Without an installed console script, use:

```bash
python -m ra.cli search --query "transformer memory mechanisms" --limit 3 --source semantic
```

### REST API

Run API locally:

```bash
python -m uvicorn ra.api.app:app --host 0.0.0.0 --port 8000
```

Interactive docs:
- Swagger UI: `http://localhost:8000/docs`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

#### `GET /health`

```bash
curl -s http://localhost:8000/health
```

#### `POST /search`

```bash
curl -s http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "transformer architecture for long-context summarization",
    "limit": 5,
    "source": "both"
  }'
```

#### `POST /retrieve`

```bash
curl -s http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"identifier":"1706.03762"}'
```

#### `POST /search/batch`

```bash
curl -s http://localhost:8000/search/batch \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {"query":"retrieval augmented generation benchmark survey","limit":5,"source":"semantic_scholar"},
      {"query":"long-context transformer memory mechanisms","limit":5,"source":"both"}
    ],
    "max_concurrency": 4
  }'
```

#### `POST /retrieve/batch`

```bash
curl -s http://localhost:8000/retrieve/batch \
  -H "Content-Type: application/json" \
  -d '{
    "requests":[
      {"identifier":"10.5555/3295222.3295349"},
      {"identifier":"1706.03762"}
    ],
    "max_concurrency": 8
  }'
```

#### `POST /answer`

Requires `OPENAI_API_KEY`.

```bash
curl -s http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"query":"What are the main limitations of retrieval-augmented generation?"}'
```

## Development Setup

Install editable package plus dev tools:

```bash
pip install -e ".[dev]"
```

Recommended checks:

```bash
ruff check src tests
mypy src
```

## Testing

Unit tests (required pre-commit command):

```bash
.venv/bin/python -m pytest tests/ -q --ignore=tests/integration
```

All tests (integration requires external APIs/network and valid credentials):

```bash
source ~/.zshrc && .venv/bin/python -m pytest tests/ -v
```

Integration-only run:

```bash
source ~/.zshrc && .venv/bin/python -m pytest tests/integration/test_agent_e2e.py -v
```

## Docker Deployment

Build and run with Docker Compose:

```bash
docker compose up --build -d
```

Service defaults:
- API: `http://localhost:8000`
- Container name: `ra-api`
- Data volume mount: `./data:/app/data`

Stop services:

```bash
docker compose down
```

Optional Chroma sidecar profile:

```bash
docker compose --profile chroma up -d
```

## Project Structure

```text
academic-research-assistant/
├── src/ra/
│   ├── agents/            # Research agent (LangChain tool-calling)
│   ├── api/               # FastAPI app, models, exception handling
│   ├── citation/          # Citation formatting and claim support utilities
│   ├── parsing/           # PDF parsing pipeline
│   ├── retrieval/         # Semantic Scholar, arXiv, unified retrieval, Chroma cache
│   ├── tools/             # Tool wrappers used by agent
│   └── utils/             # Config, logging, security, rate limiting
├── tests/
│   ├── unit/              # Unit tests
│   ├── integration/       # Network-backed integration tests
│   └── eval/              # Evaluation harness + dataset tooling
├── docs/                  # ADRs, diagrams, evaluation artifacts
├── data/                  # Runtime logs/cache/artifacts
├── Dockerfile
├── docker-compose.yml
├── CLAUDE.md
├── TODO.md
└── pyproject.toml
```
