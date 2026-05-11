# Install Arxie

This branch contains the clean single-user runtime release. It excludes development history, tests, planning documents, and internal agent workflow files.

## Requirements

- Python 3.10+
- Docker with Compose
- An OpenAI API key

## Install From Source

```bash
git clone https://github.com/mmTheBest/arxie.git
cd arxie
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`.

## Start The Local Stack

```bash
docker compose -f infra/docker-compose.paperbase.yml up -d postgres minio redis
docker compose -f infra/docker-compose.paperbase.yml run --rm paperbase-migrate
docker compose -f infra/docker-compose.paperbase.yml up -d paperbase-api paperbase-worker
```

Open:

- `http://localhost:8080/`
- `http://localhost:8080/app`

## Shortcut Launcher

```bash
arxie-local run
```

Use `arxie-local run --with-search` if you want Elasticsearch-backed search.
