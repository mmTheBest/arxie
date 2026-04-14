# System Overview

## Product Shape

Arxie is a research database plus an agent that operates on that database.

The system has two top-level layers:

- **Paperbase** — persistent corpus storage, extraction, indexing, and comparison
- **Arxie Assistant** — search, chat, synthesis, literature review, and proposal workflows

## Initial Deployment Mode

V1 is single-user and local-first.

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

