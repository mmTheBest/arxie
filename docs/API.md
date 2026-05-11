# API Overview

The browser app uses the Paperbase API served by `paperbase-api` on port `8080`.

## Runtime Health

- `GET /health`
- `GET /livez`
- `GET /readyz`

## Main Product Areas

- Collections and papers: import, membership, readiness, labels, and paper detail.
- Search: collection-aware paper, chunk, figure, table, and artifact search.
- Extraction: parse and structured extraction jobs.
- Compare: methods, results, engineering tricks, figures, and tables.
- Studies: saved study context, explicit user sources, research threads, and artifacts.
- Jobs: background job status and recovery information.

OpenAPI is available from the running service:

- `http://localhost:8080/docs`
- `http://localhost:8080/openapi.json`
