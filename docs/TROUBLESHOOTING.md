# Troubleshooting

## The app does not open

Check service health:

```bash
curl http://localhost:8080/livez
curl http://localhost:8080/readyz
```

If services are not running:

```bash
docker compose -f infra/docker-compose.paperbase.yml ps
```

## Jobs stay running

Open Library and check the processing log. A job may be stale if the worker is not running.

```bash
docker compose -f infra/docker-compose.paperbase.yml up -d paperbase-worker
```

## Search is limited

The default launcher can run without Elasticsearch. In that mode, Arxie falls back to local search behavior. Start with backend search when you need stronger collection search:

```bash
arxie-local run --with-search
```

## Extraction fails

Confirm that:

- papers are parsed first
- `OPENAI_API_KEY` is set
- the worker is running
- the object store is reachable

Retry failed papers from Library after fixing the underlying issue.
