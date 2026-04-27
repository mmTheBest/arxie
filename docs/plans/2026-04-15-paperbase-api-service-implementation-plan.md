# Paperbase API Service Milestone

Date: 2026-04-15
Branch: `feature/paperbase-foundation`

## Goal

Expose the first internal Paperbase API service so other product layers can query the local-first corpus without reaching directly into repositories and SQLAlchemy models.

## Scope

- add a FastAPI app for Paperbase
- add route groups for:
  - paper search
  - paper fetch
  - fulltext fetch
  - figures fetch
  - result comparison
- keep the service independent from `src/ra/api/app.py`
- validate user input and return structured error payloads

## Non-Goals

- authentication
- rate limiting middleware
- gateway integration into Arxie
- live Elasticsearch-backed search execution

## Verification

- `pytest tests/paperbase/test_paperbase_api.py -q`
- `git ls-files -z 'tests/paperbase/*.py' | xargs -0 .venv/bin/python -m pytest -q`

## Notes

This service is the Paperbase-facing contract layer. It should stay thin and DB-backed for now, so the next milestone can introduce a Paperbase gateway inside `src/ra/retrieval/` without rewriting the storage layer.
