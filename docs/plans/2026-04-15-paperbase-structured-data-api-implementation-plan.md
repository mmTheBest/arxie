# 2026-04-15 Paperbase Structured Data API

## Goal

Close the gap between “Paperbase stores extracted structured entities” and “a
user can browse those extracted entities for a paper through the API.”

## Implemented

- Added `GET /api/v1/papers/{paper_id}/structured-data`
- Added response contracts for:
  - datasets
  - methods
  - metrics
  - result rows
  - glossary terms
  - findings
  - engineering tricks
  - extraction runs
  - evidence spans
- Extended the paper route group so a Paperbase client can inspect extracted
  artifacts for one paper without querying the ORM directly

## Verification

- `pytest tests/paperbase/test_paperbase_api.py -q`
- `pytest tests/paperbase -q`
- `make test-clean-baseline`

## Result

Paperbase now exposes a paper-level structured inspection surface, which makes
the local-first database usable after extraction runs. This moves the platform
closer to the PRD requirement that users open a paper and inspect structured
research artifacts instead of only searching metadata or triggering jobs.
