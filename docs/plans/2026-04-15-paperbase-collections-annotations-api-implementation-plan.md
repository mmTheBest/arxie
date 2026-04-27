# 2026-04-15 Paperbase Collections And Annotations API

## Goal

Close the remaining product gap between “collections/annotations exist in the DB”
and “users can actually manage them.” This slice adds first-class Paperbase API
support for curated collections and manual annotations.

## Implemented

- Added collection endpoints:
  - `GET /api/v1/collections`
  - `POST /api/v1/collections`
  - `GET /api/v1/collections/{collection_id}`
  - `POST /api/v1/collections/{collection_id}/papers`
  - `GET /api/v1/collections/{collection_id}/papers`
- Added annotation endpoints:
  - `POST /api/v1/annotations`
  - `GET /api/v1/annotations`
- Added the supporting response/request contracts in
  `services/paperbase_api/models.py`
- Extended the repositories with collection listing and id lookup helpers needed by
  the API

## Verification

- `pytest tests/paperbase/test_collections_api.py -q`
- `pytest tests/paperbase/test_collections_api.py tests/paperbase/test_paperbase_api.py -q`

## Result

Collections and manual annotations are now a real Paperbase product surface rather
than internal schema only. This moves the platform closer to the PRD requirement
that users maintain custom field-specific databases and annotate papers directly.
