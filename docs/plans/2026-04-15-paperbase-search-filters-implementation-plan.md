# 2026-04-15 Paperbase Search Filters

## Goal

Close the gap between “Paperbase can search by title/abstract text” and “users
can filter a persistent corpus or curated collection by extracted structured
entities.”

## Implemented

- Extended `GET /api/v1/search/papers` to support:
  - `collection_id`
  - `year_gte`
  - `year_lte`
  - `venue`
  - `dataset`
  - `method`
  - `metric`
  - `extraction_state`
- Made `q` optional as long as at least one supported filter is present
- Added explicit `400 invalid_input` handling for empty search requests and
  unsupported extraction-state values

## Verification

- `pytest tests/paperbase/test_search_filters_api.py -q`
- `pytest tests/paperbase/test_search_filters_api.py tests/paperbase/test_paperbase_api.py -q`
- `pytest tests/paperbase -q`
- `make test-clean-baseline`

## Result

Paperbase can now browse a curated database through structured filters instead of
only free-text search. This moves the platform closer to the PRD requirement
that users search a persistent corpus by metadata and extracted entities within
their own field-specific collections.
