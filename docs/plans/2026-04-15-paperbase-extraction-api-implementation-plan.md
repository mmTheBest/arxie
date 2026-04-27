# 2026-04-15 Paperbase Extraction API

## Goal

Close the product gap between “collection extraction exists as internal runner
code” and “users can trigger field-specific extraction through the Paperbase API.”

## Implemented

- Added extraction profile endpoints:
  - `GET /api/v1/extraction-profiles`
  - `POST /api/v1/extraction-profiles`
- Added collection extraction endpoint:
  - `POST /api/v1/collections/{collection_id}/extract`
- Added the supporting request and response contracts in
  `services/paperbase_api/models.py`
- Extended the extraction profile repository with id lookup and filtered listing
- Wired `services/paperbase_api.app.create_app()` to accept an injectable
  extraction client factory so local tests and future workers can drive the same
  route contract

## Verification

- `pytest tests/paperbase/test_extraction_api.py -q`
- `pytest tests/paperbase/test_extraction_api.py tests/paperbase/test_collection_extraction_runner.py tests/paperbase/test_extraction_pipeline.py tests/paperbase/test_paperbase_api.py -q`
- `pytest tests/paperbase -q`
- `make test-clean-baseline`

## Result

Paperbase now has a first-class API path for defining field-specific extraction
profiles and running those schemas against curated collections. This moves the
platform closer to the PRD requirement that users maintain customized paper
databases and execute domain-specific analysis over them.
