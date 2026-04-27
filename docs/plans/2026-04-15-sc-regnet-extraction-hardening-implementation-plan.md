# scRegNet Extraction Hardening Implementation Note

## Goal

Make the local `SamplePapers` scRegNet database operational end-to-end by
removing the remaining extraction failure mode from real OpenAI
function-calling outputs and by improving collection summary readability.

## What Changed

### 1. Hardened structured extraction bundle validation

The remaining failed papers showed a consistent failure shape: OpenAI
function-calling sometimes returned valid top-level bundles but inserted stray
scalar fragments inside list fields such as `metrics`.

The extraction contract now filters invalid scalar entries out of:

- `datasets`
- `methods`
- `metrics`
- `results`
- `findings`
- `glossary_terms`
- `engineering_tricks`

This is intentionally narrow. It does not try to coerce arbitrary malformed
objects into valid entities. It only removes impossible scalar fragments so that
valid extracted entities in the same payload can still persist.

### 2. Re-ran the real scRegNet collection

After the contract hardening, the three previously failing papers were rerun
against the real `sc_regnet` preset and the local OpenAI-backed extraction path.

Current status of `SamplePapers`:

- papers in collection: `11`
- latest completed extraction runs: `11`
- latest failed extraction runs: `0`

### 3. Normalized collection summary metric aliases

The collection structured summary previously returned raw metric rows, which
produced duplicate-looking browse output such as both `AUROC` and `Area Under
the Receiver Operating Characteristic Curve`.

The summary route now canonicalizes common metric aliases and collapses obvious
duplicates in the response. This keeps the collection browse surface usable as a
field database before the richer comparison/indexing layer is finished.

## Verification

Focused tests:

- `pytest tests/paperbase/test_extraction_schema_aliases.py -q`
- `pytest tests/paperbase/test_collection_summary_api.py -q`

Broader checks:

- `pytest tests/paperbase -q`
- `make test-clean-baseline`

Operational checks:

- reran the 3 formerly failing `SamplePapers` papers against the real
  OpenAI-backed extraction client
- queried `GET /api/v1/collections/{collection_id}/structured-summary`
  against the live local DB and confirmed:
  - `paper_count=11`
  - `extracted_paper_count=11`
  - normalized metric shortlist without obvious alias duplicates

## Impact On The April 14 Design

This closes a meaningful local-first gap in the Paperbase PRD:

- a curated field collection can now be imported, parsed, extracted, and browsed
  without partial extraction failure blocking the corpus
- the collection summary behaves more like a database browse surface and less
  like a raw dump of extracted rows

The major remaining gaps are still:

- live indexing and Elasticsearch-backed retrieval
- richer comparison/read-model endpoints
- broader canonical metadata coverage such as authors and tags
- workerized background execution and larger external corpus sync
