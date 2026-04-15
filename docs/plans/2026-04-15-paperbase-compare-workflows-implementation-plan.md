# Paperbase Compare Workflows Implementation Note

## Goal

Close the biggest remaining user-facing Paperbase gap after extraction
stabilization: comparison should be a database workflow, not just a free-form
agent behavior.

## What Changed

### 1. Evidence-backed result comparison

`POST /api/v1/compare/results` now supports:

- collection-scoped filtering
- metric alias normalization for inputs and outputs
- optional evidence-span expansion for each result row

This makes result comparison closer to the PRD requirement that analysts should
be able to compare reported values and inspect the evidence behind each row.

### 2. Method-level comparison

`POST /api/v1/compare/methods` now summarizes a comparison slice by method:

- `paper_count`
- `result_count`
- datasets represented in the slice
- metrics represented in the slice
- best observed result row for the method

This provides a real database-backed answer to “compare methods across paper
sets” using the extracted result rows rather than ad hoc prompting.

### 3. Engineering-trick comparison

`POST /api/v1/compare/engineering-tricks` now summarizes recurring engineering
tricks across a collection and can filter by method family.

This is intentionally simple: engineering tricks are still paper-level entities,
so the current route groups them by normalized title and returns the papers that
contribute each trick.

## Verification

Focused checks:

- `pytest tests/paperbase/test_compare_api.py -q`
- `pytest tests/paperbase/test_paperbase_api.py tests/paperbase/test_collection_summary_api.py tests/paperbase/test_compare_api.py -q`

Broader checks:

- `pytest tests/paperbase -q`
- `make test-clean-baseline`

## Impact On The April 14 Design

This moves the project materially closer to the PRD:

- the database can now compare results with evidence
- the database can compare methods across a curated paper set
- the database can summarize engineering tricks across a topic or method family

The biggest remaining platform gaps are still:

- live Elasticsearch-backed indexing and retrieval
- figure/table extraction and comparison slices
- broader canonical metadata such as authors and paper-level tags
- workerized background jobs and external-corpus sync
