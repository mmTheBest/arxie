# Paperbase Search Read Models Milestone

Date: 2026-04-15
Branch: `feature/paperbase-foundation`

## Goal

Add the first searchable read-model contracts for Paperbase so extracted and parsed corpus data can be indexed consistently before the API and Arxie gateway layers are built.

## Scope

- add index templates for papers, chunks, and figures
- add deterministic document builders for those index types
- add a query builder that composes text, filter, and vector clauses

## Non-Goals

- live Elasticsearch writes
- API endpoints
- ranking evaluation
- comparison routes

## Verification

- `pytest tests/paperbase/test_search_read_models.py -q`
- `pytest tests/paperbase/test_search_indexing.py -q`

## Notes

This milestone is intentionally read-model-only. It provides the contract that later worker indexing and API search endpoints can share without prematurely coupling Paperbase to one Elasticsearch client implementation.
