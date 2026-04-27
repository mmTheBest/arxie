# Paperbase Extraction Milestone

Date: 2026-04-15
Branch: `feature/paperbase-foundation`

## Goal

Add the first structured extraction layer on top of the local-first parse pipeline so Paperbase can persist canonical comparison entities instead of stopping at sections and chunks.

## Scope

- add canonical `glossary_terms` storage to the Paperbase schema
- add a typed extraction bundle contract for datasets, methods, metrics, results, findings, glossary terms, and engineering tricks
- add an OpenAI-backed structured extraction client and prompt builder
- add a persistence pipeline that:
  - reads parsed sections for one paper
  - calls a schema-constrained extraction client
  - writes extraction runs plus structured entities and evidence spans
- add a collection runner so curated local corpora can execute field-specific extraction end-to-end
- keep the milestone local-first and single-user, but avoid schema choices that block later multi-user or larger-corpus expansion

## Non-Goals

- API endpoints
- Elasticsearch indexing
- distributed worker orchestration
- production model prompts

## Verification

- `pytest tests/paperbase/test_schema_contract.py -q`
- `pytest tests/paperbase/test_extraction_pipeline.py -q`
- `pytest tests/paperbase/test_extraction_client.py -q`
- `pytest tests/paperbase/test_collection_extraction_runner.py -q`
- `pytest tests/paperbase/test_extraction_contracts.py -q`

## Notes

This milestone intentionally uses parsed section text as the extraction input. That keeps the storage contract stable while the later worker service, figure/table extraction, and search indexing layers are still under construction.

Live smoke evidence on `SamplePapers` should stay lightweight:

- run one-paper extraction first with the OpenAI client
- confirm the extraction run completes and persists canonical entities
- scale to the rest of the collection only after the search/API/gateway layers are ready
