---
name: paperbase-elasticsearch
description: Use when working on Paperbase search/read models, Elasticsearch mappings, document builders, or hybrid query payloads, and you need the current safe workflow for a search layer that is partially scaffolded but not yet fully operational.
---

# Paperbase Elasticsearch

## Overview

Paperbase search is split into three pieces:

- index templates in `src/paperbase/search/index_templates.py`
- document builders in `src/paperbase/search/indexer.py`
- query composition in `src/paperbase/search/query_builder.py`

Important current-state caveat: the read-model contracts exist, but the live indexing
service is still scaffolded rather than fully deployed. This skill is for changing
and validating the search contract without pretending the background indexer already
exists.

## When To Use

- changing Paperbase search mappings
- adding fields to paper/chunk/figure documents
- validating hybrid text + vector query payloads
- debugging search-contract tests
- planning future Elasticsearch service work

Do not use this skill for corpus import, parse, or extraction flows. Use
`paperbase-corpus-ops` instead.

## Quick Reference

Key files:

- `src/paperbase/search/index_templates.py`
- `src/paperbase/search/indexer.py`
- `src/paperbase/search/query_builder.py`
- `tests/paperbase/test_search_read_models.py`

Fast verification:

```bash
.venv/bin/python -m pytest tests/paperbase/test_search_read_models.py -q
```

## Safe Workflow

1. Change the template and document builder together.
2. Update query builder only after the document field contract is clear.
3. Run the read-model tests immediately.
4. If Compose-backed Elasticsearch is live in your environment, validate the template
   shape there after the unit tests pass.
5. Do not claim “search is deployed” unless the actual indexing path and runtime
   query path are both verified.

## Design Rules

- `paper` index: metadata-first retrieval surface
- `chunk` index: section-aware evidence retrieval
- `figure` index: caption/visual artifact lookup
- embeddings are optional payload fields, not proof that vector search is live
- mapping changes must preserve the current query builder contract or update tests

## Common Mistakes

- Treating document builders as if they already write to Elasticsearch.
  They currently build payloads; they do not perform cluster writes.
- Adding fields to mappings without adding them to document builders.
- Claiming hybrid search is finished because `knn` is present in query JSON.
  A query builder is not a deployed search service.
