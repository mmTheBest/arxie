---
name: paperbase-postgres
description: Use when inspecting or operating the canonical Paperbase relational store, checking imported papers, collection membership, parsed sections, extraction runs, or annotations, and you need safe query patterns that match the Arxie schema.
---

# Paperbase Postgres

## Overview

Paperbase uses a canonical relational schema for papers, files, parses, extractions,
collections, and annotations. Local-first development currently defaults to SQLite,
but the schema and operator workflow should stay compatible with a future Postgres
deployment.

The safe rule is simple:

- read from the relational store first when validating corpus state
- prefer repository/model code for writes
- treat ad hoc SQL updates as exceptional maintenance work

## When To Use

- verifying whether a paper was imported into Paperbase
- checking which papers belong to a collection
- confirming parse output exists before extraction
- inspecting extraction runs, annotations, or evidence spans
- debugging mismatches between Arxie behavior and stored corpus state

Do not use this skill for search/read-model debugging. Use `paperbase-elasticsearch`
for that surface.

## Current State

- local default DB URL: `sqlite:///data/paperbase.db`
- migration entrypoint: `alembic upgrade head`
- canonical tables worth checking first:
  - `papers`
  - `paper_files`
  - `sections`
  - `chunks`
  - `extraction_runs`
  - `collections`
  - `collection_papers`
  - `annotations`

## Quick Reference

Initialize or migrate the schema:

```bash
PAPERBASE_DATABASE_URL=sqlite:///data/paperbase.db .venv/bin/python -m alembic upgrade head
```

Inspect collection membership:

```sql
SELECT c.title, cp.paper_id, p.canonical_title
FROM collections c
JOIN collection_papers cp ON cp.collection_id = c.id
JOIN papers p ON p.id = cp.paper_id
WHERE c.id = :collection_id
ORDER BY cp.position ASC, cp.created_at ASC;
```

Inspect parse state for one paper:

```sql
SELECT pf.storage_uri, pf.parser_status, COUNT(s.id) AS section_count, COUNT(ch.id) AS chunk_count
FROM paper_files pf
LEFT JOIN sections s ON s.paper_id = pf.paper_id
LEFT JOIN chunks ch ON ch.paper_id = pf.paper_id
WHERE pf.paper_id = :paper_id
GROUP BY pf.storage_uri, pf.parser_status;
```

Inspect extraction runs:

```sql
SELECT paper_id, model_name, prompt_version, schema_version, status, created_at
FROM extraction_runs
ORDER BY created_at DESC
LIMIT 20;
```

## Recommended Workflow

1. Confirm the DB URL you are about to inspect.
2. Prefer read-only queries first.
3. Verify corpus identity through `collections` and `collection_papers`.
4. Verify file/parse state through `paper_files`, `sections`, and `chunks`.
5. Verify extraction state through `extraction_runs` plus the derived entity tables.
6. Only then investigate assistant/API behavior.

## Common Mistakes

- Assuming Paperbase is empty because Arxie search fell back to live providers.
  The gateway intentionally fails open. Check the DB directly.
- Updating rows manually when repository code already exists.
  Use repository helpers unless this is explicit maintenance.
- Treating SQLite-specific behavior as the final production contract.
  Keep queries portable where possible.
