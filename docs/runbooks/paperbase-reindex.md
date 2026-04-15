# Paperbase Reindex Runbook

## Purpose

Use this runbook to rebuild or validate the Paperbase search/read-model contract.

Important current-state note: Paperbase search mappings and document builders exist,
but the live indexing service is still scaffolded. Today this runbook is mainly for
regenerating and validating index payload shapes so collaborators can evolve the
search layer safely.

## 1. Validate Read-Model Contracts

```bash
.venv/bin/python -m pytest tests/paperbase/test_search_read_models.py -q
```

This verifies:

- paper/chunk/figure index templates
- document builders
- hybrid query builder payloads

## 2. Rebuild Documents From The Canonical Store

Use a short Python task to regenerate the derived paper and chunk payloads from the
current DB.

```bash
.venv/bin/python - <<'PY'
from sqlalchemy import select

from paperbase.db.models import Chunk, Paper
from paperbase.db.session import make_session_factory
from paperbase.search.indexer import build_chunk_document, build_paper_document

session_factory = make_session_factory("sqlite:///data/paperbase.db")

with session_factory() as session:
    papers = session.execute(select(Paper)).scalars().all()
    chunks = session.execute(select(Chunk)).scalars().all()

paper_docs = [
    build_paper_document(
        paper_id=paper.id,
        title=paper.canonical_title,
        abstract=paper.abstract,
        year=paper.publication_year,
        venue=paper.venue,
    )
    for paper in papers
]
chunk_docs = [
    build_chunk_document(
        chunk_id=chunk.id,
        paper_id=chunk.paper_id,
        title="",
        section_title=None,
        text=chunk.text,
    )
    for chunk in chunks
]

print({"paper_docs": len(paper_docs), "chunk_docs": len(chunk_docs)})
PY
```

## 3. Validate Query Payloads

```bash
.venv/bin/python - <<'PY'
from paperbase.search.query_builder import build_search_query

query = build_search_query(
    query_text="gene regulatory network benchmark",
    filters={"year_gte": 2024, "tags": ["scRegNet"]},
    embedding_vector=[0.1, 0.2, 0.3],
)
print(query)
PY
```

Confirm the payload matches the current mappings before wiring it into any runtime
search path.

## 4. When Elasticsearch Is Live

Once the actual indexing service exists, extend this runbook with the cluster-write
steps. Until then, do not claim “reindex complete” unless both of these are true:

- the derived document payloads were rebuilt from the canonical DB
- the read-model tests passed fresh

## Troubleshooting

- Mapping changes break tests:
  update the document builders and query builder together.
- Query payload refers to missing fields:
  compare `query_builder.py` against `index_templates.py`.
- Search behavior diverges from Arxie expectations:
  verify whether the problem is in the canonical DB, the document builder, or the
  future runtime indexer path before changing mappings.
