# Arxie Reindex Runbook

## Purpose

Use this runbook to rebuild or validate the Arxie search/read-model contract.

Important current-state note: reindexing is now worker-backed. The API should
enqueue a search job, and the worker should execute the actual index rebuild.
This runbook keeps both the contract validation path and the local worker
execution path explicit.

## 1. Validate Read-Model Contracts

```bash
.venv/bin/python -m pytest tests/paperbase/test_search_read_models.py -q
```

This verifies:

- paper/chunk/figure index templates
- document builders
- hybrid query builder payloads

## 2. Enqueue A Reindex Job

```bash
.venv/bin/python - <<'PY'
from fastapi.testclient import TestClient

from paperbase.db.session import make_session_factory
from services.paperbase_api.app import create_app

session_factory = make_session_factory("sqlite:///data/paperbase.db")
client = TestClient(create_app(session_factory=session_factory))
response = client.post("/api/v1/search/reindex")
print(response.json())
PY
```

Record the returned `job_id`.

## 3. Process The Job With The Worker Runtime

```bash
.venv/bin/python - <<'PY'
from paperbase.config import load_paperbase_config
from paperbase.db.session import make_session_factory
from paperbase.search.runtime import ElasticsearchSearchBackend
from services.paperbase_worker.runtime import PaperbaseWorker

config = load_paperbase_config()
session_factory = make_session_factory("sqlite:///data/paperbase.db")
worker = PaperbaseWorker(
    session_factory=session_factory,
    search_backend=ElasticsearchSearchBackend(base_url=config.elasticsearch_url),
)
print(worker.process_next_job())
PY
```

## 4. Rebuild Documents From The Canonical Store

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

## 5. Validate Query Payloads

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

## 6. When Elasticsearch Is Live

Do not claim “reindex complete” unless all of these are true:

- the API enqueued the job successfully
- the worker processed the job to `completed`
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
