# Paperbase Ingest Runbook

## Purpose

Use this runbook to turn a local folder of PDFs into a Paperbase collection that
Arxie can search, inspect, extract from, and use as field-aware Study context.

For a normal single-user local workflow, prefer the browser workspace at `/app`
first: use **Upload PDF Folder**, then **Queue Parse**, then **Queue Extraction**.
Use the scripted steps below when you need operator control, debugging, or batch
automation.

## Canonical Local Test Corpus

When the user asks to open Arxie for local product testing without naming a
different corpus, use this source folder:

```text
/Users/mm/school/scRegNet/SamplePapers
```

This is the recurring scRegNet knowledge source for testing Library ingest,
parse, extraction, Study chat, and research-intelligence behavior. Treat it as a
source folder, not as parsed state. The parsed database may live elsewhere
depending on the active Paperbase DB URL or opened project, so verify the
runtime database before assuming the corpus is already processed.

Current maintained expectation, as of 2026-05-30:

- default local verification DB: `sqlite:///data/paperbase.db`
- imported PDFs: `12`
- parsed papers: `12`
- parsed sections: `50`
- persisted chunks: `1112`
- figures: `85`
- tables: `20`
- latest extraction status: `completed` for all `12` papers
- extracted structured evidence: `30` datasets, `31` methods, `24` metrics,
  `35` result rows, `3` findings, `0` limitations, and `41` research-design
  elements
- deterministic Study-style `benchmark_planning` smoke: completed with
  validation passing across source-library, structured-evidence,
  evidence-memory, pattern-memory, field-graph, and study-brief layers

Historical failed extraction rows can remain in the local database after failed
network or sandbox attempts. Verify latest extraction state per paper before
deciding whether the corpus needs to be reprocessed.

When this corpus is re-imported, re-parsed, re-extracted, or used as new
release/test evidence, update this runbook and
`docs/architecture/03-ingest-and-extraction.md` in the same change.

## Prerequisites

- work from the clean dev worktree
- `.venv` exists
- the Paperbase DB URL is set or you accept the default `sqlite:///data/paperbase.db`
- if you plan to run extraction, `OPENAI_API_KEY` must be available in the environment

## 1. Initialize The Database

```bash
PAPERBASE_DATABASE_URL=sqlite:///data/paperbase.db .venv/bin/python -m alembic upgrade head
```

## 2. Import A Local PDF Directory

```bash
.venv/bin/python - <<'PY'
from pathlib import Path

from paperbase.db.session import make_session_factory
from paperbase.ingest.local_library import import_local_pdf_directory

session_factory = make_session_factory("sqlite:///data/paperbase.db")
result = import_local_pdf_directory(
    source_dir=Path("/absolute/path/to/paper-folder"),
    session_factory=session_factory,
    collection_title="MyCollection",
    collection_description="Curated local corpus",
)
print(result)
PY
```

Record the `collection_id` from the DB if you will parse or extract later.

## 3. Parse Papers Into Sections And Chunks

```bash
.venv/bin/python - <<'PY'
from sqlalchemy import select

from paperbase.db.models import CollectionPaper
from paperbase.db.session import make_session_factory
from paperbase.parsing.pipeline import PaperParsePipeline

collection_id = "replace-with-collection-id"
session_factory = make_session_factory("sqlite:///data/paperbase.db")
pipeline = PaperParsePipeline(session_factory=session_factory)

with session_factory() as session:
    paper_ids = list(
        session.execute(
            select(CollectionPaper.paper_id).where(CollectionPaper.collection_id == collection_id)
        ).scalars()
    )

for paper_id in paper_ids:
    result = pipeline.parse_paper(paper_id)
    print(result)
PY
```

## 4. Enqueue Structured Extraction

```bash
.venv/bin/python - <<'PY'
from fastapi.testclient import TestClient

from paperbase.db.session import make_session_factory
from services.paperbase_api.app import create_app

collection_id = "replace-with-collection-id"
session_factory = make_session_factory("sqlite:///data/paperbase.db")
client = TestClient(create_app(session_factory=session_factory))

response = client.post(
    f"/api/v1/collections/{collection_id}/extract",
    json={
        "schema_payload": {
            "datasets": True,
            "methods": True,
            "metrics": True,
            "results": True,
            "engineering_tricks": True,
        },
        "prompt_version": "paperbase-v1",
        "schema_version": "paperbase-v1",
    },
)
print(response.json())
PY
```

Record the returned `job_id`.

## 5. Process The Extraction Job With The Worker

```bash
.venv/bin/python - <<'PY'
from paperbase.db.session import make_session_factory
from paperbase.extract.client import OpenAIExtractionClient
from services.paperbase_worker.runtime import PaperbaseWorker

session_factory = make_session_factory("sqlite:///data/paperbase.db")
worker = PaperbaseWorker(
    session_factory=session_factory,
    extraction_client_factory=lambda: OpenAIExtractionClient(model="gpt-4o-mini"),
)
print(worker.process_next_job())
PY
```

## 6. Run Structured Extraction Directly For Debugging

```bash
.venv/bin/python - <<'PY'
from paperbase.db.session import make_session_factory
from paperbase.extract.client import OpenAIExtractionClient
from paperbase.extract.runner import CollectionExtractionRunner

collection_id = "replace-with-collection-id"
schema_payload = {
    "datasets": True,
    "methods": True,
    "metrics": True,
    "results": True,
    "engineering_tricks": True,
}

session_factory = make_session_factory("sqlite:///data/paperbase.db")
runner = CollectionExtractionRunner(
    session_factory=session_factory,
    client=OpenAIExtractionClient(model="gpt-4o-mini"),
)
summary = runner.extract_collection(
    collection_id=collection_id,
    schema_payload=schema_payload,
    prompt_version="paperbase-v1",
    schema_version="paperbase-v1",
)
print(summary)
PY
```

Use the direct runner only when debugging extraction behavior. Normal product
execution should go through the queued API + worker path.

## 7. Verify The Collection

Run the Paperbase suite:

```bash
.venv/bin/python -m pytest tests/paperbase -q
```

Check the DB directly if results look wrong:

- imported papers in `papers`
- file records in `paper_files`
- parsed sections in `sections`
- extraction metadata in `extraction_runs`

For a product-level smoke test, open `/app`, create or select a Study linked to
the collection, add one explicit source if relevant, and ask for a benchmark or
experiment plan. A useful run should show:

- collection papers selected into the context pack
- structured evidence or readiness warnings
- evidence-memory, pattern-memory, and field-graph counts when extraction has
  completed
- validation status and evidence-reference counts in the artifact trace

## Troubleshooting

- No sections after parse:
  verify `paper_files.storage_uri` points to a canonical object-store URI in the
  self-hosted runtime, or to a deliberate local fallback URI in dev mode.
- Collection extraction skips papers:
  check whether parse output exists and whether the paper has a registered PDF.
- Arxie still falls back to live providers:
  confirm the collection papers were imported into the same DB URL the runtime is using.
- Browser upload is rejected:
  check `PAPERBASE_UPLOAD_MAX_FILE_COUNT`,
  `PAPERBASE_UPLOAD_MAX_SINGLE_FILE_BYTES`, and
  `PAPERBASE_UPLOAD_MAX_TOTAL_BYTES`; non-PDF multipart file parts still count
  toward configured upload limits even though only PDFs are staged for ingest.
