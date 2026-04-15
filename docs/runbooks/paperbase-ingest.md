# Paperbase Ingest Runbook

## Purpose

Use this runbook to turn a local folder of PDFs into a Paperbase collection that
Arxie can search, inspect, and extract from.

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

## 4. Run Structured Extraction

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

## 5. Verify The Collection

Run the Paperbase suite:

```bash
.venv/bin/python -m pytest tests/paperbase -q
```

Check the DB directly if results look wrong:

- imported papers in `papers`
- file records in `paper_files`
- parsed sections in `sections`
- extraction metadata in `extraction_runs`

## Troubleshooting

- No sections after parse:
  verify `paper_files.storage_uri` points to a real local file.
- Collection extraction skips papers:
  check whether parse output exists and whether the paper has a registered PDF.
- Arxie still falls back to live providers:
  confirm the collection papers were imported into the same DB URL the runtime is using.
