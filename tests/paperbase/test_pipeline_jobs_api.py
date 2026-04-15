from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from paperbase.db.bootstrap import initialize_database
from paperbase.db.models import PaperFile
from paperbase.db.repositories import CollectionRepository, PaperRepository
from paperbase.db.session import make_session_factory
from services.paperbase_api.app import create_app


def test_local_library_ingest_api_enqueues_background_job(tmp_path: Path) -> None:
    source_dir = tmp_path / "library"
    source_dir.mkdir()
    (source_dir / "paper-one.pdf").write_bytes(b"%PDF-1.4\n%stub pdf\n")

    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")
    client = TestClient(create_app(session_factory=session_factory))

    response = client.post(
        "/api/v1/ingest/local-library",
        json={
            "source_dir": str(source_dir),
            "collection_title": "Sample Library",
        },
    )

    assert response.status_code == 202
    payload = response.json()["data"]
    assert payload["job_type"] == "local_library_ingest"
    assert payload["status"] == "pending"
    assert payload["payload"]["source_dir"] == str(source_dir)
    assert payload["payload"]["collection_title"] == "Sample Library"


def test_collection_parse_api_enqueues_background_job(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%stub pdf\n")

    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")

    with session_factory() as session:
        paper = PaperRepository(session).upsert(
            provider="local_filesystem",
            external_id=str(pdf_path),
            canonical_title="Sample Paper",
        )
        session.add(
            PaperFile(
                paper_id=paper.id,
                storage_uri=pdf_path.resolve().as_uri(),
                file_kind="pdf",
                mime_type="application/pdf",
                parser_status="pending",
            )
        )
        collection = CollectionRepository(session).create(
            owner_id="local-user",
            title="SamplePapers",
            description="Curated local corpus.",
        )
        CollectionRepository(session).add_paper(collection_id=collection.id, paper_id=paper.id)
        session.commit()
        collection_id = collection.id

    client = TestClient(create_app(session_factory=session_factory))
    response = client.post(
        f"/api/v1/collections/{collection_id}/parse",
        json={"limit": 1},
    )

    assert response.status_code == 202
    payload = response.json()["data"]
    assert payload["job_type"] == "collection_parse"
    assert payload["status"] == "pending"
    assert payload["payload"]["collection_id"] == collection_id
    assert payload["payload"]["limit"] == 1
