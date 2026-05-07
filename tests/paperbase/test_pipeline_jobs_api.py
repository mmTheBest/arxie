from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from paperbase.db.bootstrap import initialize_database
from paperbase.db.models import PaperFile
from paperbase.db.repositories import CollectionRepository, PaperRepository
from paperbase.db.session import make_session_factory
from services.paperbase_api.app import create_app


class FakeJobDispatcher:
    def __init__(self) -> None:
        self.job_ids: list[str] = []

    def dispatch(self, job_id: str) -> None:
        self.job_ids.append(job_id)


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


def test_local_library_upload_api_stages_uploaded_pdfs_and_enqueues_background_job(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PAPERBASE_UPLOAD_STAGING_DIR", str(tmp_path / "uploads"))
    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")
    client = TestClient(create_app(session_factory=session_factory))

    response = client.post(
        "/api/v1/ingest/local-library-upload",
        data={
            "collection_title": "Uploaded Library",
            "collection_description": "Uploaded from browser.",
        },
        files=[
            ("files", ("nested/paper-one.pdf", b"%PDF-1.4\n%stub pdf\n", "application/pdf")),
            ("files", ("paper-two.pdf", b"%PDF-1.4\n%stub pdf\n", "application/pdf")),
        ],
    )

    assert response.status_code == 202
    payload = response.json()["data"]
    assert payload["job_type"] == "local_library_ingest"
    assert payload["status"] == "pending"
    assert payload["payload"]["collection_title"] == "Uploaded Library"
    staged_dir = Path(payload["payload"]["source_dir"])
    assert staged_dir.exists()
    assert (staged_dir / "nested" / "paper-one.pdf").exists()
    assert (staged_dir / "paper-two.pdf").exists()


def test_provider_identifier_ingest_api_enqueues_background_job(tmp_path: Path) -> None:
    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")
    client = TestClient(create_app(session_factory=session_factory))

    response = client.post(
        "/api/v1/ingest/providers",
        json={
            "collection_title": "scRegNet provider import",
            "identifiers": [
                {"kind": "doi", "value": "10.1038/example"},
                {"kind": "arxiv", "value": "2503.01682v1"},
            ],
        },
    )

    assert response.status_code == 202
    payload = response.json()["data"]
    assert payload["job_type"] == "provider_identifier_ingest"
    assert payload["status"] == "pending"
    assert payload["payload"]["collection_title"] == "scRegNet provider import"
    assert payload["payload"]["identifiers"] == [
        {"kind": "doi", "value": "10.1038/example"},
        {"kind": "arxiv", "value": "2503.01682v1"},
    ]


def test_provider_metadata_refresh_api_enqueues_background_job(tmp_path: Path) -> None:
    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")

    with session_factory() as session:
        paper = PaperRepository(session).upsert(
            provider="crossref",
            external_id="10.1038/example",
            canonical_title="Original Title",
            doi="10.1038/example",
        )
        paper_id = paper.id

    client = TestClient(create_app(session_factory=session_factory))
    response = client.post(
        "/api/v1/ingest/refresh-metadata",
        json={"paper_ids": [paper_id]},
    )

    assert response.status_code == 202
    payload = response.json()["data"]
    assert payload["job_type"] == "paper_metadata_refresh"
    assert payload["status"] == "pending"
    assert payload["payload"]["paper_ids"] == [paper_id]


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


def test_collection_parse_api_accepts_selected_paper_ids(tmp_path: Path) -> None:
    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")

    with session_factory() as session:
        paper_one = PaperRepository(session).upsert(
            provider="local_filesystem",
            external_id="/tmp/paper-one.pdf",
            canonical_title="Paper One",
        )
        paper_two = PaperRepository(session).upsert(
            provider="local_filesystem",
            external_id="/tmp/paper-two.pdf",
            canonical_title="Paper Two",
        )
        collection = CollectionRepository(session).create(
            owner_id="local-user",
            title="SamplePapers",
            description="Curated local corpus.",
        )
        CollectionRepository(session).add_paper(collection_id=collection.id, paper_id=paper_one.id)
        CollectionRepository(session).add_paper(collection_id=collection.id, paper_id=paper_two.id)
        session.commit()
        collection_id = collection.id
        paper_two_id = paper_two.id

    client = TestClient(create_app(session_factory=session_factory))
    response = client.post(
        f"/api/v1/collections/{collection_id}/parse",
        json={"paper_ids": [paper_two_id]},
    )

    assert response.status_code == 202
    payload = response.json()["data"]
    assert payload["job_type"] == "collection_parse"
    assert payload["payload"]["paper_ids"] == [paper_two_id]


def test_search_reindex_dispatches_job_when_runtime_dispatcher_is_configured(tmp_path: Path) -> None:
    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")
    dispatcher = FakeJobDispatcher()
    client = TestClient(create_app(session_factory=session_factory, job_dispatcher=dispatcher))

    response = client.post("/api/v1/search/reindex")

    assert response.status_code == 202
    payload = response.json()["data"]
    assert dispatcher.job_ids == [payload["id"]]
