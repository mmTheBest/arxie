from __future__ import annotations

from fastapi.testclient import TestClient

from paperbase.db.bootstrap import initialize_database
from paperbase.db.models import Chunk, ExtractionRun, Figure, PaperFile, Section
from paperbase.db.repositories import PaperRepository
from paperbase.db.session import make_session_factory
from services.paperbase_api.app import create_app


class FakeSearchBackend:
    def __init__(self) -> None:
        self.index_calls: list[tuple[str, dict[str, object]]] = []
        self.bulk_calls: list[tuple[str, list[dict[str, object]]]] = []
        self.search_calls: list[tuple[str, dict[str, object], int]] = []

    def ensure_index(self, index_name: str, template: dict[str, object]) -> None:
        self.index_calls.append((index_name, template))

    def bulk_index(self, index_name: str, documents: list[dict[str, object]]) -> None:
        self.bulk_calls.append((index_name, documents))

    def search(self, index_name: str, query: dict[str, object], size: int) -> list[dict[str, object]]:
        self.search_calls.append((index_name, query, size))
        return []


def test_search_status_reports_backend_configuration(tmp_path) -> None:
    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")
    backend = FakeSearchBackend()

    configured_client = TestClient(create_app(session_factory=session_factory, search_backend=backend))
    unconfigured_client = TestClient(create_app(session_factory=session_factory))

    configured_response = configured_client.get("/api/v1/search/status")
    unconfigured_response = unconfigured_client.get("/api/v1/search/status")

    assert configured_response.status_code == 200
    assert configured_response.json()["data"] == {
        "backend_configured": True,
        "backend_type": "FakeSearchBackend",
    }
    assert unconfigured_response.status_code == 200
    assert unconfigured_response.json()["data"] == {
        "backend_configured": False,
        "backend_type": None,
    }


def test_search_reindex_indexes_documents_with_configured_backend(tmp_path) -> None:
    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")
    backend = FakeSearchBackend()

    with session_factory() as session:
        paper = PaperRepository(session).upsert(
            provider="local_filesystem",
            external_id="paper-1",
            canonical_title="scLong",
            abstract="Long-range gene context modeling.",
            publication_year=2026,
            venue="Nature",
            authors=["Alice Smith"],
            tags=["scRegNet"],
        )
        session.add_all(
            [
                PaperFile(
                    paper_id=paper.id,
                    storage_uri="file:///tmp/scLong.pdf",
                    file_kind="pdf",
                    mime_type="application/pdf",
                    parser_status="parsed",
                ),
                Section(
                    paper_id=paper.id,
                    title="Methods",
                    ordinal=1,
                    text="We evaluate AUROC on scRegNetBench.",
                ),
                Chunk(
                    paper_id=paper.id,
                    section_id=None,
                    ordinal=1,
                    text="We evaluate AUROC on scRegNetBench.",
                ),
                Figure(
                    paper_id=paper.id,
                    page_number=2,
                    figure_label="Figure 1",
                    caption="Model architecture.",
                ),
                ExtractionRun(
                    paper_id=paper.id,
                    model_name="fake-extractor",
                    prompt_version="paperbase-v1",
                    schema_version="schema-v1",
                    status="completed",
                ),
            ]
        )
        session.commit()

    client = TestClient(create_app(session_factory=session_factory, search_backend=backend))

    response = client.post("/api/v1/search/reindex")

    assert response.status_code == 200
    assert response.json()["data"] == {
        "backend_type": "FakeSearchBackend",
        "indexed": {
            "papers": 1,
            "chunks": 1,
            "figures": 1,
        },
    }
    assert [call[0] for call in backend.index_calls] == [
        "paperbase-papers",
        "paperbase-chunks",
        "paperbase-figures",
    ]


def test_search_reindex_returns_503_without_configured_backend(tmp_path) -> None:
    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")
    client = TestClient(create_app(session_factory=session_factory))

    response = client.post("/api/v1/search/reindex")

    assert response.status_code == 503
    assert response.json() == {
        "error": "search_backend_unavailable",
        "message": "Search backend is not configured for reindexing.",
    }
