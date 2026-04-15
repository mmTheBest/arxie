from __future__ import annotations

from fastapi.testclient import TestClient

from paperbase.db.bootstrap import initialize_database
from paperbase.db.models import Chunk, Dataset, ExtractionRun, Figure, Method, Metric, PaperFile, Section
from paperbase.db.repositories import PaperRepository
from paperbase.db.session import make_session_factory
from services.paperbase_api.app import create_app


class FakeSearchBackend:
    def __init__(self) -> None:
        self.index_calls: list[tuple[str, dict[str, object]]] = []
        self.bulk_calls: list[tuple[str, list[dict[str, object]]]] = []
        self.search_calls: list[tuple[str, dict[str, object], int]] = []
        self.search_results: list[dict[str, object]] = []

    def ensure_index(self, index_name: str, template: dict[str, object]) -> None:
        self.index_calls.append((index_name, template))

    def bulk_index(self, index_name: str, documents: list[dict[str, object]]) -> None:
        self.bulk_calls.append((index_name, documents))

    def search(self, index_name: str, query: dict[str, object], size: int) -> list[dict[str, object]]:
        self.search_calls.append((index_name, query, size))
        return list(self.search_results)


def test_reindexer_builds_and_pushes_paper_chunk_and_figure_documents(tmp_path) -> None:
    from paperbase.search.runtime import PaperbaseSearchReindexer

    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")

    with session_factory() as session:
        paper = PaperRepository(session).upsert(
            provider="local_filesystem",
            external_id="paper-1",
            canonical_title="scLong",
            abstract="Long-range gene context modeling.",
            publication_year=2026,
            venue="Nature",
            doi="10.1000/example",
            arxiv_id="2501.00001",
            authors=["Alice Smith", "Bob Lee"],
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
                Dataset(paper_id=paper.id, normalized_name="scregnetbench", display_name="scRegNetBench"),
                Method(paper_id=paper.id, normalized_name="sclong", display_name="scLong"),
                Metric(paper_id=paper.id, normalized_name="auroc", display_name="AUROC"),
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

    backend = FakeSearchBackend()
    summary = PaperbaseSearchReindexer(session_factory=session_factory, backend=backend).reindex_all()

    assert summary["papers"] == 1
    assert summary["chunks"] == 1
    assert summary["figures"] == 1
    assert [call[0] for call in backend.index_calls] == [
        "paperbase-papers",
        "paperbase-chunks",
        "paperbase-figures",
    ]

    paper_docs = next(documents for index_name, documents in backend.bulk_calls if index_name == "paperbase-papers")
    assert paper_docs[0]["authors"] == ["Alice Smith", "Bob Lee"]
    assert paper_docs[0]["tags"] == ["scRegNet"]
    assert paper_docs[0]["datasets"] == ["scRegNetBench"]
    assert paper_docs[0]["methods"] == ["scLong"]
    assert paper_docs[0]["metrics"] == ["AUROC"]
    assert paper_docs[0]["extraction_state"] == "extracted"


def test_search_api_uses_configured_backend_when_available(tmp_path) -> None:
    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")
    backend = FakeSearchBackend()
    backend.search_results = [
        {
            "paper_id": "paper-1",
            "title": "scLong",
            "abstract": "Long-range gene context modeling.",
            "publication_year": 2026,
            "venue": "Nature",
            "provider": "local_filesystem",
            "external_id": "paper-1",
            "doi": "10.1000/example",
            "arxiv_id": "2501.00001",
            "authors": ["Alice Smith", "Bob Lee"],
            "tags": ["scRegNet"],
            "datasets": ["scRegNetBench"],
            "methods": ["scLong"],
            "metrics": ["AUROC"],
            "extraction_state": "extracted",
        }
    ]

    client = TestClient(create_app(session_factory=session_factory, search_backend=backend))

    response = client.get(
        "/api/v1/search/papers",
        params={
            "q": "gene regulatory",
            "author": "Alice Smith",
            "tag": "scRegNet",
            "dataset": "scRegNetBench",
            "method": "scLong",
            "metric": "AUROC",
            "extraction_state": "extracted",
        },
    )

    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "paper-1"
    assert response.json()["data"][0]["authors"] == ["Alice Smith", "Bob Lee"]
    assert backend.search_calls[0][0] == "paperbase-papers"
    assert {"terms": {"authors.keyword": ["Alice Smith"]}} in backend.search_calls[0][1]["bool"]["filter"]
    assert {"terms": {"metrics.keyword": ["AUROC"]}} in backend.search_calls[0][1]["bool"]["filter"]
    assert {"term": {"extraction_state": "extracted"}} in backend.search_calls[0][1]["bool"]["filter"]
