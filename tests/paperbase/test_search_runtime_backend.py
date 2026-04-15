from __future__ import annotations

from fastapi.testclient import TestClient

from paperbase.db.bootstrap import initialize_database
from paperbase.db.models import Chunk, Dataset, ExtractionRun, Figure, Method, Metric, PaperFile, Section, TableArtifact
from paperbase.db.repositories import PaperRepository
from paperbase.db.session import make_session_factory
from services.paperbase_api.app import create_app


class FakeSearchBackend:
    def __init__(self) -> None:
        self.index_calls: list[tuple[str, dict[str, object]]] = []
        self.bulk_calls: list[tuple[str, list[dict[str, object]]]] = []
        self.search_calls: list[tuple[str, dict[str, object], int]] = []
        self.search_results: dict[str, list[dict[str, object]]] = {}

    def ensure_index(self, index_name: str, template: dict[str, object]) -> None:
        self.index_calls.append((index_name, template))

    def bulk_index(self, index_name: str, documents: list[dict[str, object]]) -> None:
        self.bulk_calls.append((index_name, documents))

    def search(self, index_name: str, query: dict[str, object], size: int) -> list[dict[str, object]]:
        self.search_calls.append((index_name, query, size))
        return list(self.search_results.get(index_name, []))


def test_reindexer_builds_and_pushes_paper_chunk_figure_and_table_documents(tmp_path) -> None:
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
                TableArtifact(
                    paper_id=paper.id,
                    page_number=3,
                    table_label="Table 1",
                    caption="Benchmark comparison table.",
                    structured_payload_json={"rows": 2},
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
    assert summary["tables"] == 1
    assert [call[0] for call in backend.index_calls] == [
        "paperbase-papers",
        "paperbase-chunks",
        "paperbase-figures",
        "paperbase-tables",
    ]

    paper_docs = next(documents for index_name, documents in backend.bulk_calls if index_name == "paperbase-papers")
    table_docs = next(documents for index_name, documents in backend.bulk_calls if index_name == "paperbase-tables")
    assert paper_docs[0]["authors"] == ["Alice Smith", "Bob Lee"]
    assert paper_docs[0]["tags"] == ["scRegNet"]
    assert paper_docs[0]["datasets"] == ["scRegNetBench"]
    assert paper_docs[0]["methods"] == ["scLong"]
    assert paper_docs[0]["metrics"] == ["AUROC"]
    assert paper_docs[0]["extraction_state"] == "extracted"
    assert table_docs[0]["table_label"] == "Table 1"


def test_search_api_uses_configured_backend_when_available(tmp_path) -> None:
    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")
    backend = FakeSearchBackend()
    backend.search_results["paperbase-papers"] = [
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
    assert backend.search_calls[0][1]["knn"]["field"] == "embedding"


def test_search_api_uses_configured_backend_for_chunk_and_artifact_surfaces(tmp_path) -> None:
    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")
    backend = FakeSearchBackend()
    backend.search_results["paperbase-chunks"] = [
        {
            "chunk_id": "chunk-1",
            "paper_id": "paper-1",
            "title": "scLong",
            "section_title": "Methods",
            "text": "We evaluate AUROC on scRegNetBench.",
        }
    ]
    backend.search_results["paperbase-figures"] = [
        {
            "figure_id": "figure-1",
            "paper_id": "paper-1",
            "title": "scLong",
            "figure_label": "Figure 1",
            "caption": "Benchmark ablation figure.",
        }
    ]
    backend.search_results["paperbase-tables"] = [
        {
            "table_id": "table-1",
            "paper_id": "paper-1",
            "title": "scLong",
            "table_label": "Table 1",
            "caption": "Benchmark comparison table.",
            "structured_payload": {"rows": 2},
        }
    ]

    client = TestClient(create_app(session_factory=session_factory, search_backend=backend))

    chunk_response = client.get("/api/v1/search/chunks", params={"q": "gene regulatory"})
    artifact_response = client.get("/api/v1/search/artifacts", params={"q": "benchmark", "kind": "all"})

    assert chunk_response.status_code == 200
    assert artifact_response.status_code == 200
    assert chunk_response.json()["data"][0]["chunk_id"] == "chunk-1"
    assert {item["artifact_type"] for item in artifact_response.json()["data"]} == {"figure", "table"}
    assert backend.search_calls[0][0] == "paperbase-chunks"
    assert backend.search_calls[0][1]["knn"]["field"] == "embedding"
    assert {call[0] for call in backend.search_calls[1:]} == {"paperbase-figures", "paperbase-tables"}
