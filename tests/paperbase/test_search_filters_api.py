from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy.dialects import postgresql

from paperbase.db.bootstrap import initialize_database
from paperbase.db.models import Dataset, ExtractionRun, Method, Metric
from paperbase.db.repositories import CollectionRepository, PaperRepository
from paperbase.db.session import make_session_factory
from services.paperbase_api.app import create_app
from services.paperbase_api.routes.search import _base_paper_search_statement


def test_paper_search_sql_does_not_distinct_json_columns_for_postgres() -> None:
    compiled = str(_base_paper_search_statement().compile(dialect=postgresql.dialect()))

    assert "SELECT DISTINCT" not in compiled
    assert "papers.raw_metadata" in compiled


def test_paperbase_search_supports_collection_and_structured_filters(tmp_path) -> None:
    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")

    with session_factory() as session:
        paper_one = PaperRepository(session).upsert(
            provider="local_filesystem",
            external_id="paper-1",
            canonical_title="scLong",
            abstract="Long-range gene context modeling.",
            publication_year=2026,
            venue="Nature",
            authors=["Alice Smith", "Bob Lee"],
            tags=["single-cell", "scRegNet"],
        )
        paper_two = PaperRepository(session).upsert(
            provider="local_filesystem",
            external_id="paper-2",
            canonical_title="BaselineNet",
            abstract="Baseline model for comparison.",
            publication_year=2024,
            venue="Cell",
            authors=["Carol Jones"],
            tags=["baseline"],
        )

        collection = CollectionRepository(session).create(
            owner_id="local-user",
            title="SamplePapers",
            description="scRegNet field-specific corpus.",
        )
        CollectionRepository(session).add_paper(collection_id=collection.id, paper_id=paper_one.id)
        CollectionRepository(session).add_paper(collection_id=collection.id, paper_id=paper_two.id)

        session.add_all(
            [
                Dataset(paper_id=paper_one.id, normalized_name="scregnetbench", display_name="scRegNetBench"),
                Dataset(paper_id=paper_two.id, normalized_name="otherbench", display_name="OtherBench"),
                Method(paper_id=paper_one.id, normalized_name="sclong", display_name="scLong"),
                Method(paper_id=paper_two.id, normalized_name="baselinenet", display_name="BaselineNet"),
                Metric(paper_id=paper_one.id, normalized_name="auroc", display_name="AUROC"),
                Metric(paper_id=paper_two.id, normalized_name="accuracy", display_name="Accuracy"),
                ExtractionRun(
                    paper_id=paper_one.id,
                    model_name="fake-extractor",
                    prompt_version="paperbase-v1",
                    schema_version="schema-v1",
                    status="completed",
                ),
            ]
        )
        session.commit()
        collection_id = collection.id
        paper_one_id = paper_one.id

    client = TestClient(create_app(session_factory=session_factory))

    response = client.get(
        "/api/v1/search/papers",
        params={
            "collection_id": collection_id,
            "dataset": "scRegNetBench",
            "method": "scLong",
            "metric": "AUROC",
            "venue": "Nature",
            "year_gte": 2025,
            "author": "Alice Smith",
            "tag": "scRegNet",
            "extraction_state": "extracted",
        },
    )

    assert response.status_code == 200
    assert [item["id"] for item in response.json()["data"]] == [paper_one_id]
    assert response.json()["data"][0]["authors"] == ["Alice Smith", "Bob Lee"]
    assert response.json()["data"][0]["tags"] == ["scRegNet", "single-cell"]


def test_paperbase_sql_search_matches_scientific_multi_term_queries(tmp_path) -> None:
    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")

    with session_factory() as session:
        paper = PaperRepository(session).upsert(
            provider="local_filesystem",
            external_id="paper-1",
            canonical_title="Leveraging prior knowledge to infer gene regulatory networks",
            abstract="Benchmark evaluation for single-cell GRN inference with AUROC and AUPRC.",
            publication_year=2026,
            venue="Nature",
        )
        collection = CollectionRepository(session).create(
            owner_id="local-user",
            title="SamplePapers",
            description="scRegNet field-specific corpus.",
        )
        CollectionRepository(session).add_paper(collection_id=collection.id, paper_id=paper.id)
        session.commit()
        collection_id = collection.id
        paper_id = paper.id

    client = TestClient(create_app(session_factory=session_factory))
    response = client.get(
        "/api/v1/search/papers",
        params={
            "collection_id": collection_id,
            "q": "single-cell gene regulatory network inference using prior knowledge and benchmark evaluation with AUROC AUPRC",
        },
    )

    assert response.status_code == 200
    assert [item["id"] for item in response.json()["data"]] == [paper_id]


def test_paperbase_search_requires_query_or_filter(tmp_path) -> None:
    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")
    client = TestClient(create_app(session_factory=session_factory))

    response = client.get("/api/v1/search/papers")

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_input"
