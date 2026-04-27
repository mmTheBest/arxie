from __future__ import annotations

from fastapi.testclient import TestClient

from paperbase.db.bootstrap import initialize_database
from paperbase.db.models import Chunk, Figure, PaperFile, Section, TableArtifact
from paperbase.db.repositories import CollectionRepository, PaperRepository
from paperbase.db.session import make_session_factory
from services.paperbase_api.app import create_app


def test_chunk_and_artifact_search_support_sql_fallback(tmp_path) -> None:
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
            authors=["Alice Smith"],
            tags=["scRegNet"],
        )
        collection = CollectionRepository(session).create(
            owner_id="local-user",
            title="SamplePapers",
            description="Curated field-specific corpus.",
        )
        CollectionRepository(session).add_paper(collection_id=collection.id, paper_id=paper.id)
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
                    caption="Benchmark ablation figure.",
                ),
                TableArtifact(
                    paper_id=paper.id,
                    page_number=3,
                    table_label="Table 1",
                    caption="Benchmark comparison table.",
                    structured_payload_json={"rows": 2},
                ),
            ]
        )
        session.commit()
        collection_id = collection.id
        paper_id = paper.id

    client = TestClient(create_app(session_factory=session_factory))

    chunk_response = client.get(
        "/api/v1/search/chunks",
        params={"q": "AUROC", "collection_id": collection_id},
    )
    artifact_response = client.get(
        "/api/v1/search/artifacts",
        params={"q": "benchmark", "collection_id": collection_id, "kind": "all"},
    )

    assert chunk_response.status_code == 200
    assert artifact_response.status_code == 200
    chunk_payload = chunk_response.json()["data"][0]
    artifact_payload = artifact_response.json()["data"]
    assert chunk_payload["paper_id"] == paper_id
    assert chunk_payload["paper_title"] == "scLong"
    assert "AUROC" in chunk_payload["text"]
    assert {item["artifact_type"] for item in artifact_payload} == {"figure", "table"}
    assert {item["paper_id"] for item in artifact_payload} == {paper_id}
