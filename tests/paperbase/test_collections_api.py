from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from paperbase.db.bootstrap import initialize_database
from paperbase.db.repositories import PaperRepository
from paperbase.db.session import make_session_factory
from services.paperbase_api.app import create_app


def test_paperbase_api_manages_collections_and_annotations(tmp_path: Path) -> None:
    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")

    with session_factory() as session:
        paper = PaperRepository(session).upsert(
            provider="local_filesystem",
            external_id="/tmp/paper-a.pdf",
            canonical_title="scLong",
            abstract="Long-range gene context modeling.",
            publication_year=2026,
            venue="Nature",
        )
        paper_id = paper.id

    client = TestClient(create_app(session_factory=session_factory))

    create_collection = client.post(
        "/api/v1/collections",
        json={
            "title": "scRegNet",
            "description": "Single-cell gene regulation papers.",
            "tags": ["single-cell", "grn"],
        },
    )
    assert create_collection.status_code == 201
    collection_payload = create_collection.json()["data"]
    collection_id = collection_payload["id"]
    assert collection_payload["title"] == "scRegNet"

    add_paper = client.post(
        f"/api/v1/collections/{collection_id}/papers",
        json={"paper_id": paper_id, "position": 1, "membership_note": "Core benchmark paper"},
    )
    assert add_paper.status_code == 201
    assert add_paper.json()["data"]["paper_id"] == paper_id

    list_collections = client.get("/api/v1/collections")
    assert list_collections.status_code == 200
    listed_collection = list_collections.json()["data"][0]
    assert listed_collection["id"] == collection_id
    assert listed_collection["paper_count"] == 1
    assert listed_collection["parsed_paper_count"] == 0
    assert listed_collection["extracted_paper_count"] == 0
    assert listed_collection["latest_parse_job_status"] is None
    assert listed_collection["latest_extraction_job_status"] is None
    assert listed_collection["failed_job_count"] == 0

    collection_papers = client.get(f"/api/v1/collections/{collection_id}/papers")
    assert collection_papers.status_code == 200
    assert collection_papers.json()["data"][0]["paper"]["id"] == paper_id
    assert collection_papers.json()["data"][0]["membership_note"] == "Core benchmark paper"

    create_annotation = client.post(
        "/api/v1/annotations",
        json={
            "collection_id": collection_id,
            "target_type": "paper",
            "target_id": paper_id,
            "body": "Strong engineering trick section; keep for review.",
            "tags": ["important", "review"],
            "status": "active",
        },
    )
    assert create_annotation.status_code == 201
    annotation_id = create_annotation.json()["data"]["id"]

    list_annotations = client.get(
        "/api/v1/annotations",
        params={"target_type": "paper", "target_id": paper_id},
    )
    assert list_annotations.status_code == 200
    annotation_payload = list_annotations.json()["data"][0]
    assert annotation_payload["id"] == annotation_id
    assert annotation_payload["collection_id"] == collection_id
    assert annotation_payload["tags"] == ["important", "review"]
