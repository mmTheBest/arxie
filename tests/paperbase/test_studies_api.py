from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from paperbase.db.bootstrap import initialize_database
from paperbase.db.repositories import CollectionRepository, PaperRepository
from paperbase.db.session import make_session_factory
from services.paperbase_api.app import create_app


def test_studies_api_aliases_workspace_records_and_persists_sources(tmp_path: Path) -> None:
    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")

    source_file = tmp_path / "draft.md"
    source_file.write_text("# Draft\n\nWe need stronger ablations for the graph prior.\n")

    with session_factory() as session:
        paper = PaperRepository(session).upsert(
            provider="local_filesystem",
            external_id="/tmp/benchmark.pdf",
            canonical_title="Benchmark Paper",
        )
        collection = CollectionRepository(session).create(
            owner_id="local-user",
            title="Benchmark collection",
            description="Papers for study design.",
        )
        CollectionRepository(session).add_paper(collection_id=collection.id, paper_id=paper.id)
        collection_id = collection.id
        paper_id = paper.id

    client = TestClient(create_app(session_factory=session_factory))

    create_response = client.post(
        "/api/v1/studies",
        json={
            "title": "Graph prior study",
            "collection_id": collection_id,
            "focus_note": "Design missing ablations.",
            "pinned_paper_ids": [paper_id],
        },
    )
    assert create_response.status_code == 201
    study = create_response.json()["data"]
    study_id = study["id"]
    assert study["title"] == "Graph prior study"
    assert study["collection_id"] == collection_id

    list_response = client.get("/api/v1/studies")
    assert list_response.status_code == 200
    assert list_response.json()["data"][0]["id"] == study_id

    workspace_response = client.get(f"/api/v1/workspaces/{study_id}")
    assert workspace_response.status_code == 200
    assert workspace_response.json()["data"]["title"] == "Graph prior study"

    text_source_response = client.post(
        f"/api/v1/studies/{study_id}/sources",
        json={
            "source_type": "text",
            "title": "Current idea",
            "content": "I want to test whether graph priors help under distribution shift.",
        },
    )
    assert text_source_response.status_code == 201
    text_source = text_source_response.json()["data"]
    assert text_source["source_type"] == "text"
    assert text_source["read_status"] == "ready"
    assert "distribution shift" in text_source["summary"]

    path_source_response = client.post(
        f"/api/v1/studies/{study_id}/sources",
        json={
            "source_type": "draft_path",
            "title": "Draft",
            "path": str(source_file),
        },
    )
    assert path_source_response.status_code == 201
    path_source = path_source_response.json()["data"]
    assert path_source["source_type"] == "draft_path"
    assert path_source["path"] == str(source_file)
    assert path_source["read_status"] == "ready"
    assert "stronger ablations" in path_source["summary"]

    missing_source_response = client.post(
        f"/api/v1/studies/{study_id}/sources",
        json={
            "source_type": "results_path",
            "title": "Missing result file",
            "path": str(tmp_path / "missing.csv"),
        },
    )
    assert missing_source_response.status_code == 201
    missing_source = missing_source_response.json()["data"]
    assert missing_source["read_status"] == "error"
    assert "not found" in missing_source["error_message"].lower()

    sources_response = client.get(f"/api/v1/studies/{study_id}/sources")
    assert sources_response.status_code == 200
    assert [item["title"] for item in sources_response.json()["data"]] == [
        "Current idea",
        "Draft",
        "Missing result file",
    ]

    delete_response = client.delete(f"/api/v1/studies/{study_id}/sources/{text_source['id']}")
    assert delete_response.status_code == 204

    remaining_response = client.get(f"/api/v1/studies/{study_id}/sources")
    assert remaining_response.status_code == 200
    assert [item["id"] for item in remaining_response.json()["data"]] == [
        path_source["id"],
        missing_source["id"],
    ]
