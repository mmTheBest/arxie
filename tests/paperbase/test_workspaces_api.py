from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from paperbase.db.bootstrap import initialize_database
from paperbase.db.repositories import CollectionRepository, PaperRepository
from paperbase.db.session import make_session_factory
from services.paperbase_api.app import create_app


def test_workspaces_api_persists_saved_research_context(tmp_path: Path) -> None:
    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")

    with session_factory() as session:
        paper = PaperRepository(session).upsert(
            provider="local_filesystem",
            external_id="/tmp/scregnet.pdf",
            canonical_title="scRegNet",
            abstract="Graph learning for gene regulation.",
            publication_year=2026,
            venue="Nature",
            authors=["Mina Ma", "Alice Smith"],
            tags=["single-cell", "grn"],
        )
        collection = CollectionRepository(session).create(
            owner_id="local-user",
            title="scRegNet collection",
            description="Curated GRN benchmark papers.",
            tags=["single-cell", "benchmark"],
        )
        CollectionRepository(session).add_paper(
            collection_id=collection.id,
            paper_id=paper.id,
            position=1,
            membership_note="Pinned benchmark paper",
        )

    client = TestClient(create_app(session_factory=session_factory))

    create_response = client.post(
        "/api/v1/workspaces",
        json={
            "title": "scRegNet workspace",
            "description": "Focused workspace for GRN comparison",
            "collection_id": collection.id,
            "saved_query": "scRegNet AUROC benchmark",
            "focus_note": "Track methods using graph priors and multiome evidence.",
            "active_filters": {"metric": "AUROC", "dataset": "mESC"},
            "pinned_paper_ids": [paper.id],
        },
    )
    assert create_response.status_code == 201
    created = create_response.json()["data"]
    workspace_id = created["id"]
    assert created["title"] == "scRegNet workspace"
    assert created["saved_query"] == "scRegNet AUROC benchmark"
    assert created["pinned_paper_ids"] == [paper.id]

    list_response = client.get("/api/v1/workspaces")
    assert list_response.status_code == 200
    assert list_response.json()["data"][0]["id"] == workspace_id

    detail_response = client.get(f"/api/v1/workspaces/{workspace_id}")
    assert detail_response.status_code == 200
    detail = detail_response.json()["data"]
    assert detail["collection"]["id"] == collection.id
    assert detail["active_filters"] == {"metric": "AUROC", "dataset": "mESC"}
    assert detail["pinned_papers"][0]["id"] == paper.id
    assert detail["pinned_papers"][0]["authors"] == ["Mina Ma", "Alice Smith"]

    update_response = client.patch(
        f"/api/v1/workspaces/{workspace_id}",
        json={
            "saved_query": "scRegNet versus scMultiomeGRN AUROC",
            "focus_note": "Prioritize result tables and ablation evidence.",
            "active_filters": {"metric": "AUROC", "tag": "single-cell"},
        },
    )
    assert update_response.status_code == 200
    updated = update_response.json()["data"]
    assert updated["saved_query"] == "scRegNet versus scMultiomeGRN AUROC"
    assert updated["focus_note"] == "Prioritize result tables and ablation evidence."
    assert updated["active_filters"] == {"metric": "AUROC", "tag": "single-cell"}
