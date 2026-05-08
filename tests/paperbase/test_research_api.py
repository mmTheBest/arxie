from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from paperbase.db.bootstrap import initialize_database
from paperbase.db.models import BackgroundJob
from paperbase.db.repositories import CollectionRepository, PaperRepository
from paperbase.db.session import make_session_factory
from services.paperbase_api.app import create_app


def test_research_api_persists_threads_messages_artifacts_and_labels(tmp_path: Path) -> None:
    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")

    with session_factory() as session:
        paper = PaperRepository(session).upsert(
            provider="local_filesystem",
            external_id="/tmp/benchmark.pdf",
            canonical_title="Benchmark Paper",
        )
        collection = CollectionRepository(session).create(
            owner_id="local-user",
            title="scRegNet",
            description="Benchmark papers.",
        )
        CollectionRepository(session).add_paper(collection_id=collection.id, paper_id=paper.id)
        collection_id = collection.id
        paper_id = paper.id

    client = TestClient(create_app(session_factory=session_factory))

    label_response = client.patch(
        f"/api/v1/collections/{collection_id}/papers/{paper_id}/research-label",
        json={"user_label": "exemplar", "notes": "Strong benchmark design."},
    )
    assert label_response.status_code == 200
    label_payload = label_response.json()["data"]
    assert label_payload["collection_id"] == collection_id
    assert label_payload["paper_id"] == paper_id
    assert label_payload["user_label"] == "exemplar"
    assert "design_strength_score" in label_payload["inferred_signals"]

    labels_response = client.get(f"/api/v1/collections/{collection_id}/research-labels")
    assert labels_response.status_code == 200
    assert labels_response.json()["data"][0]["user_label"] == "exemplar"

    thread_response = client.post(
        "/api/v1/research/threads",
        json={
            "title": "GRN experiment design",
            "collection_id": collection_id,
            "selected_paper_ids": [paper_id],
        },
    )
    assert thread_response.status_code == 201
    thread = thread_response.json()["data"]
    thread_id = thread["id"]
    assert thread["collection_id"] == collection_id
    assert thread["selected_paper_ids"] == [paper_id]

    message_response = client.post(
        f"/api/v1/research/threads/{thread_id}/messages",
        json={
            "message": "Design an experiment using the strongest benchmark papers.",
            "artifact_type": "experiment_plan",
        },
    )
    assert message_response.status_code == 202
    pending_artifact = message_response.json()["data"]["artifact"]
    job = message_response.json()["data"]["job"]
    assert pending_artifact["artifact_type"] == "experiment_plan"
    assert pending_artifact["status"] == "pending"
    assert job["job_type"] == "research_agent_run"

    artifacts_response = client.get(f"/api/v1/research/artifacts?collection_id={collection_id}")
    assert artifacts_response.status_code == 200
    assert artifacts_response.json()["data"][0]["id"] == pending_artifact["id"]

    with session_factory() as session:
        jobs = session.query(BackgroundJob).all()
    assert len(jobs) == 1


def test_research_message_reuses_matching_active_job(tmp_path: Path) -> None:
    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")

    with session_factory() as session:
        collection = CollectionRepository(session).create(
            owner_id="local-user",
            title="scRegNet",
            description="Benchmark papers.",
        )
        collection_id = collection.id

    client = TestClient(create_app(session_factory=session_factory))
    thread = client.post(
        "/api/v1/research/threads",
        json={"title": "GRN experiment design", "collection_id": collection_id},
    ).json()["data"]

    first_response = client.post(
        f"/api/v1/research/threads/{thread['id']}/messages",
        json={"message": "Generate hypotheses.", "artifact_type": "hypotheses"},
    )
    duplicate_response = client.post(
        f"/api/v1/research/threads/{thread['id']}/messages",
        json={"message": "Generate hypotheses.", "artifact_type": "hypotheses"},
    )

    assert first_response.status_code == 202
    assert duplicate_response.status_code == 202
    assert duplicate_response.json()["data"]["job"]["id"] == first_response.json()["data"]["job"]["id"]


def test_research_message_infers_artifact_type_from_freeform_request(tmp_path: Path) -> None:
    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")

    with session_factory() as session:
        collection = CollectionRepository(session).create(
            owner_id="local-user",
            title="scRegNet",
            description="Benchmark papers.",
        )
        collection_id = collection.id

    client = TestClient(create_app(session_factory=session_factory))
    thread = client.post(
        "/api/v1/research/threads",
        json={"title": "GRN experiment design", "collection_id": collection_id},
    ).json()["data"]

    response = client.post(
        f"/api/v1/research/threads/{thread['id']}/messages",
        json={"message": "Generate hypotheses and gaps for this field."},
    )

    assert response.status_code == 202
    assert response.json()["data"]["artifact"]["artifact_type"] == "hypotheses"


def test_research_message_accepts_study_source_context_and_benchmark_artifact_type(tmp_path: Path) -> None:
    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")

    with session_factory() as session:
        collection = CollectionRepository(session).create(
            owner_id="local-user",
            title="scRegNet",
            description="Benchmark papers.",
        )
        collection_id = collection.id

    client = TestClient(create_app(session_factory=session_factory))
    study = client.post(
        "/api/v1/studies",
        json={"title": "GRN study", "collection_id": collection_id},
    ).json()["data"]
    source = client.post(
        f"/api/v1/studies/{study['id']}/sources",
        json={
            "source_type": "text",
            "title": "Current results",
            "content": "The current model has no distribution-shift benchmark yet.",
        },
    ).json()["data"]
    thread = client.post(
        "/api/v1/research/threads",
        json={
            "title": "GRN benchmark design",
            "collection_id": collection_id,
            "workspace_id": study["id"],
        },
    ).json()["data"]

    response = client.post(
        f"/api/v1/research/threads/{thread['id']}/messages",
        json={
            "message": "Suggest benchmarks based on my current results.",
            "artifact_type": "benchmark_plan",
            "source_ids": [source["id"]],
        },
    )

    assert response.status_code == 202
    payload = response.json()["data"]
    assert payload["artifact"]["artifact_type"] == "benchmark_plan"
    assert payload["artifact"]["input_payload"]["source_ids"] == [source["id"]]
    assert payload["job"]["payload"]["workspace_id"] == study["id"]
    assert payload["job"]["payload"]["source_ids"] == [source["id"]]
