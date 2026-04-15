from __future__ import annotations

from fastapi.testclient import TestClient

from paperbase.db.bootstrap import initialize_database
from paperbase.db.session import make_session_factory
from services.paperbase_api.app import create_app


def test_job_status_returns_enqueued_background_job(tmp_path) -> None:
    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")
    client = TestClient(create_app(session_factory=session_factory))

    enqueue_response = client.post("/api/v1/search/reindex")

    assert enqueue_response.status_code == 202
    job_id = enqueue_response.json()["data"]["id"]

    status_response = client.get(f"/api/v1/jobs/{job_id}")

    assert status_response.status_code == 200
    payload = status_response.json()["data"]
    assert payload["id"] == job_id
    assert payload["job_type"] == "search_reindex"
    assert payload["status"] == "pending"
    assert payload["payload"] == {}
    assert payload["result"] is None
    assert payload["error_message"] is None
