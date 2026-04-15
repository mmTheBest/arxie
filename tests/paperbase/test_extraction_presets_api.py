from __future__ import annotations

from fastapi.testclient import TestClient

from paperbase.db.bootstrap import initialize_database
from paperbase.db.session import make_session_factory
from services.paperbase_api.app import create_app


def test_paperbase_api_lists_and_creates_sc_regnet_preset_profiles(tmp_path) -> None:
    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")
    client = TestClient(create_app(session_factory=session_factory))

    list_response = client.get("/api/v1/extraction-profile-presets")
    assert list_response.status_code == 200
    preset = next(item for item in list_response.json()["data"] if item["name"] == "sc_regnet")
    assert preset["domain"] == "single-cell-grn"
    assert "Study" in preset["schema_payload"]
    assert "Benchmark" in preset["schema_payload"]

    create_response = client.post(
        "/api/v1/extraction-profiles",
        json={
            "name": "scRegNet profile",
            "preset_name": "sc_regnet",
        },
    )
    assert create_response.status_code == 201
    payload = create_response.json()["data"]
    assert payload["name"] == "scRegNet profile"
    assert "RegulatoryEntity" in payload["schema_payload"]
    assert payload["schema_payload"]["Study"]["required"][0] == "title"
