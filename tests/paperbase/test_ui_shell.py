from __future__ import annotations

from fastapi.testclient import TestClient

from paperbase.db.bootstrap import initialize_database
from paperbase.db.session import make_session_factory
from services.paperbase_api.app import create_app


def test_paperbase_ui_shell_and_assets_are_served(tmp_path) -> None:
    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")
    client = TestClient(create_app(session_factory=session_factory))

    landing_response = client.get("/")
    shell_response = client.get("/app")
    script_response = client.get("/ui/paperbase-ui.js")
    style_response = client.get("/ui/paperbase-ui.css")

    assert landing_response.status_code == 200
    assert "text/html" in landing_response.headers["content-type"]
    landing_html = landing_response.text
    assert "Arxie" in landing_html
    assert "research operating system" in landing_html.lower()
    assert 'href="/app"' in landing_html

    assert shell_response.status_code == 200
    assert "text/html" in shell_response.headers["content-type"]
    html = shell_response.text
    assert "Arxie Workspace" in html
    assert 'id="workspaces-panel"' in html
    assert 'id="workspace-detail-panel"' in html
    assert 'id="save-workspace-button"' in html
    assert 'id="collections-panel"' in html
    assert 'id="papers-panel"' in html
    assert 'id="summary-panel"' in html
    assert 'id="search-surfaces-panel"' in html
    assert 'id="artifacts-panel"' in html
    assert 'id="jobs-panel"' in html
    assert 'id="parse-button"' in html
    assert "/ui/paperbase-ui.css" in html
    assert "/ui/paperbase-ui.js" in html

    assert script_response.status_code == 200
    assert "/api/v1/workspaces" in script_response.text
    assert "/api/v1/collections" in script_response.text
    assert "/api/v1/jobs" in script_response.text
    assert "/api/v1/search/chunks" in script_response.text
    assert "/api/v1/search/artifacts" in script_response.text
    assert "/api/v1/compare/figures" in script_response.text
    assert "/api/v1/compare/tables" in script_response.text
    assert "/parse" in script_response.text
    assert "/api/v1/papers/" in script_response.text
    assert "/tables" in script_response.text

    assert style_response.status_code == 200
    assert "--paperbase-bg" in style_response.text
