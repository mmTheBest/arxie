from __future__ import annotations

from pathlib import Path

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
    assert 'id="app-shell"' in html
    assert 'id="app-nav"' in html
    assert 'id="sidebar-collections-list"' in html
    assert 'id="sidebar-workspaces-list"' in html
    assert 'id="library-view"' in html
    assert 'id="workspace-view"' in html
    assert 'id="compare-view"' in html
    assert 'id="jobs-view"' in html
    assert 'id="settings-view"' in html
    assert 'id="workspace-readiness-banner"' in html
    assert 'id="workspace-detail-panel"' in html
    assert 'id="save-workspace-button"' in html
    assert 'id="compare-readiness-banner"' in html
    assert 'id="parse-button"' in html
    assert 'id="local-library-upload-form"' in html
    assert 'id="local-library-upload-input"' in html
    assert 'id="local-library-upload-title-input"' in html
    assert 'id="local-library-upload-description-input"' in html
    assert 'id="local-library-upload-button"' in html
    assert 'id="local-library-form"' in html
    assert 'id="local-library-source-input"' in html
    assert 'id="local-library-title-input"' in html
    assert 'id="local-library-import-button"' in html
    assert 'data-view="library"' in html
    assert 'data-view="workspace"' in html
    assert 'data-view="compare"' in html
    assert 'data-view="jobs"' in html
    assert 'data-view="settings"' in html
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
    assert "/api/v1/compare/results" in script_response.text
    assert "/api/v1/compare/methods" in script_response.text
    assert "/api/v1/compare/engineering-tricks" in script_response.text
    assert "/api/v1/ingest/local-library-upload" in script_response.text
    assert "/api/v1/ingest/local-library" in script_response.text
    assert "/api/v1/search/status" in script_response.text
    assert "local_library_ingest" in script_response.text
    assert "collection_id" in script_response.text
    assert "/parse" in script_response.text
    assert "/api/v1/papers/" in script_response.text
    assert "/tables" in script_response.text
    assert "activeView" in script_response.text
    assert "getCollectionReadiness" in script_response.text
    assert "Run Next Step" in script_response.text
    assert "Open Workspace" in script_response.text
    assert "Evidence ready" in script_response.text
    assert "Text ready" in script_response.text
    assert "Structured evidence is not ready yet. Run extraction in Library." in script_response.text
    assert "data-library-compare-id" not in script_response.text

    assert style_response.status_code == 200
    assert "--paperbase-bg" in style_response.text


def test_library_cards_use_guided_readiness_actions() -> None:
    script_path = "services/paperbase_api/static/paperbase-ui.js"
    script = Path(script_path).read_text()

    assert "data-library-open-id" in script
    assert "data-library-next-step-id" in script
    assert "data-library-more-id" in script
    assert "data-library-compare-id" not in script
    assert "data-library-parse-id" not in script
    assert "data-library-extract-id" not in script
    assert "Open Workspace" in script
    assert "Run Next Step" in script
    assert "Ready" in script
    assert "getCollectionReadiness" in script
