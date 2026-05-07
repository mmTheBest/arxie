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
    assert 'id="sidebar-papers-list"' in html
    assert 'id="sidebar-workspaces-list"' in html
    assert 'class="sidebar-section sidebar-workspaces-section"' in html
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
    assert 'id="parse-selected-papers-button"' in html
    assert 'id="select-all-papers-button"' in html
    assert 'id="select-unextracted-papers-button"' in html
    assert 'id="clear-selected-papers-button"' in html
    assert 'id="extract-unprocessed-papers-button"' in html
    assert 'id="local-library-upload-form"' in html
    assert 'id="local-library-upload-input"' in html
    assert 'id="local-library-upload-title-input"' in html
    assert 'id="local-library-upload-description-input"' in html
    assert 'id="local-library-upload-button"' in html
    assert 'id="local-library-form"' in html
    assert 'id="local-library-source-input"' in html
    assert 'id="local-library-title-input"' in html
    assert 'id="local-library-import-button"' in html
    assert 'id="library-job-log"' in html
    assert 'id="library-collections-grid"' not in html
    assert "Choose a collection and follow the next readiness step" not in html
    assert "Parse unprocessed" in html
    assert "Parse selected" in html
    assert "Select all" in html
    assert "Select unextracted" in html
    assert "Clear selection" in html
    assert "Extract unextracted" in html
    assert "Extract selected" in html
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
    assert "renderSidebarPapers" in script_response.text
    assert "data-sidebar-paper-checkbox" in script_response.text
    assert "paper_ids" in script_response.text
    assert "is_parsed" in script_response.text
    assert "is_extracted" in script_response.text
    assert "Library activity" in script_response.text
    assert "Stale" in script_response.text
    assert "Evidence ready" in script_response.text
    assert "Text ready" in script_response.text
    assert "Structured evidence is not ready yet. Run extraction in Library." in script_response.text
    assert "data-library-compare-id" not in script_response.text

    assert style_response.status_code == 200
    assert "--paperbase-bg" in style_response.text
    assert '.app-shell[data-active-view="library"] .sidebar-workspaces-section' in style_response.text


def test_library_uses_sidebar_paper_processing_instead_of_duplicate_collection_cards() -> None:
    script_path = "services/paperbase_api/static/paperbase-ui.js"
    script = Path(script_path).read_text()

    assert "renderSidebarPapers" in script
    assert "renderLibraryLogs" in script
    assert "queueParse(unprocessedPaperIds())" in script
    assert "queueParse(selectedPaperIdsForAction" in script
    assert "selectPaperIds(extractablePaperIds())" in script
    assert "queueExtraction(extractablePaperIds())" in script
    assert "queueExtraction(selectedPaperIdsForAction" in script
    assert "data-sidebar-paper-action" in script
    assert "isStaleJob" in script
    assert "latest_job_error" in script
    assert "data-library-open-id" not in script
    assert "data-library-next-step-id" not in script
    assert "data-library-more-id" not in script
    assert "data-library-compare-id" not in script
    assert "data-library-parse-id" not in script
    assert "data-library-extract-id" not in script
    assert "getCollectionReadiness" in script
