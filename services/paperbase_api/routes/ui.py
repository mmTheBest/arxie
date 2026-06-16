"""Static UI shell routes for the Paperbase API service."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, HTMLResponse

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
REACT_APP_DIR = STATIC_DIR / "app"
REACT_APP_INDEX = REACT_APP_DIR / "index.html"
BOOTSTRAP_MARKER = '<script id="arxie-bootstrap" type="application/json"></script>'

router = APIRouter(tags=["ui"])


@router.get("/app", include_in_schema=False)
def paperbase_console(request: Request) -> HTMLResponse:
    template_path = REACT_APP_INDEX if REACT_APP_INDEX.exists() else STATIC_DIR / "index.html"
    html = template_path.read_text(encoding="utf-8")
    bootstrap: dict[str, str] | None = None
    policy = getattr(request.app.state, "hosted_security_policy", None)
    if policy is not None:
        bootstrap = policy.issue_browser_session()
    response = HTMLResponse(_inject_bootstrap_payload(html, bootstrap=bootstrap))
    if bootstrap is not None:
        response.set_cookie(
            key="arxie-hosted-session",
            value=bootstrap["session_id"],
            httponly=True,
            samesite="strict",
            secure=request.url.scheme == "https",
            path="/",
        )
    return response


@router.get("/", include_in_schema=False)
def arxie_homepage() -> FileResponse:
    return FileResponse(STATIC_DIR / "landing.html")


def _inject_bootstrap_payload(
    html: str,
    *,
    bootstrap: dict[str, str] | None,
) -> str:
    if bootstrap is None:
        return html
    serialized_bootstrap = json.dumps(
        {"csrf_token": bootstrap["csrf_token"]},
        separators=(",", ":"),
    ).replace("</", "<\\/")
    payload = BOOTSTRAP_MARKER.replace("</script>", f">{serialized_bootstrap}</script>")
    if BOOTSTRAP_MARKER in html:
        return html.replace(BOOTSTRAP_MARKER, payload, 1)
    return html.replace("</head>", f"  {payload}\n  </head>", 1)
