"""Static UI shell routes for the Paperbase API service."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

router = APIRouter(tags=["ui"])


@router.get("/app", include_in_schema=False)
def paperbase_console() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@router.get("/", include_in_schema=False)
def arxie_homepage() -> FileResponse:
    return FileResponse(STATIC_DIR / "landing.html")
