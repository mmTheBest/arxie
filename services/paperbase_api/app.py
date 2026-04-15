"""FastAPI app for the local-first Paperbase API service."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session, sessionmaker

from paperbase.db.session import make_session_factory
from ra.utils.logging_config import configure_logging_from_env
from services.paperbase_api.errors import register_exception_handlers
from services.paperbase_api.models import HealthResponse
from services.paperbase_api.routes.compare import router as compare_router
from services.paperbase_api.routes.collections import router as collections_router
from services.paperbase_api.routes.extraction import (
    default_extraction_client_factory,
    router as extraction_router,
)
from services.paperbase_api.routes.ingest import router as ingest_router
from services.paperbase_api.routes.jobs import router as jobs_router
from services.paperbase_api.routes.papers import router as papers_router
from services.paperbase_api.routes.search import router as search_router
from services.paperbase_api.routes.ui import STATIC_DIR, router as ui_router
from services.paperbase_api.routes.workspaces import router as workspaces_router


def create_app(
    *,
    session_factory: sessionmaker[Session] | None = None,
    extraction_client_factory: Callable[[], object] | None = None,
    search_backend: object | None = None,
) -> FastAPI:
    """Create the Paperbase API application."""

    configure_logging_from_env()
    resolved_session_factory = session_factory or make_session_factory()
    resolved_extraction_client_factory = (
        extraction_client_factory or default_extraction_client_factory
    )

    app = FastAPI(
        title="Paperbase API",
        summary="Local-first corpus API for papers, fulltext, figures, and comparisons.",
        description=(
            "Internal platform API for querying the Paperbase corpus. "
            "This service exposes canonical paper records, parsed fulltext, "
            "figures, structured result comparison slices, curated collections, "
            "and user annotations."
        ),
        version="0.1.0",
    )
    app.state.session_factory = resolved_session_factory
    app.state.extraction_client_factory = resolved_extraction_client_factory
    app.state.search_backend = search_backend

    register_exception_handlers(app)
    app.mount("/ui", StaticFiles(directory=Path(STATIC_DIR)), name="paperbase-ui")

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        return HealthResponse()

    app.include_router(search_router)
    app.include_router(papers_router)
    app.include_router(compare_router)
    app.include_router(collections_router)
    app.include_router(workspaces_router)
    app.include_router(ingest_router)
    app.include_router(extraction_router)
    app.include_router(jobs_router)
    app.include_router(ui_router)
    return app
