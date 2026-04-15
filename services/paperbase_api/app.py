"""FastAPI app for the local-first Paperbase API service."""

from __future__ import annotations

from fastapi import FastAPI
from sqlalchemy.orm import Session, sessionmaker

from paperbase.db.session import make_session_factory
from ra.utils.logging_config import configure_logging_from_env
from services.paperbase_api.errors import register_exception_handlers
from services.paperbase_api.models import HealthResponse
from services.paperbase_api.routes.compare import router as compare_router
from services.paperbase_api.routes.papers import router as papers_router
from services.paperbase_api.routes.search import router as search_router


def create_app(
    *,
    session_factory: sessionmaker[Session] | None = None,
) -> FastAPI:
    """Create the Paperbase API application."""

    configure_logging_from_env()
    resolved_session_factory = session_factory or make_session_factory()

    app = FastAPI(
        title="Paperbase API",
        summary="Local-first corpus API for papers, fulltext, figures, and comparisons.",
        description=(
            "Internal platform API for querying the Paperbase corpus. "
            "This service exposes canonical paper records, parsed fulltext, "
            "figures, and structured result comparison slices."
        ),
        version="0.1.0",
    )
    app.state.session_factory = resolved_session_factory

    register_exception_handlers(app)

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        return HealthResponse()

    app.include_router(search_router)
    app.include_router(papers_router)
    app.include_router(compare_router)
    return app
