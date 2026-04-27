"""FastAPI app for the local-first Paperbase API service."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session, sessionmaker

from paperbase.config import load_paperbase_config
from paperbase.db.session import make_session_factory
from paperbase.version import get_version
from ra.utils.logging_config import configure_logging_from_env
from services.paperbase_api.errors import register_exception_handlers
from services.paperbase_api.health import DependencyChecker
from services.paperbase_api.models import DependencyStatusResponse, HealthResponse, ReadinessResponse
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
    job_dispatcher: object | None = None,
    embedding_provider: object | None = None,
    dependency_checker: object | None = None,
) -> FastAPI:
    """Create the Paperbase API application."""

    configure_logging_from_env()
    config = load_paperbase_config()
    resolved_session_factory = session_factory or make_session_factory()
    resolved_extraction_client_factory = (
        extraction_client_factory or default_extraction_client_factory
    )
    resolved_dependency_checker = dependency_checker or DependencyChecker(
        session_factory=resolved_session_factory,
        config=config,
        search_backend=search_backend,
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
        version=get_version(),
    )
    app.state.session_factory = resolved_session_factory
    app.state.extraction_client_factory = resolved_extraction_client_factory
    app.state.search_backend = search_backend
    app.state.job_dispatcher = job_dispatcher
    app.state.embedding_provider = embedding_provider
    app.state.dependency_checker = resolved_dependency_checker

    register_exception_handlers(app)
    app.mount("/ui", StaticFiles(directory=Path(STATIC_DIR)), name="paperbase-ui")

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        return HealthResponse()

    @app.get("/livez", response_model=HealthResponse, tags=["system"])
    def livez() -> HealthResponse:
        return HealthResponse()

    @app.get("/readyz", response_model=ReadinessResponse, tags=["system"])
    def readyz() -> JSONResponse | ReadinessResponse:
        report = app.state.dependency_checker.check()
        payload = ReadinessResponse(
            status="ready" if report.ready else "not_ready",
            dependencies=[
                DependencyStatusResponse(
                    name=item.name,
                    ok=item.ok,
                    detail=item.detail,
                    required=item.required,
                )
                for item in report.dependencies
            ],
        )
        if report.ready:
            return payload
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=payload.model_dump(),
        )

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
