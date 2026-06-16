"""FastAPI app for the local-first Paperbase API service."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session, sessionmaker

from paperbase.config import load_paperbase_config
from paperbase.db.bootstrap import ensure_database_schema_compatible, initialize_database
from paperbase.db.session import make_session_factory
from paperbase.projects import ProjectRegistry
from paperbase.research.model_client import default_research_model_client
from paperbase.version import get_version
from ra.utils.logging_config import configure_logging_from_env
from services.paperbase_api.errors import register_exception_handlers
from services.paperbase_api.health import DependencyChecker
from services.paperbase_api.models import (
    DependencyStatusResponse,
    HealthResponse,
    ReadinessResponse,
)
from services.paperbase_api.routes import ui as ui_routes
from services.paperbase_api.routes.collections import router as collections_router
from services.paperbase_api.routes.compare import router as compare_router
from services.paperbase_api.routes.extraction import (
    default_extraction_client_factory,
)
from services.paperbase_api.routes.extraction import (
    router as extraction_router,
)
from services.paperbase_api.routes.ingest import router as ingest_router
from services.paperbase_api.routes.jobs import router as jobs_router
from services.paperbase_api.routes.papers import router as papers_router
from services.paperbase_api.routes.projects import router as projects_router
from services.paperbase_api.routes.research import router as research_router
from services.paperbase_api.routes.runtime import router as runtime_router
from services.paperbase_api.routes.search import router as search_router
from services.paperbase_api.routes.workspaces import router as workspaces_router
from services.paperbase_api.security import configure_hosted_request_security


def create_app(
    *,
    session_factory: sessionmaker[Session] | None = None,
    extraction_client_factory: Callable[[], object] | None = None,
    research_model_client_factory: Callable[[], object | None] | None = None,
    search_backend: object | None = None,
    job_dispatcher: object | None = None,
    embedding_provider: object | None = None,
    dependency_checker: object | None = None,
    project_registry: ProjectRegistry | None = None,
) -> FastAPI:
    """Create the Paperbase API application."""

    configure_logging_from_env()
    config = load_paperbase_config()
    if session_factory is None:
        initialize_database(config.database_url)
        resolved_session_factory = make_session_factory(config.database_url)
    else:
        resolved_session_factory = session_factory
        bind = resolved_session_factory.kw.get("bind")
        if bind is not None:
            ensure_database_schema_compatible(bind)
    resolved_extraction_client_factory = (
        extraction_client_factory or default_extraction_client_factory
    )
    resolved_dependency_checker = dependency_checker or DependencyChecker(
        session_factory=resolved_session_factory,
        config=config,
        search_backend=search_backend,
    )
    resolved_project_registry = project_registry or ProjectRegistry(
        registry_path=config.project_registry_path
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
    app.state.research_model_client_factory = (
        research_model_client_factory or default_research_model_client
    )
    app.state.search_backend = search_backend
    app.state.job_dispatcher = job_dispatcher
    app.state.embedding_provider = embedding_provider
    app.state.dependency_checker = resolved_dependency_checker
    app.state.project_registry = resolved_project_registry
    app.state.paperbase_config = config
    app.state.upload_staging_dir = Path(config.upload_staging_dir)
    app.state.upload_max_file_count = config.upload_max_file_count
    app.state.upload_max_single_file_bytes = config.upload_max_single_file_bytes
    app.state.upload_max_total_bytes = config.upload_max_total_bytes

    register_exception_handlers(app)
    configure_hosted_request_security(app, config=config)
    react_assets_dir = Path(ui_routes.REACT_APP_DIR) / "assets"
    if react_assets_dir.exists():
        app.mount(
            "/app/assets",
            StaticFiles(directory=react_assets_dir),
            name="paperbase-react-assets",
        )
    if not config.hosted_mode:
        app.mount("/ui", StaticFiles(directory=Path(ui_routes.STATIC_DIR)), name="paperbase-ui")

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
    app.include_router(projects_router)
    app.include_router(compare_router)
    app.include_router(collections_router)
    app.include_router(workspaces_router)
    app.include_router(research_router)
    app.include_router(ingest_router)
    app.include_router(extraction_router)
    app.include_router(jobs_router)
    app.include_router(runtime_router)
    app.include_router(ui_routes.router)
    return app
