"""Dependency helpers for the Paperbase API service."""

from __future__ import annotations

from collections.abc import Generator

from fastapi import Request
from sqlalchemy.orm import Session, sessionmaker

from paperbase.projects import ProjectNotFoundError
from services.paperbase_api.errors import PaperbaseAPIError

PROJECT_HEADER = "X-Arxie-Project-Id"


def get_project_id(request: Request) -> str | None:
    return request.headers.get(PROJECT_HEADER)


def get_session_factory(request: Request) -> sessionmaker[Session]:
    project_id = get_project_id(request)
    if project_id:
        registry = getattr(request.app.state, "project_registry", None)
        if registry is None:
            raise PaperbaseAPIError(
                status_code=503,
                error="project_registry_unavailable",
                message="Project registry is not configured.",
            )
        try:
            return registry.session_factory_for(project_id)
        except ProjectNotFoundError as exc:
            raise PaperbaseAPIError(
                status_code=404,
                error="project_not_found",
                message=f"No project found for id: {project_id}",
                details=[{"project_id": project_id}],
            ) from exc
    return request.app.state.session_factory


def get_session(request: Request) -> Generator[Session, None, None]:
    session_factory = get_session_factory(request)
    with session_factory() as session:
        yield session
