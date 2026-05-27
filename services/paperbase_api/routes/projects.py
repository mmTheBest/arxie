"""Project routes for opening isolated local Arxie library folders."""

from __future__ import annotations

from fastapi import APIRouter, Request, status

from paperbase.projects import ProjectSummary
from ra.utils.security import sanitize_user_text
from services.paperbase_api.errors import PaperbaseAPIError
from services.paperbase_api.models import (
    ProjectOpenRequest,
    ProjectResponse,
    ProjectsResponse,
    SingleProjectResponse,
)
from services.paperbase_api.path_policy import ensure_host_path_allowed

router = APIRouter(tags=["projects"])


def _project_to_response(project: ProjectSummary) -> ProjectResponse:
    return ProjectResponse(**project.to_dict())


def _registry(request: Request):
    registry = getattr(request.app.state, "project_registry", None)
    if registry is None:
        raise PaperbaseAPIError(
            status_code=503,
            error="project_registry_unavailable",
            message="Project registry is not configured.",
        )
    return registry


@router.get("/api/v1/projects", response_model=ProjectsResponse)
def list_projects(request: Request) -> ProjectsResponse:
    return ProjectsResponse(
        data=[_project_to_response(project) for project in _registry(request).list_projects()]
    )


@router.post(
    "/api/v1/projects/open",
    response_model=SingleProjectResponse,
    status_code=status.HTTP_201_CREATED,
)
def open_project(payload: ProjectOpenRequest, request: Request) -> SingleProjectResponse:
    root_path = sanitize_user_text(payload.root_path, field_name="root_path", max_length=4096)
    safe_root_path = ensure_host_path_allowed(
        root_path,
        config=request.app.state.paperbase_config,
        field_name="root_path",
    )
    title = (
        sanitize_user_text(payload.title, field_name="title", max_length=255)
        if payload.title is not None
        else None
    )
    project = _registry(request).open_project(root_path=safe_root_path, title=title)
    return SingleProjectResponse(data=_project_to_response(project))
