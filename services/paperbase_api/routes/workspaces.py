"""Workspace routes for the Paperbase API service."""

from __future__ import annotations

from pathlib import Path as FilePath

from fastapi import APIRouter, Depends, Path, Query, Response, status
from sqlalchemy.orm import Session

from paperbase.db.repositories import CollectionRepository, PaperRepository, WorkspaceRepository
from ra.utils.security import sanitize_identifier, sanitize_user_text
from services.paperbase_api.dependencies import get_session
from services.paperbase_api.errors import PaperbaseAPIError
from services.paperbase_api.models import (
    CollectionSummaryResponse,
    PaperSummaryResponse,
    SingleStudySourceResponse,
    SingleWorkspaceResponse,
    StudySourceCreateRequest,
    StudySourceResponse,
    StudySourcesResponse,
    WorkspaceCreateRequest,
    WorkspaceDetailResponse,
    WorkspaceSummaryResponse,
    WorkspaceUpdateRequest,
    WorkspacesResponse,
)

router = APIRouter(tags=["workspaces"])

SOURCE_CONTENT_LIMIT = 20000
SOURCE_SUMMARY_LIMIT = 360


def _paper_to_response(
    paper,  # noqa: ANN001
    *,
    authors: list[str] | None = None,
    tags: list[str] | None = None,
) -> PaperSummaryResponse:
    return PaperSummaryResponse(
        id=paper.id,
        title=paper.canonical_title,
        abstract=paper.abstract,
        publication_year=paper.publication_year,
        venue=paper.venue,
        provider=paper.provider,
        external_id=paper.external_id,
        doi=paper.doi,
        arxiv_id=paper.arxiv_id,
        authors=list(authors or []),
        tags=list(tags or []),
    )


def _collection_to_response(collection) -> CollectionSummaryResponse | None:  # noqa: ANN001
    if collection is None:
        return None
    return CollectionSummaryResponse(
        id=collection.id,
        owner_id=collection.owner_id,
        scope_type=collection.scope_type,
        title=collection.title,
        description=collection.description,
        extraction_profile_id=collection.extraction_profile_id,
        tags=list(collection.tags_json or []),
    )


def _workspace_to_summary_response(workspace) -> WorkspaceSummaryResponse:  # noqa: ANN001
    return WorkspaceSummaryResponse(
        id=workspace.id,
        owner_id=workspace.owner_id,
        title=workspace.title,
        description=workspace.description,
        collection_id=workspace.collection_id,
        saved_query=workspace.saved_query,
        focus_note=workspace.focus_note,
        active_filters=dict(workspace.active_filters_json or {}),
        pinned_paper_ids=list(workspace.pinned_paper_ids_json or []),
    )


def _source_to_response(source) -> StudySourceResponse:  # noqa: ANN001
    return StudySourceResponse(
        id=source.id,
        workspace_id=source.workspace_id,
        source_type=source.source_type,
        title=source.title,
        path=source.path,
        content=source.content,
        summary=source.summary,
        read_status=source.read_status,
        error_message=source.error_message,
    )


def _summarize_source_text(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    if len(cleaned) <= SOURCE_SUMMARY_LIMIT:
        return cleaned
    return f"{cleaned[:SOURCE_SUMMARY_LIMIT].rstrip()}..."


def _read_study_source_payload(payload: StudySourceCreateRequest) -> tuple[str | None, str | None, str, str | None]:
    if payload.source_type == "text":
        if not payload.content:
            raise PaperbaseAPIError(
                status_code=422,
                error="study_source_content_required",
                message="Text study sources require content.",
            )
        content = sanitize_user_text(payload.content, field_name="content", max_length=SOURCE_CONTENT_LIMIT)
        return None, content, "ready", None

    if not payload.path:
        raise PaperbaseAPIError(
            status_code=422,
            error="study_source_path_required",
            message="Path study sources require a path.",
        )

    safe_path = sanitize_user_text(payload.path, field_name="path", max_length=2000)
    source_path = FilePath(safe_path).expanduser()
    try:
        with source_path.open("r", encoding="utf-8") as source_file:
            content = source_file.read(SOURCE_CONTENT_LIMIT)
    except FileNotFoundError:
        return safe_path, None, "error", f"File not found: {safe_path}"
    except IsADirectoryError:
        return safe_path, None, "error", f"Expected a file but found a directory: {safe_path}"
    except UnicodeDecodeError:
        return safe_path, None, "error", f"Could not read text from file: {safe_path}"
    except OSError as exc:
        return safe_path, None, "error", str(exc)
    return safe_path, content, "ready", None


def _resolve_pinned_papers(
    paper_repository: PaperRepository,
    pinned_paper_ids: list[str],
) -> list[PaperSummaryResponse]:
    authors_by_paper_id = paper_repository.list_author_names_by_paper_ids(pinned_paper_ids)
    tags_by_paper_id = paper_repository.list_tags_by_paper_ids(pinned_paper_ids)

    pinned_papers: list[PaperSummaryResponse] = []
    for paper_id in pinned_paper_ids:
        paper = paper_repository.get_by_id(paper_id)
        if paper is None:
            continue
        pinned_papers.append(
            _paper_to_response(
                paper,
                authors=authors_by_paper_id.get(paper_id, []),
                tags=tags_by_paper_id.get(paper_id, []),
            )
        )
    return pinned_papers


@router.get("/api/v1/workspaces", response_model=WorkspacesResponse)
@router.get("/api/v1/studies", response_model=WorkspacesResponse)
def list_workspaces(
    owner_id: str | None = Query(None, min_length=1, max_length=128),
    session: Session = Depends(get_session),
) -> WorkspacesResponse:
    repository = WorkspaceRepository(session)
    safe_owner_id = (
        sanitize_user_text(owner_id, field_name="owner_id", max_length=128)
        if owner_id is not None
        else None
    )
    workspaces = repository.list_workspaces(owner_id=safe_owner_id)
    return WorkspacesResponse(data=[_workspace_to_summary_response(workspace) for workspace in workspaces])


@router.post(
    "/api/v1/workspaces",
    response_model=SingleWorkspaceResponse,
    status_code=status.HTTP_201_CREATED,
)
@router.post(
    "/api/v1/studies",
    response_model=SingleWorkspaceResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_workspace(
    payload: WorkspaceCreateRequest,
    session: Session = Depends(get_session),
) -> SingleWorkspaceResponse:
    workspace_repository = WorkspaceRepository(session)
    collection_repository = CollectionRepository(session)
    paper_repository = PaperRepository(session)

    owner_id = sanitize_user_text(payload.owner_id, field_name="owner_id", max_length=128)
    title = sanitize_user_text(payload.title, field_name="title", max_length=255)
    if workspace_repository.get_by_owner_title(owner_id, title) is not None:
        raise PaperbaseAPIError(
            status_code=409,
            error="workspace_conflict",
            message="A workspace with this title already exists for the owner.",
        )

    safe_collection_id = (
        sanitize_identifier(payload.collection_id, field_name="collection_id", max_length=36)
        if payload.collection_id is not None
        else None
    )
    collection = None
    if safe_collection_id is not None:
        collection = collection_repository.get_by_id(safe_collection_id)
        if collection is None:
            raise PaperbaseAPIError(
                status_code=404,
                error="collection_not_found",
                message=f"No collection found for id: {safe_collection_id}",
            )

    pinned_paper_ids: list[str] = []
    for paper_id in payload.pinned_paper_ids:
        safe_paper_id = sanitize_identifier(paper_id, field_name="pinned_paper_id", max_length=36)
        if paper_repository.get_by_id(safe_paper_id) is None:
            raise PaperbaseAPIError(
                status_code=404,
                error="paper_not_found",
                message=f"No paper found for id: {safe_paper_id}",
            )
        pinned_paper_ids.append(safe_paper_id)

    workspace = workspace_repository.create(
        owner_id=owner_id,
        title=title,
        description=payload.description,
        collection_id=safe_collection_id,
        saved_query=payload.saved_query,
        focus_note=payload.focus_note,
        active_filters=payload.active_filters,
        pinned_paper_ids=pinned_paper_ids,
    )
    return SingleWorkspaceResponse(
        data=WorkspaceDetailResponse(
            **_workspace_to_summary_response(workspace).model_dump(),
            collection=_collection_to_response(collection),
            pinned_papers=_resolve_pinned_papers(paper_repository, pinned_paper_ids),
        )
    )


@router.get("/api/v1/workspaces/{workspace_id}", response_model=SingleWorkspaceResponse)
@router.get("/api/v1/studies/{workspace_id}", response_model=SingleWorkspaceResponse)
def fetch_workspace(
    workspace_id: str = Path(..., min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> SingleWorkspaceResponse:
    workspace_repository = WorkspaceRepository(session)
    collection_repository = CollectionRepository(session)
    paper_repository = PaperRepository(session)
    safe_workspace_id = sanitize_identifier(workspace_id, field_name="workspace_id", max_length=36)
    workspace = workspace_repository.get_by_id(safe_workspace_id)
    if workspace is None:
        raise PaperbaseAPIError(
            status_code=404,
            error="workspace_not_found",
            message=f"No workspace found for id: {safe_workspace_id}",
        )

    collection = (
        collection_repository.get_by_id(workspace.collection_id)
        if workspace.collection_id is not None
        else None
    )
    pinned_paper_ids = list(workspace.pinned_paper_ids_json or [])
    return SingleWorkspaceResponse(
        data=WorkspaceDetailResponse(
            **_workspace_to_summary_response(workspace).model_dump(),
            collection=_collection_to_response(collection),
            pinned_papers=_resolve_pinned_papers(paper_repository, pinned_paper_ids),
        )
    )


@router.patch("/api/v1/workspaces/{workspace_id}", response_model=SingleWorkspaceResponse)
@router.patch("/api/v1/studies/{workspace_id}", response_model=SingleWorkspaceResponse)
def update_workspace(
    payload: WorkspaceUpdateRequest,
    workspace_id: str = Path(..., min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> SingleWorkspaceResponse:
    workspace_repository = WorkspaceRepository(session)
    collection_repository = CollectionRepository(session)
    paper_repository = PaperRepository(session)
    safe_workspace_id = sanitize_identifier(workspace_id, field_name="workspace_id", max_length=36)
    workspace = workspace_repository.get_by_id(safe_workspace_id)
    if workspace is None:
        raise PaperbaseAPIError(
            status_code=404,
            error="workspace_not_found",
            message=f"No workspace found for id: {safe_workspace_id}",
        )

    title = workspace.title
    if "title" in payload.model_fields_set and payload.title is not None:
        title = sanitize_user_text(payload.title, field_name="title", max_length=255)
        existing = workspace_repository.get_by_owner_title(workspace.owner_id, title)
        if existing is not None and existing.id != workspace.id:
            raise PaperbaseAPIError(
                status_code=409,
                error="workspace_conflict",
                message="A workspace with this title already exists for the owner.",
            )

    collection_id = workspace.collection_id
    if "collection_id" in payload.model_fields_set:
        collection_id = (
            sanitize_identifier(payload.collection_id, field_name="collection_id", max_length=36)
            if payload.collection_id is not None
            else None
        )
        if collection_id is not None and collection_repository.get_by_id(collection_id) is None:
            raise PaperbaseAPIError(
                status_code=404,
                error="collection_not_found",
                message=f"No collection found for id: {collection_id}",
            )

    pinned_paper_ids = list(workspace.pinned_paper_ids_json or [])
    if "pinned_paper_ids" in payload.model_fields_set and payload.pinned_paper_ids is not None:
        pinned_paper_ids = []
        for paper_id in payload.pinned_paper_ids:
            safe_paper_id = sanitize_identifier(paper_id, field_name="pinned_paper_id", max_length=36)
            if paper_repository.get_by_id(safe_paper_id) is None:
                raise PaperbaseAPIError(
                    status_code=404,
                    error="paper_not_found",
                    message=f"No paper found for id: {safe_paper_id}",
                )
            pinned_paper_ids.append(safe_paper_id)

    updated = workspace_repository.update(
        safe_workspace_id,
        title=title,
        description=payload.description if "description" in payload.model_fields_set else workspace.description,
        collection_id=collection_id,
        saved_query=payload.saved_query if "saved_query" in payload.model_fields_set else workspace.saved_query,
        focus_note=payload.focus_note if "focus_note" in payload.model_fields_set else workspace.focus_note,
        active_filters=(
            payload.active_filters
            if "active_filters" in payload.model_fields_set
            else dict(workspace.active_filters_json or {})
        ),
        pinned_paper_ids=pinned_paper_ids,
    )
    collection = collection_repository.get_by_id(updated.collection_id) if updated.collection_id else None
    return SingleWorkspaceResponse(
        data=WorkspaceDetailResponse(
            **_workspace_to_summary_response(updated).model_dump(),
            collection=_collection_to_response(collection),
            pinned_papers=_resolve_pinned_papers(paper_repository, pinned_paper_ids),
        )
    )


@router.get("/api/v1/studies/{workspace_id}/sources", response_model=StudySourcesResponse)
def list_study_sources(
    workspace_id: str = Path(..., min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> StudySourcesResponse:
    repository = WorkspaceRepository(session)
    safe_workspace_id = sanitize_identifier(workspace_id, field_name="workspace_id", max_length=36)
    if repository.get_by_id(safe_workspace_id) is None:
        raise PaperbaseAPIError(
            status_code=404,
            error="workspace_not_found",
            message=f"No workspace found for id: {safe_workspace_id}",
        )
    return StudySourcesResponse(
        data=[_source_to_response(source) for source in repository.list_sources(workspace_id=safe_workspace_id)]
    )


@router.post(
    "/api/v1/studies/{workspace_id}/sources",
    response_model=SingleStudySourceResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_study_source(
    payload: StudySourceCreateRequest,
    workspace_id: str = Path(..., min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> SingleStudySourceResponse:
    repository = WorkspaceRepository(session)
    safe_workspace_id = sanitize_identifier(workspace_id, field_name="workspace_id", max_length=36)
    if repository.get_by_id(safe_workspace_id) is None:
        raise PaperbaseAPIError(
            status_code=404,
            error="workspace_not_found",
            message=f"No workspace found for id: {safe_workspace_id}",
        )

    title = sanitize_user_text(payload.title, field_name="title", max_length=255)
    path, content, read_status, error_message = _read_study_source_payload(payload)
    source = repository.create_source(
        workspace_id=safe_workspace_id,
        source_type=payload.source_type,
        title=title,
        path=path,
        content=content,
        summary=_summarize_source_text(content) if content else None,
        read_status=read_status,
        error_message=error_message,
    )
    return SingleStudySourceResponse(data=_source_to_response(source))


@router.delete("/api/v1/studies/{workspace_id}/sources/{source_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_study_source(
    workspace_id: str = Path(..., min_length=1, max_length=36),
    source_id: str = Path(..., min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> Response:
    repository = WorkspaceRepository(session)
    safe_workspace_id = sanitize_identifier(workspace_id, field_name="workspace_id", max_length=36)
    safe_source_id = sanitize_identifier(source_id, field_name="source_id", max_length=36)
    source = repository.get_source(safe_source_id)
    if source is None or source.workspace_id != safe_workspace_id:
        raise PaperbaseAPIError(
            status_code=404,
            error="study_source_not_found",
            message=f"No study source found for id: {safe_source_id}",
        )
    repository.delete_source(safe_source_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
