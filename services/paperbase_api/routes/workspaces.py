"""Workspace routes for the Paperbase API service."""

from __future__ import annotations

from datetime import UTC
from pathlib import Path as FilePath
from typing import Any

from fastapi import APIRouter, Depends, Path, Query, Request, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from paperbase.db.models import ResearchArtifact, ResearchThread
from paperbase.db.repositories import (
    CollectionRepository,
    PaperRepository,
    ResearchIntelligenceRepository,
    WorkspaceRepository,
)
from paperbase.source_safety import detect_source_secret
from ra.utils.security import sanitize_identifier, sanitize_user_text
from services.paperbase_api.dependencies import get_session
from services.paperbase_api.errors import PaperbaseAPIError
from services.paperbase_api.models import (
    ArtifactFileEntryResponse,
    ArtifactFolderListingResponse,
    CollectionSummaryResponse,
    PaperSummaryResponse,
    SingleArtifactFolderListingResponse,
    SingleStudyBriefProposalResponse,
    SingleStudyBriefResponse,
    SingleStudySourceResponse,
    SingleWorkspaceResponse,
    StudyBriefPayload,
    StudyBriefProposalAcceptRequest,
    StudyBriefProposalChange,
    StudyBriefProposalResponse,
    StudyBriefResponse,
    StudyBriefUpdateRequest,
    StudySourceCreateRequest,
    StudySourceResponse,
    StudySourcesResponse,
    WorkspaceCreateRequest,
    WorkspaceDetailResponse,
    WorkspacesResponse,
    WorkspaceSummaryResponse,
    WorkspaceUpdateRequest,
)
from services.paperbase_api.path_policy import ensure_host_path_allowed

router = APIRouter(tags=["workspaces"])

SOURCE_CONTENT_LIMIT = 20000
SOURCE_SUMMARY_LIMIT = 360
ARTIFACT_BROWSER_ENTRY_LIMIT = 200
SOURCE_CONTENT_MAX_BYTES = 1 * 1024 * 1024
ARTIFACT_SOURCE_FILE_MAX_BYTES = SOURCE_CONTENT_MAX_BYTES
ARTIFACT_IGNORED_NAMES = {
    ".arxie",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "coverage",
    "dist",
    "node_modules",
}
ARTIFACT_SOURCE_SUFFIXES = {
    ".csv": "results_path",
    ".ipynb": "code_path",
    ".js": "code_path",
    ".json": "results_path",
    ".jsonl": "results_path",
    ".jsx": "code_path",
    ".md": "draft_path",
    ".markdown": "draft_path",
    ".py": "code_path",
    ".r": "code_path",
    ".rst": "draft_path",
    ".sh": "code_path",
    ".sql": "code_path",
    ".tex": "draft_path",
    ".ts": "code_path",
    ".tsx": "code_path",
    ".tsv": "results_path",
    ".txt": "draft_path",
    ".yaml": "results_path",
    ".yml": "results_path",
}
SECRET_SOURCE_ERROR_MESSAGE = (
    "Secret-like value detected; source content was not stored."
)


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


def _serialize_timestamp(value) -> str | None:  # noqa: ANN001
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    else:
        value = value.astimezone(UTC)
    return value.isoformat()


def _empty_study_brief_payload() -> StudyBriefPayload:
    return StudyBriefPayload(
        aim="",
        hypothesis="",
        constraints=[],
        confirmed_decisions=[],
        open_risks=[],
        linked_source_ids=[],
    )


def _bounded_brief_text(value: Any, *, max_length: int) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text[:max_length]


def _study_brief_items(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []

    items: list[dict[str, str]] = []
    for index, item in enumerate(value[:50], start=1):
        if isinstance(item, dict):
            title = _bounded_brief_text(
                item.get("title")
                or item.get("label")
                or item.get("name")
                or f"Item {index}",
                max_length=255,
            )
            text = _bounded_brief_text(
                item.get("text")
                or item.get("summary")
                or item.get("description")
                or item.get("decision")
                or item.get("risk")
                or title,
                max_length=5000,
            )
        else:
            title = f"Item {index}"
            text = _bounded_brief_text(item, max_length=5000)
            separator_index = text.find(":")
            if separator_index > 0:
                title = _bounded_brief_text(text[:separator_index], max_length=255) or title
                text = _bounded_brief_text(
                    text[separator_index + 1 :],
                    max_length=5000,
                )

        if text:
            items.append({"title": title or f"Item {index}", "text": text})
    return items


def _study_brief_source_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        source_id
        for source_id in (_bounded_brief_text(item, max_length=36) for item in value[:200])
        if source_id
    ]


def _study_brief_payload_from_json(payload: dict[str, Any]) -> StudyBriefPayload:
    return StudyBriefPayload(
        aim=_bounded_brief_text(payload.get("aim"), max_length=10000),
        hypothesis=_bounded_brief_text(payload.get("hypothesis"), max_length=10000),
        constraints=_study_brief_items(payload.get("constraints")),
        confirmed_decisions=_study_brief_items(
            payload.get("confirmed_decisions", payload.get("decisions"))
        ),
        open_risks=_study_brief_items(payload.get("open_risks", payload.get("risks"))),
        linked_source_ids=_study_brief_source_ids(payload.get("linked_source_ids")),
    )


def _validated_study_brief_payload(
    *,
    brief: StudyBriefPayload,
    workspace_id: str,
    workspace_repository: WorkspaceRepository,
) -> dict[str, Any]:
    brief_json = brief.model_dump(mode="json")
    linked_source_ids: list[str] = []
    for source_id in brief.linked_source_ids:
        try:
            linked_source_ids.append(
                sanitize_identifier(
                    source_id,
                    field_name="linked_source_id",
                    max_length=36,
                )
            )
        except ValueError as exc:
            raise PaperbaseAPIError(
                status_code=422,
                error="validation_error",
                message=str(exc),
                details=[{"field": "linked_source_ids", "source_id": source_id}],
            ) from exc

    owned_source_ids = {
        source.id for source in workspace_repository.list_sources(workspace_id=workspace_id)
    }
    invalid_source_ids = [
        source_id for source_id in linked_source_ids if source_id not in owned_source_ids
    ]
    if invalid_source_ids:
        raise PaperbaseAPIError(
            status_code=400,
            error="invalid_study_brief_sources",
            message="Study Brief linked sources must belong to the target Study.",
            details=[
                {"field": "linked_source_ids", "source_id": source_id}
                for source_id in invalid_source_ids
            ],
        )

    brief_json["linked_source_ids"] = linked_source_ids
    return brief_json


def _study_brief_to_response(brief, *, workspace_id: str) -> StudyBriefResponse:  # noqa: ANN001
    if brief is None:
        return StudyBriefResponse(
            id=None,
            workspace_id=workspace_id,
            brief=_empty_study_brief_payload(),
            version=0,
            updated_by=None,
            created_at=None,
            updated_at=None,
        )
    return StudyBriefResponse(
        id=brief.id,
        workspace_id=brief.workspace_id,
        brief=_study_brief_payload_from_json(dict(brief.brief_json or {})),
        version=brief.version,
        updated_by=brief.updated_by,
        created_at=_serialize_timestamp(brief.created_at),
        updated_at=_serialize_timestamp(brief.updated_at),
    )


_STUDY_BRIEF_PROPOSAL_FIELDS = (
    "aim",
    "hypothesis",
    "constraints",
    "confirmed_decisions",
    "open_risks",
    "linked_source_ids",
)


def _current_study_brief_payload(
    repository: ResearchIntelligenceRepository,
    *,
    workspace_id: str,
) -> tuple[StudyBriefPayload, int]:
    brief = repository.get_study_brief(workspace_id)
    response = _study_brief_to_response(brief, workspace_id=workspace_id)
    return response.brief, response.version


def _artifact_study_brief_update(artifact: ResearchArtifact) -> dict[str, Any]:
    output_payload = dict(artifact.output_payload_json or {})
    raw_update = output_payload.get("study_brief_update")
    if not isinstance(raw_update, dict):
        raw_update = output_payload.get("study_brief_updates")
    if not isinstance(raw_update, dict):
        return {}

    update = dict(raw_update)
    if "confirmed_decisions" not in update and "decisions" in update:
        update["confirmed_decisions"] = update["decisions"]
    if "open_risks" not in update and "risks" in update:
        update["open_risks"] = update["risks"]
    return {field: update[field] for field in _STUDY_BRIEF_PROPOSAL_FIELDS if field in update}


def _study_brief_proposal_changes(
    *,
    before: StudyBriefPayload,
    after: StudyBriefPayload,
) -> list[StudyBriefProposalChange]:
    before_json = before.model_dump(mode="json")
    after_json = after.model_dump(mode="json")
    return [
        StudyBriefProposalChange(
            field=field,
            before=before_json.get(field),
            after=after_json.get(field),
        )
        for field in _STUDY_BRIEF_PROPOSAL_FIELDS
        if before_json.get(field) != after_json.get(field)
    ]


def _ensure_study_brief_artifact(
    session: Session,
    *,
    workspace_id: str,
    artifact_id: str,
) -> ResearchArtifact:
    safe_artifact_id = sanitize_identifier(
        artifact_id,
        field_name="artifact_id",
        max_length=36,
    )
    artifact = session.get(ResearchArtifact, safe_artifact_id)
    if artifact is None:
        raise PaperbaseAPIError(
            status_code=404,
            error="research_artifact_not_found",
            message=f"No research artifact found for id: {safe_artifact_id}",
        )
    if artifact.status != "completed":
        raise PaperbaseAPIError(
            status_code=409,
            error="research_artifact_not_ready",
            message="Study Brief proposals require a completed research artifact.",
        )
    thread_workspace_id = None
    if artifact.thread_id:
        thread_workspace_id = session.execute(
            select(ResearchThread.workspace_id).where(ResearchThread.id == artifact.thread_id)
        ).scalar_one_or_none()
    if thread_workspace_id != workspace_id:
        raise PaperbaseAPIError(
            status_code=400,
            error="artifact_study_mismatch",
            message="Research artifact must belong to the target Study.",
        )
    return artifact


def _build_study_brief_proposal(
    *,
    artifact: ResearchArtifact,
    current_brief: StudyBriefPayload,
    workspace_id: str,
    workspace_repository: WorkspaceRepository,
) -> StudyBriefPayload:
    update = _artifact_study_brief_update(artifact)
    if not update:
        raise PaperbaseAPIError(
            status_code=400,
            error="study_brief_proposal_not_found",
            message="Research artifact does not include a Study Brief update proposal.",
        )

    proposed_json = current_brief.model_dump(mode="json")
    proposed_json.update(update)
    proposed_brief = _study_brief_payload_from_json(proposed_json)
    validated_json = _validated_study_brief_payload(
        brief=proposed_brief,
        workspace_id=workspace_id,
        workspace_repository=workspace_repository,
    )
    return _study_brief_payload_from_json(validated_json)


def _study_brief_proposal_response(
    *,
    workspace_id: str,
    artifact: ResearchArtifact,
    repository: ResearchIntelligenceRepository,
    workspace_repository: WorkspaceRepository,
) -> StudyBriefProposalResponse:
    current_brief, current_version = _current_study_brief_payload(
        repository,
        workspace_id=workspace_id,
    )
    proposed_brief = _build_study_brief_proposal(
        artifact=artifact,
        current_brief=current_brief,
        workspace_id=workspace_id,
        workspace_repository=workspace_repository,
    )
    return StudyBriefProposalResponse(
        workspace_id=workspace_id,
        artifact_id=artifact.id,
        artifact_type=artifact.artifact_type,
        artifact_title=artifact.title,
        current_version=current_version,
        proposed_brief=proposed_brief,
        changes=_study_brief_proposal_changes(
            before=current_brief,
            after=proposed_brief,
        ),
    )


def _source_is_stale(source, *, config) -> bool:  # noqa: ANN001
    if not source.path or source.source_size_bytes is None or source.source_mtime_ns is None:
        return False
    source_path = FilePath(source.path).expanduser()
    if not source_path.is_absolute():
        return False
    try:
        resolved_path = ensure_host_path_allowed(
            source_path,
            config=config,
            field_name="path",
        )
        current_stat = resolved_path.stat()
    except (FileNotFoundError, OSError, PaperbaseAPIError):
        return True
    return (
        current_stat.st_size != source.source_size_bytes
        or current_stat.st_mtime_ns != source.source_mtime_ns
    )


def _source_to_response(source, *, config=None) -> StudySourceResponse:  # noqa: ANN001
    is_stale = _source_is_stale(source, config=config) if config is not None else False
    return StudySourceResponse(
        id=source.id,
        workspace_id=source.workspace_id,
        source_type=source.source_type,
        title=source.title,
        path=source.path,
        content=source.content,
        summary=source.summary,
        read_status="stale" if is_stale and source.read_status == "ready" else source.read_status,
        error_message=source.error_message,
        source_size_bytes=source.source_size_bytes,
        source_mtime_ns=source.source_mtime_ns,
        is_stale=is_stale,
    )


def _summarize_source_text(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    if len(cleaned) <= SOURCE_SUMMARY_LIMIT:
        return cleaned
    return f"{cleaned[:SOURCE_SUMMARY_LIMIT].rstrip()}..."


def _read_study_source_payload(
    payload: StudySourceCreateRequest,
    *,
    config,
) -> tuple[str | None, str | None, str, str | None, int | None, int | None]:
    if payload.source_type == "text":
        if not payload.content:
            raise PaperbaseAPIError(
                status_code=422,
                error="study_source_content_required",
                message="Text study sources require content.",
            )
        content, source_size_bytes = _prepare_source_content_snapshot(payload.content)
        return None, content, "ready", None, source_size_bytes, None

    if payload.content:
        content, source_size_bytes = _prepare_source_content_snapshot(payload.content)
        display_path = (
            sanitize_user_text(payload.path, field_name="path", max_length=2000)
            if payload.path
            else None
        )
        if display_path:
            display_path_value = FilePath(display_path)
            if display_path_value.is_absolute() or ".." in display_path_value.parts:
                raise PaperbaseAPIError(
                    status_code=422,
                    error="study_source_path_invalid",
                    message="Browser-selected file paths must be relative display paths.",
                )
            if _source_type_for_artifact_file(display_path_value) is None:
                raise PaperbaseAPIError(
                    status_code=422,
                    error="study_source_file_type_unsupported",
                    message="Study source path must use a supported text-like file type.",
                )
        return display_path, content, "ready", None, source_size_bytes, None

    if not payload.path:
        raise PaperbaseAPIError(
            status_code=422,
            error="study_source_path_required",
            message="Path study sources require a path.",
        )

    safe_path = sanitize_user_text(payload.path, field_name="path", max_length=2000)
    requested_path = FilePath(safe_path).expanduser()
    if requested_path.is_symlink():
        return (
            safe_path,
            None,
            "error",
            "Symlinked study source files are not supported.",
            None,
            None,
        )
    source_path = ensure_host_path_allowed(
        requested_path,
        config=config,
        field_name="path",
    )
    if _source_type_for_artifact_file(source_path) is None:
        raise PaperbaseAPIError(
            status_code=422,
            error="study_source_file_type_unsupported",
            message="Study source path must use a supported text-like file type.",
        )
    try:
        source_stat = source_path.stat()
        if source_stat.st_size > ARTIFACT_SOURCE_FILE_MAX_BYTES:
            return (
                str(source_path),
                None,
                "error",
                "File exceeds the My Artifacts single-file limit.",
                source_stat.st_size,
                source_stat.st_mtime_ns,
            )
        with source_path.open("r", encoding="utf-8") as source_file:
            source_text = source_file.read()
    except FileNotFoundError:
        return str(source_path), None, "error", f"File not found: {source_path}", None, None
    except IsADirectoryError:
        return (
            str(source_path),
            None,
            "error",
            f"Expected a file but found a directory: {source_path}",
            None,
            None,
        )
    except UnicodeDecodeError:
        return (
            str(source_path),
            None,
            "error",
            f"Could not read text from file: {source_path}",
            None,
            None,
        )
    except OSError as exc:
        return str(source_path), None, "error", str(exc), None, None
    if detect_source_secret(source_text) is not None:
        return (
            str(source_path),
            None,
            "error",
            SECRET_SOURCE_ERROR_MESSAGE,
            source_stat.st_size,
            source_stat.st_mtime_ns,
        )
    content = source_text[:SOURCE_CONTENT_LIMIT]
    return str(source_path), content, "ready", None, source_stat.st_size, source_stat.st_mtime_ns


def _prepare_source_content_snapshot(raw_content: str) -> tuple[str, int]:
    source_size_bytes = len(raw_content.encode("utf-8"))
    if source_size_bytes > SOURCE_CONTENT_MAX_BYTES:
        raise PaperbaseAPIError(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            error="study_source_content_too_large",
            message=(
                "Study source content exceeds the single-source limit "
                f"({SOURCE_CONTENT_MAX_BYTES} bytes)."
            ),
        )
    _raise_if_source_secret_detected(raw_content)
    content = sanitize_user_text(
        raw_content[:SOURCE_CONTENT_LIMIT],
        field_name="content",
        max_length=SOURCE_CONTENT_LIMIT,
    )
    return content, source_size_bytes


def _raise_if_source_secret_detected(content: str) -> None:
    if detect_source_secret(content) is None:
        return
    raise PaperbaseAPIError(
        status_code=422,
        error="study_source_secret_detected",
        message=SECRET_SOURCE_ERROR_MESSAGE,
    )


def _source_type_for_artifact_file(path: FilePath) -> str | None:
    return ARTIFACT_SOURCE_SUFFIXES.get(path.suffix.lower())


def _artifact_entry_sort_key(child: FilePath) -> tuple[int, str]:
    try:
        is_directory = False if child.is_symlink() else child.is_dir()
    except OSError:
        is_directory = False
    return (0 if is_directory else 1, child.name.lower())


def _resolve_artifact_folder(
    *,
    root_path: str,
    relative_path: str,
    config,
) -> tuple[FilePath, FilePath, str]:
    safe_root_path = sanitize_user_text(root_path, field_name="root_path", max_length=4096)
    root = ensure_host_path_allowed(
        safe_root_path,
        config=config,
        field_name="root_path",
    )
    if not root.exists():
        raise PaperbaseAPIError(
            status_code=404,
            error="artifact_root_not_found",
            message="No My Artifacts folder found at the requested path.",
        )
    if not root.is_dir():
        raise PaperbaseAPIError(
            status_code=400,
            error="invalid_input",
            message="root_path must point to a directory.",
        )

    safe_relative_path = sanitize_user_text(
        relative_path,
        field_name="relative_path",
        max_length=2000,
        allow_empty=True,
    )
    requested_relative_path = FilePath(safe_relative_path or ".")
    if requested_relative_path.is_absolute():
        raise PaperbaseAPIError(
            status_code=403,
            error="path_not_allowed",
            message="relative_path must stay under the My Artifacts folder.",
        )
    folder = (root / requested_relative_path).resolve()
    if folder != root and root not in folder.parents:
        raise PaperbaseAPIError(
            status_code=403,
            error="path_not_allowed",
            message="relative_path must stay under the My Artifacts folder.",
        )
    if not folder.exists():
        raise PaperbaseAPIError(
            status_code=404,
            error="artifact_folder_not_found",
            message="No My Artifacts folder found at the requested relative path.",
        )
    if not folder.is_dir():
        raise PaperbaseAPIError(
            status_code=400,
            error="invalid_input",
            message="relative_path must point to a directory.",
        )
    display_relative_path = "" if folder == root else folder.relative_to(root).as_posix()
    return root, folder, display_relative_path


def _artifact_folder_listing(
    *,
    root: FilePath,
    folder: FilePath,
    relative_path: str,
) -> ArtifactFolderListingResponse:
    entries: list[ArtifactFileEntryResponse] = []
    ignored_count = 0
    truncated = False
    children = sorted(folder.iterdir(), key=_artifact_entry_sort_key)
    for child in children:
        if len(entries) >= ARTIFACT_BROWSER_ENTRY_LIMIT:
            truncated = True
            break
        if child.name.lower() in ARTIFACT_IGNORED_NAMES or child.is_symlink():
            ignored_count += 1
            continue
        child_path = child.resolve()
        if child_path != root and root not in child_path.parents:
            ignored_count += 1
            continue
        if child.is_dir():
            entries.append(
                ArtifactFileEntryResponse(
                    name=child.name,
                    relative_path=child_path.relative_to(root).as_posix(),
                    path=str(child_path),
                    entry_type="directory",
                )
            )
            continue
        source_type = _source_type_for_artifact_file(child)
        if source_type is None:
            ignored_count += 1
            continue
        try:
            child_stat = child_path.stat()
        except OSError:
            ignored_count += 1
            continue
        if child_stat.st_size > ARTIFACT_SOURCE_FILE_MAX_BYTES:
            ignored_count += 1
            continue
        entries.append(
            ArtifactFileEntryResponse(
                name=child.name,
                relative_path=child_path.relative_to(root).as_posix(),
                path=str(child_path),
                entry_type="file",
                source_type=source_type,
                size_bytes=child_stat.st_size,
                source_mtime_ns=child_stat.st_mtime_ns,
                selectable=True,
            )
        )
    return ArtifactFolderListingResponse(
        root_path=str(root),
        relative_path=relative_path,
        entries=entries,
        truncated=truncated,
        ignored_count=ignored_count,
    )


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
    return WorkspacesResponse(
        data=[_workspace_to_summary_response(workspace) for workspace in workspaces]
    )


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
    if (
        "pinned_paper_ids" in payload.model_fields_set
        and payload.pinned_paper_ids is not None
    ):
        pinned_paper_ids = []
        for paper_id in payload.pinned_paper_ids:
            safe_paper_id = sanitize_identifier(
                paper_id,
                field_name="pinned_paper_id",
                max_length=36,
            )
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
        description=(
            payload.description
            if "description" in payload.model_fields_set
            else workspace.description
        ),
        collection_id=collection_id,
        saved_query=(
            payload.saved_query
            if "saved_query" in payload.model_fields_set
            else workspace.saved_query
        ),
        focus_note=(
            payload.focus_note
            if "focus_note" in payload.model_fields_set
            else workspace.focus_note
        ),
        active_filters=(
            payload.active_filters
            if "active_filters" in payload.model_fields_set
            else dict(workspace.active_filters_json or {})
        ),
        pinned_paper_ids=pinned_paper_ids,
    )
    collection = (
        collection_repository.get_by_id(updated.collection_id)
        if updated.collection_id
        else None
    )
    return SingleWorkspaceResponse(
        data=WorkspaceDetailResponse(
            **_workspace_to_summary_response(updated).model_dump(),
            collection=_collection_to_response(collection),
            pinned_papers=_resolve_pinned_papers(paper_repository, pinned_paper_ids),
        )
    )


@router.get("/api/v1/studies/{workspace_id}/brief", response_model=SingleStudyBriefResponse)
def fetch_study_brief(
    workspace_id: str = Path(..., min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> SingleStudyBriefResponse:
    workspace_repository = WorkspaceRepository(session)
    safe_workspace_id = sanitize_identifier(
        workspace_id,
        field_name="workspace_id",
        max_length=36,
    )
    if workspace_repository.get_by_id(safe_workspace_id) is None:
        raise PaperbaseAPIError(
            status_code=404,
            error="workspace_not_found",
            message=f"No workspace found for id: {safe_workspace_id}",
        )

    brief = ResearchIntelligenceRepository(session).get_study_brief(safe_workspace_id)
    return SingleStudyBriefResponse(
        data=_study_brief_to_response(brief, workspace_id=safe_workspace_id)
    )


@router.put("/api/v1/studies/{workspace_id}/brief", response_model=SingleStudyBriefResponse)
def save_study_brief(
    payload: StudyBriefUpdateRequest,
    workspace_id: str = Path(..., min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> SingleStudyBriefResponse:
    workspace_repository = WorkspaceRepository(session)
    safe_workspace_id = sanitize_identifier(
        workspace_id,
        field_name="workspace_id",
        max_length=36,
    )
    if workspace_repository.get_by_id(safe_workspace_id) is None:
        raise PaperbaseAPIError(
            status_code=404,
            error="workspace_not_found",
            message=f"No workspace found for id: {safe_workspace_id}",
        )

    repository = ResearchIntelligenceRepository(session)
    brief_json = _validated_study_brief_payload(
        brief=payload.brief,
        workspace_id=safe_workspace_id,
        workspace_repository=workspace_repository,
    )
    saved_brief = repository.save_study_brief_if_version(
        workspace_id=safe_workspace_id,
        brief=brief_json,
        expected_version=payload.expected_version,
        updated_by="user",
    )
    if saved_brief is None:
        current_brief = repository.get_study_brief(safe_workspace_id)
        current_version = current_brief.version if current_brief is not None else 0
        raise PaperbaseAPIError(
            status_code=409,
            error="study_brief_version_conflict",
            message="Study Brief has changed since it was loaded.",
            details=[{"field": "expected_version", "current_version": current_version}],
        )
    return SingleStudyBriefResponse(
        data=_study_brief_to_response(saved_brief, workspace_id=safe_workspace_id)
    )


@router.get(
    "/api/v1/studies/{workspace_id}/brief/proposal",
    response_model=SingleStudyBriefProposalResponse,
)
def fetch_study_brief_proposal(
    workspace_id: str = Path(..., min_length=1, max_length=36),
    artifact_id: str = Query(..., min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> SingleStudyBriefProposalResponse:
    workspace_repository = WorkspaceRepository(session)
    safe_workspace_id = sanitize_identifier(
        workspace_id,
        field_name="workspace_id",
        max_length=36,
    )
    if workspace_repository.get_by_id(safe_workspace_id) is None:
        raise PaperbaseAPIError(
            status_code=404,
            error="workspace_not_found",
            message=f"No workspace found for id: {safe_workspace_id}",
        )

    artifact = _ensure_study_brief_artifact(
        session,
        workspace_id=safe_workspace_id,
        artifact_id=artifact_id,
    )
    repository = ResearchIntelligenceRepository(session)
    return SingleStudyBriefProposalResponse(
        data=_study_brief_proposal_response(
            workspace_id=safe_workspace_id,
            artifact=artifact,
            repository=repository,
            workspace_repository=workspace_repository,
        )
    )


@router.post(
    "/api/v1/studies/{workspace_id}/brief/proposal/accept",
    response_model=SingleStudyBriefResponse,
)
def accept_study_brief_proposal(
    payload: StudyBriefProposalAcceptRequest,
    workspace_id: str = Path(..., min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> SingleStudyBriefResponse:
    workspace_repository = WorkspaceRepository(session)
    safe_workspace_id = sanitize_identifier(
        workspace_id,
        field_name="workspace_id",
        max_length=36,
    )
    if workspace_repository.get_by_id(safe_workspace_id) is None:
        raise PaperbaseAPIError(
            status_code=404,
            error="workspace_not_found",
            message=f"No workspace found for id: {safe_workspace_id}",
        )

    artifact = _ensure_study_brief_artifact(
        session,
        workspace_id=safe_workspace_id,
        artifact_id=payload.artifact_id,
    )
    repository = ResearchIntelligenceRepository(session)
    current_brief, _current_version = _current_study_brief_payload(
        repository,
        workspace_id=safe_workspace_id,
    )
    _build_study_brief_proposal(
        artifact=artifact,
        current_brief=current_brief,
        workspace_id=safe_workspace_id,
        workspace_repository=workspace_repository,
    )
    brief_json = _validated_study_brief_payload(
        brief=payload.brief,
        workspace_id=safe_workspace_id,
        workspace_repository=workspace_repository,
    )
    saved_brief = repository.save_study_brief_if_version(
        workspace_id=safe_workspace_id,
        brief=brief_json,
        expected_version=payload.expected_version,
        updated_by="user_accepted_agent_proposal",
    )
    if saved_brief is None:
        current = repository.get_study_brief(safe_workspace_id)
        current_version = current.version if current is not None else 0
        raise PaperbaseAPIError(
            status_code=409,
            error="study_brief_version_conflict",
            message="Study Brief has changed since it was loaded.",
            details=[{"field": "expected_version", "current_version": current_version}],
        )
    return SingleStudyBriefResponse(
        data=_study_brief_to_response(saved_brief, workspace_id=safe_workspace_id)
    )


@router.get("/api/v1/studies/{workspace_id}/sources", response_model=StudySourcesResponse)
def list_study_sources(
    request: Request,
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
        data=[
            _source_to_response(source, config=request.app.state.paperbase_config)
            for source in repository.list_sources(workspace_id=safe_workspace_id)
        ]
    )


@router.get(
    "/api/v1/studies/{workspace_id}/artifact-files",
    response_model=SingleArtifactFolderListingResponse,
)
def browse_study_artifact_files(
    request: Request,
    workspace_id: str = Path(..., min_length=1, max_length=36),
    root_path: str = Query(..., min_length=1, max_length=4096),
    relative_path: str = Query("", max_length=2000),
    session: Session = Depends(get_session),
) -> SingleArtifactFolderListingResponse:
    repository = WorkspaceRepository(session)
    safe_workspace_id = sanitize_identifier(workspace_id, field_name="workspace_id", max_length=36)
    if repository.get_by_id(safe_workspace_id) is None:
        raise PaperbaseAPIError(
            status_code=404,
            error="workspace_not_found",
            message=f"No workspace found for id: {safe_workspace_id}",
        )

    root, folder, display_relative_path = _resolve_artifact_folder(
        root_path=root_path,
        relative_path=relative_path,
        config=request.app.state.paperbase_config,
    )
    return SingleArtifactFolderListingResponse(
        data=_artifact_folder_listing(
            root=root,
            folder=folder,
            relative_path=display_relative_path,
        )
    )


@router.post(
    "/api/v1/studies/{workspace_id}/sources",
    response_model=SingleStudySourceResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_study_source(
    request: Request,
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
    path, content, read_status, error_message, source_size_bytes, source_mtime_ns = (
        _read_study_source_payload(
            payload,
            config=request.app.state.paperbase_config,
        )
    )
    source = repository.create_source(
        workspace_id=safe_workspace_id,
        source_type=payload.source_type,
        title=title,
        path=path,
        content=content,
        summary=_summarize_source_text(content) if content else None,
        read_status=read_status,
        error_message=error_message,
        source_size_bytes=source_size_bytes,
        source_mtime_ns=source_mtime_ns,
    )
    return SingleStudySourceResponse(
        data=_source_to_response(source, config=request.app.state.paperbase_config)
    )


@router.delete(
    "/api/v1/studies/{workspace_id}/sources/{source_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
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
