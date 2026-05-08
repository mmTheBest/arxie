"""Research agent routes for collection-grounded investigation workflows."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Path, Query, Request, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from paperbase.db.models import (
    BackgroundJob,
    CollectionPaper,
    EngineeringTrick,
    ExtractionRun,
    Limitation,
    ResearchMessage,
    ResultRow,
    Section,
)
from paperbase.db.repositories import CollectionRepository, PaperRepository, ResearchRepository, WorkspaceRepository
from ra.utils.security import sanitize_identifier, sanitize_user_text
from services.paperbase_api.background_jobs import create_background_job
from services.paperbase_api.dependencies import get_session
from services.paperbase_api.errors import PaperbaseAPIError
from services.paperbase_api.models import (
    PaperResearchLabelPatchRequest,
    PaperResearchLabelResponse,
    PaperResearchLabelsResponse,
    ResearchArtifactPatchRequest,
    ResearchArtifactResponse,
    ResearchArtifactsResponse,
    ResearchMessageCreateRequest,
    ResearchMessageJobResponse,
    ResearchMessageJobResponseData,
    ResearchMessageResponse,
    ResearchThreadCreateRequest,
    ResearchThreadDetailResponse,
    ResearchThreadDetailResponseData,
    ResearchThreadResponse,
    ResearchThreadsResponse,
    SinglePaperResearchLabelResponse,
    SingleResearchArtifactResponse,
    SingleResearchThreadResponse,
)
from services.paperbase_api.routes.jobs import background_job_to_response

router = APIRouter(tags=["research"])
ACTIVE_JOB_STATUSES = {"pending", "queued", "running"}
PROMPT_VERSION = "research-agent-v1"


def _thread_to_response(thread) -> ResearchThreadResponse:  # noqa: ANN001
    return ResearchThreadResponse(
        id=thread.id,
        owner_id=thread.owner_id,
        title=thread.title,
        collection_id=thread.collection_id,
        workspace_id=thread.workspace_id,
        selected_paper_ids=list(thread.selected_paper_ids_json or []),
        status=thread.status,
    )


def _message_to_response(message) -> ResearchMessageResponse:  # noqa: ANN001
    return ResearchMessageResponse(
        id=message.id,
        thread_id=message.thread_id,
        role=message.role,
        content=message.content,
        artifact_id=message.artifact_id,
        metadata=dict(message.metadata_json or {}),
    )


def _artifact_to_response(artifact) -> ResearchArtifactResponse:  # noqa: ANN001
    return ResearchArtifactResponse(
        id=artifact.id,
        collection_id=artifact.collection_id,
        thread_id=artifact.thread_id,
        artifact_type=artifact.artifact_type,
        title=artifact.title,
        status=artifact.status,
        input_payload=dict(artifact.input_payload_json or {}),
        output_payload=dict(artifact.output_payload_json or {}),
        evidence_payload=dict(artifact.evidence_payload_json or {}),
        model_name=artifact.model_name,
        prompt_version=artifact.prompt_version,
        error_message=artifact.error_message,
    )


def _label_to_response(label) -> PaperResearchLabelResponse:  # noqa: ANN001
    return PaperResearchLabelResponse(
        id=label.id,
        collection_id=label.collection_id,
        paper_id=label.paper_id,
        user_label=label.user_label,
        inferred_label=label.inferred_label,
        inferred_signals=dict(label.inferred_signals_json or {}),
        notes=label.notes,
    )


def _ensure_collection(session: Session, collection_id: str) -> str:
    safe_collection_id = sanitize_identifier(collection_id, field_name="collection_id", max_length=36)
    if CollectionRepository(session).get_by_id(safe_collection_id) is None:
        raise PaperbaseAPIError(
            status_code=404,
            error="collection_not_found",
            message=f"No collection found for id: {safe_collection_id}",
        )
    return safe_collection_id


def _ensure_collection_paper(session: Session, *, collection_id: str, paper_id: str) -> str:
    safe_paper_id = sanitize_identifier(paper_id, field_name="paper_id", max_length=36)
    if PaperRepository(session).get_by_id(safe_paper_id) is None:
        raise PaperbaseAPIError(
            status_code=404,
            error="paper_not_found",
            message=f"No paper found for id: {safe_paper_id}",
        )
    membership = session.execute(
        select(CollectionPaper.id).where(
            CollectionPaper.collection_id == collection_id,
            CollectionPaper.paper_id == safe_paper_id,
        )
    ).scalar_one_or_none()
    if membership is None:
        raise PaperbaseAPIError(
            status_code=400,
            error="paper_not_in_collection",
            message=f"Paper {safe_paper_id} is not in collection {collection_id}.",
        )
    return safe_paper_id


def _sanitize_selected_paper_ids(
    session: Session,
    *,
    collection_id: str,
    paper_ids: list[str],
) -> list[str]:
    safe_paper_ids: list[str] = []
    seen: set[str] = set()
    for raw_paper_id in paper_ids:
        safe_paper_id = _ensure_collection_paper(
            session,
            collection_id=collection_id,
            paper_id=raw_paper_id,
        )
        if safe_paper_id in seen:
            continue
        seen.add(safe_paper_id)
        safe_paper_ids.append(safe_paper_id)
    return safe_paper_ids


def _infer_paper_research_signals(
    session: Session,
    *,
    collection_id: str,
    paper_id: str,
) -> tuple[str, dict[str, Any]]:
    _ = collection_id
    parsed_section_count = int(
        session.execute(select(func.count(Section.id)).where(Section.paper_id == paper_id)).scalar_one()
    )
    completed_extraction_count = int(
        session.execute(
            select(func.count(ExtractionRun.id)).where(
                ExtractionRun.paper_id == paper_id,
                ExtractionRun.status == "completed",
            )
        ).scalar_one()
    )
    result_count = int(
        session.execute(select(func.count(ResultRow.id)).where(ResultRow.paper_id == paper_id)).scalar_one()
    )
    limitation_count = int(
        session.execute(select(func.count(Limitation.id)).where(Limitation.paper_id == paper_id)).scalar_one()
    )
    engineering_trick_count = int(
        session.execute(
            select(func.count(EngineeringTrick.id)).where(EngineeringTrick.paper_id == paper_id)
        ).scalar_one()
    )
    design_strength_score = min(
        100,
        parsed_section_count * 5
        + completed_extraction_count * 20
        + result_count * 10
        + engineering_trick_count * 5
        + limitation_count * 3,
    )
    inferred_label = "strong_design" if design_strength_score >= 40 else "neutral"
    return inferred_label, {
        "parsed_section_count": parsed_section_count,
        "completed_extraction_count": completed_extraction_count,
        "result_count": result_count,
        "limitation_count": limitation_count,
        "engineering_trick_count": engineering_trick_count,
        "design_strength_score": design_strength_score,
    }


def _artifact_title(artifact_type: str) -> str:
    titles = {
        "field_patterns": "Field patterns",
        "hypotheses": "Hypotheses",
        "experiment_plan": "Experiment plan",
        "critique": "Critique",
        "experiment_backlog": "Experiment backlog",
        "benchmark_plan": "Benchmark plan",
        "revision_plan": "Revision plan",
        "assumption_map": "Assumption map",
    }
    return titles.get(artifact_type, "Research artifact")


def _infer_artifact_type(message: str) -> str:
    normalized = message.casefold()
    if "hypothes" in normalized or "gap" in normalized:
        return "hypotheses"
    if "critique" in normalized or "weakness" in normalized or "review" in normalized:
        return "critique"
    if "benchmark" in normalized or "baseline" in normalized:
        return "benchmark_plan"
    if "revision" in normalized or "improve" in normalized:
        return "revision_plan"
    if "assumption" in normalized:
        return "assumption_map"
    if "backlog" in normalized or "next" in normalized:
        return "experiment_backlog"
    if "pattern" in normalized or "learn from" in normalized:
        return "field_patterns"
    return "experiment_plan"


def _sanitize_source_ids(
    session: Session,
    *,
    workspace_id: str | None,
    source_ids: list[str],
) -> list[str]:
    if not source_ids:
        return []
    if workspace_id is None:
        raise PaperbaseAPIError(
            status_code=422,
            error="workspace_required_for_sources",
            message="Study sources require a research thread linked to a study.",
        )
    repository = WorkspaceRepository(session)
    safe_source_ids: list[str] = []
    for source_id in source_ids:
        safe_source_id = sanitize_identifier(source_id, field_name="source_id", max_length=36)
        source = repository.get_source(safe_source_id)
        if source is None or source.workspace_id != workspace_id:
            raise PaperbaseAPIError(
                status_code=404,
                error="study_source_not_found",
                message=f"No study source found for id: {safe_source_id}",
            )
        safe_source_ids.append(safe_source_id)
    return safe_source_ids


def _find_matching_active_research_job(
    session: Session,
    *,
    thread_id: str,
    collection_id: str,
    message: str,
    artifact_type: str,
    selected_paper_ids: list[str],
    source_ids: list[str],
) -> BackgroundJob | None:
    jobs = session.execute(
        select(BackgroundJob)
        .where(
            BackgroundJob.job_type == "research_agent_run",
            BackgroundJob.status.in_(ACTIVE_JOB_STATUSES),
        )
        .order_by(BackgroundJob.created_at.desc(), BackgroundJob.id.desc())
    ).scalars()
    for job in jobs:
        payload = dict(job.payload_json or {})
        if (
            payload.get("thread_id") == thread_id
            and payload.get("collection_id") == collection_id
            and payload.get("user_message") == message
            and payload.get("artifact_type") == artifact_type
            and list(payload.get("selected_paper_ids") or []) == selected_paper_ids
            and list(payload.get("source_ids") or []) == source_ids
        ):
            return job
    return None


@router.get("/api/v1/research/threads", response_model=ResearchThreadsResponse)
def list_research_threads(
    collection_id: str | None = Query(None, min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> ResearchThreadsResponse:
    safe_collection_id = (
        _ensure_collection(session, collection_id) if collection_id is not None else None
    )
    threads = ResearchRepository(session).list_threads(collection_id=safe_collection_id)
    return ResearchThreadsResponse(data=[_thread_to_response(thread) for thread in threads])


@router.post(
    "/api/v1/research/threads",
    response_model=SingleResearchThreadResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_research_thread(
    payload: ResearchThreadCreateRequest,
    session: Session = Depends(get_session),
) -> SingleResearchThreadResponse:
    safe_collection_id = _ensure_collection(session, payload.collection_id)
    selected_paper_ids = _sanitize_selected_paper_ids(
        session,
        collection_id=safe_collection_id,
        paper_ids=list(payload.selected_paper_ids),
    )
    thread = ResearchRepository(session).create_thread(
        owner_id=sanitize_user_text(payload.owner_id, field_name="owner_id", max_length=128),
        title=sanitize_user_text(payload.title, field_name="title", max_length=255),
        collection_id=safe_collection_id,
        workspace_id=(
            sanitize_identifier(payload.workspace_id, field_name="workspace_id", max_length=36)
            if payload.workspace_id is not None
            else None
        ),
        selected_paper_ids=selected_paper_ids,
    )
    return SingleResearchThreadResponse(data=_thread_to_response(thread))


@router.get(
    "/api/v1/research/threads/{thread_id}",
    response_model=ResearchThreadDetailResponse,
)
def fetch_research_thread(
    thread_id: str = Path(..., min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> ResearchThreadDetailResponse:
    safe_thread_id = sanitize_identifier(thread_id, field_name="thread_id", max_length=36)
    repository = ResearchRepository(session)
    thread = repository.get_thread(safe_thread_id)
    if thread is None:
        raise PaperbaseAPIError(
            status_code=404,
            error="research_thread_not_found",
            message=f"No research thread found for id: {safe_thread_id}",
        )
    artifacts = [
        artifact
        for artifact in repository.list_artifacts(collection_id=thread.collection_id)
        if artifact.thread_id == thread.id
    ]
    return ResearchThreadDetailResponse(
        data=ResearchThreadDetailResponseData(
            **_thread_to_response(thread).model_dump(),
            messages=[_message_to_response(message) for message in repository.list_messages(thread_id=thread.id)],
            artifacts=[_artifact_to_response(artifact) for artifact in artifacts],
        )
    )


@router.post(
    "/api/v1/research/threads/{thread_id}/messages",
    response_model=ResearchMessageJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_research_message(
    payload: ResearchMessageCreateRequest,
    request: Request,
    thread_id: str = Path(..., min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> ResearchMessageJobResponse:
    safe_thread_id = sanitize_identifier(thread_id, field_name="thread_id", max_length=36)
    repository = ResearchRepository(session)
    thread = repository.get_thread(safe_thread_id)
    if thread is None:
        raise PaperbaseAPIError(
            status_code=404,
            error="research_thread_not_found",
            message=f"No research thread found for id: {safe_thread_id}",
        )

    message_text = sanitize_user_text(payload.message, field_name="message", max_length=20000)
    artifact_type = payload.artifact_type or _infer_artifact_type(message_text)
    selected_paper_ids = list(thread.selected_paper_ids_json or [])
    source_ids = _sanitize_source_ids(
        session,
        workspace_id=thread.workspace_id,
        source_ids=list(payload.source_ids),
    )
    active_job = _find_matching_active_research_job(
        session,
        thread_id=thread.id,
        collection_id=thread.collection_id,
        message=message_text,
        artifact_type=artifact_type,
        selected_paper_ids=selected_paper_ids,
        source_ids=source_ids,
    )
    if active_job is not None:
        active_payload = dict(active_job.payload_json or {})
        artifact_id = active_payload.get("artifact_id")
        message_id = active_payload.get("message_id")
        artifact = repository.get_artifact(artifact_id) if isinstance(artifact_id, str) else None
        if artifact is not None and isinstance(message_id, str):
            stored_message = session.get(ResearchMessage, message_id)
            if stored_message is not None:
                return ResearchMessageJobResponse(
                    data=ResearchMessageJobResponseData(
                        message=_message_to_response(stored_message),
                        artifact=_artifact_to_response(artifact),
                        job=background_job_to_response(active_job),
                    )
                )

    user_message = repository.create_message(
        thread_id=thread.id,
        role="user",
        content=message_text,
    )
    artifact = repository.create_artifact(
        collection_id=thread.collection_id,
        thread_id=thread.id,
        artifact_type=artifact_type,
        title=_artifact_title(artifact_type),
        status="pending",
        input_payload={
            "message": message_text,
            "selected_paper_ids": selected_paper_ids,
            "source_ids": source_ids,
        },
        prompt_version=PROMPT_VERSION,
    )
    job = create_background_job(
        session_factory=request.app.state.session_factory,
        job_type="research_agent_run",
        payload_json={
            "thread_id": thread.id,
            "message_id": user_message.id,
            "artifact_id": artifact.id,
            "collection_id": thread.collection_id,
            "user_message": message_text,
            "artifact_type": artifact_type,
            "selected_paper_ids": selected_paper_ids,
            "workspace_id": thread.workspace_id,
            "source_ids": source_ids,
        },
        dispatcher=request.app.state.job_dispatcher,
    )
    return ResearchMessageJobResponse(
        data=ResearchMessageJobResponseData(
            message=_message_to_response(user_message),
            artifact=_artifact_to_response(artifact),
            job=background_job_to_response(job),
        )
    )


@router.get("/api/v1/research/artifacts", response_model=ResearchArtifactsResponse)
def list_research_artifacts(
    collection_id: str | None = Query(None, min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> ResearchArtifactsResponse:
    safe_collection_id = (
        _ensure_collection(session, collection_id) if collection_id is not None else None
    )
    artifacts = ResearchRepository(session).list_artifacts(collection_id=safe_collection_id)
    return ResearchArtifactsResponse(data=[_artifact_to_response(artifact) for artifact in artifacts])


@router.get(
    "/api/v1/research/artifacts/{artifact_id}",
    response_model=SingleResearchArtifactResponse,
)
def fetch_research_artifact(
    artifact_id: str = Path(..., min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> SingleResearchArtifactResponse:
    safe_artifact_id = sanitize_identifier(artifact_id, field_name="artifact_id", max_length=36)
    artifact = ResearchRepository(session).get_artifact(safe_artifact_id)
    if artifact is None:
        raise PaperbaseAPIError(
            status_code=404,
            error="research_artifact_not_found",
            message=f"No research artifact found for id: {safe_artifact_id}",
        )
    return SingleResearchArtifactResponse(data=_artifact_to_response(artifact))


@router.patch(
    "/api/v1/research/artifacts/{artifact_id}",
    response_model=SingleResearchArtifactResponse,
)
def update_research_artifact(
    payload: ResearchArtifactPatchRequest,
    artifact_id: str = Path(..., min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> SingleResearchArtifactResponse:
    safe_artifact_id = sanitize_identifier(artifact_id, field_name="artifact_id", max_length=36)
    repository = ResearchRepository(session)
    artifact = repository.get_artifact(safe_artifact_id)
    if artifact is None:
        raise PaperbaseAPIError(
            status_code=404,
            error="research_artifact_not_found",
            message=f"No research artifact found for id: {safe_artifact_id}",
        )
    updated = repository.update_artifact(
        safe_artifact_id,
        title=(
            sanitize_user_text(payload.title, field_name="title", max_length=255)
            if payload.title is not None
            else None
        ),
        status=(
            sanitize_user_text(payload.status, field_name="status", max_length=64)
            if payload.status is not None
            else None
        ),
    )
    return SingleResearchArtifactResponse(data=_artifact_to_response(updated))


@router.get(
    "/api/v1/collections/{collection_id}/research-labels",
    response_model=PaperResearchLabelsResponse,
)
def list_paper_research_labels(
    collection_id: str = Path(..., min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> PaperResearchLabelsResponse:
    safe_collection_id = _ensure_collection(session, collection_id)
    labels = ResearchRepository(session).list_labels(collection_id=safe_collection_id)
    return PaperResearchLabelsResponse(data=[_label_to_response(label) for label in labels])


@router.patch(
    "/api/v1/collections/{collection_id}/papers/{paper_id}/research-label",
    response_model=SinglePaperResearchLabelResponse,
)
def update_paper_research_label(
    payload: PaperResearchLabelPatchRequest,
    collection_id: str = Path(..., min_length=1, max_length=36),
    paper_id: str = Path(..., min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> SinglePaperResearchLabelResponse:
    safe_collection_id = _ensure_collection(session, collection_id)
    safe_paper_id = _ensure_collection_paper(
        session,
        collection_id=safe_collection_id,
        paper_id=paper_id,
    )
    inferred_label, inferred_signals = _infer_paper_research_signals(
        session,
        collection_id=safe_collection_id,
        paper_id=safe_paper_id,
    )
    label = ResearchRepository(session).upsert_label(
        collection_id=safe_collection_id,
        paper_id=safe_paper_id,
        user_label=payload.user_label,
        inferred_label=inferred_label,
        inferred_signals=inferred_signals,
        notes=(
            sanitize_user_text(payload.notes, field_name="notes", max_length=10000, allow_empty=True)
            if payload.notes is not None
            else None
        ),
    )
    return SinglePaperResearchLabelResponse(data=_label_to_response(label))
