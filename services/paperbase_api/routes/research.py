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
from paperbase.db.repositories import (
    CollectionRepository,
    PaperRepository,
    ResearchAgentRunRepository,
    ResearchRepository,
    WorkspaceRepository,
)
from paperbase.research.skill_policies import policy_for_skill
from paperbase.research.skills import artifact_type_for_skill, select_research_skill
from ra.utils.security import sanitize_identifier, sanitize_user_text
from services.paperbase_api.background_jobs import create_background_job
from services.paperbase_api.dependencies import get_project_id, get_session, get_session_factory
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
    ResearchAgentRunResponse,
    ResearchAgentStepResponse,
    ResearchSuggestionResponse,
    ResearchSuggestionsResponse,
    ResearchValidationReportResponse,
    ResearchThreadCreateRequest,
    ResearchThreadDetailResponse,
    ResearchThreadDetailResponseData,
    ResearchThreadResponse,
    ResearchThreadsResponse,
    SingleResearchAgentRunResponse,
    SinglePaperResearchLabelResponse,
    SingleResearchArtifactResponse,
    SingleResearchThreadResponse,
    StudyContextPackResponse,
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
    output_payload = dict(artifact.output_payload_json or {})
    return ResearchArtifactResponse(
        id=artifact.id,
        collection_id=artifact.collection_id,
        thread_id=artifact.thread_id,
        artifact_type=artifact.artifact_type,
        title=artifact.title,
        status=artifact.status,
        input_payload=dict(artifact.input_payload_json or {}),
        output_payload=output_payload,
        evidence_payload=dict(artifact.evidence_payload_json or {}),
        model_name=artifact.model_name,
        prompt_version=artifact.prompt_version,
        error_message=artifact.error_message,
        is_saved=bool(output_payload.get("is_saved")),
        saved_format=(
            str(output_payload.get("saved_format"))
            if output_payload.get("saved_format") is not None
            else None
        ),
        saved_title=(
            str(output_payload.get("saved_title"))
            if output_payload.get("saved_title") is not None
            else None
        ),
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


def _run_to_response(session: Session, run) -> ResearchAgentRunResponse:  # noqa: ANN001
    repository = ResearchAgentRunRepository(session)
    context_pack = repository.get_context_pack(run_id=run.id)
    validation_report = repository.get_validation_report(run_id=run.id)
    context_response = None
    if context_pack is not None:
        context = dict(context_pack.context_json or {})
        context_response = StudyContextPackResponse(
            id=context_pack.id,
            run_id=context_pack.run_id,
            collection_id=context_pack.collection_id,
            workspace_id=context_pack.workspace_id,
            task_type=context_pack.task_type,
            cache_key=context_pack.cache_key,
            context_summary={
                "paper_count": len(context.get("papers", []) or []),
                "source_count": len(context.get("sources", []) or []),
                "task_type": context.get("task_type"),
            },
            selected_item_counts=dict(context_pack.selected_item_counts_json or {}),
            readiness_warnings=list(context_pack.readiness_warnings_json or []),
        )
    validation_response = None
    if validation_report is not None:
        validation_response = ResearchValidationReportResponse(
            id=validation_report.id,
            run_id=validation_report.run_id,
            artifact_id=validation_report.artifact_id,
            harness_status=validation_report.harness_status,
            missing_evidence=list(validation_report.missing_evidence_json or []),
            unsupported_claims=list(validation_report.unsupported_claims_json or []),
            readiness_blockers=list(validation_report.readiness_blockers_json or []),
            report=dict(validation_report.report_json or {}),
        )
    return ResearchAgentRunResponse(
        id=run.id,
        thread_id=run.thread_id,
        artifact_id=run.artifact_id,
        collection_id=run.collection_id,
        workspace_id=run.workspace_id,
        skill_id=run.skill_id,
        artifact_type=run.artifact_type,
        model_policy=run.model_policy,
        status=run.status,
        input_json=dict(run.input_json or {}),
        model_name=run.model_name,
        error_message=run.error_message,
        steps=[
            ResearchAgentStepResponse(
                id=step.id,
                run_id=step.run_id,
                ordinal=step.ordinal,
                step_type=step.step_type,
                label=step.label,
                status=step.status,
                input_json=dict(step.input_json or {}),
                output_json=dict(step.output_json or {}),
                error_message=step.error_message,
            )
            for step in repository.list_steps(run_id=run.id)
        ],
        context_pack=context_response,
        validation_report=validation_response,
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


def _resolve_thread_workspace_id(
    session: Session,
    *,
    collection_id: str,
    workspace_id: str | None,
) -> str | None:
    if workspace_id is None:
        return None
    safe_workspace_id = sanitize_identifier(
        workspace_id,
        field_name="workspace_id",
        max_length=36,
    )
    workspace = WorkspaceRepository(session).get_by_id(safe_workspace_id)
    if workspace is None:
        raise PaperbaseAPIError(
            status_code=404,
            error="workspace_not_found",
            message=f"No workspace found for id: {safe_workspace_id}",
        )
    if workspace.collection_id is not None and workspace.collection_id != collection_id:
        raise PaperbaseAPIError(
            status_code=400,
            error="workspace_collection_mismatch",
            message="Research thread workspace must belong to the selected collection.",
        )
    return safe_workspace_id


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
        "literature_review": "Literature review",
        "comparison": "Comparison",
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
    if "literature review" in normalized or "synthesize" in normalized or "themes" in normalized:
        return "literature_review"
    if any(term in normalized for term in ("compare", "comparison", "contrast", "rank", "ranking")):
        return "comparison"
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


def _collection_processing_counts(session: Session, *, collection_id: str) -> dict[str, int]:
    member_paper_ids = select(CollectionPaper.paper_id).where(CollectionPaper.collection_id == collection_id)
    paper_count = int(
        session.execute(
            select(func.count()).select_from(CollectionPaper).where(
                CollectionPaper.collection_id == collection_id
            )
        ).scalar_one()
    )
    parsed_count = int(
        session.execute(
            select(func.count(func.distinct(Section.paper_id))).where(Section.paper_id.in_(member_paper_ids))
        ).scalar_one()
    )
    extracted_count = int(
        session.execute(
            select(func.count(func.distinct(ExtractionRun.paper_id))).where(
                ExtractionRun.paper_id.in_(member_paper_ids),
                ExtractionRun.status == "completed",
            )
        ).scalar_one()
    )
    return {
        "paper_count": paper_count,
        "parsed_count": parsed_count,
        "extracted_count": extracted_count,
    }


def _suggestions_for_counts(counts: dict[str, int]) -> list[ResearchSuggestionResponse]:
    readiness = (
        "evidence_ready"
        if counts["extracted_count"] > 0
        else "text_ready"
        if counts["parsed_count"] > 0
        else "imported"
    )
    suggestions = [
        ResearchSuggestionResponse(
            id="literature_review_synthesis",
            label="Synthesize themes",
            instruction=(
                "Synthesize this collection into major themes, consensus points, "
                "controversies, research gaps, and future directions."
            ),
            skill_id="literature_review",
            artifact_type="literature_review",
            readiness=readiness,
        ),
        ResearchSuggestionResponse(
            id="evidence_quality_check",
            label="Check evidence coverage",
            instruction=(
                "Check evidence coverage, unsupported claims, missing context, "
                "and reproducibility risks for the current study."
            ),
            skill_id="quality_harness",
            artifact_type="critique",
            readiness=readiness,
        ),
    ]
    if counts["extracted_count"] > 0:
        suggestions.insert(
            1,
            ResearchSuggestionResponse(
                id="benchmark_ablation_plan",
                label="Plan benchmarks",
                instruction=(
                    "Build a benchmark, baseline, metric, and ablation plan grounded "
                    "in the strongest extracted evidence."
                ),
                skill_id="benchmark_planning",
                artifact_type="benchmark_plan",
                readiness=readiness,
            ),
        )
    else:
        suggestions.append(
            ResearchSuggestionResponse(
                id="benchmark_ablation_plan",
                label="Prepare benchmark plan",
                instruction=(
                    "Draft a benchmark and ablation plan from available paper metadata, "
                    "and list what extraction should add next."
                ),
                skill_id="benchmark_planning",
                artifact_type="benchmark_plan",
                readiness=readiness,
            )
        )
    return suggestions


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


@router.get("/api/v1/research/suggestions", response_model=ResearchSuggestionsResponse)
def list_research_suggestions(
    collection_id: str = Query(..., min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> ResearchSuggestionsResponse:
    safe_collection_id = _ensure_collection(session, collection_id)
    counts = _collection_processing_counts(session, collection_id=safe_collection_id)
    return ResearchSuggestionsResponse(data=_suggestions_for_counts(counts))


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
    workspace_id = _resolve_thread_workspace_id(
        session,
        collection_id=safe_collection_id,
        workspace_id=payload.workspace_id,
    )
    thread = ResearchRepository(session).create_thread(
        owner_id=sanitize_user_text(payload.owner_id, field_name="owner_id", max_length=128),
        title=sanitize_user_text(payload.title, field_name="title", max_length=255),
        collection_id=safe_collection_id,
        workspace_id=workspace_id,
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
    inferred_artifact_type = payload.artifact_type or _infer_artifact_type(message_text)
    skill_id = select_research_skill(
        message_text,
        suggestion_id=payload.suggestion_id,
        artifact_type=inferred_artifact_type,
    )
    if payload.artifact_type is not None:
        artifact_type = payload.artifact_type
    elif payload.suggestion_id is not None or skill_id in {"literature_review", "quality_harness"}:
        artifact_type = artifact_type_for_skill(skill_id, fallback=inferred_artifact_type)
    else:
        artifact_type = inferred_artifact_type
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
                        run_id=(
                            str(active_payload.get("run_id"))
                            if active_payload.get("run_id") is not None
                            else None
                        ),
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
            "skill_id": skill_id,
            "suggestion_id": payload.suggestion_id,
        },
        prompt_version=PROMPT_VERSION,
    )
    policy = policy_for_skill(skill_id)
    run = ResearchAgentRunRepository(session).create_run(
        thread_id=thread.id,
        artifact_id=artifact.id,
        collection_id=thread.collection_id,
        workspace_id=thread.workspace_id,
        skill_id=skill_id,
        artifact_type=artifact_type,
        model_policy=policy.model_policy,
        input_json={
            "message": message_text,
            "selected_paper_ids": selected_paper_ids,
            "source_ids": source_ids,
            "suggestion_id": payload.suggestion_id,
        },
    )
    job = create_background_job(
        session_factory=get_session_factory(request),
        job_type="research_agent_run",
        payload_json={
            "run_id": run.id,
            "thread_id": thread.id,
            "message_id": user_message.id,
            "artifact_id": artifact.id,
            "collection_id": thread.collection_id,
            "user_message": message_text,
            "artifact_type": artifact_type,
            "skill_id": skill_id,
            "suggestion_id": payload.suggestion_id,
            "selected_paper_ids": selected_paper_ids,
            "workspace_id": thread.workspace_id,
            "source_ids": source_ids,
        },
        dispatcher=request.app.state.job_dispatcher,
        project_id=get_project_id(request),
    )
    return ResearchMessageJobResponse(
        data=ResearchMessageJobResponseData(
            message=_message_to_response(user_message),
            artifact=_artifact_to_response(artifact),
            job=background_job_to_response(job),
            run_id=run.id,
        )
    )


@router.get(
    "/api/v1/research/runs/{run_id}",
    response_model=SingleResearchAgentRunResponse,
)
def fetch_research_run(
    run_id: str = Path(..., min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> SingleResearchAgentRunResponse:
    safe_run_id = sanitize_identifier(run_id, field_name="run_id", max_length=36)
    run = ResearchAgentRunRepository(session).get_run(safe_run_id)
    if run is None:
        raise PaperbaseAPIError(
            status_code=404,
            error="research_run_not_found",
            message=f"No research agent run found for id: {safe_run_id}",
        )
    return SingleResearchAgentRunResponse(data=_run_to_response(session, run))


@router.get(
    "/api/v1/research/artifacts/{artifact_id}/run",
    response_model=SingleResearchAgentRunResponse,
)
def fetch_research_artifact_run(
    artifact_id: str = Path(..., min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> SingleResearchAgentRunResponse:
    safe_artifact_id = sanitize_identifier(artifact_id, field_name="artifact_id", max_length=36)
    run = ResearchAgentRunRepository(session).get_run_for_artifact(safe_artifact_id)
    if run is None:
        raise PaperbaseAPIError(
            status_code=404,
            error="research_run_not_found",
            message=f"No research agent run found for artifact id: {safe_artifact_id}",
        )
    return SingleResearchAgentRunResponse(data=_run_to_response(session, run))


@router.get("/api/v1/research/artifacts", response_model=ResearchArtifactsResponse)
def list_research_artifacts(
    collection_id: str | None = Query(None, min_length=1, max_length=36),
    saved_only: bool = Query(False),
    session: Session = Depends(get_session),
) -> ResearchArtifactsResponse:
    safe_collection_id = (
        _ensure_collection(session, collection_id) if collection_id is not None else None
    )
    artifacts = ResearchRepository(session).list_artifacts(collection_id=safe_collection_id)
    if saved_only:
        artifacts = [
            artifact
            for artifact in artifacts
            if bool((artifact.output_payload_json or {}).get("is_saved"))
        ]
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
    output_payload = dict(artifact.output_payload_json or {})
    if payload.is_saved is not None:
        output_payload["is_saved"] = payload.is_saved
    if payload.saved_format is not None:
        output_payload["saved_format"] = sanitize_user_text(
            payload.saved_format,
            field_name="saved_format",
            max_length=64,
        )
    if payload.saved_title is not None:
        output_payload["saved_title"] = sanitize_user_text(
            payload.saved_title,
            field_name="saved_title",
            max_length=255,
        )

    updated = repository.update_artifact(
        safe_artifact_id,
        title=(
            sanitize_user_text(payload.title, field_name="title", max_length=255)
            if payload.title is not None
            else None
        ),
        status=None,
        output_payload=output_payload if output_payload != dict(artifact.output_payload_json or {}) else None,
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
