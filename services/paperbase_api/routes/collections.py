"""Collection and annotation routes for the Paperbase API service."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Path, Query, Request, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from paperbase.db.models import (
    BackgroundJob,
    CollectionPaper,
    Dataset,
    EngineeringTrick,
    ExtractionRun,
    Figure,
    GlossaryTerm,
    Limitation,
    Method,
    Metric,
    Paper,
    ResearchDesignElement,
    ResultRow,
    Section,
    TableArtifact,
)
from paperbase.db.repositories import (
    AnnotationRepository,
    CollectionRepository,
    PaperRepository,
)
from ra.utils.security import sanitize_identifier, sanitize_user_text
from services.paperbase_api.background_jobs import create_background_job
from services.paperbase_api.dependencies import get_session
from services.paperbase_api.errors import PaperbaseAPIError
from services.paperbase_api.models import (
    AnnotationCreateRequest,
    AnnotationResponse,
    AnnotationsResponse,
    CollectionCreateRequest,
    CollectionPaperCreateRequest,
    CollectionPaperMembershipResponse,
    CollectionPapersResponse,
    CollectionStructuredSummaryResponse,
    CollectionStructuredSummaryResponseData,
    CollectionSummaryEngineeringTrickResponse,
    CollectionSummaryFigureResponse,
    CollectionSummaryGlossaryTermResponse,
    CollectionSummaryLimitationResponse,
    CollectionSummaryNamedArtifactResponse,
    CollectionSummaryResearchDesignElementResponse,
    CollectionSummaryResultRowResponse,
    CollectionSummaryTableResponse,
    CollectionsResponse,
    CollectionSummaryResponse,
    PaperSummaryResponse,
    RunCollectionParseRequest,
    SingleAnnotationResponse,
    SingleBackgroundJobResponse,
    SingleCollectionPaperResponse,
    SingleCollectionResponse,
)
from services.paperbase_api.routes.jobs import background_job_to_response
from services.paperbase_api.normalization import (
    canonicalize_metric_display_name,
    normalize_summary_key,
)

router = APIRouter(tags=["collections"])
ACTIVE_JOB_STATUSES = {"pending", "queued", "running"}


def _canonicalize_artifact_display_name(value: str, *, artifact_type: str) -> str:
    if artifact_type == "metric":
        return canonicalize_metric_display_name(value)
    return value.strip()


def _build_named_artifact_summary(items, *, artifact_type: str) -> list[CollectionSummaryNamedArtifactResponse]:  # noqa: ANN001
    summarized: dict[str, CollectionSummaryNamedArtifactResponse] = {}
    for item in items:
        display_name = _canonicalize_artifact_display_name(item.display_name, artifact_type=artifact_type)
        key = normalize_summary_key(display_name)
        summarized.setdefault(
            key,
            CollectionSummaryNamedArtifactResponse(id=item.id, display_name=display_name),
        )
    return list(summarized.values())


def _job_collection_id(job: BackgroundJob) -> str | None:
    result_json = dict(job.result_json or {})
    payload_json = dict(job.payload_json or {})
    return result_json.get("collection_id") or payload_json.get("collection_id")


def _active_status_wins(current_status: str | None, candidate_status: str) -> bool:
    if current_status is None:
        return True
    if candidate_status == "running" and current_status in {"pending", "queued"}:
        return True
    return candidate_status in ACTIVE_JOB_STATUSES and current_status not in ACTIVE_JOB_STATUSES


def _find_matching_active_collection_job(
    session: Session,
    *,
    job_type: str,
    job_payload: dict[str, object],
) -> BackgroundJob | None:
    jobs = session.execute(
        select(BackgroundJob)
        .where(
            BackgroundJob.job_type == job_type,
            BackgroundJob.status.in_(ACTIVE_JOB_STATUSES),
        )
        .order_by(BackgroundJob.created_at.desc(), BackgroundJob.id.desc())
    ).scalars()
    for job in jobs:
        payload_json = dict(job.payload_json or {})
        if payload_json == job_payload:
            return job
    return None


def _collection_readiness_inputs(session: Session, collection_id: str) -> dict[str, int | str | None]:
    member_paper_ids = select(CollectionPaper.paper_id).where(CollectionPaper.collection_id == collection_id)
    paper_count = session.execute(
        select(func.count()).select_from(CollectionPaper).where(CollectionPaper.collection_id == collection_id)
    ).scalar_one()
    parsed_paper_count = session.execute(
        select(func.count(func.distinct(Section.paper_id))).where(Section.paper_id.in_(member_paper_ids))
    ).scalar_one()
    extracted_paper_count = session.execute(
        select(func.count(func.distinct(ExtractionRun.paper_id))).where(
            ExtractionRun.paper_id.in_(member_paper_ids),
            ExtractionRun.status == "completed",
        )
    ).scalar_one()

    latest_job_status = None
    latest_parse_job_status = None
    latest_extraction_job_status = None
    failed_job_count = 0
    jobs = session.execute(
        select(BackgroundJob).order_by(
            BackgroundJob.created_at.desc(),
            BackgroundJob.id.desc(),
        )
    ).scalars()
    for job in jobs:
        if _job_collection_id(job) != collection_id:
            continue
        if _active_status_wins(latest_job_status, job.status):
            latest_job_status = job.status
        if job.job_type == "collection_parse" and _active_status_wins(latest_parse_job_status, job.status):
            latest_parse_job_status = job.status
        if job.job_type == "collection_extract" and _active_status_wins(latest_extraction_job_status, job.status):
            latest_extraction_job_status = job.status
        if job.status == "failed":
            failed_job_count += 1

    return {
        "paper_count": paper_count,
        "parsed_paper_count": parsed_paper_count,
        "extracted_paper_count": extracted_paper_count,
        "latest_job_status": latest_job_status,
        "latest_parse_job_status": latest_parse_job_status,
        "latest_extraction_job_status": latest_extraction_job_status,
        "failed_job_count": failed_job_count,
    }


def _sanitize_collection_paper_ids(
    session: Session,
    *,
    collection_id: str,
    paper_ids: list[str] | None,
) -> list[str] | None:
    if paper_ids is None:
        return None

    member_ids = set(
        session.execute(
            select(CollectionPaper.paper_id).where(CollectionPaper.collection_id == collection_id)
        ).scalars()
    )
    safe_paper_ids: list[str] = []
    seen: set[str] = set()
    for raw_paper_id in paper_ids:
        safe_paper_id = sanitize_identifier(raw_paper_id, field_name="paper_id", max_length=36)
        if safe_paper_id not in member_ids:
            raise PaperbaseAPIError(
                status_code=400,
                error="paper_not_in_collection",
                message=f"Paper {safe_paper_id} is not in collection {collection_id}.",
            )
        if safe_paper_id in seen:
            continue
        seen.add(safe_paper_id)
        safe_paper_ids.append(safe_paper_id)
    return safe_paper_ids


def _collection_paper_job_state(
    session: Session,
    *,
    collection_id: str,
    paper_ids: list[str],
) -> dict[str, dict[str, str | None]]:
    state_by_paper_id = {
        paper_id: {
            "latest_parse_job_status": None,
            "latest_extraction_job_status": None,
            "latest_job_error": None,
        }
        for paper_id in paper_ids
    }
    paper_id_set = set(paper_ids)
    jobs = session.execute(
        select(BackgroundJob).order_by(
            BackgroundJob.created_at.desc(),
            BackgroundJob.id.desc(),
        )
    ).scalars()
    for job in jobs:
        if _job_collection_id(job) != collection_id:
            continue
        if job.job_type not in {"collection_parse", "collection_extract"}:
            continue

        payload_json = dict(job.payload_json or {})
        payload_paper_ids = payload_json.get("paper_ids")
        if payload_paper_ids is None:
            affected_paper_ids = paper_ids
        else:
            affected_paper_ids = [
                paper_id for paper_id in payload_paper_ids if paper_id in paper_id_set
            ]

        for paper_id in affected_paper_ids:
            paper_state = state_by_paper_id[paper_id]
            if job.job_type == "collection_parse" and _active_status_wins(
                paper_state["latest_parse_job_status"],
                job.status,
            ):
                paper_state["latest_parse_job_status"] = job.status
                if job.error_message and paper_state["latest_job_error"] is None:
                    paper_state["latest_job_error"] = job.error_message
            if job.job_type == "collection_extract" and _active_status_wins(
                paper_state["latest_extraction_job_status"],
                job.status,
            ):
                paper_state["latest_extraction_job_status"] = job.status
                if job.error_message and paper_state["latest_job_error"] is None:
                    paper_state["latest_job_error"] = job.error_message
    return state_by_paper_id


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


def _collection_to_response(collection, session: Session) -> CollectionSummaryResponse:  # noqa: ANN001
    readiness = _collection_readiness_inputs(session, collection.id)
    return CollectionSummaryResponse(
        id=collection.id,
        owner_id=collection.owner_id,
        scope_type=collection.scope_type,
        title=collection.title,
        description=collection.description,
        extraction_profile_id=collection.extraction_profile_id,
        tags=list(collection.tags_json or []),
        paper_count=int(readiness["paper_count"] or 0),
        parsed_paper_count=int(readiness["parsed_paper_count"] or 0),
        extracted_paper_count=int(readiness["extracted_paper_count"] or 0),
        latest_job_status=readiness["latest_job_status"],
        latest_parse_job_status=readiness["latest_parse_job_status"],
        latest_extraction_job_status=readiness["latest_extraction_job_status"],
        failed_job_count=int(readiness["failed_job_count"] or 0),
    )


def _annotation_to_response(annotation) -> AnnotationResponse:  # noqa: ANN001
    return AnnotationResponse(
        id=annotation.id,
        author_id=annotation.author_id,
        collection_id=annotation.collection_id,
        target_type=annotation.target_type,
        target_id=annotation.target_id,
        body=annotation.body,
        tags=list(annotation.tags_json or []),
        status=annotation.status,
    )


@router.get("/api/v1/collections", response_model=CollectionsResponse)
def list_collections(
    owner_id: str | None = Query(None, min_length=1, max_length=128),
    session: Session = Depends(get_session),
) -> CollectionsResponse:
    repository = CollectionRepository(session)
    safe_owner_id = (
        sanitize_user_text(owner_id, field_name="owner_id", max_length=128)
        if owner_id is not None
        else None
    )
    collections = repository.list_collections(owner_id=safe_owner_id)
    return CollectionsResponse(data=[_collection_to_response(collection, session) for collection in collections])


@router.post(
    "/api/v1/collections",
    response_model=SingleCollectionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_collection(
    payload: CollectionCreateRequest,
    session: Session = Depends(get_session),
) -> SingleCollectionResponse:
    repository = CollectionRepository(session)
    owner_id = sanitize_user_text(payload.owner_id, field_name="owner_id", max_length=128)
    title = sanitize_user_text(payload.title, field_name="title", max_length=255)

    if repository.get_by_owner_title(owner_id, title) is not None:
        raise PaperbaseAPIError(
            status_code=409,
            error="collection_conflict",
            message="A collection with this title already exists for the owner.",
        )

    collection = repository.create(
        owner_id=owner_id,
        title=title,
        description=payload.description,
        scope_type=payload.scope_type,
        tags=list(payload.tags),
        extraction_profile_id=payload.extraction_profile_id,
    )
    return SingleCollectionResponse(data=_collection_to_response(collection, session))


@router.get("/api/v1/collections/{collection_id}", response_model=SingleCollectionResponse)
def fetch_collection(
    collection_id: str = Path(..., min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> SingleCollectionResponse:
    repository = CollectionRepository(session)
    safe_collection_id = sanitize_identifier(collection_id, field_name="collection_id", max_length=36)
    collection = repository.get_by_id(safe_collection_id)
    if collection is None:
        raise PaperbaseAPIError(
            status_code=404,
            error="collection_not_found",
            message=f"No collection found for id: {safe_collection_id}",
        )
    return SingleCollectionResponse(data=_collection_to_response(collection, session))


@router.post(
    "/api/v1/collections/{collection_id}/papers",
    response_model=SingleCollectionPaperResponse,
    status_code=status.HTTP_201_CREATED,
)
def add_collection_paper(
    payload: CollectionPaperCreateRequest,
    collection_id: str = Path(..., min_length=1, max_length=36),
    session: Session = Depends(get_session),
 ) -> SingleCollectionPaperResponse:
    collection_repository = CollectionRepository(session)
    paper_repository = PaperRepository(session)
    safe_collection_id = sanitize_identifier(collection_id, field_name="collection_id", max_length=36)
    safe_paper_id = sanitize_identifier(payload.paper_id, field_name="paper_id", max_length=36)

    collection = collection_repository.get_by_id(safe_collection_id)
    if collection is None:
        raise PaperbaseAPIError(
            status_code=404,
            error="collection_not_found",
            message=f"No collection found for id: {safe_collection_id}",
        )

    paper = paper_repository.get_by_id(safe_paper_id)
    if paper is None:
        raise PaperbaseAPIError(
            status_code=404,
            error="paper_not_found",
            message=f"No paper found for id: {safe_paper_id}",
        )

    membership = collection_repository.add_paper(
        collection_id=safe_collection_id,
        paper_id=safe_paper_id,
        position=payload.position,
        membership_note=payload.membership_note,
    )

    return SingleCollectionPaperResponse(
        data=CollectionPaperMembershipResponse(
            id=membership.id,
            collection_id=membership.collection_id,
            paper_id=membership.paper_id,
            position=membership.position,
            membership_note=membership.membership_note,
            paper=_paper_to_response(paper),
        )
    )


@router.get("/api/v1/collections/{collection_id}/papers", response_model=CollectionPapersResponse)
def list_collection_papers(
    collection_id: str = Path(..., min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> CollectionPapersResponse:
    collection_repository = CollectionRepository(session)
    paper_repository = PaperRepository(session)
    safe_collection_id = sanitize_identifier(collection_id, field_name="collection_id", max_length=36)
    collection = collection_repository.get_by_id(safe_collection_id)
    if collection is None:
        raise PaperbaseAPIError(
            status_code=404,
            error="collection_not_found",
            message=f"No collection found for id: {safe_collection_id}",
        )

    memberships = collection_repository.list_papers(safe_collection_id)
    paper_ids = [membership.paper_id for membership in memberships]
    section_counts = dict(
        session.execute(
            select(Section.paper_id, func.count(Section.id))
            .where(Section.paper_id.in_(paper_ids))
            .group_by(Section.paper_id)
        ).all()
    ) if paper_ids else {}
    completed_extraction_counts = dict(
        session.execute(
            select(ExtractionRun.paper_id, func.count(ExtractionRun.id))
            .where(
                ExtractionRun.paper_id.in_(paper_ids),
                ExtractionRun.status == "completed",
            )
            .group_by(ExtractionRun.paper_id)
        ).all()
    ) if paper_ids else {}
    job_state_by_paper_id = _collection_paper_job_state(
        session,
        collection_id=safe_collection_id,
        paper_ids=paper_ids,
    )
    data: list[CollectionPaperMembershipResponse] = []
    for membership in memberships:
        paper = paper_repository.get_by_id(membership.paper_id)
        if paper is None:
            continue
        parsed_section_count = int(section_counts.get(paper.id, 0) or 0)
        completed_extraction_count = int(completed_extraction_counts.get(paper.id, 0) or 0)
        job_state = job_state_by_paper_id.get(paper.id, {})
        data.append(
            CollectionPaperMembershipResponse(
                id=membership.id,
                collection_id=membership.collection_id,
                paper_id=membership.paper_id,
                position=membership.position,
                membership_note=membership.membership_note,
                paper=_paper_to_response(
                    paper,
                    authors=paper_repository.list_author_names(paper.id),
                    tags=paper_repository.list_tags(paper.id),
                ),
                is_parsed=parsed_section_count > 0,
                is_extracted=completed_extraction_count > 0,
                parsed_section_count=parsed_section_count,
                completed_extraction_count=completed_extraction_count,
                latest_parse_job_status=job_state.get("latest_parse_job_status"),
                latest_extraction_job_status=job_state.get("latest_extraction_job_status"),
                latest_job_error=job_state.get("latest_job_error"),
            )
        )
    return CollectionPapersResponse(data=data)


@router.get(
    "/api/v1/collections/{collection_id}/structured-summary",
    response_model=CollectionStructuredSummaryResponse,
)
def fetch_collection_structured_summary(
    collection_id: str = Path(..., min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> CollectionStructuredSummaryResponse:
    collection_repository = CollectionRepository(session)
    safe_collection_id = sanitize_identifier(collection_id, field_name="collection_id", max_length=36)
    collection = collection_repository.get_by_id(safe_collection_id)
    if collection is None:
        raise PaperbaseAPIError(
            status_code=404,
            error="collection_not_found",
            message=f"No collection found for id: {safe_collection_id}",
        )

    member_paper_ids = select(CollectionPaper.paper_id).where(CollectionPaper.collection_id == safe_collection_id)

    readiness = _collection_readiness_inputs(session, safe_collection_id)

    datasets = session.execute(
        select(Dataset)
        .where(Dataset.paper_id.in_(member_paper_ids))
        .order_by(Dataset.display_name.asc(), Dataset.created_at.asc())
    ).scalars().all()
    methods = session.execute(
        select(Method)
        .where(Method.paper_id.in_(member_paper_ids))
        .order_by(Method.display_name.asc(), Method.created_at.asc())
    ).scalars().all()
    metrics = session.execute(
        select(Metric)
        .where(Metric.paper_id.in_(member_paper_ids))
        .order_by(Metric.display_name.asc(), Metric.created_at.asc())
    ).scalars().all()
    figures = session.execute(
        select(Figure)
        .where(Figure.paper_id.in_(member_paper_ids))
        .order_by(Figure.page_number.asc(), Figure.created_at.asc())
        .limit(12)
    ).scalars().all()
    tables = session.execute(
        select(TableArtifact)
        .where(TableArtifact.paper_id.in_(member_paper_ids))
        .order_by(TableArtifact.page_number.asc(), TableArtifact.created_at.asc())
        .limit(12)
    ).scalars().all()
    glossary_terms = session.execute(
        select(GlossaryTerm)
        .where(GlossaryTerm.paper_id.in_(member_paper_ids))
        .order_by(GlossaryTerm.term.asc(), GlossaryTerm.created_at.asc())
    ).scalars().all()
    limitations = session.execute(
        select(Limitation)
        .where(Limitation.paper_id.in_(member_paper_ids))
        .order_by(Limitation.created_at.asc())
    ).scalars().all()
    engineering_tricks = session.execute(
        select(EngineeringTrick)
        .where(EngineeringTrick.paper_id.in_(member_paper_ids))
        .order_by(EngineeringTrick.title.asc(), EngineeringTrick.created_at.asc())
    ).scalars().all()
    research_design_elements = session.execute(
        select(ResearchDesignElement)
        .where(ResearchDesignElement.paper_id.in_(member_paper_ids))
        .order_by(ResearchDesignElement.element_type.asc(), ResearchDesignElement.created_at.asc())
        .limit(24)
    ).scalars().all()
    result_rows = session.execute(
        select(ResultRow, Paper, Dataset, Method, Metric)
        .join(Paper, Paper.id == ResultRow.paper_id)
        .outerjoin(Dataset, Dataset.id == ResultRow.dataset_id)
        .outerjoin(Method, Method.id == ResultRow.method_id)
        .outerjoin(Metric, Metric.id == ResultRow.metric_id)
        .where(ResultRow.paper_id.in_(member_paper_ids))
        .order_by(ResultRow.value_numeric.desc().nullslast(), ResultRow.created_at.asc())
        .limit(10)
    ).all()

    return CollectionStructuredSummaryResponse(
        data=CollectionStructuredSummaryResponseData(
            collection_id=safe_collection_id,
            paper_count=int(readiness["paper_count"] or 0),
            parsed_paper_count=int(readiness["parsed_paper_count"] or 0),
            extracted_paper_count=int(readiness["extracted_paper_count"] or 0),
            latest_job_status=readiness["latest_job_status"],
            latest_parse_job_status=readiness["latest_parse_job_status"],
            latest_extraction_job_status=readiness["latest_extraction_job_status"],
            failed_job_count=int(readiness["failed_job_count"] or 0),
            datasets=_build_named_artifact_summary(datasets, artifact_type="dataset"),
            methods=_build_named_artifact_summary(methods, artifact_type="method"),
            metrics=_build_named_artifact_summary(metrics, artifact_type="metric"),
            figures=[
                CollectionSummaryFigureResponse(
                    id=item.id,
                    page_number=item.page_number,
                    figure_label=item.figure_label,
                    caption=item.caption,
                )
                for item in figures
            ],
            tables=[
                CollectionSummaryTableResponse(
                    id=item.id,
                    page_number=item.page_number,
                    table_label=item.table_label,
                    caption=item.caption,
                )
                for item in tables
            ],
            glossary_terms=[
                CollectionSummaryGlossaryTermResponse(
                    id=item.id,
                    term=item.term,
                    definition=item.definition,
                )
                for item in glossary_terms
            ],
            limitations=[
                CollectionSummaryLimitationResponse(
                    id=item.id,
                    statement=item.statement,
                )
                for item in limitations
            ],
            engineering_tricks=[
                CollectionSummaryEngineeringTrickResponse(
                    id=item.id,
                    title=item.title,
                    description=item.description,
                )
                for item in engineering_tricks
            ],
            research_design_elements=[
                CollectionSummaryResearchDesignElementResponse(
                    id=item.id,
                    paper_id=item.paper_id,
                    element_type=item.element_type,
                    title=item.title,
                    description=item.description,
                    metadata=dict(item.metadata_json or {}),
                )
                for item in research_design_elements
            ],
            top_result_rows=[
                CollectionSummaryResultRowResponse(
                    id=result_row.id,
                    paper_id=paper.id,
                    paper_title=paper.canonical_title,
                    dataset=dataset.display_name.strip() if dataset is not None else None,
                    method=method.display_name.strip() if method is not None else None,
                    metric=(
                        canonicalize_metric_display_name(metric.display_name)
                        if metric is not None
                        else None
                    ),
                    value_numeric=result_row.value_numeric,
                    value_text=result_row.value_text,
                )
                for result_row, paper, dataset, method, metric in result_rows
            ],
        )
    )


@router.post(
    "/api/v1/collections/{collection_id}/parse",
    response_model=SingleBackgroundJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def parse_collection(
    payload: RunCollectionParseRequest,
    request: Request,
    collection_id: str = Path(..., min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> SingleBackgroundJobResponse:
    collection_repository = CollectionRepository(session)
    safe_collection_id = sanitize_identifier(collection_id, field_name="collection_id", max_length=36)
    collection = collection_repository.get_by_id(safe_collection_id)
    if collection is None:
        raise PaperbaseAPIError(
            status_code=404,
            error="collection_not_found",
            message=f"No collection found for id: {safe_collection_id}",
        )

    job_payload = {
        "collection_id": safe_collection_id,
        "limit": payload.limit,
        "paper_ids": _sanitize_collection_paper_ids(
            session,
            collection_id=safe_collection_id,
            paper_ids=payload.paper_ids,
        ),
    }
    active_job = _find_matching_active_collection_job(
        session,
        job_type="collection_parse",
        job_payload=job_payload,
    )
    if active_job is not None:
        return SingleBackgroundJobResponse(data=background_job_to_response(active_job))

    job = create_background_job(
        session_factory=request.app.state.session_factory,
        job_type="collection_parse",
        payload_json=job_payload,
        dispatcher=request.app.state.job_dispatcher,
    )

    return SingleBackgroundJobResponse(data=background_job_to_response(job))


@router.post(
    "/api/v1/annotations",
    response_model=SingleAnnotationResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_annotation(
    payload: AnnotationCreateRequest,
    session: Session = Depends(get_session),
) -> SingleAnnotationResponse:
    annotation_repository = AnnotationRepository(session)
    collection_repository = CollectionRepository(session)
    paper_repository = PaperRepository(session)

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

    target_type = sanitize_user_text(payload.target_type, field_name="target_type", max_length=64)
    target_id = sanitize_identifier(payload.target_id, field_name="target_id", max_length=36)
    if target_type == "paper" and paper_repository.get_by_id(target_id) is None:
        raise PaperbaseAPIError(
            status_code=404,
            error="paper_not_found",
            message=f"No paper found for id: {target_id}",
        )

    annotation = annotation_repository.create(
        author_id=sanitize_user_text(payload.author_id, field_name="author_id", max_length=128),
        collection_id=collection_id,
        target_type=target_type,
        target_id=target_id,
        body=sanitize_user_text(payload.body, field_name="body", max_length=20000),
        tags=list(payload.tags),
        status=payload.status,
    )
    return SingleAnnotationResponse(data=_annotation_to_response(annotation))


@router.get("/api/v1/annotations", response_model=AnnotationsResponse)
def list_annotations(
    target_type: str = Query(..., min_length=1, max_length=64),
    target_id: str = Query(..., min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> AnnotationsResponse:
    annotation_repository = AnnotationRepository(session)
    annotations = annotation_repository.list_for_target(
        target_type=sanitize_user_text(target_type, field_name="target_type", max_length=64),
        target_id=sanitize_identifier(target_id, field_name="target_id", max_length=36),
    )
    return AnnotationsResponse(data=[_annotation_to_response(annotation) for annotation in annotations])
