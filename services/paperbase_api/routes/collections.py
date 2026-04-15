"""Collection and annotation routes for the Paperbase API service."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Path, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from paperbase.db.models import (
    CollectionPaper,
    Dataset,
    EngineeringTrick,
    ExtractionRun,
    GlossaryTerm,
    Method,
    Metric,
    Paper,
    ResultRow,
)
from paperbase.db.repositories import AnnotationRepository, CollectionRepository, PaperRepository
from ra.utils.security import sanitize_identifier, sanitize_user_text
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
    CollectionSummaryGlossaryTermResponse,
    CollectionSummaryNamedArtifactResponse,
    CollectionSummaryResultRowResponse,
    CollectionsResponse,
    CollectionSummaryResponse,
    PaperSummaryResponse,
    SingleAnnotationResponse,
    SingleCollectionPaperResponse,
    SingleCollectionResponse,
)
from services.paperbase_api.normalization import (
    canonicalize_metric_display_name,
    normalize_summary_key,
)

router = APIRouter(tags=["collections"])


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


def _paper_to_response(paper) -> PaperSummaryResponse:  # noqa: ANN001
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
    )


def _collection_to_response(collection) -> CollectionSummaryResponse:  # noqa: ANN001
    return CollectionSummaryResponse(
        id=collection.id,
        owner_id=collection.owner_id,
        scope_type=collection.scope_type,
        title=collection.title,
        description=collection.description,
        extraction_profile_id=collection.extraction_profile_id,
        tags=list(collection.tags_json or []),
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
    return CollectionsResponse(data=[_collection_to_response(collection) for collection in collections])


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
    return SingleCollectionResponse(data=_collection_to_response(collection))


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
    return SingleCollectionResponse(data=_collection_to_response(collection))


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
    data: list[CollectionPaperMembershipResponse] = []
    for membership in memberships:
        paper = paper_repository.get_by_id(membership.paper_id)
        if paper is None:
            continue
        data.append(
            CollectionPaperMembershipResponse(
                id=membership.id,
                collection_id=membership.collection_id,
                paper_id=membership.paper_id,
                position=membership.position,
                membership_note=membership.membership_note,
                paper=_paper_to_response(paper),
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

    paper_count = session.execute(
        select(func.count()).select_from(CollectionPaper).where(CollectionPaper.collection_id == safe_collection_id)
    ).scalar_one()
    extracted_paper_count = session.execute(
        select(func.count(func.distinct(ExtractionRun.paper_id)))
        .where(
            ExtractionRun.paper_id.in_(member_paper_ids),
            ExtractionRun.status == "completed",
        )
    ).scalar_one()

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
    glossary_terms = session.execute(
        select(GlossaryTerm)
        .where(GlossaryTerm.paper_id.in_(member_paper_ids))
        .order_by(GlossaryTerm.term.asc(), GlossaryTerm.created_at.asc())
    ).scalars().all()
    engineering_tricks = session.execute(
        select(EngineeringTrick)
        .where(EngineeringTrick.paper_id.in_(member_paper_ids))
        .order_by(EngineeringTrick.title.asc(), EngineeringTrick.created_at.asc())
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
            paper_count=paper_count,
            extracted_paper_count=extracted_paper_count,
            datasets=_build_named_artifact_summary(datasets, artifact_type="dataset"),
            methods=_build_named_artifact_summary(methods, artifact_type="method"),
            metrics=_build_named_artifact_summary(metrics, artifact_type="metric"),
            glossary_terms=[
                CollectionSummaryGlossaryTermResponse(
                    id=item.id,
                    term=item.term,
                    definition=item.definition,
                )
                for item in glossary_terms
            ],
            engineering_tricks=[
                CollectionSummaryEngineeringTrickResponse(
                    id=item.id,
                    title=item.title,
                    description=item.description,
                )
                for item in engineering_tricks
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
