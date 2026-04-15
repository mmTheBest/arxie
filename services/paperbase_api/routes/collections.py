"""Collection and annotation routes for the Paperbase API service."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Path, Query, status
from sqlalchemy.orm import Session

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
    CollectionsResponse,
    CollectionSummaryResponse,
    PaperSummaryResponse,
    SingleAnnotationResponse,
    SingleCollectionPaperResponse,
    SingleCollectionResponse,
)

router = APIRouter(tags=["collections"])


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
