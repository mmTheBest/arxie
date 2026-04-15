"""Extraction profile and collection extraction routes for the Paperbase API service."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, Depends, Path, Query, Request, status
from sqlalchemy.orm import Session

from paperbase.db.repositories import CollectionRepository, ExtractionProfileRepository
from paperbase.extract.client import OpenAIExtractionClient
from paperbase.extract.runner import CollectionExtractionRunner
from ra.utils.security import sanitize_identifier, sanitize_user_text
from services.paperbase_api.dependencies import get_session
from services.paperbase_api.errors import PaperbaseAPIError
from services.paperbase_api.models import (
    CollectionExtractionSummaryResponse,
    ExtractionProfileCreateRequest,
    ExtractionProfileResponse,
    ExtractionProfilesResponse,
    RunCollectionExtractionRequest,
    SingleCollectionExtractionSummaryResponse,
    SingleExtractionProfileResponse,
)

router = APIRouter(tags=["extraction"])


def get_extraction_client_factory(request: Request) -> Callable[[], object]:
    return request.app.state.extraction_client_factory


def _profile_to_response(profile) -> ExtractionProfileResponse:  # noqa: ANN001
    return ExtractionProfileResponse(
        id=profile.id,
        owner_id=profile.owner_id,
        name=profile.name,
        description=profile.description,
        scope_type=profile.scope_type,
        schema_payload=dict(profile.schema_payload or {}),
        active=profile.active,
    )


@router.get("/api/v1/extraction-profiles", response_model=ExtractionProfilesResponse)
def list_extraction_profiles(
    owner_id: str | None = Query(None, min_length=1, max_length=128),
    session: Session = Depends(get_session),
) -> ExtractionProfilesResponse:
    repository = ExtractionProfileRepository(session)
    safe_owner_id = (
        sanitize_user_text(owner_id, field_name="owner_id", max_length=128)
        if owner_id is not None
        else None
    )
    profiles = repository.list_profiles(owner_id=safe_owner_id)
    return ExtractionProfilesResponse(data=[_profile_to_response(profile) for profile in profiles])


@router.post(
    "/api/v1/extraction-profiles",
    response_model=SingleExtractionProfileResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_extraction_profile(
    payload: ExtractionProfileCreateRequest,
    session: Session = Depends(get_session),
) -> SingleExtractionProfileResponse:
    repository = ExtractionProfileRepository(session)
    profile = repository.create(
        owner_id=sanitize_user_text(payload.owner_id, field_name="owner_id", max_length=128),
        name=sanitize_user_text(payload.name, field_name="name", max_length=255),
        description=payload.description,
        scope_type=payload.scope_type,
        schema_payload=dict(payload.schema_payload),
        active=payload.active,
    )
    return SingleExtractionProfileResponse(data=_profile_to_response(profile))


@router.post(
    "/api/v1/collections/{collection_id}/extract",
    response_model=SingleCollectionExtractionSummaryResponse,
)
def extract_collection(
    payload: RunCollectionExtractionRequest,
    request: Request,
    collection_id: str = Path(..., min_length=1, max_length=36),
    session: Session = Depends(get_session),
    extraction_client_factory: Callable[[], object] = Depends(get_extraction_client_factory),
) -> SingleCollectionExtractionSummaryResponse:
    collection_repository = CollectionRepository(session)
    extraction_profile_repository = ExtractionProfileRepository(session)

    safe_collection_id = sanitize_identifier(collection_id, field_name="collection_id", max_length=36)
    collection = collection_repository.get_by_id(safe_collection_id)
    if collection is None:
        raise PaperbaseAPIError(
            status_code=404,
            error="collection_not_found",
            message=f"No collection found for id: {safe_collection_id}",
        )

    schema_payload = dict(payload.schema_payload)
    extraction_profile_id = None
    if payload.extraction_profile_id is not None:
        extraction_profile_id = sanitize_identifier(
            payload.extraction_profile_id,
            field_name="extraction_profile_id",
            max_length=36,
        )
        profile = extraction_profile_repository.get_by_id(extraction_profile_id)
        if profile is None:
            raise PaperbaseAPIError(
                status_code=404,
                error="extraction_profile_not_found",
                message=f"No extraction profile found for id: {extraction_profile_id}",
            )
        if not schema_payload:
            schema_payload = dict(profile.schema_payload or {})

    if not schema_payload:
        raise PaperbaseAPIError(
            status_code=400,
            error="invalid_input",
            message="Provide schema_payload or extraction_profile_id with a stored schema.",
        )

    session_factory = request.app.state.session_factory
    runner = CollectionExtractionRunner(
        session_factory=session_factory,
        client=extraction_client_factory(),
    )
    summary = runner.extract_collection(
        collection_id=safe_collection_id,
        schema_payload=schema_payload,
        prompt_version=sanitize_user_text(payload.prompt_version, field_name="prompt_version", max_length=64),
        schema_version=sanitize_user_text(payload.schema_version, field_name="schema_version", max_length=64),
        extraction_profile_id=extraction_profile_id,
        limit=payload.limit,
    )
    return SingleCollectionExtractionSummaryResponse(
        data=CollectionExtractionSummaryResponse(
            collection_id=summary.collection_id,
            extracted_paper_count=summary.extracted_paper_count,
            extraction_run_ids=list(summary.extraction_run_ids),
            skipped_paper_ids=list(summary.skipped_paper_ids),
        )
    )


def default_extraction_client_factory() -> object:
    return OpenAIExtractionClient()
