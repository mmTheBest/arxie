"""Extraction profile and collection extraction routes for the Paperbase API service."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Path, Query, Request, status
from sqlalchemy.orm import Session

from paperbase.db.repositories import (
    CollectionRepository,
    ExtractionProfileRepository,
)
from paperbase.extract.client import OpenAIExtractionClient
from paperbase.profiles import get_profile_preset, list_profile_presets
from ra.utils.security import sanitize_identifier, sanitize_user_text
from services.paperbase_api.background_jobs import create_background_job
from services.paperbase_api.dependencies import get_session
from services.paperbase_api.errors import PaperbaseAPIError
from services.paperbase_api.models import (
    ExtractionProfileCreateRequest,
    ExtractionProfilePresetResponse,
    ExtractionProfilePresetsResponse,
    ExtractionProfileResponse,
    ExtractionProfilesResponse,
    RunCollectionExtractionRequest,
    SingleBackgroundJobResponse,
    SingleExtractionProfileResponse,
)
from services.paperbase_api.routes.jobs import background_job_to_response

router = APIRouter(tags=["extraction"])


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


@router.get(
    "/api/v1/extraction-profile-presets",
    response_model=ExtractionProfilePresetsResponse,
)
def list_extraction_profile_presets() -> ExtractionProfilePresetsResponse:
    presets = list_profile_presets()
    return ExtractionProfilePresetsResponse(
        data=[
            ExtractionProfilePresetResponse(
                name=preset["name"],
                title=preset["title"],
                domain=preset["domain"],
                description=preset["description"],
                schema_payload=dict(preset["schema_payload"]),
            )
            for preset in presets
        ]
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
    schema_payload = dict(payload.schema_payload)
    if payload.preset_name is not None:
        preset_name = sanitize_identifier(
            payload.preset_name,
            field_name="preset_name",
            max_length=128,
        )
        preset = get_profile_preset(preset_name)
        if preset is None:
            raise PaperbaseAPIError(
                status_code=400,
                error="invalid_input",
                message=f"Unknown extraction profile preset: {preset_name}",
            )
        if not schema_payload:
            schema_payload = dict(preset["schema_payload"])
    if not schema_payload:
        raise PaperbaseAPIError(
            status_code=400,
            error="invalid_input",
            message="Provide schema_payload or a valid preset_name.",
        )
    profile = repository.create(
        owner_id=sanitize_user_text(payload.owner_id, field_name="owner_id", max_length=128),
        name=sanitize_user_text(payload.name, field_name="name", max_length=255),
        description=payload.description,
        scope_type=payload.scope_type,
        schema_payload=schema_payload,
        active=payload.active,
    )
    return SingleExtractionProfileResponse(data=_profile_to_response(profile))


@router.post(
    "/api/v1/collections/{collection_id}/extract",
    response_model=SingleBackgroundJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def extract_collection(
    payload: RunCollectionExtractionRequest,
    request: Request,
    collection_id: str = Path(..., min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> SingleBackgroundJobResponse:
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

    job = create_background_job(
        session_factory=request.app.state.session_factory,
        job_type="collection_extract",
        payload_json={
            "collection_id": safe_collection_id,
            "schema_payload": schema_payload,
            "prompt_version": sanitize_user_text(
                payload.prompt_version,
                field_name="prompt_version",
                max_length=64,
            ),
            "schema_version": sanitize_user_text(
                payload.schema_version,
                field_name="schema_version",
                max_length=64,
            ),
            "extraction_profile_id": extraction_profile_id,
            "limit": payload.limit,
        },
        dispatcher=request.app.state.job_dispatcher,
    )
    return SingleBackgroundJobResponse(
        data=background_job_to_response(job)
    )


def default_extraction_client_factory() -> object:
    return OpenAIExtractionClient()
