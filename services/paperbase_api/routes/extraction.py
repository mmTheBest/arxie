"""Extraction profile and collection extraction routes for the Paperbase API service."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Path, Query, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from paperbase.db.models import BackgroundJob, CollectionPaper
from paperbase.db.repositories import (
    CollectionRepository,
    ExtractionProfileRepository,
)
from paperbase.extract.client import OpenAIExtractionClient
from paperbase.profiles import get_profile_preset, list_profile_presets
from ra.utils.security import sanitize_identifier, sanitize_user_text
from services.paperbase_api.background_jobs import create_background_job
from services.paperbase_api.dependencies import get_project_id, get_session, get_session_factory
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
ACTIVE_JOB_STATUSES = {"pending", "queued", "running"}


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


def _find_matching_active_extraction_job(
    session: Session,
    *,
    job_payload: dict[str, object],
) -> BackgroundJob | None:
    jobs = session.execute(
        select(BackgroundJob)
        .where(
            BackgroundJob.job_type == "collection_extract",
            BackgroundJob.status.in_(ACTIVE_JOB_STATUSES),
        )
        .order_by(BackgroundJob.created_at.desc(), BackgroundJob.id.desc())
    ).scalars()
    for job in jobs:
        payload_json = dict(job.payload_json or {})
        if payload_json == job_payload:
            return job
    return None


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

    job_payload = {
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
        "paper_ids": _sanitize_collection_paper_ids(
            session,
            collection_id=safe_collection_id,
            paper_ids=payload.paper_ids,
        ),
    }
    active_job = _find_matching_active_extraction_job(session, job_payload=job_payload)
    if active_job is not None:
        return SingleBackgroundJobResponse(data=background_job_to_response(active_job))

    job = create_background_job(
        session_factory=get_session_factory(request),
        job_type="collection_extract",
        payload_json=job_payload,
        dispatcher=request.app.state.job_dispatcher,
        project_id=get_project_id(request),
    )
    return SingleBackgroundJobResponse(
        data=background_job_to_response(job)
    )


def default_extraction_client_factory() -> object:
    return OpenAIExtractionClient()
