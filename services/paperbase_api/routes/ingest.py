"""Ingest routes for Paperbase local-first corpus operations."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.orm import Session

from ra.utils.security import sanitize_identifier, sanitize_user_text
from services.paperbase_api.background_jobs import create_background_job
from services.paperbase_api.dependencies import get_project_id, get_session, get_session_factory
from services.paperbase_api.errors import PaperbaseAPIError
from services.paperbase_api.models import (
    LocalLibraryIngestRequest,
    PaperMetadataRefreshRequest,
    ProviderIdentifierIngestRequest,
    SingleBackgroundJobResponse,
)
from services.paperbase_api.path_policy import ensure_host_path_allowed
from services.paperbase_api.routes.jobs import background_job_to_response
from services.paperbase_api.upload_parser import stage_streamed_local_library_upload

router = APIRouter(tags=["ingest"])


def _cleanup_staged_upload(staged_dir: Path) -> None:
    shutil.rmtree(staged_dir, ignore_errors=True)


@router.post(
    "/api/v1/ingest/local-library",
    response_model=SingleBackgroundJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def queue_local_library_ingest(
    payload: LocalLibraryIngestRequest,
    request: Request,
    session: Session = Depends(get_session),
) -> SingleBackgroundJobResponse:
    source_dir = Path(
        sanitize_user_text(payload.source_dir, field_name="source_dir", max_length=4096)
    ).expanduser()
    source_dir = ensure_host_path_allowed(
        source_dir,
        config=request.app.state.paperbase_config,
        field_name="source_dir",
    )
    if not source_dir.exists():
        raise PaperbaseAPIError(
            status_code=404,
            error="source_dir_not_found",
            message=f"No local directory found at: {source_dir}",
        )
    if not source_dir.is_dir():
        raise PaperbaseAPIError(
            status_code=400,
            error="invalid_input",
            message=f"source_dir is not a directory: {source_dir}",
        )

    job = create_background_job(
        session_factory=get_session_factory(request),
        job_type="local_library_ingest",
        payload_json={
            "source_dir": str(source_dir),
            "owner_id": sanitize_user_text(
                payload.owner_id,
                field_name="owner_id",
                max_length=128,
            ),
            "collection_title": payload.collection_title,
            "collection_description": payload.collection_description,
        },
        dispatcher=request.app.state.job_dispatcher,
        project_id=get_project_id(request),
    )

    return SingleBackgroundJobResponse(data=background_job_to_response(job))


@router.post(
    "/api/v1/ingest/local-library-upload",
    response_model=SingleBackgroundJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["files"],
                        "properties": {
                            "files": {
                                "type": "array",
                                "items": {"type": "string", "format": "binary"},
                            },
                            "owner_id": {"type": "string", "default": "local-user"},
                            "collection_title": {"type": "string"},
                            "collection_description": {"type": "string"},
                        },
                    }
                }
            },
        }
    },
)
async def queue_local_library_upload_ingest(
    request: Request,
    session: Session = Depends(get_session),
) -> SingleBackgroundJobResponse:
    del session
    staging_root = Path(request.app.state.upload_staging_dir).expanduser()
    upload = await stage_streamed_local_library_upload(
        request=request,
        staging_root=staging_root,
        max_file_count=request.app.state.upload_max_file_count,
        max_single_file_bytes=request.app.state.upload_max_single_file_bytes,
        max_total_bytes=request.app.state.upload_max_total_bytes,
    )

    try:
        safe_owner_id = sanitize_user_text(upload.owner_id, field_name="owner_id", max_length=128)
        safe_collection_title = (
            sanitize_user_text(upload.collection_title, field_name="collection_title", max_length=255)
            if upload.collection_title is not None and upload.collection_title.strip()
            else None
        )
        safe_collection_description = (
            sanitize_user_text(
                upload.collection_description,
                field_name="collection_description",
                max_length=5000,
            )
            if upload.collection_description is not None and upload.collection_description.strip()
            else None
        )

        job = create_background_job(
            session_factory=get_session_factory(request),
            job_type="local_library_ingest",
            payload_json={
                "source_dir": str(upload.staged_dir),
                "owner_id": safe_owner_id,
                "collection_title": safe_collection_title,
                "collection_description": safe_collection_description,
            },
            dispatcher=request.app.state.job_dispatcher,
            project_id=get_project_id(request),
        )
    except ValueError as exc:
        _cleanup_staged_upload(upload.staged_dir)
        raise PaperbaseAPIError(
            status_code=400,
            error="invalid_input",
            message=str(exc),
        ) from exc
    except Exception:
        _cleanup_staged_upload(upload.staged_dir)
        raise
    return SingleBackgroundJobResponse(data=background_job_to_response(job))


@router.post(
    "/api/v1/ingest/providers",
    response_model=SingleBackgroundJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def queue_provider_identifier_ingest(
    payload: ProviderIdentifierIngestRequest,
    request: Request,
    session: Session = Depends(get_session),
) -> SingleBackgroundJobResponse:
    owner_id = sanitize_user_text(payload.owner_id, field_name="owner_id", max_length=128)
    collection_id = (
        sanitize_identifier(payload.collection_id, field_name="collection_id", max_length=36)
        if payload.collection_id is not None
        else None
    )
    collection_title = (
        sanitize_user_text(payload.collection_title, field_name="collection_title", max_length=255)
        if payload.collection_title is not None
        else None
    )

    job = create_background_job(
        session_factory=get_session_factory(request),
        job_type="provider_identifier_ingest",
        payload_json={
            "owner_id": owner_id,
            "collection_id": collection_id,
            "collection_title": collection_title,
            "collection_description": payload.collection_description,
            "identifiers": [
                {
                    "kind": sanitize_identifier(
                        item.kind,
                        field_name="identifier_kind",
                        max_length=32,
                    ),
                    "value": sanitize_identifier(
                        item.value,
                        field_name="identifier_value",
                        max_length=512,
                    ),
                }
                for item in payload.identifiers
            ],
        },
        dispatcher=request.app.state.job_dispatcher,
        project_id=get_project_id(request),
    )

    return SingleBackgroundJobResponse(data=background_job_to_response(job))


@router.post(
    "/api/v1/ingest/refresh-metadata",
    response_model=SingleBackgroundJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def queue_paper_metadata_refresh(
    payload: PaperMetadataRefreshRequest,
    request: Request,
    session: Session = Depends(get_session),
) -> SingleBackgroundJobResponse:
    del session
    job = create_background_job(
        session_factory=get_session_factory(request),
        job_type="paper_metadata_refresh",
        payload_json={
            "paper_ids": [
                sanitize_identifier(paper_id, field_name="paper_id", max_length=36)
                for paper_id in payload.paper_ids
            ]
        },
        dispatcher=request.app.state.job_dispatcher,
        project_id=get_project_id(request),
    )

    return SingleBackgroundJobResponse(data=background_job_to_response(job))
