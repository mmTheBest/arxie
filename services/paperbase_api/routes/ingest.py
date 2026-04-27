"""Ingest routes for Paperbase local-first corpus operations."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile, status
from sqlalchemy.orm import Session

from ra.utils.security import sanitize_identifier, sanitize_user_text
from services.paperbase_api.background_jobs import create_background_job
from services.paperbase_api.dependencies import get_session
from services.paperbase_api.errors import PaperbaseAPIError
from services.paperbase_api.models import (
    LocalLibraryIngestRequest,
    PaperMetadataRefreshRequest,
    ProviderIdentifierIngestRequest,
    SingleBackgroundJobResponse,
)
from services.paperbase_api.routes.jobs import background_job_to_response

router = APIRouter(tags=["ingest"])


def _safe_uploaded_relative_path(filename: str | None) -> Path:
    raw_name = (filename or "").replace("\\", "/").strip()
    candidate = PurePosixPath(raw_name or "upload.pdf")
    safe_parts = [part for part in candidate.parts if part not in {"", ".", "..", "/"}]
    if not safe_parts:
        safe_parts = ["upload.pdf"]
    return Path(*safe_parts)


def _stage_uploaded_pdf_directory(
    *,
    staging_root: Path,
    files: list[UploadFile],
) -> Path:
    staged_dir = staging_root / uuid4().hex
    staged_dir.mkdir(parents=True, exist_ok=True)

    stored_any = False
    for upload in files:
        relative_path = _safe_uploaded_relative_path(upload.filename)
        if relative_path.suffix.lower() != ".pdf":
            continue

        destination = staged_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            while True:
                chunk = upload.file.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        stored_any = True

    if not stored_any:
        raise PaperbaseAPIError(
            status_code=400,
            error="invalid_input",
            message="At least one PDF file is required for local library upload.",
        )

    return staged_dir


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
        session_factory=request.app.state.session_factory,
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
    )

    return SingleBackgroundJobResponse(data=background_job_to_response(job))


@router.post(
    "/api/v1/ingest/local-library-upload",
    response_model=SingleBackgroundJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def queue_local_library_upload_ingest(
    request: Request,
    session: Session = Depends(get_session),
    files: list[UploadFile] = File(...),
    owner_id: str = Form("local-user"),
    collection_title: str | None = Form(None),
    collection_description: str | None = Form(None),
) -> SingleBackgroundJobResponse:
    del session
    staging_root = Path(request.app.state.upload_staging_dir).expanduser()
    staging_root.mkdir(parents=True, exist_ok=True)
    staged_dir = _stage_uploaded_pdf_directory(staging_root=staging_root, files=files)

    safe_owner_id = sanitize_user_text(owner_id, field_name="owner_id", max_length=128)
    safe_collection_title = (
        sanitize_user_text(collection_title, field_name="collection_title", max_length=255)
        if collection_title is not None and collection_title.strip()
        else None
    )
    safe_collection_description = (
        sanitize_user_text(
            collection_description,
            field_name="collection_description",
            max_length=5000,
        )
        if collection_description is not None and collection_description.strip()
        else None
    )

    job = create_background_job(
        session_factory=request.app.state.session_factory,
        job_type="local_library_ingest",
        payload_json={
            "source_dir": str(staged_dir),
            "owner_id": safe_owner_id,
            "collection_title": safe_collection_title,
            "collection_description": safe_collection_description,
        },
        dispatcher=request.app.state.job_dispatcher,
    )
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
        session_factory=request.app.state.session_factory,
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
        session_factory=request.app.state.session_factory,
        job_type="paper_metadata_refresh",
        payload_json={
            "paper_ids": [
                sanitize_identifier(paper_id, field_name="paper_id", max_length=36)
                for paper_id in payload.paper_ids
            ]
        },
        dispatcher=request.app.state.job_dispatcher,
    )

    return SingleBackgroundJobResponse(data=background_job_to_response(job))
