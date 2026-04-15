"""Ingest routes for Paperbase local-first corpus operations."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.orm import Session

from paperbase.db.repositories import BackgroundJobRepository
from ra.utils.security import sanitize_user_text
from services.paperbase_api.dependencies import get_session
from services.paperbase_api.errors import PaperbaseAPIError
from services.paperbase_api.models import LocalLibraryIngestRequest, SingleBackgroundJobResponse
from services.paperbase_api.routes.jobs import background_job_to_response

router = APIRouter(tags=["ingest"])


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

    with request.app.state.session_factory() as job_session:
        job = BackgroundJobRepository(job_session).create(
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
        )

    return SingleBackgroundJobResponse(data=background_job_to_response(job))
