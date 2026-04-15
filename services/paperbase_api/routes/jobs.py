"""Background job routes and serializers for the Paperbase API service."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Path
from sqlalchemy.orm import Session

from paperbase.db.models import BackgroundJob
from paperbase.db.repositories import BackgroundJobRepository
from ra.utils.security import sanitize_identifier
from services.paperbase_api.dependencies import get_session
from services.paperbase_api.errors import PaperbaseAPIError
from services.paperbase_api.models import BackgroundJobResponse, SingleBackgroundJobResponse

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


def _serialize_timestamp(value) -> str | None:  # noqa: ANN001
    if value is None:
        return None
    return value.isoformat()


def background_job_to_response(job: BackgroundJob) -> BackgroundJobResponse:
    return BackgroundJobResponse(
        id=job.id,
        job_type=job.job_type,
        status=job.status,
        payload=dict(job.payload_json or {}),
        result=dict(job.result_json) if job.result_json is not None else None,
        error_message=job.error_message,
        created_at=_serialize_timestamp(job.created_at),
        started_at=_serialize_timestamp(job.started_at),
        finished_at=_serialize_timestamp(job.finished_at),
    )


@router.get(
    "/{job_id}",
    response_model=SingleBackgroundJobResponse,
)
def get_background_job(
    job_id: str = Path(..., min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> SingleBackgroundJobResponse:
    repository = BackgroundJobRepository(session)
    safe_job_id = sanitize_identifier(job_id, field_name="job_id", max_length=36)
    job = repository.get_by_id(safe_job_id)
    if job is None:
        raise PaperbaseAPIError(
            status_code=404,
            error="job_not_found",
            message=f"No background job found for id: {safe_job_id}",
        )

    return SingleBackgroundJobResponse(data=background_job_to_response(job))
