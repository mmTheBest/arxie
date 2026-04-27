"""Shared background job enqueue helpers for the Paperbase API service."""

from __future__ import annotations

from paperbase.db.models import BackgroundJob
from paperbase.db.repositories import BackgroundJobRepository
from services.paperbase_api.errors import PaperbaseAPIError


def create_background_job(
    *,
    session_factory,
    job_type: str,
    payload_json: dict[str, object] | None,
    dispatcher: object | None = None,
) -> BackgroundJob:
    with session_factory() as session:
        job = BackgroundJobRepository(session).create(
            job_type=job_type,
            payload_json=payload_json or {},
        )

    if dispatcher is None:
        return job

    try:
        dispatcher.dispatch(job.id)
    except Exception as exc:  # noqa: BLE001
        with session_factory() as session:
            BackgroundJobRepository(session).mark_failed(
                job.id,
                error_message=f"dispatch failed: {exc}",
            )
        raise PaperbaseAPIError(
            status_code=503,
            error="job_dispatch_failed",
            message=f"Unable to dispatch background job {job.id}.",
        ) from exc

    return job
