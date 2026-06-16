"""Errors and exception handlers for the Paperbase API service."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from services.paperbase_api.models import ErrorResponse

logger = logging.getLogger(__name__)


class PaperbaseAPIError(Exception):
    """Typed API error with status code and machine-readable error key."""

    def __init__(
        self,
        *,
        status_code: int,
        error: str,
        message: str,
        details: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error = error
        self.message = message
        self.details = details


def _error_response(
    *,
    status_code: int,
    error: str,
    message: str,
    details: list[dict[str, Any]] | None = None,
) -> JSONResponse:
    payload = ErrorResponse(error=error, message=message, details=details)
    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(exclude_none=True),
    )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(PaperbaseAPIError)
    async def _paperbase_api_error_handler(  # noqa: ANN202
        request: Request,
        exc: PaperbaseAPIError,
    ):
        _ = request
        return _error_response(
            status_code=exc.status_code,
            error=exc.error,
            message=exc.message,
            details=exc.details,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(  # noqa: ANN202
        request: Request,
        exc: RequestValidationError,
    ):
        _ = request
        return _error_response(
            status_code=422,
            error="validation_error",
            message="Invalid request payload.",
            details=[_redact_validation_error(item) for item in exc.errors()],
        )

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(  # noqa: ANN202
        request: Request,
        exc: Exception,
    ):
        logger.exception("Unhandled Paperbase API exception for path=%s", request.url.path, exc_info=exc)
        return _error_response(
            status_code=500,
            error="internal_error",
            message="Internal server error.",
        )


def _redact_validation_error(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _redact_validation_error(item)
            for key, item in value.items()
            if key != "input"
        }
    if isinstance(value, list):
        return [_redact_validation_error(item) for item in value]
    return value
