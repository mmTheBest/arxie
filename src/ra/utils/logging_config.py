"""Structured logging configuration utilities."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_RA_HANDLER_ATTR = "_ra_logging_handlers"
_RA_CONFIGURED_ATTR = "_ra_logging_configured"
_DEFAULT_LOG_LEVEL = "INFO"
_DEFAULT_LOG_DIR = "data/logs"

_RESERVED_RECORD_KEYS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}


def parse_log_level(value: str | None) -> int:
    """Parse a user-provided logging level, defaulting to INFO for invalid values."""

    if value is None:
        return logging.INFO
    normalized = value.strip().upper()
    if not normalized:
        return logging.INFO
    level = getattr(logging, normalized, None)
    if isinstance(level, int):
        return level
    return logging.INFO


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


class StructuredJsonFormatter(logging.Formatter):
    """JSON formatter with stable top-level fields and optional extras."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key.startswith("_") or key in _RESERVED_RECORD_KEYS:
                continue
            payload[key] = _json_safe(value)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        return json.dumps(payload, ensure_ascii=False)


def _remove_existing_ra_handlers(root: logging.Logger) -> None:
    handlers = getattr(root, _RA_HANDLER_ATTR, ())
    for handler in list(handlers):
        try:
            root.removeHandler(handler)
        finally:
            handler.close()


def configure_logging(
    *,
    log_level: str | None = _DEFAULT_LOG_LEVEL,
    log_dir: str | Path = _DEFAULT_LOG_DIR,
) -> None:
    """Configure process-wide structured logging for console + file handlers."""

    root = logging.getLogger()
    _remove_existing_ra_handlers(root)

    level = parse_log_level(log_level)
    root.setLevel(level)

    formatter = StructuredJsonFormatter()

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path / "ra.log", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    root.addHandler(stream_handler)
    root.addHandler(file_handler)

    setattr(root, _RA_HANDLER_ATTR, (stream_handler, file_handler))
    setattr(root, _RA_CONFIGURED_ATTR, True)


def configure_logging_from_env() -> None:
    """Configure logging using environment variables with safe defaults."""

    configure_logging(
        log_level=os.getenv("RA_LOG_LEVEL", _DEFAULT_LOG_LEVEL),
        log_dir=os.getenv("RA_LOG_DIR", _DEFAULT_LOG_DIR),
    )
