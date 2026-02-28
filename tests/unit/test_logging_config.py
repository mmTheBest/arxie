from __future__ import annotations

import json
import logging

from ra.utils.config import RAConfig
from ra.utils.logging import setup_logging
from ra.utils.logging_config import configure_logging, parse_log_level


def _reset_root_logger() -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(logging.NOTSET)
    setattr(root, "_ra_logging_configured", False)


def test_parse_log_level_defaults_to_info_for_invalid_values() -> None:
    assert parse_log_level("DEBUG") == logging.DEBUG
    assert parse_log_level("warning") == logging.WARNING
    assert parse_log_level(None) == logging.INFO
    assert parse_log_level("not-a-level") == logging.INFO


def test_configure_logging_writes_structured_json(tmp_path) -> None:
    _reset_root_logger()

    configure_logging(log_dir=tmp_path, log_level="DEBUG")
    logger = logging.getLogger("ra.test.logging")
    logger.debug(
        "structured logging works",
        extra={"event": "unit_test_event", "paper_id": "paper-123"},
    )

    for handler in logging.getLogger().handlers:
        handler.flush()

    line = (tmp_path / "ra.log").read_text(encoding="utf-8").splitlines()[-1]
    payload = json.loads(line)
    assert payload["level"] == "DEBUG"
    assert payload["logger"] == "ra.test.logging"
    assert payload["message"] == "structured logging works"
    assert payload["event"] == "unit_test_event"
    assert payload["paper_id"] == "paper-123"


def test_setup_logging_honors_configured_log_level(tmp_path) -> None:
    _reset_root_logger()

    cfg = RAConfig(
        openai_api_key="sk-test-0000",
        ra_model="gpt-4o-mini",
        semantic_scholar_api_key=None,
        ra_log_dir=str(tmp_path),
        ra_log_level="WARNING",
    )
    setup_logging(cfg)
    logger = logging.getLogger("ra.test.bridge")

    logger.info("this should not be written")
    logger.warning("this should be written", extra={"event": "warning_event"})

    for handler in logging.getLogger().handlers:
        handler.flush()

    lines = (tmp_path / "ra.log").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["level"] == "WARNING"
    assert payload["message"] == "this should be written"
    assert payload["event"] == "warning_event"
