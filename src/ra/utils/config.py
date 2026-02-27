"""Environment-based configuration for the Research Assistant.

This module centralizes environment variable parsing and ensures required secrets
are present. It is intentionally lightweight (no pydantic) to keep runtime
surface small.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping


def mask_secret(value: str | None) -> str | None:
    """Mask a secret for logging.

    Shows only the last 4 characters (if present). Examples:
      - None -> None
      - "" -> ""
      - "abcd" -> "***abcd"
      - "sk-123456" -> "***3456"

    Note: We intentionally do *not* preserve length.
    """

    if value is None:
        return None
    if value == "":
        return ""
    tail = value[-4:]
    return f"***{tail}"


@dataclass(frozen=True, slots=True)
class RAConfig:
    openai_api_key: str
    ra_model: str = "gpt-4o-mini"
    semantic_scholar_api_key: str | None = None
    ra_log_dir: str = "data/logs"
    ra_log_level: str = "INFO"

    def to_log_dict(self) -> dict[str, str | None]:
        """A safe-to-log view of config values."""

        return {
            "openai_api_key": mask_secret(self.openai_api_key),
            "ra_model": self.ra_model,
            "semantic_scholar_api_key": mask_secret(self.semantic_scholar_api_key),
            "ra_log_dir": self.ra_log_dir,
            "ra_log_level": self.ra_log_level,
        }

    def __repr__(self) -> str:  # pragma: no cover
        # Avoid leaking secrets through debug logs or exceptions.
        safe = self.to_log_dict()
        return "RAConfig(" + ", ".join(f"{k}={safe[k]!r}" for k in safe.keys()) + ")"


def load_config(env: Mapping[str, str] | None = None) -> RAConfig:
    """Load config from environment variables.

    Required:
      - OPENAI_API_KEY

    Optional:
      - RA_MODEL (default: gpt-4o-mini)
      - SEMANTIC_SCHOLAR_API_KEY
      - RA_LOG_DIR (default: data/logs)
      - RA_LOG_LEVEL (default: INFO)
    """

    env = env or os.environ

    openai_api_key = (env.get("OPENAI_API_KEY") or "").strip()
    if not openai_api_key:
        raise ValueError(
            "Missing required environment variable: OPENAI_API_KEY. "
            "Set it to your OpenAI API key (e.g., via a .env file or shell env)."
        )

    return RAConfig(
        openai_api_key=openai_api_key,
        ra_model=(env.get("RA_MODEL") or "gpt-4o-mini").strip() or "gpt-4o-mini",
        semantic_scholar_api_key=(env.get("SEMANTIC_SCHOLAR_API_KEY") or "").strip()
        or None,
        ra_log_dir=(env.get("RA_LOG_DIR") or "data/logs").strip() or "data/logs",
        ra_log_level=(env.get("RA_LOG_LEVEL") or "INFO").strip() or "INFO",
    )
