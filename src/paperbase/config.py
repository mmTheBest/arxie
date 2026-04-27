"""Configuration helpers for the Paperbase platform layer."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class PaperbaseConfig:
    database_url: str = "sqlite:///data/paperbase.db"
    elasticsearch_url: str = "http://localhost:9200"
    redis_url: str = "redis://localhost:6379/0"
    object_store_endpoint: str = "http://localhost:9000"
    object_store_bucket: str = "paperbase"
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    worker_poll_interval_seconds: float = 2.0


def load_paperbase_config(env: Mapping[str, str] | None = None) -> PaperbaseConfig:
    """Load Paperbase runtime configuration from environment variables."""

    resolved_env = env or os.environ
    return PaperbaseConfig(
        database_url=(
            resolved_env.get("PAPERBASE_DATABASE_URL")
            or "sqlite:///data/paperbase.db"
        ).strip(),
        elasticsearch_url=(
            resolved_env.get("PAPERBASE_ELASTICSEARCH_URL") or "http://localhost:9200"
        ).strip(),
        redis_url=(resolved_env.get("PAPERBASE_REDIS_URL") or "redis://localhost:6379/0").strip(),
        object_store_endpoint=(
            resolved_env.get("PAPERBASE_OBJECT_STORE_ENDPOINT") or "http://localhost:9000"
        ).strip(),
        object_store_bucket=(
            resolved_env.get("PAPERBASE_OBJECT_STORE_BUCKET") or "paperbase"
        ).strip(),
        api_host=(resolved_env.get("PAPERBASE_API_HOST") or "0.0.0.0").strip(),
        api_port=int((resolved_env.get("PAPERBASE_API_PORT") or "8080").strip()),
        worker_poll_interval_seconds=float(
            (resolved_env.get("PAPERBASE_WORKER_POLL_INTERVAL_SECONDS") or "2.0").strip()
        ),
    )
