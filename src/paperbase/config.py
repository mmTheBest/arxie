"""Configuration helpers for the Paperbase platform layer."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PaperbaseConfig:
    database_url: str = "sqlite:///data/paperbase.db"
    elasticsearch_url: str = "http://localhost:9200"
    redis_url: str = "redis://localhost:6379/0"
    object_store_endpoint: str = "http://localhost:9000"
    object_store_bucket: str = "paperbase"


def load_paperbase_config() -> PaperbaseConfig:
    """Load local-first Paperbase configuration from environment variables."""

    env = os.environ
    return PaperbaseConfig(
        database_url=(env.get("PAPERBASE_DATABASE_URL") or "sqlite:///data/paperbase.db").strip(),
        elasticsearch_url=(env.get("PAPERBASE_ELASTICSEARCH_URL") or "http://localhost:9200").strip(),
        redis_url=(env.get("PAPERBASE_REDIS_URL") or "redis://localhost:6379/0").strip(),
        object_store_endpoint=(env.get("PAPERBASE_OBJECT_STORE_ENDPOINT") or "http://localhost:9000").strip(),
        object_store_bucket=(env.get("PAPERBASE_OBJECT_STORE_BUCKET") or "paperbase").strip(),
    )

