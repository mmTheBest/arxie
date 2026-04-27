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
    worker_queue_backend: str = "db"
    worker_queue_name: str = "paperbase:jobs"
    worker_retry_limit: int = 3
    worker_stale_job_seconds: float = 900.0
    object_store_endpoint: str = "http://localhost:9000"
    object_store_backend: str = "filesystem"
    object_store_bucket: str = "paperbase"
    object_store_access_key: str | None = None
    object_store_secret_key: str | None = None
    object_store_local_root: str = "data/object-store"
    download_cache_dir: str = "data/cache/paperbase-downloads"
    download_cache_ttl_seconds: int = 86400
    embedding_provider: str = "auto"
    embedding_model: str = "text-embedding-3-small"
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
        worker_queue_backend=(
            resolved_env.get("PAPERBASE_WORKER_QUEUE_BACKEND") or "db"
        ).strip(),
        worker_queue_name=(
            resolved_env.get("PAPERBASE_WORKER_QUEUE_NAME") or "paperbase:jobs"
        ).strip(),
        worker_retry_limit=int((resolved_env.get("PAPERBASE_WORKER_RETRY_LIMIT") or "3").strip()),
        worker_stale_job_seconds=float(
            (resolved_env.get("PAPERBASE_WORKER_STALE_JOB_SECONDS") or "900").strip()
        ),
        object_store_endpoint=(
            resolved_env.get("PAPERBASE_OBJECT_STORE_ENDPOINT") or "http://localhost:9000"
        ).strip(),
        object_store_backend=(
            resolved_env.get("PAPERBASE_OBJECT_STORE_BACKEND") or "filesystem"
        ).strip(),
        object_store_bucket=(
            resolved_env.get("PAPERBASE_OBJECT_STORE_BUCKET") or "paperbase"
        ).strip(),
        object_store_access_key=(
            (resolved_env.get("PAPERBASE_OBJECT_STORE_ACCESS_KEY") or "").strip() or None
        ),
        object_store_secret_key=(
            (resolved_env.get("PAPERBASE_OBJECT_STORE_SECRET_KEY") or "").strip() or None
        ),
        object_store_local_root=(
            resolved_env.get("PAPERBASE_OBJECT_STORE_LOCAL_ROOT") or "data/object-store"
        ).strip(),
        download_cache_dir=(
            resolved_env.get("PAPERBASE_DOWNLOAD_CACHE_DIR") or "data/cache/paperbase-downloads"
        ).strip(),
        download_cache_ttl_seconds=int(
            (resolved_env.get("PAPERBASE_DOWNLOAD_CACHE_TTL_SECONDS") or "86400").strip()
        ),
        embedding_provider=(resolved_env.get("PAPERBASE_EMBEDDING_PROVIDER") or "auto").strip(),
        embedding_model=(
            resolved_env.get("PAPERBASE_EMBEDDING_MODEL") or "text-embedding-3-small"
        ).strip(),
        api_host=(resolved_env.get("PAPERBASE_API_HOST") or "0.0.0.0").strip(),
        api_port=int((resolved_env.get("PAPERBASE_API_PORT") or "8080").strip()),
        worker_poll_interval_seconds=float(
            (resolved_env.get("PAPERBASE_WORKER_POLL_INTERVAL_SECONDS") or "2.0").strip()
        ),
    )
