"""Production entrypoint for the Paperbase worker service."""

from __future__ import annotations

from pathlib import Path

from paperbase.config import load_paperbase_config
from paperbase.db.session import make_session_factory
from paperbase.jobs import build_job_queue
from paperbase.object_store import build_object_store
from paperbase.search import ElasticsearchSearchBackend
from paperbase.search.embeddings import build_embedding_provider
from ra.utils.logging_config import configure_logging_from_env
from services.paperbase_api.routes.extraction import default_extraction_client_factory
from services.paperbase_worker.runtime import PaperbaseWorker


def build_worker() -> PaperbaseWorker:
    config = load_paperbase_config()
    job_queue = build_job_queue(config)
    object_store = build_object_store(config)
    object_store.ensure_bucket()
    return PaperbaseWorker(
        session_factory=make_session_factory(config.database_url),
        search_backend=ElasticsearchSearchBackend(base_url=config.elasticsearch_url),
        extraction_client_factory=default_extraction_client_factory,
        job_consumer=job_queue,
        job_dispatcher=job_queue,
        retry_limit=config.worker_retry_limit,
        embedding_provider=build_embedding_provider(config),
        object_store=object_store,
        download_cache_dir=Path(config.download_cache_dir),
        download_cache_ttl_seconds=config.download_cache_ttl_seconds,
    )


def main() -> None:
    config = load_paperbase_config()
    configure_logging_from_env()
    worker = build_worker()

    while True:
        processed_job_id = worker.process_next_dispatched_job(
            timeout_seconds=config.worker_poll_interval_seconds
        )
        if processed_job_id is None:
            continue


if __name__ == "__main__":
    main()
