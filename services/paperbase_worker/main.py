"""Production entrypoint for the Paperbase worker service."""

from __future__ import annotations

from pathlib import Path

from paperbase.config import load_paperbase_config
from paperbase.db.session import make_session_factory
from paperbase.jobs import build_job_queue
from paperbase.object_store import build_object_store
from paperbase.projects import ProjectRegistry
from paperbase.research.model_client import default_research_model_client
from paperbase.search import ElasticsearchSearchBackend
from paperbase.search.embeddings import build_embedding_provider
from ra.utils.logging_config import configure_logging_from_env
from services.paperbase_api.routes.extraction import default_extraction_client_factory
from services.paperbase_worker.runtime import PaperbaseWorker


def _build_search_backend(config):
    if not config.require_search_backend:
        return None
    return ElasticsearchSearchBackend(base_url=config.elasticsearch_url)


def build_worker() -> PaperbaseWorker:
    config = load_paperbase_config()
    job_queue = build_job_queue(config, project_id=config.worker_project_id)
    object_store = build_object_store(config)
    object_store.ensure_bucket()
    session_factory = (
        ProjectRegistry(registry_path=config.project_registry_path).session_factory_for(
            config.worker_project_id
        )
        if config.worker_project_id
        else make_session_factory(config.database_url)
    )
    return PaperbaseWorker(
        session_factory=session_factory,
        search_backend=_build_search_backend(config),
        extraction_client_factory=default_extraction_client_factory,
        research_model_client_factory=default_research_model_client,
        job_consumer=job_queue,
        job_dispatcher=job_queue,
        project_id=config.worker_project_id,
        retry_limit=config.worker_retry_limit,
        embedding_provider=build_embedding_provider(config),
        object_store=object_store,
        download_cache_dir=Path(config.download_cache_dir),
        download_cache_ttl_seconds=config.download_cache_ttl_seconds,
        stale_running_seconds=config.worker_stale_job_seconds,
        backend_retrieval_enabled=config.agent_context_backend_retrieval,
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
