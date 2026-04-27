"""Production entrypoint for the Paperbase worker service."""

from __future__ import annotations

import time

from paperbase.config import load_paperbase_config
from paperbase.db.session import make_session_factory
from paperbase.search import ElasticsearchSearchBackend
from ra.utils.logging_config import configure_logging_from_env
from services.paperbase_api.routes.extraction import default_extraction_client_factory
from services.paperbase_worker.runtime import PaperbaseWorker


def build_worker() -> PaperbaseWorker:
    config = load_paperbase_config()
    return PaperbaseWorker(
        session_factory=make_session_factory(config.database_url),
        search_backend=ElasticsearchSearchBackend(base_url=config.elasticsearch_url),
        extraction_client_factory=default_extraction_client_factory,
    )


def main() -> None:
    config = load_paperbase_config()
    configure_logging_from_env()
    worker = build_worker()

    while True:
        processed_job_id = worker.process_next_job()
        if processed_job_id is None:
            time.sleep(config.worker_poll_interval_seconds)


if __name__ == "__main__":
    main()
