"""Production entrypoint for the Paperbase API service."""

from __future__ import annotations

import uvicorn

from paperbase.config import load_paperbase_config
from paperbase.db.session import make_session_factory
from paperbase.jobs import build_job_queue
from paperbase.search import ElasticsearchSearchBackend
from paperbase.search.embeddings import build_embedding_provider
from services.paperbase_api.app import create_app


def create_runtime_app():
    config = load_paperbase_config()
    return create_app(
        session_factory=make_session_factory(config.database_url),
        search_backend=ElasticsearchSearchBackend(base_url=config.elasticsearch_url),
        job_dispatcher=build_job_queue(config),
        embedding_provider=build_embedding_provider(config),
    )


def main() -> None:
    config = load_paperbase_config()
    uvicorn.run(
        create_runtime_app(),
        host=config.api_host,
        port=config.api_port,
    )


if __name__ == "__main__":
    main()
