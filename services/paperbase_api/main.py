"""Production entrypoint for the Paperbase API service."""

from __future__ import annotations

import uvicorn

from paperbase.config import load_paperbase_config
from paperbase.db.session import make_session_factory
from paperbase.search import ElasticsearchSearchBackend
from services.paperbase_api.app import create_app


def create_runtime_app():
    config = load_paperbase_config()
    return create_app(
        session_factory=make_session_factory(config.database_url),
        search_backend=ElasticsearchSearchBackend(base_url=config.elasticsearch_url),
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
