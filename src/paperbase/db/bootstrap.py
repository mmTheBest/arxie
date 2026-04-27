"""Database bootstrap helpers for Paperbase."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy.engine import Engine, make_url

from paperbase.config import load_paperbase_config
from paperbase.db.models import Base
from paperbase.db.session import make_engine


def _ensure_sqlite_parent_directory(database_url: str) -> None:
    url = make_url(database_url)
    if url.get_backend_name() != "sqlite":
        return

    database = url.database
    if not database or database == ":memory:":
        return

    database_path = Path(database)
    if not database_path.is_absolute():
        database_path = Path.cwd() / database_path

    database_path.parent.mkdir(parents=True, exist_ok=True)


def initialize_database(database_url: str | None = None) -> Engine:
    """Create local schema and return the configured engine."""

    resolved_url = database_url or load_paperbase_config().database_url
    _ensure_sqlite_parent_directory(resolved_url)
    engine = make_engine(resolved_url)
    Base.metadata.create_all(engine)
    return engine
