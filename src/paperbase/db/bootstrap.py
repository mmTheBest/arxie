"""Database bootstrap helpers for Paperbase."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import inspect, text
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
    ensure_database_schema_compatible(engine)
    return engine


def ensure_database_schema_compatible(engine: Engine) -> None:
    """Create schema and apply small additive compatibility upgrades."""

    Base.metadata.create_all(engine)
    _ensure_study_source_file_metadata_columns(engine)
    _ensure_research_agent_attempt_number_columns(engine)


def _ensure_study_source_file_metadata_columns(engine: Engine) -> None:
    inspector = inspect(engine)
    if "study_sources" not in inspector.get_table_names():
        return
    column_names = {column["name"] for column in inspector.get_columns("study_sources")}
    missing_columns = [
        column_name
        for column_name in ("source_size_bytes", "source_mtime_ns")
        if column_name not in column_names
    ]
    if not missing_columns:
        return

    with engine.begin() as connection:
        for column_name in missing_columns:
            connection.execute(
                text(f"ALTER TABLE study_sources ADD COLUMN {column_name} INTEGER")
            )


def _ensure_research_agent_attempt_number_columns(engine: Engine) -> None:
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    attempt_tables = (
        "research_agent_steps",
        "study_context_packs",
        "research_validation_reports",
    )
    missing_attempt_tables = [
        table_name
        for table_name in attempt_tables
        if table_name in table_names
        and "attempt_number"
        not in {column["name"] for column in inspector.get_columns(table_name)}
    ]
    if not missing_attempt_tables:
        return

    with engine.begin() as connection:
        for table_name in missing_attempt_tables:
            connection.execute(
                text(
                    f"ALTER TABLE {table_name} "
                    "ADD COLUMN attempt_number INTEGER NOT NULL DEFAULT 1"
                )
            )
