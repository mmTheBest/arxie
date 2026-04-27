from __future__ import annotations

from pathlib import Path

from sqlalchemy import inspect

from paperbase.db.bootstrap import initialize_database


def test_initialize_database_creates_sqlite_parent_directory_and_schema(tmp_path: Path) -> None:
    database_path = tmp_path / "nested" / "paperbase.sqlite3"

    engine = initialize_database(f"sqlite:///{database_path}")

    assert database_path.parent.exists()
    assert database_path.exists()

    inspector = inspect(engine)
    assert "papers" in inspector.get_table_names()
    assert "collections" in inspector.get_table_names()
    assert "annotations" in inspector.get_table_names()
