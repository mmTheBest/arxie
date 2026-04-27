"""Operational database commands for Paperbase."""

from __future__ import annotations

from pathlib import Path

import typer
from alembic import command
from alembic.config import Config

from paperbase.config import load_paperbase_config
from paperbase.object_store import build_object_store

app = typer.Typer(help="Paperbase database operations.")


def _alembic_config() -> Config:
    repo_root = Path(__file__).resolve().parents[3]
    config = Config(str(repo_root / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", load_paperbase_config().database_url)
    return config


@app.command("upgrade")
def upgrade(revision: str = "head") -> None:
    """Apply Paperbase Alembic migrations."""

    command.upgrade(_alembic_config(), revision)
    build_object_store(load_paperbase_config()).ensure_bucket()


@app.command("current")
def current() -> None:
    """Print the current Paperbase migration revision."""

    command.current(_alembic_config())


def main() -> None:
    app()


if __name__ == "__main__":
    main()
