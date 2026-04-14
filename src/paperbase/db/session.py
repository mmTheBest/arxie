"""SQLAlchemy session helpers for Paperbase."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from paperbase.config import load_paperbase_config


def make_engine(database_url: str | None = None) -> Engine:
    url = database_url or load_paperbase_config().database_url
    return create_engine(url, future=True)


def make_session_factory(database_url: str | None = None) -> sessionmaker[Session]:
    engine = make_engine(database_url=database_url)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

