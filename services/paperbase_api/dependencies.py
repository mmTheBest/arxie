"""Dependency helpers for the Paperbase API service."""

from __future__ import annotations

from collections.abc import Generator

from fastapi import Request
from sqlalchemy.orm import Session, sessionmaker


def get_session_factory(request: Request) -> sessionmaker[Session]:
    return request.app.state.session_factory


def get_session(request: Request) -> Generator[Session, None, None]:
    session_factory = get_session_factory(request)
    with session_factory() as session:
        yield session
