"""Database primitives for Paperbase."""

from paperbase.db.bootstrap import initialize_database
from paperbase.db.models import Base

__all__ = ["Base", "initialize_database"]
