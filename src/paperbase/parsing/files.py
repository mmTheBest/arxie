"""Helpers for selecting parseable paper source files."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from paperbase.db.models import PaperFile


def load_active_pdf_file(session: Session, *, paper_id: str) -> PaperFile | None:
    """Return the newest PDF file row for a paper.

    Local reimports can add a new content-hash-specific file row while older
    rows remain for provenance. The newest row is the active parse source.
    """

    return session.execute(
        select(PaperFile)
        .where(
            PaperFile.paper_id == paper_id,
            PaperFile.file_kind == "pdf",
        )
        .order_by(PaperFile.created_at.desc(), PaperFile.id.desc())
        .limit(1)
    ).scalar_one_or_none()
