"""Paperbase-backed retrieval adapter for Arxie runtime usage."""

from __future__ import annotations

import asyncio
import logging

from sqlalchemy import String, cast, func, or_, select
from sqlalchemy.orm import Session, sessionmaker

from paperbase.db.models import Paper as PaperRecord
from paperbase.db.models import Section as PaperSectionRecord
from paperbase.db.session import make_session_factory
from ra.parsing import Section
from ra.retrieval.unified import Paper, normalize_arxiv_id, normalize_doi

logger = logging.getLogger(__name__)


def _paper_source(provider: str) -> str:
    normalized = provider.strip().lower()
    if normalized == "semantic_scholar":
        return "semantic_scholar"
    if normalized == "arxiv":
        return "arxiv"
    return "both"


def _authors_from_metadata(raw_metadata: object) -> list[str]:
    if not isinstance(raw_metadata, dict):
        return []

    authors = raw_metadata.get("authors")
    if isinstance(authors, list):
        return [str(author).strip() for author in authors if str(author).strip()]

    return []


def _pdf_url_from_metadata(raw_metadata: object) -> str | None:
    if not isinstance(raw_metadata, dict):
        return None

    pdf_url = raw_metadata.get("pdf_url")
    if not isinstance(pdf_url, str):
        return None

    clean = pdf_url.strip()
    if clean.startswith(("http://", "https://")):
        return clean
    return None


def _to_ra_paper(record: PaperRecord) -> Paper:
    return Paper(
        id=record.id,
        title=record.canonical_title,
        abstract=record.abstract,
        authors=_authors_from_metadata(record.raw_metadata),
        year=record.publication_year,
        venue=record.venue,
        citation_count=None,
        pdf_url=_pdf_url_from_metadata(record.raw_metadata),
        doi=record.doi,
        arxiv_id=record.arxiv_id,
        source=_paper_source(record.provider),
    )


class PaperbaseGateway:
    """Read-only adapter over the local Paperbase database."""

    def __init__(self, *, session_factory: sessionmaker[Session] | None = None) -> None:
        self.session_factory = session_factory or make_session_factory()

    async def search(self, query: str, limit: int = 10) -> list[Paper]:
        return await asyncio.to_thread(self._search_sync, query, limit)

    def _search_sync(self, query: str, limit: int) -> list[Paper]:
        pattern = f"%{query.lower()}%"
        with self.session_factory() as session:
            records = session.execute(
                select(PaperRecord)
                .where(
                    or_(
                        func.lower(PaperRecord.canonical_title).like(pattern),
                        func.lower(cast(func.coalesce(PaperRecord.abstract, ""), String)).like(pattern),
                    )
                )
                .order_by(PaperRecord.publication_year.desc(), PaperRecord.created_at.desc())
                .limit(limit)
            ).scalars().all()
            return [_to_ra_paper(record) for record in records]

    async def get_paper(self, identifier: str) -> Paper | None:
        return await asyncio.to_thread(self._get_paper_sync, identifier)

    def _get_paper_sync(self, identifier: str) -> Paper | None:
        normalized_identifier = identifier.strip()
        normalized_doi = normalize_doi(normalized_identifier)
        normalized_arxiv = normalize_arxiv_id(normalized_identifier)
        if normalized_identifier.upper().startswith("DOI:"):
            normalized_doi = normalize_doi(normalized_identifier[4:])
        if normalized_identifier.upper().startswith("ARXIV:"):
            normalized_arxiv = normalize_arxiv_id(normalized_identifier[6:])

        with self.session_factory() as session:
            record = session.get(PaperRecord, normalized_identifier)
            if record is None and normalized_doi:
                record = session.execute(
                    select(PaperRecord).where(func.lower(PaperRecord.doi) == normalized_doi)
                ).scalar_one_or_none()
            if record is None and normalized_arxiv:
                record = session.execute(
                    select(PaperRecord).where(func.lower(PaperRecord.arxiv_id) == normalized_arxiv.lower())
                ).scalar_one_or_none()
            if record is None:
                record = session.execute(
                    select(PaperRecord).where(PaperRecord.external_id == normalized_identifier)
                ).scalar_one_or_none()
            return _to_ra_paper(record) if record is not None else None

    async def get_stored_sections(self, identifier: str) -> list[Section]:
        return await asyncio.to_thread(self._get_stored_sections_sync, identifier)

    def _get_stored_sections_sync(self, identifier: str) -> list[Section]:
        paper = self._get_paper_sync(identifier)
        if paper is None:
            return []

        with self.session_factory() as session:
            sections = session.execute(
                select(PaperSectionRecord)
                .where(PaperSectionRecord.paper_id == paper.id)
                .order_by(PaperSectionRecord.ordinal.asc())
            ).scalars().all()
            return [
                Section(
                    title=section.title,
                    content=section.text,
                    page_start=section.page_start or 0,
                )
                for section in sections
            ]

    async def close(self) -> None:
        return None
