"""Paperbase-backed retrieval adapter for Arxie runtime usage."""

from __future__ import annotations

import asyncio
import logging

from sqlalchemy import String, cast, func, or_, select
from sqlalchemy.orm import Session, sessionmaker

from paperbase.db.models import CollectionPaper
from paperbase.db.models import Paper as PaperRecord
from paperbase.db.models import Section as PaperSectionRecord
from paperbase.db.models import Workspace as WorkspaceRecord
from paperbase.db.session import make_session_factory
from ra.parsing import Section
from ra.retrieval.unified import Paper, WorkspaceContext, normalize_arxiv_id, normalize_doi

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


def _abstract_from_sections(session: Session, paper_id: str) -> str | None:
    section = session.execute(
        select(PaperSectionRecord)
        .where(
            PaperSectionRecord.paper_id == paper_id,
            func.lower(PaperSectionRecord.title).like("%abstract%"),
        )
        .order_by(PaperSectionRecord.ordinal.asc())
        .limit(1)
    ).scalar_one_or_none()
    if section is None:
        return None
    text = (section.text or "").strip()
    return text or None


def _to_ra_paper(record: PaperRecord, *, abstract_override: str | None = None) -> Paper:
    return Paper(
        id=record.id,
        title=record.canonical_title,
        abstract=abstract_override or record.abstract,
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

    async def get_papers_by_ids(self, paper_ids: list[str]) -> list[Paper]:
        return await asyncio.to_thread(self._get_papers_by_ids_sync, paper_ids)

    def _get_papers_by_ids_sync(self, paper_ids: list[str]) -> list[Paper]:
        if not paper_ids:
            return []

        with self.session_factory() as session:
            papers: list[Paper] = []
            for paper_id in paper_ids:
                record = session.get(PaperRecord, paper_id)
                if record is None:
                    continue
                abstract_override = None
                if not record.abstract:
                    abstract_override = _abstract_from_sections(session, record.id)
                papers.append(_to_ra_paper(record, abstract_override=abstract_override))
            return papers

    async def get_collection_papers(
        self,
        collection_id: str,
        *,
        query: str | None = None,
        limit: int = 50,
    ) -> list[Paper]:
        return await asyncio.to_thread(
            self._get_collection_papers_sync,
            collection_id,
            query,
            limit,
        )

    def _get_collection_papers_sync(
        self,
        collection_id: str,
        query: str | None = None,
        limit: int = 50,
    ) -> list[Paper]:
        query_tokens = {
            token
            for token in str(query or "").lower().split()
            if token
        }
        with self.session_factory() as session:
            records = session.execute(
                select(PaperRecord, CollectionPaper)
                .join(CollectionPaper, CollectionPaper.paper_id == PaperRecord.id)
                .where(CollectionPaper.collection_id == collection_id)
                .order_by(CollectionPaper.position.asc(), PaperRecord.created_at.asc())
            ).all()

            ranked: list[tuple[int, int, Paper]] = []
            for position, (record, membership) in enumerate(records):
                abstract_override = None
                if not record.abstract:
                    abstract_override = _abstract_from_sections(session, record.id)
                paper = _to_ra_paper(record, abstract_override=abstract_override)
                haystack = f"{paper.title} {paper.abstract or ''}".lower()
                score = sum(1 for token in query_tokens if token in haystack)
                membership_position = membership.position if membership.position is not None else position
                ranked.append((score, membership_position, paper))

            ranked.sort(key=lambda item: (-item[0], item[1], item[2].title.lower()))
            return [paper for _, _, paper in ranked[:limit]]

    async def get_workspace_context(self, workspace_id: str) -> WorkspaceContext | None:
        return await asyncio.to_thread(self._get_workspace_context_sync, workspace_id)

    def _get_workspace_context_sync(self, workspace_id: str) -> WorkspaceContext | None:
        with self.session_factory() as session:
            workspace = session.get(WorkspaceRecord, workspace_id)
            if workspace is None:
                return None
            return WorkspaceContext(
                workspace_id=workspace.id,
                title=workspace.title,
                collection_id=workspace.collection_id,
                saved_query=workspace.saved_query,
                focus_note=workspace.focus_note,
                active_filters=dict(workspace.active_filters_json or {}),
                pinned_paper_ids=list(workspace.pinned_paper_ids_json or []),
            )

    async def close(self) -> None:
        return None
