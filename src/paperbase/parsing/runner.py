"""Collection-level parse orchestration for Paperbase."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session, sessionmaker

from paperbase.db.repositories import CollectionRepository
from paperbase.figures.pipeline import FigureExtractionPipeline
from paperbase.parsing.pipeline import PaperParsePipeline
from paperbase.storage import StorageResolver
from paperbase.tables.pipeline import TableExtractionPipeline


@dataclass(frozen=True, slots=True)
class CollectionParseSummary:
    collection_id: str
    parsed_paper_count: int
    skipped_paper_ids: list[str]
    section_count: int
    chunk_count: int
    figure_count: int
    table_count: int


class CollectionParseRunner:
    """Run the stored PDF parse pipeline across a curated collection."""

    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        parser_factory: Callable[[], object] | None = None,
        object_store: object | None = None,
        cache_dir: Path | None = None,
        cache_ttl_seconds: int = 86400,
    ) -> None:
        self.session_factory = session_factory
        self.parser_factory = parser_factory
        self.object_store = object_store
        self.cache_dir = cache_dir
        self.cache_ttl_seconds = cache_ttl_seconds

    def parse_collection(
        self,
        *,
        collection_id: str,
        limit: int | None = None,
    ) -> CollectionParseSummary:
        with self.session_factory() as session:
            memberships = CollectionRepository(session).list_papers(collection_id)

        parsed_paper_count = 0
        skipped_paper_ids: list[str] = []
        section_count = 0
        chunk_count = 0
        figure_count = 0
        table_count = 0

        for membership in memberships[:limit]:
            storage_resolver = StorageResolver(
                object_store=self.object_store,
                cache_dir=None if self.cache_dir is None else self.cache_dir,
                cache_ttl_seconds=self.cache_ttl_seconds,
            )
            try:
                parser = self.parser_factory() if self.parser_factory is not None else None
                result = PaperParsePipeline(
                    session_factory=self.session_factory,
                    parser=parser,
                    storage_resolver=storage_resolver,
                ).parse_paper(membership.paper_id)
            except Exception:  # noqa: BLE001
                skipped_paper_ids.append(membership.paper_id)
                continue

            parsed_paper_count += 1
            section_count += result.section_count
            chunk_count += result.chunk_count
            try:
                figure_count += FigureExtractionPipeline(
                    session_factory=self.session_factory,
                    storage_resolver=storage_resolver,
                ).extract_and_store(membership.paper_id).figure_count
                table_count += TableExtractionPipeline(
                    session_factory=self.session_factory,
                    storage_resolver=storage_resolver,
                ).extract_and_store(membership.paper_id).table_count
            except Exception:  # noqa: BLE001
                continue

        return CollectionParseSummary(
            collection_id=collection_id,
            parsed_paper_count=parsed_paper_count,
            skipped_paper_ids=skipped_paper_ids,
            section_count=section_count,
            chunk_count=chunk_count,
            figure_count=figure_count,
            table_count=table_count,
        )
