"""Collection-level parse orchestration for Paperbase."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy.orm import Session, sessionmaker

from paperbase.db.repositories import CollectionRepository
from paperbase.parsing.pipeline import PaperParsePipeline


@dataclass(frozen=True, slots=True)
class CollectionParseSummary:
    collection_id: str
    parsed_paper_count: int
    skipped_paper_ids: list[str]
    section_count: int
    chunk_count: int


class CollectionParseRunner:
    """Run the stored PDF parse pipeline across a curated collection."""

    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        parser_factory: Callable[[], object] | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.parser_factory = parser_factory

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

        for membership in memberships[:limit]:
            try:
                parser = self.parser_factory() if self.parser_factory is not None else None
                result = PaperParsePipeline(
                    session_factory=self.session_factory,
                    parser=parser,
                ).parse_paper(membership.paper_id)
            except Exception:  # noqa: BLE001
                skipped_paper_ids.append(membership.paper_id)
                continue

            parsed_paper_count += 1
            section_count += result.section_count
            chunk_count += result.chunk_count

        return CollectionParseSummary(
            collection_id=collection_id,
            parsed_paper_count=parsed_paper_count,
            skipped_paper_ids=skipped_paper_ids,
            section_count=section_count,
            chunk_count=chunk_count,
        )
