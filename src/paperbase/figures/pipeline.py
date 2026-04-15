"""Placeholder figure extraction pipeline."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from sqlalchemy import delete
from sqlalchemy.orm import Session, sessionmaker

from paperbase.db.models import Figure, PaperFile
from paperbase.db.repositories import PaperFileRepository
from paperbase.figures.models import FigureCandidate

FigureExtractor = Callable[[Path], Sequence[FigureCandidate]]


@dataclass(frozen=True, slots=True)
class FigureExtractionResult:
    paper_id: str
    figure_count: int


def _path_from_storage_uri(storage_uri: str) -> Path:
    parsed = urlparse(storage_uri)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    return Path(storage_uri)


def _placeholder_figure_extractor(pdf_path: Path) -> list[FigureCandidate]:  # noqa: ARG001
    return []


class FigureExtractionPipeline:
    """Persist figure metadata produced by a placeholder extraction adapter."""

    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        extractor: FigureExtractor | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.extractor = extractor or _placeholder_figure_extractor

    def extract_and_store(self, paper_id: str) -> FigureExtractionResult:
        with self.session_factory() as session:
            file_record = self._get_primary_pdf_file(session, paper_id)

        pdf_path = _path_from_storage_uri(file_record.storage_uri)
        candidates = list(self.extractor(pdf_path))

        with self.session_factory() as session:
            session.execute(delete(Figure).where(Figure.paper_id == paper_id))
            for candidate in candidates:
                session.add(
                    Figure(
                        paper_id=paper_id,
                        page_number=candidate.page_number,
                        figure_label=candidate.figure_label,
                        caption=candidate.caption,
                        storage_uri=candidate.storage_uri,
                        bbox_json=dict(candidate.bbox_json),
                    )
                )
            session.commit()

        return FigureExtractionResult(paper_id=paper_id, figure_count=len(candidates))

    def _get_primary_pdf_file(self, session: Session, paper_id: str) -> PaperFile:
        file_records = PaperFileRepository(session).list_for_paper(paper_id=paper_id, file_kind="pdf")
        if not file_records:
            raise ValueError(f"No PDF file registered for paper_id={paper_id}")
        return file_records[0]
