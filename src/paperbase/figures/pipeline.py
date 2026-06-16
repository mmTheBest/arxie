"""Phase-1 figure extraction pipeline."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import fitz
from sqlalchemy import delete
from sqlalchemy.orm import Session, sessionmaker

from paperbase.db.models import Figure, PaperFile
from paperbase.figures.models import FigureCandidate
from paperbase.parsing.files import load_active_pdf_file
from paperbase.storage import StorageResolver

FigureExtractor = Callable[[Path], Sequence[FigureCandidate]]


@dataclass(frozen=True, slots=True)
class FigureExtractionResult:
    paper_id: str
    figure_count: int
_FIGURE_LABEL_PATTERN = re.compile(
    r"^\s*(?:Figure|Fig\.?)\s*(?P<index>\d+[A-Za-z]?)\s*[:.\-]?\s*(?P<caption>.+?)\s*$",
    re.IGNORECASE,
)


def _extract_text_lines(page: fitz.Page) -> list[tuple[str, tuple[float, float, float, float]]]:
    text_lines: list[tuple[str, tuple[float, float, float, float]]] = []
    text_page = page.get_text("dict")

    for block in text_page.get("blocks", []):
        if block.get("type") != 0:
            continue

        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(span.get("text", "") for span in spans).strip()
            if not text:
                continue
            bbox = tuple(line.get("bbox") or block.get("bbox") or (0.0, 0.0, 0.0, 0.0))
            if len(bbox) != 4:
                bbox = (0.0, 0.0, 0.0, 0.0)
            text_lines.append((text, bbox))

    return text_lines


def _default_figure_extractor(pdf_path: Path) -> list[FigureCandidate]:
    candidates: list[FigureCandidate] = []

    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            for text, bbox in _extract_text_lines(page):
                match = _FIGURE_LABEL_PATTERN.match(text)
                if match is None:
                    continue

                label_index = match.group("index")
                caption = match.group("caption").strip()
                candidates.append(
                    FigureCandidate(
                        page_number=page_number,
                        figure_label=f"Figure {label_index}",
                        caption=caption,
                        storage_uri=f"{pdf_path.resolve().as_uri()}#page={page_number}",
                        bbox_json={
                            "x0": bbox[0],
                            "y0": bbox[1],
                            "x1": bbox[2],
                            "y1": bbox[3],
                        },
                    )
                )

    return candidates


class FigureExtractionPipeline:
    """Persist figure metadata produced by a PDF caption extractor."""

    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        extractor: FigureExtractor | None = None,
        storage_resolver: StorageResolver | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.extractor = extractor or _default_figure_extractor
        self.storage_resolver = storage_resolver or StorageResolver()

    def extract_and_store(self, paper_id: str) -> FigureExtractionResult:
        with self.session_factory() as session:
            file_record = self._get_primary_pdf_file(session, paper_id)

        pdf_path = self.storage_resolver.resolve(file_record.storage_uri)
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
                        storage_uri=_artifact_storage_uri(
                            source_storage_uri=file_record.storage_uri,
                            candidate_storage_uri=candidate.storage_uri,
                            page_number=candidate.page_number,
                        ),
                        bbox_json=dict(candidate.bbox_json),
                    )
                )
            session.commit()

        return FigureExtractionResult(paper_id=paper_id, figure_count=len(candidates))

    def _get_primary_pdf_file(self, session: Session, paper_id: str) -> PaperFile:
        file_record = load_active_pdf_file(session, paper_id=paper_id)
        if file_record is None:
            raise ValueError(f"No PDF file registered for paper_id={paper_id}")
        return file_record


def _artifact_storage_uri(
    *,
    source_storage_uri: str,
    candidate_storage_uri: str | None,
    page_number: int | None,
) -> str | None:
    source_parsed = urlparse(source_storage_uri)
    if source_parsed.scheme in {"http", "https"}:
        suffix = f"#page={page_number}" if page_number is not None else ""
        return f"{source_storage_uri.split('#', 1)[0]}{suffix}"
    return candidate_storage_uri
