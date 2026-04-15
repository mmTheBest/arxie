"""Paper and artifact fetch routes for the Paperbase API service."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Path
from sqlalchemy import select
from sqlalchemy.orm import Session

from paperbase.db.models import Figure, Paper, Section
from ra.utils.security import sanitize_identifier
from services.paperbase_api.dependencies import get_session
from services.paperbase_api.errors import PaperbaseAPIError
from services.paperbase_api.models import (
    FigureResponse,
    FiguresResponse,
    FulltextResponse,
    FulltextResponseData,
    PaperDetailResponse,
    SectionResponse,
    SinglePaperResponse,
)

router = APIRouter(prefix="/api/v1/papers", tags=["papers"])


def _paper_to_response(paper: Paper) -> PaperDetailResponse:
    return PaperDetailResponse(
        id=paper.id,
        title=paper.canonical_title,
        abstract=paper.abstract,
        publication_year=paper.publication_year,
        venue=paper.venue,
        provider=paper.provider,
        external_id=paper.external_id,
        doi=paper.doi,
        arxiv_id=paper.arxiv_id,
    )


def _get_paper_or_404(session: Session, paper_id: str) -> Paper:
    paper = session.get(Paper, paper_id)
    if paper is None:
        raise PaperbaseAPIError(
            status_code=404,
            error="paper_not_found",
            message=f"No paper found for id: {paper_id}",
        )
    return paper


@router.get(
    "/{paper_id}",
    response_model=SinglePaperResponse,
)
def fetch_paper(
    paper_id: str = Path(..., min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> SinglePaperResponse:
    normalized_paper_id = sanitize_identifier(paper_id, field_name="paper_id", max_length=36)
    paper = _get_paper_or_404(session, normalized_paper_id)
    return SinglePaperResponse(data=_paper_to_response(paper))


@router.get(
    "/{paper_id}/fulltext",
    response_model=FulltextResponse,
)
def fetch_paper_fulltext(
    paper_id: str = Path(..., min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> FulltextResponse:
    normalized_paper_id = sanitize_identifier(paper_id, field_name="paper_id", max_length=36)
    paper = _get_paper_or_404(session, normalized_paper_id)
    sections = session.execute(
        select(Section)
        .where(Section.paper_id == normalized_paper_id)
        .order_by(Section.ordinal.asc())
    ).scalars().all()

    return FulltextResponse(
        data=FulltextResponseData(
            paper_id=paper.id,
            title=paper.canonical_title,
            sections=[
                SectionResponse(
                    id=section.id,
                    title=section.title,
                    ordinal=section.ordinal,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    text=section.text,
                )
                for section in sections
            ],
        )
    )


@router.get(
    "/{paper_id}/figures",
    response_model=FiguresResponse,
)
def fetch_paper_figures(
    paper_id: str = Path(..., min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> FiguresResponse:
    normalized_paper_id = sanitize_identifier(paper_id, field_name="paper_id", max_length=36)
    _get_paper_or_404(session, normalized_paper_id)
    figures = session.execute(
        select(Figure)
        .where(Figure.paper_id == normalized_paper_id)
        .order_by(Figure.page_number.asc(), Figure.created_at.asc())
    ).scalars().all()

    return FiguresResponse(
        data=[
            FigureResponse(
                id=figure.id,
                page_number=figure.page_number,
                figure_label=figure.figure_label,
                caption=figure.caption,
                storage_uri=figure.storage_uri,
            )
            for figure in figures
        ]
    )
