"""Search routes for the Paperbase API service."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import String, cast, func, or_, select
from sqlalchemy.orm import Session

from paperbase.db.models import Paper
from ra.utils.security import sanitize_user_text
from services.paperbase_api.dependencies import get_session
from services.paperbase_api.models import PaperSummaryResponse, SearchPapersResponse

router = APIRouter(prefix="/api/v1/search", tags=["search"])


@router.get(
    "/papers",
    response_model=SearchPapersResponse,
)
def search_papers(
    q: str = Query(..., min_length=1, max_length=1000),
    limit: int = Query(10, ge=1, le=100),
    session: Session = Depends(get_session),
) -> SearchPapersResponse:
    query = sanitize_user_text(q, field_name="q", max_length=1000)
    pattern = f"%{query.lower()}%"

    papers = session.execute(
        select(Paper)
        .where(
            or_(
                func.lower(Paper.canonical_title).like(pattern),
                func.lower(cast(func.coalesce(Paper.abstract, ""), String)).like(pattern),
            )
        )
        .order_by(Paper.publication_year.desc(), Paper.created_at.desc())
        .limit(limit)
    ).scalars().all()

    return SearchPapersResponse(
        data=[
            PaperSummaryResponse(
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
            for paper in papers
        ]
    )
