"""Search routes for the Paperbase API service."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import String, cast, exists, func, not_, or_, select
from sqlalchemy.orm import Session

from paperbase.db.models import CollectionPaper, Dataset, ExtractionRun, Method, Metric, Paper
from ra.utils.security import sanitize_identifier, sanitize_user_text
from services.paperbase_api.dependencies import get_session
from services.paperbase_api.errors import PaperbaseAPIError
from services.paperbase_api.models import PaperSummaryResponse, SearchPapersResponse

router = APIRouter(prefix="/api/v1/search", tags=["search"])


@router.get(
    "/papers",
    response_model=SearchPapersResponse,
)
def search_papers(
    q: str | None = Query(None, min_length=1, max_length=1000),
    limit: int = Query(10, ge=1, le=100),
    collection_id: str | None = Query(None, min_length=1, max_length=36),
    year_gte: int | None = Query(None, ge=0, le=9999),
    year_lte: int | None = Query(None, ge=0, le=9999),
    venue: str | None = Query(None, min_length=1, max_length=255),
    dataset: str | None = Query(None, min_length=1, max_length=255),
    method: str | None = Query(None, min_length=1, max_length=255),
    metric: str | None = Query(None, min_length=1, max_length=255),
    extraction_state: str | None = Query(None, min_length=1, max_length=32),
    session: Session = Depends(get_session),
) -> SearchPapersResponse:
    statement = select(Paper).distinct()
    predicates: list[object] = []
    has_filter = False

    if q is not None:
        query = sanitize_user_text(q, field_name="q", max_length=1000)
        pattern = f"%{query.lower()}%"
        predicates.append(
            or_(
                func.lower(Paper.canonical_title).like(pattern),
                func.lower(cast(func.coalesce(Paper.abstract, ""), String)).like(pattern),
            )
        )
        has_filter = True

    if collection_id is not None:
        safe_collection_id = sanitize_identifier(
            collection_id,
            field_name="collection_id",
            max_length=36,
        )
        predicates.append(
            exists(
                select(CollectionPaper.id).where(
                    CollectionPaper.collection_id == safe_collection_id,
                    CollectionPaper.paper_id == Paper.id,
                )
            )
        )
        has_filter = True

    if year_gte is not None:
        predicates.append(Paper.publication_year >= year_gte)
        has_filter = True

    if year_lte is not None:
        predicates.append(Paper.publication_year <= year_lte)
        has_filter = True

    if venue is not None:
        safe_venue = sanitize_user_text(venue, field_name="venue", max_length=255)
        predicates.append(func.lower(cast(func.coalesce(Paper.venue, ""), String)) == safe_venue.lower())
        has_filter = True

    if dataset is not None:
        safe_dataset = sanitize_user_text(dataset, field_name="dataset", max_length=255)
        predicates.append(
            exists(
                select(Dataset.id).where(
                    Dataset.paper_id == Paper.id,
                    func.lower(Dataset.display_name) == safe_dataset.lower(),
                )
            )
        )
        has_filter = True

    if method is not None:
        safe_method = sanitize_user_text(method, field_name="method", max_length=255)
        predicates.append(
            exists(
                select(Method.id).where(
                    Method.paper_id == Paper.id,
                    func.lower(Method.display_name) == safe_method.lower(),
                )
            )
        )
        has_filter = True

    if metric is not None:
        safe_metric = sanitize_user_text(metric, field_name="metric", max_length=255)
        predicates.append(
            exists(
                select(Metric.id).where(
                    Metric.paper_id == Paper.id,
                    func.lower(Metric.display_name) == safe_metric.lower(),
                )
            )
        )
        has_filter = True

    if extraction_state is not None:
        safe_extraction_state = sanitize_user_text(
            extraction_state,
            field_name="extraction_state",
            max_length=32,
        ).lower()
        extracted_exists = exists(
            select(ExtractionRun.id).where(
                ExtractionRun.paper_id == Paper.id,
                ExtractionRun.status == "completed",
            )
        )
        if safe_extraction_state == "extracted":
            predicates.append(extracted_exists)
        elif safe_extraction_state == "unextracted":
            predicates.append(not_(extracted_exists))
        else:
            raise PaperbaseAPIError(
                status_code=400,
                error="invalid_input",
                message="extraction_state must be 'extracted' or 'unextracted'.",
            )
        has_filter = True

    if not has_filter:
        raise PaperbaseAPIError(
            status_code=400,
            error="invalid_input",
            message="Provide q or at least one supported filter.",
        )

    papers = session.execute(
        statement
        .where(*predicates)
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
