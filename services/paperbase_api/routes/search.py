"""Search routes for the Paperbase API service."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request, status
from sqlalchemy import String, cast, exists, func, not_, or_, select
from sqlalchemy.orm import Session

from paperbase.db.models import (
    Author,
    CollectionPaper,
    Dataset,
    ExtractionRun,
    Method,
    Metric,
    Paper,
    PaperAuthor,
    PaperTag,
    Tag,
)
from paperbase.db.repositories import BackgroundJobRepository, PaperRepository
from paperbase.search.query_builder import build_search_query
from ra.utils.security import sanitize_identifier, sanitize_user_text
from services.paperbase_api.dependencies import get_session
from services.paperbase_api.errors import PaperbaseAPIError
from services.paperbase_api.models import (
    PaperSummaryResponse,
    SearchPapersResponse,
    SearchStatusResponse,
    SearchStatusResponseData,
    SingleBackgroundJobResponse,
)
from services.paperbase_api.routes.jobs import background_job_to_response

router = APIRouter(prefix="/api/v1/search", tags=["search"])


@router.get(
    "/status",
    response_model=SearchStatusResponse,
)
def search_status(request: Request) -> SearchStatusResponse:
    search_backend = request.app.state.search_backend
    return SearchStatusResponse(
        data=SearchStatusResponseData(
            backend_configured=search_backend is not None,
            backend_type=search_backend.__class__.__name__ if search_backend is not None else None,
        )
    )


@router.post(
    "/reindex",
    response_model=SingleBackgroundJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def search_reindex(request: Request) -> SingleBackgroundJobResponse:
    with request.app.state.session_factory() as session:
        repository = BackgroundJobRepository(session)
        job = repository.create(job_type="search_reindex", payload_json={})
    return SingleBackgroundJobResponse(data=background_job_to_response(job))


@router.get(
    "/papers",
    response_model=SearchPapersResponse,
)
def search_papers(
    request: Request,
    q: str | None = Query(None, min_length=1, max_length=1000),
    limit: int = Query(10, ge=1, le=100),
    collection_id: str | None = Query(None, min_length=1, max_length=36),
    year_gte: int | None = Query(None, ge=0, le=9999),
    year_lte: int | None = Query(None, ge=0, le=9999),
    venue: str | None = Query(None, min_length=1, max_length=255),
    author: str | None = Query(None, min_length=1, max_length=255),
    tag: str | None = Query(None, min_length=1, max_length=255),
    dataset: str | None = Query(None, min_length=1, max_length=255),
    method: str | None = Query(None, min_length=1, max_length=255),
    metric: str | None = Query(None, min_length=1, max_length=255),
    extraction_state: str | None = Query(None, min_length=1, max_length=32),
    session: Session = Depends(get_session),
) -> SearchPapersResponse:
    statement = select(Paper).distinct()
    predicates: list[object] = []
    has_filter = False
    search_filters: dict[str, object] = {}
    search_backend = request.app.state.search_backend

    safe_query: str | None = None
    if q is not None:
        safe_query = sanitize_user_text(q, field_name="q", max_length=1000)
        pattern = f"%{safe_query.lower()}%"
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
        search_filters["collection_id"] = safe_collection_id

    if year_gte is not None:
        predicates.append(Paper.publication_year >= year_gte)
        has_filter = True
        search_filters["year_gte"] = year_gte

    if year_lte is not None:
        predicates.append(Paper.publication_year <= year_lte)
        has_filter = True
        search_filters["year_lte"] = year_lte

    if venue is not None:
        safe_venue = sanitize_user_text(venue, field_name="venue", max_length=255)
        predicates.append(func.lower(cast(func.coalesce(Paper.venue, ""), String)) == safe_venue.lower())
        has_filter = True
        search_filters["venue"] = [safe_venue]

    if author is not None:
        safe_author = sanitize_user_text(author, field_name="author", max_length=255)
        predicates.append(
            exists(
                select(PaperAuthor.id)
                .join(Author, Author.id == PaperAuthor.author_id)
                .where(
                    PaperAuthor.paper_id == Paper.id,
                    func.lower(Author.display_name) == safe_author.lower(),
                )
            )
        )
        has_filter = True
        search_filters["authors"] = [safe_author]

    if tag is not None:
        safe_tag = sanitize_user_text(tag, field_name="tag", max_length=255)
        predicates.append(
            exists(
                select(PaperTag.id)
                .join(Tag, Tag.id == PaperTag.tag_id)
                .where(
                    PaperTag.paper_id == Paper.id,
                    func.lower(Tag.display_name) == safe_tag.lower(),
                )
            )
        )
        has_filter = True
        search_filters["tags"] = [safe_tag]

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
        search_filters["datasets"] = [safe_dataset]

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
        search_filters["methods"] = [safe_method]

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
        search_filters["metrics"] = [safe_metric]

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
        search_filters["extraction_state"] = safe_extraction_state

    if not has_filter:
        raise PaperbaseAPIError(
            status_code=400,
            error="invalid_input",
            message="Provide q or at least one supported filter.",
        )

    if search_backend is not None and collection_id is None:
        backend_filters = {
            key: value for key, value in search_filters.items() if key != "collection_id"
        }
        documents = search_backend.search(
            "paperbase-papers",
            build_search_query(query_text=safe_query, filters=backend_filters),
            limit,
        )
        return SearchPapersResponse(
            data=[
                PaperSummaryResponse(
                    id=str(document.get("paper_id", "")),
                    title=str(document.get("title", "")),
                    abstract=document.get("abstract"),
                    publication_year=document.get("publication_year"),
                    venue=document.get("venue"),
                    provider=str(document.get("provider", "")),
                    external_id=str(document.get("external_id", "")),
                    doi=document.get("doi"),
                    arxiv_id=document.get("arxiv_id"),
                    authors=list(document.get("authors", [])),
                    tags=list(document.get("tags", [])),
                )
                for document in documents
            ]
        )

    papers = session.execute(
        statement
        .where(*predicates)
        .order_by(Paper.publication_year.desc(), Paper.created_at.desc())
        .limit(limit)
    ).scalars().all()
    repository = PaperRepository(session)
    paper_ids = [paper.id for paper in papers]
    authors_by_paper_id = repository.list_author_names_by_paper_ids(paper_ids)
    tags_by_paper_id = repository.list_tags_by_paper_ids(paper_ids)

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
                authors=authors_by_paper_id.get(paper.id, []),
                tags=tags_by_paper_id.get(paper.id, []),
            )
            for paper in papers
        ]
    )
