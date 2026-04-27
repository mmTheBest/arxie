"""Search routes for the Paperbase API service."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request, status
from sqlalchemy import String, cast, exists, func, not_, or_, select
from sqlalchemy.orm import Session

from paperbase.db.models import (
    Author,
    Chunk,
    CollectionPaper,
    Dataset,
    ExtractionRun,
    Figure,
    Method,
    Metric,
    Paper,
    PaperAuthor,
    PaperTag,
    Section,
    TableArtifact,
    Tag,
)
from paperbase.db.repositories import PaperRepository
from paperbase.search.query_builder import build_search_query
from ra.utils.security import sanitize_identifier, sanitize_user_text
from services.paperbase_api.background_jobs import create_background_job
from services.paperbase_api.dependencies import get_session
from services.paperbase_api.errors import PaperbaseAPIError
from services.paperbase_api.models import (
    PaperSummaryResponse,
    SearchArtifactHitResponse,
    SearchArtifactsResponse,
    SearchChunkHitResponse,
    SearchChunksResponse,
    SearchPapersResponse,
    SearchStatusResponse,
    SearchStatusResponseData,
    SingleBackgroundJobResponse,
)
from services.paperbase_api.routes.jobs import background_job_to_response

router = APIRouter(prefix="/api/v1/search", tags=["search"])


def _search_backend_query(
    query_text: str | None,
    *,
    filters: dict[str, object] | None = None,
    embedding_provider: object | None = None,
) -> dict[str, object]:
    embedding_vector = None
    if query_text and embedding_provider is not None:
        embedding_vector = embedding_provider.embed(query_text)
    return build_search_query(
        query_text=query_text,
        filters=filters,
        embedding_vector=embedding_vector,
    )


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
    job = create_background_job(
        session_factory=request.app.state.session_factory,
        job_type="search_reindex",
        payload_json={},
        dispatcher=request.app.state.job_dispatcher,
    )
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
        search_filters["collection_ids"] = [safe_collection_id]

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

    if search_backend is not None:
        documents = search_backend.search(
            "paperbase-papers",
            _search_backend_query(
                safe_query,
                filters=search_filters,
                embedding_provider=request.app.state.embedding_provider,
            ),
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


@router.get(
    "/chunks",
    response_model=SearchChunksResponse,
)
def search_chunks(
    request: Request,
    q: str = Query(..., min_length=1, max_length=1000),
    limit: int = Query(10, ge=1, le=100),
    collection_id: str | None = Query(None, min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> SearchChunksResponse:
    safe_query = sanitize_user_text(q, field_name="q", max_length=1000)
    safe_collection_id = (
        sanitize_identifier(collection_id, field_name="collection_id", max_length=36)
        if collection_id is not None
        else None
    )
    search_backend = request.app.state.search_backend

    if search_backend is not None:
        documents = search_backend.search(
            "paperbase-chunks",
            _search_backend_query(
                safe_query,
                filters={"collection_ids": [safe_collection_id]} if safe_collection_id is not None else None,
                embedding_provider=request.app.state.embedding_provider,
            ),
            limit,
        )
        return SearchChunksResponse(
            data=[
                SearchChunkHitResponse(
                    chunk_id=str(document.get("chunk_id", "")),
                    paper_id=str(document.get("paper_id", "")),
                    paper_title=str(document.get("title", "")),
                    section_title=(str(document.get("section_title")) if document.get("section_title") else None),
                    text=str(document.get("text", "")),
                )
                for document in documents
            ]
        )

    pattern = f"%{safe_query.lower()}%"
    statement = (
        select(Chunk, Paper, Section)
        .join(Paper, Paper.id == Chunk.paper_id)
        .outerjoin(Section, Section.id == Chunk.section_id)
        .where(
            or_(
                func.lower(cast(func.coalesce(Chunk.text, ""), String)).like(pattern),
                func.lower(cast(func.coalesce(Paper.canonical_title, ""), String)).like(pattern),
                func.lower(cast(func.coalesce(Section.title, ""), String)).like(pattern),
            )
        )
        .order_by(Chunk.created_at.asc())
    )
    if safe_collection_id is not None:
        statement = statement.join(
            CollectionPaper,
            CollectionPaper.paper_id == Chunk.paper_id,
        ).where(CollectionPaper.collection_id == safe_collection_id)
    statement = statement.limit(limit)

    rows = session.execute(statement).all()
    return SearchChunksResponse(
        data=[
            SearchChunkHitResponse(
                chunk_id=chunk.id,
                paper_id=paper.id,
                paper_title=paper.canonical_title,
                section_title=section.title if section is not None else None,
                text=chunk.text,
            )
            for chunk, paper, section in rows
        ]
    )


@router.get(
    "/artifacts",
    response_model=SearchArtifactsResponse,
)
def search_artifacts(
    request: Request,
    q: str = Query(..., min_length=1, max_length=1000),
    kind: str = Query("all", pattern="^(all|figure|table)$"),
    limit: int = Query(10, ge=1, le=100),
    collection_id: str | None = Query(None, min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> SearchArtifactsResponse:
    safe_query = sanitize_user_text(q, field_name="q", max_length=1000)
    safe_kind = sanitize_user_text(kind, field_name="kind", max_length=16).lower()
    safe_collection_id = (
        sanitize_identifier(collection_id, field_name="collection_id", max_length=36)
        if collection_id is not None
        else None
    )
    search_backend = request.app.state.search_backend

    if search_backend is not None:
        documents: list[SearchArtifactHitResponse] = []
        query = _search_backend_query(
            safe_query,
            filters={"collection_ids": [safe_collection_id]} if safe_collection_id is not None else None,
            embedding_provider=request.app.state.embedding_provider,
        )
        if safe_kind in {"all", "figure"}:
            for document in search_backend.search("paperbase-figures", query, limit):
                documents.append(
                    SearchArtifactHitResponse(
                        artifact_type="figure",
                        artifact_id=str(document.get("figure_id", "")),
                        paper_id=str(document.get("paper_id", "")),
                        paper_title=str(document.get("title", "")),
                        label=(str(document.get("figure_label")) if document.get("figure_label") else None),
                        caption=(str(document.get("caption")) if document.get("caption") else None),
                    )
                )
        if safe_kind in {"all", "table"}:
            for document in search_backend.search("paperbase-tables", query, limit):
                documents.append(
                    SearchArtifactHitResponse(
                        artifact_type="table",
                        artifact_id=str(document.get("table_id", "")),
                        paper_id=str(document.get("paper_id", "")),
                        paper_title=str(document.get("title", "")),
                        label=(str(document.get("table_label")) if document.get("table_label") else None),
                        caption=(str(document.get("caption")) if document.get("caption") else None),
                        structured_payload=dict(document.get("structured_payload") or {}),
                    )
                )
        return SearchArtifactsResponse(data=documents[:limit])

    pattern = f"%{safe_query.lower()}%"
    items: list[SearchArtifactHitResponse] = []

    if safe_kind in {"all", "figure"}:
        figure_statement = (
            select(Figure, Paper)
            .join(Paper, Paper.id == Figure.paper_id)
            .where(
                or_(
                    func.lower(cast(func.coalesce(Figure.caption, ""), String)).like(pattern),
                    func.lower(cast(func.coalesce(Figure.figure_label, ""), String)).like(pattern),
                    func.lower(cast(func.coalesce(Paper.canonical_title, ""), String)).like(pattern),
                )
            )
            .order_by(Figure.created_at.asc())
        )
        if safe_collection_id is not None:
            figure_statement = figure_statement.join(
                CollectionPaper,
                CollectionPaper.paper_id == Figure.paper_id,
            ).where(CollectionPaper.collection_id == safe_collection_id)
        figure_statement = figure_statement.limit(limit)
        items.extend(
            SearchArtifactHitResponse(
                artifact_type="figure",
                artifact_id=figure.id,
                paper_id=paper.id,
                paper_title=paper.canonical_title,
                page_number=figure.page_number,
                label=figure.figure_label,
                caption=figure.caption,
            )
            for figure, paper in session.execute(figure_statement).all()
        )

    if safe_kind in {"all", "table"}:
        table_statement = (
            select(TableArtifact, Paper)
            .join(Paper, Paper.id == TableArtifact.paper_id)
            .where(
                or_(
                    func.lower(cast(func.coalesce(TableArtifact.caption, ""), String)).like(pattern),
                    func.lower(cast(func.coalesce(TableArtifact.table_label, ""), String)).like(pattern),
                    func.lower(cast(func.coalesce(Paper.canonical_title, ""), String)).like(pattern),
                )
            )
            .order_by(TableArtifact.created_at.asc())
        )
        if safe_collection_id is not None:
            table_statement = table_statement.join(
                CollectionPaper,
                CollectionPaper.paper_id == TableArtifact.paper_id,
            ).where(CollectionPaper.collection_id == safe_collection_id)
        table_statement = table_statement.limit(limit)
        items.extend(
            SearchArtifactHitResponse(
                artifact_type="table",
                artifact_id=table.id,
                paper_id=paper.id,
                paper_title=paper.canonical_title,
                page_number=table.page_number,
                label=table.table_label,
                caption=table.caption,
                structured_payload=dict(table.structured_payload_json or {}),
            )
            for table, paper in session.execute(table_statement).all()
        )

    return SearchArtifactsResponse(data=items[:limit])
