"""Paper and artifact fetch routes for the Paperbase API service."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Path
from sqlalchemy import select
from sqlalchemy.orm import Session

from paperbase.db.models import (
    Dataset,
    EngineeringTrick,
    EvidenceSpan,
    ExtractionRun,
    Figure,
    Finding,
    GlossaryTerm,
    Method,
    Metric,
    Paper,
    ResultRow,
    Section,
)
from ra.utils.security import sanitize_identifier
from services.paperbase_api.dependencies import get_session
from services.paperbase_api.errors import PaperbaseAPIError
from services.paperbase_api.models import (
    EngineeringTrickArtifactResponse,
    EvidenceSpanArtifactResponse,
    ExtractionRunArtifactResponse,
    FindingArtifactResponse,
    FigureResponse,
    FiguresResponse,
    FulltextResponse,
    FulltextResponseData,
    GlossaryTermArtifactResponse,
    PaperDetailResponse,
    PaperStructuredDataResponse,
    PaperStructuredDataResponseData,
    ResultRowArtifactResponse,
    SectionResponse,
    SinglePaperResponse,
    StructuredNamedArtifactResponse,
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


@router.get(
    "/{paper_id}/structured-data",
    response_model=PaperStructuredDataResponse,
)
def fetch_paper_structured_data(
    paper_id: str = Path(..., min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> PaperStructuredDataResponse:
    normalized_paper_id = sanitize_identifier(paper_id, field_name="paper_id", max_length=36)
    _get_paper_or_404(session, normalized_paper_id)

    datasets = session.execute(
        select(Dataset)
        .where(Dataset.paper_id == normalized_paper_id)
        .order_by(Dataset.display_name.asc(), Dataset.created_at.asc())
    ).scalars().all()
    methods = session.execute(
        select(Method)
        .where(Method.paper_id == normalized_paper_id)
        .order_by(Method.display_name.asc(), Method.created_at.asc())
    ).scalars().all()
    metrics = session.execute(
        select(Metric)
        .where(Metric.paper_id == normalized_paper_id)
        .order_by(Metric.display_name.asc(), Metric.created_at.asc())
    ).scalars().all()
    glossary_terms = session.execute(
        select(GlossaryTerm)
        .where(GlossaryTerm.paper_id == normalized_paper_id)
        .order_by(GlossaryTerm.term.asc(), GlossaryTerm.created_at.asc())
    ).scalars().all()
    findings = session.execute(
        select(Finding)
        .where(Finding.paper_id == normalized_paper_id)
        .order_by(Finding.created_at.asc())
    ).scalars().all()
    engineering_tricks = session.execute(
        select(EngineeringTrick)
        .where(EngineeringTrick.paper_id == normalized_paper_id)
        .order_by(EngineeringTrick.title.asc(), EngineeringTrick.created_at.asc())
    ).scalars().all()
    extraction_runs = session.execute(
        select(ExtractionRun)
        .where(ExtractionRun.paper_id == normalized_paper_id)
        .order_by(ExtractionRun.created_at.desc())
    ).scalars().all()
    evidence_spans = session.execute(
        select(EvidenceSpan)
        .where(EvidenceSpan.paper_id == normalized_paper_id)
        .order_by(EvidenceSpan.created_at.asc())
    ).scalars().all()
    result_rows = session.execute(
        select(ResultRow, Dataset, Method, Metric)
        .outerjoin(Dataset, Dataset.id == ResultRow.dataset_id)
        .outerjoin(Method, Method.id == ResultRow.method_id)
        .outerjoin(Metric, Metric.id == ResultRow.metric_id)
        .where(ResultRow.paper_id == normalized_paper_id)
        .order_by(ResultRow.created_at.asc())
    ).all()

    return PaperStructuredDataResponse(
        data=PaperStructuredDataResponseData(
            paper_id=normalized_paper_id,
            datasets=[
                StructuredNamedArtifactResponse(
                    id=dataset.id,
                    display_name=dataset.display_name,
                    metadata=dict(dataset.metadata_json or {}),
                )
                for dataset in datasets
            ],
            methods=[
                StructuredNamedArtifactResponse(
                    id=method.id,
                    display_name=method.display_name,
                    metadata=dict(method.metadata_json or {}),
                )
                for method in methods
            ],
            metrics=[
                StructuredNamedArtifactResponse(
                    id=metric.id,
                    display_name=metric.display_name,
                    metadata=dict(metric.metadata_json or {}),
                )
                for metric in metrics
            ],
            result_rows=[
                ResultRowArtifactResponse(
                    id=result_row.id,
                    dataset_id=dataset.id if dataset is not None else result_row.dataset_id,
                    dataset=dataset.display_name if dataset is not None else None,
                    method_id=method.id if method is not None else result_row.method_id,
                    method=method.display_name if method is not None else None,
                    metric_id=metric.id if metric is not None else result_row.metric_id,
                    metric=metric.display_name if metric is not None else None,
                    value_numeric=result_row.value_numeric,
                    value_text=result_row.value_text,
                    comparator_text=result_row.comparator_text,
                    notes=result_row.notes,
                )
                for result_row, dataset, method, metric in result_rows
            ],
            glossary_terms=[
                GlossaryTermArtifactResponse(
                    id=term.id,
                    term=term.term,
                    definition=term.definition,
                    metadata=dict(term.metadata_json or {}),
                )
                for term in glossary_terms
            ],
            findings=[
                FindingArtifactResponse(
                    id=finding.id,
                    statement=finding.statement,
                    polarity=finding.polarity,
                    metadata=dict(finding.metadata_json or {}),
                )
                for finding in findings
            ],
            engineering_tricks=[
                EngineeringTrickArtifactResponse(
                    id=trick.id,
                    title=trick.title,
                    description=trick.description,
                    metadata=dict(trick.metadata_json or {}),
                )
                for trick in engineering_tricks
            ],
            extraction_runs=[
                ExtractionRunArtifactResponse(
                    id=run.id,
                    model_name=run.model_name,
                    prompt_version=run.prompt_version,
                    schema_version=run.schema_version,
                    status=run.status,
                )
                for run in extraction_runs
            ],
            evidence_spans=[
                EvidenceSpanArtifactResponse(
                    id=span.id,
                    extraction_run_id=span.extraction_run_id,
                    target_type=span.target_type,
                    target_id=span.target_id,
                    page_number=span.page_number,
                    quote_text=span.quote_text,
                    section_id=span.section_id,
                    chunk_id=span.chunk_id,
                )
                for span in evidence_spans
            ],
        )
    )
