"""Comparison routes for the Paperbase API service."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from paperbase.db.models import CollectionPaper, Dataset, Method, Metric, Paper, ResultRow
from ra.utils.security import sanitize_identifier, sanitize_user_text
from services.paperbase_api.dependencies import get_session
from services.paperbase_api.models import (
    CompareResultItemResponse,
    CompareResultsRequest,
    CompareResultsResponse,
)

router = APIRouter(prefix="/api/v1/compare", tags=["compare"])


@router.post(
    "/results",
    response_model=CompareResultsResponse,
)
def compare_results(
    payload: CompareResultsRequest,
    session: Session = Depends(get_session),
) -> CompareResultsResponse:
    dataset_name = sanitize_user_text(payload.dataset, field_name="dataset", max_length=255)
    metric_name = sanitize_user_text(payload.metric, field_name="metric", max_length=255)

    statement = (
        select(ResultRow, Paper, Dataset, Method, Metric)
        .join(Paper, Paper.id == ResultRow.paper_id)
        .join(Dataset, Dataset.id == ResultRow.dataset_id)
        .join(Metric, Metric.id == ResultRow.metric_id)
        .outerjoin(Method, Method.id == ResultRow.method_id)
        .where(
            func.lower(Dataset.display_name) == dataset_name.lower(),
            func.lower(Metric.display_name) == metric_name.lower(),
        )
        .order_by(ResultRow.value_numeric.desc(), Paper.canonical_title.asc())
    )

    if payload.collection_id is not None:
        collection_id = sanitize_identifier(
            payload.collection_id,
            field_name="collection_id",
            max_length=36,
        )
        statement = statement.join(
            CollectionPaper,
            CollectionPaper.paper_id == ResultRow.paper_id,
        ).where(CollectionPaper.collection_id == collection_id)

    rows = session.execute(statement).all()
    return CompareResultsResponse(
        data=[
            CompareResultItemResponse(
                paper_id=paper.id,
                paper_title=paper.canonical_title,
                dataset=dataset.display_name,
                method=method.display_name if method is not None else None,
                metric=metric.display_name,
                value_numeric=result_row.value_numeric,
                value_text=result_row.value_text,
                comparator_text=result_row.comparator_text,
                notes=result_row.notes,
            )
            for result_row, paper, dataset, method, metric in rows
        ]
    )
