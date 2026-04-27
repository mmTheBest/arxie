"""Comparison routes for the Paperbase API service."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from paperbase.db.models import (
    CollectionPaper,
    Dataset,
    EngineeringTrick,
    EvidenceSpan,
    Figure,
    Method,
    Metric,
    Paper,
    ResultRow,
    TableArtifact,
)
from paperbase.db.repositories import CollectionRepository
from ra.utils.security import sanitize_identifier, sanitize_user_text
from services.paperbase_api.dependencies import get_session
from services.paperbase_api.errors import PaperbaseAPIError
from services.paperbase_api.models import (
    CompareEngineeringTrickItemResponse,
    CompareEngineeringTricksRequest,
    CompareEngineeringTricksResponse,
    CompareEvidenceSpanResponse,
    CompareFigureItemResponse,
    CompareFigureTableRequest,
    CompareFiguresResponse,
    CompareMethodBestResultResponse,
    CompareMethodItemResponse,
    CompareMethodsRequest,
    CompareMethodsResponse,
    ComparePaperReferenceResponse,
    CompareResultItemResponse,
    CompareResultsRequest,
    CompareResultsResponse,
    CompareTableItemResponse,
    CompareTablesResponse,
)
from services.paperbase_api.normalization import canonicalize_metric_display_name, normalize_summary_key

router = APIRouter(prefix="/api/v1/compare", tags=["compare"])


def _resolve_collection_id(session: Session, raw_collection_id: str | None) -> str | None:
    if raw_collection_id is None:
        return None
    collection_id = sanitize_identifier(
        raw_collection_id,
        field_name="collection_id",
        max_length=36,
    )
    if CollectionRepository(session).get_by_id(collection_id) is None:
        raise PaperbaseAPIError(
            status_code=404,
            error="collection_not_found",
            message=f"No collection found for id: {collection_id}",
        )
    return collection_id


def _load_result_rows(
    session: Session,
    *,
    collection_id: str | None,
    dataset_name: str | None,
    metric_name: str | None,
) -> list[tuple[ResultRow, Paper, Dataset | None, Method | None, Metric | None]]:
    statement = (
        select(ResultRow, Paper, Dataset, Method, Metric)
        .join(Paper, Paper.id == ResultRow.paper_id)
        .outerjoin(Dataset, Dataset.id == ResultRow.dataset_id)
        .outerjoin(Method, Method.id == ResultRow.method_id)
        .outerjoin(Metric, Metric.id == ResultRow.metric_id)
    )

    if collection_id is not None:
        statement = statement.join(
            CollectionPaper,
            CollectionPaper.paper_id == ResultRow.paper_id,
        ).where(CollectionPaper.collection_id == collection_id)

    rows = list(session.execute(statement).all())
    if dataset_name is not None:
        dataset_filter = dataset_name.casefold()
        rows = [
            row
            for row in rows
            if row[2] is not None and row[2].display_name.strip().casefold() == dataset_filter
        ]
    if metric_name is not None:
        metric_filter = canonicalize_metric_display_name(metric_name).casefold()
        rows = [
            row
            for row in rows
            if row[4] is not None
            and canonicalize_metric_display_name(row[4].display_name).casefold() == metric_filter
        ]

    rows.sort(
        key=lambda row: (
            row[0].value_numeric is None,
            -(row[0].value_numeric or 0.0),
            row[1].canonical_title.casefold(),
        )
    )
    return rows


def _load_result_evidence(
    session: Session,
    *,
    result_row_ids: list[str],
) -> dict[str, list[CompareEvidenceSpanResponse]]:
    if not result_row_ids:
        return {}

    evidence_rows = session.execute(
        select(EvidenceSpan)
        .where(
            EvidenceSpan.target_type == "result_row",
            EvidenceSpan.target_id.in_(result_row_ids),
        )
        .order_by(EvidenceSpan.page_number.asc().nullslast(), EvidenceSpan.created_at.asc())
    ).scalars().all()

    evidence_by_result_row: dict[str, list[CompareEvidenceSpanResponse]] = defaultdict(list)
    for evidence in evidence_rows:
        if evidence.target_id is None:
            continue
        evidence_by_result_row[evidence.target_id].append(
            CompareEvidenceSpanResponse(
                id=evidence.id,
                page_number=evidence.page_number,
                quote_text=evidence.quote_text,
            )
        )
    return evidence_by_result_row


def _resolve_method_paper_ids(session: Session, method_name: str | None) -> set[str] | None:
    if method_name is None:
        return None
    return {
        paper_id
        for paper_id in session.execute(
            select(Method.paper_id).where(Method.display_name.ilike(method_name))
        ).scalars()
    }


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
    collection_id = _resolve_collection_id(session, payload.collection_id)
    rows = _load_result_rows(
        session,
        collection_id=collection_id,
        dataset_name=dataset_name,
        metric_name=metric_name,
    )
    evidence_by_result_row = (
        _load_result_evidence(
            session,
            result_row_ids=[result_row.id for result_row, _, _, _, _ in rows],
        )
        if payload.include_evidence
        else {}
    )

    return CompareResultsResponse(
        data=[
            CompareResultItemResponse(
                result_row_id=result_row.id,
                paper_id=paper.id,
                paper_title=paper.canonical_title,
                dataset=dataset.display_name.strip() if dataset is not None else dataset_name,
                method=method.display_name if method is not None else None,
                metric=canonicalize_metric_display_name(metric.display_name) if metric is not None else metric_name,
                value_numeric=result_row.value_numeric,
                value_text=result_row.value_text,
                comparator_text=result_row.comparator_text,
                notes=result_row.notes,
                evidence_spans=evidence_by_result_row.get(result_row.id, []),
            )
            for result_row, paper, dataset, method, metric in rows
        ]
    )


@router.post(
    "/methods",
    response_model=CompareMethodsResponse,
)
def compare_methods(
    payload: CompareMethodsRequest,
    session: Session = Depends(get_session),
) -> CompareMethodsResponse:
    collection_id = _resolve_collection_id(session, payload.collection_id)
    dataset_name = (
        sanitize_user_text(payload.dataset, field_name="dataset", max_length=255)
        if payload.dataset is not None
        else None
    )
    metric_name = (
        sanitize_user_text(payload.metric, field_name="metric", max_length=255)
        if payload.metric is not None
        else None
    )
    rows = _load_result_rows(
        session,
        collection_id=collection_id,
        dataset_name=dataset_name,
        metric_name=metric_name,
    )

    grouped: dict[str, dict[str, Any]] = {}
    for result_row, paper, dataset, method, metric in rows:
        if method is None:
            continue
        method_name = method.display_name.strip()
        entry = grouped.setdefault(
            method_name,
            {
                "method": method_name,
                "paper_ids": set(),
                "datasets": set(),
                "metrics": set(),
                "result_count": 0,
                "best_row": None,
            },
        )
        entry["paper_ids"].add(paper.id)
        if dataset is not None:
            entry["datasets"].add(dataset.display_name.strip())
        if metric is not None:
            entry["metrics"].add(canonicalize_metric_display_name(metric.display_name))
        entry["result_count"] += 1

        best_row = entry["best_row"]
        if best_row is None:
            entry["best_row"] = (result_row, paper, dataset, metric)
        else:
            best_result_row = best_row[0]
            current_value = result_row.value_numeric if result_row.value_numeric is not None else float("-inf")
            best_value = (
                best_result_row.value_numeric
                if best_result_row.value_numeric is not None
                else float("-inf")
            )
            if current_value > best_value:
                entry["best_row"] = (result_row, paper, dataset, metric)

    items = sorted(
        grouped.values(),
        key=lambda entry: (
            -(
                entry["best_row"][0].value_numeric
                if entry["best_row"] is not None and entry["best_row"][0].value_numeric is not None
                else float("-inf")
            ),
            entry["method"].casefold(),
        ),
    )[: payload.limit]

    response_items: list[CompareMethodItemResponse] = []
    for entry in items:
        best_row = entry["best_row"]
        best_result = None
        if best_row is not None:
            result_row, paper, dataset, metric = best_row
            best_result = CompareMethodBestResultResponse(
                paper_id=paper.id,
                paper_title=paper.canonical_title,
                dataset=dataset.display_name.strip() if dataset is not None else None,
                metric=canonicalize_metric_display_name(metric.display_name) if metric is not None else None,
                value_numeric=result_row.value_numeric,
                value_text=result_row.value_text,
            )
        response_items.append(
            CompareMethodItemResponse(
                method=entry["method"],
                paper_count=len(entry["paper_ids"]),
                result_count=entry["result_count"],
                datasets=sorted(entry["datasets"]),
                metrics=sorted(entry["metrics"]),
                best_result=best_result,
            )
        )

    return CompareMethodsResponse(data=response_items)


@router.post(
    "/engineering-tricks",
    response_model=CompareEngineeringTricksResponse,
)
def compare_engineering_tricks(
    payload: CompareEngineeringTricksRequest,
    session: Session = Depends(get_session),
) -> CompareEngineeringTricksResponse:
    collection_id = _resolve_collection_id(session, payload.collection_id)
    method_name = (
        sanitize_user_text(payload.method, field_name="method", max_length=255)
        if payload.method is not None
        else None
    )

    statement = select(EngineeringTrick, Paper).join(Paper, Paper.id == EngineeringTrick.paper_id)
    if collection_id is not None:
        statement = statement.join(
            CollectionPaper,
            CollectionPaper.paper_id == EngineeringTrick.paper_id,
        ).where(CollectionPaper.collection_id == collection_id)

    rows = list(session.execute(statement).all())
    if method_name is not None:
        allowed_paper_ids = {
            paper_id
            for paper_id in session.execute(
                select(Method.paper_id).where(Method.display_name.ilike(method_name))
            ).scalars()
        }
        rows = [row for row in rows if row[1].id in allowed_paper_ids]

    grouped: dict[str, dict[str, Any]] = {}
    for trick, paper in rows:
        key = normalize_summary_key(trick.title)
        entry = grouped.setdefault(
            key,
            {
                "title": trick.title.strip(),
                "description": trick.description,
                "papers": {},
            },
        )
        entry["papers"].setdefault(
            paper.id,
            ComparePaperReferenceResponse(paper_id=paper.id, paper_title=paper.canonical_title),
        )

    items = sorted(
        grouped.values(),
        key=lambda entry: (-len(entry["papers"]), entry["title"].casefold()),
    )[: payload.limit]

    return CompareEngineeringTricksResponse(
        data=[
            CompareEngineeringTrickItemResponse(
                title=entry["title"],
                description=entry["description"],
                paper_count=len(entry["papers"]),
                papers=list(entry["papers"].values()),
            )
            for entry in items
        ]
    )


@router.post(
    "/figures",
    response_model=CompareFiguresResponse,
)
def compare_figures(
    payload: CompareFigureTableRequest,
    session: Session = Depends(get_session),
) -> CompareFiguresResponse:
    collection_id = _resolve_collection_id(session, payload.collection_id)
    method_name = (
        sanitize_user_text(payload.method, field_name="method", max_length=255)
        if payload.method is not None
        else None
    )

    statement = select(Figure, Paper).join(Paper, Paper.id == Figure.paper_id)
    if collection_id is not None:
        statement = statement.join(
            CollectionPaper,
            CollectionPaper.paper_id == Figure.paper_id,
        ).where(CollectionPaper.collection_id == collection_id)

    rows = list(session.execute(statement).all())
    allowed_paper_ids = _resolve_method_paper_ids(session, method_name)
    if allowed_paper_ids is not None:
        rows = [row for row in rows if row[1].id in allowed_paper_ids]

    rows = rows[: payload.limit]
    return CompareFiguresResponse(
        data=[
            CompareFigureItemResponse(
                id=figure.id,
                paper_id=paper.id,
                paper_title=paper.canonical_title,
                page_number=figure.page_number,
                figure_label=figure.figure_label,
                caption=figure.caption,
                storage_uri=figure.storage_uri,
            )
            for figure, paper in rows
        ]
    )


@router.post(
    "/tables",
    response_model=CompareTablesResponse,
)
def compare_tables(
    payload: CompareFigureTableRequest,
    session: Session = Depends(get_session),
) -> CompareTablesResponse:
    collection_id = _resolve_collection_id(session, payload.collection_id)
    method_name = (
        sanitize_user_text(payload.method, field_name="method", max_length=255)
        if payload.method is not None
        else None
    )

    statement = select(TableArtifact, Paper).join(Paper, Paper.id == TableArtifact.paper_id)
    if collection_id is not None:
        statement = statement.join(
            CollectionPaper,
            CollectionPaper.paper_id == TableArtifact.paper_id,
        ).where(CollectionPaper.collection_id == collection_id)

    rows = list(session.execute(statement).all())
    allowed_paper_ids = _resolve_method_paper_ids(session, method_name)
    if allowed_paper_ids is not None:
        rows = [row for row in rows if row[1].id in allowed_paper_ids]

    rows = rows[: payload.limit]
    return CompareTablesResponse(
        data=[
            CompareTableItemResponse(
                id=table.id,
                paper_id=paper.id,
                paper_title=paper.canonical_title,
                page_number=table.page_number,
                table_label=table.table_label,
                caption=table.caption,
                storage_uri=table.storage_uri,
                structured_payload=dict(table.structured_payload_json or {}),
            )
            for table, paper in rows
        ]
    )
