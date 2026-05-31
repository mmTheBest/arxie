"""Task-aware context assembly for Paperbase research-agent runs."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from paperbase.db.models import (
    Chunk,
    CollectionPaper,
    Dataset,
    EngineeringTrick,
    Finding,
    Limitation,
    Method,
    Metric,
    Paper,
    ResearchDesignElement,
    ResearchGraphEdge,
    ResearchGraphNode,
    ResearchMemoryRecord,
    ResultRow,
    Section,
    StudySource,
    Workspace,
)
from paperbase.db.repositories import ResearchIntelligenceRepository
from paperbase.research.context_ranking import (
    rank_context_papers,
    rank_context_sources,
)
from paperbase.research.memory_builder import ResearchIntelligenceMemoryBuilder
from paperbase.research.task_descriptors import task_descriptor_for
from paperbase.search.index_names import search_index_name
from paperbase.search.query_builder import build_search_query

DEFAULT_CONTEXT_PAPER_LIMIT = 40
DEFAULT_CONTEXT_PAPER_CANDIDATE_LIMIT = 80
DEFAULT_CONTEXT_SOURCE_LIMIT = 24
NAMED_ENTITY_PREVIEW_LIMIT = 12
EVIDENCE_MEMORY_LIMIT = 12
PATTERN_MEMORY_LIMIT = 12
FIELD_GRAPH_NODE_LIMIT = 24
FIELD_GRAPH_EDGE_LIMIT = 32
INTELLIGENCE_LAYER_CANDIDATE_MULTIPLIER = 4
CONTEXT_TEXT_LIMIT = 720
CONTEXT_SHORT_TEXT_LIMIT = 360
PAPER_SECTION_TEXT_LIMIT = 900
CONTEXT_VALUE_LIST_LIMIT = 12
CONTEXT_VALUE_DICT_FIELD_LIMIT = 12
STUDY_BRIEF_TEXT_LIMIT = 720
STUDY_BRIEF_LIST_LIMIT = 8
STUDY_BRIEF_LIST_ITEM_TEXT_LIMIT = 240
STUDY_BRIEF_DICT_FIELD_LIMIT = 12
SOURCE_FACT_LIMIT_PER_SOURCE = 8
SOURCE_FACT_TEXT_LIMIT = 360
BACKEND_CONTEXT_SEARCH_LIMIT = 12
BACKEND_CONTEXT_CHUNK_LIMIT = 8
BACKEND_CONTEXT_PAPER_LIMIT = 12
BACKEND_CONTEXT_CHUNK_TEXT_LIMIT = 720
BACKEND_CONTEXT_ENTITY_LIMIT = 8
BACKEND_CONTEXT_FAILURE_ERROR = "backend_search_failed"
SAFE_BACKEND_ERROR_CODE_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
SAFE_BACKEND_ERROR_TYPE_RE = re.compile(r"[^A-Za-z0-9_.-]+")
SOURCE_ERROR_PATH_PLACEHOLDER = "[registered_path]"
SOURCE_ERROR_ABSOLUTE_PATH_RE = re.compile(r"(?:(?<=\s)|(?<=:)|^)/[^\s]+")
SOURCE_FACT_LABELS = {
    "claim": "claim",
    "hypothesis": "claim",
    "method": "method",
    "approach": "method",
    "model": "method",
    "dataset": "dataset",
    "data": "dataset",
    "metric": "metric",
    "measure": "metric",
    "result": "result",
    "finding": "result",
    "observation": "result",
    "constraint": "constraint",
    "requirement": "constraint",
    "risk": "constraint",
    "limitation": "constraint",
    "open question": "open_question",
    "question": "open_question",
}


@dataclass(frozen=True, slots=True)
class ResearchContextPack:
    context: dict[str, Any]
    selected_item_counts: dict[str, int]
    readiness_warnings: list[str]
    cache_key: str


@dataclass(frozen=True, slots=True)
class BackendContextRetrieval:
    enabled: bool
    status: str
    paper_ids: list[str]
    chunks_by_paper_id: dict[str, list[dict[str, Any]]]
    diagnostics: dict[str, Any]
    warning: str | None = None


class PaperbaseResearchContextBuilder:
    """Build bounded agent context from papers, structured evidence, and user sources."""

    def __init__(
        self,
        session: Session,
        *,
        search_backend: object | None = None,
        embedding_provider: object | None = None,
        project_id: str | None = None,
        backend_retrieval_enabled: bool = False,
    ) -> None:
        self.session = session
        self.search_backend = search_backend
        self.embedding_provider = embedding_provider
        self.project_id = project_id
        self.backend_retrieval_enabled = backend_retrieval_enabled

    def build(
        self,
        *,
        collection_id: str,
        task_type: str,
        message: str,
        selected_paper_ids: list[str],
        workspace_id: str | None,
        source_ids: list[str],
    ) -> ResearchContextPack:
        study = self._study_context(workspace_id)
        pinned_paper_ids = list(study.get("pinned_paper_ids") or []) if study else []
        backend_retrieval = self._backend_context_retrieval(
            collection_id=collection_id,
            message=message,
        )
        default_paper_ids = self._default_collection_paper_ids(collection_id)
        paper_ids, role_by_paper_id = self._paper_candidates(
            collection_id=collection_id,
            selected_paper_ids=selected_paper_ids,
            pinned_paper_ids=pinned_paper_ids,
            backend_paper_ids=backend_retrieval.paper_ids,
            default_paper_ids=default_paper_ids,
            limit=DEFAULT_CONTEXT_PAPER_CANDIDATE_LIMIT,
        )
        intelligence_layers = self._intelligence_layers(
            collection_id=collection_id,
            task_type=task_type,
            workspace_id=workspace_id,
        )
        papers = rank_context_papers(
            task_type,
            self._paper_summaries(
                paper_ids,
                role_by_paper_id=role_by_paper_id,
                backend_chunks_by_paper_id=backend_retrieval.chunks_by_paper_id,
            ),
            limit=DEFAULT_CONTEXT_PAPER_LIMIT,
        )
        sources = rank_context_sources(
            task_type,
            self._study_sources(workspace_id=workspace_id, source_ids=source_ids),
            limit=DEFAULT_CONTEXT_SOURCE_LIMIT,
        )
        readiness_warnings = self._readiness_warnings(
            papers=papers,
            source_ids=source_ids,
            sources=sources,
            backend_retrieval=backend_retrieval,
        )
        context = {
            "collection_id": collection_id,
            "task_type": task_type,
            "message": message,
            "papers": papers,
            "sources": sources,
            "intelligence_layers": intelligence_layers,
        }
        if study:
            context["study"] = study
        if backend_retrieval.enabled:
            context["backend_retrieval"] = backend_retrieval.diagnostics
        selected_item_counts = {
            "papers": len(papers),
            "sources": len(sources),
            "source_facts": sum(
                len(source.get("source_facts", []))
                for source in sources
                if isinstance(source, dict)
            ),
            "study": 1 if study else 0,
            "sections": sum(len(paper.get("sections", [])) for paper in papers),
            "structured_evidence": sum(self._structured_signal_count(paper) for paper in papers),
            "evidence_memory": len(intelligence_layers["evidence_memory"]),
            "pattern_memory": len(intelligence_layers["pattern_memory"]),
            "graph_nodes": len(intelligence_layers["field_graph"]["nodes"]),
            "graph_edges": len(intelligence_layers["field_graph"]["edges"]),
            "study_brief": 1 if intelligence_layers["study_brief"] else 0,
        }
        if backend_retrieval.enabled:
            selected_item_counts["retrieved_chunks"] = sum(
                len(chunks) for chunks in backend_retrieval.chunks_by_paper_id.values()
            )
            selected_item_counts["backend_papers"] = len(backend_retrieval.paper_ids)
        return ResearchContextPack(
            context=context,
            selected_item_counts=selected_item_counts,
            readiness_warnings=readiness_warnings,
            cache_key=self._cache_key(
                collection_id=collection_id,
                task_type=task_type,
                message=message,
                workspace_id=workspace_id,
                study=study,
                workspace_cache=self._workspace_cache_input(workspace_id),
                paper_ids=[
                    paper["paper_id"] for paper in papers if isinstance(paper.get("paper_id"), str)
                ],
                source_ids=source_ids,
                source_cache=self._source_cache_inputs(
                    workspace_id=workspace_id,
                    source_ids=source_ids,
                ),
                intelligence_layers=intelligence_layers,
                backend_retrieval=backend_retrieval,
            ),
        )

    def _default_collection_paper_ids(self, collection_id: str) -> list[str]:
        return list(
            self.session.execute(
                select(CollectionPaper.paper_id)
                .where(CollectionPaper.collection_id == collection_id)
                .order_by(CollectionPaper.position.asc(), CollectionPaper.created_at.asc())
                .limit(DEFAULT_CONTEXT_PAPER_CANDIDATE_LIMIT)
            ).scalars()
        )

    def _backend_context_retrieval(
        self,
        *,
        collection_id: str,
        message: str,
    ) -> BackendContextRetrieval:
        if not self.backend_retrieval_enabled:
            return BackendContextRetrieval(
                enabled=False,
                status="disabled",
                paper_ids=[],
                chunks_by_paper_id={},
                diagnostics={},
            )

        indexes = {
            "chunks": search_index_name("chunks", project_id=self.project_id),
            "papers": search_index_name("papers", project_id=self.project_id),
        }
        if self.search_backend is None:
            return BackendContextRetrieval(
                enabled=True,
                status="unavailable",
                paper_ids=[],
                chunks_by_paper_id={},
                diagnostics={
                    "enabled": True,
                    "status": "unavailable",
                    "reason": "search_backend_not_configured",
                    "indexes": indexes,
                    "chunk_count": 0,
                    "paper_count": 0,
                    "dropped_hit_count": 0,
                    "paper_candidates": [],
                },
                warning=(
                    "Backend-assisted context retrieval unavailable; "
                    "using deterministic SQL context."
                ),
            )

        try:
            chunk_documents = self._backend_search_documents(
                index_name=indexes["chunks"],
                collection_id=collection_id,
                message=message,
                limit=BACKEND_CONTEXT_SEARCH_LIMIT,
            )
            paper_documents = self._backend_search_documents(
                index_name=indexes["papers"],
                collection_id=collection_id,
                message=message,
                limit=BACKEND_CONTEXT_SEARCH_LIMIT,
            )
            chunks_by_paper_id, chunk_paper_ids, dropped_chunk_hits = self._valid_backend_chunks(
                collection_id=collection_id,
                documents=chunk_documents,
            )
            paper_candidates, paper_candidate_ids, dropped_paper_hits = (
                self._valid_backend_paper_candidates(
                    collection_id=collection_id,
                    documents=paper_documents,
                )
            )
        except Exception as exc:  # noqa: BLE001
            return BackendContextRetrieval(
                enabled=True,
                status="failed",
                paper_ids=[],
                chunks_by_paper_id={},
                diagnostics=self._backend_failure_diagnostics(
                    exc=exc,
                    indexes=indexes,
                ),
                warning=(
                    "Backend-assisted context retrieval failed; using deterministic SQL context."
                ),
            )

        paper_ids = self._dedupe_ordered([*chunk_paper_ids, *paper_candidate_ids])
        return BackendContextRetrieval(
            enabled=True,
            status="success",
            paper_ids=paper_ids,
            chunks_by_paper_id=chunks_by_paper_id,
            diagnostics={
                "enabled": True,
                "status": "success",
                "indexes": indexes,
                "chunk_count": sum(len(chunks) for chunks in chunks_by_paper_id.values()),
                "paper_count": len(paper_ids),
                "dropped_hit_count": dropped_chunk_hits + dropped_paper_hits,
                "paper_candidates": paper_candidates,
            },
        )

    def _backend_search_documents(
        self,
        *,
        index_name: str,
        collection_id: str,
        message: str,
        limit: int,
    ) -> list[dict[str, object]]:
        filters: dict[str, object] = {"collection_ids": [collection_id]}
        if self.project_id:
            filters["project_id"] = self.project_id
        query_text = message.strip() or None
        embedding_vector = None
        if query_text and self.embedding_provider is not None:
            embedding_vector = self.embedding_provider.embed(query_text)
        query = build_search_query(
            query_text=query_text,
            filters=filters,
            embedding_vector=embedding_vector,
            k=limit,
        )
        documents = self.search_backend.search(index_name, query, limit)
        return [document for document in documents if isinstance(document, dict)]

    def _valid_backend_chunks(
        self,
        *,
        collection_id: str,
        documents: list[dict[str, object]],
    ) -> tuple[dict[str, list[dict[str, Any]]], list[str], int]:
        chunk_ids = [
            str(document.get("chunk_id"))
            for document in documents
            if isinstance(document.get("chunk_id"), str) and document.get("chunk_id")
        ]
        if not chunk_ids:
            return {}, [], len(documents)

        rows = self.session.execute(
            select(Chunk, Paper, Section)
            .join(Paper, Paper.id == Chunk.paper_id)
            .outerjoin(Section, Section.id == Chunk.section_id)
            .join(CollectionPaper, CollectionPaper.paper_id == Chunk.paper_id)
            .where(
                Chunk.id.in_(chunk_ids),
                CollectionPaper.collection_id == collection_id,
            )
        ).all()
        rows_by_chunk_id = {chunk.id: (chunk, paper, section) for chunk, paper, section in rows}
        chunks_by_paper_id: dict[str, list[dict[str, Any]]] = {}
        paper_ids: list[str] = []
        dropped_hits = 0
        seen_chunk_ids: set[str] = set()

        for document in documents:
            raw_chunk_id = document.get("chunk_id")
            if not isinstance(raw_chunk_id, str) or not raw_chunk_id:
                dropped_hits += 1
                continue
            if raw_chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(raw_chunk_id)
            row = rows_by_chunk_id.get(raw_chunk_id)
            if row is None:
                dropped_hits += 1
                continue
            retrieved_chunk_count = sum(len(chunks) for chunks in chunks_by_paper_id.values())
            if retrieved_chunk_count >= BACKEND_CONTEXT_CHUNK_LIMIT:
                break
            chunk, paper, section = row
            chunk_context = {
                "chunk_id": chunk.id,
                "section_title": (
                    self._bounded_text(section.title, limit=CONTEXT_SHORT_TEXT_LIMIT)
                    if section is not None
                    else None
                ),
                "text": self._bounded_text(
                    chunk.text,
                    limit=BACKEND_CONTEXT_CHUNK_TEXT_LIMIT,
                ),
                "retrieval_rank": retrieved_chunk_count + 1,
            }
            chunks_by_paper_id.setdefault(paper.id, []).append(chunk_context)
            paper_ids.append(paper.id)
        return chunks_by_paper_id, self._dedupe_ordered(paper_ids), dropped_hits

    def _valid_backend_paper_candidates(
        self,
        *,
        collection_id: str,
        documents: list[dict[str, object]],
    ) -> tuple[list[dict[str, Any]], list[str], int]:
        paper_ids = [
            str(document.get("paper_id"))
            for document in documents
            if isinstance(document.get("paper_id"), str) and document.get("paper_id")
        ]
        if not paper_ids:
            return [], [], len(documents)

        rows = self.session.execute(
            select(Paper)
            .join(CollectionPaper, CollectionPaper.paper_id == Paper.id)
            .where(
                Paper.id.in_(paper_ids),
                CollectionPaper.collection_id == collection_id,
            )
        ).scalars()
        papers_by_id = {paper.id: paper for paper in rows}
        candidates: list[dict[str, Any]] = []
        valid_paper_ids: list[str] = []
        dropped_hits = 0
        seen_paper_ids: set[str] = set()

        for document in documents:
            raw_paper_id = document.get("paper_id")
            if not isinstance(raw_paper_id, str) or not raw_paper_id:
                dropped_hits += 1
                continue
            if raw_paper_id in seen_paper_ids:
                continue
            seen_paper_ids.add(raw_paper_id)
            paper = papers_by_id.get(raw_paper_id)
            if paper is None:
                dropped_hits += 1
                continue
            if len(candidates) >= BACKEND_CONTEXT_PAPER_LIMIT:
                break
            candidates.append(
                {
                    "paper_id": paper.id,
                    "title": self._bounded_text(
                        paper.canonical_title,
                        limit=CONTEXT_SHORT_TEXT_LIMIT,
                    ),
                    "datasets": self._backend_entity_list(document.get("datasets")),
                    "methods": self._backend_entity_list(document.get("methods")),
                    "metrics": self._backend_entity_list(document.get("metrics")),
                    "retrieval_rank": len(candidates) + 1,
                }
            )
            valid_paper_ids.append(paper.id)
        return candidates, valid_paper_ids, dropped_hits

    def _backend_entity_list(self, value: object) -> list[str]:
        if not isinstance(value, list | tuple):
            return []
        return [
            self._bounded_text(str(item), limit=CONTEXT_SHORT_TEXT_LIMIT)
            for item in value[:BACKEND_CONTEXT_ENTITY_LIMIT]
            if item is not None and str(item).strip()
        ]

    def _backend_failure_diagnostics(
        self,
        *,
        exc: Exception,
        indexes: dict[str, str],
    ) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {
            "enabled": True,
            "status": "failed",
            "error": BACKEND_CONTEXT_FAILURE_ERROR,
            "reason": BACKEND_CONTEXT_FAILURE_ERROR,
            "error_type": self._safe_backend_error_type(exc),
            "indexes": indexes,
            "chunk_count": 0,
            "paper_count": 0,
            "dropped_hit_count": 0,
            "paper_candidates": [],
        }
        error_code = self._safe_backend_error_code(exc)
        if error_code is not None:
            diagnostics["error_code"] = error_code
        return diagnostics

    def _safe_backend_error_type(self, exc: Exception) -> str:
        error_type = type(exc).__name__ or "Exception"
        safe_type = SAFE_BACKEND_ERROR_TYPE_RE.sub("_", error_type).strip("_.-")
        return self._bounded_text(safe_type or "Exception", limit=CONTEXT_SHORT_TEXT_LIMIT)

    def _safe_backend_error_code(self, exc: Exception) -> int | str | None:
        for attr_name in ("status_code", "code", "errno"):
            value = getattr(exc, attr_name, None)
            if isinstance(value, int):
                return value
            if isinstance(value, str) and SAFE_BACKEND_ERROR_CODE_RE.fullmatch(value):
                return value
        return None

    def _dedupe_ordered(self, values: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            deduped.append(value)
        return deduped

    def _paper_candidates(
        self,
        *,
        collection_id: str,
        selected_paper_ids: list[str],
        pinned_paper_ids: list[str],
        backend_paper_ids: list[str],
        default_paper_ids: list[str],
        limit: int,
    ) -> tuple[list[str], dict[str, str]]:
        paper_ids: list[str] = []
        role_by_paper_id: dict[str, str] = {}
        seen: set[str] = set()
        valid_paper_ids = self._collection_member_paper_ids(
            collection_id=collection_id,
            paper_ids=[
                *selected_paper_ids,
                *pinned_paper_ids,
                *backend_paper_ids,
                *default_paper_ids,
            ],
        )

        def add_candidates(candidate_ids: list[str], role: str) -> None:
            for paper_id in candidate_ids:
                if paper_id not in valid_paper_ids:
                    continue
                if paper_id in seen:
                    continue
                if len(paper_ids) >= limit:
                    return
                paper_ids.append(paper_id)
                role_by_paper_id[paper_id] = role
                seen.add(paper_id)

        add_candidates(selected_paper_ids, "selected")
        add_candidates(pinned_paper_ids, "pinned_context")
        add_candidates(backend_paper_ids, "backend_retrieved")
        add_candidates(default_paper_ids, "collection_default")
        return paper_ids, role_by_paper_id

    def _collection_member_paper_ids(
        self,
        *,
        collection_id: str,
        paper_ids: list[str],
    ) -> set[str]:
        candidate_ids = self._dedupe_ordered(
            [paper_id for paper_id in paper_ids if isinstance(paper_id, str) and paper_id]
        )
        if not candidate_ids:
            return set()
        return set(
            self.session.execute(
                select(CollectionPaper.paper_id).where(
                    CollectionPaper.collection_id == collection_id,
                    CollectionPaper.paper_id.in_(candidate_ids),
                )
            ).scalars()
        )

    def _paper_summaries(
        self,
        paper_ids: list[str],
        *,
        role_by_paper_id: dict[str, str],
        backend_chunks_by_paper_id: dict[str, list[dict[str, Any]]] | None = None,
    ) -> list[dict[str, Any]]:
        backend_chunks_by_paper_id = backend_chunks_by_paper_id or {}
        papers = {
            paper.id: paper
            for paper in self.session.execute(select(Paper).where(Paper.id.in_(paper_ids)))
            .scalars()
            .all()
        }
        summaries: list[dict[str, Any]] = []
        for paper_id in paper_ids:
            paper = papers.get(paper_id)
            if paper is None:
                continue
            context_role = role_by_paper_id.get(paper.id, "collection_default")
            datasets, dataset_count = self._named_item_preview(Dataset, paper_id=paper_id)
            methods, method_count = self._named_item_preview(Method, paper_id=paper_id)
            metrics, metric_count = self._named_item_preview(Metric, paper_id=paper_id)
            summary = {
                "paper_id": paper.id,
                "title": self._bounded_text(
                    paper.canonical_title,
                    limit=CONTEXT_SHORT_TEXT_LIMIT,
                ),
                "context_role": context_role,
                "context_reason": self._context_reason(context_role),
                "intelligence_layer": "source_library",
                "abstract": self._bounded_optional_text(
                    paper.abstract,
                    limit=CONTEXT_TEXT_LIMIT,
                ),
                "publication_year": paper.publication_year,
                "sections": [
                    {
                        "title": self._bounded_text(
                            section.title,
                            limit=CONTEXT_SHORT_TEXT_LIMIT,
                        ),
                        "text": self._bounded_text(
                            section.text,
                            limit=PAPER_SECTION_TEXT_LIMIT,
                        ),
                    }
                    for section in self.session.execute(
                        select(Section)
                        .where(Section.paper_id == paper_id)
                        .order_by(Section.ordinal.asc())
                        .limit(4)
                    ).scalars()
                ],
                "datasets": datasets,
                "dataset_count": dataset_count,
                "methods": methods,
                "method_count": method_count,
                "metrics": metrics,
                "metric_count": metric_count,
                "limitations": [
                    self._bounded_text(item.statement, limit=CONTEXT_TEXT_LIMIT)
                    for item in self.session.execute(
                        select(Limitation)
                        .where(Limitation.paper_id == paper_id)
                        .order_by(Limitation.created_at.asc())
                        .limit(8)
                    ).scalars()
                ],
                "findings": [
                    self._bounded_text(item.statement, limit=CONTEXT_TEXT_LIMIT)
                    for item in self.session.execute(
                        select(Finding)
                        .where(Finding.paper_id == paper_id)
                        .order_by(Finding.created_at.asc())
                        .limit(8)
                    ).scalars()
                ],
                "engineering_tricks": [
                    self._bounded_text(item.title, limit=CONTEXT_SHORT_TEXT_LIMIT)
                    for item in self.session.execute(
                        select(EngineeringTrick)
                        .where(EngineeringTrick.paper_id == paper_id)
                        .order_by(EngineeringTrick.title.asc())
                        .limit(8)
                    ).scalars()
                ],
                "research_design_elements": [
                    {
                        "element_type": item.element_type,
                        "title": self._bounded_text(
                            item.title,
                            limit=CONTEXT_SHORT_TEXT_LIMIT,
                        ),
                        "description": self._bounded_text(
                            item.description,
                            limit=CONTEXT_TEXT_LIMIT,
                        ),
                        "metadata": self._bounded_context_value(dict(item.metadata_json or {})),
                    }
                    for item in self.session.execute(
                        select(ResearchDesignElement)
                        .where(ResearchDesignElement.paper_id == paper_id)
                        .order_by(
                            ResearchDesignElement.element_type.asc(),
                            ResearchDesignElement.created_at.asc(),
                        )
                        .limit(12)
                    ).scalars()
                ],
                "results": self._result_rows(paper_id),
            }
            if paper.id in backend_chunks_by_paper_id:
                summary["retrieved_chunks"] = backend_chunks_by_paper_id[paper.id]
            summaries.append(summary)
        return summaries

    def _context_reason(self, role: str) -> str:
        return {
            "selected": "The user selected this paper for the research thread.",
            "pinned_context": "The paper is pinned in the active Study.",
            "backend_retrieved": "The paper was retrieved by backend-assisted context search.",
            "collection_default": "The paper was included from the active collection scope.",
        }.get(role, "The paper was included as collection evidence.")

    def _named_item_preview(self, model, *, paper_id: str) -> tuple[list[str], int]:  # noqa: ANN001
        total_count = int(
            self.session.execute(
                select(func.count()).select_from(model).where(model.paper_id == paper_id)
            ).scalar_one()
            or 0
        )
        items = [
            item.display_name
            for item in self.session.execute(
                select(model)
                .where(model.paper_id == paper_id)
                .order_by(model.display_name.asc())
                .limit(NAMED_ENTITY_PREVIEW_LIMIT)
            ).scalars()
        ]
        return items, total_count

    def _result_rows(self, paper_id: str) -> list[dict[str, Any]]:
        rows = self.session.execute(
            select(ResultRow, Dataset, Method, Metric)
            .outerjoin(Dataset, Dataset.id == ResultRow.dataset_id)
            .outerjoin(Method, Method.id == ResultRow.method_id)
            .outerjoin(Metric, Metric.id == ResultRow.metric_id)
            .where(ResultRow.paper_id == paper_id)
            .order_by(ResultRow.value_numeric.desc().nullslast(), ResultRow.created_at.asc())
            .limit(12)
        ).all()
        return [
            {
                "dataset": dataset.display_name if dataset is not None else None,
                "method": method.display_name if method is not None else None,
                "metric": metric.display_name if metric is not None else None,
                "value_numeric": row.value_numeric,
                "value_text": self._bounded_optional_text(
                    row.value_text,
                    limit=CONTEXT_SHORT_TEXT_LIMIT,
                ),
                "notes": self._bounded_optional_text(row.notes, limit=CONTEXT_TEXT_LIMIT),
            }
            for row, dataset, method, metric in rows
        ]

    def _intelligence_layers(
        self,
        *,
        collection_id: str,
        task_type: str,
        workspace_id: str | None,
    ) -> dict[str, Any]:
        self._ensure_research_intelligence(collection_id=collection_id)
        repository = ResearchIntelligenceRepository(self.session)
        evidence_records = self._memory_records(
            repository=repository,
            collection_id=collection_id,
            workspace_id=workspace_id,
            memory_type="evidence",
            limit=EVIDENCE_MEMORY_LIMIT,
        )
        pattern_records = self._memory_records(
            repository=repository,
            collection_id=collection_id,
            workspace_id=workspace_id,
            memory_type="pattern",
            limit=PATTERN_MEMORY_LIMIT,
        )
        nodes = self._graph_nodes(
            repository=repository,
            collection_id=collection_id,
            workspace_id=workspace_id,
            limit=FIELD_GRAPH_NODE_LIMIT,
        )
        edges = self._graph_edges(
            repository=repository,
            collection_id=collection_id,
            workspace_id=workspace_id,
            limit=FIELD_GRAPH_EDGE_LIMIT,
        )
        return {
            "evidence_memory": [
                self._serialize_memory_record(record, intelligence_layer="evidence_memory")
                for record in evidence_records
            ],
            "pattern_memory": [
                self._serialize_memory_record(record, intelligence_layer="pattern_memory")
                for record in pattern_records
            ],
            "field_graph": {
                "nodes": self._rank_graph_nodes(
                    task_type=task_type,
                    nodes=nodes,
                    workspace_id=workspace_id,
                    limit=FIELD_GRAPH_NODE_LIMIT,
                ),
                "edges": self._rank_graph_edges(
                    task_type=task_type,
                    edges=edges,
                    workspace_id=workspace_id,
                    limit=FIELD_GRAPH_EDGE_LIMIT,
                ),
            },
            "study_brief": self._study_brief(
                repository=repository,
                workspace_id=workspace_id,
            ),
        }

    def _ensure_research_intelligence(self, *, collection_id: str) -> None:
        ResearchIntelligenceMemoryBuilder(self.session).build(collection_id)

    def _memory_records(
        self,
        *,
        repository: ResearchIntelligenceRepository,
        collection_id: str,
        workspace_id: str | None,
        memory_type: str,
        limit: int,
    ) -> list[ResearchMemoryRecord]:
        candidate_limit = self._candidate_pool_limit(limit)
        if workspace_id is None:
            records = list(
                repository.list_memory_records(
                    collection_id=collection_id,
                    memory_types=[memory_type],
                    limit=candidate_limit,
                )
            )
            return self._rank_memory_records(records, workspace_id=workspace_id, limit=limit)

        workspace_records = list(
            repository.list_memory_records(
                collection_id=collection_id,
                workspace_id=workspace_id,
                memory_types=[memory_type],
                limit=candidate_limit,
            )
        )
        global_records = list(
            repository.list_memory_records(
                collection_id=collection_id,
                memory_types=[memory_type],
                limit=candidate_limit,
            )
        )
        records = self._dedupe_memory_records([*workspace_records, *global_records])
        return self._rank_memory_records(records, workspace_id=workspace_id, limit=limit)

    def _candidate_pool_limit(self, output_limit: int) -> int:
        return max(output_limit, output_limit * INTELLIGENCE_LAYER_CANDIDATE_MULTIPLIER)

    def _dedupe_memory_records(
        self,
        records: list[ResearchMemoryRecord],
    ) -> list[ResearchMemoryRecord]:
        deduped: list[ResearchMemoryRecord] = []
        seen: set[tuple[str, str]] = set()
        for record in records:
            key = (record.memory_type, record.version_key)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(record)
        return deduped

    def _rank_memory_records(
        self,
        records: list[ResearchMemoryRecord],
        *,
        workspace_id: str | None,
        limit: int,
    ) -> list[ResearchMemoryRecord]:
        return sorted(
            records,
            key=lambda record: (
                0 if workspace_id is not None and record.workspace_id == workspace_id else 1,
                -self._memory_selection_score(record),
                record.memory_type,
                record.title,
                record.version_key,
                record.id,
            ),
        )[:limit]

    def _graph_nodes(
        self,
        *,
        repository: ResearchIntelligenceRepository,
        collection_id: str,
        workspace_id: str | None,
        limit: int,
    ) -> list[ResearchGraphNode]:
        candidate_limit = self._candidate_pool_limit(limit)
        if workspace_id is None:
            return list(
                repository.list_graph_nodes(
                    collection_id=collection_id,
                    limit=candidate_limit,
                )
            )

        workspace_nodes = list(
            repository.list_graph_nodes(
                collection_id=collection_id,
                workspace_id=workspace_id,
                limit=candidate_limit,
            )
        )
        global_nodes = list(
            repository.list_graph_nodes(collection_id=collection_id, limit=candidate_limit)
        )
        return self._dedupe_graph_nodes([*workspace_nodes, *global_nodes])

    def _dedupe_graph_nodes(
        self,
        nodes: list[ResearchGraphNode],
    ) -> list[ResearchGraphNode]:
        deduped: list[ResearchGraphNode] = []
        seen: set[tuple[str, str]] = set()
        for node in nodes:
            key = (node.node_type, node.stable_key)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(node)
        return deduped

    def _graph_edges(
        self,
        *,
        repository: ResearchIntelligenceRepository,
        collection_id: str,
        workspace_id: str | None,
        limit: int,
    ) -> list[ResearchGraphEdge]:
        candidate_limit = self._candidate_pool_limit(limit)
        if workspace_id is None:
            return list(
                repository.list_graph_edges(
                    collection_id=collection_id,
                    limit=candidate_limit,
                )
            )

        workspace_edges = list(
            repository.list_graph_edges(
                collection_id=collection_id,
                workspace_id=workspace_id,
                limit=candidate_limit,
            )
        )
        global_edges = list(
            repository.list_graph_edges(collection_id=collection_id, limit=candidate_limit)
        )
        return self._dedupe_graph_edges([*workspace_edges, *global_edges])

    def _dedupe_graph_edges(
        self,
        edges: list[ResearchGraphEdge],
    ) -> list[ResearchGraphEdge]:
        deduped: list[ResearchGraphEdge] = []
        seen: set[tuple[str, str, str]] = set()
        for edge in edges:
            key = (edge.source_node_id, edge.target_node_id, edge.edge_type)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(edge)
        return deduped

    def _serialize_memory_record(
        self,
        record: ResearchMemoryRecord,
        *,
        intelligence_layer: str,
    ) -> dict[str, Any]:
        source_refs = self._bounded_list(record.source_refs_json, limit=12)
        selection_features = {
            "source_refs": len(source_refs),
            "has_paper": bool(record.paper_id),
            "confidence": record.confidence,
        }
        return {
            "memory_record_id": record.id,
            "memory_type": record.memory_type,
            "paper_id": record.paper_id,
            "title": self._bounded_text(record.title, limit=CONTEXT_SHORT_TEXT_LIMIT),
            "summary": self._bounded_text(record.summary, limit=CONTEXT_TEXT_LIMIT),
            "payload": self._safe_memory_payload(record),
            "source_refs": source_refs,
            "confidence": record.confidence,
            "version_key": record.version_key,
            "updated_at": self._datetime_signal(record.updated_at),
            "content_digest": self._memory_record_digest(record),
            "context_role": intelligence_layer,
            "context_reason": self._memory_context_reason(intelligence_layer),
            "selection_score": self._memory_selection_score(record),
            "selection_features": selection_features,
            "intelligence_layer": intelligence_layer,
        }

    def _safe_memory_payload(self, record: ResearchMemoryRecord) -> dict[str, Any]:
        payload = dict(record.payload_json or {})
        if record.memory_type == "pattern":
            return {
                "pattern_type": self._bounded_context_value(payload.get("pattern_type")),
                "items": self._bounded_list(payload.get("items"), limit=12),
            }
        if record.memory_type != "evidence":
            return {}
        paper = dict(payload.get("paper") or {})
        return {
            "paper": {
                key: self._bounded_context_value(paper.get(key))
                for key in ("id", "title", "abstract", "publication_year", "venue")
                if paper.get(key) is not None
            },
            "sections": self._bounded_list(payload.get("sections"), limit=4),
            "datasets": self._bounded_list(payload.get("datasets"), limit=8),
            "methods": self._bounded_list(payload.get("methods"), limit=8),
            "metrics": self._bounded_list(payload.get("metrics"), limit=8),
            "findings": self._bounded_list(payload.get("findings"), limit=8),
            "limitations": self._bounded_list(payload.get("limitations"), limit=8),
            "research_design_elements": self._bounded_list(
                payload.get("research_design_elements"),
                limit=8,
            ),
            "result_rows": self._bounded_list(payload.get("result_rows"), limit=8),
        }

    def _memory_context_reason(self, intelligence_layer: str) -> str:
        return {
            "evidence_memory": "Included as derived evidence memory for the active collection.",
            "pattern_memory": "Included as derived pattern memory for task-level research norms.",
        }.get(intelligence_layer, "Included as derived research memory.")

    def _memory_selection_score(self, record: ResearchMemoryRecord) -> float:
        source_ref_score = min(6, len(record.source_refs_json or [])) * 1.5
        confidence_score = (record.confidence if record.confidence is not None else 0.5) * 5
        paper_score = 2.0 if record.paper_id else 0.0
        return round(source_ref_score + confidence_score + paper_score, 4)

    def _scope_rank(self, item_workspace_id: str | None, *, workspace_id: str | None) -> int:
        if workspace_id is None:
            return 0
        return 0 if item_workspace_id == workspace_id else 1

    def _rank_graph_nodes(
        self,
        *,
        task_type: str,
        nodes: list[ResearchGraphNode],
        workspace_id: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        descriptor = task_descriptor_for(task_type)
        serialized = [
            (
                node,
                self._serialize_graph_node(
                    node,
                    descriptor_weights=descriptor.context_feature_weights,
                ),
            )
            for node in nodes
        ]
        ranked = sorted(
            serialized,
            key=lambda item: (
                self._scope_rank(item[0].workspace_id, workspace_id=workspace_id),
                -float(item[1]["selection_score"]),
                item[1]["node_type"],
                item[1]["label"],
                item[1]["stable_key"],
            ),
        )
        return [item for _node, item in ranked[:limit]]

    def _serialize_graph_node(
        self,
        node: ResearchGraphNode,
        *,
        descriptor_weights: Mapping[str, float],
    ) -> dict[str, Any]:
        feature_name = self._node_feature_name(node.node_type)
        selection_features = {
            "node_type": node.node_type,
            "task_feature": feature_name,
            "paper_ids": len(self._safe_payload_list(node.payload_json, "paper_ids")),
            "occurrence_count": int(dict(node.payload_json or {}).get("occurrence_count") or 1),
        }
        score = float(descriptor_weights.get(feature_name, 1.0)) + min(
            6,
            selection_features["paper_ids"] or selection_features["occurrence_count"],
        )
        return {
            "graph_node_id": node.id,
            "node_type": node.node_type,
            "stable_key": node.stable_key,
            "label": self._bounded_text(node.label, limit=CONTEXT_SHORT_TEXT_LIMIT),
            "payload": self._safe_graph_payload(node.payload_json),
            "updated_at": self._datetime_signal(node.updated_at),
            "content_digest": self._graph_node_digest(node),
            "context_role": "field_graph_node",
            "context_reason": f"Included as a {node.node_type} node in the collection field graph.",
            "selection_score": round(score, 4),
            "selection_features": selection_features,
            "intelligence_layer": "field_graph",
        }

    def _rank_graph_edges(
        self,
        *,
        task_type: str,
        edges: list[ResearchGraphEdge],
        workspace_id: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        descriptor = task_descriptor_for(task_type)
        serialized = [
            (
                edge,
                self._serialize_graph_edge(
                    edge,
                    descriptor_weights=descriptor.context_feature_weights,
                ),
            )
            for edge in edges
        ]
        ranked = sorted(
            serialized,
            key=lambda item: (
                self._scope_rank(item[0].workspace_id, workspace_id=workspace_id),
                -float(item[1]["selection_score"]),
                item[1]["edge_type"],
                item[1]["graph_edge_id"],
            ),
        )
        return [item for _edge, item in ranked[:limit]]

    def _serialize_graph_edge(
        self,
        edge: ResearchGraphEdge,
        *,
        descriptor_weights: Mapping[str, float],
    ) -> dict[str, Any]:
        feature_name = self._edge_feature_name(edge.edge_type)
        evidence_refs = self._bounded_list(edge.evidence_refs_json, limit=12)
        selection_features = {
            "edge_type": edge.edge_type,
            "task_feature": feature_name,
            "evidence_refs": len(evidence_refs),
            "weight": edge.weight,
        }
        score = (
            float(descriptor_weights.get(feature_name, 1.0))
            + min(6, len(evidence_refs))
            + float(edge.weight or 0)
        )
        return {
            "graph_edge_id": edge.id,
            "source_node_id": edge.source_node_id,
            "target_node_id": edge.target_node_id,
            "edge_type": edge.edge_type,
            "evidence_refs": evidence_refs,
            "weight": edge.weight,
            "payload": self._safe_graph_payload(edge.payload_json),
            "updated_at": self._datetime_signal(edge.updated_at),
            "content_digest": self._graph_edge_digest(edge),
            "context_role": "field_graph_edge",
            "context_reason": (
                f"Included as a {edge.edge_type} relation in the collection field graph."
            ),
            "selection_score": round(score, 4),
            "selection_features": selection_features,
            "intelligence_layer": "field_graph",
        }

    def _study_brief(
        self,
        *,
        repository: ResearchIntelligenceRepository,
        workspace_id: str | None,
    ) -> dict[str, Any] | None:
        if workspace_id is None:
            return None
        brief = repository.get_study_brief(workspace_id)
        if brief is None:
            return None
        brief_payload = dict(brief.brief_json or {})
        brief_preview, brief_counts = self._study_brief_preview(brief_payload)
        selection_features = {
            "has_aim": bool(brief_payload.get("aim")),
            "constraints": self._brief_list_count(brief_payload.get("constraints")),
            "open_risks": self._brief_list_count(brief_payload.get("open_risks")),
            "version": brief.version,
        }
        return {
            "study_brief_id": brief.id,
            "workspace_id": brief.workspace_id,
            "brief": brief_preview,
            "brief_counts": brief_counts,
            "version": brief.version,
            "updated_by": brief.updated_by,
            "updated_at": self._datetime_signal(brief.updated_at),
            "content_digest": self._json_digest(
                {
                    "brief": brief_payload,
                    "version": brief.version,
                    "updated_by": brief.updated_by,
                }
            ),
            "context_role": "study_brief",
            "context_reason": "Included as the active Study Brief for user-owned research state.",
            "selection_score": round(10.0 + brief.version, 4),
            "selection_features": selection_features,
            "intelligence_layer": "study_brief",
        }

    def _study_brief_preview(
        self,
        brief_payload: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        preview: dict[str, Any] = {}
        counts: dict[str, Any] = {"fields": len(brief_payload)}
        for key, value in brief_payload.items():
            if isinstance(value, str):
                preview[key] = self._bounded_text(value, limit=STUDY_BRIEF_TEXT_LIMIT)
                counts[key] = {"characters": len(value)}
                continue
            if isinstance(value, list):
                preview[key] = [
                    self._bounded_study_brief_value(item) for item in value[:STUDY_BRIEF_LIST_LIMIT]
                ]
                counts[key] = {
                    "included": min(len(value), STUDY_BRIEF_LIST_LIMIT),
                    "total": len(value),
                }
                continue
            if isinstance(value, dict):
                items = list(value.items())
                preview[key] = {
                    str(item_key): self._bounded_study_brief_value(item_value)
                    for item_key, item_value in items[:STUDY_BRIEF_DICT_FIELD_LIMIT]
                }
                counts[key] = {
                    "included": min(len(items), STUDY_BRIEF_DICT_FIELD_LIMIT),
                    "total": len(items),
                }
                continue
            preview[key] = value
        return preview, counts

    def _bounded_study_brief_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._bounded_text(value, limit=STUDY_BRIEF_LIST_ITEM_TEXT_LIMIT)
        if isinstance(value, list):
            return [
                self._bounded_study_brief_value(item) for item in value[:STUDY_BRIEF_LIST_LIMIT]
            ]
        if isinstance(value, dict):
            return {
                str(item_key): self._bounded_study_brief_value(item_value)
                for item_key, item_value in list(value.items())[:STUDY_BRIEF_DICT_FIELD_LIMIT]
            }
        return value

    def _brief_list_count(self, value: Any) -> int:
        return len(value) if isinstance(value, list) else 0

    def _study_sources(
        self,
        *,
        workspace_id: str | None,
        source_ids: list[str],
    ) -> list[dict[str, Any]]:
        if workspace_id is None or not source_ids:
            return []
        sources = {
            source.id: source
            for source in self.session.execute(
                select(StudySource)
                .where(StudySource.workspace_id == workspace_id, StudySource.id.in_(source_ids))
                .order_by(StudySource.created_at.asc(), StudySource.id.asc())
            ).scalars()
        }
        summaries: list[dict[str, Any]] = []
        for source_id in source_ids:
            source = sources.get(source_id)
            if source is None:
                continue
            source_facts = self._source_facts(source)
            summaries.append(
                {
                    "source_id": source.id,
                    "source_type": source.source_type,
                    "title": self._bounded_text(
                        source.title,
                        limit=CONTEXT_SHORT_TEXT_LIMIT,
                    ),
                    "source_locator": "registered_path" if source.path else "inline_text",
                    "has_path": bool(source.path),
                    "summary": self._bounded_text(
                        source.summary or self._summarize_text(source.content or ""),
                        limit=CONTEXT_TEXT_LIMIT,
                    ),
                    "read_status": source.read_status,
                    "error_message": self._safe_source_error_message(source),
                    "source_facts": source_facts,
                    "source_fact_count": len(source_facts),
                }
            )
        return summaries

    def _safe_source_error_message(self, source: StudySource) -> str | None:
        message = self._bounded_optional_text(source.error_message, limit=CONTEXT_TEXT_LIMIT)
        if message is None:
            return None
        if source.path:
            message = message.replace(source.path, SOURCE_ERROR_PATH_PLACEHOLDER)
        return SOURCE_ERROR_ABSOLUTE_PATH_RE.sub(SOURCE_ERROR_PATH_PLACEHOLDER, message)

    def _source_facts(self, source: StudySource) -> list[dict[str, Any]]:
        text = source.content or source.summary or ""
        facts: list[dict[str, Any]] = []
        for line in text.splitlines():
            if len(facts) >= SOURCE_FACT_LIMIT_PER_SOURCE:
                break
            parsed = self._parse_source_fact_line(line)
            if parsed is None:
                continue
            fact_type, fact_text = parsed
            facts.append(
                {
                    "fact_id": f"{source.id}:fact:{len(facts)}",
                    "fact_type": fact_type,
                    "text": self._bounded_text(fact_text, limit=SOURCE_FACT_TEXT_LIMIT),
                    "source_id": source.id,
                    "source_type": source.source_type,
                    "source_title": self._bounded_text(
                        source.title,
                        limit=CONTEXT_SHORT_TEXT_LIMIT,
                    ),
                    "support_status": "user_provided",
                    "supporting_layers": ["source_library"],
                    "evidence_references": [
                        {
                            "reference_type": "study_source",
                            "source_id": source.id,
                            "label": self._bounded_text(
                                source.title,
                                limit=CONTEXT_SHORT_TEXT_LIMIT,
                            ),
                        }
                    ],
                    "context_role": "source_fact",
                    "context_reason": ("Extracted from an explicitly selected Study source."),
                    "intelligence_layer": "source_library",
                }
            )
        return facts

    def _parse_source_fact_line(self, line: str) -> tuple[str, str] | None:
        normalized = line.strip().lstrip("-*").strip()
        if not normalized or ":" not in normalized:
            return None
        label, value = normalized.split(":", 1)
        label_key = " ".join(label.strip().casefold().replace("_", " ").split())
        fact_type = SOURCE_FACT_LABELS.get(label_key)
        fact_text = value.strip()
        if fact_type is None or not fact_text:
            return None
        return fact_type, fact_text

    def _study_context(self, workspace_id: str | None) -> dict[str, Any] | None:
        if workspace_id is None:
            return None
        workspace = self.session.get(Workspace, workspace_id)
        if workspace is None:
            return None
        return {
            "workspace_id": workspace.id,
            "title": workspace.title,
            "description": workspace.description,
            "saved_query": workspace.saved_query,
            "focus_note": workspace.focus_note,
            "active_filters": dict(workspace.active_filters_json or {}),
            "pinned_paper_ids": list(workspace.pinned_paper_ids_json or []),
        }

    def _readiness_warnings(
        self,
        *,
        papers: list[dict[str, Any]],
        source_ids: list[str],
        sources: list[dict[str, Any]],
        backend_retrieval: BackendContextRetrieval,
    ) -> list[str]:
        warnings: list[str] = []
        if not papers:
            warnings.append("No paper evidence was available.")
        if papers and not any(paper.get("sections") for paper in papers):
            warnings.append("Selected papers are not parsed, so full-text evidence is missing.")
        if papers and not any(self._structured_signal_count(paper) for paper in papers):
            warnings.append("Structured extraction is missing for the selected papers.")
        if source_ids and not sources:
            warnings.append("Requested study sources were not available in the context pack.")
        if backend_retrieval.warning:
            warnings.append(backend_retrieval.warning)
        return warnings

    def _structured_signal_count(self, paper: dict[str, Any]) -> int:
        return sum(
            len(paper.get(key, []))
            for key in (
                "datasets",
                "methods",
                "metrics",
                "limitations",
                "findings",
                "engineering_tricks",
                "research_design_elements",
                "results",
            )
        )

    def _cache_key(
        self,
        *,
        collection_id: str,
        task_type: str,
        message: str,
        workspace_id: str | None,
        study: dict[str, Any] | None,
        workspace_cache: dict[str, Any] | None,
        paper_ids: list[str],
        source_ids: list[str],
        source_cache: list[dict[str, Any]],
        intelligence_layers: dict[str, Any],
        backend_retrieval: BackendContextRetrieval,
    ) -> str:
        digest_payload = {
            "collection_id": collection_id,
            "task_type": task_type,
            "message": message,
            "workspace_id": workspace_id,
            "study": study,
            "workspace_cache": workspace_cache,
            "paper_ids": paper_ids,
            "source_ids": source_ids,
            "source_cache": source_cache,
            "intelligence_layers": self._cache_layer_inputs(intelligence_layers),
        }
        if backend_retrieval.enabled:
            digest_payload["backend_retrieval"] = self._cache_backend_retrieval_input(
                backend_retrieval
            )
        digest = self._json_digest(digest_payload)
        return "|".join([f"collection:{collection_id}", f"task:{task_type}", f"inputs:{digest}"])

    def _cache_backend_retrieval_input(
        self,
        backend_retrieval: BackendContextRetrieval,
    ) -> dict[str, Any]:
        return {
            "status": backend_retrieval.status,
            "paper_ids": list(backend_retrieval.paper_ids),
            "chunks": [
                {
                    "paper_id": paper_id,
                    "chunk_id": chunk.get("chunk_id"),
                    "text_digest": self._json_digest(chunk.get("text")),
                    "rank": chunk.get("retrieval_rank"),
                }
                for paper_id, chunks in sorted(backend_retrieval.chunks_by_paper_id.items())
                for chunk in chunks
                if isinstance(chunk, dict)
            ],
            "paper_candidates": [
                {
                    "paper_id": item.get("paper_id"),
                    "rank": item.get("retrieval_rank"),
                    "datasets": item.get("datasets"),
                    "methods": item.get("methods"),
                    "metrics": item.get("metrics"),
                }
                for item in backend_retrieval.diagnostics.get("paper_candidates", [])
                if isinstance(item, dict)
            ],
            "dropped_hit_count": backend_retrieval.diagnostics.get("dropped_hit_count"),
        }

    def _json_digest(self, payload: Any) -> str:
        serialized = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:32]

    def _cache_layer_inputs(self, intelligence_layers: dict[str, Any]) -> dict[str, Any]:
        field_graph = intelligence_layers.get("field_graph") or {}
        study_brief = intelligence_layers.get("study_brief")
        return {
            "evidence_memory": [
                {
                    "id": item.get("memory_record_id"),
                    "version_key": item.get("version_key"),
                    "score": item.get("selection_score"),
                    "updated_at": item.get("updated_at"),
                    "content_digest": item.get("content_digest"),
                }
                for item in intelligence_layers.get("evidence_memory", [])
                if isinstance(item, dict)
            ],
            "pattern_memory": [
                {
                    "id": item.get("memory_record_id"),
                    "version_key": item.get("version_key"),
                    "score": item.get("selection_score"),
                    "updated_at": item.get("updated_at"),
                    "content_digest": item.get("content_digest"),
                }
                for item in intelligence_layers.get("pattern_memory", [])
                if isinstance(item, dict)
            ],
            "graph_nodes": [
                {
                    "id": item.get("graph_node_id"),
                    "stable_key": item.get("stable_key"),
                    "score": item.get("selection_score"),
                    "updated_at": item.get("updated_at"),
                    "content_digest": item.get("content_digest"),
                }
                for item in field_graph.get("nodes", [])
                if isinstance(item, dict)
            ],
            "graph_edges": [
                {
                    "id": item.get("graph_edge_id"),
                    "edge_type": item.get("edge_type"),
                    "score": item.get("selection_score"),
                    "updated_at": item.get("updated_at"),
                    "content_digest": item.get("content_digest"),
                }
                for item in field_graph.get("edges", [])
                if isinstance(item, dict)
            ],
            "study_brief": {
                "id": study_brief.get("study_brief_id"),
                "version": study_brief.get("version"),
                "updated_at": study_brief.get("updated_at"),
                "content_digest": study_brief.get("content_digest"),
            }
            if isinstance(study_brief, dict)
            else None,
        }

    def _workspace_cache_input(self, workspace_id: str | None) -> dict[str, Any] | None:
        if workspace_id is None:
            return None
        workspace = self.session.get(Workspace, workspace_id)
        if workspace is None:
            return {"workspace_id": workspace_id, "missing": True}
        return {
            "workspace_id": workspace.id,
            "updated_at": self._datetime_signal(workspace.updated_at),
            "content_digest": self._json_digest(
                {
                    "title": workspace.title,
                    "description": workspace.description,
                    "saved_query": workspace.saved_query,
                    "focus_note": workspace.focus_note,
                    "active_filters": dict(workspace.active_filters_json or {}),
                    "pinned_paper_ids": list(workspace.pinned_paper_ids_json or []),
                }
            ),
        }

    def _source_cache_inputs(
        self,
        *,
        workspace_id: str | None,
        source_ids: list[str],
    ) -> list[dict[str, Any]]:
        if workspace_id is None or not source_ids:
            return []
        sources = {
            source.id: source
            for source in self.session.execute(
                select(StudySource).where(
                    StudySource.workspace_id == workspace_id,
                    StudySource.id.in_(source_ids),
                )
            ).scalars()
        }
        cache_inputs: list[dict[str, Any]] = []
        for source_id in source_ids:
            source = sources.get(source_id)
            if source is None:
                cache_inputs.append({"source_id": source_id, "missing": True})
                continue
            cache_inputs.append(
                {
                    "source_id": source.id,
                    "updated_at": self._datetime_signal(source.updated_at),
                    "content_digest": self._json_digest(
                        {
                            "source_type": source.source_type,
                            "title": source.title,
                            "summary": source.summary,
                            "content": source.content,
                            "read_status": source.read_status,
                            "error_message": source.error_message,
                            "has_path": bool(source.path),
                        }
                    ),
                }
            )
        return cache_inputs

    def _memory_record_digest(self, record: ResearchMemoryRecord) -> str:
        return self._json_digest(
            {
                "title": record.title,
                "summary": record.summary,
                "payload": record.payload_json,
                "source_refs": record.source_refs_json,
                "confidence": record.confidence,
            }
        )

    def _graph_node_digest(self, node: ResearchGraphNode) -> str:
        return self._json_digest(
            {
                "label": node.label,
                "payload": node.payload_json,
            }
        )

    def _graph_edge_digest(self, edge: ResearchGraphEdge) -> str:
        return self._json_digest(
            {
                "source_node_id": edge.source_node_id,
                "target_node_id": edge.target_node_id,
                "evidence_refs": edge.evidence_refs_json,
                "weight": edge.weight,
                "payload": edge.payload_json,
            }
        )

    def _datetime_signal(self, value: datetime | None) -> str | None:
        return value.isoformat(timespec="microseconds") if value is not None else None

    def _summarize_text(self, text: str) -> str:
        cleaned = " ".join(text.strip().split())
        if len(cleaned) <= 360:
            return cleaned
        return f"{cleaned[:360].rstrip()}..."

    def _bounded_text(self, text: str | None, *, limit: int) -> str:
        cleaned = " ".join(str(text or "").strip().split())
        if len(cleaned) <= limit:
            return cleaned
        return f"{cleaned[:limit].rstrip()}..."

    def _bounded_optional_text(self, text: str | None, *, limit: int) -> str | None:
        if text is None:
            return None
        return self._bounded_text(text, limit=limit)

    def _bounded_context_value(
        self,
        value: Any,
        *,
        text_limit: int = CONTEXT_TEXT_LIMIT,
        list_limit: int = CONTEXT_VALUE_LIST_LIMIT,
        dict_field_limit: int = CONTEXT_VALUE_DICT_FIELD_LIMIT,
    ) -> Any:
        if isinstance(value, str):
            return self._bounded_text(value, limit=text_limit)
        if isinstance(value, list):
            return [
                self._bounded_context_value(
                    item,
                    text_limit=text_limit,
                    list_limit=list_limit,
                    dict_field_limit=dict_field_limit,
                )
                for item in value[:list_limit]
            ]
        if isinstance(value, dict):
            return {
                str(item_key): self._bounded_context_value(
                    item_value,
                    text_limit=text_limit,
                    list_limit=list_limit,
                    dict_field_limit=dict_field_limit,
                )
                for item_key, item_value in list(value.items())[:dict_field_limit]
            }
        return value

    def _bounded_list(self, value: Any, *, limit: int) -> list[Any]:
        if not isinstance(value, list):
            return []
        return [self._bounded_context_value(item) for item in value[:limit]]

    def _safe_payload_list(self, payload: dict[str, Any] | None, key: str) -> list[Any]:
        if not isinstance(payload, dict):
            return []
        value = payload.get(key)
        return value if isinstance(value, list) else []

    def _safe_graph_payload(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        safe = dict(payload)
        safe.pop("external_id", None)
        safe.pop("path", None)
        if "result_rows" in safe:
            safe["result_rows"] = self._bounded_list(safe["result_rows"], limit=12)
        return self._bounded_context_value(safe)

    def _node_feature_name(self, node_type: str) -> str:
        return {
            "paper": "direct_evidence",
            "method": "methods",
            "dataset": "datasets",
            "metric": "metrics",
            "finding": "findings",
            "gap": "limitations",
            "study_constraint": "study_brief",
        }.get(node_type, "direct_evidence")

    def _edge_feature_name(self, edge_type: str) -> str:
        return {
            "uses_method": "methods",
            "mentions_dataset": "datasets",
            "reports_metric": "metrics",
            "validated_by": "results",
            "supports": "findings",
            "leaves_gap": "limitations",
            "compares_against": "baselines",
            "contradicts": "contradictions",
        }.get(edge_type, "direct_evidence")
