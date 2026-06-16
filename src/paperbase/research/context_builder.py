"""Task-aware context assembly for Paperbase research-agent runs."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from paperbase.db.models import (
    Chunk,
    Collection,
    CollectionPaper,
    Dataset,
    EngineeringTrick,
    EvidenceSpan,
    ExtractionProfile,
    ExtractionRun,
    Figure,
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
    StudyBrief,
    StudySource,
    TableArtifact,
    Workspace,
)
from paperbase.db.repositories import ResearchIntelligenceRepository
from paperbase.extract.quality import (
    CollectionExtractionQuality,
    PaperExtractionQuality,
    build_collection_extraction_quality,
)
from paperbase.research.context_ranking import (
    rank_context_papers,
    rank_context_sources,
)
from paperbase.research.memory_builder import ResearchIntelligenceMemoryBuilder
from paperbase.research.task_descriptors import task_descriptor_for
from paperbase.search.index_names import search_index_name
from paperbase.search.query_builder import (
    RESULT_ROW_SEARCH_TEXT_FIELDS,
    build_search_query,
)

DEFAULT_CONTEXT_PAPER_LIMIT = 40
DEFAULT_CONTEXT_PAPER_CANDIDATE_LIMIT = 80
DEFAULT_CONTEXT_CHUNK_LIMIT = 16
DEFAULT_CONTEXT_EVIDENCE_SPAN_LIMIT = 16
DEFAULT_CONTEXT_FIGURE_LIMIT = 8
DEFAULT_CONTEXT_TABLE_LIMIT = 8
DEFAULT_CONTEXT_STRUCTURED_ENTITY_LIMIT = 24
DEFAULT_CONTEXT_RESULT_EVIDENCE_LIMIT = 16
DEFAULT_CONTEXT_SOURCE_LIMIT = 24
CONTEXT_MATERIALIZATION_ITEM_LIMIT = 8
NAMED_ENTITY_PREVIEW_LIMIT = 12
EVIDENCE_MEMORY_LIMIT = 12
PATTERN_MEMORY_LIMIT = 12
SOURCE_FACT_MEMORY_LIMIT = 16
FIELD_GRAPH_NODE_LIMIT = 24
FIELD_GRAPH_EDGE_LIMIT = 32
INTELLIGENCE_LAYER_CANDIDATE_MULTIPLIER = 4
CONTEXT_TEXT_LIMIT = 720
CONTEXT_SHORT_TEXT_LIMIT = 360
PAPER_SECTION_TEXT_LIMIT = 900
CONTEXT_CHUNK_TEXT_LIMIT = 900
CONTEXT_VALUE_LIST_LIMIT = 12
CONTEXT_VALUE_DICT_FIELD_LIMIT = 12
STUDY_BRIEF_TEXT_LIMIT = 720
STUDY_BRIEF_LIST_LIMIT = 8
STUDY_BRIEF_LIST_ITEM_TEXT_LIMIT = 240
STUDY_BRIEF_DICT_FIELD_LIMIT = 12
SOURCE_FACT_NODE_TYPES = {
    "user_claim",
    "assumption",
    "contradiction",
    "extension",
    "project_constraint",
    "method_context",
    "dataset_context",
    "metric_context",
    "result_context",
    "open_question",
    "note_context",
}
CHUNK_QUERY_STOPWORDS = {
    "about",
    "across",
    "after",
    "against",
    "also",
    "and",
    "are",
    "because",
    "current",
    "design",
    "does",
    "for",
    "from",
    "has",
    "have",
    "into",
    "paper",
    "papers",
    "request",
    "research",
    "show",
    "study",
    "that",
    "the",
    "their",
    "this",
    "with",
}
TASK_CHUNK_KEYWORDS = {
    "experiment_planning": {
        "ablation",
        "ablations",
        "baseline",
        "baselines",
        "benchmark",
        "control",
        "dataset",
        "datasets",
        "evaluation",
        "experiment",
        "experiments",
        "method",
        "methods",
        "metric",
        "metrics",
        "result",
        "results",
    },
    "benchmark_planning": {
        "baseline",
        "baselines",
        "benchmark",
        "benchmarks",
        "comparison",
        "dataset",
        "datasets",
        "evaluation",
        "leaderboard",
        "metric",
        "metrics",
        "result",
        "results",
    },
    "revision_planning": {
        "claim",
        "claims",
        "contradiction",
        "contradictions",
        "discussion",
        "evidence",
        "finding",
        "findings",
        "limitation",
        "limitations",
        "risk",
        "risks",
    },
    "quality_harness": {
        "claim",
        "claims",
        "evidence",
        "limitation",
        "limitations",
        "method",
        "methods",
        "result",
        "results",
        "support",
        "unsupported",
        "validation",
    },
    "literature_review": {
        "background",
        "discussion",
        "finding",
        "findings",
        "gap",
        "gaps",
        "limitation",
        "limitations",
        "method",
        "methods",
        "result",
        "results",
        "theme",
        "themes",
    },
}
TASK_CHUNK_SECTION_HINTS = {
    "experiment_planning": {
        "ablation",
        "analysis",
        "evaluation",
        "experiment",
        "method",
        "result",
    },
    "benchmark_planning": {
        "benchmark",
        "comparison",
        "dataset",
        "evaluation",
        "experiment",
        "metric",
        "result",
    },
    "revision_planning": {
        "conclusion",
        "discussion",
        "limitation",
        "related",
    },
    "quality_harness": {
        "discussion",
        "evaluation",
        "limitation",
        "method",
        "result",
    },
    "literature_review": {
        "abstract",
        "background",
        "conclusion",
        "discussion",
        "introduction",
        "result",
    },
}
CONTEXT_PACK_CACHE_VERSION = "study-context-pack-v6-claim-evidence-maps"
CONTEXT_MATERIALIZATION_VERSION = "context-materialization-v3"
BACKEND_CHUNK_TASK_KEYWORD_SCORE_CAP = 4
BACKEND_CHUNK_ANCHORED_EVIDENCE_SCORE = 9.0
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ResearchContextPack:
    context: dict[str, Any]
    selected_item_counts: dict[str, int]
    readiness_warnings: list[str]
    cache_key: str


class PaperbaseResearchContextBuilder:
    """Build bounded agent context from papers, structured evidence, and user sources."""

    def __init__(
        self,
        session: Session,
        *,
        search_backend: object | None = None,
        embedding_provider: object | None = None,
        project_id: str | None = None,
    ) -> None:
        self.session = session
        self.search_backend = search_backend
        self.embedding_provider = embedding_provider
        self.project_id = project_id

    def build(
        self,
        *,
        collection_id: str,
        task_type: str,
        message: str,
        selected_paper_ids: list[str],
        workspace_id: str | None,
        source_ids: list[str],
        cache_key_override: str | None = None,
    ) -> ResearchContextPack:
        study = self._study_context(workspace_id)
        pinned_paper_ids = list(study.get("pinned_paper_ids") or []) if study else []
        retrieval = self._retrieval_context(
            collection_id=collection_id,
            task_type=task_type,
            message=message,
            limit=DEFAULT_CONTEXT_PAPER_CANDIDATE_LIMIT,
        )
        retrieval_chunk_candidates = [
            chunk
            for chunk in retrieval.pop("chunks", [])
            if isinstance(chunk, dict)
        ]
        retrieval_figure_candidates = [
            figure
            for figure in retrieval.pop("figures", [])
            if isinstance(figure, dict)
        ]
        retrieval_table_candidates = [
            table
            for table in retrieval.pop("tables", [])
            if isinstance(table, dict)
        ]
        retrieval_structured_entity_candidates = [
            entity
            for entity in retrieval.pop("structured_entities", [])
            if isinstance(entity, dict)
        ]
        retrieval_result_row_candidates = [
            row
            for row in retrieval.pop("result_rows", [])
            if isinstance(row, dict)
        ]
        default_paper_ids = self._default_collection_paper_ids(collection_id)
        paper_ids, role_by_paper_id = self._paper_candidates(
            selected_paper_ids=selected_paper_ids,
            pinned_paper_ids=pinned_paper_ids,
            retrieval_paper_ids=[
                str(paper_id)
                for paper_id in retrieval.get("paper_ids", [])
                if isinstance(paper_id, str)
            ],
            default_paper_ids=default_paper_ids,
            limit=DEFAULT_CONTEXT_PAPER_CANDIDATE_LIMIT,
        )
        extraction_quality = self._collection_extraction_quality(collection_id)
        extraction_quality_by_paper_id = {
            paper_quality.paper_id: paper_quality
            for paper_quality in extraction_quality.papers
        }
        intelligence_layers = self._intelligence_layers(
            collection_id=collection_id,
            task_type=task_type,
            workspace_id=workspace_id,
            source_ids=source_ids,
        )
        papers = rank_context_papers(
            task_type,
            self._paper_summaries(
                paper_ids,
                role_by_paper_id=role_by_paper_id,
                extraction_quality_by_paper_id=extraction_quality_by_paper_id,
            ),
            limit=DEFAULT_CONTEXT_PAPER_LIMIT,
        )
        chunks = self._chunks_for_selected_papers(
            retrieval_chunk_candidates,
            papers=papers,
            limit=DEFAULT_CONTEXT_CHUNK_LIMIT,
        )
        sql_chunks = self._sql_fallback_chunks(
            collection_id=collection_id,
            task_type=task_type,
            message=message,
            papers=papers,
            existing_chunk_ids=[
                str(chunk["chunk_id"])
                for chunk in chunks
                if isinstance(chunk.get("chunk_id"), str)
            ],
            limit=max(DEFAULT_CONTEXT_CHUNK_LIMIT - len(chunks), 0),
        )
        chunks = [*chunks, *sql_chunks]
        retrieval["sql_chunk_count"] = len(sql_chunks)
        retrieval["selected_chunk_count"] = len(chunks)
        evidence_spans = self._evidence_spans_for_context(
            collection_id=collection_id,
            message=message,
            papers=papers,
            chunks=chunks,
            limit=DEFAULT_CONTEXT_EVIDENCE_SPAN_LIMIT,
        )
        figures = self._figures_for_context(
            collection_id=collection_id,
            task_type=task_type,
            message=message,
            papers=papers,
            limit=DEFAULT_CONTEXT_FIGURE_LIMIT,
            retrieval_candidates=retrieval_figure_candidates,
        )
        tables = self._tables_for_context(
            collection_id=collection_id,
            task_type=task_type,
            message=message,
            papers=papers,
            limit=DEFAULT_CONTEXT_TABLE_LIMIT,
            retrieval_candidates=retrieval_table_candidates,
        )
        retrieval["selected_figure_count"] = len(figures)
        retrieval["selected_table_count"] = len(tables)
        structured_entities = self._structured_entities_for_context(
            collection_id=collection_id,
            task_type=task_type,
            message=message,
            papers=papers,
            limit=DEFAULT_CONTEXT_STRUCTURED_ENTITY_LIMIT,
            retrieval_candidates=retrieval_structured_entity_candidates,
        )
        retrieval["selected_structured_entity_count"] = len(
            [
                entity
                for entity in structured_entities
                if entity.get("context_role") == "retrieval_structured_entity"
            ]
        )
        result_evidence = self._result_evidence_for_context(
            collection_id=collection_id,
            task_type=task_type,
            message=message,
            papers=papers,
            limit=DEFAULT_CONTEXT_RESULT_EVIDENCE_LIMIT,
            retrieval_candidates=retrieval_result_row_candidates,
        )
        context_materialization = self._context_materialization(
            papers=papers,
            structured_entities=structured_entities,
            result_evidence=result_evidence,
            evidence_spans=evidence_spans,
        )
        retrieval["selected_result_evidence_count"] = len(
            [
                result
                for result in result_evidence
                if result.get("context_role") == "retrieval_result_evidence"
            ]
        )
        sources = rank_context_sources(
            task_type,
            self._study_sources(workspace_id=workspace_id, source_ids=source_ids),
            limit=DEFAULT_CONTEXT_SOURCE_LIMIT,
        )
        extraction_quality_context = self._extraction_quality_context(
            extraction_quality,
            papers=papers,
        )
        readiness_warnings = self._readiness_warnings(
            papers=papers,
            source_ids=source_ids,
            sources=sources,
        )
        context = {
            "collection_id": collection_id,
            "task_type": task_type,
            "message": message,
            "papers": papers,
            "chunks": chunks,
            "evidence_spans": evidence_spans,
            "figures": figures,
            "tables": tables,
            "structured_entities": structured_entities,
            "result_evidence": result_evidence,
            "sources": sources,
            "retrieval": retrieval,
            "context_materialization": context_materialization,
            "intelligence_layers": intelligence_layers,
            "extraction_quality": extraction_quality_context,
        }
        if study:
            context["study"] = study
        selected_extraction_quality = extraction_quality_context["selected_papers"]
        cache_key = cache_key_override
        if cache_key is None:
            cache_key = self._cache_key(
                collection_id=collection_id,
                task_type=task_type,
                message=message,
                workspace_id=workspace_id,
                study=study,
                workspace_cache=self._workspace_cache_input(workspace_id),
                paper_ids=[
                    paper["paper_id"]
                    for paper in papers
                    if isinstance(paper.get("paper_id"), str)
                ],
                source_ids=source_ids,
                source_cache=self._source_cache_inputs(
                    workspace_id=workspace_id,
                    source_ids=source_ids,
                ),
                retrieval=retrieval,
                chunks=chunks,
                evidence_spans=evidence_spans,
                figures=figures,
                tables=tables,
                structured_entities=structured_entities,
                result_evidence=result_evidence,
                context_materialization=context_materialization,
                intelligence_layers=intelligence_layers,
                extraction_quality=extraction_quality_context,
                paper_extraction_quality=self._paper_extraction_quality_cache_inputs(
                    papers
                ),
            )
        return ResearchContextPack(
            context=context,
            selected_item_counts={
                "papers": len(papers),
                "sources": len(sources),
                "study": 1 if study else 0,
                "chunks": len(chunks),
                "evidence_spans": len(evidence_spans),
                "figures": len(figures),
                "tables": len(tables),
                "structured_entities": len(structured_entities),
                "result_evidence": len(result_evidence),
                "context_materialization": 1 if context_materialization else 0,
                "sections": sum(len(paper.get("sections", [])) for paper in papers),
                "structured_evidence": sum(
                    self._structured_signal_count(paper) for paper in papers
                ),
                "evidence_memory": len(intelligence_layers["evidence_memory"]),
                "pattern_memory": len(intelligence_layers["pattern_memory"]),
                "source_fact_memory": len(intelligence_layers["source_fact_memory"]),
                "graph_nodes": len(intelligence_layers["field_graph"]["nodes"]),
                "graph_edges": len(intelligence_layers["field_graph"]["edges"]),
                "study_brief": 1 if intelligence_layers["study_brief"] else 0,
                "retrieval_matches": sum(
                    1
                    for paper in papers
                    if paper.get("context_role") == "retrieval_match"
                ),
                "fresh_extraction_papers": selected_extraction_quality[
                    "fresh_paper_count"
                ],
                "stale_extraction_papers": selected_extraction_quality[
                    "stale_paper_count"
                ],
                "missing_extraction_papers": selected_extraction_quality[
                    "missing_extraction_paper_count"
                ],
                "unresolved_evidence_spans": selected_extraction_quality[
                    "unresolved_evidence_span_count"
                ],
            },
            readiness_warnings=readiness_warnings,
            cache_key=cache_key,
        )

    def cache_lookup_key(
        self,
        *,
        collection_id: str,
        task_type: str,
        message: str,
        prompt_version: str | None,
        selected_paper_ids: list[str],
        workspace_id: str | None,
        source_ids: list[str],
    ) -> str:
        self._ensure_research_intelligence(
            collection_id=collection_id,
            workspace_id=workspace_id,
        )
        source_ids = self._unique_strings(source_ids)
        digest_payload = {
            "version": CONTEXT_PACK_CACHE_VERSION,
            "collection_id": collection_id,
            "task_type": task_type,
            "message": message,
            "prompt_version": prompt_version,
            "workspace_id": workspace_id,
            "selected_paper_ids": self._unique_strings(selected_paper_ids),
            "source_ids": source_ids,
            "project_id": self.project_id,
            "search_backend": self._component_cache_fingerprint(self.search_backend),
            "embedding_provider": self._component_cache_fingerprint(
                self.embedding_provider
            ),
            "collection": self._collection_cache_input(collection_id),
            "workspace": self._workspace_cache_input(workspace_id),
            "study_brief": self._study_brief_cache_input(workspace_id),
            "sources": self._source_cache_inputs(
                workspace_id=workspace_id,
                source_ids=source_ids,
            ),
            "research_intelligence": self._research_intelligence_cache_inputs(
                collection_id=collection_id,
                workspace_id=workspace_id,
            ),
        }
        digest = self._json_digest(digest_payload)
        return "|".join(
            [f"collection:{collection_id}", f"task:{task_type}", f"inputs:{digest}"]
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

    def _paper_candidates(
        self,
        *,
        selected_paper_ids: list[str],
        pinned_paper_ids: list[str],
        retrieval_paper_ids: list[str],
        default_paper_ids: list[str],
        limit: int,
    ) -> tuple[list[str], dict[str, str]]:
        paper_ids: list[str] = []
        role_by_paper_id: dict[str, str] = {}
        seen: set[str] = set()

        def add_candidates(candidate_ids: list[str], role: str) -> None:
            for paper_id in candidate_ids:
                if paper_id in seen:
                    continue
                if len(paper_ids) >= limit:
                    return
                paper_ids.append(paper_id)
                role_by_paper_id[paper_id] = role
                seen.add(paper_id)

        add_candidates(selected_paper_ids, "selected")
        add_candidates(pinned_paper_ids, "pinned_context")
        add_candidates(retrieval_paper_ids, "retrieval_match")
        add_candidates(default_paper_ids, "collection_default")
        return paper_ids, role_by_paper_id

    def _retrieval_context(
        self,
        *,
        collection_id: str,
        task_type: str,
        message: str,
        limit: int,
    ) -> dict[str, Any]:
        base_context: dict[str, Any] = {
            "backend_status": "unconfigured",
            "chunk_hit_count": 0,
            "figure_hit_count": 0,
            "table_hit_count": 0,
            "structured_entity_hit_count": 0,
            "result_row_hit_count": 0,
            "sql_chunk_count": 0,
            "selected_figure_count": 0,
            "selected_table_count": 0,
            "selected_structured_entity_count": 0,
            "selected_result_evidence_count": 0,
            "paper_ids": [],
        }
        if self.search_backend is None:
            return base_context

        query_text = message.strip()
        if not query_text:
            return {
                **base_context,
                "backend_status": "skipped",
            }

        try:
            filters: dict[str, object] = {"collection_ids": [collection_id]}
            if self.project_id:
                filters["project_id"] = self.project_id
            embedding_vector = (
                self.embedding_provider.embed(query_text)
                if self.embedding_provider is not None
                else None
            )
            query = build_search_query(
                query_text=query_text,
                filters=filters,
                embedding_vector=embedding_vector,
                k=limit,
            )
            chunk_documents = self.search_backend.search(
                search_index_name("chunks", project_id=self.project_id),
                query,
                limit,
            )
            figure_documents = self.search_backend.search(
                search_index_name("figures", project_id=self.project_id),
                query,
                limit,
            )
            table_documents = self.search_backend.search(
                search_index_name("tables", project_id=self.project_id),
                query,
                limit,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Agent context search failed for collection_id=%s; "
                "falling back to SQL candidates: %s",
                collection_id,
                exc,
            )
            return {
                **base_context,
                "backend_status": "fallback",
            }

        try:
            structured_entity_documents = self.search_backend.search(
                search_index_name("structured-entities", project_id=self.project_id),
                query,
                limit,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Agent structured-entity context search failed for "
                "collection_id=%s; continuing without structured entity hits: %s",
                collection_id,
                exc,
            )
            structured_entity_documents = []

        try:
            result_row_query = build_search_query(
                query_text=query_text,
                filters=filters,
                embedding_vector=embedding_vector,
                text_fields=RESULT_ROW_SEARCH_TEXT_FIELDS,
                k=limit,
            )
            result_row_documents = self.search_backend.search(
                search_index_name("result-rows", project_id=self.project_id),
                result_row_query,
                limit,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Agent result-row context search failed for "
                "collection_id=%s; continuing without result row hits: %s",
                collection_id,
                exc,
            )
            result_row_documents = []

        retrieval_chunks = self._validated_retrieval_chunks(
            collection_id=collection_id,
            task_type=task_type,
            message=message,
            documents=chunk_documents,
            limit=limit,
        )
        retrieval_figures = self._validated_retrieval_artifacts(
            collection_id=collection_id,
            documents=figure_documents,
            artifact_id_key="figure_id",
            canonical_loader=self._canonical_retrieval_figures,
            limit=limit,
        )
        retrieval_tables = self._validated_retrieval_artifacts(
            collection_id=collection_id,
            documents=table_documents,
            artifact_id_key="table_id",
            canonical_loader=self._canonical_retrieval_tables,
            limit=limit,
        )
        retrieval_structured_entities = self._validated_retrieval_structured_entities(
            collection_id=collection_id,
            documents=structured_entity_documents,
            limit=limit,
        )
        retrieval_result_rows = self._validated_retrieval_result_rows(
            collection_id=collection_id,
            documents=result_row_documents,
            limit=limit,
        )
        return {
            "backend_status": "used",
            "chunk_hit_count": len(chunk_documents),
            "figure_hit_count": len(figure_documents),
            "table_hit_count": len(table_documents),
            "structured_entity_hit_count": len(structured_entity_documents),
            "result_row_hit_count": len(result_row_documents),
            "paper_ids": self._unique_strings(
                [
                    str(chunk.get("paper_id"))
                    for chunk in retrieval_chunks
                    if isinstance(chunk.get("paper_id"), str)
                ]
                + [
                    str(figure.get("paper_id"))
                    for figure in retrieval_figures
                    if isinstance(figure.get("paper_id"), str)
                ]
                + [
                    str(table.get("paper_id"))
                    for table in retrieval_tables
                    if isinstance(table.get("paper_id"), str)
                ]
                + [
                    str(entity.get("paper_id"))
                    for entity in retrieval_structured_entities
                    if isinstance(entity.get("paper_id"), str)
                ]
                + [
                    str(row.get("paper_id"))
                    for row in retrieval_result_rows
                    if isinstance(row.get("paper_id"), str)
                ]
            ),
            "chunks": retrieval_chunks,
            "figures": retrieval_figures,
            "tables": retrieval_tables,
            "structured_entities": retrieval_structured_entities,
            "result_rows": retrieval_result_rows,
        }

    def _validated_retrieval_chunks(
        self,
        *,
        collection_id: str,
        task_type: str,
        message: str,
        documents: list[dict[str, object]],
        limit: int,
    ) -> list[dict[str, Any]]:
        chunk_ids = self._unique_strings(
            [
                str(document.get("chunk_id")).strip()
                for document in documents
                if isinstance(document, Mapping)
                and isinstance(document.get("chunk_id"), str)
                and document.get("chunk_id")
            ]
        )
        if not chunk_ids:
            return []

        canonical_chunks = self._canonical_retrieval_chunks(
            collection_id=collection_id,
            chunk_ids=chunk_ids,
        )
        paper_ids = self._unique_strings(
            [
                str(chunk["paper_id"])
                for chunk in canonical_chunks.values()
                if isinstance(chunk.get("paper_id"), str)
            ]
        )
        chunk_span_rank_features, section_span_rank_features = (
            self._anchored_span_rank_features(
                collection_id=collection_id,
                paper_ids=paper_ids,
                query_terms=self._chunk_terms(message),
            )
            if paper_ids
            else ({}, {})
        )
        task_keywords = TASK_CHUNK_KEYWORDS.get(
            task_type,
            TASK_CHUNK_KEYWORDS["literature_review"],
        )
        section_hints = TASK_CHUNK_SECTION_HINTS.get(
            task_type,
            TASK_CHUNK_SECTION_HINTS["literature_review"],
        )
        candidates: list[tuple[float, int, int, dict[str, Any]]] = []
        seen_chunk_ids: set[str] = set()
        for backend_rank, document in enumerate(documents, start=1):
            if not isinstance(document, Mapping):
                continue
            paper_id = document.get("paper_id")
            chunk_id = document.get("chunk_id")
            if not isinstance(paper_id, str):
                continue
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                continue
            if self.project_id and document.get("project_id") != self.project_id:
                continue

            chunk_id = chunk_id.strip()
            if chunk_id in seen_chunk_ids:
                continue
            canonical_chunk = canonical_chunks.get(chunk_id)
            if canonical_chunk is None:
                continue
            if canonical_chunk["paper_id"] != paper_id:
                continue
            text = canonical_chunk["text"]
            if not isinstance(text, str) or not text.strip():
                continue
            seen_chunk_ids.add(chunk_id)
            section_title = str(canonical_chunk.get("section_title") or "")
            row_terms = self._chunk_terms(
                " ".join(
                    [
                        str(canonical_chunk.get("paper_title") or ""),
                        section_title,
                        text,
                    ]
                )
            )
            task_keyword_hits = len(task_keywords & row_terms)
            task_keyword_score_hits = min(
                task_keyword_hits,
                BACKEND_CHUNK_TASK_KEYWORD_SCORE_CAP,
            )
            task_section_hint = any(
                hint in section_title.lower()
                for hint in section_hints
            )
            chunk_span_features = chunk_span_rank_features.get(chunk_id, {})
            section_id = str(canonical_chunk.get("section_id") or "").strip()
            section_span_features = (
                section_span_rank_features.get(section_id, {})
                if section_id
                else {}
            )
            anchored_evidence_span_count = int(
                chunk_span_features.get("count", 0)
            )
            anchored_evidence_query_overlap = int(
                chunk_span_features.get("query_overlap", 0)
            )
            section_anchored_evidence_span_count = int(
                section_span_features.get("count", 0)
            )
            section_anchored_evidence_query_overlap = int(
                section_span_features.get("query_overlap", 0)
            )
            selection_score = round(
                (1.0 / backend_rank)
                + (task_keyword_score_hits * 1.5)
                + (2.0 if task_section_hint else 0.0)
                + (
                    anchored_evidence_span_count
                    * BACKEND_CHUNK_ANCHORED_EVIDENCE_SCORE
                )
                + (anchored_evidence_query_overlap * 3.0)
                + (section_anchored_evidence_span_count * 2.0)
                + (
                    section_anchored_evidence_query_overlap * 1.0
                    if anchored_evidence_span_count == 0
                    else 0.0
                ),
                4,
            )
            candidates.append(
                (
                    selection_score,
                    backend_rank,
                    len(candidates),
                    {
                        "chunk_id": chunk_id,
                        "paper_id": paper_id,
                        "paper_title": self._bounded_text(
                            canonical_chunk["paper_title"],
                            limit=CONTEXT_SHORT_TEXT_LIMIT,
                        ),
                        "section_title": self._bounded_optional_text(
                            section_title,
                            limit=CONTEXT_SHORT_TEXT_LIMIT,
                        ),
                        "text": self._bounded_text(
                            text,
                            limit=CONTEXT_CHUNK_TEXT_LIMIT,
                        ),
                        "context_role": "retrieval_chunk",
                        "context_reason": (
                            "Matched backend chunk retrieval for the current request."
                        ),
                        "selection_score": selection_score,
                        "selection_features": {
                            "backend_rank": backend_rank,
                            "retrieval_match": True,
                            "task_keyword_hits": task_keyword_hits,
                            "task_keyword_score_hits": task_keyword_score_hits,
                            "task_section_hint": task_section_hint,
                            "anchored_evidence_span_count": (
                                anchored_evidence_span_count
                            ),
                            "anchored_evidence_query_overlap": (
                                anchored_evidence_query_overlap
                            ),
                            "section_anchored_evidence_span_count": (
                                section_anchored_evidence_span_count
                            ),
                            "section_anchored_evidence_query_overlap": (
                                section_anchored_evidence_query_overlap
                            ),
                            "has_anchored_evidence_span": bool(
                                anchored_evidence_span_count
                                or section_anchored_evidence_span_count
                            ),
                        },
                        "intelligence_layer": "retrieval",
                    },
                )
            )
        ranked = sorted(candidates, key=lambda item: (-item[0], item[1], item[2]))
        return [chunk for *_ranking, chunk in ranked[:limit]]

    def _validated_retrieval_artifacts(
        self,
        *,
        collection_id: str,
        documents: list[dict[str, object]],
        artifact_id_key: str,
        canonical_loader: Callable[..., dict[str, dict[str, Any]]],
        limit: int,
    ) -> list[dict[str, Any]]:
        artifact_ids = self._unique_strings(
            [
                str(document.get(artifact_id_key)).strip()
                for document in documents
                if isinstance(document, Mapping)
                and isinstance(document.get(artifact_id_key), str)
                and document.get(artifact_id_key)
            ]
        )
        if not artifact_ids:
            return []

        canonical_artifacts = canonical_loader(
            collection_id=collection_id,
            artifact_ids=artifact_ids,
        )
        artifacts: list[dict[str, Any]] = []
        seen_artifact_ids: set[str] = set()
        for backend_rank, document in enumerate(documents, start=1):
            if len(artifacts) >= limit:
                break
            if not isinstance(document, Mapping):
                continue
            paper_id = document.get("paper_id")
            artifact_id = document.get(artifact_id_key)
            if not isinstance(paper_id, str):
                continue
            if not isinstance(artifact_id, str) or not artifact_id.strip():
                continue
            if self.project_id and document.get("project_id") != self.project_id:
                continue

            artifact_id = artifact_id.strip()
            if artifact_id in seen_artifact_ids:
                continue
            canonical_artifact = canonical_artifacts.get(artifact_id)
            if canonical_artifact is None:
                continue
            if canonical_artifact["paper_id"] != paper_id:
                continue
            seen_artifact_ids.add(artifact_id)
            artifacts.append(
                {
                    artifact_id_key: artifact_id,
                    "paper_id": paper_id,
                    "backend_rank": backend_rank,
                }
            )
        return artifacts

    def _validated_retrieval_structured_entities(
        self,
        *,
        collection_id: str,
        documents: list[dict[str, object]],
        limit: int,
    ) -> list[dict[str, Any]]:
        entity_keys: list[tuple[str, str]] = []
        for document in documents:
            if not isinstance(document, Mapping):
                continue
            entity_type = self._normalized_retrieval_entity_type(
                document.get("entity_type")
            )
            entity_id = document.get("entity_id")
            if entity_type is None:
                continue
            if not isinstance(entity_id, str) or not entity_id.strip():
                continue
            entity_keys.append((entity_type, entity_id.strip()))

        if not entity_keys:
            return []

        canonical_entities = self._canonical_retrieval_structured_entities(
            collection_id=collection_id,
            entity_keys=self._unique_entity_keys(entity_keys),
        )
        entities: list[dict[str, Any]] = []
        seen_entity_keys: set[tuple[str, str]] = set()
        for backend_rank, document in enumerate(documents, start=1):
            if len(entities) >= limit:
                break
            if not isinstance(document, Mapping):
                continue
            paper_id = document.get("paper_id")
            entity_type = self._normalized_retrieval_entity_type(
                document.get("entity_type")
            )
            entity_id = document.get("entity_id")
            if not isinstance(paper_id, str):
                continue
            if entity_type is None:
                continue
            if not isinstance(entity_id, str) or not entity_id.strip():
                continue
            if self.project_id and document.get("project_id") != self.project_id:
                continue

            entity_key = (entity_type, entity_id.strip())
            if entity_key in seen_entity_keys:
                continue
            canonical_entity = canonical_entities.get(entity_key)
            if canonical_entity is None:
                continue
            if canonical_entity["paper_id"] != paper_id:
                continue
            seen_entity_keys.add(entity_key)
            entities.append(
                {
                    "entity_id": entity_key[1],
                    "entity_type": entity_type,
                    "paper_id": paper_id,
                    "backend_rank": backend_rank,
                }
            )
        return entities

    def _canonical_retrieval_structured_entities(
        self,
        *,
        collection_id: str,
        entity_keys: list[tuple[str, str]],
    ) -> dict[tuple[str, str], dict[str, Any]]:
        entities: dict[tuple[str, str], dict[str, Any]] = {}
        for entity_type, model in (
            ("dataset", Dataset),
            ("method", Method),
            ("metric", Metric),
        ):
            entity_ids = [
                entity_id
                for key_entity_type, entity_id in entity_keys
                if key_entity_type == entity_type
            ]
            if not entity_ids:
                continue
            rows = self.session.execute(
                select(
                    model.id.label("entity_id"),
                    model.paper_id.label("paper_id"),
                )
                .join(CollectionPaper, CollectionPaper.paper_id == model.paper_id)
                .where(
                    CollectionPaper.collection_id == collection_id,
                    model.id.in_(entity_ids),
                )
            )
            for row in rows:
                entities[(entity_type, row.entity_id)] = {
                    "paper_id": row.paper_id,
                }
        return entities

    def _validated_retrieval_result_rows(
        self,
        *,
        collection_id: str,
        documents: list[dict[str, object]],
        limit: int,
    ) -> list[dict[str, Any]]:
        result_row_ids = self._unique_strings(
            [
                str(document.get("result_row_id")).strip()
                for document in documents
                if isinstance(document, Mapping)
                and isinstance(document.get("result_row_id"), str)
                and document.get("result_row_id")
            ]
        )
        if not result_row_ids:
            return []

        canonical_rows = self._canonical_retrieval_result_rows(
            collection_id=collection_id,
            result_row_ids=result_row_ids,
        )
        result_rows: list[dict[str, Any]] = []
        seen_result_row_ids: set[str] = set()
        for backend_rank, document in enumerate(documents, start=1):
            if len(result_rows) >= limit:
                break
            if not isinstance(document, Mapping):
                continue
            paper_id = document.get("paper_id")
            result_row_id = document.get("result_row_id")
            if not isinstance(paper_id, str):
                continue
            if not isinstance(result_row_id, str) or not result_row_id.strip():
                continue
            if self.project_id and document.get("project_id") != self.project_id:
                continue

            result_row_id = result_row_id.strip()
            if result_row_id in seen_result_row_ids:
                continue
            canonical_row = canonical_rows.get(result_row_id)
            if canonical_row is None:
                continue
            if canonical_row["paper_id"] != paper_id:
                continue
            seen_result_row_ids.add(result_row_id)
            result_rows.append(
                {
                    "result_row_id": result_row_id,
                    "paper_id": paper_id,
                    "backend_rank": backend_rank,
                }
            )
        return result_rows

    def _canonical_retrieval_result_rows(
        self,
        *,
        collection_id: str,
        result_row_ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        if not result_row_ids:
            return {}

        rows = self.session.execute(
            select(
                ResultRow.id.label("result_row_id"),
                ResultRow.paper_id.label("paper_id"),
            )
            .join(CollectionPaper, CollectionPaper.paper_id == ResultRow.paper_id)
            .where(
                CollectionPaper.collection_id == collection_id,
                ResultRow.id.in_(result_row_ids),
            )
        )
        return {
            row.result_row_id: {
                "paper_id": row.paper_id,
            }
            for row in rows
        }

    def _structured_entity_backend_rank_by_key(
        self,
        candidates: list[dict[str, Any]],
    ) -> dict[tuple[str, str], int]:
        ranks: dict[tuple[str, str], int] = {}
        for candidate in candidates:
            entity_type = self._normalized_retrieval_entity_type(
                candidate.get("entity_type")
            )
            entity_id = candidate.get("entity_id")
            backend_rank = candidate.get("backend_rank")
            if entity_type is None:
                continue
            if not isinstance(entity_id, str) or not entity_id.strip():
                continue
            if not isinstance(backend_rank, int) or isinstance(backend_rank, bool):
                continue
            if backend_rank <= 0:
                continue
            ranks.setdefault((entity_type, entity_id.strip()), backend_rank)
        return ranks

    def _result_row_backend_rank_by_id(
        self,
        candidates: list[dict[str, Any]],
    ) -> dict[str, int]:
        ranks: dict[str, int] = {}
        for candidate in candidates:
            result_row_id = candidate.get("result_row_id")
            backend_rank = candidate.get("backend_rank")
            if not isinstance(result_row_id, str) or not result_row_id.strip():
                continue
            if not isinstance(backend_rank, int) or isinstance(backend_rank, bool):
                continue
            if backend_rank <= 0:
                continue
            ranks.setdefault(result_row_id.strip(), backend_rank)
        return ranks

    @staticmethod
    def _normalized_retrieval_entity_type(value: object) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip().lower()
        if normalized in {"dataset", "method", "metric"}:
            return normalized
        return None

    @staticmethod
    def _unique_entity_keys(
        entity_keys: list[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        unique_keys: list[tuple[str, str]] = []
        seen_keys: set[tuple[str, str]] = set()
        for entity_key in entity_keys:
            if entity_key in seen_keys:
                continue
            unique_keys.append(entity_key)
            seen_keys.add(entity_key)
        return unique_keys

    def _canonical_retrieval_chunks(
        self,
        *,
        collection_id: str,
        chunk_ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        rows = self.session.execute(
            select(
                Chunk.id.label("chunk_id"),
                Chunk.paper_id.label("paper_id"),
                Chunk.section_id.label("section_id"),
                Chunk.text.label("text"),
                Paper.canonical_title.label("paper_title"),
                Section.title.label("section_title"),
            )
            .join(CollectionPaper, CollectionPaper.paper_id == Chunk.paper_id)
            .join(Paper, Paper.id == Chunk.paper_id)
            .outerjoin(Section, Section.id == Chunk.section_id)
            .where(
                CollectionPaper.collection_id == collection_id,
                Chunk.id.in_(chunk_ids),
            )
        )
        return {
            row.chunk_id: {
                "paper_id": row.paper_id,
                "section_id": row.section_id,
                "text": row.text,
                "paper_title": row.paper_title,
                "section_title": row.section_title,
            }
            for row in rows
        }

    def _canonical_retrieval_figures(
        self,
        *,
        collection_id: str,
        artifact_ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        rows = self.session.execute(
            select(
                Figure.id.label("figure_id"),
                Figure.paper_id.label("paper_id"),
            )
            .join(CollectionPaper, CollectionPaper.paper_id == Figure.paper_id)
            .where(
                CollectionPaper.collection_id == collection_id,
                Figure.id.in_(artifact_ids),
            )
        )
        return {
            row.figure_id: {
                "paper_id": row.paper_id,
            }
            for row in rows
        }

    def _canonical_retrieval_tables(
        self,
        *,
        collection_id: str,
        artifact_ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        rows = self.session.execute(
            select(
                TableArtifact.id.label("table_id"),
                TableArtifact.paper_id.label("paper_id"),
            )
            .join(CollectionPaper, CollectionPaper.paper_id == TableArtifact.paper_id)
            .where(
                CollectionPaper.collection_id == collection_id,
                TableArtifact.id.in_(artifact_ids),
            )
        )
        return {
            row.table_id: {
                "paper_id": row.paper_id,
            }
            for row in rows
        }

    def _artifact_backend_rank_by_id(
        self,
        candidates: list[dict[str, Any]],
        *,
        id_key: str,
    ) -> dict[str, int]:
        ranks: dict[str, int] = {}
        for candidate in candidates:
            artifact_id = candidate.get(id_key)
            backend_rank = candidate.get("backend_rank")
            if not isinstance(artifact_id, str) or not artifact_id.strip():
                continue
            if not isinstance(backend_rank, int) or isinstance(backend_rank, bool):
                continue
            if backend_rank <= 0:
                continue
            ranks.setdefault(artifact_id.strip(), backend_rank)
        return ranks

    def _chunks_for_selected_papers(
        self,
        chunks: list[dict[str, Any]],
        *,
        papers: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        selected_paper_ids = {
            paper["paper_id"]
            for paper in papers
            if isinstance(paper.get("paper_id"), str)
        }
        selected_chunks = [
            chunk
            for chunk in chunks
            if isinstance(chunk.get("paper_id"), str)
            and chunk["paper_id"] in selected_paper_ids
        ]
        return selected_chunks[:limit]

    def _sql_fallback_chunks(
        self,
        *,
        collection_id: str,
        task_type: str,
        message: str,
        papers: list[dict[str, Any]],
        existing_chunk_ids: list[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        paper_ids = [
            str(paper["paper_id"])
            for paper in papers
            if isinstance(paper.get("paper_id"), str)
        ]
        if not paper_ids:
            return []

        existing_ids = set(existing_chunk_ids)
        paper_rank = {paper_id: index + 1 for index, paper_id in enumerate(paper_ids)}
        query_terms = self._chunk_terms(message)
        task_keywords = TASK_CHUNK_KEYWORDS.get(
            task_type,
            TASK_CHUNK_KEYWORDS["literature_review"],
        )
        section_hints = TASK_CHUNK_SECTION_HINTS.get(
            task_type,
            TASK_CHUNK_SECTION_HINTS["literature_review"],
        )
        chunk_span_rank_features, section_span_rank_features = (
            self._anchored_span_rank_features(
                collection_id=collection_id,
                paper_ids=paper_ids,
                query_terms=query_terms,
            )
        )
        rows = self.session.execute(
            select(
                Chunk.id.label("chunk_id"),
                Chunk.paper_id.label("paper_id"),
                Chunk.section_id.label("section_id"),
                Chunk.text.label("text"),
                Chunk.ordinal.label("chunk_ordinal"),
                Paper.canonical_title.label("paper_title"),
                Section.title.label("section_title"),
                Section.ordinal.label("section_ordinal"),
            )
            .join(CollectionPaper, CollectionPaper.paper_id == Chunk.paper_id)
            .join(Paper, Paper.id == Chunk.paper_id)
            .outerjoin(Section, Section.id == Chunk.section_id)
            .where(
                CollectionPaper.collection_id == collection_id,
                Chunk.paper_id.in_(paper_ids),
            )
            .order_by(
                CollectionPaper.position.asc(),
                Section.ordinal.asc().nullslast(),
                Chunk.ordinal.asc(),
                Chunk.id.asc(),
            )
        ).all()

        candidates: list[tuple[float, int, int, int, int, dict[str, Any]]] = []
        seen_chunk_ids: set[str] = set()
        paper_count = max(len(paper_rank), 1)
        for row_index, row in enumerate(rows):
            chunk_id = str(row.chunk_id or "").strip()
            if not chunk_id or chunk_id in existing_ids or chunk_id in seen_chunk_ids:
                continue
            text = str(row.text or "").strip()
            if not text:
                continue
            seen_chunk_ids.add(chunk_id)
            section_title = str(row.section_title or "")
            row_terms = self._chunk_terms(
                " ".join([str(row.paper_title or ""), section_title, text])
            )
            query_overlap = len(query_terms & row_terms)
            task_keyword_hits = len(task_keywords & row_terms)
            task_section_hint = any(
                hint in section_title.lower()
                for hint in section_hints
            )
            chunk_span_features = chunk_span_rank_features.get(chunk_id, {})
            section_id = str(row.section_id or "").strip()
            section_span_features = (
                section_span_rank_features.get(section_id, {})
                if section_id
                else {}
            )
            anchored_evidence_span_count = int(
                chunk_span_features.get("count", 0)
            )
            anchored_evidence_query_overlap = int(
                chunk_span_features.get("query_overlap", 0)
            )
            section_anchored_evidence_span_count = int(
                section_span_features.get("count", 0)
            )
            section_anchored_evidence_query_overlap = int(
                section_span_features.get("query_overlap", 0)
            )
            row_paper_rank = paper_rank.get(row.paper_id, paper_count + 1)
            paper_rank_score = max(
                0.0,
                1.0 - ((row_paper_rank - 1) / paper_count),
            )
            selection_score = round(
                (query_overlap * 4.0)
                + (task_keyword_hits * 1.5)
                + (2.0 if task_section_hint else 0.0)
                + (anchored_evidence_span_count * 8.0)
                + (anchored_evidence_query_overlap * 3.0)
                + (section_anchored_evidence_span_count * 2.0)
                + (
                    section_anchored_evidence_query_overlap * 1.0
                    if anchored_evidence_span_count == 0
                    else 0.0
                )
                + paper_rank_score,
                4,
            )
            section_ordinal = (
                int(row.section_ordinal) if row.section_ordinal is not None else 9999
            )
            chunk_ordinal = (
                int(row.chunk_ordinal) if row.chunk_ordinal is not None else 9999
            )
            candidates.append(
                (
                    selection_score,
                    row_paper_rank,
                    section_ordinal,
                    chunk_ordinal,
                    row_index,
                    {
                        "chunk_id": chunk_id,
                        "paper_id": row.paper_id,
                        "paper_title": self._bounded_text(
                            row.paper_title,
                            limit=CONTEXT_SHORT_TEXT_LIMIT,
                        ),
                        "section_title": self._bounded_optional_text(
                            section_title if section_title else None,
                            limit=CONTEXT_SHORT_TEXT_LIMIT,
                        ),
                        "text": self._bounded_text(
                            text,
                            limit=CONTEXT_CHUNK_TEXT_LIMIT,
                        ),
                        "context_role": "sql_chunk",
                        "context_reason": (
                            "Selected from canonical SQL chunks for the current "
                            "request."
                        ),
                        "selection_score": selection_score,
                        "selection_features": {
                            "sql_fallback": True,
                            "query_overlap": query_overlap,
                            "task_keyword_hits": task_keyword_hits,
                            "task_section_hint": task_section_hint,
                            "anchored_evidence_span_count": (
                                anchored_evidence_span_count
                            ),
                            "anchored_evidence_query_overlap": (
                                anchored_evidence_query_overlap
                            ),
                            "section_anchored_evidence_span_count": (
                                section_anchored_evidence_span_count
                            ),
                            "section_anchored_evidence_query_overlap": (
                                section_anchored_evidence_query_overlap
                            ),
                            "has_anchored_evidence_span": bool(
                                anchored_evidence_span_count
                                or section_anchored_evidence_span_count
                            ),
                            "paper_rank": row_paper_rank,
                        },
                        "intelligence_layer": "retrieval",
                    },
                )
            )

        ranked = sorted(
            candidates,
            key=lambda item: (-item[0], item[1], item[2], item[3], item[4]),
        )
        return [chunk for *_ranking, chunk in ranked[:limit]]

    def _anchored_span_rank_features(
        self,
        *,
        collection_id: str,
        paper_ids: list[str],
        query_terms: set[str],
    ) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
        rows = self.session.execute(
            select(
                EvidenceSpan.chunk_id.label("chunk_id"),
                EvidenceSpan.section_id.label("section_id"),
                EvidenceSpan.quote_text.label("quote_text"),
            )
            .join(CollectionPaper, CollectionPaper.paper_id == EvidenceSpan.paper_id)
            .where(
                CollectionPaper.collection_id == collection_id,
                EvidenceSpan.paper_id.in_(paper_ids),
                EvidenceSpan.quote_text.is_not(None),
                or_(
                    EvidenceSpan.section_id.is_not(None),
                    EvidenceSpan.chunk_id.is_not(None),
                ),
            )
        ).all()
        chunk_features: dict[str, dict[str, int]] = {}
        section_features: dict[str, dict[str, int]] = {}

        def add_feature(
            features_by_id: dict[str, dict[str, int]],
            key: str | None,
            query_overlap: int,
        ) -> None:
            if not key:
                return
            features = features_by_id.setdefault(
                key,
                {"count": 0, "query_overlap": 0},
            )
            features["count"] += 1
            features["query_overlap"] = max(
                features["query_overlap"],
                query_overlap,
            )

        for row in rows:
            quote_text = str(row.quote_text or "").strip()
            if not quote_text:
                continue
            query_overlap = len(query_terms & self._chunk_terms(quote_text))
            chunk_id = str(row.chunk_id or "").strip()
            section_id = str(row.section_id or "").strip()
            add_feature(chunk_features, chunk_id or None, query_overlap)
            add_feature(section_features, section_id or None, query_overlap)

        return chunk_features, section_features

    def _evidence_spans_for_context(
        self,
        *,
        collection_id: str,
        message: str,
        papers: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        paper_ids = [
            str(paper["paper_id"])
            for paper in papers
            if isinstance(paper.get("paper_id"), str)
        ]
        if not paper_ids:
            return []

        selected_chunk_ids = {
            str(chunk["chunk_id"])
            for chunk in chunks
            if isinstance(chunk.get("chunk_id"), str)
        }
        paper_rank = {paper_id: index + 1 for index, paper_id in enumerate(paper_ids)}
        query_terms = self._chunk_terms(message)
        rows = self.session.execute(
            select(
                EvidenceSpan.id.label("evidence_span_id"),
                EvidenceSpan.paper_id.label("paper_id"),
                EvidenceSpan.section_id.label("section_id"),
                EvidenceSpan.chunk_id.label("chunk_id"),
                EvidenceSpan.target_type.label("target_type"),
                EvidenceSpan.target_id.label("target_id"),
                EvidenceSpan.page_number.label("page_number"),
                EvidenceSpan.quote_text.label("quote_text"),
                Paper.canonical_title.label("paper_title"),
                Section.title.label("section_title"),
            )
            .join(CollectionPaper, CollectionPaper.paper_id == EvidenceSpan.paper_id)
            .join(Paper, Paper.id == EvidenceSpan.paper_id)
            .outerjoin(Section, Section.id == EvidenceSpan.section_id)
            .where(
                CollectionPaper.collection_id == collection_id,
                EvidenceSpan.paper_id.in_(paper_ids),
                EvidenceSpan.quote_text.is_not(None),
                or_(
                    EvidenceSpan.section_id.is_not(None),
                    EvidenceSpan.chunk_id.is_not(None),
                ),
            )
            .order_by(
                CollectionPaper.position.asc(),
                EvidenceSpan.created_at.asc(),
                EvidenceSpan.id.asc(),
            )
        ).all()

        candidates: list[tuple[float, int, int, dict[str, Any]]] = []
        paper_count = max(len(paper_rank), 1)
        for row_index, row in enumerate(rows):
            quote_text = str(row.quote_text or "").strip()
            if not quote_text:
                continue
            section_title = str(row.section_title or "")
            span_terms = self._chunk_terms(
                " ".join(
                    [
                        str(row.paper_title or ""),
                        section_title,
                        str(row.target_type or ""),
                        quote_text,
                    ]
                )
            )
            query_overlap = len(query_terms & span_terms)
            chunk_selected = (
                isinstance(row.chunk_id, str)
                and bool(row.chunk_id)
                and row.chunk_id in selected_chunk_ids
            )
            has_chunk_anchor = isinstance(row.chunk_id, str) and bool(row.chunk_id)
            has_section_anchor = (
                isinstance(row.section_id, str) and bool(row.section_id)
            )
            selection_score = round(
                (query_overlap * 3.0)
                + (2.0 if chunk_selected else 0.0)
                + (1.0 if has_chunk_anchor else 0.0),
                4,
            )
            candidates.append(
                (
                    selection_score,
                    paper_rank.get(row.paper_id, paper_count + 1),
                    row_index,
                    {
                        "evidence_span_id": row.evidence_span_id,
                        "paper_id": row.paper_id,
                        "paper_title": self._bounded_text(
                            row.paper_title,
                            limit=CONTEXT_SHORT_TEXT_LIMIT,
                        ),
                        "target_type": row.target_type,
                        "target_id": row.target_id,
                        "section_id": row.section_id,
                        "section_title": self._bounded_optional_text(
                            section_title if section_title else None,
                            limit=CONTEXT_SHORT_TEXT_LIMIT,
                        ),
                        "chunk_id": row.chunk_id,
                        "quote_text": self._bounded_text(
                            quote_text,
                            limit=CONTEXT_TEXT_LIMIT,
                        ),
                        "page_number": row.page_number,
                        "context_role": "anchored_evidence_span",
                        "context_reason": (
                            "Anchored extraction evidence span for the current context."
                        ),
                        "selection_score": selection_score,
                        "selection_features": {
                            "query_overlap": query_overlap,
                            "chunk_selected": chunk_selected,
                            "has_chunk_anchor": has_chunk_anchor,
                            "has_section_anchor": has_section_anchor,
                        },
                        "intelligence_layer": "direct_evidence",
                    },
                )
            )

        ranked = sorted(candidates, key=lambda item: (-item[0], item[1], item[2]))
        return [span for *_ranking, span in ranked[:limit]]

    def _figures_for_context(
        self,
        *,
        collection_id: str,
        task_type: str,
        message: str,
        papers: list[dict[str, Any]],
        limit: int,
        retrieval_candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        paper_ids = [
            str(paper["paper_id"])
            for paper in papers
            if isinstance(paper.get("paper_id"), str)
        ]
        if not paper_ids:
            return []

        paper_rank = {paper_id: index + 1 for index, paper_id in enumerate(paper_ids)}
        paper_count = max(len(paper_rank), 1)
        query_terms = self._chunk_terms(message)
        task_keywords = TASK_CHUNK_KEYWORDS.get(
            task_type,
            TASK_CHUNK_KEYWORDS["literature_review"],
        )
        backend_rank_by_figure_id = self._artifact_backend_rank_by_id(
            retrieval_candidates,
            id_key="figure_id",
        )
        rows = self.session.execute(
            select(
                Figure.id.label("figure_id"),
                Figure.paper_id.label("paper_id"),
                Figure.page_number.label("page_number"),
                Figure.figure_label.label("figure_label"),
                Figure.caption.label("caption"),
                Figure.storage_uri.label("storage_uri"),
                Paper.canonical_title.label("paper_title"),
            )
            .join(CollectionPaper, CollectionPaper.paper_id == Figure.paper_id)
            .join(Paper, Paper.id == Figure.paper_id)
            .where(
                CollectionPaper.collection_id == collection_id,
                Figure.paper_id.in_(paper_ids),
            )
            .order_by(
                CollectionPaper.position.asc(),
                Figure.page_number.asc().nullslast(),
                Figure.figure_label.asc().nullslast(),
                Figure.created_at.asc(),
                Figure.id.asc(),
            )
        ).all()

        candidates: list[tuple[float, int, int, dict[str, Any]]] = []
        for row_index, row in enumerate(rows):
            label = str(row.figure_label or "").strip()
            caption = str(row.caption or "").strip()
            if not label and not caption:
                continue
            row_terms = self._chunk_terms(
                " ".join(
                    [
                        str(row.paper_title or ""),
                        "figure",
                        label,
                        caption,
                    ]
                )
            )
            query_overlap = len(query_terms & row_terms)
            task_keyword_hits = len(task_keywords & row_terms)
            backend_rank = backend_rank_by_figure_id.get(row.figure_id)
            retrieval_match = backend_rank is not None
            row_paper_rank = paper_rank.get(row.paper_id, paper_count + 1)
            paper_rank_score = max(
                0.0,
                1.0 - ((row_paper_rank - 1) / paper_count),
            )
            retrieval_score = (
                50.0 + (1.0 / backend_rank)
                if backend_rank is not None and backend_rank > 0
                else 0.0
            )
            selection_score = round(
                (query_overlap * 3.0)
                + (task_keyword_hits * 1.5)
                + (0.5 if caption else 0.0)
                + paper_rank_score
                + retrieval_score,
                4,
            )
            selection_features: dict[str, Any] = {
                "query_overlap": query_overlap,
                "task_keyword_hits": task_keyword_hits,
                "has_caption": bool(caption),
                "has_asset": bool(row.storage_uri),
                "paper_rank": row_paper_rank,
                "retrieval_match": retrieval_match,
            }
            if backend_rank is not None:
                selection_features["backend_rank"] = backend_rank
            candidates.append(
                (
                    selection_score,
                    row_paper_rank,
                    row_index,
                    {
                        "figure_id": row.figure_id,
                        "paper_id": row.paper_id,
                        "paper_title": self._bounded_text(
                            row.paper_title,
                            limit=CONTEXT_SHORT_TEXT_LIMIT,
                        ),
                        "figure_label": self._bounded_optional_text(
                            label if label else None,
                            limit=CONTEXT_SHORT_TEXT_LIMIT,
                        ),
                        "caption": self._bounded_optional_text(
                            caption if caption else None,
                            limit=CONTEXT_TEXT_LIMIT,
                        ),
                        "page_number": row.page_number,
                        "has_asset": bool(row.storage_uri),
                        "context_role": (
                            "retrieval_figure"
                            if retrieval_match
                            else "figure_evidence"
                        ),
                        "context_reason": (
                            "Matched backend figure retrieval for the current request."
                            if retrieval_match
                            else "Selected figure caption evidence for the current request."
                        ),
                        "selection_score": selection_score,
                        "selection_features": selection_features,
                        "intelligence_layer": "visual_evidence",
                    },
                )
            )

        ranked = sorted(candidates, key=lambda item: (-item[0], item[1], item[2]))
        return [figure for *_ranking, figure in ranked[:limit]]

    def _tables_for_context(
        self,
        *,
        collection_id: str,
        task_type: str,
        message: str,
        papers: list[dict[str, Any]],
        limit: int,
        retrieval_candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        paper_ids = [
            str(paper["paper_id"])
            for paper in papers
            if isinstance(paper.get("paper_id"), str)
        ]
        if not paper_ids:
            return []

        paper_rank = {paper_id: index + 1 for index, paper_id in enumerate(paper_ids)}
        paper_count = max(len(paper_rank), 1)
        query_terms = self._chunk_terms(message)
        task_keywords = TASK_CHUNK_KEYWORDS.get(
            task_type,
            TASK_CHUNK_KEYWORDS["literature_review"],
        )
        backend_rank_by_table_id = self._artifact_backend_rank_by_id(
            retrieval_candidates,
            id_key="table_id",
        )
        rows = self.session.execute(
            select(
                TableArtifact.id.label("table_id"),
                TableArtifact.paper_id.label("paper_id"),
                TableArtifact.page_number.label("page_number"),
                TableArtifact.table_label.label("table_label"),
                TableArtifact.caption.label("caption"),
                TableArtifact.storage_uri.label("storage_uri"),
                TableArtifact.structured_payload_json.label("structured_payload"),
                Paper.canonical_title.label("paper_title"),
            )
            .join(CollectionPaper, CollectionPaper.paper_id == TableArtifact.paper_id)
            .join(Paper, Paper.id == TableArtifact.paper_id)
            .where(
                CollectionPaper.collection_id == collection_id,
                TableArtifact.paper_id.in_(paper_ids),
            )
            .order_by(
                CollectionPaper.position.asc(),
                TableArtifact.page_number.asc().nullslast(),
                TableArtifact.table_label.asc().nullslast(),
                TableArtifact.created_at.asc(),
                TableArtifact.id.asc(),
            )
        ).all()

        candidates: list[tuple[float, int, int, dict[str, Any]]] = []
        for row_index, row in enumerate(rows):
            label = str(row.table_label or "").strip()
            caption = str(row.caption or "").strip()
            structured_payload = (
                dict(row.structured_payload)
                if isinstance(row.structured_payload, Mapping)
                else {}
            )
            if not label and not caption and not structured_payload:
                continue
            payload_text = json.dumps(
                structured_payload,
                ensure_ascii=False,
                sort_keys=True,
            )
            row_terms = self._chunk_terms(
                " ".join(
                    [
                        str(row.paper_title or ""),
                        "table",
                        label,
                        caption,
                        payload_text,
                    ]
                )
            )
            query_overlap = len(query_terms & row_terms)
            task_keyword_hits = len(task_keywords & row_terms)
            backend_rank = backend_rank_by_table_id.get(row.table_id)
            retrieval_match = backend_rank is not None
            row_paper_rank = paper_rank.get(row.paper_id, paper_count + 1)
            paper_rank_score = max(
                0.0,
                1.0 - ((row_paper_rank - 1) / paper_count),
            )
            retrieval_score = (
                50.0 + (1.0 / backend_rank)
                if backend_rank is not None and backend_rank > 0
                else 0.0
            )
            selection_score = round(
                (query_overlap * 3.0)
                + (task_keyword_hits * 1.5)
                + (1.0 if structured_payload else 0.0)
                + (0.5 if caption else 0.0)
                + paper_rank_score
                + retrieval_score,
                4,
            )
            selection_features: dict[str, Any] = {
                "query_overlap": query_overlap,
                "task_keyword_hits": task_keyword_hits,
                "has_caption": bool(caption),
                "has_structured_payload": bool(structured_payload),
                "has_asset": bool(row.storage_uri),
                "paper_rank": row_paper_rank,
                "retrieval_match": retrieval_match,
            }
            if backend_rank is not None:
                selection_features["backend_rank"] = backend_rank
            candidates.append(
                (
                    selection_score,
                    row_paper_rank,
                    row_index,
                    {
                        "table_id": row.table_id,
                        "paper_id": row.paper_id,
                        "paper_title": self._bounded_text(
                            row.paper_title,
                            limit=CONTEXT_SHORT_TEXT_LIMIT,
                        ),
                        "table_label": self._bounded_optional_text(
                            label if label else None,
                            limit=CONTEXT_SHORT_TEXT_LIMIT,
                        ),
                        "caption": self._bounded_optional_text(
                            caption if caption else None,
                            limit=CONTEXT_TEXT_LIMIT,
                        ),
                        "page_number": row.page_number,
                        "structured_payload": self._bounded_context_value(
                            structured_payload
                        ),
                        "has_asset": bool(row.storage_uri),
                        "context_role": (
                            "retrieval_table"
                            if retrieval_match
                            else "table_evidence"
                        ),
                        "context_reason": (
                            "Matched backend table retrieval for the current request."
                            if retrieval_match
                            else "Selected table evidence for the current request."
                        ),
                        "selection_score": selection_score,
                        "selection_features": selection_features,
                        "intelligence_layer": "table_evidence",
                    },
                )
            )

        ranked = sorted(candidates, key=lambda item: (-item[0], item[1], item[2]))
        return [table for *_ranking, table in ranked[:limit]]

    def _structured_entities_for_context(
        self,
        *,
        collection_id: str,
        task_type: str,
        message: str,
        papers: list[dict[str, Any]],
        limit: int,
        retrieval_candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        paper_ids = [
            str(paper["paper_id"])
            for paper in papers
            if isinstance(paper.get("paper_id"), str)
        ]
        if not paper_ids:
            return []

        paper_rank = {paper_id: index + 1 for index, paper_id in enumerate(paper_ids)}
        paper_count = max(len(paper_rank), 1)
        query_terms = self._chunk_terms(message)
        task_keywords = TASK_CHUNK_KEYWORDS.get(
            task_type,
            TASK_CHUNK_KEYWORDS["literature_review"],
        )
        result_count_by_entity = self._result_count_by_entity(paper_ids)
        backend_rank_by_entity_key = self._structured_entity_backend_rank_by_key(
            retrieval_candidates
        )
        entity_models = [
            ("dataset", Dataset),
            ("method", Method),
            ("metric", Metric),
        ]

        candidates: list[tuple[float, int, int, int, dict[str, Any]]] = []
        for type_index, (entity_type, model) in enumerate(entity_models):
            rows = self.session.execute(
                select(
                    model.id.label("entity_id"),
                    model.paper_id.label("paper_id"),
                    model.normalized_name.label("normalized_name"),
                    model.display_name.label("display_name"),
                    model.metadata_json.label("metadata"),
                    Paper.canonical_title.label("paper_title"),
                )
                .join(CollectionPaper, CollectionPaper.paper_id == model.paper_id)
                .join(Paper, Paper.id == model.paper_id)
                .where(
                    CollectionPaper.collection_id == collection_id,
                    model.paper_id.in_(paper_ids),
                )
                .order_by(
                    CollectionPaper.position.asc(),
                    model.display_name.asc(),
                    model.created_at.asc(),
                    model.id.asc(),
                )
            ).all()
            for row_index, row in enumerate(rows):
                display_name = str(row.display_name or "").strip()
                normalized_name = str(row.normalized_name or "").strip()
                if not display_name and not normalized_name:
                    continue
                metadata = (
                    dict(row.metadata)
                    if isinstance(row.metadata, Mapping)
                    else {}
                )
                metadata_text = json.dumps(
                    metadata,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                row_terms = self._chunk_terms(
                    " ".join(
                        [
                            str(row.paper_title or ""),
                            entity_type,
                            normalized_name,
                            display_name,
                            metadata_text,
                        ]
                    )
                )
                query_overlap = len(query_terms & row_terms)
                task_keyword_hits = len(task_keywords & row_terms)
                result_row_count = result_count_by_entity.get(
                    (entity_type, row.entity_id),
                    0,
                )
                backend_rank = backend_rank_by_entity_key.get(
                    (entity_type, row.entity_id)
                )
                retrieval_match = backend_rank is not None
                row_paper_rank = paper_rank.get(row.paper_id, paper_count + 1)
                paper_rank_score = max(
                    0.0,
                    1.0 - ((row_paper_rank - 1) / paper_count),
                )
                retrieval_score = (
                    50.0 + (1.0 / backend_rank)
                    if backend_rank is not None and backend_rank > 0
                    else 0.0
                )
                selection_score = round(
                    (query_overlap * 3.0)
                    + (task_keyword_hits * 1.5)
                    + (result_row_count * 0.75)
                    + paper_rank_score
                    + retrieval_score,
                    4,
                )
                selection_features: dict[str, Any] = {
                    "query_overlap": query_overlap,
                    "task_keyword_hits": task_keyword_hits,
                    "result_row_count": result_row_count,
                    "paper_rank": row_paper_rank,
                    "retrieval_match": retrieval_match,
                }
                if backend_rank is not None:
                    selection_features["backend_rank"] = backend_rank
                candidates.append(
                    (
                        selection_score,
                        row_paper_rank,
                        type_index,
                        row_index,
                        {
                            "entity_id": row.entity_id,
                            "entity_type": entity_type,
                            "paper_id": row.paper_id,
                            "paper_title": self._bounded_text(
                                row.paper_title,
                                limit=CONTEXT_SHORT_TEXT_LIMIT,
                            ),
                            "normalized_name": self._bounded_optional_text(
                                normalized_name if normalized_name else None,
                                limit=CONTEXT_SHORT_TEXT_LIMIT,
                            ),
                            "display_name": self._bounded_text(
                                display_name or normalized_name,
                                limit=CONTEXT_SHORT_TEXT_LIMIT,
                            ),
                            "metadata": self._bounded_context_value(metadata),
                            "result_row_count": result_row_count,
                            "context_role": (
                                "retrieval_structured_entity"
                                if retrieval_match
                                else "structured_entity"
                            ),
                            "context_reason": (
                                "Matched backend structured-entity retrieval for "
                                "the current request."
                                if retrieval_match
                                else "Selected structured extraction entity for "
                                "the current request."
                            ),
                            "selection_score": selection_score,
                            "selection_features": selection_features,
                            "intelligence_layer": "structured_evidence",
                        },
                    )
                )

        ranked = sorted(
            candidates,
            key=lambda item: (-item[0], item[1], item[2], item[3]),
        )
        return [entity for *_ranking, entity in ranked[:limit]]

    def _result_evidence_for_context(
        self,
        *,
        collection_id: str,
        task_type: str,
        message: str,
        papers: list[dict[str, Any]],
        limit: int,
        retrieval_candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        paper_ids = [
            str(paper["paper_id"])
            for paper in papers
            if isinstance(paper.get("paper_id"), str)
        ]
        if not paper_ids:
            return []

        paper_rank = {paper_id: index + 1 for index, paper_id in enumerate(paper_ids)}
        paper_count = max(len(paper_rank), 1)
        query_terms = self._chunk_terms(message)
        task_keywords = TASK_CHUNK_KEYWORDS.get(
            task_type,
            TASK_CHUNK_KEYWORDS["literature_review"],
        )
        backend_rank_by_result_row_id = self._result_row_backend_rank_by_id(
            retrieval_candidates
        )
        rows = self.session.execute(
            select(ResultRow, Dataset, Method, Metric, Paper)
            .join(CollectionPaper, CollectionPaper.paper_id == ResultRow.paper_id)
            .join(Paper, Paper.id == ResultRow.paper_id)
            .outerjoin(Dataset, Dataset.id == ResultRow.dataset_id)
            .outerjoin(Method, Method.id == ResultRow.method_id)
            .outerjoin(Metric, Metric.id == ResultRow.metric_id)
            .where(
                CollectionPaper.collection_id == collection_id,
                ResultRow.paper_id.in_(paper_ids),
            )
            .order_by(
                CollectionPaper.position.asc(),
                ResultRow.value_numeric.desc().nullslast(),
                ResultRow.created_at.asc(),
                ResultRow.id.asc(),
            )
        ).all()

        candidates: list[tuple[float, int, int, dict[str, Any]]] = []
        for row_index, (result, dataset, method, metric, paper) in enumerate(rows):
            dataset_name = dataset.display_name if dataset is not None else None
            method_name = method.display_name if method is not None else None
            metric_name = metric.display_name if metric is not None else None
            row_terms = self._chunk_terms(
                " ".join(
                    [
                        str(paper.canonical_title or ""),
                        str(dataset_name or ""),
                        str(method_name or ""),
                        str(metric_name or ""),
                        str(result.split_name or ""),
                        str(result.value_text or ""),
                        str(result.comparator_text or ""),
                        str(result.notes or ""),
                    ]
                )
            )
            query_overlap = len(query_terms & row_terms)
            task_keyword_hits = len(task_keywords & row_terms)
            backend_rank = backend_rank_by_result_row_id.get(result.id)
            retrieval_match = backend_rank is not None
            row_paper_rank = paper_rank.get(result.paper_id, paper_count + 1)
            paper_rank_score = max(
                0.0,
                1.0 - ((row_paper_rank - 1) / paper_count),
            )
            has_complete_join = bool(dataset_name and method_name and metric_name)
            retrieval_score = (
                50.0 + (1.0 / backend_rank)
                if backend_rank is not None and backend_rank > 0
                else 0.0
            )
            selection_score = round(
                (query_overlap * 3.0)
                + (task_keyword_hits * 1.5)
                + (1.0 if has_complete_join else 0.0)
                + paper_rank_score
                + retrieval_score,
                4,
            )
            selection_features: dict[str, Any] = {
                "query_overlap": query_overlap,
                "task_keyword_hits": task_keyword_hits,
                "has_complete_entity_join": has_complete_join,
                "paper_rank": row_paper_rank,
                "retrieval_match": retrieval_match,
            }
            if backend_rank is not None:
                selection_features["backend_rank"] = backend_rank
            candidates.append(
                (
                    selection_score,
                    row_paper_rank,
                    row_index,
                    {
                        "result_row_id": result.id,
                        "paper_id": result.paper_id,
                        "paper_title": self._bounded_text(
                            paper.canonical_title,
                            limit=CONTEXT_SHORT_TEXT_LIMIT,
                        ),
                        "dataset_id": result.dataset_id,
                        "dataset": self._bounded_optional_text(
                            dataset_name,
                            limit=CONTEXT_SHORT_TEXT_LIMIT,
                        ),
                        "method_id": result.method_id,
                        "method": self._bounded_optional_text(
                            method_name,
                            limit=CONTEXT_SHORT_TEXT_LIMIT,
                        ),
                        "metric_id": result.metric_id,
                        "metric": self._bounded_optional_text(
                            metric_name,
                            limit=CONTEXT_SHORT_TEXT_LIMIT,
                        ),
                        "split_name": self._bounded_optional_text(
                            result.split_name,
                            limit=CONTEXT_SHORT_TEXT_LIMIT,
                        ),
                        "value_numeric": result.value_numeric,
                        "value_text": self._bounded_optional_text(
                            result.value_text,
                            limit=CONTEXT_SHORT_TEXT_LIMIT,
                        ),
                        "comparator_text": self._bounded_optional_text(
                            result.comparator_text,
                            limit=CONTEXT_SHORT_TEXT_LIMIT,
                        ),
                        "notes": self._bounded_optional_text(
                            result.notes,
                            limit=CONTEXT_TEXT_LIMIT,
                        ),
                        "context_role": (
                            "retrieval_result_evidence"
                            if retrieval_match
                            else "result_evidence"
                        ),
                        "context_reason": (
                            "Matched backend result-row retrieval for the current request."
                            if retrieval_match
                            else "Selected joined structured result evidence for the "
                            "current request."
                        ),
                        "selection_score": selection_score,
                        "selection_features": selection_features,
                        "intelligence_layer": "structured_evidence",
                    },
                )
            )

        ranked = sorted(candidates, key=lambda item: (-item[0], item[1], item[2]))
        return [result for *_ranking, result in ranked[:limit]]

    def _context_materialization(
        self,
        *,
        papers: list[dict[str, Any]],
        structured_entities: list[dict[str, Any]],
        result_evidence: list[dict[str, Any]],
        evidence_spans: list[dict[str, Any]],
    ) -> dict[str, Any]:
        method_items: dict[str, dict[str, Any]] = {}
        dataset_items: dict[str, dict[str, Any]] = {}
        metric_items: dict[str, dict[str, Any]] = {}
        baseline_items: dict[str, dict[str, Any]] = {}
        benchmark_items: dict[str, dict[str, Any]] = {}
        limitation_items: dict[str, dict[str, Any]] = {}
        claim_items: dict[str, dict[str, Any]] = {}

        def item_for(
            items: dict[str, dict[str, Any]],
            raw_label: Any,
        ) -> dict[str, Any] | None:
            label = self._bounded_optional_text(
                raw_label,
                limit=CONTEXT_SHORT_TEXT_LIMIT,
            )
            if label is None:
                return None
            key = self._normalize_label(label)
            if key not in items:
                items[key] = {
                    "label": label,
                    "paper_ids": set(),
                    "structured_entity_ids": set(),
                    "result_row_ids": set(),
                    "limitation_ids": set(),
                    "claim_ids": set(),
                    "evidence_span_ids": set(),
                    "method_names": set(),
                    "dataset_names": set(),
                    "metric_names": set(),
                    "baseline_names": set(),
                }
            return items[key]

        for entity in structured_entities:
            if not isinstance(entity, dict):
                continue
            entity_type = entity.get("entity_type")
            items_by_type = {
                "method": method_items,
                "dataset": dataset_items,
                "metric": metric_items,
            }.get(entity_type)
            if items_by_type is None:
                continue
            item = item_for(
                items_by_type,
                entity.get("display_name") or entity.get("normalized_name"),
            )
            if item is None:
                continue
            self._set_add_string(item["paper_ids"], entity.get("paper_id"))
            self._set_add_string(item["structured_entity_ids"], entity.get("entity_id"))

        for result in result_evidence:
            if not isinstance(result, dict):
                continue
            method_name = result.get("method")
            dataset_name = result.get("dataset")
            metric_name = result.get("metric")
            result_row_id = result.get("result_row_id")
            paper_id = result.get("paper_id")

            method_item = item_for(method_items, method_name)
            if method_item is not None:
                self._set_add_string(method_item["paper_ids"], paper_id)
                self._set_add_string(method_item["result_row_ids"], result_row_id)
                self._set_add_string(method_item["dataset_names"], dataset_name)
                self._set_add_string(method_item["metric_names"], metric_name)

            dataset_item = item_for(dataset_items, dataset_name)
            if dataset_item is not None:
                self._set_add_string(dataset_item["paper_ids"], paper_id)
                self._set_add_string(dataset_item["result_row_ids"], result_row_id)
                self._set_add_string(dataset_item["method_names"], method_name)
                self._set_add_string(dataset_item["metric_names"], metric_name)

            metric_item = item_for(metric_items, metric_name)
            if metric_item is not None:
                self._set_add_string(metric_item["paper_ids"], paper_id)
                self._set_add_string(metric_item["result_row_ids"], result_row_id)
                self._set_add_string(metric_item["method_names"], method_name)
                self._set_add_string(metric_item["dataset_names"], dataset_name)

            baseline_item = item_for(baseline_items, result.get("comparator_text"))
            if baseline_item is not None:
                self._set_add_string(baseline_item["paper_ids"], paper_id)
                self._set_add_string(baseline_item["result_row_ids"], result_row_id)
                self._set_add_string(baseline_item["method_names"], method_name)
                self._set_add_string(baseline_item["dataset_names"], dataset_name)
                self._set_add_string(baseline_item["metric_names"], metric_name)

            benchmark_item = item_for(
                benchmark_items,
                self._benchmark_table_label(result),
            )
            if benchmark_item is not None:
                self._set_add_string(benchmark_item["paper_ids"], paper_id)
                self._set_add_string(benchmark_item["result_row_ids"], result_row_id)
                self._set_add_string(benchmark_item["method_names"], method_name)
                self._set_add_string(benchmark_item["dataset_names"], dataset_name)
                self._set_add_string(benchmark_item["metric_names"], metric_name)
                self._set_add_string(
                    benchmark_item["baseline_names"],
                    result.get("comparator_text"),
                )

            claim_item = item_for(claim_items, "Structured result evidence")
            if claim_item is not None:
                self._set_add_string(claim_item["paper_ids"], paper_id)
                self._set_add_string(claim_item["claim_ids"], result_row_id)
                self._set_add_string(claim_item["result_row_ids"], result_row_id)
                self._set_add_string(claim_item["method_names"], method_name)
                self._set_add_string(claim_item["dataset_names"], dataset_name)
                self._set_add_string(claim_item["metric_names"], metric_name)
                self._set_add_string(
                    claim_item["baseline_names"],
                    result.get("comparator_text"),
                )

        finding_count = 0
        for paper in papers:
            if not isinstance(paper, dict):
                continue
            paper_id = paper.get("paper_id")
            raw_findings = paper.get("findings")
            if not isinstance(raw_findings, list):
                continue
            if not raw_findings:
                continue
            finding_count += len(raw_findings)
            finding_item = item_for(claim_items, "Finding evidence")
            if finding_item is None:
                continue
            self._set_add_string(finding_item["paper_ids"], paper_id)
            if isinstance(paper_id, str) and paper_id:
                for index, _finding in enumerate(raw_findings):
                    finding_item["claim_ids"].add(f"{paper_id}:finding:{index}")

        limitation_count = sum(
            len(paper.get("limitations") or [])
            for paper in papers
            if isinstance(paper.get("limitations"), list)
        )
        for paper in papers:
            if not isinstance(paper, dict):
                continue
            paper_id = paper.get("paper_id")
            raw_limitations = paper.get("limitations")
            if not isinstance(raw_limitations, list):
                continue
            if not raw_limitations:
                continue
            claim_item = item_for(claim_items, "Limitation evidence")
            for index, limitation in enumerate(raw_limitations):
                category = self._limitation_category(limitation)
                limitation_item = item_for(limitation_items, category)
                if limitation_item is None:
                    continue
                self._set_add_string(limitation_item["paper_ids"], paper_id)
                if isinstance(paper_id, str) and paper_id:
                    limitation_id = f"{paper_id}:{index}"
                    limitation_item["limitation_ids"].add(limitation_id)
                    if claim_item is not None:
                        self._set_add_string(claim_item["paper_ids"], paper_id)
                        claim_item["claim_ids"].add(f"{paper_id}:limitation:{index}")
                        claim_item["limitation_ids"].add(limitation_id)

        for span in evidence_spans:
            if not isinstance(span, dict):
                continue
            target_type = str(span.get("target_type") or "").casefold()
            label = {
                "finding": "Finding evidence",
                "limitation": "Limitation evidence",
                "result": "Structured result evidence",
                "result_row": "Structured result evidence",
            }.get(target_type)
            if label is None:
                continue
            claim_item = item_for(claim_items, label)
            if claim_item is None:
                continue
            self._set_add_string(claim_item["paper_ids"], span.get("paper_id"))
            self._set_add_string(
                claim_item["evidence_span_ids"],
                span.get("evidence_span_id"),
            )

        claim_count = sum(len(item["claim_ids"]) for item in claim_items.values())
        claim_evidence_span_ids = {
            span_id
            for item in claim_items.values()
            for span_id in item["evidence_span_ids"]
        }

        return {
            "version": CONTEXT_MATERIALIZATION_VERSION,
            "summary": {
                "paper_count": len(papers),
                "structured_entity_count": len(structured_entities),
                "result_evidence_count": len(result_evidence),
                "method_count": len(method_items),
                "dataset_count": len(dataset_items),
                "metric_count": len(metric_items),
                "baseline_count": len(baseline_items),
                "limitation_count": limitation_count,
                "benchmark_table_count": len(benchmark_items),
                "limitation_category_count": len(limitation_items),
                "finding_count": finding_count,
                "claim_count": claim_count,
                "claim_evidence_span_count": len(claim_evidence_span_ids),
                "claim_evidence_map_count": len(claim_items),
            },
            "method_map": self._materialization_items(
                method_items,
                cross_count_fields=("dataset_count", "metric_count"),
            ),
            "dataset_map": self._materialization_items(
                dataset_items,
                cross_count_fields=("method_count", "metric_count"),
            ),
            "metric_map": self._materialization_items(
                metric_items,
                cross_count_fields=("method_count", "dataset_count"),
            ),
            "baseline_map": self._materialization_items(
                baseline_items,
                cross_count_fields=("method_count", "dataset_count", "metric_count"),
            ),
            "benchmark_table": self._materialization_items(
                benchmark_items,
                cross_count_fields=(
                    "method_count",
                    "dataset_count",
                    "metric_count",
                    "baseline_count",
                ),
            ),
            "limitation_inventory": self._materialization_items(
                limitation_items,
                cross_count_fields=(),
            ),
            "claim_evidence_map": self._materialization_items(
                claim_items,
                cross_count_fields=(
                    "method_count",
                    "dataset_count",
                    "metric_count",
                    "baseline_count",
                ),
            ),
        }

    def _materialization_items(
        self,
        items: dict[str, dict[str, Any]],
        *,
        cross_count_fields: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        materialized: list[dict[str, Any]] = []
        for item in items.values():
            result: dict[str, Any] = {
                "label": item["label"],
                "paper_count": len(item["paper_ids"]),
            }
            structured_entity_count = len(item["structured_entity_ids"])
            result_evidence_count = len(item["result_row_ids"])
            limitation_count = len(item["limitation_ids"])
            claim_count = len(item["claim_ids"])
            evidence_span_count = len(item["evidence_span_ids"])
            if structured_entity_count > 0:
                result["structured_entity_count"] = structured_entity_count
            if result_evidence_count > 0:
                result["result_evidence_count"] = result_evidence_count
            if limitation_count > 0:
                result["limitation_count"] = limitation_count
            if claim_count > 0:
                result["claim_count"] = claim_count
            if evidence_span_count > 0:
                result["evidence_span_count"] = evidence_span_count
            cross_sets = {
                "method_count": item["method_names"],
                "dataset_count": item["dataset_names"],
                "metric_count": item["metric_names"],
                "baseline_count": item["baseline_names"],
            }
            for field_name in cross_count_fields:
                cross_count = len(cross_sets[field_name])
                if cross_count > 0:
                    result[field_name] = cross_count
            materialized.append(result)
        materialized.sort(
            key=lambda item: (
                -int(item.get("result_evidence_count") or 0),
                -int(item.get("limitation_count") or 0),
                -int(item.get("evidence_span_count") or 0),
                -int(item.get("claim_count") or 0),
                -int(item.get("structured_entity_count") or 0),
                item["label"].casefold(),
            )
        )
        return materialized[:CONTEXT_MATERIALIZATION_ITEM_LIMIT]

    def _benchmark_table_label(self, result: dict[str, Any]) -> str | None:
        parts = [
            self._bounded_optional_text(result.get("dataset"), limit=CONTEXT_SHORT_TEXT_LIMIT),
            self._bounded_optional_text(result.get("method"), limit=CONTEXT_SHORT_TEXT_LIMIT),
            self._bounded_optional_text(result.get("metric"), limit=CONTEXT_SHORT_TEXT_LIMIT),
            self._bounded_optional_text(
                result.get("comparator_text"),
                limit=CONTEXT_SHORT_TEXT_LIMIT,
            ),
        ]
        label_parts = [part for part in parts if part]
        if len(label_parts) < 2:
            return None
        return self._bounded_text(" - ".join(label_parts), limit=CONTEXT_SHORT_TEXT_LIMIT)

    @staticmethod
    def _limitation_category(statement: Any) -> str:
        if not isinstance(statement, str):
            return "Other limitations"
        lowered = statement.casefold()
        category_keywords = (
            ("Data coverage", ("cohort", "corpus", "data", "dataset", "sample", "split")),
            (
                "Compute and scale",
                ("budget", "comput", "memory", "runtime", "scalab", "scale"),
            ),
            ("Baseline coverage", ("baseline", "comparator", "comparison")),
            (
                "Evaluation coverage",
                ("ablation", "evaluat", "metric", "validat"),
            ),
            (
                "Generalization",
                ("cross-domain", "external", "generaliz", "out-of-distribution"),
            ),
            ("Reproducibility", ("code", "implement", "reproduc")),
        )
        for category, keywords in category_keywords:
            if any(keyword in lowered for keyword in keywords):
                return category
        return "Other limitations"

    def _set_add_string(self, values: set[str], value: Any) -> None:
        if isinstance(value, str) and value:
            values.add(value)

    def _result_count_by_entity(
        self,
        paper_ids: list[str],
    ) -> dict[tuple[str, str], int]:
        result_count_by_entity: dict[tuple[str, str], int] = {}
        rows = self.session.execute(
            select(
                ResultRow.dataset_id.label("dataset_id"),
                ResultRow.method_id.label("method_id"),
                ResultRow.metric_id.label("metric_id"),
            )
            .where(ResultRow.paper_id.in_(paper_ids))
        ).all()
        for row in rows:
            for entity_type, entity_id in (
                ("dataset", row.dataset_id),
                ("method", row.method_id),
                ("metric", row.metric_id),
            ):
                if not isinstance(entity_id, str) or not entity_id:
                    continue
                key = (entity_type, entity_id)
                result_count_by_entity[key] = (
                    result_count_by_entity.get(key, 0) + 1
                )
        return result_count_by_entity

    def _paper_summaries(
        self,
        paper_ids: list[str],
        *,
        role_by_paper_id: dict[str, str],
        extraction_quality_by_paper_id: dict[str, PaperExtractionQuality],
    ) -> list[dict[str, Any]]:
        papers = {
            paper.id: paper
            for paper in self.session.execute(
                select(Paper).where(Paper.id.in_(paper_ids))
            )
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
                        "metadata": self._bounded_context_value(
                            dict(item.metadata_json or {})
                        ),
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
            paper_quality = extraction_quality_by_paper_id.get(paper_id)
            if paper_quality is not None:
                summary["extraction_quality"] = self._paper_extraction_quality_context(
                    paper_quality
                )
            summaries.append(summary)
        return summaries

    def _context_reason(self, role: str) -> str:
        return {
            "selected": "The user selected this paper for the research thread.",
            "pinned_context": "The paper is pinned in the active Study.",
            "retrieval_match": "The paper matched backend chunk retrieval for the current request.",
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

    def _collection_extraction_quality(
        self,
        collection_id: str,
    ) -> CollectionExtractionQuality:
        collection = self.session.get(Collection, collection_id)
        extraction_profile_id = collection.extraction_profile_id if collection else None
        schema_payload = None
        if extraction_profile_id is not None:
            profile = self.session.get(ExtractionProfile, extraction_profile_id)
            if profile is not None:
                schema_payload = dict(profile.schema_payload or {})
        return build_collection_extraction_quality(
            self.session,
            collection_id=collection_id,
            collection_extraction_profile_id=extraction_profile_id,
            collection_schema_payload=schema_payload,
        )

    def _paper_extraction_quality_context(
        self,
        quality: PaperExtractionQuality,
    ) -> dict[str, Any]:
        return {
            "freshness_status": quality.freshness_status,
            "stale_reasons": list(quality.stale_reasons),
            "structured_entity_count": quality.structured_entity_count,
            "entity_counts": dict(quality.entity_counts),
            "evidence_span_count": quality.evidence_span_count,
            "anchored_evidence_span_count": quality.anchored_evidence_span_count,
            "unresolved_evidence_span_count": quality.unresolved_evidence_span_count,
            "evidence_span_anchor_diagnostics": dict(
                quality.evidence_span_anchor_diagnostics
            ),
            "missing_structured_evidence": list(quality.missing_structured_evidence),
        }

    def _extraction_quality_context(
        self,
        quality: CollectionExtractionQuality,
        *,
        papers: list[dict[str, Any]],
    ) -> dict[str, Any]:
        quality_by_paper_id = {
            paper_quality.paper_id: paper_quality
            for paper_quality in quality.papers
        }
        selected_quality = [
            quality_by_paper_id[paper["paper_id"]]
            for paper in papers
            if isinstance(paper.get("paper_id"), str)
            and paper["paper_id"] in quality_by_paper_id
        ]
        return {
            "collection": {
                "paper_count": quality.paper_count,
                "papers_with_completed_extraction_count": (
                    quality.papers_with_completed_extraction_count
                ),
                "fresh_paper_count": quality.fresh_paper_count,
                "stale_paper_count": quality.stale_paper_count,
                "missing_extraction_paper_count": (
                    quality.missing_extraction_paper_count
                ),
                "total_structured_entity_count": (
                    quality.total_structured_entity_count
                ),
                "total_evidence_span_count": quality.total_evidence_span_count,
                "anchored_evidence_span_count": quality.anchored_evidence_span_count,
                "unresolved_evidence_span_count": (
                    quality.unresolved_evidence_span_count
                ),
                "missing_structured_evidence": list(quality.missing_structured_evidence),
            },
            "selected_papers": self._extraction_quality_summary(selected_quality),
        }

    def _extraction_quality_summary(
        self,
        paper_qualities: list[PaperExtractionQuality],
    ) -> dict[str, Any]:
        entity_categories = (
            list(paper_qualities[0].entity_counts.keys()) if paper_qualities else []
        )
        entity_totals = {
            category: sum(
                paper_quality.entity_counts.get(category, 0)
                for paper_quality in paper_qualities
            )
            for category in entity_categories
        }
        return {
            "paper_count": len(paper_qualities),
            "fresh_paper_count": sum(
                1
                for quality in paper_qualities
                if quality.freshness_status == "fresh"
            ),
            "stale_paper_count": sum(
                1
                for quality in paper_qualities
                if quality.freshness_status == "stale"
            ),
            "missing_extraction_paper_count": sum(
                1
                for quality in paper_qualities
                if quality.freshness_status == "missing_extraction"
            ),
            "total_structured_entity_count": sum(
                quality.structured_entity_count for quality in paper_qualities
            ),
            "total_evidence_span_count": sum(
                quality.evidence_span_count for quality in paper_qualities
            ),
            "anchored_evidence_span_count": sum(
                quality.anchored_evidence_span_count for quality in paper_qualities
            ),
            "unresolved_evidence_span_count": sum(
                quality.unresolved_evidence_span_count for quality in paper_qualities
            ),
            "stale_reasons": self._ordered_unique(
                reason
                for quality in paper_qualities
                if quality.freshness_status == "stale"
                for reason in quality.stale_reasons
            ),
            "missing_structured_evidence": [
                category for category, total in entity_totals.items() if total == 0
            ],
        }

    def _intelligence_layers(
        self,
        *,
        collection_id: str,
        task_type: str,
        workspace_id: str | None,
        source_ids: list[str],
    ) -> dict[str, Any]:
        self._ensure_research_intelligence(
            collection_id=collection_id,
            workspace_id=workspace_id,
        )
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
        source_fact_records = self._source_fact_memory_records(
            collection_id=collection_id,
            workspace_id=workspace_id,
            source_ids=source_ids,
            limit=SOURCE_FACT_MEMORY_LIMIT,
        )
        nodes = self._graph_nodes(
            repository=repository,
            collection_id=collection_id,
            workspace_id=workspace_id,
            source_ids=source_ids,
            limit=FIELD_GRAPH_NODE_LIMIT,
        )
        edges = self._graph_edges(
            repository=repository,
            collection_id=collection_id,
            workspace_id=workspace_id,
            source_ids=source_ids,
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
            "source_fact_memory": [
                self._serialize_memory_record(
                    record,
                    intelligence_layer="source_fact_memory",
                )
                for record in source_fact_records
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

    def _ensure_research_intelligence(
        self,
        *,
        collection_id: str,
        workspace_id: str | None,
    ) -> None:
        builder = ResearchIntelligenceMemoryBuilder(self.session)
        if not builder.is_current(collection_id):
            builder.build(collection_id)
        if workspace_id is not None and not builder.is_current(
            collection_id,
            workspace_id=workspace_id,
        ):
            builder.build(collection_id, workspace_id=workspace_id)

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

    def _source_fact_memory_records(
        self,
        *,
        collection_id: str,
        workspace_id: str | None,
        source_ids: list[str],
        limit: int,
    ) -> list[ResearchMemoryRecord]:
        if workspace_id is None or not source_ids:
            return []
        selected_source_ids = set(source_ids)
        candidate_records = self._selected_source_fact_memory_records_query(
            collection_id=collection_id,
            workspace_id=workspace_id,
            source_ids=source_ids,
        )
        selected_records = [
            record
            for record in candidate_records
            if self._memory_record_source_id(record) in selected_source_ids
        ]
        return self._rank_memory_records(
            selected_records,
            workspace_id=workspace_id,
            limit=limit,
        )

    def _memory_record_source_id(self, record: ResearchMemoryRecord) -> str | None:
        source_id = dict(record.payload_json or {}).get("source_id")
        return source_id if isinstance(source_id, str) and source_id else None

    def _selected_source_fact_memory_records_query(
        self,
        *,
        collection_id: str,
        workspace_id: str,
        source_ids: list[str],
    ) -> list[ResearchMemoryRecord]:
        prefixes = [
            self._source_fact_stable_prefix(
                collection_id=collection_id,
                workspace_id=workspace_id,
                source_id=source_id,
            )
            for source_id in self._unique_strings(source_ids)
        ]
        if not prefixes:
            return []
        statement = (
            select(ResearchMemoryRecord)
            .where(
                ResearchMemoryRecord.collection_id == collection_id,
                ResearchMemoryRecord.workspace_id == workspace_id,
                ResearchMemoryRecord.memory_type == "source_fact",
                or_(
                    *(
                        ResearchMemoryRecord.version_key.startswith(prefix)
                        for prefix in prefixes
                    )
                ),
            )
            .order_by(
                ResearchMemoryRecord.title.asc(),
                ResearchMemoryRecord.version_key.asc(),
                ResearchMemoryRecord.id.asc(),
            )
        )
        return list(self.session.execute(statement).scalars())

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
        source_ids: list[str],
        limit: int,
    ) -> list[ResearchGraphNode]:
        candidate_limit = self._candidate_pool_limit(limit)
        if workspace_id is None:
            nodes = list(
                repository.list_graph_nodes(
                    collection_id=collection_id,
                    limit=candidate_limit,
                )
            )
            return self._filter_source_fact_graph_nodes(nodes, source_ids=[])

        workspace_nodes = list(
            repository.list_graph_nodes(
                collection_id=collection_id,
                workspace_id=workspace_id,
                limit=candidate_limit,
            )
        )
        selected_source_fact_nodes = self._selected_source_fact_graph_nodes_query(
            collection_id=collection_id,
            workspace_id=workspace_id,
            source_ids=source_ids,
        )
        global_nodes = list(
            repository.list_graph_nodes(collection_id=collection_id, limit=candidate_limit)
        )
        nodes = self._dedupe_graph_nodes(
            [*selected_source_fact_nodes, *workspace_nodes, *global_nodes]
        )
        return self._filter_source_fact_graph_nodes(nodes, source_ids=source_ids)

    def _selected_source_fact_graph_nodes_query(
        self,
        *,
        collection_id: str,
        workspace_id: str,
        source_ids: list[str],
    ) -> list[ResearchGraphNode]:
        prefixes = [
            self._source_fact_stable_prefix(
                collection_id=collection_id,
                workspace_id=workspace_id,
                source_id=source_id,
            )
            for source_id in self._unique_strings(source_ids)
        ]
        if not prefixes:
            return []
        statement = (
            select(ResearchGraphNode)
            .where(
                ResearchGraphNode.collection_id == collection_id,
                ResearchGraphNode.workspace_id == workspace_id,
                ResearchGraphNode.node_type.in_(SOURCE_FACT_NODE_TYPES),
                or_(
                    *(
                        ResearchGraphNode.stable_key.startswith(prefix)
                        for prefix in prefixes
                    )
                ),
            )
            .order_by(
                ResearchGraphNode.node_type.asc(),
                ResearchGraphNode.label.asc(),
                ResearchGraphNode.stable_key.asc(),
            )
        )
        return list(self.session.execute(statement).scalars())

    def _filter_source_fact_graph_nodes(
        self,
        nodes: list[ResearchGraphNode],
        *,
        source_ids: list[str],
    ) -> list[ResearchGraphNode]:
        selected_source_ids = set(source_ids)
        filtered: list[ResearchGraphNode] = []
        for node in nodes:
            if not self._is_source_fact_graph_node(node):
                filtered.append(node)
                continue
            if selected_source_ids and self._graph_node_source_id(node) in selected_source_ids:
                filtered.append(node)
        return filtered

    def _is_source_fact_graph_node(self, node: ResearchGraphNode) -> bool:
        if node.node_type in SOURCE_FACT_NODE_TYPES:
            return dict(node.payload_json or {}).get("origin") == "study_source"
        return False

    def _graph_node_source_id(self, node: ResearchGraphNode) -> str | None:
        source_id = dict(node.payload_json or {}).get("source_id")
        return source_id if isinstance(source_id, str) and source_id else None


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
        source_ids: list[str],
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
        selected_source_fact_edges = self._selected_source_fact_graph_edges_query(
            collection_id=collection_id,
            workspace_id=workspace_id,
            source_ids=source_ids,
        )
        workspace_edges = self._filter_source_fact_graph_edges(
            [*selected_source_fact_edges, *workspace_edges],
            source_ids=source_ids,
        )
        global_edges = list(
            repository.list_graph_edges(collection_id=collection_id, limit=candidate_limit)
        )
        return self._dedupe_graph_edges([*workspace_edges, *global_edges])

    def _selected_source_fact_graph_edges_query(
        self,
        *,
        collection_id: str,
        workspace_id: str,
        source_ids: list[str],
    ) -> list[ResearchGraphEdge]:
        selected_nodes = self._selected_source_fact_graph_nodes_query(
            collection_id=collection_id,
            workspace_id=workspace_id,
            source_ids=source_ids,
        )
        selected_node_ids = [node.id for node in selected_nodes]
        if not selected_node_ids:
            return []
        statement = (
            select(ResearchGraphEdge)
            .where(
                ResearchGraphEdge.collection_id == collection_id,
                ResearchGraphEdge.workspace_id == workspace_id,
                or_(
                    ResearchGraphEdge.source_node_id.in_(selected_node_ids),
                    ResearchGraphEdge.target_node_id.in_(selected_node_ids),
                ),
            )
            .order_by(
                ResearchGraphEdge.edge_type.asc(),
                ResearchGraphEdge.source_node_id.asc(),
                ResearchGraphEdge.target_node_id.asc(),
            )
        )
        return list(self.session.execute(statement).scalars())

    def _filter_source_fact_graph_edges(
        self,
        edges: list[ResearchGraphEdge],
        *,
        source_ids: list[str],
    ) -> list[ResearchGraphEdge]:
        selected_source_ids = set(source_ids)
        filtered: list[ResearchGraphEdge] = []
        for edge in edges:
            if not self._is_source_fact_graph_edge(edge):
                filtered.append(edge)
                continue
            if selected_source_ids and self._graph_edge_source_id(edge) in selected_source_ids:
                filtered.append(edge)
        return filtered

    def _is_source_fact_graph_edge(self, edge: ResearchGraphEdge) -> bool:
        return dict(edge.payload_json or {}).get("origin") == "source_fact_evidence_link"

    def _graph_edge_source_id(self, edge: ResearchGraphEdge) -> str | None:
        source_id = dict(edge.payload_json or {}).get("source_id")
        return source_id if isinstance(source_id, str) and source_id else None

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
        if record.memory_type == "source_fact":
            return {
                key: self._bounded_context_value(payload.get(key))
                for key in (
                    "origin",
                    "fact_id",
                    "fact_type",
                    "fact_text",
                    "source_id",
                    "source_type",
                    "source_title",
                    "source_locator",
                    "read_status",
                    "extraction_rule",
                )
                if payload.get(key) is not None
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
            "source_fact_memory": (
                "Included as typed user-source facts from selected Study artifacts."
            ),
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
                    self._bounded_study_brief_value(item)
                    for item in value[:STUDY_BRIEF_LIST_LIMIT]
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
                self._bounded_study_brief_value(item)
                for item in value[:STUDY_BRIEF_LIST_LIMIT]
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
            is_stale = self._study_source_is_stale(source)
            read_status = (
                "stale"
                if is_stale and source.read_status == "ready"
                else source.read_status
            )
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
                    "read_status": read_status,
                    "stale": is_stale,
                    "error_message": self._bounded_optional_text(
                        source.error_message,
                        limit=CONTEXT_TEXT_LIMIT,
                    ),
                }
            )
        return summaries

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
    ) -> list[str]:
        warnings: list[str] = []
        if not papers:
            warnings.append("No paper evidence was available.")
        if papers and not any(paper.get("sections") for paper in papers):
            warnings.append("Selected papers are not parsed, so full-text evidence is missing.")
        has_structured_signal = any(self._structured_signal_count(paper) for paper in papers)
        if papers and not has_structured_signal:
            warnings.append("Structured extraction is missing for the selected papers.")
        quality_items = [
            dict(paper.get("extraction_quality") or {})
            for paper in papers
            if isinstance(paper.get("extraction_quality"), dict)
        ]
        stale_quality_items = [
            item for item in quality_items if item.get("freshness_status") == "stale"
        ]
        if stale_quality_items:
            stale_reasons = self._ordered_unique(
                reason
                for item in stale_quality_items
                for reason in item.get("stale_reasons", [])
                if isinstance(reason, str)
            )
            reason_text = ", ".join(stale_reasons) if stale_reasons else "unknown reason"
            warnings.append(
                "Structured extraction is stale for "
                f"{len(stale_quality_items)} selected "
                f"{self._plural('paper', len(stale_quality_items))}: {reason_text}."
            )
        missing_extraction_count = sum(
            1
            for item in quality_items
            if item.get("freshness_status") == "missing_extraction"
            and int(item.get("structured_entity_count") or 0) == 0
        )
        if missing_extraction_count and has_structured_signal:
            warnings.append(
                "Structured extraction is missing for "
                f"{missing_extraction_count} selected "
                f"{self._plural('paper', missing_extraction_count)}."
            )
        unresolved_evidence_span_count = sum(
            int(item.get("unresolved_evidence_span_count") or 0)
            for item in quality_items
        )
        if unresolved_evidence_span_count:
            warnings.append(
                f"{unresolved_evidence_span_count} extracted evidence "
                f"{self._plural('span', unresolved_evidence_span_count)} "
                f"{self._is_or_are(unresolved_evidence_span_count)} unresolved; "
                "citation grounding may be weaker."
            )
        if source_ids and not sources:
            warnings.append("Requested study sources were not available in the context pack.")
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

    def _collection_cache_input(self, collection_id: str) -> dict[str, Any]:
        collection = self.session.get(Collection, collection_id)
        extraction_profile = (
            self.session.get(ExtractionProfile, collection.extraction_profile_id)
            if collection is not None and collection.extraction_profile_id is not None
            else None
        )
        paper_rows = self.session.execute(
            select(
                CollectionPaper.paper_id,
                CollectionPaper.position,
                CollectionPaper.updated_at.label("membership_updated_at"),
                Paper.updated_at.label("paper_updated_at"),
                Paper.canonical_title,
                Paper.abstract,
                Paper.publication_year,
                Paper.venue,
                Paper.doi,
                Paper.arxiv_id,
                Paper.raw_metadata,
            )
            .join(Paper, Paper.id == CollectionPaper.paper_id)
            .where(CollectionPaper.collection_id == collection_id)
            .order_by(CollectionPaper.position.asc(), CollectionPaper.created_at.asc())
        ).all()
        paper_ids = [row.paper_id for row in paper_rows]
        return {
            "collection": {
                "collection_id": collection_id,
                "missing": collection is None,
                "updated_at": (
                    self._datetime_signal(collection.updated_at)
                    if collection is not None
                    else None
                ),
                "extraction_profile_id": (
                    collection.extraction_profile_id
                    if collection is not None
                    else None
                ),
            },
            "extraction_profile": {
                "extraction_profile_id": (
                    collection.extraction_profile_id
                    if collection is not None
                    else None
                ),
                "missing": (
                    collection is not None
                    and collection.extraction_profile_id is not None
                    and extraction_profile is None
                ),
                "updated_at": (
                    self._datetime_signal(extraction_profile.updated_at)
                    if extraction_profile is not None
                    else None
                ),
                "content_digest": (
                    self._json_digest(
                        {
                            "name": extraction_profile.name,
                            "description": extraction_profile.description,
                            "scope_type": extraction_profile.scope_type,
                            "schema_payload": extraction_profile.schema_payload,
                            "active": extraction_profile.active,
                        }
                    )
                    if extraction_profile is not None
                    else None
                ),
            },
            "papers": [
                {
                    "paper_id": row.paper_id,
                    "position": row.position,
                    "membership_updated_at": self._datetime_signal(
                        row.membership_updated_at
                    ),
                    "paper_updated_at": self._datetime_signal(row.paper_updated_at),
                    "content_digest": self._json_digest(
                        {
                            "canonical_title": row.canonical_title,
                            "abstract": row.abstract,
                            "publication_year": row.publication_year,
                            "venue": row.venue,
                            "doi": row.doi,
                            "arxiv_id": row.arxiv_id,
                            "raw_metadata": row.raw_metadata,
                        }
                    ),
                }
                for row in paper_rows
            ],
            "paper_scoped_rows": self._paper_scoped_cache_inputs(paper_ids),
            "extraction_runs": self._extraction_run_cache_inputs(paper_ids),
        }

    def _paper_scoped_cache_inputs(self, paper_ids: list[str]) -> dict[str, Any]:
        if not paper_ids:
            return {}
        models = (
            ("sections", Section),
            ("chunks", Chunk),
            ("figures", Figure),
            ("tables", TableArtifact),
            ("datasets", Dataset),
            ("methods", Method),
            ("metrics", Metric),
            ("result_rows", ResultRow),
            ("findings", Finding),
            ("limitations", Limitation),
            ("engineering_tricks", EngineeringTrick),
            ("research_design_elements", ResearchDesignElement),
            ("evidence_spans", EvidenceSpan),
        )
        return {
            name: self._model_row_cache_inputs(model, paper_ids)
            for name, model in models
        }

    def _model_row_cache_inputs(self, model, paper_ids: list[str]) -> list[dict[str, Any]]:  # noqa: ANN001
        rows = self.session.execute(
            select(model.id, model.paper_id, model.updated_at)
            .where(model.paper_id.in_(paper_ids))
            .order_by(model.paper_id.asc(), model.id.asc())
        ).all()
        return [
            {
                "id": row.id,
                "paper_id": row.paper_id,
                "updated_at": self._datetime_signal(row.updated_at),
            }
            for row in rows
        ]

    def _extraction_run_cache_inputs(self, paper_ids: list[str]) -> list[dict[str, Any]]:
        if not paper_ids:
            return []
        rows = self.session.execute(
            select(ExtractionRun)
            .where(ExtractionRun.paper_id.in_(paper_ids))
            .order_by(ExtractionRun.paper_id.asc(), ExtractionRun.created_at.asc())
        ).scalars()
        return [
            {
                "id": run.id,
                "paper_id": run.paper_id,
                "extraction_profile_id": run.extraction_profile_id,
                "model_name": run.model_name,
                "prompt_version": run.prompt_version,
                "schema_version": run.schema_version,
                "status": run.status,
                "updated_at": self._datetime_signal(run.updated_at),
                "diagnostics_digest": self._json_digest(run.diagnostics_json or {}),
            }
            for run in rows
        ]

    def _study_brief_cache_input(
        self,
        workspace_id: str | None,
    ) -> dict[str, Any] | None:
        if workspace_id is None:
            return None
        brief = self.session.execute(
            select(StudyBrief).where(StudyBrief.workspace_id == workspace_id).limit(1)
        ).scalar_one_or_none()
        if brief is None:
            return {"workspace_id": workspace_id, "missing": True}
        return {
            "study_brief_id": brief.id,
            "workspace_id": brief.workspace_id,
            "version": brief.version,
            "updated_at": self._datetime_signal(brief.updated_at),
            "content_digest": self._json_digest(brief.brief_json or {}),
        }

    def _research_intelligence_cache_inputs(
        self,
        *,
        collection_id: str,
        workspace_id: str | None,
    ) -> dict[str, Any]:
        workspace_ids = [None] if workspace_id is None else [None, workspace_id]
        return {
            "memory_records": self._scoped_research_row_cache_inputs(
                ResearchMemoryRecord,
                collection_id=collection_id,
                workspace_ids=workspace_ids,
            ),
            "graph_nodes": self._scoped_research_row_cache_inputs(
                ResearchGraphNode,
                collection_id=collection_id,
                workspace_ids=workspace_ids,
            ),
            "graph_edges": self._scoped_research_row_cache_inputs(
                ResearchGraphEdge,
                collection_id=collection_id,
                workspace_ids=workspace_ids,
            ),
        }

    def _scoped_research_row_cache_inputs(
        self,
        model,  # noqa: ANN001
        *,
        collection_id: str,
        workspace_ids: list[str | None],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for scoped_workspace_id in workspace_ids:
            statement = select(model.id, model.workspace_id, model.updated_at).where(
                model.collection_id == collection_id
            )
            if scoped_workspace_id is None:
                statement = statement.where(model.workspace_id.is_(None))
            else:
                statement = statement.where(model.workspace_id == scoped_workspace_id)
            statement = statement.order_by(model.workspace_id.asc(), model.id.asc())
            rows.extend(
                {
                    "id": row.id,
                    "workspace_id": row.workspace_id,
                    "updated_at": self._datetime_signal(row.updated_at),
                }
                for row in self.session.execute(statement).all()
            )
        return rows

    def _component_cache_fingerprint(self, component: object | None) -> dict[str, Any]:
        if component is None:
            return {"available": False}
        fingerprint: dict[str, Any] = {
            "available": True,
            "class": (
                f"{component.__class__.__module__}."
                f"{component.__class__.__qualname__}"
            ),
        }
        for attr_name in (
            "context_cache_fingerprint",
            "cache_fingerprint",
            "fingerprint",
            "model_name",
            "model",
            "index_version",
        ):
            value = getattr(component, attr_name, None)
            if value is None:
                continue
            if callable(value):
                try:
                    value = value()
                except TypeError:
                    continue
            fingerprint[attr_name] = self._bounded_context_value(
                value,
                text_limit=CONTEXT_SHORT_TEXT_LIMIT,
            )
        return fingerprint

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
        retrieval: dict[str, Any],
        chunks: list[dict[str, Any]],
        evidence_spans: list[dict[str, Any]],
        figures: list[dict[str, Any]],
        tables: list[dict[str, Any]],
        structured_entities: list[dict[str, Any]],
        result_evidence: list[dict[str, Any]],
        context_materialization: dict[str, Any],
        intelligence_layers: dict[str, Any],
        extraction_quality: dict[str, Any],
        paper_extraction_quality: list[dict[str, Any]],
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
            "retrieval": retrieval,
            "chunks": [
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "paper_id": chunk.get("paper_id"),
                    "section_title": chunk.get("section_title"),
                    "score": chunk.get("selection_score"),
                }
                for chunk in chunks
            ],
            "evidence_spans": [
                {
                    "evidence_span_id": span.get("evidence_span_id"),
                    "paper_id": span.get("paper_id"),
                    "chunk_id": span.get("chunk_id"),
                    "section_id": span.get("section_id"),
                    "target_type": span.get("target_type"),
                    "target_id": span.get("target_id"),
                    "quote_text": span.get("quote_text"),
                    "score": span.get("selection_score"),
                }
                for span in evidence_spans
            ],
            "figures": [
                {
                    "figure_id": figure.get("figure_id"),
                    "paper_id": figure.get("paper_id"),
                    "figure_label": figure.get("figure_label"),
                    "caption": figure.get("caption"),
                    "page_number": figure.get("page_number"),
                    "has_asset": figure.get("has_asset"),
                    "score": figure.get("selection_score"),
                }
                for figure in figures
            ],
            "tables": [
                {
                    "table_id": table.get("table_id"),
                    "paper_id": table.get("paper_id"),
                    "table_label": table.get("table_label"),
                    "caption": table.get("caption"),
                    "page_number": table.get("page_number"),
                    "structured_payload": table.get("structured_payload"),
                    "has_asset": table.get("has_asset"),
                    "score": table.get("selection_score"),
                }
                for table in tables
            ],
            "structured_entities": [
                {
                    "entity_id": entity.get("entity_id"),
                    "entity_type": entity.get("entity_type"),
                    "paper_id": entity.get("paper_id"),
                    "display_name": entity.get("display_name"),
                    "normalized_name": entity.get("normalized_name"),
                    "metadata": entity.get("metadata"),
                    "result_row_count": entity.get("result_row_count"),
                    "score": entity.get("selection_score"),
                }
                for entity in structured_entities
            ],
            "result_evidence": [
                {
                    "result_row_id": result.get("result_row_id"),
                    "paper_id": result.get("paper_id"),
                    "dataset_id": result.get("dataset_id"),
                    "method_id": result.get("method_id"),
                    "metric_id": result.get("metric_id"),
                    "split_name": result.get("split_name"),
                    "value_numeric": result.get("value_numeric"),
                    "value_text": result.get("value_text"),
                    "comparator_text": result.get("comparator_text"),
                    "notes": result.get("notes"),
                    "score": result.get("selection_score"),
                }
                for result in result_evidence
            ],
            "context_materialization": context_materialization,
            "intelligence_layers": self._cache_layer_inputs(intelligence_layers),
            "extraction_quality": extraction_quality,
            "paper_extraction_quality": paper_extraction_quality,
        }
        digest = self._json_digest(digest_payload)
        return "|".join(
            [f"collection:{collection_id}", f"task:{task_type}", f"inputs:{digest}"]
        )

    def _paper_extraction_quality_cache_inputs(
        self,
        papers: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        cache_inputs: list[dict[str, Any]] = []
        for paper in papers:
            quality = paper.get("extraction_quality")
            if not isinstance(quality, dict):
                continue
            cache_inputs.append(
                {
                    "paper_id": paper.get("paper_id"),
                    "freshness_status": quality.get("freshness_status"),
                    "stale_reasons": list(quality.get("stale_reasons") or []),
                    "structured_entity_count": quality.get("structured_entity_count"),
                    "entity_counts": dict(quality.get("entity_counts") or {}),
                    "evidence_span_count": quality.get("evidence_span_count"),
                    "anchored_evidence_span_count": quality.get(
                        "anchored_evidence_span_count"
                    ),
                    "unresolved_evidence_span_count": quality.get(
                        "unresolved_evidence_span_count"
                    ),
                    "evidence_span_anchor_diagnostics": dict(
                        quality.get("evidence_span_anchor_diagnostics") or {}
                    ),
                    "missing_structured_evidence": list(
                        quality.get("missing_structured_evidence") or []
                    ),
                }
            )
        return cache_inputs

    def _json_digest(self, payload: Any) -> str:
        serialized = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:32]

    def _source_fact_stable_prefix(
        self,
        *,
        collection_id: str,
        workspace_id: str,
        source_id: str,
    ) -> str:
        return (
            self._stable_key(
                "collection",
                collection_id,
                "workspace",
                workspace_id,
                "source",
                source_id,
                "fact",
            )
            + ":"
        )

    def _stable_key(self, *parts: str) -> str:
        raw_key = ":".join(self._normalize_label(part) for part in parts if part)
        if len(raw_key) <= 240:
            return raw_key
        digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]
        return f"{raw_key[:223].rstrip(':')}:{digest}"

    def _normalize_label(self, value: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
        return normalized.strip("-") or "unknown"

    def _chunk_terms(self, text: str) -> set[str]:
        return {
            term
            for term in re.findall(r"[a-z0-9]+", text.lower())
            if len(term) >= 3 and term not in CHUNK_QUERY_STOPWORDS
        }

    def _unique_strings(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        cleaned: list[str] = []
        for value in values:
            if not isinstance(value, str):
                continue
            stripped = value.strip()
            if not stripped or stripped in seen:
                continue
            seen.add(stripped)
            cleaned.append(stripped)
        return cleaned

    def _ordered_unique(self, values) -> list[str]:  # noqa: ANN001
        seen: set[str] = set()
        ordered: list[str] = []
        for value in values:
            if not isinstance(value, str) or not value:
                continue
            if value in seen:
                continue
            seen.add(value)
            ordered.append(value)
        return ordered

    def _plural(self, singular: str, count: int) -> str:
        return singular if count == 1 else f"{singular}s"

    def _is_or_are(self, count: int) -> str:
        return "is" if count == 1 else "are"

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
            "source_fact_memory": [
                {
                    "id": item.get("memory_record_id"),
                    "version_key": item.get("version_key"),
                    "score": item.get("selection_score"),
                    "updated_at": item.get("updated_at"),
                    "content_digest": item.get("content_digest"),
                }
                for item in intelligence_layers.get("source_fact_memory", [])
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
            current_file_signal = self._study_source_file_signal(source)
            cache_inputs.append(
                {
                    "source_id": source.id,
                    "updated_at": self._datetime_signal(source.updated_at),
                    "current_file_signal": current_file_signal,
                    "content_digest": self._json_digest(
                        {
                            "source_type": source.source_type,
                            "title": source.title,
                            "summary": source.summary,
                            "content": source.content,
                            "read_status": source.read_status,
                            "error_message": source.error_message,
                            "has_path": bool(source.path),
                            "source_size_bytes": source.source_size_bytes,
                            "source_mtime_ns": source.source_mtime_ns,
                        }
                    ),
                }
            )
        return cache_inputs

    def _study_source_file_signal(self, source: StudySource) -> dict[str, Any] | None:
        if not source.path or source.source_size_bytes is None or source.source_mtime_ns is None:
            return None
        path = Path(source.path).expanduser()
        if not path.is_absolute():
            return None
        try:
            stat = path.stat()
        except OSError:
            return {"missing": True}
        return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    def _study_source_is_stale(self, source: StudySource) -> bool:
        signal = self._study_source_file_signal(source)
        if signal is None:
            return False
        return (
            signal.get("missing") is True
            or signal.get("size") != source.source_size_bytes
            or signal.get("mtime_ns") != source.source_mtime_ns
        )

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
        return [
            self._bounded_context_value(item)
            for item in value[:limit]
        ]

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
            "baseline": "baselines",
            "finding": "findings",
            "gap": "limitations",
            "study_constraint": "study_brief",
            "user_claim": "source_claims",
            "assumption": "source_claims",
            "contradiction": "source_claims",
            "extension": "source_claims",
            "project_constraint": "study_brief",
            "method_context": "methods",
            "dataset_context": "datasets",
            "metric_context": "metrics",
            "result_context": "results",
            "open_question": "source_claims",
            "note_context": "source_claims",
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
