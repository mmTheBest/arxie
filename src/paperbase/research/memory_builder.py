"""Deterministic derived-memory builder for collection research intelligence."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from csv import DictReader
from dataclasses import dataclass
from io import StringIO
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from paperbase.db.models import (
    CollectionPaper,
    Dataset,
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
)
from paperbase.db.repositories import ResearchIntelligenceRepository
from paperbase.source_safety import detect_source_secret


@dataclass(frozen=True, slots=True)
class ResearchIntelligenceMemoryBuildSummary:
    memory_type_counts: dict[str, int]
    graph_node_type_counts: dict[str, int]
    graph_edge_type_counts: dict[str, int]
    readiness_warnings: list[str]


@dataclass(slots=True)
class _GeneratedBuildRecords:
    memory_type_counts: Counter[str]
    graph_node_type_counts: Counter[str]
    graph_edge_type_counts: Counter[str]
    memory_record_keys: set[tuple[str, str]]
    graph_node_ids: set[str]
    graph_edge_keys: set[tuple[str, str, str]]


@dataclass(frozen=True, slots=True)
class _ResultTableRow:
    values: dict[str, str]
    display_headers: dict[str, str]
    header_order: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _SourceFactEntityTarget:
    entity_type: str
    edge_type: str
    node_id: str
    label: str
    normalized_name: str
    source_refs: tuple[dict[str, Any], ...]


_BUILDER_EDGE_TYPES = {
    "uses_method",
    "mentions_dataset",
    "reports_metric",
    "validated_by",
    "compares_against",
    "tests",
    "supports",
    "leaves_gap",
    "assumes",
    "contradicts",
    "extends",
}

_EXPERIMENT_ELEMENT_TYPES = {
    "experiment",
    "experimental-design",
    "ablation",
    "evaluation",
    "evaluation-protocol",
    "validation",
    "test",
    "control",
    "controlled-experiment",
    "experimental-variable",
    "analysis",
}

_SOURCE_FACT_MAX_PER_SOURCE = 24
_SOURCE_FACT_TEXT_LIMIT = 700
_BUILD_METADATA_MEMORY_TYPE = "build_metadata"
_BUILD_METADATA_SCHEMA_VERSION = "research-intelligence-memory-builder-v1"
_SOURCE_FACT_RULES: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (
        re.compile(r"^(?:claim|draft claim|hypothesis)\s*:\s*(?P<text>.+)$", re.IGNORECASE),
        "draft_claim",
        "user_claim",
    ),
    (
        re.compile(r"^(?:assumption|assumes)\s*:\s*(?P<text>.+)$", re.IGNORECASE),
        "assumption",
        "assumption",
    ),
    (
        re.compile(
            r"^(?:contradiction|contradicts|counterevidence|counter evidence)\s*:"
            r"\s*(?P<text>.+)$",
            re.IGNORECASE,
        ),
        "contradiction",
        "contradiction",
    ),
    (
        re.compile(
            r"^(?:extension|extends|builds on|follow[- ]up)\s*:\s*(?P<text>.+)$",
            re.IGNORECASE,
        ),
        "extension",
        "extension",
    ),
    (
        re.compile(r"^(?:constraint|requirement)\s*:\s*(?P<text>.+)$", re.IGNORECASE),
        "constraint",
        "project_constraint",
    ),
    (
        re.compile(r"^(?:method|approach|implementation)\s*:\s*(?P<text>.+)$", re.IGNORECASE),
        "method_context",
        "method_context",
    ),
    (
        re.compile(r"^(?:dataset|data)\s*:\s*(?P<text>.+)$", re.IGNORECASE),
        "dataset_context",
        "dataset_context",
    ),
    (
        re.compile(r"^(?:metric|evaluation)\s*:\s*(?P<text>.+)$", re.IGNORECASE),
        "metric_context",
        "metric_context",
    ),
    (
        re.compile(r"^(?:result|finding|observation)\s*:\s*(?P<text>.+)$", re.IGNORECASE),
        "result_context",
        "result_context",
    ),
    (
        re.compile(r"^(?:open question|question|risk|todo)\s*:\s*(?P<text>.+)$", re.IGNORECASE),
        "open_question",
        "open_question",
    ),
)

_INFERRED_SOURCE_FACT_RULES: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (
        re.compile(
            r"\b(?:we|our|this\s+draft|the\s+draft)\s+"
            r"(?:argue|claim|hypothesi[sz]e|propose|show|find|suggest)\b",
            re.IGNORECASE,
        ),
        "draft_claim",
        "user_claim",
    ),
    (
        re.compile(r"\b(?:we\s+)?assum(?:e|es|ed|ing)|\bassumption\b", re.IGNORECASE),
        "assumption",
        "assumption",
    ),
    (
        re.compile(
            r"\b(?:method|approach|implementation|pipeline|algorithm|model|baseline|"
            r"compare|compares|compared|uses|using|train|trains|evaluate|evaluates)\b",
            re.IGNORECASE,
        ),
        "method_context",
        "method_context",
    ),
    (
        re.compile(
            r"\b(?:dataset|data|benchmark|cohort|corpus|split|held[- ]out)\b",
            re.IGNORECASE,
        ),
        "dataset_context",
        "dataset_context",
    ),
    (
        re.compile(
            r"\b(?:metric|metrics|AUROC|AUC|F1|accuracy|precision|recall|RMSE|MAE|R2)\b",
            re.IGNORECASE,
        ),
        "metric_context",
        "metric_context",
    ),
    (
        re.compile(
            r"\b(?:result|results|reports?|shows?|achieves?|reaches?|improves?|"
            r"outperform|score|scores)\b",
            re.IGNORECASE,
        ),
        "result_context",
        "result_context",
    ),
    (
        re.compile(r"\b(?:open question|question|risk|limitation|todo|unclear)\b", re.IGNORECASE),
        "open_question",
        "open_question",
    ),
)

_SOURCE_TYPE_DEFAULT_FACT: dict[str, tuple[str, str]] = {
    "draft_path": ("draft_claim", "user_claim"),
    "results_path": ("result_context", "result_context"),
    "code_path": ("method_context", "method_context"),
    "text": ("note_context", "note_context"),
}

_SOURCE_FACT_TYPE_LABELS: dict[str, str] = {
    "draft_claim": "Draft claim",
    "assumption": "Assumption",
    "contradiction": "Contradiction",
    "extension": "Extension",
    "constraint": "Constraint",
    "method_context": "Method context",
    "dataset_context": "Dataset context",
    "metric_context": "Metric context",
    "result_context": "Result context",
    "open_question": "Open question",
    "note_context": "Study note",
}


class ResearchIntelligenceMemoryBuilder:
    """Build inspectable evidence, pattern, and graph memory from existing records."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.repository = ResearchIntelligenceRepository(session)

    def build(
        self,
        collection_id: str,
        workspace_id: str | None = None,
    ) -> ResearchIntelligenceMemoryBuildSummary:
        generated = _GeneratedBuildRecords(
            memory_type_counts=Counter(),
            graph_node_type_counts=Counter(),
            graph_edge_type_counts=Counter(),
            memory_record_keys=set(),
            graph_node_ids=set(),
            graph_edge_keys=set(),
        )
        try:
            snapshots = self._collection_snapshots(collection_id)
            input_signature = self._build_input_signature_from_snapshots(
                collection_id=collection_id,
                workspace_id=workspace_id,
                snapshots=snapshots,
            )
            readiness_warnings = self._readiness_warnings(snapshots)

            paper_nodes = self._write_paper_evidence_and_nodes(
                collection_id=collection_id,
                workspace_id=workspace_id,
                snapshots=snapshots,
                generated=generated,
            )
            entity_nodes = self._write_entity_nodes(
                collection_id=collection_id,
                workspace_id=workspace_id,
                snapshots=snapshots,
                generated=generated,
            )
            baseline_nodes = self._write_baseline_nodes(
                collection_id=collection_id,
                workspace_id=workspace_id,
                snapshots=snapshots,
                generated=generated,
            )
            self._write_finding_and_gap_nodes(
                collection_id=collection_id,
                workspace_id=workspace_id,
                snapshots=snapshots,
                paper_nodes=paper_nodes,
                generated=generated,
            )
            self._write_experiment_nodes_and_edges(
                collection_id=collection_id,
                workspace_id=workspace_id,
                snapshots=snapshots,
                entity_nodes=entity_nodes,
                generated=generated,
            )
            self._write_result_edges(
                collection_id=collection_id,
                workspace_id=workspace_id,
                snapshots=snapshots,
                paper_nodes=paper_nodes,
                entity_nodes=entity_nodes,
                baseline_nodes=baseline_nodes,
                generated=generated,
            )
            self._write_pattern_memory(
                collection_id=collection_id,
                workspace_id=workspace_id,
                snapshots=snapshots,
                generated=generated,
            )
            self._write_study_source_fact_memory(
                collection_id=collection_id,
                workspace_id=workspace_id,
                snapshots=snapshots,
                entity_nodes=entity_nodes,
                generated=generated,
            )
            self._write_study_constraint_nodes(
                collection_id=collection_id,
                workspace_id=workspace_id,
                generated=generated,
            )
            self._upsert_build_marker(
                collection_id=collection_id,
                workspace_id=workspace_id,
                input_signature=input_signature,
                generated=generated,
            )
            self._prune_stale_builder_rows(
                collection_id=collection_id,
                workspace_id=workspace_id,
                generated=generated,
            )
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise

        return ResearchIntelligenceMemoryBuildSummary(
            memory_type_counts=dict(generated.memory_type_counts),
            graph_node_type_counts=dict(generated.graph_node_type_counts),
            graph_edge_type_counts=dict(generated.graph_edge_type_counts),
            readiness_warnings=readiness_warnings,
        )

    def is_current(
        self,
        collection_id: str,
        workspace_id: str | None = None,
    ) -> bool:
        marker = self._build_marker_record(
            collection_id=collection_id,
            workspace_id=workspace_id,
        )
        if marker is None:
            return False
        payload = dict(marker.payload_json or {})
        if payload.get("schema_version") != _BUILD_METADATA_SCHEMA_VERSION:
            return False
        snapshots = self._collection_snapshots(collection_id)
        expected_signature = self._build_input_signature_from_snapshots(
            collection_id=collection_id,
            workspace_id=workspace_id,
            snapshots=snapshots,
        )
        return payload.get("input_signature") == expected_signature

    def _upsert_memory_record(
        self,
        *,
        generated: _GeneratedBuildRecords,
        **kwargs: Any,
    ) -> ResearchMemoryRecord:
        record = self.repository.upsert_memory_record(**kwargs, commit=False)
        generated.memory_type_counts[record.memory_type] += 1
        generated.memory_record_keys.add((record.memory_type, record.version_key))
        return record

    def _upsert_build_marker(
        self,
        *,
        collection_id: str,
        workspace_id: str | None,
        input_signature: str,
        generated: _GeneratedBuildRecords,
    ) -> ResearchMemoryRecord:
        version_key = self._build_marker_version_key(collection_id)
        record = self.repository.upsert_memory_record(
            collection_id=collection_id,
            workspace_id=workspace_id,
            memory_type=_BUILD_METADATA_MEMORY_TYPE,
            version_key=version_key,
            title="Research intelligence build metadata",
            summary="Internal freshness marker for derived research intelligence.",
            payload={
                "origin": "research_intelligence_memory_builder",
                "schema_version": _BUILD_METADATA_SCHEMA_VERSION,
                "input_signature": input_signature,
                "scope": "workspace" if workspace_id is not None else "collection",
                "workspace_id": workspace_id,
            },
            source_refs=[],
            confidence=1.0,
            commit=False,
        )
        generated.memory_record_keys.add((record.memory_type, record.version_key))
        return record

    def _upsert_graph_node(
        self,
        *,
        generated: _GeneratedBuildRecords,
        **kwargs: Any,
    ) -> ResearchGraphNode:
        node = self.repository.upsert_graph_node(**kwargs, commit=False)
        generated.graph_node_type_counts[node.node_type] += 1
        generated.graph_node_ids.add(node.id)
        return node

    def _upsert_graph_edge(
        self,
        *,
        generated: _GeneratedBuildRecords,
        **kwargs: Any,
    ) -> ResearchGraphEdge:
        edge = self.repository.upsert_graph_edge(**kwargs, commit=False)
        generated.graph_edge_type_counts[edge.edge_type] += 1
        generated.graph_edge_keys.add(
            (edge.source_node_id, edge.target_node_id, edge.edge_type)
        )
        return edge

    def _prune_stale_builder_rows(
        self,
        *,
        collection_id: str,
        workspace_id: str | None,
        generated: _GeneratedBuildRecords,
    ) -> None:
        existing_records = self._select_memory_records(
            collection_id=collection_id,
            workspace_id=workspace_id,
        )
        for record in existing_records:
            record_key = (record.memory_type, record.version_key)
            if (
                self._is_builder_memory_record(record, collection_id=collection_id)
                and record_key not in generated.memory_record_keys
            ):
                self.session.delete(record)

        existing_nodes = self._select_graph_nodes(
            collection_id=collection_id,
            workspace_id=workspace_id,
        )
        builder_node_ids = {
            node.id
            for node in existing_nodes
            if self._is_builder_graph_node(
                node,
                collection_id=collection_id,
                workspace_id=workspace_id,
            )
        }
        stale_node_ids = builder_node_ids - generated.graph_node_ids

        existing_edges = self._select_graph_edges(
            collection_id=collection_id,
            workspace_id=workspace_id,
        )
        stale_edge_ids = {
            edge.id
            for edge in existing_edges
            if edge.edge_type in _BUILDER_EDGE_TYPES
            and (
                edge.source_node_id in builder_node_ids
                or edge.target_node_id in builder_node_ids
            )
            and (
                edge.source_node_id,
                edge.target_node_id,
                edge.edge_type,
            )
            not in generated.graph_edge_keys
        }
        for edge in existing_edges:
            if edge.id in stale_edge_ids:
                self.session.delete(edge)

        referenced_node_ids = {
            node_id
            for edge in existing_edges
            if edge.id not in stale_edge_ids
            for node_id in (edge.source_node_id, edge.target_node_id)
        }
        for node in existing_nodes:
            if node.id in stale_node_ids and node.id not in referenced_node_ids:
                self.session.delete(node)

        self.session.flush()

    def _select_memory_records(
        self,
        *,
        collection_id: str,
        workspace_id: str | None,
    ) -> list[ResearchMemoryRecord]:
        statement = select(ResearchMemoryRecord).where(
            ResearchMemoryRecord.collection_id == collection_id
        )
        if workspace_id is None:
            statement = statement.where(ResearchMemoryRecord.workspace_id.is_(None))
        else:
            statement = statement.where(ResearchMemoryRecord.workspace_id == workspace_id)
        return list(self.session.execute(statement).scalars())

    def _select_graph_nodes(
        self,
        *,
        collection_id: str,
        workspace_id: str | None,
    ) -> list[ResearchGraphNode]:
        statement = select(ResearchGraphNode).where(
            ResearchGraphNode.collection_id == collection_id
        )
        if workspace_id is None:
            statement = statement.where(ResearchGraphNode.workspace_id.is_(None))
        else:
            statement = statement.where(ResearchGraphNode.workspace_id == workspace_id)
        return list(self.session.execute(statement).scalars())

    def _select_graph_edges(
        self,
        *,
        collection_id: str,
        workspace_id: str | None,
    ) -> list[ResearchGraphEdge]:
        statement = select(ResearchGraphEdge).where(
            ResearchGraphEdge.collection_id == collection_id
        )
        if workspace_id is None:
            statement = statement.where(ResearchGraphEdge.workspace_id.is_(None))
        else:
            statement = statement.where(ResearchGraphEdge.workspace_id == workspace_id)
        return list(self.session.execute(statement).scalars())

    def _is_builder_memory_record(
        self,
        record: ResearchMemoryRecord,
        *,
        collection_id: str,
    ) -> bool:
        source_fact_prefix = f"{self._stable_key('collection', collection_id, 'workspace')}:"
        if record.memory_type == _BUILD_METADATA_MEMORY_TYPE:
            return record.version_key == self._build_marker_version_key(collection_id)
        if record.memory_type == "source_fact":
            return record.version_key.startswith(source_fact_prefix)
        return record.version_key.startswith(
            (
                f"{self._stable_key('collection', collection_id, 'paper')}:",
                f"{self._stable_key('collection', collection_id, 'pattern')}:",
            )
        )

    def _is_builder_graph_node(
        self,
        node: ResearchGraphNode,
        *,
        collection_id: str,
        workspace_id: str | None,
    ) -> bool:
        prefixes = [
            f"{self._stable_key('collection', collection_id, 'paper')}:",
            f"{self._stable_key('collection', collection_id, 'method')}:",
            f"{self._stable_key('collection', collection_id, 'dataset')}:",
            f"{self._stable_key('collection', collection_id, 'metric')}:",
            f"{self._stable_key('collection', collection_id, 'baseline')}:",
            f"{self._stable_key('collection', collection_id, 'experiment')}:",
        ]
        if workspace_id is not None:
            prefixes.append(
                f"{self._stable_key('collection', collection_id, 'workspace', workspace_id)}:"
            )
        return node.stable_key.startswith(tuple(prefixes))

    def _build_marker_record(
        self,
        *,
        collection_id: str,
        workspace_id: str | None,
    ) -> ResearchMemoryRecord | None:
        statement = select(ResearchMemoryRecord).where(
            ResearchMemoryRecord.collection_id == collection_id,
            ResearchMemoryRecord.memory_type == _BUILD_METADATA_MEMORY_TYPE,
            ResearchMemoryRecord.version_key == self._build_marker_version_key(
                collection_id
            ),
        )
        if workspace_id is None:
            statement = statement.where(ResearchMemoryRecord.workspace_id.is_(None))
        else:
            statement = statement.where(ResearchMemoryRecord.workspace_id == workspace_id)
        return self.session.execute(statement.limit(1)).scalar_one_or_none()

    def _build_marker_version_key(self, collection_id: str) -> str:
        return self._stable_key("collection", collection_id, "build", "metadata")

    def _build_input_signature_from_snapshots(
        self,
        *,
        collection_id: str,
        workspace_id: str | None,
        snapshots: list[dict[str, Any]],
    ) -> str:
        return self._json_digest(
            {
                "schema_version": _BUILD_METADATA_SCHEMA_VERSION,
                "collection_id": collection_id,
                "workspace_id": workspace_id,
                "collection_snapshots": snapshots,
                "workspace_inputs": self._workspace_signature_inputs(workspace_id),
            }
        )

    def _workspace_signature_inputs(
        self,
        workspace_id: str | None,
    ) -> dict[str, Any] | None:
        if workspace_id is None:
            return None
        sources = self.session.execute(
            select(StudySource)
            .where(
                StudySource.workspace_id == workspace_id,
                StudySource.read_status == "ready",
            )
            .order_by(StudySource.created_at.asc(), StudySource.id.asc())
        ).scalars()
        study_brief = self.repository.get_study_brief(workspace_id)
        return {
            "workspace_id": workspace_id,
            "ready_sources": [
                self._study_source_signature_input(source) for source in sources
            ],
            "study_brief": self._study_brief_signature_input(study_brief),
        }

    def _study_source_signature_input(self, source: StudySource) -> dict[str, Any]:
        return {
            "id": source.id,
            "source_type": source.source_type,
            "title": source.title,
            "has_path": bool(source.path),
            "content": source.content,
            "summary": source.summary,
            "read_status": source.read_status,
            "source_size_bytes": source.source_size_bytes,
            "source_mtime_ns": source.source_mtime_ns,
        }

    def _study_brief_signature_input(
        self,
        study_brief: StudyBrief | None,
    ) -> dict[str, Any]:
        if study_brief is None:
            return {"missing": True}
        return {
            "id": study_brief.id,
            "version": study_brief.version,
            "updated_by": study_brief.updated_by,
            "content_digest": self._json_digest(study_brief.brief_json or {}),
        }

    def _json_digest(self, payload: Any) -> str:
        serialized = json.dumps(
            payload,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:32]

    def _collection_snapshots(self, collection_id: str) -> list[dict[str, Any]]:
        paper_rows = self.session.execute(
            select(Paper, CollectionPaper)
            .join(CollectionPaper, CollectionPaper.paper_id == Paper.id)
            .where(CollectionPaper.collection_id == collection_id)
            .order_by(
                CollectionPaper.position.asc().nullslast(),
                CollectionPaper.created_at.asc(),
                Paper.canonical_title.asc(),
            )
        ).all()
        if not paper_rows:
            return []

        paper_ids = [paper.id for paper, _membership in paper_rows]
        sections_by_paper = self._rows_by_paper_id(
            Section,
            paper_ids=paper_ids,
            order_by=(Section.ordinal.asc(), Section.created_at.asc()),
        )
        datasets_by_paper = self._rows_by_paper_id(
            Dataset,
            paper_ids=paper_ids,
            order_by=(Dataset.display_name.asc(), Dataset.id.asc()),
        )
        methods_by_paper = self._rows_by_paper_id(
            Method,
            paper_ids=paper_ids,
            order_by=(Method.display_name.asc(), Method.id.asc()),
        )
        metrics_by_paper = self._rows_by_paper_id(
            Metric,
            paper_ids=paper_ids,
            order_by=(Metric.display_name.asc(), Metric.id.asc()),
        )
        findings_by_paper = self._rows_by_paper_id(
            Finding,
            paper_ids=paper_ids,
            order_by=(Finding.created_at.asc(), Finding.id.asc()),
        )
        limitations_by_paper = self._rows_by_paper_id(
            Limitation,
            paper_ids=paper_ids,
            order_by=(Limitation.created_at.asc(), Limitation.id.asc()),
        )
        design_elements_by_paper = self._rows_by_paper_id(
            ResearchDesignElement,
            paper_ids=paper_ids,
            order_by=(
                ResearchDesignElement.element_type.asc(),
                ResearchDesignElement.title.asc(),
                ResearchDesignElement.id.asc(),
            ),
        )
        result_rows_by_paper = self._rows_by_paper_id(
            ResultRow,
            paper_ids=paper_ids,
            order_by=(ResultRow.created_at.asc(), ResultRow.id.asc()),
        )

        snapshots: list[dict[str, Any]] = []
        for paper, membership in paper_rows:
            datasets = [
                self._named_entity_snapshot(dataset) for dataset in datasets_by_paper[paper.id]
            ]
            methods = [
                self._named_entity_snapshot(method) for method in methods_by_paper[paper.id]
            ]
            metrics = [
                self._named_entity_snapshot(metric) for metric in metrics_by_paper[paper.id]
            ]
            dataset_by_id = {dataset["id"]: dataset for dataset in datasets}
            method_by_id = {method["id"]: method for method in methods}
            metric_by_id = {metric["id"]: metric for metric in metrics}
            snapshots.append(
                {
                    "paper": {
                        "id": paper.id,
                        "title": paper.canonical_title,
                        "abstract": paper.abstract,
                        "publication_year": paper.publication_year,
                        "venue": paper.venue,
                        "provider": paper.provider,
                        "external_id": paper.external_id,
                        "membership_note": membership.membership_note,
                    },
                    "sections": [
                        {
                            "id": section.id,
                            "title": section.title,
                            "ordinal": section.ordinal,
                            "page_start": section.page_start,
                            "page_end": section.page_end,
                            "text": self._bounded_text(section.text, limit=420),
                        }
                        for section in sections_by_paper[paper.id][:6]
                    ],
                    "datasets": datasets,
                    "methods": methods,
                    "metrics": metrics,
                    "findings": [
                        {
                            "id": finding.id,
                            "statement": self._clean_text(finding.statement),
                            "polarity": finding.polarity,
                        }
                        for finding in findings_by_paper[paper.id]
                    ],
                    "limitations": [
                        {
                            "id": limitation.id,
                            "statement": self._clean_text(limitation.statement),
                        }
                        for limitation in limitations_by_paper[paper.id]
                    ],
                    "research_design_elements": [
                        {
                            "id": element.id,
                            "element_type": element.element_type,
                            "title": self._clean_text(element.title),
                            "description": self._bounded_text(element.description, limit=420),
                            "metadata": dict(element.metadata_json or {}),
                        }
                        for element in design_elements_by_paper[paper.id]
                    ],
                    "result_rows": [
                        self._result_row_snapshot(
                            row,
                            dataset_by_id=dataset_by_id,
                            method_by_id=method_by_id,
                            metric_by_id=metric_by_id,
                        )
                        for row in result_rows_by_paper[paper.id]
                    ],
                }
            )
        return snapshots

    def _rows_by_paper_id(
        self,
        model,  # noqa: ANN001
        *,
        paper_ids: list[str],
        order_by: tuple[Any, ...],
    ) -> dict[str, list[Any]]:
        rows = self.session.execute(
            select(model).where(model.paper_id.in_(paper_ids)).order_by(*order_by)
        ).scalars()
        grouped: dict[str, list[Any]] = defaultdict(list)
        for row in rows:
            grouped[row.paper_id].append(row)
        return grouped

    def _named_entity_snapshot(self, item: Dataset | Method | Metric) -> dict[str, Any]:
        return {
            "id": item.id,
            "normalized_name": item.normalized_name,
            "display_name": item.display_name,
            "metadata": dict(item.metadata_json or {}),
        }

    def _result_row_snapshot(
        self,
        row: ResultRow,
        *,
        dataset_by_id: dict[str, dict[str, Any]],
        method_by_id: dict[str, dict[str, Any]],
        metric_by_id: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        dataset = dataset_by_id.get(row.dataset_id or "")
        method = method_by_id.get(row.method_id or "")
        metric = metric_by_id.get(row.metric_id or "")
        return {
            "id": row.id,
            "dataset_id": row.dataset_id,
            "dataset": dataset["display_name"] if dataset else None,
            "method_id": row.method_id,
            "method": method["display_name"] if method else None,
            "metric_id": row.metric_id,
            "metric": metric["display_name"] if metric else None,
            "split_name": row.split_name,
            "value_numeric": row.value_numeric,
            "value_text": row.value_text,
            "comparator_text": row.comparator_text,
            "notes": self._clean_text(row.notes or ""),
        }

    def _write_paper_evidence_and_nodes(
        self,
        *,
        collection_id: str,
        workspace_id: str | None,
        snapshots: list[dict[str, Any]],
        generated: _GeneratedBuildRecords,
    ) -> dict[str, str]:
        paper_nodes: dict[str, str] = {}
        for snapshot in snapshots:
            paper = snapshot["paper"]
            summary = self._paper_evidence_summary(snapshot)
            source_refs = self._source_refs_for_paper(snapshot)
            self._upsert_memory_record(
                generated=generated,
                collection_id=collection_id,
                workspace_id=workspace_id,
                paper_id=paper["id"],
                memory_type="evidence",
                version_key=self._stable_key(
                    "collection",
                    collection_id,
                    "paper",
                    paper["id"],
                    "summary",
                ),
                title=f"Evidence: {paper['title']}",
                summary=summary,
                payload=snapshot,
                source_refs=source_refs,
                confidence=1.0,
            )
            node = self._upsert_graph_node(
                generated=generated,
                collection_id=collection_id,
                workspace_id=workspace_id,
                node_type="paper",
                stable_key=self._stable_key("collection", collection_id, "paper", paper["id"]),
                label=paper["title"],
                payload={
                    "paper_id": paper["id"],
                    "publication_year": paper["publication_year"],
                    "venue": paper["venue"],
                    "source_ref": {"paper_id": paper["id"], "target_type": "paper"},
                },
            )
            paper_nodes[paper["id"]] = node.id
        return paper_nodes

    def _write_entity_nodes(
        self,
        *,
        collection_id: str,
        workspace_id: str | None,
        snapshots: list[dict[str, Any]],
        generated: _GeneratedBuildRecords,
    ) -> dict[tuple[str, str], str]:
        entity_nodes: dict[tuple[str, str], str] = {}
        for entity_type, key in (
            ("method", "methods"),
            ("dataset", "datasets"),
            ("metric", "metrics"),
        ):
            for item in self._group_named_entities(snapshots, key=key, entity_type=entity_type):
                stable_key = self._stable_key(
                    "collection",
                    collection_id,
                    entity_type,
                    item["normalized_name"],
                )
                node = self._upsert_graph_node(
                    generated=generated,
                    collection_id=collection_id,
                    workspace_id=workspace_id,
                    node_type=entity_type,
                    stable_key=stable_key,
                    label=item["label"],
                    payload={
                        "normalized_name": item["normalized_name"],
                        "paper_ids": item["paper_ids"],
                        "source_ids": item["source_ids"],
                        "occurrence_count": item["occurrence_count"],
                    },
                )
                for source_id in item["source_ids"]:
                    entity_nodes[(entity_type, source_id)] = node.id
        return entity_nodes

    def _write_baseline_nodes(
        self,
        *,
        collection_id: str,
        workspace_id: str | None,
        snapshots: list[dict[str, Any]],
        generated: _GeneratedBuildRecords,
    ) -> dict[str, str]:
        baseline_nodes: dict[str, str] = {}
        for baseline in self._group_baselines(snapshots):
            stable_key = self._stable_key(
                "collection",
                collection_id,
                "baseline",
                baseline["normalized_name"],
            )
            node = self._upsert_graph_node(
                generated=generated,
                collection_id=collection_id,
                workspace_id=workspace_id,
                node_type="baseline",
                stable_key=stable_key,
                label=baseline["label"],
                payload={
                    "normalized_name": baseline["normalized_name"],
                    "paper_ids": baseline["paper_ids"],
                    "source_ids": baseline["source_ids"],
                    "occurrence_count": baseline["occurrence_count"],
                },
            )
            baseline_nodes[baseline["normalized_name"]] = node.id
        return baseline_nodes

    def _write_finding_and_gap_nodes(
        self,
        *,
        collection_id: str,
        workspace_id: str | None,
        snapshots: list[dict[str, Any]],
        paper_nodes: dict[str, str],
        generated: _GeneratedBuildRecords,
    ) -> None:
        for snapshot in snapshots:
            paper = snapshot["paper"]
            paper_node_id = paper_nodes.get(paper["id"])
            for finding in snapshot["findings"]:
                finding_node = self._upsert_graph_node(
                    generated=generated,
                    collection_id=collection_id,
                    workspace_id=workspace_id,
                    node_type="finding",
                    stable_key=self._stable_key(
                        "collection",
                        collection_id,
                        "paper",
                        paper["id"],
                        "finding",
                        finding["id"],
                    ),
                    label=finding["statement"],
                    payload={
                        "paper_id": paper["id"],
                        "finding_id": finding["id"],
                        "polarity": finding["polarity"],
                    },
                )
                if paper_node_id is not None:
                    self._upsert_graph_edge(
                        generated=generated,
                        collection_id=collection_id,
                        workspace_id=workspace_id,
                        source_node_id=paper_node_id,
                        target_node_id=finding_node.id,
                        edge_type="supports",
                        evidence_refs=[
                            {
                                "paper_id": paper["id"],
                                "target_type": "finding",
                                "target_id": finding["id"],
                            }
                        ],
                    )

            for limitation in snapshot["limitations"]:
                gap_node = self._upsert_graph_node(
                    generated=generated,
                    collection_id=collection_id,
                    workspace_id=workspace_id,
                    node_type="gap",
                    stable_key=self._stable_key(
                        "collection",
                        collection_id,
                        "paper",
                        paper["id"],
                        "limitation",
                        limitation["id"],
                    ),
                    label=limitation["statement"],
                    payload={
                        "paper_id": paper["id"],
                        "limitation_id": limitation["id"],
                    },
                )
                if paper_node_id is not None:
                    self._upsert_graph_edge(
                        generated=generated,
                        collection_id=collection_id,
                        workspace_id=workspace_id,
                        source_node_id=paper_node_id,
                        target_node_id=gap_node.id,
                        edge_type="leaves_gap",
                        evidence_refs=[
                            {
                                "paper_id": paper["id"],
                                "target_type": "limitation",
                                "target_id": limitation["id"],
                            }
                        ],
                    )

    def _write_experiment_nodes_and_edges(
        self,
        *,
        collection_id: str,
        workspace_id: str | None,
        snapshots: list[dict[str, Any]],
        entity_nodes: dict[tuple[str, str], str],
        generated: _GeneratedBuildRecords,
    ) -> None:
        for snapshot in snapshots:
            paper = snapshot["paper"]
            paper_id = paper["id"]
            for element in snapshot["research_design_elements"]:
                if not self._is_experiment_design_element(element):
                    continue
                source_ref = {
                    "paper_id": paper_id,
                    "target_type": "research_design_element",
                    "target_id": element["id"],
                }
                experiment_node = self._upsert_graph_node(
                    generated=generated,
                    collection_id=collection_id,
                    workspace_id=workspace_id,
                    node_type="experiment",
                    stable_key=self._stable_key(
                        "collection",
                        collection_id,
                        "experiment",
                        paper_id,
                        element["id"],
                    ),
                    label=element["title"] or self._experiment_fallback_label(element),
                    payload={
                        "origin": "research_design_element",
                        "paper_id": paper_id,
                        "design_element_id": element["id"],
                        "element_type": element["element_type"],
                        "source_ref": source_ref,
                    },
                )
                self._write_experiment_entity_edges(
                    collection_id=collection_id,
                    workspace_id=workspace_id,
                    snapshot=snapshot,
                    experiment_node=experiment_node,
                    element=element,
                    source_ref=source_ref,
                    entity_nodes=entity_nodes,
                    generated=generated,
                )

    def _write_experiment_entity_edges(
        self,
        *,
        collection_id: str,
        workspace_id: str | None,
        snapshot: dict[str, Any],
        experiment_node: ResearchGraphNode,
        element: dict[str, Any],
        source_ref: dict[str, Any],
        entity_nodes: dict[tuple[str, str], str],
        generated: _GeneratedBuildRecords,
    ) -> None:
        text = self._experiment_design_text(element)
        if not text:
            return
        linked_targets: set[tuple[str, str]] = set()
        for entity_type, key in (
            ("method", "methods"),
            ("dataset", "datasets"),
            ("metric", "metrics"),
        ):
            for item in snapshot[key]:
                node_id = entity_nodes.get((entity_type, item["id"]))
                if node_id is None or (entity_type, node_id) in linked_targets:
                    continue
                if not self._text_mentions_entity_label(
                    text,
                    label=item["display_name"],
                    normalized_name=item["normalized_name"],
                ):
                    continue
                linked_targets.add((entity_type, node_id))
                self._upsert_graph_edge(
                    generated=generated,
                    collection_id=collection_id,
                    workspace_id=workspace_id,
                    source_node_id=experiment_node.id,
                    target_node_id=node_id,
                    edge_type="tests",
                    evidence_refs=[
                        dict(source_ref),
                        {
                            "paper_id": snapshot["paper"]["id"],
                            "target_type": entity_type,
                            "target_id": item["id"],
                        },
                    ],
                    weight=0.8,
                    payload={
                        "origin": "design_element_evidence_link",
                        "design_element_id": element["id"],
                        "design_element_type": element["element_type"],
                        "target_entity_type": entity_type,
                        "target_label": item["display_name"],
                        "match_rule": "label_or_normalized_token_sequence",
                    },
                )

    def _write_result_edges(
        self,
        *,
        collection_id: str,
        workspace_id: str | None,
        snapshots: list[dict[str, Any]],
        paper_nodes: dict[str, str],
        entity_nodes: dict[tuple[str, str], str],
        baseline_nodes: dict[str, str],
        generated: _GeneratedBuildRecords,
    ) -> None:
        result_edges: dict[tuple[str, str, str], dict[str, Any]] = {}
        for snapshot in snapshots:
            paper = snapshot["paper"]
            paper_node_id = paper_nodes.get(paper["id"])
            for method in snapshot["methods"]:
                method_node_id = entity_nodes.get(("method", method["id"]))
                if paper_node_id is not None and method_node_id is not None:
                    self._upsert_graph_edge(
                        generated=generated,
                        collection_id=collection_id,
                        workspace_id=workspace_id,
                        source_node_id=paper_node_id,
                        target_node_id=method_node_id,
                        edge_type="uses_method",
                        evidence_refs=[
                            {
                                "paper_id": paper["id"],
                                "target_type": "method",
                                "target_id": method["id"],
                            }
                        ],
                    )

            for dataset in snapshot["datasets"]:
                dataset_node_id = entity_nodes.get(("dataset", dataset["id"]))
                if paper_node_id is not None and dataset_node_id is not None:
                    self._upsert_graph_edge(
                        generated=generated,
                        collection_id=collection_id,
                        workspace_id=workspace_id,
                        source_node_id=paper_node_id,
                        target_node_id=dataset_node_id,
                        edge_type="mentions_dataset",
                        evidence_refs=[
                            {
                                "paper_id": paper["id"],
                                "target_type": "dataset",
                                "target_id": dataset["id"],
                            }
                        ],
                    )

            for metric in snapshot["metrics"]:
                metric_node_id = entity_nodes.get(("metric", metric["id"]))
                if paper_node_id is not None and metric_node_id is not None:
                    self._upsert_graph_edge(
                        generated=generated,
                        collection_id=collection_id,
                        workspace_id=workspace_id,
                        source_node_id=paper_node_id,
                        target_node_id=metric_node_id,
                        edge_type="reports_metric",
                        evidence_refs=[
                            {
                                "paper_id": paper["id"],
                                "target_type": "metric",
                                "target_id": metric["id"],
                            }
                        ],
                    )

            for row in snapshot["result_rows"]:
                method_node_id = entity_nodes.get(("method", row["method_id"] or ""))
                dataset_node_id = entity_nodes.get(("dataset", row["dataset_id"] or ""))
                metric_node_id = entity_nodes.get(("metric", row["metric_id"] or ""))
                comparator_text = self._clean_text(row["comparator_text"] or "")
                baseline_key = self._baseline_normalized_name(comparator_text)
                baseline_node_id = (
                    baseline_nodes.get(baseline_key) if baseline_key is not None else None
                )
                evidence_ref = {
                    "paper_id": paper["id"],
                    "target_type": "result_row",
                    "target_id": row["id"],
                }
                if method_node_id is not None and dataset_node_id is not None:
                    result_edge = result_edges.setdefault(
                        (method_node_id, dataset_node_id, "validated_by"),
                        {"evidence_refs": [], "result_rows": []},
                    )
                    result_edge["evidence_refs"].append(evidence_ref)
                    result_edge["result_rows"].append(
                        {
                            "paper_id": paper["id"],
                            "result_row_id": row["id"],
                            "method_id": row["method_id"],
                            "dataset_id": row["dataset_id"],
                            "split_name": row["split_name"],
                        }
                    )
                if method_node_id is not None and metric_node_id is not None:
                    result_edge = result_edges.setdefault(
                        (method_node_id, metric_node_id, "reports_metric"),
                        {"evidence_refs": [], "result_rows": []},
                    )
                    result_edge["evidence_refs"].append(evidence_ref)
                    result_edge["result_rows"].append(
                        {
                            "paper_id": paper["id"],
                            "result_row_id": row["id"],
                            "method_id": row["method_id"],
                            "metric_id": row["metric_id"],
                            "split_name": row["split_name"],
                            "value": self._result_value(row),
                        }
                    )
                if method_node_id is not None and baseline_node_id is not None:
                    result_edge = result_edges.setdefault(
                        (method_node_id, baseline_node_id, "compares_against"),
                        {"evidence_refs": [], "result_rows": []},
                    )
                    result_edge["evidence_refs"].append(evidence_ref)
                    result_edge["result_rows"].append(
                        {
                            "paper_id": paper["id"],
                            "result_row_id": row["id"],
                            "method_id": row["method_id"],
                            "comparator_text": comparator_text,
                            "metric_id": row["metric_id"],
                            "split_name": row["split_name"],
                            "value": self._result_value(row),
                        }
                    )
        for (source_node_id, target_node_id, edge_type), edge_payload in sorted(
            result_edges.items(),
            key=lambda item: item[0],
        ):
            self._upsert_graph_edge(
                generated=generated,
                collection_id=collection_id,
                workspace_id=workspace_id,
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                edge_type=edge_type,
                evidence_refs=edge_payload["evidence_refs"],
                payload={"result_rows": edge_payload["result_rows"]},
            )

    def _write_pattern_memory(
        self,
        *,
        collection_id: str,
        workspace_id: str | None,
        snapshots: list[dict[str, Any]],
        generated: _GeneratedBuildRecords,
    ) -> None:
        pattern_specs = [
            (
                "method",
                "Recurring methods",
                self._group_named_entities(snapshots, key="methods", entity_type="method"),
            ),
            (
                "dataset",
                "Recurring datasets",
                self._group_named_entities(snapshots, key="datasets", entity_type="dataset"),
            ),
            (
                "metric",
                "Recurring metrics",
                self._group_named_entities(snapshots, key="metrics", entity_type="metric"),
            ),
            ("ablation", "Ablation patterns", self._ablation_patterns(snapshots)),
            ("limitation", "Limitation patterns", self._limitation_patterns(snapshots)),
            ("validation_norm", "Validation norms", self._validation_norm_patterns(snapshots)),
        ]
        for pattern_type, title, items in pattern_specs:
            if not items:
                continue
            self._upsert_memory_record(
                generated=generated,
                collection_id=collection_id,
                workspace_id=workspace_id,
                memory_type="pattern",
                version_key=self._stable_key(
                    "collection",
                    collection_id,
                    "pattern",
                    pattern_type,
                ),
                title=title,
                summary=self._pattern_summary(pattern_type=pattern_type, items=items),
                payload={"pattern_type": pattern_type, "items": items},
                source_refs=self._pattern_source_refs(items),
                confidence=1.0,
            )

    def _write_study_source_fact_memory(
        self,
        *,
        collection_id: str,
        workspace_id: str | None,
        snapshots: list[dict[str, Any]],
        entity_nodes: dict[tuple[str, str], str],
        generated: _GeneratedBuildRecords,
    ) -> None:
        if workspace_id is None:
            return
        link_targets = self._source_fact_entity_link_targets(
            snapshots=snapshots,
            entity_nodes=entity_nodes,
        )
        sources = self.session.execute(
            select(StudySource)
            .where(
                StudySource.workspace_id == workspace_id,
                StudySource.read_status == "ready",
            )
            .order_by(StudySource.created_at.asc(), StudySource.id.asc())
        ).scalars()
        for source in sources:
            for fact in self._study_source_facts(source):
                fact_id = self._source_fact_id(source_id=source.id, fact=fact)
                source_ref = {
                    "reference_type": "study_source",
                    "source_id": source.id,
                    "source_type": source.source_type,
                    "target_type": "source_fact",
                    "target_id": fact_id,
                }
                payload = {
                    "origin": "study_source",
                    "fact_id": fact_id,
                    "fact_type": fact["fact_type"],
                    "fact_text": fact["fact_text"],
                    "source_id": source.id,
                    "source_type": source.source_type,
                    "source_title": source.title,
                    "source_locator": "registered_path" if source.path else "inline_text",
                    "read_status": source.read_status,
                    "extraction_rule": fact["extraction_rule"],
                }
                version_key = self._stable_key(
                    "collection",
                    collection_id,
                    "workspace",
                    workspace_id,
                    "source",
                    source.id,
                    "fact",
                    fact_id,
                )
                summary = f"{self._source_fact_type_label(fact['fact_type'])}: {fact['fact_text']}"
                self._upsert_memory_record(
                    generated=generated,
                    collection_id=collection_id,
                    workspace_id=workspace_id,
                    memory_type="source_fact",
                    version_key=version_key,
                    title=summary,
                    summary=summary,
                    payload=payload,
                    source_refs=[source_ref],
                    confidence=0.85,
                )
                source_fact_node = self._upsert_graph_node(
                    generated=generated,
                    collection_id=collection_id,
                    workspace_id=workspace_id,
                    node_type=fact["node_type"],
                    stable_key=version_key,
                    label=fact["fact_text"],
                    payload=payload,
                )
                self._write_source_fact_entity_links(
                    collection_id=collection_id,
                    workspace_id=workspace_id,
                    source_fact_node=source_fact_node,
                    source_ref=source_ref,
                    payload=payload,
                    link_targets=link_targets,
                    generated=generated,
                )

    def _source_fact_entity_link_targets(
        self,
        *,
        snapshots: list[dict[str, Any]],
        entity_nodes: dict[tuple[str, str], str],
    ) -> list[_SourceFactEntityTarget]:
        targets: list[_SourceFactEntityTarget] = []
        for entity_type, key, edge_type in (
            ("method", "methods", "uses_method"),
            ("dataset", "datasets", "mentions_dataset"),
            ("metric", "metrics", "reports_metric"),
        ):
            for item in self._group_named_entities(
                snapshots,
                key=key,
                entity_type=entity_type,
            ):
                node_id = next(
                    (
                        entity_nodes.get((entity_type, source_id))
                        for source_id in item["source_ids"]
                        if entity_nodes.get((entity_type, source_id)) is not None
                    ),
                    None,
                )
                if node_id is None:
                    continue
                targets.append(
                    _SourceFactEntityTarget(
                        entity_type=entity_type,
                        edge_type=edge_type,
                        node_id=node_id,
                        label=item["label"],
                        normalized_name=item["normalized_name"],
                        source_refs=tuple(dict(ref) for ref in item["source_refs"]),
                    )
                )
        return targets

    def _write_source_fact_entity_links(
        self,
        *,
        collection_id: str,
        workspace_id: str,
        source_fact_node: ResearchGraphNode,
        source_ref: dict[str, Any],
        payload: dict[str, Any],
        link_targets: list[_SourceFactEntityTarget],
        generated: _GeneratedBuildRecords,
    ) -> None:
        fact_text = str(payload.get("fact_text") or "")
        semantic_edge_type = self._source_fact_semantic_edge_type(
            str(payload.get("fact_type") or "")
        )
        for target in link_targets:
            if not self._source_fact_mentions_entity(fact_text, target):
                continue
            evidence_refs = [
                dict(source_ref),
                *[dict(ref) for ref in target.source_refs],
            ]
            edge_payload = {
                "origin": "source_fact_evidence_link",
                "source_fact_id": payload["fact_id"],
                "source_fact_type": payload["fact_type"],
                "source_id": payload["source_id"],
                "target_entity_type": target.entity_type,
                "target_label": target.label,
                "match_rule": "label_or_normalized_token_sequence",
            }
            self._upsert_graph_edge(
                generated=generated,
                collection_id=collection_id,
                workspace_id=workspace_id,
                source_node_id=source_fact_node.id,
                target_node_id=target.node_id,
                edge_type=target.edge_type,
                evidence_refs=evidence_refs,
                weight=0.7,
                payload=edge_payload,
            )
            if semantic_edge_type is None or semantic_edge_type == target.edge_type:
                continue
            self._upsert_graph_edge(
                generated=generated,
                collection_id=collection_id,
                workspace_id=workspace_id,
                source_node_id=source_fact_node.id,
                target_node_id=target.node_id,
                edge_type=semantic_edge_type,
                evidence_refs=[dict(ref) for ref in evidence_refs],
                weight=0.75,
                payload={
                    **edge_payload,
                    "base_edge_type": target.edge_type,
                    "semantic_edge_rule": "source_fact_type",
                },
            )

    def _source_fact_semantic_edge_type(self, fact_type: str) -> str | None:
        if fact_type == "assumption":
            return "assumes"
        if fact_type == "contradiction":
            return "contradicts"
        if fact_type == "extension":
            return "extends"
        return None

    def _source_fact_mentions_entity(
        self,
        fact_text: str,
        target: _SourceFactEntityTarget,
    ) -> bool:
        return self._text_mentions_entity_label(
            fact_text,
            label=target.label,
            normalized_name=target.normalized_name,
        )

    def _contains_entity_label(self, *, fact_text: str, label: str) -> bool:
        label = self._clean_text(label)
        if not label:
            return False
        pattern = rf"(?<![A-Za-z0-9]){re.escape(label)}(?![A-Za-z0-9])"
        return re.search(pattern, fact_text, flags=re.IGNORECASE) is not None

    def _text_mentions_entity_label(
        self,
        text: str,
        *,
        label: str,
        normalized_name: str,
    ) -> bool:
        if self._contains_entity_label(fact_text=text, label=label):
            return True

        normalized_text = self._normalize_label(text)
        normalized_label = self._normalize_label(label)
        normalized_entity_name = self._normalize_label(normalized_name)
        padded_text = f"-{normalized_text}-"
        return any(
            len(candidate) >= 3 and f"-{candidate}-" in padded_text
            for candidate in (normalized_label, normalized_entity_name)
        )

    def _is_experiment_design_element(self, element: dict[str, Any]) -> bool:
        element_type = self._normalize_label(str(element.get("element_type") or ""))
        return element_type in _EXPERIMENT_ELEMENT_TYPES

    def _experiment_design_text(self, element: dict[str, Any]) -> str:
        metadata = element.get("metadata")
        metadata_terms: list[str] = []
        if isinstance(metadata, dict):
            for value in metadata.values():
                if isinstance(value, str):
                    metadata_terms.append(value)
                elif isinstance(value, list | tuple):
                    metadata_terms.extend(str(item) for item in value if item is not None)
        return self._clean_text(
            " ".join(
                str(part)
                for part in (
                    element.get("title"),
                    element.get("description"),
                    " ".join(metadata_terms),
                )
                if part
            )
        )

    def _experiment_fallback_label(self, element: dict[str, Any]) -> str:
        element_type = self._clean_text(str(element.get("element_type") or "Experiment"))
        return element_type.replace("_", " ").title()

    def _study_source_facts(self, source: StudySource) -> list[dict[str, str]]:
        facts: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        source_content = (
            source.content
            if isinstance(source.content, str) and source.content.strip()
            else ""
        )
        source_summary = source.summary if isinstance(source.summary, str) else ""
        if (
            detect_source_secret(source_content) is not None
            or detect_source_secret(source_summary) is not None
        ):
            return []
        source_text = source_content or source_summary
        for line in self._source_fact_lines(source_text):
            for pattern, fact_type, node_type in _SOURCE_FACT_RULES:
                match = pattern.match(line)
                if match is None:
                    continue
                fact_text = self._bounded_text(
                    match.group("text"),
                    limit=_SOURCE_FACT_TEXT_LIMIT,
                )
                self._append_source_fact(
                    facts,
                    seen,
                    fact_type=fact_type,
                    node_type=node_type,
                    fact_text=fact_text,
                    extraction_rule=pattern.pattern,
                )
                break
            if len(facts) >= _SOURCE_FACT_MAX_PER_SOURCE:
                break
        if facts:
            return facts

        facts = self._inferred_source_facts(source, source_text)
        if facts:
            return facts

        fallback = self._fallback_source_fact(source)
        return [fallback] if fallback is not None else []

    def _append_source_fact(
        self,
        facts: list[dict[str, str]],
        seen: set[tuple[str, str]],
        *,
        fact_type: str,
        node_type: str,
        fact_text: str,
        extraction_rule: str,
    ) -> None:
        bounded_text = self._bounded_text(fact_text, limit=_SOURCE_FACT_TEXT_LIMIT)
        key = (fact_type, bounded_text.casefold())
        if not bounded_text or key in seen or len(facts) >= _SOURCE_FACT_MAX_PER_SOURCE:
            return
        seen.add(key)
        facts.append(
            {
                "fact_type": fact_type,
                "node_type": node_type,
                "fact_text": bounded_text,
                "extraction_rule": extraction_rule,
            }
        )

    def _inferred_source_facts(
        self,
        source: StudySource,
        source_text: str,
    ) -> list[dict[str, str]]:
        facts: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()

        if source.source_type == "results_path":
            self._append_result_table_source_facts(facts, seen, source_text=source_text)
            if facts:
                return facts

        for line in self._source_fact_lines(source_text):
            for pattern, fact_type, node_type in _INFERRED_SOURCE_FACT_RULES:
                if pattern.search(line) is None:
                    continue
                self._append_source_fact(
                    facts,
                    seen,
                    fact_type=fact_type,
                    node_type=node_type,
                    fact_text=line,
                    extraction_rule=f"inferred:{fact_type}",
                )
                break
            if len(facts) >= _SOURCE_FACT_MAX_PER_SOURCE:
                break
        return facts

    def _append_result_table_source_facts(
        self,
        facts: list[dict[str, str]],
        seen: set[tuple[str, str]],
        *,
        source_text: str,
    ) -> None:
        rows = self._result_table_rows(source_text)
        if not rows:
            return
        for row in rows:
            dataset = self._result_table_value(row, "dataset", "data", "benchmark")
            method = self._result_table_value(row, "method", "model", "approach")
            value = self._result_table_value(row, "value", "result", "score_value")
            metric = self._result_table_value(row, "metric", "measure")
            if value and not metric:
                metric = self._result_table_value(row, "score")

            if dataset:
                self._append_source_fact(
                    facts,
                    seen,
                    fact_type="dataset_context",
                    node_type="dataset_context",
                    fact_text=f"Result table dataset: {dataset}.",
                    extraction_rule="inferred:result_table",
                )
            if method:
                self._append_source_fact(
                    facts,
                    seen,
                    fact_type="method_context",
                    node_type="method_context",
                    fact_text=f"Result table method: {method}.",
                    extraction_rule="inferred:result_table",
                )
            if metric:
                self._append_source_fact(
                    facts,
                    seen,
                    fact_type="metric_context",
                    node_type="metric_context",
                    fact_text=f"Result table metric: {metric}.",
                    extraction_rule="inferred:result_table",
                )
            if dataset and method and metric and value:
                self._append_source_fact(
                    facts,
                    seen,
                    fact_type="result_context",
                    node_type="result_context",
                    fact_text=f"{dataset} / {method} reports {metric} = {value}.",
                    extraction_rule="inferred:result_table",
                )
            elif dataset and method:
                for metric, value in self._wide_result_table_values(row):
                    self._append_source_fact(
                        facts,
                        seen,
                        fact_type="metric_context",
                        node_type="metric_context",
                        fact_text=f"Result table metric: {metric}.",
                        extraction_rule="inferred:result_table",
                    )
                    self._append_source_fact(
                        facts,
                        seen,
                        fact_type="result_context",
                        node_type="result_context",
                        fact_text=f"{dataset} / {method} reports {metric} = {value}.",
                        extraction_rule="inferred:result_table",
                    )
                    if len(facts) >= _SOURCE_FACT_MAX_PER_SOURCE:
                        break
            if len(facts) >= _SOURCE_FACT_MAX_PER_SOURCE:
                break

    def _result_table_rows(self, source_text: str) -> list[_ResultTableRow]:
        csv_rows = self._csv_result_table_rows(source_text)
        if csv_rows:
            return csv_rows
        return self._markdown_result_table_rows(source_text)

    def _csv_result_table_rows(self, source_text: str) -> list[_ResultTableRow]:
        lines = [
            line.strip()
            for line in source_text.splitlines()
            if line.strip() and "," in line
        ]
        if len(lines) < 2:
            return []
        try:
            rows = list(DictReader(StringIO("\n".join(lines))))
        except Exception:
            return []
        header_order, display_headers = self._result_table_headers(rows)
        normalized_rows: list[_ResultTableRow] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            values: dict[str, str] = {}
            for key, value in row.items():
                if key is None:
                    continue
                normalized_key = self._clean_text(str(key)).casefold()
                if not normalized_key:
                    continue
                values[normalized_key] = self._clean_result_table_value(value)
            if any(values.values()):
                normalized_rows.append(
                    _ResultTableRow(
                        values=values,
                        display_headers=display_headers,
                        header_order=header_order,
                    )
                )
        return normalized_rows

    def _markdown_result_table_rows(self, source_text: str) -> list[_ResultTableRow]:
        table_lines = [
            line.strip()
            for line in source_text.splitlines()
            if line.strip().startswith("|") and line.strip().endswith("|")
        ]
        if len(table_lines) < 3:
            return []
        raw_header = [self._clean_text(cell) for cell in table_lines[0].strip("|").split("|")]
        header_order = tuple(header.casefold() for header in raw_header if header)
        display_headers = {header.casefold(): header for header in raw_header if header}
        rows: list[_ResultTableRow] = []
        for line in table_lines[2:]:
            cells = [self._clean_result_table_value(cell) for cell in line.strip("|").split("|")]
            if len(cells) != len(raw_header):
                continue
            values = {
                key: value
                for key, value in zip(header_order, cells, strict=False)
                if key
            }
            if any(values.values()):
                rows.append(
                    _ResultTableRow(
                        values=values,
                        display_headers=display_headers,
                        header_order=header_order,
                    )
                )
        return rows

    def _result_table_headers(
        self,
        rows: list[dict[str | None, str | list[str] | None]],
    ) -> tuple[tuple[str, ...], dict[str, str]]:
        first_row = rows[0] if rows else {}
        raw_headers = [
            self._clean_text(str(key))
            for key in first_row
            if key is not None and self._clean_text(str(key))
        ]
        header_order = tuple(header.casefold() for header in raw_headers)
        display_headers = {header.casefold(): header for header in raw_headers}
        return header_order, display_headers

    def _clean_result_table_value(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, list | tuple):
            value = " ".join(str(item) for item in value if item is not None)
        return self._clean_text(str(value))

    def _result_table_value(self, row: _ResultTableRow, *keys: str) -> str:
        for key in keys:
            value = row.values.get(key)
            if value:
                return self._bounded_text(value, limit=160)
        return ""

    def _wide_result_table_values(self, row: _ResultTableRow) -> list[tuple[str, str]]:
        context_keys = {
            "dataset",
            "data",
            "benchmark",
            "method",
            "model",
            "approach",
            "metric",
            "measure",
            "value",
            "result",
            "score_value",
        }
        metric_values: list[tuple[str, str]] = []
        for key in row.header_order:
            if key in context_keys:
                continue
            value = row.values.get(key)
            if not value:
                continue
            metric = self._bounded_text(row.display_headers.get(key, key), limit=160)
            bounded_value = self._bounded_text(value, limit=160)
            if metric and bounded_value:
                metric_values.append((metric, bounded_value))
        return metric_values

    def _source_fact_lines(self, source_text: str) -> list[str]:
        lines: list[str] = []
        for raw_line in re.split(r"[\r\n]+", source_text):
            line = self._clean_text(re.sub(r"^(?:[-*]|\d+[.)])\s*", "", raw_line))
            if line:
                lines.append(line)
        return lines

    def _fallback_source_fact(self, source: StudySource) -> dict[str, str] | None:
        fact_type, node_type = _SOURCE_TYPE_DEFAULT_FACT.get(
            source.source_type,
            ("note_context", "note_context"),
        )
        source_text = source.summary or source.content or ""
        fact_text = self._bounded_text(source_text, limit=_SOURCE_FACT_TEXT_LIMIT)
        if not fact_text:
            return None
        return {
            "fact_type": fact_type,
            "node_type": node_type,
            "fact_text": fact_text,
            "extraction_rule": f"default:{source.source_type}",
        }

    def _source_fact_id(self, *, source_id: str, fact: dict[str, str]) -> str:
        raw_key = "|".join(
            (
                source_id,
                fact["fact_type"],
                fact["node_type"],
                fact["fact_text"],
            )
        )
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]

    def _source_fact_type_label(self, fact_type: str) -> str:
        return _SOURCE_FACT_TYPE_LABELS.get(fact_type, fact_type.replace("_", " ").title())

    def _write_study_constraint_nodes(
        self,
        *,
        collection_id: str,
        workspace_id: str | None,
        generated: _GeneratedBuildRecords,
    ) -> None:
        if workspace_id is None:
            return
        study_brief = self.repository.get_study_brief(workspace_id)
        if study_brief is None:
            return
        constraints = study_brief.brief_json.get("constraints")
        if not isinstance(constraints, list):
            return
        for index, constraint in enumerate(constraints):
            if isinstance(constraint, str):
                label = self._clean_text(constraint)
                payload = {"constraint": label}
            elif isinstance(constraint, dict):
                label = self._clean_text(
                    str(constraint.get("title") or constraint.get("text") or "")
                )
                payload = dict(constraint)
            else:
                continue
            if not label:
                continue
            self._upsert_graph_node(
                generated=generated,
                collection_id=collection_id,
                workspace_id=workspace_id,
                node_type="study_constraint",
                stable_key=self._stable_key(
                    "collection",
                    collection_id,
                    "workspace",
                    workspace_id,
                    "constraint",
                    str(index),
                    label,
                ),
                label=label,
                payload=payload,
            )

    def _group_named_entities(
        self,
        snapshots: list[dict[str, Any]],
        *,
        key: str,
        entity_type: str,
    ) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for snapshot in snapshots:
            paper_id = snapshot["paper"]["id"]
            for item in snapshot[key]:
                normalized_name = item["normalized_name"] or self._normalize_label(
                    item["display_name"]
                )
                group = grouped.setdefault(
                    normalized_name,
                    {
                        "pattern_type": entity_type,
                        "normalized_name": normalized_name,
                        "label": item["display_name"],
                        "paper_ids": [],
                        "source_ids": [],
                        "source_refs": [],
                        "occurrence_count": 0,
                    },
                )
                if paper_id not in group["paper_ids"]:
                    group["paper_ids"].append(paper_id)
                group["source_ids"].append(item["id"])
                group["source_refs"].append(
                    {
                        "paper_id": paper_id,
                        "target_type": entity_type,
                        "target_id": item["id"],
                    }
                )
                group["occurrence_count"] += 1
        return [
            self._sorted_occurrence_item(item)
            for item in sorted(grouped.values(), key=lambda value: value["label"].lower())
        ]

    def _group_baselines(self, snapshots: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for snapshot in snapshots:
            paper_id = snapshot["paper"]["id"]
            for row in snapshot["result_rows"]:
                label = self._clean_text(row["comparator_text"] or "")
                if not label:
                    continue
                normalized_name = self._baseline_normalized_name(label)
                if normalized_name is None:
                    continue
                group = grouped.setdefault(
                    normalized_name,
                    {
                        "normalized_name": normalized_name,
                        "label": label,
                        "paper_ids": [],
                        "source_ids": [],
                        "source_refs": [],
                        "occurrence_count": 0,
                    },
                )
                if paper_id not in group["paper_ids"]:
                    group["paper_ids"].append(paper_id)
                group["source_ids"].append(row["id"])
                group["source_refs"].append(
                    {
                        "paper_id": paper_id,
                        "target_type": "result_row",
                        "target_id": row["id"],
                    }
                )
                group["occurrence_count"] += 1
        return [
            self._sorted_occurrence_item(item)
            for item in sorted(grouped.values(), key=lambda value: value["label"].lower())
        ]

    def _ablation_patterns(self, snapshots: list[dict[str, Any]]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for snapshot in snapshots:
            paper_id = snapshot["paper"]["id"]
            for element in snapshot["research_design_elements"]:
                if "ablation" not in self._normalize_label(element["element_type"]):
                    continue
                items.append(
                    {
                        "pattern_type": "ablation",
                        "label": element["title"],
                        "description": element["description"],
                        "paper_ids": [paper_id],
                        "source_ids": [element["id"]],
                        "source_refs": [
                            {
                                "paper_id": paper_id,
                                "target_type": "research_design_element",
                                "target_id": element["id"],
                            }
                        ],
                        "occurrence_count": 1,
                    }
                )
        return sorted(items, key=lambda item: item["label"].lower())

    def _limitation_patterns(self, snapshots: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for snapshot in snapshots:
            paper_id = snapshot["paper"]["id"]
            for limitation in snapshot["limitations"]:
                label = limitation["statement"]
                key = self._normalize_label(label)
                group = grouped.setdefault(
                    key,
                    {
                        "pattern_type": "limitation",
                        "label": label,
                        "paper_ids": [],
                        "source_ids": [],
                        "source_refs": [],
                        "occurrence_count": 0,
                    },
                )
                if paper_id not in group["paper_ids"]:
                    group["paper_ids"].append(paper_id)
                group["source_ids"].append(limitation["id"])
                group["source_refs"].append(
                    {
                        "paper_id": paper_id,
                        "target_type": "limitation",
                        "target_id": limitation["id"],
                    }
                )
                group["occurrence_count"] += 1
        return [
            self._sorted_occurrence_item(item)
            for item in sorted(grouped.values(), key=lambda value: value["label"].lower())
        ]

    def _validation_norm_patterns(self, snapshots: list[dict[str, Any]]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for snapshot in snapshots:
            paper_id = snapshot["paper"]["id"]
            for row in snapshot["result_rows"]:
                if not (row["metric"] or row["dataset"] or row["method"]):
                    continue
                items.append(
                    {
                        "pattern_type": "validation_norm",
                        "label": self._validation_norm_label(row),
                        "paper_ids": [paper_id],
                        "source_ids": [row["id"]],
                        "source_refs": [
                            {
                                "paper_id": paper_id,
                                "target_type": "result_row",
                                "target_id": row["id"],
                            }
                        ],
                        "occurrence_count": 1,
                        "dataset": row["dataset"],
                        "method": row["method"],
                        "metric": row["metric"],
                        "split_name": row["split_name"],
                        "value": self._result_value(row),
                    }
                )
        return sorted(items, key=lambda item: item["label"].lower())

    def _paper_evidence_summary(self, snapshot: dict[str, Any]) -> str:
        paper = snapshot["paper"]
        fragments = [paper["title"]]
        if paper["abstract"]:
            fragments.append(self._bounded_text(paper["abstract"], limit=220))
        if snapshot["sections"]:
            section_titles = ", ".join(section["title"] for section in snapshot["sections"][:3])
            fragments.append(f"Parsed sections: {section_titles}.")
        structured_counts = self._structured_counts(snapshot)
        if structured_counts:
            fragments.append(
                "Structured evidence: "
                + ", ".join(
                    f"{count} {name.replace('_', ' ')}"
                    for name, count in structured_counts.items()
                )
                + "."
            )
        return self._bounded_text(" ".join(fragments), limit=700)

    def _structured_counts(self, snapshot: dict[str, Any]) -> dict[str, int]:
        counts = {
            "datasets": len(snapshot["datasets"]),
            "methods": len(snapshot["methods"]),
            "metrics": len(snapshot["metrics"]),
            "findings": len(snapshot["findings"]),
            "limitations": len(snapshot["limitations"]),
            "research_design_elements": len(snapshot["research_design_elements"]),
            "result_rows": len(snapshot["result_rows"]),
        }
        return {name: count for name, count in counts.items() if count}

    def _source_refs_for_paper(self, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
        paper_id = snapshot["paper"]["id"]
        refs = [{"paper_id": paper_id, "target_type": "paper", "target_id": paper_id}]
        for key, target_type in (
            ("sections", "section"),
            ("datasets", "dataset"),
            ("methods", "method"),
            ("metrics", "metric"),
            ("findings", "finding"),
            ("limitations", "limitation"),
            ("research_design_elements", "research_design_element"),
            ("result_rows", "result_row"),
        ):
            refs.extend(
                {"paper_id": paper_id, "target_type": target_type, "target_id": item["id"]}
                for item in snapshot[key]
            )
        return refs

    def _pattern_source_refs(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        refs: list[dict[str, Any]] = []
        for item in items:
            refs.extend(dict(source_ref) for source_ref in item.get("source_refs", []))
        return refs

    def _pattern_summary(self, *, pattern_type: str, items: list[dict[str, Any]]) -> str:
        labels = ", ".join(item["label"] for item in items[:5])
        more = "" if len(items) <= 5 else f", plus {len(items) - 5} more"
        return f"Observed {len(items)} {pattern_type.replace('_', ' ')} pattern(s): {labels}{more}."

    def _readiness_warnings(self, snapshots: list[dict[str, Any]]) -> list[str]:
        warnings: list[str] = []
        if not snapshots:
            return ["No collection papers were available for memory building."]
        if not any(snapshot["sections"] for snapshot in snapshots):
            warnings.append("No parsed sections were available for memory building.")
        if not any(self._structured_counts(snapshot) for snapshot in snapshots):
            warnings.append("No structured extraction records were available for memory building.")
        return warnings

    def _sorted_occurrence_item(self, item: dict[str, Any]) -> dict[str, Any]:
        item["paper_ids"] = sorted(item["paper_ids"])
        item["source_ids"] = sorted(item["source_ids"])
        item["source_refs"] = sorted(
            item.get("source_refs", []),
            key=lambda source_ref: (
                source_ref["paper_id"],
                source_ref["target_type"],
                source_ref["target_id"],
            ),
        )
        return item

    def _result_value(self, row: dict[str, Any]) -> float | str | None:
        return row["value_numeric"] if row["value_numeric"] is not None else row["value_text"]

    def _validation_norm_label(self, row: dict[str, Any]) -> str:
        parts = [
            part
            for part in (
                row["method"],
                row["dataset"],
                row["metric"],
                row["split_name"],
            )
            if part
        ]
        return " / ".join(parts) if parts else "Result row validation"

    def _stable_key(self, *parts: str) -> str:
        raw_key = ":".join(self._normalize_label(part) for part in parts if part)
        if len(raw_key) <= 240:
            return raw_key
        digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]
        return f"{raw_key[:223].rstrip(':')}:{digest}"

    def _normalize_label(self, value: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
        return normalized.strip("-") or "unknown"

    def _baseline_normalized_name(self, value: str) -> str | None:
        cleaned = self._clean_text(value)
        if not cleaned or re.search(r"[a-z0-9]", cleaned.lower()) is None:
            return None
        return self._normalize_label(cleaned)

    def _clean_text(self, value: str) -> str:
        return " ".join(value.strip().split())

    def _bounded_text(self, value: str, *, limit: int) -> str:
        cleaned = self._clean_text(value)
        if len(cleaned) <= limit:
            return cleaned
        return f"{cleaned[:limit].rstrip()}..."
