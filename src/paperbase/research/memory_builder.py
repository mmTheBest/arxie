"""Deterministic derived-memory builder for collection research intelligence."""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
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
)
from paperbase.db.repositories import ResearchIntelligenceRepository


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


_BUILDER_EDGE_TYPES = {
    "uses_method",
    "mentions_dataset",
    "reports_metric",
    "validated_by",
    "supports",
    "leaves_gap",
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
            self._write_finding_and_gap_nodes(
                collection_id=collection_id,
                workspace_id=workspace_id,
                snapshots=snapshots,
                paper_nodes=paper_nodes,
                generated=generated,
            )
            self._write_result_edges(
                collection_id=collection_id,
                workspace_id=workspace_id,
                snapshots=snapshots,
                paper_nodes=paper_nodes,
                entity_nodes=entity_nodes,
                generated=generated,
            )
            self._write_pattern_memory(
                collection_id=collection_id,
                workspace_id=workspace_id,
                snapshots=snapshots,
                generated=generated,
            )
            self._write_study_constraint_nodes(
                collection_id=collection_id,
                workspace_id=workspace_id,
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
        ]
        if workspace_id is not None:
            prefixes.append(
                f"{self._stable_key('collection', collection_id, 'workspace', workspace_id)}:"
            )
        return node.stable_key.startswith(tuple(prefixes))

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

    def _write_result_edges(
        self,
        *,
        collection_id: str,
        workspace_id: str | None,
        snapshots: list[dict[str, Any]],
        paper_nodes: dict[str, str],
        entity_nodes: dict[tuple[str, str], str],
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
                label = self._clean_text(str(constraint.get("title") or constraint.get("text") or ""))
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
                normalized_name = item["normalized_name"] or self._normalize_label(item["display_name"])
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

    def _clean_text(self, value: str) -> str:
        return " ".join(value.strip().split())

    def _bounded_text(self, value: str, *, limit: int) -> str:
        cleaned = self._clean_text(value)
        if len(cleaned) <= limit:
            return cleaned
        return f"{cleaned[:limit].rstrip()}..."
