"""Task-aware context assembly for Paperbase research-agent runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from paperbase.db.models import (
    CollectionPaper,
    Dataset,
    EngineeringTrick,
    Limitation,
    Method,
    Metric,
    Paper,
    ResearchDesignElement,
    ResultRow,
    Section,
    StudySource,
    Workspace,
)

DEFAULT_CONTEXT_PAPER_LIMIT = 40


@dataclass(frozen=True, slots=True)
class ResearchContextPack:
    context: dict[str, Any]
    selected_item_counts: dict[str, int]
    readiness_warnings: list[str]
    cache_key: str


class PaperbaseResearchContextBuilder:
    """Build bounded agent context from papers, structured evidence, and user sources."""

    def __init__(self, session: Session) -> None:
        self.session = session

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
        default_paper_ids = [] if selected_paper_ids else self._default_collection_paper_ids(collection_id)
        paper_ids = selected_paper_ids or default_paper_ids
        role_by_paper_id = self._paper_roles(
            selected_paper_ids=selected_paper_ids,
            paper_ids=paper_ids,
            pinned_paper_ids=pinned_paper_ids,
        )
        for pinned_paper_id in pinned_paper_ids:
            if pinned_paper_id not in paper_ids:
                paper_ids.append(pinned_paper_id)
                role_by_paper_id[pinned_paper_id] = "pinned_context"
        papers = self._paper_summaries(paper_ids, role_by_paper_id=role_by_paper_id)
        sources = self._study_sources(workspace_id=workspace_id, source_ids=source_ids)
        readiness_warnings = self._readiness_warnings(papers=papers, source_ids=source_ids, sources=sources)
        context = {
            "collection_id": collection_id,
            "task_type": task_type,
            "message": message,
            "papers": papers,
            "sources": sources,
        }
        if study:
            context["study"] = study
        return ResearchContextPack(
            context=context,
            selected_item_counts={
                "papers": len(papers),
                "sources": len(sources),
                "study": 1 if study else 0,
                "sections": sum(len(paper.get("sections", [])) for paper in papers),
                "structured_evidence": sum(self._structured_signal_count(paper) for paper in papers),
            },
            readiness_warnings=readiness_warnings,
            cache_key=self._cache_key(
                collection_id=collection_id,
                task_type=task_type,
                paper_ids=paper_ids,
                source_ids=source_ids,
            ),
        )

    def _default_collection_paper_ids(self, collection_id: str) -> list[str]:
        return list(
            self.session.execute(
                select(CollectionPaper.paper_id)
                .where(CollectionPaper.collection_id == collection_id)
                .order_by(CollectionPaper.position.asc(), CollectionPaper.created_at.asc())
                .limit(DEFAULT_CONTEXT_PAPER_LIMIT)
            ).scalars()
        )

    def _paper_roles(
        self,
        *,
        selected_paper_ids: list[str],
        paper_ids: list[str],
        pinned_paper_ids: list[str],
    ) -> dict[str, str]:
        roles = {paper_id: "collection_default" for paper_id in paper_ids}
        for paper_id in pinned_paper_ids:
            if paper_id in roles:
                roles[paper_id] = "pinned_context"
        for paper_id in selected_paper_ids:
            roles[paper_id] = "selected"
        return roles

    def _paper_summaries(
        self,
        paper_ids: list[str],
        *,
        role_by_paper_id: dict[str, str],
    ) -> list[dict[str, Any]]:
        papers = {
            paper.id: paper
            for paper in self.session.execute(select(Paper).where(Paper.id.in_(paper_ids))).scalars().all()
        }
        summaries: list[dict[str, Any]] = []
        for paper_id in paper_ids:
            paper = papers.get(paper_id)
            if paper is None:
                continue
            summaries.append(
                {
                    "paper_id": paper.id,
                    "title": paper.canonical_title,
                    "context_role": role_by_paper_id.get(paper.id, "collection_default"),
                    "context_reason": self._context_reason(role_by_paper_id.get(paper.id, "collection_default")),
                    "abstract": paper.abstract,
                    "publication_year": paper.publication_year,
                    "sections": [
                        {"title": section.title, "text": section.text[:900]}
                        for section in self.session.execute(
                            select(Section)
                            .where(Section.paper_id == paper_id)
                            .order_by(Section.ordinal.asc())
                            .limit(4)
                        ).scalars()
                    ],
                    "datasets": self._named_items(Dataset, paper_id=paper_id),
                    "methods": self._named_items(Method, paper_id=paper_id),
                    "metrics": self._named_items(Metric, paper_id=paper_id),
                    "limitations": [
                        item.statement
                        for item in self.session.execute(
                            select(Limitation)
                            .where(Limitation.paper_id == paper_id)
                            .order_by(Limitation.created_at.asc())
                            .limit(8)
                        ).scalars()
                    ],
                    "engineering_tricks": [
                        item.title
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
                            "title": item.title,
                            "description": item.description,
                            "metadata": dict(item.metadata_json or {}),
                        }
                        for item in self.session.execute(
                            select(ResearchDesignElement)
                            .where(ResearchDesignElement.paper_id == paper_id)
                            .order_by(ResearchDesignElement.element_type.asc(), ResearchDesignElement.created_at.asc())
                            .limit(12)
                        ).scalars()
                    ],
                    "results": self._result_rows(paper_id),
                }
            )
        return summaries

    def _context_reason(self, role: str) -> str:
        return {
            "selected": "The user selected this paper for the research thread.",
            "pinned_context": "The paper is pinned in the active Study.",
            "collection_default": "The paper was included from the active collection scope.",
        }.get(role, "The paper was included as collection evidence.")

    def _named_items(self, model, *, paper_id: str) -> list[str]:  # noqa: ANN001
        return [
            item.display_name
            for item in self.session.execute(
                select(model)
                .where(model.paper_id == paper_id)
                .order_by(model.display_name.asc())
            ).scalars()
        ]

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
                "value_text": row.value_text,
                "notes": row.notes,
            }
            for row, dataset, method, metric in rows
        ]

    def _study_sources(self, *, workspace_id: str | None, source_ids: list[str]) -> list[dict[str, Any]]:
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
            summaries.append(
                {
                    "source_id": source.id,
                    "source_type": source.source_type,
                    "title": source.title,
                    "source_locator": "registered_path" if source.path else "inline_text",
                    "has_path": bool(source.path),
                    "summary": source.summary or self._summarize_text(source.content or ""),
                    "read_status": source.read_status,
                    "error_message": source.error_message,
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
        if papers and not any(self._structured_signal_count(paper) for paper in papers):
            warnings.append("Structured extraction is missing for the selected papers.")
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
        paper_ids: list[str],
        source_ids: list[str],
    ) -> str:
        paper_digest = self._id_list_digest(paper_ids)
        source_digest = self._id_list_digest(source_ids)
        return "|".join([collection_id, task_type, f"papers:{paper_digest}", f"sources:{source_digest}"])

    def _id_list_digest(self, ids: list[str]) -> str:
        payload = json.dumps(ids, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def _summarize_text(self, text: str) -> str:
        cleaned = " ".join(text.strip().split())
        if len(cleaned) <= 360:
            return cleaned
        return f"{cleaned[:360].rstrip()}..."
