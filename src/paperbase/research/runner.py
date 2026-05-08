"""Deterministic local research-agent runner backed by Paperbase evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from paperbase.db.models import (
    CollectionPaper,
    Dataset,
    EngineeringTrick,
    Limitation,
    Method,
    Metric,
    Paper,
    ResearchDesignElement,
    ResearchArtifact,
    ResearchMessage,
    ResultRow,
    Section,
    StudySource,
)
from paperbase.db.repositories import ResearchRepository


@dataclass(frozen=True, slots=True)
class ResearchAgentRunResult:
    collection_id: str
    thread_id: str
    artifact_id: str
    artifact_type: str
    evidence_paper_count: int


class PaperbaseResearchAgentRunner:
    """Synthesize research artifacts from a prepared local paper collection."""

    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        model_name: str = "local-research-agent",
        prompt_version: str = "research-agent-v1",
    ) -> None:
        self.session_factory = session_factory
        self.model_name = model_name
        self.prompt_version = prompt_version

    def run(self, payload: dict[str, Any]) -> ResearchAgentRunResult:
        thread_id = str(payload["thread_id"])
        artifact_id = str(payload["artifact_id"])
        collection_id = str(payload["collection_id"])
        user_message = str(payload.get("user_message") or "")
        artifact_type = self._resolve_artifact_type(
            requested=payload.get("artifact_type"),
            message=user_message,
        )
        workspace_id = str(payload["workspace_id"]) if payload.get("workspace_id") else None
        selected_paper_ids = [str(item) for item in list(payload.get("selected_paper_ids") or [])]
        source_ids = [str(item) for item in list(payload.get("source_ids") or [])]

        with self.session_factory() as session:
            artifact = session.get(ResearchArtifact, artifact_id)
            if artifact is None:
                raise ValueError(f"No research artifact found for id: {artifact_id}")

            evidence_payload = self._build_evidence_payload(
                session,
                collection_id=collection_id,
                selected_paper_ids=selected_paper_ids,
                workspace_id=workspace_id,
                source_ids=source_ids,
            )
            output_payload = self._build_output_payload(
                artifact_type=artifact_type,
                user_message=user_message,
                evidence_payload=evidence_payload,
            )

            repository = ResearchRepository(session)
            repository.update_artifact(
                artifact_id,
                title=output_payload["title"],
                status="completed",
                output_payload=output_payload,
                evidence_payload=evidence_payload,
                model_name=self.model_name,
                prompt_version=self.prompt_version,
                error_message=None,
            )
            repository.create_message(
                thread_id=thread_id,
                role="assistant",
                content=self._assistant_summary(output_payload),
                artifact_id=artifact_id,
                metadata={"artifact_type": artifact_type},
            )

        return ResearchAgentRunResult(
            collection_id=collection_id,
            thread_id=thread_id,
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            evidence_paper_count=len(evidence_payload["papers"]),
        )

    def mark_artifact_failed(self, *, artifact_id: str, error_message: str) -> None:
        with self.session_factory() as session:
            repository = ResearchRepository(session)
            if repository.get_artifact(artifact_id) is None:
                return
            repository.update_artifact(
                artifact_id,
                status="failed",
                error_message=error_message,
            )

    def _build_evidence_payload(
        self,
        session: Session,
        *,
        collection_id: str,
        selected_paper_ids: list[str],
        workspace_id: str | None,
        source_ids: list[str],
    ) -> dict[str, Any]:
        paper_ids = selected_paper_ids or list(
            session.execute(
                select(CollectionPaper.paper_id)
                .where(CollectionPaper.collection_id == collection_id)
                .order_by(CollectionPaper.position.asc(), CollectionPaper.created_at.asc())
            ).scalars()
        )
        papers = {
            paper.id: paper
            for paper in session.execute(select(Paper).where(Paper.id.in_(paper_ids))).scalars().all()
        }
        paper_summaries: list[dict[str, Any]] = []
        for paper_id in paper_ids:
            paper = papers.get(paper_id)
            if paper is None:
                continue
            section_snippets = [
                {"title": section.title, "text": section.text[:600]}
                for section in session.execute(
                    select(Section)
                    .where(Section.paper_id == paper_id)
                    .order_by(Section.ordinal.asc())
                    .limit(3)
                ).scalars()
            ]
            paper_summaries.append(
                {
                    "paper_id": paper.id,
                    "title": paper.canonical_title,
                    "sections": section_snippets,
                    "datasets": [
                        item.display_name
                        for item in session.execute(
                            select(Dataset)
                            .where(Dataset.paper_id == paper_id)
                            .order_by(Dataset.display_name.asc())
                        ).scalars()
                    ],
                    "methods": [
                        item.display_name
                        for item in session.execute(
                            select(Method)
                            .where(Method.paper_id == paper_id)
                            .order_by(Method.display_name.asc())
                        ).scalars()
                    ],
                    "metrics": [
                        item.display_name
                        for item in session.execute(
                            select(Metric)
                            .where(Metric.paper_id == paper_id)
                            .order_by(Metric.display_name.asc())
                        ).scalars()
                    ],
                    "limitations": [
                        item.statement
                        for item in session.execute(
                            select(Limitation)
                            .where(Limitation.paper_id == paper_id)
                            .order_by(Limitation.created_at.asc())
                            .limit(5)
                        ).scalars()
                    ],
                    "engineering_tricks": [
                        item.title
                        for item in session.execute(
                            select(EngineeringTrick)
                            .where(EngineeringTrick.paper_id == paper_id)
                            .order_by(EngineeringTrick.title.asc())
                            .limit(5)
                        ).scalars()
                    ],
                    "research_design_elements": [
                        {
                            "element_type": item.element_type,
                            "title": item.title,
                            "description": item.description,
                            "metadata": dict(item.metadata_json or {}),
                        }
                        for item in session.execute(
                            select(ResearchDesignElement)
                            .where(ResearchDesignElement.paper_id == paper_id)
                            .order_by(
                                ResearchDesignElement.element_type.asc(),
                                ResearchDesignElement.created_at.asc(),
                            )
                            .limit(12)
                        ).scalars()
                    ],
                    "results": self._result_rows_for_paper(session, paper_id=paper_id),
                }
            )
        return {
            "collection_id": collection_id,
            "papers": paper_summaries,
            "sources": self._study_sources_for_run(
                session,
                workspace_id=workspace_id,
                source_ids=source_ids,
            ),
        }

    def _study_sources_for_run(
        self,
        session: Session,
        *,
        workspace_id: str | None,
        source_ids: list[str],
    ) -> list[dict[str, Any]]:
        if workspace_id is None or not source_ids:
            return []
        sources = {
            source.id: source
            for source in session.execute(
                select(StudySource)
                .where(
                    StudySource.workspace_id == workspace_id,
                    StudySource.id.in_(source_ids),
                )
                .order_by(StudySource.created_at.asc(), StudySource.id.asc())
            ).scalars()
        }
        source_summaries: list[dict[str, Any]] = []
        for source_id in source_ids:
            source = sources.get(source_id)
            if source is None:
                continue
            source_summaries.append(
                {
                    "source_id": source.id,
                    "source_type": source.source_type,
                    "title": source.title,
                    "path": source.path,
                    "summary": source.summary or self._summarize_text(source.content or ""),
                    "read_status": source.read_status,
                    "error_message": source.error_message,
                }
            )
        return source_summaries

    def _result_rows_for_paper(self, session: Session, *, paper_id: str) -> list[dict[str, Any]]:
        rows = session.execute(
            select(ResultRow, Dataset, Method, Metric)
            .outerjoin(Dataset, Dataset.id == ResultRow.dataset_id)
            .outerjoin(Method, Method.id == ResultRow.method_id)
            .outerjoin(Metric, Metric.id == ResultRow.metric_id)
            .where(ResultRow.paper_id == paper_id)
            .order_by(ResultRow.value_numeric.desc().nullslast(), ResultRow.created_at.asc())
            .limit(10)
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

    def _build_output_payload(
        self,
        *,
        artifact_type: str,
        user_message: str,
        evidence_payload: dict[str, Any],
    ) -> dict[str, Any]:
        papers = list(evidence_payload.get("papers") or [])
        sources = [
            source
            for source in list(evidence_payload.get("sources") or [])
            if isinstance(source, dict)
        ]
        methods = self._unique(item for paper in papers for item in paper.get("methods", []))
        datasets = self._unique(item for paper in papers for item in paper.get("datasets", []))
        limitations = self._unique(item for paper in papers for item in paper.get("limitations", []))
        design_elements = [
            item
            for paper in papers
            for item in paper.get("research_design_elements", [])
            if isinstance(item, dict)
        ]
        result_notes = self._unique(
            item.get("notes")
            for paper in papers
            for item in paper.get("results", [])
            if item.get("notes")
        )
        baselines = [
            method
            for method in methods
            if "baseline" in method.casefold() or "standard" in method.casefold()
        ] or ["Compare against the strongest method reported in each exemplar paper."]

        common = {
            "artifact_type": artifact_type,
            "request": user_message,
            "paper_count": len(papers),
            "evidence_basis": [paper["title"] for paper in papers],
            "source_context": [
                {
                    "title": str(source.get("title") or "Study source"),
                    "source_type": str(source.get("source_type") or "text"),
                    "summary": str(source.get("summary") or ""),
                }
                for source in sources
            ],
            "general_methodology": [
                "Separate claims, experimental variables, and evaluation criteria before writing new experiments.",
                "Use paper-level evidence as constraints, then identify where a new study can challenge one assumption.",
            ],
        }
        if artifact_type == "benchmark_plan":
            return {
                **common,
                "title": "Benchmark plan",
                "benchmark_recommendations": [
                    "Reuse the strongest recurring dataset and metric from the collection as the primary benchmark.",
                    "Add at least one stress test that targets a weakness visible in the current study context.",
                    "Report matched baselines under identical splits before adding new model variants.",
                ],
                "datasets": datasets[:8],
                "metrics_or_result_logic": result_notes[:5]
                or ["Tie each benchmark to a measurable claim and a prior-paper comparison point."],
                "source_gaps": [
                    source["summary"]
                    for source in common["source_context"]
                    if source.get("summary")
                ][:5],
            }
        if artifact_type == "revision_plan":
            return {
                **common,
                "title": "Revision plan",
                "revision_priorities": [
                    "State the main claim as a falsifiable comparison against prior work.",
                    "Add controls or ablations for each extracted limitation that applies to the user's study.",
                    "Align evaluation tables with datasets, metrics, and baselines that recur across the collection.",
                ],
                "paper_backed_risks": limitations[:6],
                "source_context_risks": [
                    source["summary"]
                    for source in common["source_context"]
                    if source.get("summary")
                ][:5],
            }
        if artifact_type == "assumption_map":
            return {
                **common,
                "title": "Assumption map",
                "assumptions_to_challenge": [
                    "The main method advantage remains after controlling for preprocessing and splits.",
                    "The graph prior improves generalization rather than only fitting the benchmark distribution.",
                    "Reported gains are robust to ablations that remove one design choice at a time.",
                ],
                "validation_tests": [
                    "Map each assumption to a measurable comparison.",
                    "Require one paper-grounded baseline and one study-context check per assumption.",
                ],
            }
        if artifact_type == "experiment_backlog":
            return {
                **common,
                "title": "Experiment backlog",
                "backlog_items": [
                    "Reproduce the primary baseline on the collection's recurring benchmark.",
                    "Run an ablation for the strongest claimed design component.",
                    "Add a failure-case analysis tied to extracted limitations.",
                    "Compare against the best reported method using the same metric family.",
                ],
                "candidate_baselines": baselines,
                "datasets": datasets[:8],
            }
        if artifact_type == "hypotheses":
            return {
                **common,
                "title": "Collection-grounded hypotheses",
                "hypotheses": self._hypotheses(methods=methods, datasets=datasets, result_notes=result_notes),
                "validation_plan": [
                    "Define a measurable claim for each hypothesis.",
                    "Reuse collection datasets and metrics where possible to make comparisons interpretable.",
                ],
            }
        if artifact_type == "critique":
            return {
                **common,
                "title": "Study critique",
                "strengths": methods[:5] or ["The collection provides reusable methodological anchors."],
                "risks": limitations[:5] or ["Explicit validity threats were not extracted yet."],
                "revision_priorities": [
                    "Turn each limitation into a control, ablation, or exclusion criterion.",
                    "Add evidence-backed rationale wherever a design choice follows prior work.",
                ],
            }
        if artifact_type == "field_patterns":
            return {
                **common,
                "title": "Field patterns",
                "patterns": [
                    f"Common method family: {method}" for method in methods[:5]
                ] or ["The collection needs structured extraction before method patterns are reliable."],
                "datasets": datasets[:8],
                "reasoning_patterns": result_notes[:5],
            }
        return {
            **common,
            "title": "Experiment plan",
            "objective": "Design a study whose claims can be compared against the selected paper collection.",
            "baselines": baselines,
            "datasets": datasets[:8],
            "ablations": [
                *[
                    item["title"]
                    for item in design_elements
                    if item.get("element_type") == "ablation" and item.get("title")
                ][:3],
                "Remove one model assumption at a time and report the same metrics used by reference papers.",
                "Compare full model, simplified variant, and baseline under identical data splits.",
            ],
            "controls": [
                "Lock preprocessing and train/test splits before comparing methods.",
                "Report failure cases and sensitivity to key experimental variables.",
            ],
            "evaluation_protocol": [
                "Use the collection's recurring datasets and metrics as the primary benchmark frame.",
                "Add a validity-threat section that explicitly maps back to extracted limitations.",
            ],
            "reasoning_logic": result_notes[:5]
            or ["The design should make the challenged assumption observable through controlled comparisons."],
        }

    def _resolve_artifact_type(self, *, requested: object, message: str) -> str:
        valid_types = {
            "field_patterns",
            "hypotheses",
            "experiment_plan",
            "critique",
            "experiment_backlog",
            "benchmark_plan",
            "revision_plan",
            "assumption_map",
        }
        if requested in valid_types:
            return str(requested)
        normalized = message.casefold()
        if "hypothes" in normalized or "gap" in normalized:
            return "hypotheses"
        if "critique" in normalized or "weakness" in normalized or "review" in normalized:
            return "critique"
        if "benchmark" in normalized or "baseline" in normalized:
            return "benchmark_plan"
        if "revision" in normalized or "improve" in normalized:
            return "revision_plan"
        if "assumption" in normalized:
            return "assumption_map"
        if "backlog" in normalized or "next" in normalized:
            return "experiment_backlog"
        if "pattern" in normalized or "learn from" in normalized:
            return "field_patterns"
        return "experiment_plan"

    def _summarize_text(self, text: str) -> str:
        cleaned = " ".join(text.strip().split())
        if len(cleaned) <= 360:
            return cleaned
        return f"{cleaned[:360].rstrip()}..."

    def _assistant_summary(self, output_payload: dict[str, Any]) -> str:
        title = str(output_payload.get("title") or "Research artifact")
        paper_count = int(output_payload.get("paper_count") or 0)
        return f"{title} generated from {paper_count} collection paper(s)."

    def _hypotheses(
        self,
        *,
        methods: list[str],
        datasets: list[str],
        result_notes: list[str],
    ) -> list[dict[str, str]]:
        method = methods[0] if methods else "the target method"
        dataset = datasets[0] if datasets else "the benchmark corpus"
        rationale = result_notes[0] if result_notes else "Prior results suggest a testable design assumption."
        return [
            {
                "claim": f"{method} improves reliability on {dataset} when the key design assumption is isolated.",
                "rationale": rationale,
                "test": "Run matched baselines and ablations under the same evaluation protocol.",
            }
        ]

    def _unique(self, values) -> list[str]:  # noqa: ANN001
        seen: set[str] = set()
        cleaned: list[str] = []
        for value in values:
            if not isinstance(value, str):
                continue
            item = value.strip()
            if not item or item.casefold() in seen:
                continue
            seen.add(item.casefold())
            cleaned.append(item)
        return cleaned
