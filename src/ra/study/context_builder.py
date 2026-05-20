"""Task-aware context packing for study-agent runs."""

from __future__ import annotations

import re
from uuid import uuid4

from ra.retrieval.unified import Paper
from ra.study.brief_service import StudyBriefService
from ra.study.models import (
    StudyContextPack,
    StudyPaperContextRef,
    StudySourceContextRef,
    StudyTaskType,
)

_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")
_TASK_TERMS: dict[StudyTaskType, tuple[str, ...]] = {
    StudyTaskType.DESIGN_EXPERIMENTS: (
        "experiment",
        "ablation",
        "baseline",
        "dataset",
        "metric",
        "benchmark",
        "evaluation",
        "method",
    ),
    StudyTaskType.FIND_BENCHMARKS: (
        "benchmark",
        "dataset",
        "metric",
        "baseline",
        "leaderboard",
        "evaluation",
    ),
    StudyTaskType.REVIEW_DRAFT_CLAIMS: (
        "claim",
        "support",
        "evidence",
        "contradict",
        "limitation",
        "draft",
    ),
}


class StudyContextBuilder:
    """Build explicit bounded context for a study task."""

    def __init__(self, *, service: StudyBriefService, max_papers: int = 8) -> None:
        self.service = service
        self.max_papers = max(1, int(max_papers))

    def build_context(
        self,
        *,
        study_id: str,
        task_type: StudyTaskType | str,
        query: str,
        papers: list[Paper] | tuple[Paper, ...] | None = None,
    ) -> StudyContextPack:
        task = task_type if isinstance(task_type, StudyTaskType) else StudyTaskType(str(task_type))
        brief = self.service.get_brief(study_id)
        sources = self.service.list_sources(study_id)
        paper_refs = self._paper_refs(task, query, papers or [])
        source_refs = tuple(
            StudySourceContextRef(
                source_id=source.source_id,
                source_title=source.title,
                source_type=source.source_type,
                summary=source.summary,
                version=source.version,
            )
            for source in sources
        )

        brief_fields = {
            key: value
            for key, value in {
                "title": brief.title,
                "research_goal": brief.research_goal,
                "collection_id": brief.collection_id,
                "domain": brief.domain,
                "current_method": brief.current_method,
                "datasets": ", ".join(brief.datasets),
                "metrics": ", ".join(brief.metrics),
                "constraints": "; ".join(brief.constraints),
                "decisions": "; ".join(brief.decisions),
                "risks": "; ".join(brief.risks),
                "open_questions": "; ".join(brief.open_questions),
            }.items()
            if value
        }

        missing_context: list[str] = []
        if not paper_refs:
            missing_context.append("No paper metadata was provided for literature grounding.")
        if not source_refs:
            missing_context.append("No user sources are attached to this study.")

        reasons = {
            "brief": "Study brief fields provide the user's goal, constraints, and current method.",
        }
        for source in source_refs:
            reasons[f"source:{source.source_id}"] = (
                f"{source.source_type.value} source attached to the active study."
            )
        for paper in paper_refs:
            reasons[f"paper:{paper.paper_id}"] = (
                "Paper metadata matched task/query terms for this study run."
            )

        return StudyContextPack(
            context_pack_id=f"ctx-{uuid4().hex[:12]}",
            study_id=brief.study_id,
            task_type=task,
            query=str(query or "").strip(),
            brief_fields=brief_fields,
            source_refs=source_refs,
            paper_refs=paper_refs,
            selection_reasons=reasons,
            missing_context=tuple(missing_context),
            source_versions={
                "brief": brief.version,
                **{source.source_id: source.version for source in source_refs},
            },
        )

    def _paper_refs(
        self,
        task_type: StudyTaskType,
        query: str,
        papers: list[Paper] | tuple[Paper, ...],
    ) -> tuple[StudyPaperContextRef, ...]:
        query_tokens = set(_tokens(query))
        task_tokens = set(_TASK_TERMS[task_type])
        scored: list[StudyPaperContextRef] = []
        for paper in papers:
            paper_id = str(getattr(paper, "id", "") or "").strip()
            title = str(getattr(paper, "title", "") or "").strip()
            if not paper_id or not title:
                continue
            abstract = getattr(paper, "abstract", None)
            haystack_tokens = set(_tokens(f"{title} {abstract or ''}"))
            overlap = len(haystack_tokens & (query_tokens | task_tokens))
            relevance = min(1.0, overlap / max(1, len(task_tokens)))
            scored.append(
                StudyPaperContextRef(
                    paper_id=paper_id,
                    title=title,
                    abstract=str(abstract) if abstract else None,
                    year=getattr(paper, "year", None),
                    relevance_score=round(relevance, 6),
                    source=str(getattr(paper, "source", "semantic_scholar")),
                )
            )
        return tuple(
            sorted(scored, key=lambda item: (-item.relevance_score, item.title.lower()))[
                : self.max_papers
            ]
        )


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text or "").lower())
