"""Deterministic study-agent tool registry."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ra.study.models import StudyContextPack, StudyToolCategory

_HINT_RE = re.compile(
    r"\b("
    r"ablation|baseline|benchmark|dataset|datasets|metric|metrics|"
    r"accuracy|auc|f1|faithfulness|reranking|retrieval|hotpotqa|"
    r"limitation|claim|improve|improves|improved"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class StudyTool:
    """Tool contract exposed to the deterministic study-agent runtime."""

    name: str
    description: str
    category: StudyToolCategory
    side_effect: bool
    readiness_requirements: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StudyToolResult:
    """Structured result from a deterministic study tool call."""

    tool_name: str
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


class StudyToolRegistry:
    """Registry of deterministic, readiness-aware study tools."""

    def __init__(self) -> None:
        self._tools: dict[str, tuple[StudyTool, Callable[[StudyContextPack], StudyToolResult]]] = {}
        self.register(
            StudyTool(
                name="inspect_study_brief",
                description="Inspect durable study brief fields.",
                category=StudyToolCategory.READ,
                side_effect=False,
            ),
            _inspect_study_brief,
        )
        self.register(
            StudyTool(
                name="inspect_study_sources",
                description="Inspect attached user-provided study sources.",
                category=StudyToolCategory.READ,
                side_effect=False,
            ),
            _inspect_study_sources,
        )
        self.register(
            StudyTool(
                name="extract_benchmark_hints",
                description="Extract benchmark, dataset, metric, baseline, and ablation hints.",
                category=StudyToolCategory.READ,
                side_effect=False,
                readiness_requirements=("paper_metadata_or_user_source",),
            ),
            _extract_benchmark_hints,
        )
        self.register(
            StudyTool(
                name="check_draft_claims",
                description="Identify draft-like claims that need paper evidence.",
                category=StudyToolCategory.READ,
                side_effect=False,
                readiness_requirements=("user_source",),
            ),
            _check_draft_claims,
        )

    def register(
        self,
        tool: StudyTool,
        handler: Callable[[StudyContextPack], StudyToolResult],
    ) -> None:
        self._tools[tool.name] = (tool, handler)

    def get(self, name: str) -> StudyTool:
        return self._tools[name][0]

    def call(self, name: str, context: StudyContextPack) -> StudyToolResult:
        return self._tools[name][1](context)


def default_study_tool_registry() -> StudyToolRegistry:
    return StudyToolRegistry()


def _inspect_study_brief(context: StudyContextPack) -> StudyToolResult:
    return StudyToolResult(
        tool_name="inspect_study_brief",
        summary="Study brief inspected.",
        data={"fields": dict(context.brief_fields)},
    )


def _inspect_study_sources(context: StudyContextPack) -> StudyToolResult:
    return StudyToolResult(
        tool_name="inspect_study_sources",
        summary=f"{len(context.source_refs)} user source(s) inspected.",
        data={
            "sources": [
                {
                    "source_id": source.source_id,
                    "title": source.source_title,
                    "type": source.source_type.value,
                    "summary": source.summary,
                }
                for source in context.source_refs
            ]
        },
        warnings=tuple(
            message
            for message in context.missing_context
            if message.startswith("No user sources")
        ),
    )


def _extract_benchmark_hints(context: StudyContextPack) -> StudyToolResult:
    hints: set[str] = set()
    for paper in context.paper_refs:
        hints.update(_hints(f"{paper.title} {paper.abstract or ''}"))
    for source in context.source_refs:
        hints.update(_hints(source.summary))
    for value in context.brief_fields.values():
        hints.update(_hints(value))
    return StudyToolResult(
        tool_name="extract_benchmark_hints",
        summary=f"{len(hints)} benchmark or experiment hint(s) identified.",
        data={"hints": sorted(hints)},
        warnings=tuple(context.missing_context),
    )


def _check_draft_claims(context: StudyContextPack) -> StudyToolResult:
    claims: list[dict[str, str]] = []
    for source in context.source_refs:
        if source.source_type.value not in {"draft", "note"}:
            continue
        summary = source.summary
        if _looks_like_claim(summary):
            claims.append(
                {
                    "source_id": source.source_id,
                    "source_title": source.source_title,
                    "claim": summary,
                    "status": "needs_literature_support"
                    if not context.paper_refs
                    else "candidate_support_available",
                }
            )
    return StudyToolResult(
        tool_name="check_draft_claims",
        summary=f"{len(claims)} draft claim(s) inspected.",
        data={"claims": claims},
        warnings=tuple(context.missing_context),
    )


def _hints(text: str) -> set[str]:
    return {match.group(1).lower() for match in _HINT_RE.finditer(str(text or ""))}


def _looks_like_claim(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(marker in lowered for marker in ("claim", "improve", "outperform", "show", "argue"))
