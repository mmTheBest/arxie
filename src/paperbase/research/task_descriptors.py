"""Shared task descriptors for research-agent context and validation needs."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True, slots=True)
class ResearchTaskDescriptor:
    """Deterministic description of what a research task needs from context."""

    task_type: str
    required_context_types: frozenset[str]
    readiness_requirements: frozenset[str]
    required_output_fields: frozenset[str]
    allowed_internal_tools: frozenset[str]
    context_feature_weights: Mapping[str, float]


_COMMON_TOOLS = frozenset(
    {
        "search_collection_evidence",
        "read_structured_extractions",
        "read_research_memory",
        "read_field_graph",
        "read_study_brief",
    }
)


TASK_DESCRIPTORS: Mapping[str, ResearchTaskDescriptor] = MappingProxyType(
    {
        "experiment_planning": ResearchTaskDescriptor(
            task_type="experiment_planning",
            required_context_types=frozenset(
                {
                    "methods",
                    "datasets",
                    "metrics",
                    "results",
                    "ablations",
                    "benchmark_papers",
                    "pinned_papers",
                }
            ),
            readiness_requirements=frozenset({"parsed_text", "structured_extraction"}),
            required_output_fields=frozenset(
                {"experiment_plan", "baselines", "ablations", "metrics", "evidence_references"}
            ),
            allowed_internal_tools=_COMMON_TOOLS,
            context_feature_weights=MappingProxyType(
                {
                    "selected": 4.0,
                    "pinned": 3.5,
                    "methods": 3.0,
                    "datasets": 2.25,
                    "metrics": 2.25,
                    "results": 4.0,
                    "ablations": 3.25,
                    "benchmark_signal": 2.5,
                    "baselines": 2.0,
                    "direct_evidence": 1.0,
                }
            ),
        ),
        "benchmark_planning": ResearchTaskDescriptor(
            task_type="benchmark_planning",
            required_context_types=frozenset(
                {"results", "metrics", "datasets", "baselines", "comparison_sections"}
            ),
            readiness_requirements=frozenset({"parsed_text", "structured_extraction"}),
            required_output_fields=frozenset(
                {"benchmark_plan", "datasets", "metrics", "baselines", "evidence_references"}
            ),
            allowed_internal_tools=_COMMON_TOOLS,
            context_feature_weights=MappingProxyType(
                {
                    "selected": 2.5,
                    "pinned": 2.5,
                    "results": 5.0,
                    "metrics": 4.0,
                    "datasets": 3.5,
                    "baselines": 3.0,
                    "benchmark_signal": 3.0,
                    "comparison_sections": 3.0,
                    "methods": 2.0,
                    "direct_evidence": 1.0,
                }
            ),
        ),
        "revision_planning": ResearchTaskDescriptor(
            task_type="revision_planning",
            required_context_types=frozenset(
                {
                    "user_draft_sources",
                    "evidence_references",
                    "contradictions",
                    "limitations",
                    "study_brief",
                }
            ),
            readiness_requirements=frozenset({"source_context", "parsed_text"}),
            required_output_fields=frozenset(
                {"revision_plan", "claim_fixes", "missing_citations", "evidence_references"}
            ),
            allowed_internal_tools=_COMMON_TOOLS,
            context_feature_weights=MappingProxyType(
                {
                    "draft_source": 6.0,
                    "selected": 4.0,
                    "pinned": 4.0,
                    "limitations": 3.5,
                    "contradictions": 3.0,
                    "evidence_references": 2.5,
                    "source_claims": 2.25,
                    "findings": 2.0,
                    "direct_evidence": 1.0,
                }
            ),
        ),
        "literature_review": ResearchTaskDescriptor(
            task_type="literature_review",
            required_context_types=frozenset(
                {"themes", "findings", "limitations", "broad_evidence_coverage"}
            ),
            readiness_requirements=frozenset({"parsed_text"}),
            required_output_fields=frozenset(
                {"themes", "gaps", "future_directions", "evidence_references"}
            ),
            allowed_internal_tools=_COMMON_TOOLS,
            context_feature_weights=MappingProxyType(
                {
                    "selected": 2.0,
                    "pinned": 2.0,
                    "findings": 4.0,
                    "limitations": 3.0,
                    "sections": 2.0,
                    "themes": 2.0,
                    "broad_evidence": 2.0,
                    "direct_evidence": 1.0,
                }
            ),
        ),
        "quality_harness": ResearchTaskDescriptor(
            task_type="quality_harness",
            required_context_types=frozenset(
                {"direct_evidence", "source_claims", "validation_warnings", "evidence_references"}
            ),
            readiness_requirements=frozenset({"parsed_text", "evidence_references"}),
            required_output_fields=frozenset(
                {
                    "quality_report",
                    "unsupported_claims",
                    "readiness_blockers",
                    "evidence_references",
                }
            ),
            allowed_internal_tools=_COMMON_TOOLS,
            context_feature_weights=MappingProxyType(
                {
                    "selected": 2.5,
                    "pinned": 2.5,
                    "direct_evidence": 4.0,
                    "evidence_references": 3.5,
                    "source_claims": 3.0,
                    "validation_warnings": 3.0,
                    "sections": 2.0,
                    "results": 2.0,
                    "limitations": 2.0,
                }
            ),
        ),
    }
)


def task_descriptor_for(task_type: str) -> ResearchTaskDescriptor:
    """Return the descriptor for a known task, falling back to literature review."""

    return TASK_DESCRIPTORS.get(task_type, TASK_DESCRIPTORS["literature_review"])
