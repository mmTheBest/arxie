"""Typed read-tool metadata for Paperbase Study-agent traces."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from paperbase.research.task_descriptors import task_descriptor_for


@dataclass(frozen=True, slots=True)
class StudyAgentToolSpec:
    """Static contract for a Study-agent tool surface."""

    name: str
    category: str
    description: str
    context_groups: tuple[str, ...]
    count_keys: tuple[str, ...]
    readiness_requirements: frozenset[str]
    side_effects: bool = False


_REQUIREMENT_COUNT_KEYS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "parsed_text": ("sections", "chunks"),
        "structured_extraction": (
            "structured_evidence",
            "structured_entities",
            "result_evidence",
            "figures",
            "tables",
        ),
        "source_context": ("sources", "source_fact_memory"),
        "evidence_references": (
            "evidence_spans",
            "structured_evidence",
            "result_evidence",
        ),
        "derived_memory": (
            "evidence_memory",
            "pattern_memory",
            "source_fact_memory",
        ),
        "field_graph": ("graph_nodes", "graph_edges"),
        "study_brief": ("study_brief",),
    }
)


class StudyAgentToolRegistry:
    """Evaluate task-scoped read-tool availability from a context summary."""

    def __init__(self, tools: tuple[StudyAgentToolSpec, ...]) -> None:
        self._tools = tools
        self._tool_by_name = {tool.name: tool for tool in tools}

    def registered_tool_names(self) -> frozenset[str]:
        """Return registered tool names for descriptor coverage tests."""

        return frozenset(tool.name for tool in self._tools)

    def trace_summary(
        self,
        *,
        task_type: str,
        selected_item_counts: Mapping[str, Any],
    ) -> dict[str, list[dict[str, Any]]]:
        descriptor = task_descriptor_for(task_type)
        available: list[dict[str, Any]] = []
        blocked: list[dict[str, Any]] = []
        for tool_name in descriptor.allowed_internal_tools:
            tool = self._tool_by_name.get(tool_name)
            if tool is None:
                continue
            missing_requirements = [
                requirement
                for requirement in sorted(tool.readiness_requirements)
                if not self._requirement_available(
                    requirement,
                    selected_item_counts=selected_item_counts,
                )
            ]
            item = self._trace_item(tool, selected_item_counts=selected_item_counts)
            if missing_requirements:
                item["missing_requirements"] = missing_requirements
                blocked.append(item)
            else:
                available.append(item)
        return {
            "available": sorted(available, key=lambda item: str(item["name"])),
            "blocked": sorted(blocked, key=lambda item: str(item["name"])),
        }

    def _trace_item(
        self,
        tool: StudyAgentToolSpec,
        *,
        selected_item_counts: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "name": tool.name,
            "category": tool.category,
            "side_effects": tool.side_effects,
            "context_groups": list(tool.context_groups),
            "readiness_requirements": sorted(tool.readiness_requirements),
            "selected_context_counts": {
                key: count
                for key in tool.count_keys
                if (count := self._count(selected_item_counts, key)) > 0
            },
        }

    def _requirement_available(
        self,
        requirement: str,
        *,
        selected_item_counts: Mapping[str, Any],
    ) -> bool:
        count_keys = _REQUIREMENT_COUNT_KEYS.get(requirement)
        if count_keys is None:
            return False
        return any(self._count(selected_item_counts, key) > 0 for key in count_keys)

    def _count(self, selected_item_counts: Mapping[str, Any], key: str) -> int:
        value = selected_item_counts.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
        return 0


def default_study_agent_tool_registry() -> StudyAgentToolRegistry:
    """Return the built-in Study-agent read-tool registry."""

    return StudyAgentToolRegistry(
        (
            StudyAgentToolSpec(
                name="search_collection_evidence",
                category="read",
                description="Read selected paper, chunk, and direct evidence-span context.",
                context_groups=("papers", "chunks", "evidence_spans"),
                count_keys=("papers", "chunks", "evidence_spans"),
                readiness_requirements=frozenset({"parsed_text"}),
            ),
            StudyAgentToolSpec(
                name="read_structured_extractions",
                category="read",
                description="Read extracted entities, result rows, figures, and tables.",
                context_groups=(
                    "structured_entities",
                    "result_evidence",
                    "figures",
                    "tables",
                    "evidence_spans",
                ),
                count_keys=(
                    "structured_entities",
                    "result_evidence",
                    "figures",
                    "tables",
                    "evidence_spans",
                ),
                readiness_requirements=frozenset({"structured_extraction"}),
            ),
            StudyAgentToolSpec(
                name="read_research_memory",
                category="read",
                description="Read derived evidence, pattern, and source-fact memory.",
                context_groups=(
                    "intelligence_layers.evidence_memory",
                    "intelligence_layers.pattern_memory",
                    "intelligence_layers.source_fact_memory",
                ),
                count_keys=(
                    "evidence_memory",
                    "pattern_memory",
                    "source_fact_memory",
                ),
                readiness_requirements=frozenset({"derived_memory"}),
            ),
            StudyAgentToolSpec(
                name="read_field_graph",
                category="read",
                description="Read selected field-graph nodes and edges.",
                context_groups=(
                    "intelligence_layers.field_graph.nodes",
                    "intelligence_layers.field_graph.edges",
                ),
                count_keys=("graph_nodes", "graph_edges"),
                readiness_requirements=frozenset({"field_graph"}),
            ),
            StudyAgentToolSpec(
                name="read_study_brief",
                category="read",
                description="Read the current reviewed Study Brief.",
                context_groups=("intelligence_layers.study_brief",),
                count_keys=("study_brief",),
                readiness_requirements=frozenset({"study_brief"}),
            ),
            StudyAgentToolSpec(
                name="read_study_sources",
                category="read",
                description="Read explicit user-provided Study sources and source facts.",
                context_groups=("sources", "intelligence_layers.source_fact_memory"),
                count_keys=("sources", "source_fact_memory"),
                readiness_requirements=frozenset({"source_context"}),
            ),
        )
    )
