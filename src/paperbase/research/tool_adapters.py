"""Executable bounded read-tool observations for Paperbase Study-agent traces."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from typing import Any

_TOOL_GROUPS: Mapping[str, tuple[str, ...]] = {
    "search_collection_evidence": ("papers", "chunks", "evidence_spans"),
    "read_structured_extractions": (
        "structured_entities",
        "result_evidence",
        "figures",
        "tables",
        "evidence_spans",
    ),
    "read_research_memory": (
        "intelligence_layers.evidence_memory",
        "intelligence_layers.pattern_memory",
        "intelligence_layers.source_fact_memory",
    ),
    "read_field_graph": (
        "intelligence_layers.field_graph.nodes",
        "intelligence_layers.field_graph.edges",
    ),
    "read_study_brief": ("intelligence_layers.study_brief",),
    "read_study_sources": (
        "sources",
        "intelligence_layers.source_fact_memory",
    ),
}

_COUNT_KEY_BY_GROUP: Mapping[str, str] = {
    "intelligence_layers.evidence_memory": "evidence_memory",
    "intelligence_layers.pattern_memory": "pattern_memory",
    "intelligence_layers.source_fact_memory": "source_fact_memory",
    "intelligence_layers.field_graph.nodes": "graph_nodes",
    "intelligence_layers.field_graph.edges": "graph_edges",
    "intelligence_layers.study_brief": "study_brief",
}

_SAFE_REF_KEYS_BY_COUNT_KEY: Mapping[str, tuple[str, ...]] = {
    "papers": ("paper_id", "title"),
    "chunks": ("chunk_id", "paper_id", "section_title", "context_role"),
    "evidence_spans": ("evidence_span_id", "paper_id", "target_type", "target_id"),
    "structured_entities": ("entity_id", "entity_type", "paper_id", "name", "display_name"),
    "result_evidence": (
        "result_row_id",
        "paper_id",
        "dataset_id",
        "method_id",
        "metric_id",
    ),
    "figures": ("figure_id", "paper_id", "label"),
    "tables": ("table_id", "paper_id", "label"),
    "evidence_memory": ("memory_record_id", "paper_id", "title"),
    "pattern_memory": ("memory_record_id", "paper_id", "title"),
    "source_fact_memory": ("memory_record_id", "source_id", "fact_type"),
    "graph_nodes": ("graph_node_id", "node_type"),
    "graph_edges": ("graph_edge_id", "edge_type", "source_node_id", "target_node_id"),
    "sources": ("source_id", "source_type", "title", "read_status", "stale"),
}

_FACET_KEYS_BY_COUNT_KEY: Mapping[str, tuple[str, ...]] = {
    "chunks": ("context_role", "section_type"),
    "evidence_spans": ("target_type", "anchor_mode"),
    "structured_entities": ("entity_type",),
    "evidence_memory": ("memory_type",),
    "pattern_memory": ("memory_type",),
    "source_fact_memory": ("memory_type", "fact_type"),
    "graph_nodes": ("node_type",),
    "graph_edges": ("edge_type",),
    "sources": ("source_type", "read_status", "stale"),
}

_PRESENCE_KEYS_BY_COUNT_KEY: Mapping[str, tuple[str, ...]] = {
    "result_evidence": ("dataset_id", "method_id", "metric_id", "comparator_text"),
}

_STUDY_BRIEF_FIELDS = (
    "aim",
    "hypothesis",
    "constraints",
    "confirmed_decisions",
    "open_risks",
    "linked_source_ids",
)

_MAX_SAMPLE_REFS = 3
_MAX_FACET_VALUES = 6


class StudyAgentReadToolExecutor:
    """Execute read-tool adapters against the already bounded context pack."""

    def execute(
        self,
        *,
        context: Mapping[str, Any],
        read_tools: Mapping[str, Any],
        task_type: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        executed = [
            observation
            for item in self._tool_items(read_tools.get("available"))
            if (
                observation := self._executed_observation(
                    item,
                    context=context,
                    task_type=task_type,
                )
            )
        ]
        blocked = [
            self._blocked_observation(item, task_type=task_type)
            for item in self._tool_items(read_tools.get("blocked"))
        ]
        return {"executed": executed, "blocked": blocked}

    def _executed_observation(
        self,
        item: Mapping[str, Any],
        *,
        context: Mapping[str, Any],
        task_type: str | None,
    ) -> dict[str, Any] | None:
        name = self._tool_name(item)
        if not name:
            return None
        groups = _TOOL_GROUPS.get(name)
        if groups is None:
            return None
        observed_counts: dict[str, int] = {}
        sample_refs: dict[str, list[dict[str, Any]]] = {}
        observed_facets: dict[str, dict[str, dict[str, int]]] = {}
        for group in groups:
            count_key = _COUNT_KEY_BY_GROUP.get(group, group)
            value = self._context_value(context, group)
            count = self._context_count(value)
            if count > 0:
                observed_counts[count_key] = count
            facets = self._observed_facets(count_key=count_key, value=value)
            if facets:
                observed_facets[count_key] = facets
            refs = self._sample_refs(count_key=count_key, value=value)
            if refs:
                sample_refs[count_key] = refs

        observation: dict[str, Any] = {
            "name": name,
            "status": "completed",
            "side_effects": item.get("side_effects") is True,
            "observed_counts": observed_counts,
        }
        if task_type:
            observation["task_type"] = task_type
        if observed_facets:
            observation["observed_facets"] = observed_facets
        if sample_refs:
            observation["sample_refs"] = sample_refs
        return observation

    def _blocked_observation(
        self,
        item: Mapping[str, Any],
        *,
        task_type: str | None,
    ) -> dict[str, Any]:
        observation = {
            "name": self._tool_name(item) or "unknown_read_tool",
            "status": "blocked",
            "side_effects": item.get("side_effects") is True,
            "missing_requirements": self._string_list(item.get("missing_requirements")),
            "observed_counts": {},
        }
        if task_type:
            observation["task_type"] = task_type
        return observation

    def _tool_items(self, value: Any) -> list[Mapping[str, Any]]:
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, Mapping)]

    def _tool_name(self, item: Mapping[str, Any]) -> str | None:
        name = item.get("name")
        return name if isinstance(name, str) and name else None

    def _string_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, str) and item]

    def _context_value(self, context: Mapping[str, Any], path: str) -> Any:
        current: Any = context
        for segment in path.split("."):
            if not isinstance(current, Mapping):
                return None
            current = current.get(segment)
        return current

    def _context_count(self, value: Any) -> int:
        if isinstance(value, list):
            return len([item for item in value if isinstance(item, Mapping)])
        if isinstance(value, Mapping):
            return 1 if value else 0
        return 0

    def _sample_refs(self, *, count_key: str, value: Any) -> list[dict[str, Any]]:
        safe_keys = _SAFE_REF_KEYS_BY_COUNT_KEY.get(count_key)
        if safe_keys is None:
            if count_key == "study_brief" and isinstance(value, Mapping):
                version = value.get("version")
                return [{"version": version}] if isinstance(version, int) else []
            return []
        if not isinstance(value, list):
            return []
        refs: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, Mapping):
                continue
            ref = self._safe_ref(item, safe_keys=safe_keys)
            if ref:
                refs.append(ref)
            if len(refs) >= _MAX_SAMPLE_REFS:
                break
        return refs

    def _observed_facets(
        self,
        *,
        count_key: str,
        value: Any,
    ) -> dict[str, dict[str, int]]:
        if count_key == "study_brief" and isinstance(value, Mapping):
            return self._study_brief_facets(value)
        if not isinstance(value, list):
            return {}

        facets: dict[str, dict[str, int]] = {}
        for key in _FACET_KEYS_BY_COUNT_KEY.get(count_key, ()):
            counter = Counter[str]()
            for item in value:
                if not isinstance(item, Mapping):
                    continue
                facet_value = self._safe_facet_value(
                    self._safe_item_value(item, count_key=count_key, key=key)
                )
                if facet_value:
                    counter[facet_value] += 1
            if counter:
                facets[key] = self._bounded_counter(counter)

        presence_counter = Counter[str]()
        for item in value:
            if not isinstance(item, Mapping):
                continue
            for key in _PRESENCE_KEYS_BY_COUNT_KEY.get(count_key, ()):
                if self._has_value(
                    self._safe_item_value(item, count_key=count_key, key=key)
                ):
                    presence_counter[key] += 1
        if presence_counter:
            facets["field_presence"] = self._bounded_counter(presence_counter)
        return facets

    def _safe_item_value(
        self,
        item: Mapping[str, Any],
        *,
        count_key: str,
        key: str,
    ) -> Any:
        value = item.get(key)
        if value is not None:
            return value
        if count_key == "source_fact_memory" and key in {"fact_type", "source_id"}:
            payload = item.get("payload")
            if isinstance(payload, Mapping):
                return payload.get(key)
        return None

    def _study_brief_facets(self, value: Mapping[str, Any]) -> dict[str, dict[str, int]]:
        fields_present = {
            field: 1 for field in _STUDY_BRIEF_FIELDS if self._has_value(value.get(field))
        }
        return {"fields_present": fields_present} if fields_present else {}

    def _safe_facet_value(self, value: Any) -> str | None:
        if isinstance(value, bool):
            return "true" if value else "false"
        if not isinstance(value, str):
            return None
        normalized = value.strip().lower()
        if not normalized or len(normalized) > 64:
            return None
        if not all(char.isalnum() or char in "_:-" for char in normalized):
            return None
        return normalized

    def _has_value(self, value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, bool):
            return value
        if isinstance(value, int | float):
            return math.isfinite(value) if isinstance(value, float) else True
        if isinstance(value, list | tuple | set | Mapping):
            return bool(value)
        return False

    def _bounded_counter(self, counter: Counter[str]) -> dict[str, int]:
        return {
            key: count
            for key, count in sorted(
                counter.items(),
                key=lambda item: (-item[1], item[0]),
            )[:_MAX_FACET_VALUES]
        }

    def _safe_ref(
        self,
        item: Mapping[str, Any],
        *,
        safe_keys: tuple[str, ...],
    ) -> dict[str, Any]:
        ref: dict[str, Any] = {}
        for key in safe_keys:
            value = self._safe_item_value(item, count_key="", key=key)
            if key in {"fact_type", "source_id"} and item.get("memory_type") == "source_fact":
                value = self._safe_item_value(
                    item,
                    count_key="source_fact_memory",
                    key=key,
                )
            if isinstance(value, str) and value:
                ref[key] = value
            elif isinstance(value, bool):
                ref[key] = value
            elif isinstance(value, int):
                ref[key] = value
            elif isinstance(value, float) and math.isfinite(value):
                ref[key] = value
        return ref


def default_study_agent_read_tool_executor() -> StudyAgentReadToolExecutor:
    """Return the built-in bounded read-tool executor."""

    return StudyAgentReadToolExecutor()
