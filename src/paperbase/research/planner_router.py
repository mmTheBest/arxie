"""Deterministic planner/router metadata for research-agent run traces."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from paperbase.research.skill_policies import ResearchSkillPolicy
from paperbase.research.task_descriptors import task_descriptor_for

PLANNER_ROUTER_METADATA_VERSION = "planner_router_v1"
PLANNER_ROUTER_WARNING_LIMIT = 8
PLANNER_ROUTER_WARNING_TEXT_LIMIT = 240
PLANNER_ROUTER_COUNT_LIMIT = 24
PLANNER_ROUTER_DICT_FIELD_LIMIT = 16
PLANNER_ROUTER_LIST_LIMIT = 16
PLANNER_ROUTER_TEXT_LIMIT = 240
PLANNER_ROUTER_FLOW = ["context", "plan", "tool_call", "synthesis", "validation"]
SKILL_SELECTION_SOURCES = frozenset({"payload", "suggestion", "artifact_type", "message"})


def build_planner_router_metadata(
    *,
    skill_id: str,
    policy: ResearchSkillPolicy,
    payload: Mapping[str, Any],
    selected_context_summary: Mapping[str, Any],
    selected_item_counts: Mapping[str, Any],
    readiness_warnings: list[str],
) -> dict[str, Any]:
    """Build bounded operational metadata for the existing plan/tool-call trace."""

    descriptor = task_descriptor_for(skill_id)
    descriptor_fallback = descriptor.task_type != skill_id
    return {
        "metadata_version": PLANNER_ROUTER_METADATA_VERSION,
        "routing": {
            "skill_id": skill_id,
            "artifact_type": policy.artifact_type,
            "selected_skill_source": skill_selection_source_for_payload(payload),
            "descriptor_task_type": descriptor.task_type,
            "descriptor_fallback": descriptor_fallback,
        },
        "model_policy": {
            "policy": policy.model_policy,
            "allow_deterministic_fallback": policy.allow_deterministic_fallback,
            "model_required": policy.model_policy == "required",
        },
        "task_descriptor": {
            "task_type": descriptor.task_type,
            "descriptor_fallback": descriptor_fallback,
            "required_context_types": sorted(descriptor.required_context_types),
            "readiness_requirements": sorted(descriptor.readiness_requirements),
            "required_output_fields": sorted(descriptor.required_output_fields),
            "allowed_internal_tools": sorted(descriptor.allowed_internal_tools),
        },
        "context": {
            "selected_context": _bounded_metadata_value(selected_context_summary),
            "selected_item_counts": _bounded_count_map(selected_item_counts),
            "readiness_warning_count": len(readiness_warnings),
            "readiness_warnings": _bounded_warning_list(readiness_warnings),
        },
        "execution_contract": {
            "flow": list(PLANNER_ROUTER_FLOW),
            "metadata_only": True,
            "autonomous_tool_loop": False,
            "model_routing_changed": False,
        },
    }


def skill_selection_source_for_payload(payload: Mapping[str, Any]) -> str:
    """Return the operational source that selected a skill for this payload."""

    explicit_source = payload.get("skill_selection_source")
    if isinstance(explicit_source, str) and explicit_source in SKILL_SELECTION_SOURCES:
        return explicit_source
    if payload.get("suggestion_id"):
        return "suggestion"
    if payload.get("artifact_type"):
        return "artifact_type"
    if payload.get("user_message"):
        return "message"
    if payload.get("skill_id"):
        return "payload"
    return "message"


def _bounded_warning_list(readiness_warnings: list[str]) -> list[str]:
    return [
        _bounded_text(warning, limit=PLANNER_ROUTER_WARNING_TEXT_LIMIT)
        for warning in readiness_warnings[:PLANNER_ROUTER_WARNING_LIMIT]
    ]


def _bounded_count_map(selected_item_counts: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key in sorted(selected_item_counts):
        if len(counts) >= PLANNER_ROUTER_COUNT_LIMIT:
            break
        value = selected_item_counts[key]
        if not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, int):
            continue
        counts[key] = value
    return counts


def _bounded_metadata_value(value: Any) -> Any:
    if isinstance(value, str):
        return _bounded_text(value, limit=PLANNER_ROUTER_TEXT_LIMIT)
    if isinstance(value, bool | int | float) or value is None:
        return value
    if isinstance(value, Mapping):
        bounded: dict[str, Any] = {}
        for key, nested_value in list(value.items())[:PLANNER_ROUTER_DICT_FIELD_LIMIT]:
            if not isinstance(key, str):
                continue
            bounded[key] = _bounded_metadata_value(nested_value)
        return bounded
    if isinstance(value, list | tuple):
        return [
            _bounded_metadata_value(item)
            for item in list(value)[:PLANNER_ROUTER_LIST_LIMIT]
        ]
    return _bounded_text(str(value), limit=PLANNER_ROUTER_TEXT_LIMIT)


def _bounded_text(value: object, *, limit: int) -> str:
    text = " ".join(str(value or "").strip().split())
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}..."
