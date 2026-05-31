"""Report-only artifact evaluation diagnostics for research-agent outputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from paperbase.research.skills import ARTIFACT_SKILL_MAP, artifact_type_for_skill
from paperbase.research.task_descriptors import task_descriptor_for

ARTIFACT_EVALUATION_METADATA_VERSION = "artifact_evaluation_v1"
ARTIFACT_EVALUATION_MODE = "report_only"
ARTIFACT_EVALUATION_LIST_LIMIT = 16
ARTIFACT_EVALUATION_COUNT_LIMIT = 24
ARTIFACT_EVALUATION_TEXT_LIMIT = 240
PAPER_REFERENCE_TYPES = frozenset(
    {
        "paper",
        "section",
        "chunk",
        "paper_chunk",
        "structured_evidence",
    }
)
USER_SOURCE_REFERENCE_TYPES = frozenset(
    {
        "study_source",
        "study_brief",
        "source_library",
        "user_source",
    }
)
PAPER_DERIVED_LAYERS = frozenset(
    {
        "evidence_memory",
        "pattern_memory",
        "field_graph",
    }
)


def build_artifact_evaluation(
    *,
    output_payload: Mapping[str, Any],
    recommendations: list[dict[str, Any]],
    artifact_schema: Mapping[str, Any],
    recommendation_report: Mapping[str, Any],
    context_selection_counts: Mapping[str, Any],
    readiness_warnings: list[str],
    readiness_blockers: list[str],
    planner_router_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build bounded deterministic report-only artifact evaluation metadata."""

    skill_id = _resolved_skill_id(
        output_payload=output_payload,
        planner_router_metadata=planner_router_metadata,
    )
    artifact_type = _string_or_empty(output_payload.get("artifact_type"))
    expected_artifact_type = _expected_artifact_type(
        skill_id=skill_id,
        artifact_type=artifact_type,
        planner_router_metadata=planner_router_metadata,
    )
    descriptor = task_descriptor_for(skill_id)
    descriptor_fallback = descriptor.task_type != skill_id
    checks = {
        "task_artifact_alignment": _task_artifact_alignment_check(
            artifact_type=artifact_type,
            expected_artifact_type=expected_artifact_type,
        ),
        "required_output_fields": _required_output_fields_check(
            output_payload=output_payload,
            required_output_fields=sorted(descriptor.required_output_fields),
        ),
        "artifact_schema": _artifact_schema_check(artifact_schema),
        "evidence_reference_coverage": _evidence_reference_coverage_check(
            output_payload=output_payload,
            recommendations=recommendations,
            recommendation_report=recommendation_report,
        ),
        "recommendation_support_coverage": _recommendation_support_coverage_check(
            recommendation_report
        ),
        "context_readiness": _context_readiness_check(
            context_selection_counts=context_selection_counts,
            readiness_warnings=readiness_warnings,
            readiness_blockers=readiness_blockers,
        ),
        "source_separation": _source_separation_check(
            output_payload=output_payload,
            recommendations=recommendations,
        ),
    }
    summary = _summary_for_checks(checks)
    evaluation: dict[str, Any] = {
        "metadata_version": ARTIFACT_EVALUATION_METADATA_VERSION,
        "mode": ARTIFACT_EVALUATION_MODE,
        "status": "needs_attention"
        if summary["needs_attention_check_count"]
        else "passed",
        "task": {
            "skill_id": _bounded_text(skill_id),
            "artifact_type": _bounded_text(artifact_type),
            "expected_artifact_type": _bounded_text(expected_artifact_type),
            "descriptor_task_type": _bounded_text(descriptor.task_type),
            "descriptor_fallback": descriptor_fallback,
        },
        "summary": summary,
        "checks": checks,
    }
    planning = _planner_router_summary(planner_router_metadata)
    if planning:
        evaluation["planning"] = planning
    return evaluation


def _resolved_skill_id(
    *,
    output_payload: Mapping[str, Any],
    planner_router_metadata: Mapping[str, Any] | None,
) -> str:
    skill_id = _string_or_empty(output_payload.get("skill_id"))
    if skill_id:
        return skill_id
    routing = _planner_routing(planner_router_metadata)
    routed_skill_id = _string_or_empty(routing.get("skill_id"))
    if routed_skill_id:
        return routed_skill_id
    artifact_type = _string_or_empty(output_payload.get("artifact_type"))
    if artifact_type in ARTIFACT_SKILL_MAP:
        return ARTIFACT_SKILL_MAP[artifact_type]
    return "literature_review"


def _expected_artifact_type(
    *,
    skill_id: str,
    artifact_type: str,
    planner_router_metadata: Mapping[str, Any] | None,
) -> str:
    routing = _planner_routing(planner_router_metadata)
    routed_artifact_type = _string_or_empty(routing.get("artifact_type"))
    if routed_artifact_type:
        return routed_artifact_type
    return artifact_type_for_skill(skill_id, fallback=artifact_type or "artifact")


def _task_artifact_alignment_check(
    *,
    artifact_type: str,
    expected_artifact_type: str,
) -> dict[str, Any]:
    matches_expected = bool(artifact_type) and artifact_type == expected_artifact_type
    return {
        "status": "passed" if matches_expected else "needs_attention",
        "artifact_type": _bounded_text(artifact_type),
        "expected_artifact_type": _bounded_text(expected_artifact_type),
        "matches_expected": matches_expected,
    }


def _required_output_fields_check(
    *,
    output_payload: Mapping[str, Any],
    required_output_fields: list[str],
) -> dict[str, Any]:
    missing_fields = [
        field_name
        for field_name in required_output_fields
        if not _has_meaningful_value(output_payload.get(field_name))
    ]
    return {
        "status": "needs_attention" if missing_fields else "passed",
        "required_fields": _bounded_string_list(required_output_fields),
        "missing_fields": _bounded_string_list(missing_fields),
        "missing_field_count": len(missing_fields),
    }


def _artifact_schema_check(artifact_schema: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": _status_or_needs_attention(artifact_schema.get("status")),
        "error_count": _safe_int(artifact_schema.get("error_count")),
        "missing_required_fields": _bounded_string_list(
            artifact_schema.get("missing_required_fields")
        ),
        "invalid_fields": _bounded_string_list(artifact_schema.get("invalid_fields")),
    }


def _evidence_reference_coverage_check(
    *,
    output_payload: Mapping[str, Any],
    recommendations: list[dict[str, Any]],
    recommendation_report: Mapping[str, Any],
) -> dict[str, Any]:
    top_level_reference_count = len(_top_level_references(output_payload))
    recommendation_reference_count = sum(
        len(_recommendation_references(recommendation))
        for recommendation in recommendations
    )
    recommendations_with_references = len(
        [
            recommendation
            for recommendation in recommendations
            if _recommendation_references(recommendation)
        ]
    )
    recommendations_missing_references = _safe_int(
        recommendation_report.get("recommendations_missing_evidence_references")
    )
    status = "passed"
    if top_level_reference_count == 0 or recommendations_missing_references:
        status = "needs_attention"
    return {
        "status": status,
        "top_level_reference_count": top_level_reference_count,
        "recommendation_reference_count": recommendation_reference_count,
        "total_reference_count": top_level_reference_count
        + recommendation_reference_count,
        "recommendation_count": len(recommendations),
        "recommendations_with_references": recommendations_with_references,
        "recommendations_missing_references": recommendations_missing_references,
    }


def _recommendation_support_coverage_check(
    recommendation_report: Mapping[str, Any],
) -> dict[str, Any]:
    issue_count = sum(
        _safe_int(recommendation_report.get(key))
        for key in (
            "recommendations_missing_support_status",
            "recommendations_missing_supporting_layers",
            "recommendations_invalid_support_status",
            "recommendations_missing_evidence_references",
            "recommendations_speculative_as_supported",
            "recommendations_with_unavailable_layers",
            "recommendations_with_invalid_layers",
        )
    )
    return {
        "status": "needs_attention" if issue_count else "passed",
        "recommendation_count": _safe_int(recommendation_report.get("recommendation_count")),
        "issue_count": issue_count,
        "missing_support_status": _safe_int(
            recommendation_report.get("recommendations_missing_support_status")
        ),
        "missing_supporting_layers": _safe_int(
            recommendation_report.get("recommendations_missing_supporting_layers")
        ),
        "invalid_support_status": _safe_int(
            recommendation_report.get("recommendations_invalid_support_status")
        ),
        "missing_evidence_references": _safe_int(
            recommendation_report.get("recommendations_missing_evidence_references")
        ),
        "speculative_as_supported": _safe_int(
            recommendation_report.get("recommendations_speculative_as_supported")
        ),
        "with_unavailable_layers": _safe_int(
            recommendation_report.get("recommendations_with_unavailable_layers")
        ),
        "with_invalid_layers": _safe_int(
            recommendation_report.get("recommendations_with_invalid_layers")
        ),
        "support_status_counts": _bounded_count_map(
            recommendation_report.get("recommendation_support_status_counts")
        ),
        "supporting_layer_counts": _bounded_count_map(
            recommendation_report.get("recommendation_supporting_layer_counts")
        ),
    }


def _context_readiness_check(
    *,
    context_selection_counts: Mapping[str, Any],
    readiness_warnings: list[str],
    readiness_blockers: list[str],
) -> dict[str, Any]:
    return {
        "status": "needs_attention" if readiness_blockers else "passed",
        "readiness_warning_count": len(readiness_warnings),
        "readiness_blocker_count": len(readiness_blockers),
        "context_selection_counts": _bounded_count_map(context_selection_counts),
    }


def _source_separation_check(
    *,
    output_payload: Mapping[str, Any],
    recommendations: list[dict[str, Any]],
) -> dict[str, Any]:
    top_level_references = _top_level_references(output_payload)
    recommendation_references = [
        reference
        for recommendation in recommendations
        for reference in _recommendation_references(recommendation)
    ]
    all_references = top_level_references + recommendation_references
    declared_user_provided = [
        recommendation
        for recommendation in recommendations
        if _recommendation_support_status(recommendation) == "user_provided"
    ]
    user_provided_missing_user_refs = len(
        [
            recommendation
            for recommendation in declared_user_provided
            if not _has_reference_type(
                _recommendation_references(recommendation),
                USER_SOURCE_REFERENCE_TYPES,
            )
        ]
    )
    declared_paper_backed = len(
        [
            recommendation
            for recommendation in recommendations
            if set(_recommendation_supporting_layers(recommendation)) & PAPER_DERIVED_LAYERS
            or _has_reference_type(
                _recommendation_references(recommendation),
                PAPER_REFERENCE_TYPES,
            )
        ]
    )
    status = "needs_attention" if user_provided_missing_user_refs else "passed"
    return {
        "status": status,
        "paper_reference_count": _reference_count(all_references, PAPER_REFERENCE_TYPES),
        "user_source_reference_count": _reference_count(
            all_references,
            USER_SOURCE_REFERENCE_TYPES,
        ),
        "memory_reference_count": _memory_reference_count(all_references),
        "declared_user_provided_recommendation_count": len(declared_user_provided),
        "declared_paper_backed_recommendation_count": declared_paper_backed,
        "user_provided_recommendations_missing_user_source_refs": (
            user_provided_missing_user_refs
        ),
    }


def _summary_for_checks(checks: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    check_count = len(checks)
    needs_attention_count = len(
        [check for check in checks.values() if check.get("status") == "needs_attention"]
    )
    passed_count = check_count - needs_attention_count
    score = round(passed_count / check_count, 3) if check_count else 1.0
    return {
        "check_count": check_count,
        "passed_check_count": passed_count,
        "needs_attention_check_count": needs_attention_count,
        "score": score,
    }


def _planner_router_summary(
    planner_router_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(planner_router_metadata, Mapping):
        return {}
    routing = _planner_routing(planner_router_metadata)
    context = planner_router_metadata.get("context")
    context = context if isinstance(context, Mapping) else {}
    return {
        "selected_skill_source": _bounded_text(routing.get("selected_skill_source")),
        "readiness_warning_count": _safe_int(context.get("readiness_warning_count")),
        "selected_item_counts": _bounded_count_map(context.get("selected_item_counts")),
    }


def _planner_routing(
    planner_router_metadata: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if not isinstance(planner_router_metadata, Mapping):
        return {}
    routing = planner_router_metadata.get("routing")
    return routing if isinstance(routing, Mapping) else {}


def _top_level_references(output_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    references = output_payload.get("evidence_references")
    if not isinstance(references, list):
        return []
    return [reference for reference in references if isinstance(reference, dict)]


def _recommendation_references(recommendation: Mapping[str, Any]) -> list[dict[str, Any]]:
    references = recommendation.get("evidence_references")
    if not isinstance(references, list):
        return []
    return [reference for reference in references if isinstance(reference, dict)]


def _recommendation_support_status(recommendation: Mapping[str, Any]) -> str | None:
    status = recommendation.get("support_status")
    return status.strip() if isinstance(status, str) and status.strip() else None


def _recommendation_supporting_layers(recommendation: Mapping[str, Any]) -> list[str]:
    layers = recommendation.get("supporting_layers")
    if isinstance(layers, str):
        layer = layers.strip()
        return [layer] if layer else []
    if not isinstance(layers, list):
        return []
    return [layer.strip() for layer in layers if isinstance(layer, str) and layer.strip()]


def _has_reference_type(
    references: list[dict[str, Any]],
    reference_types: frozenset[str],
) -> bool:
    return any(_reference_type(reference) in reference_types for reference in references)


def _reference_count(
    references: list[dict[str, Any]],
    reference_types: frozenset[str],
) -> int:
    return len(
        [
            reference
            for reference in references
            if _reference_type(reference) in reference_types
        ]
    )


def _memory_reference_count(references: list[dict[str, Any]]) -> int:
    return len(
        [
            reference
            for reference in references
            if reference.get("memory_record_id")
            or reference.get("graph_node_id")
            or reference.get("graph_edge_id")
        ]
    )


def _reference_type(reference: Mapping[str, Any]) -> str:
    return _string_or_empty(reference.get("reference_type"))


def _has_meaningful_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list | tuple | set | dict):
        return bool(value)
    return True


def _bounded_count_map(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    counts: dict[str, int] = {}
    for key in sorted(value):
        if len(counts) >= ARTIFACT_EVALUATION_COUNT_LIMIT:
            break
        item = value[key]
        if isinstance(key, str) and type(item) is int:
            counts[_bounded_text(key)] = item
    return counts


def _bounded_string_list(value: Any) -> list[str]:
    if not isinstance(value, list | tuple | set | frozenset):
        return []
    strings = [
        _bounded_text(item)
        for item in value
        if isinstance(item, str) and item.strip()
    ]
    return strings[:ARTIFACT_EVALUATION_LIST_LIMIT]


def _status_or_needs_attention(value: Any) -> str:
    return value if value in {"passed", "needs_attention"} else "needs_attention"


def _safe_int(value: Any) -> int:
    return value if type(value) is int else 0


def _string_or_empty(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _bounded_text(value: Any) -> str:
    text = " ".join(str(value or "").strip().split())
    if len(text) <= ARTIFACT_EVALUATION_TEXT_LIMIT:
        return text
    return f"{text[:ARTIFACT_EVALUATION_TEXT_LIMIT].rstrip()}..."
