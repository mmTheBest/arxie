"""Quality harnesses for Paperbase research-agent outputs."""

from __future__ import annotations

from collections import Counter
from typing import Any

from paperbase.research.artifact_evaluation import build_artifact_evaluation

SUPPORT_STATUS_VALUES = (
    "supported",
    "mixed",
    "inferred",
    "user_provided",
    "speculative",
)
SUPPORTING_LAYER_VALUES = (
    "evidence_memory",
    "pattern_memory",
    "field_graph",
    "study_brief",
    "source_library",
)
EVIDENCE_BACKED_STATUSES = {"supported", "mixed", "user_provided"}
REFERENCE_BACKED_LAYERS = {"evidence_memory", "field_graph", "study_brief", "source_library"}
SUPPORTING_LAYER_REQUIRED_STATUSES = {"supported", "mixed", "inferred", "user_provided"}
RECOMMENDATION_FIELD_NAME = "recommendations"
RECOMMENDATION_FIELD_SUFFIX = "_recommendations"
ARTIFACT_REQUIRED_FIELDS = ("title", "artifact_type")


def validate_research_output(
    *,
    context: dict[str, Any],
    output_payload: dict[str, Any],
    readiness_warnings: list[str],
    context_selection_counts: dict[str, int] | None = None,
    planner_router_metadata: dict[str, Any] | None = None,
    forced_status: str | None = None,
) -> dict[str, Any]:
    papers = [paper for paper in context.get("papers", []) if isinstance(paper, dict)]
    raw_references = output_payload.get("evidence_references")
    references = [
        reference
        for reference in raw_references
        if isinstance(reference, dict)
    ] if isinstance(raw_references, list) else []
    readiness_blockers = list(output_payload.get("readiness_blockers") or [])
    missing_evidence: list[str] = []
    unsupported_claims: list[str] = []
    recommendations = _recommendations(output_payload)
    available_layers = _available_layers(context)

    if not papers:
        missing_evidence.append("No paper evidence was available.")
    if (
        papers
        and forced_status != "blocked"
        and not references
    ):
        missing_evidence.append("No explicit evidence references were provided.")
    if output_payload.get("summary") and not papers:
        unsupported_claims.append(str(output_payload["summary"]))

    recommendation_diagnostics = _recommendation_diagnostics(
        recommendations=recommendations,
        available_layers=available_layers,
    )
    artifact_schema_diagnostics = _artifact_schema_diagnostics(
        output_payload=output_payload,
        recommendations=recommendations,
    )
    selected_context_counts = _context_selection_counts(
        context=context,
        explicit_counts=context_selection_counts,
    )
    artifact_evaluation = build_artifact_evaluation(
        output_payload=output_payload,
        recommendations=recommendations,
        artifact_schema=artifact_schema_diagnostics,
        recommendation_report=recommendation_diagnostics["report"],
        context_selection_counts=selected_context_counts,
        readiness_warnings=readiness_warnings,
        readiness_blockers=readiness_blockers,
        planner_router_metadata=planner_router_metadata,
    )
    missing_evidence.extend(recommendation_diagnostics["missing_evidence"])
    unsupported_claims.extend(recommendation_diagnostics["unsupported_claims"])

    if forced_status == "blocked" or readiness_blockers:
        harness_status = "blocked"
    elif missing_evidence or unsupported_claims:
        harness_status = "needs_attention"
    else:
        harness_status = "passed"

    return {
        "harness_status": harness_status,
        "missing_evidence": missing_evidence,
        "unsupported_claims": unsupported_claims,
        "readiness_blockers": readiness_blockers,
        "readiness_warnings": readiness_warnings,
        "evidence_reference_count": len(references),
        "evidence_paper_count": len(papers),
        "source_context_count": len([source for source in context.get("sources", []) if isinstance(source, dict)]),
        "context_selection_counts": selected_context_counts,
        "artifact_schema": artifact_schema_diagnostics,
        "artifact_evaluation": artifact_evaluation,
        **recommendation_diagnostics["report"],
    }


def _context_selection_counts(
    *,
    context: dict[str, Any],
    explicit_counts: dict[str, int] | None,
) -> dict[str, int]:
    if explicit_counts is not None:
        return {
            key: value
            for key, value in explicit_counts.items()
            if isinstance(key, str) and type(value) is int
        }

    intelligence_layers = context.get("intelligence_layers")
    if not isinstance(intelligence_layers, dict):
        intelligence_layers = {}
    field_graph = intelligence_layers.get("field_graph")
    if not isinstance(field_graph, dict):
        field_graph = {}
    return {
        "papers": len(
            [paper for paper in context.get("papers", []) if isinstance(paper, dict)]
        ),
        "sources": len(
            [source for source in context.get("sources", []) if isinstance(source, dict)]
        ),
        "evidence_memory": len(
            [
                item
                for item in intelligence_layers.get("evidence_memory", [])
                if isinstance(item, dict)
            ]
        ),
        "pattern_memory": len(
            [
                item
                for item in intelligence_layers.get("pattern_memory", [])
                if isinstance(item, dict)
            ]
        ),
        "graph_nodes": len(
            [item for item in field_graph.get("nodes", []) if isinstance(item, dict)]
        ),
        "graph_edges": len(
            [item for item in field_graph.get("edges", []) if isinstance(item, dict)]
        ),
        "study_brief": 1 if isinstance(intelligence_layers.get("study_brief"), dict) else 0,
    }


def _artifact_schema_diagnostics(
    *,
    output_payload: dict[str, Any],
    recommendations: list[dict[str, Any]],
) -> dict[str, Any]:
    missing_required_fields: list[str] = []
    invalid_fields: list[str] = []
    errors: list[dict[str, str]] = []

    for field_name in ARTIFACT_REQUIRED_FIELDS:
        if field_name not in output_payload or output_payload.get(field_name) is None:
            missing_required_fields.append(field_name)
            errors.append(
                {
                    "field": field_name,
                    "code": "missing_required",
                    "message": f"Required artifact field '{field_name}' is missing.",
                }
            )
            continue
        value = output_payload.get(field_name)
        if not isinstance(value, str) or not value.strip():
            invalid_fields.append(field_name)
            errors.append(
                {
                    "field": field_name,
                    "code": "invalid_required",
                    "message": f"Required artifact field '{field_name}' must be a non-empty string.",
                }
            )

    if "evidence_references" in output_payload:
        references = output_payload.get("evidence_references")
        if not isinstance(references, list):
            invalid_fields.append("evidence_references")
            errors.append(
                {
                    "field": "evidence_references",
                    "code": "invalid_list",
                    "message": "Artifact field 'evidence_references' must be a list when present.",
                }
            )
        elif any(not isinstance(reference, dict) for reference in references):
            invalid_fields.append("evidence_references")
            errors.append(
                {
                    "field": "evidence_references",
                    "code": "invalid_item",
                    "message": "Artifact field 'evidence_references' must contain objects.",
                }
            )

    recommendation_field_names = [
        field_name
        for field_name in output_payload
        if isinstance(field_name, str) and _is_recommendation_field(field_name)
    ]
    for field_name in recommendation_field_names:
        raw_items = output_payload.get(field_name)
        if not isinstance(raw_items, list):
            invalid_fields.append(field_name)
            errors.append(
                {
                    "field": field_name,
                    "code": "invalid_list",
                    "message": f"Artifact field '{field_name}' must be a list when present.",
                }
            )
            continue
        if any(
            not isinstance(item, dict)
            and not (isinstance(item, str) and item.strip())
            for item in raw_items
        ):
            invalid_fields.append(field_name)
            errors.append(
                {
                    "field": field_name,
                    "code": "invalid_item",
                    "message": f"Artifact field '{field_name}' must contain objects or non-empty strings.",
                }
            )

    return {
        "status": "needs_attention" if errors else "passed",
        "required_field_count": len(ARTIFACT_REQUIRED_FIELDS),
        "missing_required_fields": missing_required_fields,
        "invalid_fields": _unique_strings(invalid_fields),
        "recommendation_field_count": len(recommendation_field_names),
        "recommendation_count": len(recommendations),
        "error_count": len(errors),
        "errors": errors,
    }


def _unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _recommendations(output_payload: dict[str, Any]) -> list[dict[str, Any]]:
    top_level_recommendations = _recommendation_items(
        output_payload.get(RECOMMENDATION_FIELD_NAME),
        field_name=RECOMMENDATION_FIELD_NAME,
    )
    recommendations = list(top_level_recommendations)
    labeled_top_level_texts = {
        _recommendation_label(item).casefold()
        for item in top_level_recommendations
        if _recommendation_support_status(item)
        or _recommendation_supporting_layers(item)
        or _recommendation_references(item)
    }
    for field_name, raw_items in output_payload.items():
        if field_name == RECOMMENDATION_FIELD_NAME:
            continue
        if not _is_recommendation_field(field_name):
            continue
        for item in _recommendation_items(raw_items, field_name=field_name):
            if (
                item.get("_normalized_from_string") is True
                and _recommendation_label(item).casefold() in labeled_top_level_texts
            ):
                continue
            recommendations.append(item)
    return recommendations


def _is_recommendation_field(field_name: str) -> bool:
    return field_name == RECOMMENDATION_FIELD_NAME or field_name.endswith(
        RECOMMENDATION_FIELD_SUFFIX
    )


def _recommendation_items(raw_items: Any, *, field_name: str) -> list[dict[str, Any]]:
    if not isinstance(raw_items, list):
        return []
    recommendations: list[dict[str, Any]] = []
    for item in raw_items:
        if isinstance(item, dict):
            recommendations.append({"recommendation_field": field_name, **item})
        elif isinstance(item, str) and item.strip():
            recommendations.append(
                {
                    "recommendation_field": field_name,
                    "title": item.strip(),
                    "_normalized_from_string": True,
                }
            )
    return recommendations


def _recommendation_support_status(recommendation: dict[str, Any]) -> str | None:
    status = recommendation.get("support_status")
    return status if isinstance(status, str) and status else None


def _recommendation_references(recommendation: dict[str, Any]) -> list[dict[str, Any]]:
    references = recommendation.get("evidence_references")
    if not isinstance(references, list):
        return []
    return [reference for reference in references if isinstance(reference, dict)]


def _available_layers(context: dict[str, Any]) -> set[str]:
    layers: set[str] = set()
    if any(isinstance(paper, dict) for paper in context.get("papers", [])):
        layers.add("source_library")
    if any(isinstance(source, dict) for source in context.get("sources", [])):
        layers.add("source_library")

    intelligence_layers = context.get("intelligence_layers")
    if not isinstance(intelligence_layers, dict):
        return layers
    if any(isinstance(item, dict) for item in intelligence_layers.get("evidence_memory", [])):
        layers.add("evidence_memory")
    if any(isinstance(item, dict) for item in intelligence_layers.get("pattern_memory", [])):
        layers.add("pattern_memory")
    field_graph = intelligence_layers.get("field_graph")
    if isinstance(field_graph, dict) and (
        any(isinstance(item, dict) for item in field_graph.get("nodes", []))
        or any(isinstance(item, dict) for item in field_graph.get("edges", []))
    ):
        layers.add("field_graph")
    if isinstance(intelligence_layers.get("study_brief"), dict):
        layers.add("study_brief")
    return layers


def _recommendation_diagnostics(
    *,
    recommendations: list[dict[str, Any]],
    available_layers: set[str],
) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    layer_counts: Counter[str] = Counter()
    missing_evidence: list[str] = []
    unsupported_claims: list[str] = []
    missing_support_status = 0
    missing_supporting_layers = 0
    invalid_support_status = 0
    missing_evidence_references = 0
    speculative_as_supported = 0
    unavailable_layer_recommendations = 0
    invalid_layer_recommendations = 0

    for recommendation in recommendations:
        label = _recommendation_label(recommendation)
        status = _recommendation_support_status(recommendation)
        if status is None:
            missing_support_status += 1
            missing_evidence.append(f"Recommendation is missing support_status: {label}")
        elif status not in SUPPORT_STATUS_VALUES:
            invalid_support_status += 1
            missing_evidence.append(f"Recommendation has invalid support_status '{status}': {label}")
        else:
            status_counts[status] += 1

        layers = _recommendation_supporting_layers(recommendation)
        if status in SUPPORTING_LAYER_REQUIRED_STATUSES and not layers:
            missing_supporting_layers += 1
            missing_evidence.append(f"Recommendation is missing supporting_layers: {label}")
        invalid_layers = [layer for layer in layers if layer not in SUPPORTING_LAYER_VALUES]
        if invalid_layers:
            invalid_layer_recommendations += 1
            missing_evidence.append(
                "Recommendation cites invalid supporting layer(s) "
                f"{', '.join(sorted(set(invalid_layers)))}: {label}"
            )
        valid_layers = [layer for layer in layers if layer in SUPPORTING_LAYER_VALUES]
        layer_counts.update(valid_layers)
        unavailable_layers = sorted({layer for layer in valid_layers if layer not in available_layers})
        if unavailable_layers:
            unavailable_layer_recommendations += 1
            unsupported_claims.append(
                "Recommendation cites unavailable supporting layer(s) "
                f"{', '.join(unavailable_layers)}: {label}"
            )

        if _needs_recommendation_references(status, valid_layers) and not _recommendation_references(recommendation):
            missing_evidence_references += 1
            missing_evidence.append(f"Recommendation is missing evidence references: {label}")
        if status == "supported" and _is_speculative_recommendation(recommendation):
            speculative_as_supported += 1
            unsupported_claims.append(f"Recommendation is speculative but marked supported: {label}")

    return {
        "missing_evidence": missing_evidence,
        "unsupported_claims": unsupported_claims,
        "report": {
            "recommendation_count": len(recommendations),
            "recommendations_missing_support_status": missing_support_status,
            "recommendations_missing_supporting_layers": missing_supporting_layers,
            "recommendations_invalid_support_status": invalid_support_status,
            "recommendations_missing_evidence_references": missing_evidence_references,
            "recommendations_speculative_as_supported": speculative_as_supported,
            "recommendations_with_unavailable_layers": unavailable_layer_recommendations,
            "recommendations_with_invalid_layers": invalid_layer_recommendations,
            "recommendation_support_status_counts": {
                status: status_counts[status]
                for status in SUPPORT_STATUS_VALUES
                if status_counts[status]
            },
            "recommendation_supporting_layer_counts": {
                layer: layer_counts[layer]
                for layer in SUPPORTING_LAYER_VALUES
                if layer_counts[layer]
            },
            "available_supporting_layers": sorted(available_layers),
        },
    }


def _recommendation_supporting_layers(recommendation: dict[str, Any]) -> list[str]:
    layers = recommendation.get("supporting_layers")
    if isinstance(layers, str):
        layer = layers.strip()
        return [layer] if layer else []
    if not isinstance(layers, list):
        return []
    return [layer.strip() for layer in layers if isinstance(layer, str) and layer.strip()]


def _needs_recommendation_references(status: str | None, layers: list[str]) -> bool:
    if status in EVIDENCE_BACKED_STATUSES:
        return True
    return bool(set(layers) & REFERENCE_BACKED_LAYERS)


def _is_speculative_recommendation(recommendation: dict[str, Any]) -> bool:
    if recommendation.get("is_speculative") is True or recommendation.get("speculative") is True:
        return True
    for key in ("claim_type", "evidence_type"):
        value = recommendation.get(key)
        if isinstance(value, str) and value.casefold() == "speculative":
            return True
    text = " ".join(
        str(recommendation.get(key) or "")
        for key in ("title", "detail", "claim", "rationale")
    ).casefold()
    return "speculative" in text


def _recommendation_label(recommendation: dict[str, Any]) -> str:
    for key in ("title", "claim", "summary", "detail"):
        value = recommendation.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:160]
    return "Untitled recommendation"
