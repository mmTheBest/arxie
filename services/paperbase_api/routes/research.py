"""Research agent routes for collection-grounded investigation workflows."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, Path, Query, Request, status
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from paperbase.db.models import (
    BackgroundJob,
    CollectionPaper,
    EngineeringTrick,
    ExtractionRun,
    Limitation,
    ResearchAgentRun,
    ResearchArtifact,
    ResearchMessage,
    ResultRow,
    Section,
)
from paperbase.db.repositories import (
    CollectionRepository,
    PaperRepository,
    ResearchAgentRunRepository,
    ResearchRepository,
    WorkspaceRepository,
)
from paperbase.research.output_contracts import (
    REDACTED_PRIVATE_TRACE_TEXT,
    redact_private_trace_text,
    strip_private_trace_fields,
)
from paperbase.research.skill_policies import policy_for_skill
from paperbase.research.skills import artifact_type_for_skill, select_research_skill
from ra.utils.security import sanitize_identifier, sanitize_user_text
from services.paperbase_api.background_jobs import create_background_job
from services.paperbase_api.dependencies import get_project_id, get_session, get_session_factory
from services.paperbase_api.errors import PaperbaseAPIError
from services.paperbase_api.models import (
    PaperResearchLabelPatchRequest,
    PaperResearchLabelResponse,
    PaperResearchLabelsResponse,
    ResearchAgentRunResponse,
    ResearchAgentStepResponse,
    ResearchArtifactPatchRequest,
    ResearchArtifactResponse,
    ResearchArtifactsResponse,
    ResearchMessageCreateRequest,
    ResearchMessageJobResponse,
    ResearchMessageJobResponseData,
    ResearchMessageResponse,
    ResearchSuggestionResponse,
    ResearchSuggestionsResponse,
    ResearchThreadCreateRequest,
    ResearchThreadDetailResponse,
    ResearchThreadDetailResponseData,
    ResearchThreadResponse,
    ResearchThreadsResponse,
    ResearchValidationReportResponse,
    SinglePaperResearchLabelResponse,
    SingleResearchAgentRunResponse,
    SingleResearchArtifactResponse,
    SingleResearchThreadResponse,
    StudyContextPackResponse,
)
from services.paperbase_api.routes.jobs import background_job_to_response

router = APIRouter(tags=["research"])
ACTIVE_JOB_STATUSES = {"pending", "queued", "running"}


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)
PROMPT_VERSION = "research-agent-v1"
SELECTION_DIAGNOSTIC_LIMIT = 8
SELECTION_FEATURE_LIMIT = 12
SELECTION_LABEL_LIMIT = 160
SELECTION_FEATURE_PRIVATE_KEY_PARTS = (
    "paper_id",
    "path",
    "payload",
    "prompt",
    "quote",
    "raw",
    "secret",
    "source_ref",
    "text",
    "token",
)
CONTEXT_MATERIALIZATION_TOP_ITEM_LIMIT = 8
CONTEXT_MATERIALIZATION_PUBLIC_VERSIONS = {
    "context-materialization-v1",
    "context-materialization-v2",
    "context-materialization-v3",
}
CONTEXT_MATERIALIZATION_SAFE_LABEL_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9 ._+()\-]{0,159}$"
)
CONTEXT_MATERIALIZATION_SECRET_LABEL_PATTERN = re.compile(
    r"(api[\s_-]?key|authorization|bearer|credential|openai|password|"
    r"secret|sk-[a-z0-9]|token)",
    re.IGNORECASE,
)
CONTEXT_MATERIALIZATION_PRIVATE_LABEL_PARTS = (
    "path",
    "private",
    "prompt",
    "raw",
    "secret",
    "token",
)
CONTEXT_MATERIALIZATION_SUMMARY_KEYS = (
    "paper_count",
    "structured_entity_count",
    "result_evidence_count",
    "method_count",
    "dataset_count",
    "metric_count",
    "baseline_count",
    "limitation_count",
    "benchmark_table_count",
    "limitation_category_count",
    "finding_count",
    "claim_count",
    "claim_evidence_span_count",
    "claim_evidence_map_count",
)
CONTEXT_MATERIALIZATION_COUNT_KEYS = (
    "paper_count",
    "structured_entity_count",
    "result_evidence_count",
    "method_count",
    "dataset_count",
    "metric_count",
    "baseline_count",
    "limitation_count",
    "claim_count",
    "evidence_span_count",
)
CONTEXT_MATERIALIZATION_MAP_KEYS = {
    "method_map": "methods",
    "dataset_map": "datasets",
    "metric_map": "metrics",
    "baseline_map": "baselines",
    "benchmark_table": "benchmark_table",
    "limitation_inventory": "limitation_inventory",
    "claim_evidence_map": "claim_evidence_map",
}
TASK_EVIDENCE_ROLE_FIELDS = {
    "available_groups": "task_evidence_role_available_groups",
    "satisfied_groups": "task_evidence_role_satisfied_groups",
    "missing_groups": "task_evidence_role_missing_groups",
    "referenced_roles": "task_evidence_reference_roles",
    "available_roles": "task_evidence_available_roles",
}
TASK_EVIDENCE_ROLE_LIST_LIMIT = 12
VALIDATION_ISSUE_LIST_LIMIT = 12
VALIDATION_ISSUE_TEXT_LIMIT = 320
VALIDATION_SOURCE_LIST_FIELDS = (
    "missing_evidence",
    "unsupported_claims",
    "readiness_blockers",
)
SUPPORT_LABEL_COUNT_LIMIT = 12
SUPPORT_LABEL_REPORT_FIELDS = {
    "recommendation_support_status_counts",
    "recommendation_supporting_layer_counts",
}
RECOMMENDATION_HEALTH_FIELDS = {
    "recommendation_count": "recommendation_count",
    "missing_support_status": "recommendations_missing_support_status",
    "invalid_support_status": "recommendations_invalid_support_status",
    "missing_supporting_layers": "recommendations_missing_supporting_layers",
    "missing_evidence_references": "recommendations_missing_evidence_references",
    "speculative_as_supported": "recommendations_speculative_as_supported",
    "unavailable_layers": "recommendations_with_unavailable_layers",
    "invalid_layers": "recommendations_with_invalid_layers",
    "source_fact_marked_supported": (
        "recommendations_with_source_fact_supported_status"
    ),
}
RECOMMENDATION_HEALTH_REPORT_FIELDS = {
    *RECOMMENDATION_HEALTH_FIELDS.values(),
    "available_supporting_layers",
}
REFERENCE_INTEGRITY_FIELDS = {
    "artifact_references": "evidence_reference_count",
    "valid_references": "valid_evidence_reference_count",
    "invalid_references": "invalid_evidence_reference_count",
    "unverifiable_references": "unverifiable_evidence_reference_count",
    "incompatible_reference_types": "incompatible_evidence_reference_type_count",
}
SUPPORT_SCREENING_FIELDS = {
    "chunk_span": {
        "checked_recommendations": "span_chunk_support_checked_recommendation_count",
        "weak_recommendations": "span_chunk_support_weak_recommendation_count",
        "unavailable_references": "span_chunk_support_unavailable_reference_count",
    },
    "structured_evidence": {
        "checked_recommendations": (
            "structured_evidence_support_checked_recommendation_count"
        ),
        "weak_recommendations": (
            "structured_evidence_support_weak_recommendation_count"
        ),
        "unavailable_references": (
            "structured_evidence_support_unavailable_reference_count"
        ),
    },
    "source_fact_memory": {
        "checked_recommendations": "source_fact_support_checked_recommendation_count",
        "weak_recommendations": "source_fact_support_weak_recommendation_count",
        "unavailable_references": "source_fact_support_unavailable_reference_count",
    },
    "entailment": {
        "checked_recommendations": "entailment_support_checked_recommendation_count",
        "unknown_recommendations": "entailment_support_unknown_recommendation_count",
        "weak_recommendations": "entailment_support_weak_recommendation_count",
    },
}
SUPPORT_SCREENING_REPORT_FIELDS = {
    report_key
    for group_fields in SUPPORT_SCREENING_FIELDS.values()
    for report_key in group_fields.values()
}
TASK_QUALITY_SECTION_FIELDS = {
    "required_sections": "task_quality_required_sections",
    "missing_required_sections": "task_quality_missing_required_sections",
}
TASK_QUALITY_SECTION_LIST_LIMIT = 12
VALIDATION_PAYLOAD_IDENTITY_FIELDS = {
    "harness_status",
    "validation_issues",
    "validation_issue_counts",
}
VALIDATION_PAYLOAD_MARKER_FIELDS = {
    *VALIDATION_PAYLOAD_IDENTITY_FIELDS,
    *VALIDATION_SOURCE_LIST_FIELDS,
    *SUPPORT_LABEL_REPORT_FIELDS,
    *RECOMMENDATION_HEALTH_REPORT_FIELDS,
    *REFERENCE_INTEGRITY_FIELDS.values(),
    *SUPPORT_SCREENING_REPORT_FIELDS,
    *TASK_EVIDENCE_ROLE_FIELDS.values(),
    "task_quality_artifact_type",
    "task_quality_checked_artifact_count",
    *TASK_QUALITY_SECTION_FIELDS.values(),
}
PUBLIC_TRACE_PRIVATE_KEY_EXEMPTIONS = {"prompt_version"}
PUBLIC_TRACE_PRIVATE_KEY_EXACT = {
    "context",
    "context_label",
    "messages",
}
PUBLIC_TRACE_PRIVATE_KEY_PARTS = (
    "access_key",
    "accesskey",
    "api_key",
    "apikey",
    "authorization",
    "auth_key",
    "authkey",
    "bearer",
    "chain_of_thought",
    "credential",
    "hidden_reasoning",
    "internal_reasoning",
    "openai",
    "password",
    "private_key",
    "privatekey",
    "prompt",
    "raw_model_response",
    "raw_response",
    "reasoning_trace",
    "secret",
    "sk_",
    "system_prompt",
    "token",
)
PUBLIC_TRACE_PRIVATE_VALUE_LABEL_PARTS = (
    "api_key",
    "authorization",
    "bearer",
    "chain_of_thought",
    "context_label",
    "credential",
    "hidden_reasoning",
    "internal_reasoning",
    "openai",
    "password",
    "prompt",
    "raw_model_response",
    "raw_response",
    "reasoning_trace",
    "secret",
    "sk_",
    "system_prompt",
    "token",
)


def _thread_to_response(thread) -> ResearchThreadResponse:  # noqa: ANN001
    return ResearchThreadResponse(
        id=thread.id,
        owner_id=thread.owner_id,
        title=thread.title,
        collection_id=thread.collection_id,
        workspace_id=thread.workspace_id,
        selected_paper_ids=list(thread.selected_paper_ids_json or []),
        status=thread.status,
    )


def _message_to_response(message) -> ResearchMessageResponse:  # noqa: ANN001
    return ResearchMessageResponse(
        id=message.id,
        thread_id=message.thread_id,
        role=message.role,
        content=message.content,
        artifact_id=message.artifact_id,
        metadata=dict(message.metadata_json or {}),
    )


def _artifact_to_response(artifact) -> ResearchArtifactResponse:  # noqa: ANN001
    output_payload = dict(artifact.output_payload_json or {})
    return ResearchArtifactResponse(
        id=artifact.id,
        collection_id=artifact.collection_id,
        thread_id=artifact.thread_id,
        artifact_type=artifact.artifact_type,
        title=artifact.title,
        status=artifact.status,
        input_payload=dict(artifact.input_payload_json or {}),
        output_payload=output_payload,
        evidence_payload=_public_trace_payload(dict(artifact.evidence_payload_json or {})),
        model_name=artifact.model_name,
        prompt_version=artifact.prompt_version,
        error_message=redact_private_trace_text(artifact.error_message),
        is_saved=bool(output_payload.get("is_saved")),
        saved_format=(
            str(output_payload.get("saved_format"))
            if output_payload.get("saved_format") is not None
            else None
        ),
        saved_title=(
            str(output_payload.get("saved_title"))
            if output_payload.get("saved_title") is not None
            else None
        ),
    )


def _label_to_response(label) -> PaperResearchLabelResponse:  # noqa: ANN001
    return PaperResearchLabelResponse(
        id=label.id,
        collection_id=label.collection_id,
        paper_id=label.paper_id,
        user_label=label.user_label,
        inferred_label=label.inferred_label,
        inferred_signals=dict(label.inferred_signals_json or {}),
        notes=label.notes,
    )


def _intelligence_layers_summary(
    *,
    context: dict[str, Any],
    selected_item_counts: dict[str, Any],
) -> dict[str, Any]:
    layers = context.get("intelligence_layers")
    if not isinstance(layers, dict):
        layers = {}
    field_graph = layers.get("field_graph")
    if not isinstance(field_graph, dict):
        field_graph = {}
    study_brief = layers.get("study_brief")
    study_brief_version = (
        study_brief.get("version")
        if isinstance(study_brief, dict)
        else None
    )
    study_brief_count = _summary_count(
        selected_item_counts,
        "study_brief",
        1 if isinstance(study_brief, dict) else 0,
    )
    return {
        "evidence_memory": _summary_count(
            selected_item_counts,
            "evidence_memory",
            _dict_list_count(layers.get("evidence_memory")),
        ),
        "pattern_memory": _summary_count(
            selected_item_counts,
            "pattern_memory",
            _dict_list_count(layers.get("pattern_memory")),
        ),
        "source_fact_memory": _summary_count(
            selected_item_counts,
            "source_fact_memory",
            _dict_list_count(layers.get("source_fact_memory")),
        ),
        "field_graph": {
            "nodes": _summary_count(
                selected_item_counts,
                "graph_nodes",
                _dict_list_count(field_graph.get("nodes")),
            ),
            "edges": _summary_count(
                selected_item_counts,
                "graph_edges",
                _dict_list_count(field_graph.get("edges")),
            ),
        },
        "study_brief": {
            "included": study_brief_count > 0,
            "version": study_brief_version,
        },
    }


def _summary_count(
    selected_item_counts: dict[str, Any],
    key: str,
    fallback: int,
) -> int:
    value = selected_item_counts.get(key)
    if isinstance(value, int):
        return value
    return fallback


def _dict_list_count(value: Any) -> int:
    if not isinstance(value, list):
        return 0
    return len([item for item in value if isinstance(item, dict)])


def _json_object(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _json_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _validation_issue_items(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    issues: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        code = _materialization_label(item.get("code"))
        message = _validation_issue_text(item.get("message"))
        if code is None or message is None:
            continue
        issue = {
            "code": code,
            "message": message,
        }
        for key in ("category", "severity", "source"):
            label = _materialization_label(item.get(key))
            if label is not None:
                issue[key] = label
        remediation = _validation_issue_text(item.get("remediation"))
        if remediation is not None:
            issue["remediation"] = remediation
        if issue:
            issues.append(issue)
        if len(issues) >= VALIDATION_ISSUE_LIST_LIMIT:
            break
    return issues


def _validation_issue_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.strip().split())
    if not text:
        return None
    redacted = redact_private_trace_text(text)
    if (
        redacted is None
        or redacted == REDACTED_PRIVATE_TRACE_TEXT
        or _contains_private_trace_label(redacted)
    ):
        return None
    return redacted[:VALIDATION_ISSUE_TEXT_LIMIT]


def _public_validation_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.strip().split())
    if not text:
        return None
    redacted = redact_private_trace_text(text)
    if redacted is None:
        return None
    if _contains_private_trace_label(redacted):
        return REDACTED_PRIVATE_TRACE_TEXT
    return redacted[:VALIDATION_ISSUE_TEXT_LIMIT]


def _public_validation_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        text = _public_validation_text(item)
        if text is not None:
            items.append(text)
    return items


def _validation_issue_counts(issues: list[dict[str, str]]) -> dict[str, Any]:
    if not issues:
        return {}
    counts: dict[str, Any] = {"total": len(issues)}
    for issue_key, count_key in (
        ("category", "by_category"),
        ("severity", "by_severity"),
        ("code", "by_code"),
        ("source", "by_source"),
    ):
        issue_counts: dict[str, int] = {}
        for issue in issues:
            label = issue.get(issue_key)
            if label is None:
                continue
            issue_counts[label] = issue_counts.get(label, 0) + 1
        if issue_counts:
            counts[count_key] = issue_counts
    return counts


def _public_trace_payload(value: Any) -> Any:
    stripped = strip_private_trace_fields(value)
    return _public_trace_value(stripped)


def _public_trace_value(value: Any) -> Any:
    if isinstance(value, dict):
        if _is_validation_payload(value):
            return _public_validation_payload(value)
        public: dict[str, Any] = {}
        for key, item in value.items():
            if _is_private_trace_key(key):
                continue
            public[key] = _public_trace_value(item)
        return public
    if isinstance(value, list):
        return [_public_trace_value(item) for item in value]
    if isinstance(value, str):
        return redact_private_trace_text(value)
    return value


def _is_validation_payload(value: dict[str, Any]) -> bool:
    if any(key in value for key in VALIDATION_PAYLOAD_IDENTITY_FIELDS):
        return True
    nested_report = value.get("report")
    return isinstance(nested_report, dict) and any(
        key in nested_report for key in VALIDATION_PAYLOAD_IDENTITY_FIELDS
    )


def _is_private_trace_key(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    normalized = re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")
    if normalized in PUBLIC_TRACE_PRIVATE_KEY_EXEMPTIONS:
        return False
    if normalized in PUBLIC_TRACE_PRIVATE_KEY_EXACT:
        return True
    return any(part in normalized for part in PUBLIC_TRACE_PRIVATE_KEY_PARTS)


def _contains_private_trace_label(value: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    if not normalized:
        return False
    if normalized in PUBLIC_TRACE_PRIVATE_KEY_EXACT:
        return True
    return any(part in normalized for part in PUBLIC_TRACE_PRIVATE_VALUE_LABEL_PARTS)


def _public_validation_payload(report: dict[str, Any]) -> dict[str, Any]:
    public: dict[str, Any] = {}
    harness_status = _public_validation_text(report.get("harness_status"))
    if harness_status is not None:
        public["harness_status"] = harness_status
    attempt_number = _safe_nonnegative_int(report.get("attempt_number"))
    if attempt_number > 0:
        public["attempt_number"] = attempt_number
    for field_name in VALIDATION_SOURCE_LIST_FIELDS:
        if field_name in report:
            public[field_name] = _public_validation_text_list(report.get(field_name))

    validation_issues = _validation_issue_items(report.get("validation_issues"))
    if "validation_issues" in report or "validation_issue_counts" in report:
        public["validation_issues"] = validation_issues
        public["validation_issue_counts"] = _validation_issue_counts(validation_issues)

    support_label_counts = _support_label_counts(report)
    if any(support_label_counts.values()):
        public["support_label_counts"] = support_label_counts
    recommendation_health = _recommendation_health(report)
    if recommendation_health:
        public["recommendation_health"] = recommendation_health
    reference_integrity = _reference_integrity(report)
    if reference_integrity:
        public["reference_integrity"] = reference_integrity
    support_screening = _support_screening(report)
    if support_screening:
        public["support_screening"] = support_screening
    task_evidence_roles = _task_evidence_roles(report)
    if task_evidence_roles:
        public["task_evidence_roles"] = task_evidence_roles
    task_quality = _task_quality(report)
    if task_quality:
        public["task_quality"] = task_quality

    nested_report = _json_object(report.get("report"))
    if nested_report:
        public_report = _public_validation_report(nested_report)
        if public_report:
            public["report"] = public_report
    return public


def _public_residual_report(value: dict[str, Any]) -> dict[str, Any]:
    redacted = _public_residual_trace_value(strip_private_trace_fields(value))
    return redacted if isinstance(redacted, dict) else {}


def _public_residual_trace_value(value: Any) -> Any:
    if isinstance(value, dict):
        public: dict[str, Any] = {}
        for key, item in value.items():
            if key in VALIDATION_PAYLOAD_MARKER_FIELDS or _is_private_trace_key(key):
                continue
            redacted_item = _public_residual_trace_value(item)
            if redacted_item is not None:
                public[key] = redacted_item
        return public
    if isinstance(value, list):
        return [
            redacted_item
            for item in value
            if (redacted_item := _public_residual_trace_value(item)) is not None
        ]
    if isinstance(value, str):
        return _public_validation_text(value)
    return value


def _json_list_count(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _support_label_counts(report: dict[str, Any]) -> dict[str, Any]:
    support_statuses = report.get("recommendation_support_status_counts")
    supporting_layers = report.get("recommendation_supporting_layer_counts")
    return {
        "support_statuses": _safe_count_map(
            support_statuses,
            limit=SUPPORT_LABEL_COUNT_LIMIT,
        ),
        "supporting_layers": _safe_count_map(
            supporting_layers,
            limit=SUPPORT_LABEL_COUNT_LIMIT,
        ),
    }


def _safe_count_map(value: Any, *, limit: int) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    counts: dict[str, int] = {}
    for raw_key, raw_count in value.items():
        label = _materialization_label(raw_key)
        count = _safe_nonnegative_int(raw_count)
        if label is None or count <= 0 or label in counts:
            continue
        counts[label] = count
        if len(counts) >= limit:
            break
    return counts


def _recommendation_health(report: dict[str, Any]) -> dict[str, Any]:
    if not any(report_key in report for report_key in RECOMMENDATION_HEALTH_REPORT_FIELDS):
        return {}

    summary: dict[str, Any] = {
        response_key: _safe_nonnegative_int(report.get(report_key))
        for response_key, report_key in RECOMMENDATION_HEALTH_FIELDS.items()
    }
    available_layers = _safe_validation_label_list(
        report.get("available_supporting_layers"),
        limit=SUPPORT_LABEL_COUNT_LIMIT,
    )
    if available_layers:
        summary["available_supporting_layers"] = available_layers
    return summary


def _reference_integrity(report: dict[str, Any]) -> dict[str, int]:
    if not any(report_key in report for report_key in REFERENCE_INTEGRITY_FIELDS.values()):
        return {}
    return {
        response_key: _safe_nonnegative_int(report.get(report_key))
        for response_key, report_key in REFERENCE_INTEGRITY_FIELDS.items()
    }


def _support_screening(report: dict[str, Any]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for response_group, group_fields in SUPPORT_SCREENING_FIELDS.items():
        if not any(report_key in report for report_key in group_fields.values()):
            continue
        summary[response_group] = {
            response_key: _safe_nonnegative_int(report.get(report_key))
            for response_key, report_key in group_fields.items()
        }
    return summary


def _task_evidence_role_list(value: Any) -> list[str]:
    return _safe_validation_label_list(
        value,
        limit=TASK_EVIDENCE_ROLE_LIST_LIMIT,
    )


def _safe_validation_label_list(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    labels: list[str] = []
    seen: set[str] = set()
    for item in value:
        label = _materialization_label(item)
        if label is None or label in seen:
            continue
        labels.append(label)
        seen.add(label)
        if len(labels) >= limit:
            break
    return labels


def _task_evidence_roles(report: dict[str, Any]) -> dict[str, list[str]]:
    summary: dict[str, list[str]] = {}
    for response_key, report_key in TASK_EVIDENCE_ROLE_FIELDS.items():
        labels = _task_evidence_role_list(report.get(report_key))
        if labels:
            summary[response_key] = labels
    return summary


def _task_quality(report: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    artifact_type = _materialization_label(report.get("task_quality_artifact_type"))
    if artifact_type is not None:
        summary["artifact_type"] = artifact_type
    checked_artifact_count = _safe_nonnegative_int(
        report.get("task_quality_checked_artifact_count")
    )
    if checked_artifact_count > 0:
        summary["checked_artifact_count"] = checked_artifact_count
    for response_key, report_key in TASK_QUALITY_SECTION_FIELDS.items():
        labels = _safe_validation_label_list(
            report.get(report_key),
            limit=TASK_QUALITY_SECTION_LIST_LIMIT,
        )
        if labels:
            summary[response_key] = labels
    return summary


def _public_validation_report(report: dict[str, Any]) -> dict[str, Any]:
    public_report = dict(report)
    for report_key in SUPPORT_LABEL_REPORT_FIELDS:
        public_report.pop(report_key, None)
    for report_key in RECOMMENDATION_HEALTH_REPORT_FIELDS:
        public_report.pop(report_key, None)
    for report_key in REFERENCE_INTEGRITY_FIELDS.values():
        public_report.pop(report_key, None)
    for report_key in SUPPORT_SCREENING_REPORT_FIELDS:
        public_report.pop(report_key, None)
    for report_key in TASK_EVIDENCE_ROLE_FIELDS.values():
        labels = _task_evidence_role_list(public_report.get(report_key))
        if labels:
            public_report[report_key] = labels
        else:
            public_report.pop(report_key, None)
    for report_key in [
        key for key in public_report if isinstance(key, str) and key.startswith("task_quality_")
    ]:
        public_report.pop(report_key, None)
    for report_key in VALIDATION_SOURCE_LIST_FIELDS:
        public_report.pop(report_key, None)
    public_report.pop("validation_issues", None)
    public_report.pop("validation_issue_counts", None)
    return _public_residual_report(public_report)


def _retrieval_summary(
    *,
    context: dict[str, Any],
    selected_item_counts: dict[str, Any],
) -> dict[str, Any]:
    retrieval = context.get("retrieval")
    if not isinstance(retrieval, dict):
        return {}

    raw_paper_ids = retrieval.get("paper_ids")
    paper_ids = (
        [item for item in raw_paper_ids if isinstance(item, str)][:20]
        if isinstance(raw_paper_ids, list)
        else []
    )
    summary: dict[str, Any] = {
        "backend_status": (
            retrieval.get("backend_status")
            if isinstance(retrieval.get("backend_status"), str)
            else "unknown"
        ),
        "retrieval_match_count": _summary_count(
            selected_item_counts,
            "retrieval_matches",
            len(paper_ids),
        ),
        "paper_ids": paper_ids,
    }
    for field_name in (
        "chunk_hit_count",
        "figure_hit_count",
        "table_hit_count",
        "structured_entity_hit_count",
        "result_row_hit_count",
        "sql_chunk_count",
        "selected_chunk_count",
        "selected_figure_count",
        "selected_table_count",
        "selected_structured_entity_count",
        "selected_result_evidence_count",
    ):
        value = retrieval.get(field_name)
        is_valid_count = (
            isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
        )
        summary[field_name] = value if is_valid_count else 0
    return summary


def _bounded_diagnostic_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return stripped[:SELECTION_LABEL_LIMIT]


def _selection_score(value: Any) -> float | int | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return round(float(value), 4)


def _selection_features(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    features: dict[str, Any] = {}
    for key, feature_value in value.items():
        if len(features) >= SELECTION_FEATURE_LIMIT:
            break
        if not isinstance(key, str) or not key:
            continue
        normalized_key = key.lower()
        if any(part in normalized_key for part in SELECTION_FEATURE_PRIVATE_KEY_PARTS):
            continue
        if isinstance(feature_value, bool):
            features[key] = feature_value
        elif isinstance(feature_value, int) and not isinstance(feature_value, bool):
            features[key] = feature_value
        elif isinstance(feature_value, float):
            features[key] = round(feature_value, 4)
    return features


def _selection_item(
    item: Any,
    *,
    id_field: str,
    item_type: str,
    label_fields: tuple[str, ...],
    extra_fields: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    item_id = _bounded_diagnostic_text(item.get(id_field))
    if item_id is None:
        return None
    diagnostic: dict[str, Any] = {
        "item_id": item_id,
        "item_type": item_type,
    }
    for field_name in extra_fields:
        field_value = _bounded_diagnostic_text(item.get(field_name))
        if field_value is not None:
            diagnostic[field_name] = field_value
    for field_name in label_fields:
        label = _bounded_diagnostic_text(item.get(field_name))
        if label is not None:
            diagnostic["label"] = label
            break
    for field_name in ("context_role", "context_reason"):
        value = _bounded_diagnostic_text(item.get(field_name))
        if value is not None:
            diagnostic[field_name] = value
    score = _selection_score(item.get("selection_score"))
    if score is not None:
        diagnostic["selection_score"] = score
    features = _selection_features(item.get("selection_features"))
    if features:
        diagnostic["selection_features"] = features
    return diagnostic


def _selection_items(
    items: Any,
    *,
    id_field: str,
    item_type: str,
    label_fields: tuple[str, ...],
    extra_fields: tuple[str, ...] = (),
    limit: int = SELECTION_DIAGNOSTIC_LIMIT,
) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    diagnostics: list[dict[str, Any]] = []
    for item in items:
        diagnostic = _selection_item(
            item,
            id_field=id_field,
            item_type=item_type,
            label_fields=label_fields,
            extra_fields=extra_fields,
        )
        if diagnostic is None:
            continue
        diagnostics.append(diagnostic)
        if len(diagnostics) >= limit:
            break
    return diagnostics


def _selection_diagnostics(context: dict[str, Any]) -> dict[str, Any]:
    layers = context.get("intelligence_layers")
    if not isinstance(layers, dict):
        layers = {}
    field_graph = layers.get("field_graph")
    if not isinstance(field_graph, dict):
        field_graph = {}
    memory_records = [
        *_selection_items(
            layers.get("evidence_memory"),
            id_field="memory_record_id",
            item_type="evidence_memory",
            label_fields=("title", "memory_type"),
        ),
        *_selection_items(
            layers.get("pattern_memory"),
            id_field="memory_record_id",
            item_type="pattern_memory",
            label_fields=("title", "memory_type"),
        ),
        *_selection_items(
            layers.get("source_fact_memory"),
            id_field="memory_record_id",
            item_type="source_fact_memory",
            label_fields=("title", "memory_type"),
        ),
    ][:SELECTION_DIAGNOSTIC_LIMIT]
    graph_items = [
        *_selection_items(
            field_graph.get("nodes"),
            id_field="graph_node_id",
            item_type="field_graph_node",
            label_fields=("label", "node_type"),
        ),
        *_selection_items(
            field_graph.get("edges"),
            id_field="graph_edge_id",
            item_type="field_graph_edge",
            label_fields=("edge_type",),
        ),
    ][:SELECTION_DIAGNOSTIC_LIMIT]
    return {
        "papers": _selection_items(
            context.get("papers"),
            id_field="paper_id",
            item_type="paper",
            label_fields=("title", "paper_title"),
        ),
        "chunks": _selection_items(
            context.get("chunks"),
            id_field="chunk_id",
            item_type="chunk",
            label_fields=("section_title", "chunk_id"),
            extra_fields=("paper_id",),
        ),
        "memory_records": memory_records,
        "graph_items": graph_items,
        "limits": {
            "per_group": SELECTION_DIAGNOSTIC_LIMIT,
            "features_per_item": SELECTION_FEATURE_LIMIT,
        },
    }


def _safe_nonnegative_int(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return 0


def _materialization_label(value: Any) -> str | None:
    label = _bounded_diagnostic_text(value)
    if label is None:
        return None
    normalized = label.lower()
    if any(part in normalized for part in CONTEXT_MATERIALIZATION_PRIVATE_LABEL_PARTS):
        return None
    if not CONTEXT_MATERIALIZATION_SAFE_LABEL_PATTERN.fullmatch(label):
        return None
    if CONTEXT_MATERIALIZATION_SECRET_LABEL_PATTERN.search(label):
        return None
    return label


def _context_materialization_top_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    items: list[dict[str, Any]] = []
    for raw_item in value:
        if not isinstance(raw_item, dict):
            continue
        label = _materialization_label(raw_item.get("label"))
        if label is None:
            continue
        item = {"label": label}
        for count_key in CONTEXT_MATERIALIZATION_COUNT_KEYS:
            count = _safe_nonnegative_int(raw_item.get(count_key))
            if count > 0:
                item[count_key] = count
        items.append(item)
        if len(items) >= CONTEXT_MATERIALIZATION_TOP_ITEM_LIMIT:
            break
    return items


def _context_materialization_summary(context: dict[str, Any]) -> dict[str, Any]:
    materialization = context.get("context_materialization")
    if not isinstance(materialization, dict):
        return {}
    raw_summary = materialization.get("summary")
    summary = {
        key: _safe_nonnegative_int(raw_summary.get(key))
        for key in CONTEXT_MATERIALIZATION_SUMMARY_KEYS
    } if isinstance(raw_summary, dict) else {}
    top_items = {
        response_key: items
        for map_key, response_key in CONTEXT_MATERIALIZATION_MAP_KEYS.items()
        if (
            items := _context_materialization_top_items(
                materialization.get(map_key)
            )
        )
    }
    version = materialization.get("version")
    safe_version = (
        version
        if isinstance(version, str)
        and version in CONTEXT_MATERIALIZATION_PUBLIC_VERSIONS
        else "unknown"
    )
    if not summary and not top_items:
        return {}
    return {
        "version": safe_version,
        "summary": summary,
        "top_items": top_items,
    }


def _run_to_response(session: Session, run) -> ResearchAgentRunResponse:  # noqa: ANN001
    repository = ResearchAgentRunRepository(session)
    context_pack = repository.get_context_pack(run_id=run.id)
    validation_report = repository.get_validation_report(run_id=run.id)
    context_response = None
    if context_pack is not None:
        context = _json_object(context_pack.context_json)
        selected_item_counts = _json_object(context_pack.selected_item_counts_json)
        intelligence_layers_summary = _intelligence_layers_summary(
            context=context,
            selected_item_counts=selected_item_counts,
        )
        context_response = StudyContextPackResponse(
            id=context_pack.id,
            run_id=context_pack.run_id,
            attempt_number=context_pack.attempt_number,
            collection_id=context_pack.collection_id,
            workspace_id=context_pack.workspace_id,
            task_type=context_pack.task_type,
            cache_key=context_pack.cache_key,
            context_summary={
                "paper_count": _json_list_count(context.get("papers")),
                "source_count": _json_list_count(context.get("sources")),
                "task_type": context.get("task_type"),
                "intelligence_layers": intelligence_layers_summary,
            },
            intelligence_layers_summary=intelligence_layers_summary,
            retrieval_summary=_retrieval_summary(
                context=context,
                selected_item_counts=selected_item_counts,
            ),
            context_materialization_summary=_context_materialization_summary(context),
            selection_diagnostics=_selection_diagnostics(context),
            selected_item_counts=selected_item_counts,
            readiness_warnings=_json_string_list(
                context_pack.readiness_warnings_json
            ),
        )
    validation_response = None
    if validation_report is not None:
        report = _json_object(validation_report.report_json)
        public_report = _public_validation_report(report)
        validation_issues = _validation_issue_items(report.get("validation_issues"))
        validation_response = ResearchValidationReportResponse(
            id=validation_report.id,
            run_id=validation_report.run_id,
            attempt_number=validation_report.attempt_number,
            artifact_id=validation_report.artifact_id,
            harness_status=validation_report.harness_status,
            missing_evidence=_public_validation_text_list(
                validation_report.missing_evidence_json
            ),
            unsupported_claims=_public_validation_text_list(
                validation_report.unsupported_claims_json
            ),
            readiness_blockers=_public_validation_text_list(
                validation_report.readiness_blockers_json
            ),
            validation_issues=validation_issues,
            validation_issue_counts=_validation_issue_counts(validation_issues),
            support_label_counts=_support_label_counts(report),
            recommendation_health=_recommendation_health(report),
            reference_integrity=_reference_integrity(report),
            support_screening=_support_screening(report),
            task_evidence_roles=_task_evidence_roles(report),
            task_quality=_task_quality(report),
            report=public_report,
        )
    return ResearchAgentRunResponse(
        id=run.id,
        thread_id=run.thread_id,
        artifact_id=run.artifact_id,
        collection_id=run.collection_id,
        workspace_id=run.workspace_id,
        skill_id=run.skill_id,
        artifact_type=run.artifact_type,
        model_policy=run.model_policy,
        status=run.status,
        input_json=_public_trace_payload(dict(run.input_json or {})),
        model_name=run.model_name,
        error_message=redact_private_trace_text(run.error_message),
        steps=[
            ResearchAgentStepResponse(
                id=step.id,
                run_id=step.run_id,
                attempt_number=step.attempt_number,
                ordinal=step.ordinal,
                step_type=step.step_type,
                label=step.label,
                status=step.status,
                input_json=_public_trace_payload(dict(step.input_json or {})),
                output_json=_public_trace_payload(dict(step.output_json or {})),
                error_message=redact_private_trace_text(step.error_message),
            )
            for step in repository.list_steps(run_id=run.id)
        ],
        context_pack=context_response,
        validation_report=validation_response,
    )


def _ensure_collection(session: Session, collection_id: str) -> str:
    safe_collection_id = sanitize_identifier(
        collection_id,
        field_name="collection_id",
        max_length=36,
    )
    if CollectionRepository(session).get_by_id(safe_collection_id) is None:
        raise PaperbaseAPIError(
            status_code=404,
            error="collection_not_found",
            message=f"No collection found for id: {safe_collection_id}",
        )
    return safe_collection_id


def _ensure_collection_paper(session: Session, *, collection_id: str, paper_id: str) -> str:
    safe_paper_id = sanitize_identifier(paper_id, field_name="paper_id", max_length=36)
    if PaperRepository(session).get_by_id(safe_paper_id) is None:
        raise PaperbaseAPIError(
            status_code=404,
            error="paper_not_found",
            message=f"No paper found for id: {safe_paper_id}",
        )
    membership = session.execute(
        select(CollectionPaper.id).where(
            CollectionPaper.collection_id == collection_id,
            CollectionPaper.paper_id == safe_paper_id,
        )
    ).scalar_one_or_none()
    if membership is None:
        raise PaperbaseAPIError(
            status_code=400,
            error="paper_not_in_collection",
            message=f"Paper {safe_paper_id} is not in collection {collection_id}.",
        )
    return safe_paper_id


def _resolve_thread_workspace_id(
    session: Session,
    *,
    collection_id: str,
    workspace_id: str | None,
) -> str | None:
    if workspace_id is None:
        return None
    safe_workspace_id = sanitize_identifier(
        workspace_id,
        field_name="workspace_id",
        max_length=36,
    )
    workspace = WorkspaceRepository(session).get_by_id(safe_workspace_id)
    if workspace is None:
        raise PaperbaseAPIError(
            status_code=404,
            error="workspace_not_found",
            message=f"No workspace found for id: {safe_workspace_id}",
        )
    if workspace.collection_id is not None and workspace.collection_id != collection_id:
        raise PaperbaseAPIError(
            status_code=400,
            error="workspace_collection_mismatch",
            message="Research thread workspace must belong to the selected collection.",
        )
    return safe_workspace_id


def _sanitize_selected_paper_ids(
    session: Session,
    *,
    collection_id: str,
    paper_ids: list[str],
) -> list[str]:
    safe_paper_ids: list[str] = []
    seen: set[str] = set()
    for raw_paper_id in paper_ids:
        safe_paper_id = _ensure_collection_paper(
            session,
            collection_id=collection_id,
            paper_id=raw_paper_id,
        )
        if safe_paper_id in seen:
            continue
        seen.add(safe_paper_id)
        safe_paper_ids.append(safe_paper_id)
    return safe_paper_ids


def _infer_paper_research_signals(
    session: Session,
    *,
    collection_id: str,
    paper_id: str,
) -> tuple[str, dict[str, Any]]:
    _ = collection_id
    parsed_section_count = int(
        session.execute(
            select(func.count(Section.id)).where(Section.paper_id == paper_id)
        ).scalar_one()
    )
    completed_extraction_count = int(
        session.execute(
            select(func.count(ExtractionRun.id)).where(
                ExtractionRun.paper_id == paper_id,
                ExtractionRun.status == "completed",
            )
        ).scalar_one()
    )
    result_count = int(
        session.execute(
            select(func.count(ResultRow.id)).where(ResultRow.paper_id == paper_id)
        ).scalar_one()
    )
    limitation_count = int(
        session.execute(
            select(func.count(Limitation.id)).where(Limitation.paper_id == paper_id)
        ).scalar_one()
    )
    engineering_trick_count = int(
        session.execute(
            select(func.count(EngineeringTrick.id)).where(EngineeringTrick.paper_id == paper_id)
        ).scalar_one()
    )
    design_strength_score = min(
        100,
        parsed_section_count * 5
        + completed_extraction_count * 20
        + result_count * 10
        + engineering_trick_count * 5
        + limitation_count * 3,
    )
    inferred_label = "strong_design" if design_strength_score >= 40 else "neutral"
    return inferred_label, {
        "parsed_section_count": parsed_section_count,
        "completed_extraction_count": completed_extraction_count,
        "result_count": result_count,
        "limitation_count": limitation_count,
        "engineering_trick_count": engineering_trick_count,
        "design_strength_score": design_strength_score,
    }


def _artifact_title(artifact_type: str) -> str:
    titles = {
        "field_patterns": "Field patterns",
        "hypotheses": "Hypotheses",
        "literature_review": "Literature review",
        "comparison": "Comparison",
        "experiment_plan": "Experiment plan",
        "critique": "Critique",
        "experiment_backlog": "Experiment backlog",
        "benchmark_plan": "Benchmark plan",
        "revision_plan": "Revision plan",
        "assumption_map": "Assumption map",
    }
    return titles.get(artifact_type, "Research artifact")


def _infer_artifact_type(message: str) -> str:
    normalized = message.casefold()
    if "literature review" in normalized or "synthesize" in normalized or "themes" in normalized:
        return "literature_review"
    if any(term in normalized for term in ("compare", "comparison", "contrast", "rank", "ranking")):
        return "comparison"
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


def _sanitize_source_ids(
    session: Session,
    *,
    workspace_id: str | None,
    source_ids: list[str],
) -> list[str]:
    if not source_ids:
        return []
    if workspace_id is None:
        raise PaperbaseAPIError(
            status_code=422,
            error="workspace_required_for_sources",
            message="Study sources require a research thread linked to a study.",
        )
    repository = WorkspaceRepository(session)
    safe_source_ids: list[str] = []
    for source_id in source_ids:
        safe_source_id = sanitize_identifier(source_id, field_name="source_id", max_length=36)
        source = repository.get_source(safe_source_id)
        if source is None or source.workspace_id != workspace_id:
            raise PaperbaseAPIError(
                status_code=404,
                error="study_source_not_found",
                message=f"No study source found for id: {safe_source_id}",
            )
        safe_source_ids.append(safe_source_id)
    return safe_source_ids


def _collection_processing_counts(
    session: Session,
    *,
    collection_id: str,
) -> dict[str, int]:
    member_paper_ids = select(CollectionPaper.paper_id).where(
        CollectionPaper.collection_id == collection_id
    )
    paper_count = int(
        session.execute(
            select(func.count()).select_from(CollectionPaper).where(
                CollectionPaper.collection_id == collection_id
            )
        ).scalar_one()
    )
    parsed_count = int(
        session.execute(
            select(func.count(func.distinct(Section.paper_id))).where(Section.paper_id.in_(member_paper_ids))
        ).scalar_one()
    )
    extracted_count = int(
        session.execute(
            select(func.count(func.distinct(ExtractionRun.paper_id))).where(
                ExtractionRun.paper_id.in_(member_paper_ids),
                ExtractionRun.status == "completed",
            )
        ).scalar_one()
    )
    return {
        "paper_count": paper_count,
        "parsed_count": parsed_count,
        "extracted_count": extracted_count,
    }


def _suggestions_for_counts(counts: dict[str, int]) -> list[ResearchSuggestionResponse]:
    readiness = (
        "evidence_ready"
        if counts["extracted_count"] > 0
        else "text_ready"
        if counts["parsed_count"] > 0
        else "imported"
    )
    suggestions = [
        ResearchSuggestionResponse(
            id="literature_review_synthesis",
            label="Synthesize themes",
            instruction=(
                "Synthesize this collection into major themes, consensus points, "
                "controversies, research gaps, and future directions."
            ),
            skill_id="literature_review",
            artifact_type="literature_review",
            readiness=readiness,
        ),
        ResearchSuggestionResponse(
            id="evidence_quality_check",
            label="Check evidence coverage",
            instruction=(
                "Check evidence coverage, unsupported claims, missing context, "
                "and reproducibility risks for the current study."
            ),
            skill_id="quality_harness",
            artifact_type="critique",
            readiness=readiness,
        ),
    ]
    if counts["extracted_count"] > 0:
        suggestions.insert(
            1,
            ResearchSuggestionResponse(
                id="benchmark_ablation_plan",
                label="Plan benchmarks",
                instruction=(
                    "Build a benchmark, baseline, metric, and ablation plan grounded "
                    "in the strongest extracted evidence."
                ),
                skill_id="benchmark_planning",
                artifact_type="benchmark_plan",
                readiness=readiness,
            ),
        )
    else:
        suggestions.append(
            ResearchSuggestionResponse(
                id="benchmark_ablation_plan",
                label="Prepare benchmark plan",
                instruction=(
                    "Draft a benchmark and ablation plan from available paper metadata, "
                    "and list what extraction should add next."
                ),
                skill_id="benchmark_planning",
                artifact_type="benchmark_plan",
                readiness=readiness,
            )
        )
    return suggestions


def _find_matching_active_research_job(
    session: Session,
    *,
    thread_id: str,
    collection_id: str,
    message: str,
    artifact_type: str,
    selected_paper_ids: list[str],
    source_ids: list[str],
) -> BackgroundJob | None:
    jobs = session.execute(
        select(BackgroundJob)
        .where(
            BackgroundJob.job_type == "research_agent_run",
            BackgroundJob.status.in_(ACTIVE_JOB_STATUSES),
        )
        .order_by(BackgroundJob.created_at.desc(), BackgroundJob.id.desc())
    ).scalars()
    for job in jobs:
        payload = dict(job.payload_json or {})
        if (
            payload.get("thread_id") == thread_id
            and payload.get("collection_id") == collection_id
            and payload.get("user_message") == message
            and payload.get("artifact_type") == artifact_type
            and list(payload.get("selected_paper_ids") or []) == selected_paper_ids
            and list(payload.get("source_ids") or []) == source_ids
        ):
            return job
    return None


def _find_retry_user_message(
    repository: ResearchRepository,
    *,
    thread_id: str,
    message_text: str,
) -> ResearchMessage | None:
    messages = list(repository.list_messages(thread_id=thread_id))
    for message in reversed(messages):
        if message.role == "user" and message.content == message_text:
            return message
    return None


def _retry_output_payload(output_payload: dict[str, Any]) -> dict[str, Any]:
    preserved: dict[str, Any] = {}
    for key in ("is_saved", "saved_format", "saved_title"):
        value = output_payload.get(key)
        if value is not None:
            preserved[key] = value
    return preserved


def _find_active_retry_job_for_artifact(
    session: Session,
    *,
    artifact_id: str,
) -> BackgroundJob | None:
    jobs = session.execute(
        select(BackgroundJob)
        .where(
            BackgroundJob.job_type == "research_agent_run",
            BackgroundJob.status.in_(ACTIVE_JOB_STATUSES),
        )
        .order_by(BackgroundJob.created_at.desc(), BackgroundJob.id.desc())
    ).scalars()
    for job in jobs:
        payload = dict(job.payload_json or {})
        if payload.get("artifact_id") == artifact_id and isinstance(
            payload.get("retry_of_run_id"),
            str,
        ):
            return job
    return None


def _active_research_job_response(
    session: Session,
    repository: ResearchRepository,
    *,
    job: BackgroundJob,
) -> ResearchMessageJobResponse | None:
    payload = dict(job.payload_json or {})
    artifact_id = payload.get("artifact_id")
    message_id = payload.get("message_id")
    if not isinstance(artifact_id, str) or not isinstance(message_id, str):
        return None
    artifact = repository.get_artifact(artifact_id)
    message = session.get(ResearchMessage, message_id)
    if artifact is None or message is None:
        return None
    return ResearchMessageJobResponse(
        data=ResearchMessageJobResponseData(
            message=_message_to_response(message),
            artifact=_artifact_to_response(artifact),
            job=background_job_to_response(job),
            run_id=str(payload["run_id"]) if payload.get("run_id") is not None else None,
        )
    )


@router.get("/api/v1/research/suggestions", response_model=ResearchSuggestionsResponse)
def list_research_suggestions(
    collection_id: str = Query(..., min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> ResearchSuggestionsResponse:
    safe_collection_id = _ensure_collection(session, collection_id)
    counts = _collection_processing_counts(session, collection_id=safe_collection_id)
    return ResearchSuggestionsResponse(data=_suggestions_for_counts(counts))


@router.get("/api/v1/research/threads", response_model=ResearchThreadsResponse)
def list_research_threads(
    collection_id: str | None = Query(None, min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> ResearchThreadsResponse:
    safe_collection_id = (
        _ensure_collection(session, collection_id) if collection_id is not None else None
    )
    threads = ResearchRepository(session).list_threads(collection_id=safe_collection_id)
    return ResearchThreadsResponse(data=[_thread_to_response(thread) for thread in threads])


@router.post(
    "/api/v1/research/threads",
    response_model=SingleResearchThreadResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_research_thread(
    payload: ResearchThreadCreateRequest,
    session: Session = Depends(get_session),
) -> SingleResearchThreadResponse:
    safe_collection_id = _ensure_collection(session, payload.collection_id)
    selected_paper_ids = _sanitize_selected_paper_ids(
        session,
        collection_id=safe_collection_id,
        paper_ids=list(payload.selected_paper_ids),
    )
    workspace_id = _resolve_thread_workspace_id(
        session,
        collection_id=safe_collection_id,
        workspace_id=payload.workspace_id,
    )
    thread = ResearchRepository(session).create_thread(
        owner_id=sanitize_user_text(payload.owner_id, field_name="owner_id", max_length=128),
        title=sanitize_user_text(payload.title, field_name="title", max_length=255),
        collection_id=safe_collection_id,
        workspace_id=workspace_id,
        selected_paper_ids=selected_paper_ids,
    )
    return SingleResearchThreadResponse(data=_thread_to_response(thread))


@router.get(
    "/api/v1/research/threads/{thread_id}",
    response_model=ResearchThreadDetailResponse,
)
def fetch_research_thread(
    thread_id: str = Path(..., min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> ResearchThreadDetailResponse:
    safe_thread_id = sanitize_identifier(thread_id, field_name="thread_id", max_length=36)
    repository = ResearchRepository(session)
    thread = repository.get_thread(safe_thread_id)
    if thread is None:
        raise PaperbaseAPIError(
            status_code=404,
            error="research_thread_not_found",
            message=f"No research thread found for id: {safe_thread_id}",
        )
    artifacts = [
        artifact
        for artifact in repository.list_artifacts(collection_id=thread.collection_id)
        if artifact.thread_id == thread.id
    ]
    return ResearchThreadDetailResponse(
        data=ResearchThreadDetailResponseData(
            **_thread_to_response(thread).model_dump(),
            messages=[
                _message_to_response(message)
                for message in repository.list_messages(thread_id=thread.id)
            ],
            artifacts=[_artifact_to_response(artifact) for artifact in artifacts],
        )
    )


@router.post(
    "/api/v1/research/threads/{thread_id}/messages",
    response_model=ResearchMessageJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_research_message(
    payload: ResearchMessageCreateRequest,
    request: Request,
    thread_id: str = Path(..., min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> ResearchMessageJobResponse:
    safe_thread_id = sanitize_identifier(thread_id, field_name="thread_id", max_length=36)
    repository = ResearchRepository(session)
    thread = repository.get_thread(safe_thread_id)
    if thread is None:
        raise PaperbaseAPIError(
            status_code=404,
            error="research_thread_not_found",
            message=f"No research thread found for id: {safe_thread_id}",
        )

    message_text = sanitize_user_text(payload.message, field_name="message", max_length=20000)
    inferred_artifact_type = payload.artifact_type or _infer_artifact_type(message_text)
    skill_id = select_research_skill(
        message_text,
        suggestion_id=payload.suggestion_id,
        artifact_type=inferred_artifact_type,
    )
    if payload.artifact_type is not None:
        artifact_type = payload.artifact_type
    elif payload.suggestion_id is not None or skill_id in {"literature_review", "quality_harness"}:
        artifact_type = artifact_type_for_skill(skill_id, fallback=inferred_artifact_type)
    else:
        artifact_type = inferred_artifact_type
    selected_paper_ids = list(thread.selected_paper_ids_json or [])
    source_ids = _sanitize_source_ids(
        session,
        workspace_id=thread.workspace_id,
        source_ids=list(payload.source_ids),
    )
    active_job = _find_matching_active_research_job(
        session,
        thread_id=thread.id,
        collection_id=thread.collection_id,
        message=message_text,
        artifact_type=artifact_type,
        selected_paper_ids=selected_paper_ids,
        source_ids=source_ids,
    )
    if active_job is not None:
        active_payload = dict(active_job.payload_json or {})
        artifact_id = active_payload.get("artifact_id")
        message_id = active_payload.get("message_id")
        artifact = repository.get_artifact(artifact_id) if isinstance(artifact_id, str) else None
        if artifact is not None and isinstance(message_id, str):
            stored_message = session.get(ResearchMessage, message_id)
            if stored_message is not None:
                return ResearchMessageJobResponse(
                    data=ResearchMessageJobResponseData(
                        message=_message_to_response(stored_message),
                        artifact=_artifact_to_response(artifact),
                        job=background_job_to_response(active_job),
                        run_id=(
                            str(active_payload.get("run_id"))
                            if active_payload.get("run_id") is not None
                            else None
                        ),
                    )
                )

    user_message = repository.create_message(
        thread_id=thread.id,
        role="user",
        content=message_text,
    )
    artifact = repository.create_artifact(
        collection_id=thread.collection_id,
        thread_id=thread.id,
        artifact_type=artifact_type,
        title=_artifact_title(artifact_type),
        status="pending",
        input_payload={
            "message": message_text,
            "selected_paper_ids": selected_paper_ids,
            "source_ids": source_ids,
            "skill_id": skill_id,
            "suggestion_id": payload.suggestion_id,
        },
        prompt_version=PROMPT_VERSION,
    )
    policy = policy_for_skill(skill_id)
    run = ResearchAgentRunRepository(session).create_run(
        thread_id=thread.id,
        artifact_id=artifact.id,
        collection_id=thread.collection_id,
        workspace_id=thread.workspace_id,
        skill_id=skill_id,
        artifact_type=artifact_type,
        model_policy=policy.model_policy,
        input_json={
            "message": message_text,
            "selected_paper_ids": selected_paper_ids,
            "source_ids": source_ids,
            "suggestion_id": payload.suggestion_id,
        },
    )
    job = create_background_job(
        session_factory=get_session_factory(request),
        job_type="research_agent_run",
        payload_json={
            "run_id": run.id,
            "thread_id": thread.id,
            "message_id": user_message.id,
            "artifact_id": artifact.id,
            "collection_id": thread.collection_id,
            "user_message": message_text,
            "artifact_type": artifact_type,
            "skill_id": skill_id,
            "suggestion_id": payload.suggestion_id,
            "selected_paper_ids": selected_paper_ids,
            "workspace_id": thread.workspace_id,
            "source_ids": source_ids,
        },
        dispatcher=request.app.state.job_dispatcher,
        project_id=get_project_id(request),
    )
    return ResearchMessageJobResponse(
        data=ResearchMessageJobResponseData(
            message=_message_to_response(user_message),
            artifact=_artifact_to_response(artifact),
            job=background_job_to_response(job),
            run_id=run.id,
        )
    )


@router.get(
    "/api/v1/research/runs/{run_id}",
    response_model=SingleResearchAgentRunResponse,
)
def fetch_research_run(
    run_id: str = Path(..., min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> SingleResearchAgentRunResponse:
    safe_run_id = sanitize_identifier(run_id, field_name="run_id", max_length=36)
    run = ResearchAgentRunRepository(session).get_run(safe_run_id)
    if run is None:
        raise PaperbaseAPIError(
            status_code=404,
            error="research_run_not_found",
            message=f"No research agent run found for id: {safe_run_id}",
        )
    return SingleResearchAgentRunResponse(data=_run_to_response(session, run))


@router.get(
    "/api/v1/research/artifacts/{artifact_id}/run",
    response_model=SingleResearchAgentRunResponse,
)
def fetch_research_artifact_run(
    artifact_id: str = Path(..., min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> SingleResearchAgentRunResponse:
    safe_artifact_id = sanitize_identifier(artifact_id, field_name="artifact_id", max_length=36)
    run = ResearchAgentRunRepository(session).get_run_for_artifact(safe_artifact_id)
    if run is None:
        raise PaperbaseAPIError(
            status_code=404,
            error="research_run_not_found",
            message=f"No research agent run found for artifact id: {safe_artifact_id}",
        )
    return SingleResearchAgentRunResponse(data=_run_to_response(session, run))


@router.get("/api/v1/research/artifacts", response_model=ResearchArtifactsResponse)
def list_research_artifacts(
    collection_id: str | None = Query(None, min_length=1, max_length=36),
    saved_only: bool = Query(False),
    session: Session = Depends(get_session),
) -> ResearchArtifactsResponse:
    safe_collection_id = (
        _ensure_collection(session, collection_id) if collection_id is not None else None
    )
    artifacts = ResearchRepository(session).list_artifacts(collection_id=safe_collection_id)
    if saved_only:
        artifacts = [
            artifact
            for artifact in artifacts
            if bool((artifact.output_payload_json or {}).get("is_saved"))
        ]
    return ResearchArtifactsResponse(
        data=[_artifact_to_response(artifact) for artifact in artifacts]
    )


@router.post(
    "/api/v1/research/artifacts/{artifact_id}/retry",
    response_model=ResearchMessageJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def retry_research_artifact(
    request: Request,
    artifact_id: str = Path(..., min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> ResearchMessageJobResponse:
    safe_artifact_id = sanitize_identifier(artifact_id, field_name="artifact_id", max_length=36)
    repository = ResearchRepository(session)
    artifact = repository.get_artifact(safe_artifact_id)
    if artifact is None:
        raise PaperbaseAPIError(
            status_code=404,
            error="research_artifact_not_found",
            message=f"No research artifact found for id: {safe_artifact_id}",
        )
    active_job = _find_active_retry_job_for_artifact(
        session,
        artifact_id=safe_artifact_id,
    )
    if active_job is not None:
        active_response = _active_research_job_response(
            session,
            repository,
            job=active_job,
        )
        if active_response is not None:
            return active_response

    run_repository = ResearchAgentRunRepository(session)
    latest_run = run_repository.get_run_for_artifact(safe_artifact_id)
    output_payload = dict(artifact.output_payload_json or {})
    original_error_message = artifact.error_message
    original_prompt_version = artifact.prompt_version
    is_retryable_model_block = (
        artifact.status == "blocked"
        and output_payload.get("model_required") is True
        and latest_run is not None
        and latest_run.status == "blocked"
        and latest_run.model_policy == "required"
    )
    if not is_retryable_model_block:
        session.rollback()
        raise PaperbaseAPIError(
            status_code=409,
            error="research_artifact_not_retryable",
            message="Only blocked model-required research artifacts can be retried.",
        )
    if artifact.thread_id is None:
        raise PaperbaseAPIError(
            status_code=409,
            error="research_artifact_not_retryable",
            message="Retry requires a research thread with the original user message.",
        )

    input_payload = dict(artifact.input_payload_json or {})
    run_input = dict(latest_run.input_json or {})
    message_text = str(
        run_input.get("message") or input_payload.get("message") or ""
    ).strip()
    if not message_text:
        raise PaperbaseAPIError(
            status_code=409,
            error="research_artifact_not_retryable",
            message="Retry requires the original research message.",
        )
    user_message = _find_retry_user_message(
        repository,
        thread_id=artifact.thread_id,
        message_text=message_text,
    )
    if user_message is None:
        raise PaperbaseAPIError(
            status_code=409,
            error="research_artifact_not_retryable",
            message="Retry requires the original research message.",
        )

    selected_paper_ids = [
        str(item)
        for item in list(
            run_input.get("selected_paper_ids")
            or input_payload.get("selected_paper_ids")
            or []
        )
    ]
    source_ids = [
        str(item)
        for item in list(run_input.get("source_ids") or input_payload.get("source_ids") or [])
    ]
    suggestion_id_value = input_payload.get("suggestion_id") or run_input.get("suggestion_id")
    suggestion_id = str(suggestion_id_value) if suggestion_id_value is not None else None

    transition = session.execute(
        update(ResearchArtifact)
        .where(
            ResearchArtifact.id == safe_artifact_id,
            ResearchArtifact.status == "blocked",
        )
        .values(
            status="pending",
            output_payload_json=_retry_output_payload(output_payload),
            prompt_version=artifact.prompt_version or PROMPT_VERSION,
            error_message=None,
        )
    )
    if transition.rowcount != 1:
        session.rollback()
        active_job = _find_active_retry_job_for_artifact(
            session,
            artifact_id=safe_artifact_id,
        )
        if active_job is not None:
            active_response = _active_research_job_response(
                session,
                repository,
                job=active_job,
            )
            if active_response is not None:
                return active_response
        raise PaperbaseAPIError(
            status_code=409,
            error="research_artifact_not_retryable",
            message="This research artifact is already retrying or no longer blocked.",
        )
    retry_run = ResearchAgentRun(
        thread_id=artifact.thread_id,
        artifact_id=artifact.id,
        collection_id=artifact.collection_id,
        workspace_id=latest_run.workspace_id,
        skill_id=latest_run.skill_id,
        artifact_type=artifact.artifact_type,
        model_policy=latest_run.model_policy,
        status="pending",
        input_json={
            "message": message_text,
            "selected_paper_ids": selected_paper_ids,
            "source_ids": source_ids,
            "suggestion_id": suggestion_id,
        },
    )
    session.add(retry_run)
    session.flush()
    job = BackgroundJob(
        job_type="research_agent_run",
        status="pending",
        payload_json={
            "run_id": retry_run.id,
            "thread_id": artifact.thread_id,
            "message_id": user_message.id,
            "artifact_id": artifact.id,
            "collection_id": artifact.collection_id,
            "user_message": message_text,
            "artifact_type": artifact.artifact_type,
            "skill_id": latest_run.skill_id,
            "suggestion_id": suggestion_id,
            "selected_paper_ids": selected_paper_ids,
            "workspace_id": latest_run.workspace_id,
            "source_ids": source_ids,
            "retry_of_run_id": latest_run.id,
        },
    )
    session.add(job)
    session.commit()
    session.refresh(retry_run)
    session.refresh(job)
    session.refresh(artifact)
    try:
        if request.app.state.job_dispatcher is not None:
            request.app.state.job_dispatcher.dispatch(
                job.id,
                project_id=get_project_id(request),
            )
    except Exception as exc:
        session.delete(retry_run)
        artifact.status = "blocked"
        artifact.output_payload_json = output_payload
        artifact.error_message = original_error_message
        artifact.prompt_version = original_prompt_version
        job.status = "failed"
        job.error_message = "dispatch failed: unable to dispatch research retry job"
        job.finished_at = _utc_now()
        session.commit()
        raise PaperbaseAPIError(
            status_code=503,
            error="job_dispatch_failed",
            message=f"Unable to dispatch background job {job.id}.",
        ) from exc
    session.refresh(artifact)
    return ResearchMessageJobResponse(
        data=ResearchMessageJobResponseData(
            message=_message_to_response(user_message),
            artifact=_artifact_to_response(artifact),
            job=background_job_to_response(job),
            run_id=retry_run.id,
        )
    )


@router.get(
    "/api/v1/research/artifacts/{artifact_id}",
    response_model=SingleResearchArtifactResponse,
)
def fetch_research_artifact(
    artifact_id: str = Path(..., min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> SingleResearchArtifactResponse:
    safe_artifact_id = sanitize_identifier(artifact_id, field_name="artifact_id", max_length=36)
    artifact = ResearchRepository(session).get_artifact(safe_artifact_id)
    if artifact is None:
        raise PaperbaseAPIError(
            status_code=404,
            error="research_artifact_not_found",
            message=f"No research artifact found for id: {safe_artifact_id}",
        )
    return SingleResearchArtifactResponse(data=_artifact_to_response(artifact))


@router.patch(
    "/api/v1/research/artifacts/{artifact_id}",
    response_model=SingleResearchArtifactResponse,
)
def update_research_artifact(
    payload: ResearchArtifactPatchRequest,
    artifact_id: str = Path(..., min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> SingleResearchArtifactResponse:
    safe_artifact_id = sanitize_identifier(artifact_id, field_name="artifact_id", max_length=36)
    repository = ResearchRepository(session)
    artifact = repository.get_artifact(safe_artifact_id)
    if artifact is None:
        raise PaperbaseAPIError(
            status_code=404,
            error="research_artifact_not_found",
            message=f"No research artifact found for id: {safe_artifact_id}",
        )
    output_payload = dict(artifact.output_payload_json or {})
    if payload.is_saved is not None:
        output_payload["is_saved"] = payload.is_saved
    if payload.saved_format is not None:
        output_payload["saved_format"] = sanitize_user_text(
            payload.saved_format,
            field_name="saved_format",
            max_length=64,
        )
    if payload.saved_title is not None:
        output_payload["saved_title"] = sanitize_user_text(
            payload.saved_title,
            field_name="saved_title",
            max_length=255,
        )

    updated = repository.update_artifact(
        safe_artifact_id,
        title=(
            sanitize_user_text(payload.title, field_name="title", max_length=255)
            if payload.title is not None
            else None
        ),
        status=None,
        output_payload=(
            output_payload
            if output_payload != dict(artifact.output_payload_json or {})
            else None
        ),
    )
    return SingleResearchArtifactResponse(data=_artifact_to_response(updated))


@router.get(
    "/api/v1/collections/{collection_id}/research-labels",
    response_model=PaperResearchLabelsResponse,
)
def list_paper_research_labels(
    collection_id: str = Path(..., min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> PaperResearchLabelsResponse:
    safe_collection_id = _ensure_collection(session, collection_id)
    labels = ResearchRepository(session).list_labels(collection_id=safe_collection_id)
    return PaperResearchLabelsResponse(data=[_label_to_response(label) for label in labels])


@router.patch(
    "/api/v1/collections/{collection_id}/papers/{paper_id}/research-label",
    response_model=SinglePaperResearchLabelResponse,
)
def update_paper_research_label(
    payload: PaperResearchLabelPatchRequest,
    collection_id: str = Path(..., min_length=1, max_length=36),
    paper_id: str = Path(..., min_length=1, max_length=36),
    session: Session = Depends(get_session),
) -> SinglePaperResearchLabelResponse:
    safe_collection_id = _ensure_collection(session, collection_id)
    safe_paper_id = _ensure_collection_paper(
        session,
        collection_id=safe_collection_id,
        paper_id=paper_id,
    )
    inferred_label, inferred_signals = _infer_paper_research_signals(
        session,
        collection_id=safe_collection_id,
        paper_id=safe_paper_id,
    )
    label = ResearchRepository(session).upsert_label(
        collection_id=safe_collection_id,
        paper_id=safe_paper_id,
        user_label=payload.user_label,
        inferred_label=inferred_label,
        inferred_signals=inferred_signals,
        notes=(
            sanitize_user_text(
                payload.notes,
                field_name="notes",
                max_length=10000,
                allow_empty=True,
            )
            if payload.notes is not None
            else None
        ),
    )
    return SinglePaperResearchLabelResponse(data=_label_to_response(label))
