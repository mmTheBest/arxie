"""Schema contracts for model-backed Paperbase research-agent artifacts."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from paperbase.research.task_quality import task_section_has_meaningful_content

_REFERENCE_TARGET_FIELDS = (
    "paper_id",
    "source_id",
    "study_source_id",
    "study_brief_id",
    "chunk_id",
    "evidence_span_id",
    "span_id",
    "evidence_id",
    "artifact_id",
    "run_id",
    "figure_id",
    "table_id",
    "entity_id",
    "result_id",
    "result_row_id",
    "dataset_id",
    "method_id",
    "metric_id",
    "finding_id",
    "limitation_id",
    "research_design_id",
    "memory_record_id",
    "graph_node_id",
    "graph_edge_id",
)
_PRIVATE_MODEL_OUTPUT_KEYS = frozenset(
    {
        "chain_of_thought",
        "hidden_chain_of_thought",
        "internal_reasoning",
        "hidden_reasoning",
        "reasoning_trace",
        "prompt",
        "prompt_payload",
        "system_prompt",
        "messages",
        "context",
        "deterministic_preview",
        "raw_response",
        "raw_model_response",
    }
)
_PRIVATE_TRACE_TEXT_MARKERS = (
    "chain_of_thought",
    "chain-of-thought",
    "chain of thought",
    "hidden_chain_of_thought",
    "internal_reasoning",
    "internal reasoning",
    "hidden_reasoning",
    "hidden reasoning",
    "reasoning_trace",
    "prompt_payload",
    "system_prompt",
    "system prompt",
    "deterministic_preview",
    "raw_response",
    "raw response",
    "raw_model_response",
    "raw model response",
    "return json only",
)
_PRIVATE_TRACE_KEY_PATTERN = re.compile(
    r"""(?ix)
    ['"]?
    (
        chain_of_thought
        |hidden_chain_of_thought
        |internal_reasoning
        |hidden_reasoning
        |reasoning_trace
        |prompt
        |prompt_payload
        |system_prompt
        |messages
        |context
        |deterministic_preview
        |raw_response
        |raw_model_response
    )
    ['"]?
    \s*[:=]
    """
)
REDACTED_PRIVATE_TRACE_TEXT = "Trace detail redacted."
_RUNTIME_OUTPUT_KEYS = frozenset(
    {
        "artifact_type",
        "skill_id",
        "model_backed",
        "model_name",
        "model_required",
        "setup_error",
        "schema_validation",
    }
)
_DISALLOWED_MODEL_OUTPUT_KEYS = _PRIVATE_MODEL_OUTPUT_KEYS | _RUNTIME_OUTPUT_KEYS


class ResearchModelOutputContractError(ValueError):
    """Raised when a model-backed artifact cannot satisfy its output schema."""


class _FlexibleModel(BaseModel):
    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)


class EvidenceReference(_FlexibleModel):
    reference_type: str = Field(min_length=1)
    label: str | None = Field(default=None, min_length=1)
    paper_id: str | None = Field(default=None, min_length=1)
    source_id: str | None = Field(default=None, min_length=1)
    study_source_id: str | None = Field(default=None, min_length=1)
    study_brief_id: str | None = Field(default=None, min_length=1)
    chunk_id: str | None = Field(default=None, min_length=1)
    evidence_span_id: str | None = Field(default=None, min_length=1)
    span_id: str | None = Field(default=None, min_length=1)
    evidence_id: str | None = Field(default=None, min_length=1)
    artifact_id: str | None = Field(default=None, min_length=1)
    run_id: str | None = Field(default=None, min_length=1)
    figure_id: str | None = Field(default=None, min_length=1)
    table_id: str | None = Field(default=None, min_length=1)
    entity_id: str | None = Field(default=None, min_length=1)
    result_id: str | None = Field(default=None, min_length=1)
    result_row_id: str | None = Field(default=None, min_length=1)
    dataset_id: str | None = Field(default=None, min_length=1)
    method_id: str | None = Field(default=None, min_length=1)
    metric_id: str | None = Field(default=None, min_length=1)
    finding_id: str | None = Field(default=None, min_length=1)
    limitation_id: str | None = Field(default=None, min_length=1)
    research_design_id: str | None = Field(default=None, min_length=1)
    memory_record_id: str | None = Field(default=None, min_length=1)
    graph_node_id: str | None = Field(default=None, min_length=1)
    graph_edge_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _require_reference_anchor(self) -> EvidenceReference:
        if _has_reference_anchor(self.label):
            return self
        for field_name in _REFERENCE_TARGET_FIELDS:
            if _has_reference_anchor(getattr(self, field_name, None)):
                return self
        extra_fields = self.model_extra or {}
        for field_name in _REFERENCE_TARGET_FIELDS:
            if _has_reference_anchor(extra_fields.get(field_name)):
                return self
        raise ValueError(
            "Evidence reference requires reference_type plus at least one "
            "target id or label."
        )


class Recommendation(_FlexibleModel):
    title: str = Field(min_length=1)
    detail: str | None = None
    support_status: str | None = None
    supporting_layers: list[str] = Field(default_factory=list)
    evidence_references: list[EvidenceReference] = Field(default_factory=list)


class ResearchArtifactOutput(_FlexibleModel):
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    recommendations: list[Recommendation] = Field(default_factory=list)
    evidence_references: list[EvidenceReference] = Field(default_factory=list)
    assumptions_or_inferences: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class LiteratureReviewOutput(ResearchArtifactOutput):
    themes: list[dict[str, Any] | str] = Field(default_factory=list)
    consensus: list[str] = Field(default_factory=list)
    controversies: list[str] = Field(default_factory=list)
    research_gaps: list[str] = Field(default_factory=list)
    future_directions: list[str] = Field(default_factory=list)


class CritiqueOutput(ResearchArtifactOutput):
    quality_checks: list[dict[str, Any]] = Field(default_factory=list)


class ComparisonOutput(ResearchArtifactOutput):
    comparison_rows: list[dict[str, Any]] = Field(default_factory=list)
    method_families: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)


class BenchmarkPlanOutput(ResearchArtifactOutput):
    benchmark_recommendations: list[Recommendation | str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    metrics_or_result_logic: list[str] = Field(default_factory=list)
    candidate_baselines: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _require_task_sections(self) -> BenchmarkPlanOutput:
        missing_sections = _missing_required_output_sections(
            self,
            (
                "benchmark_recommendations",
                "datasets",
                "metrics_or_result_logic",
                "candidate_baselines",
            ),
        )
        if missing_sections:
            raise ValueError(
                "Benchmark plan requires non-empty task section(s): "
                f"{', '.join(missing_sections)}"
            )
        return self


class ExperimentPlanOutput(ResearchArtifactOutput):
    objective: str | None = None
    baselines: list[str] = Field(default_factory=list)
    ablations: list[str] = Field(default_factory=list)
    controls: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _require_task_sections(self) -> ExperimentPlanOutput:
        missing_sections = _missing_required_output_sections(
            self,
            ("objective", "baselines", "ablations", "controls"),
        )
        if missing_sections:
            raise ValueError(
                "Experiment plan requires non-empty task section(s): "
                f"{', '.join(missing_sections)}"
            )
        return self


class FieldPatternsOutput(ResearchArtifactOutput):
    patterns: list[str] = Field(default_factory=list)
    method_patterns: list[str] = Field(default_factory=list)
    dataset_patterns: list[str] = Field(default_factory=list)
    metric_patterns: list[str] = Field(default_factory=list)
    limitation_patterns: list[str] = Field(default_factory=list)


class ExperimentBacklogOutput(ResearchArtifactOutput):
    backlog_items: list[dict[str, Any] | str] = Field(default_factory=list)
    prioritization_rule: str | None = None


class AssumptionMapOutput(ResearchArtifactOutput):
    assumptions_to_challenge: list[str] = Field(default_factory=list)
    challenge_tests: list[dict[str, Any] | str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _require_task_sections(self) -> AssumptionMapOutput:
        missing_sections = _missing_required_output_sections(
            self,
            ("assumptions_to_challenge", "challenge_tests", "evidence_gaps"),
        )
        if missing_sections:
            raise ValueError(
                "Assumption map requires non-empty task section(s): "
                f"{', '.join(missing_sections)}"
            )
        return self


class HypothesesOutput(ResearchArtifactOutput):
    hypotheses: list[dict[str, Any] | str] = Field(default_factory=list)
    validation_plan: list[dict[str, Any] | str] = Field(default_factory=list)


class RevisionPlanOutput(ResearchArtifactOutput):
    revision_priorities: list[dict[str, Any] | str] = Field(default_factory=list)
    paper_backed_risks: list[str] = Field(default_factory=list)
    source_context_risks: list[str] = Field(default_factory=list)


_CONTRACTS: dict[str, type[ResearchArtifactOutput]] = {
    "literature_review": LiteratureReviewOutput,
    "quality_harness": CritiqueOutput,
    "comparison": ComparisonOutput,
    "benchmark_planning": BenchmarkPlanOutput,
    "experiment_planning": ExperimentPlanOutput,
    "field_pattern_analysis": FieldPatternsOutput,
    "experiment_backlog": ExperimentBacklogOutput,
    "assumption_mapping": AssumptionMapOutput,
    "hypothesis_generation": HypothesesOutput,
    "revision_planning": RevisionPlanOutput,
}

_CONTRACT_DETAILS: dict[str, dict[str, Any]] = {
    "literature_review": {
        "schema_name": "literature_review",
        "required_fields": ["title", "summary"],
        "artifact_fields": ["themes", "research_gaps", "future_directions"],
        "recommendation_fields": ["recommendations"],
    },
    "quality_harness": {
        "schema_name": "critique",
        "required_fields": ["title", "summary"],
        "artifact_fields": ["quality_checks"],
        "recommendation_fields": ["recommendations"],
    },
    "comparison": {
        "schema_name": "comparison",
        "required_fields": ["title", "summary"],
        "artifact_fields": ["comparison_rows", "method_families", "datasets", "metrics"],
        "recommendation_fields": ["recommendations"],
    },
    "benchmark_planning": {
        "schema_name": "benchmark_plan",
        "required_fields": [
            "title",
            "summary",
            "benchmark_recommendations",
            "datasets",
            "metrics_or_result_logic",
            "candidate_baselines",
        ],
        "artifact_fields": [
            "benchmark_recommendations",
            "datasets",
            "metrics_or_result_logic",
            "candidate_baselines",
        ],
        "recommendation_fields": ["recommendations", "benchmark_recommendations"],
    },
    "experiment_planning": {
        "schema_name": "experiment_plan",
        "required_fields": [
            "title",
            "summary",
            "objective",
            "baselines",
            "ablations",
            "controls",
        ],
        "artifact_fields": ["objective", "baselines", "ablations", "controls"],
        "recommendation_fields": ["recommendations"],
    },
    "field_pattern_analysis": {
        "schema_name": "field_patterns",
        "required_fields": ["title", "summary"],
        "artifact_fields": [
            "patterns",
            "method_patterns",
            "dataset_patterns",
            "metric_patterns",
            "limitation_patterns",
        ],
        "recommendation_fields": ["recommendations"],
    },
    "experiment_backlog": {
        "schema_name": "experiment_backlog",
        "required_fields": ["title", "summary"],
        "artifact_fields": ["backlog_items", "prioritization_rule"],
        "recommendation_fields": ["recommendations"],
    },
    "assumption_mapping": {
        "schema_name": "assumption_map",
        "required_fields": [
            "title",
            "summary",
            "assumptions_to_challenge",
            "challenge_tests",
            "evidence_gaps",
        ],
        "artifact_fields": ["assumptions_to_challenge", "challenge_tests", "evidence_gaps"],
        "recommendation_fields": ["recommendations"],
    },
    "hypothesis_generation": {
        "schema_name": "hypotheses",
        "required_fields": ["title", "summary"],
        "artifact_fields": ["hypotheses", "validation_plan"],
        "recommendation_fields": ["recommendations"],
    },
    "revision_planning": {
        "schema_name": "revision_plan",
        "required_fields": ["title", "summary"],
        "artifact_fields": ["revision_priorities", "paper_backed_risks", "source_context_risks"],
        "recommendation_fields": ["recommendations"],
    },
}


def output_contract_prompt(
    *,
    skill_id: str,
    artifact_type: str,
) -> dict[str, Any]:
    """Return a compact prompt-facing contract for one research skill output."""

    details = dict(_CONTRACT_DETAILS.get(skill_id) or {})
    schema_name = str(details.get("schema_name") or artifact_type)
    return {
        "schema_name": schema_name,
        "artifact_type": artifact_type,
        "required_fields": list(details.get("required_fields") or ["title", "summary"]),
        "common_list_fields": [
            "recommendations",
            "evidence_references",
            "assumptions_or_inferences",
            "next_actions",
            "limitations",
        ],
        "artifact_fields": list(details.get("artifact_fields") or []),
        "recommendation_fields": list(
            details.get("recommendation_fields") or ["recommendations"]
        ),
        "rules": [
            "Return a JSON object, not markdown.",
            "Use arrays for every list field, even when there is one item.",
            "Recommendation objects need at least a title; include detail, support_status, "
            "supporting_layers, and evidence_references when available.",
            "Evidence references need reference_type plus a target id such as paper_id, "
            "source_id, chunk_id, evidence_span_id, result_row_id, memory_record_id, "
            "or a label.",
            "Every field listed in required_fields needs non-empty content.",
            "Do not include runtime metadata, prompt echoes, context echoes, or private "
            "reasoning fields.",
            "Keep paper evidence, user-provided source facts, and assumptions_or_inferences "
            "separate.",
            "Use support_status=user_provided with supporting_layers=[source_fact_memory] "
            "for selected Study source facts; do not mark source_fact_memory as "
            "supported paper evidence.",
        ],
    }


def normalize_model_output_payload(
    *,
    skill_id: str,
    artifact_type: str,
    payload: Any,
) -> dict[str, Any]:
    """Validate and normalize a model-produced artifact payload."""

    if not isinstance(payload, dict):
        raise ResearchModelOutputContractError(
            f"{artifact_type} schema expected a JSON object."
        )
    contract = _CONTRACTS.get(skill_id, ResearchArtifactOutput)
    try:
        model = contract.model_validate(payload)
    except ValidationError as exc:
        raise ResearchModelOutputContractError(
            f"{artifact_type} schema validation failed: {_validation_summary(exc)}"
        ) from exc
    return _strip_disallowed_model_fields(model.model_dump(mode="json", exclude_none=True))


def strip_private_trace_fields(value: Any) -> Any:
    """Remove prompt echoes and private reasoning fields from persisted trace payloads."""

    return _strip_keys(value, _PRIVATE_MODEL_OUTPUT_KEYS)


def redact_private_trace_text(value: str | None) -> str | None:
    """Replace error text that appears to contain private prompt or reasoning data."""

    if value is None:
        return None
    normalized = value.casefold()
    if _PRIVATE_TRACE_KEY_PATTERN.search(value) or any(
        marker in normalized for marker in _PRIVATE_TRACE_TEXT_MARKERS
    ):
        return REDACTED_PRIVATE_TRACE_TEXT
    return value


def _has_reference_anchor(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(_has_reference_anchor(item) for item in value)
    return False


def _missing_required_output_sections(
    model: BaseModel,
    section_names: tuple[str, ...],
) -> list[str]:
    return [
        section_name
        for section_name in section_names
        if not _has_required_output_value(getattr(model, section_name, None))
    ]


def _has_required_output_value(value: Any) -> bool:
    if isinstance(value, BaseModel):
        return _has_required_output_value(value.model_dump(mode="json", exclude_none=True))
    return task_section_has_meaningful_content(value)


def _strip_disallowed_model_fields(value: Any) -> Any:
    return _strip_keys(value, _DISALLOWED_MODEL_OUTPUT_KEYS)


def _strip_keys(value: Any, disallowed_keys: frozenset[str]) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_keys(item, disallowed_keys)
            for key, item in value.items()
            if key not in disallowed_keys
        }
    if isinstance(value, list):
        return [_strip_keys(item, disallowed_keys) for item in value]
    return value


def _validation_summary(exc: ValidationError) -> str:
    issues: list[str] = []
    for error in exc.errors()[:5]:
        location = ".".join(str(part) for part in error.get("loc", ()))
        message = str(error.get("msg") or "invalid value")
        issues.append(f"{location or '<root>'}: {message}")
    return "; ".join(issues)
