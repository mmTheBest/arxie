"""Pydantic contracts for structured extraction outputs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


def _normalize_named_artifact_payload(
    raw: object,
    *,
    aliases: tuple[str, ...],
) -> object:
    if not isinstance(raw, dict):
        return raw

    data = dict(raw)
    if "display_name" not in data:
        for alias in aliases:
            alias_value = data.get(alias)
            if isinstance(alias_value, str) and alias_value.strip():
                data["display_name"] = alias_value.strip()
                break

    metadata = dict(data.get("metadata") or {})
    reserved_keys = {"display_name", "normalized_name", "metadata", "evidence_spans", *aliases}
    for key, value in list(data.items()):
        if key in reserved_keys:
            continue
        if isinstance(value, (str, int, float, bool)):
            metadata.setdefault(key, value)
    if metadata:
        data["metadata"] = metadata
    return data


class EvidenceSpanPayload(BaseModel):
    target_type: str
    quote_text: str
    page_number: int | None = None
    start_char: int | None = None
    end_char: int | None = None


class DatasetExtraction(BaseModel):
    display_name: str
    normalized_name: str | None = None
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)
    evidence_spans: list[EvidenceSpanPayload] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_aliases(cls, data: object) -> object:
        return _normalize_named_artifact_payload(data, aliases=("dataset_name", "name"))


class MethodExtraction(BaseModel):
    display_name: str
    normalized_name: str | None = None
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)
    evidence_spans: list[EvidenceSpanPayload] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_aliases(cls, data: object) -> object:
        return _normalize_named_artifact_payload(
            data,
            aliases=("canonical_name", "method_name", "name"),
        )


class MetricExtraction(BaseModel):
    display_name: str
    normalized_name: str | None = None
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)
    evidence_spans: list[EvidenceSpanPayload] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_aliases(cls, data: object) -> object:
        return _normalize_named_artifact_payload(data, aliases=("metric_name", "name"))


class ResultExtraction(BaseModel):
    dataset_name: str
    method_name: str
    metric_name: str
    value_numeric: float | None = None
    value_text: str | None = None
    comparator_text: str | None = None
    notes: str | None = None
    evidence_spans: list[EvidenceSpanPayload] = Field(default_factory=list)


class FindingExtraction(BaseModel):
    statement: str
    polarity: str | None = None
    evidence_spans: list[EvidenceSpanPayload] = Field(default_factory=list)


class LimitationExtraction(BaseModel):
    statement: str
    evidence_spans: list[EvidenceSpanPayload] = Field(default_factory=list)


class GlossaryTermExtraction(BaseModel):
    term: str
    definition: str
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)
    evidence_spans: list[EvidenceSpanPayload] = Field(default_factory=list)


class EngineeringTrickExtraction(BaseModel):
    title: str
    description: str
    evidence_spans: list[EvidenceSpanPayload] = Field(default_factory=list)


ResearchDesignElementType = Literal[
    "research_question",
    "hypothesis",
    "task_definition",
    "baseline_method",
    "control",
    "ablation",
    "evaluation_protocol",
    "experimental_variable",
    "validity_threat",
    "reproducibility_signal",
    "reasoning_pattern",
    "claimed_contribution",
]


class ResearchDesignElementExtraction(BaseModel):
    element_type: ResearchDesignElementType
    title: str
    description: str
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)
    evidence_spans: list[EvidenceSpanPayload] = Field(default_factory=list)
