"""Pydantic contracts for structured extraction outputs."""

from __future__ import annotations

from pydantic import BaseModel, Field


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


class MethodExtraction(BaseModel):
    display_name: str
    normalized_name: str | None = None
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)
    evidence_spans: list[EvidenceSpanPayload] = Field(default_factory=list)


class MetricExtraction(BaseModel):
    display_name: str
    normalized_name: str | None = None
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)
    evidence_spans: list[EvidenceSpanPayload] = Field(default_factory=list)


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


class EngineeringTrickExtraction(BaseModel):
    title: str
    description: str
    evidence_spans: list[EvidenceSpanPayload] = Field(default_factory=list)
