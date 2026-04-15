"""Pydantic contracts for structured Paperbase extraction payloads."""

from __future__ import annotations

from pydantic import BaseModel, Field

from paperbase.schemas.extraction import (
    DatasetExtraction,
    EngineeringTrickExtraction,
    FindingExtraction,
    GlossaryTermExtraction,
    MethodExtraction,
    MetricExtraction,
    ResultExtraction,
)


class StructuredExtractionBundle(BaseModel):
    datasets: list[DatasetExtraction] = Field(default_factory=list)
    methods: list[MethodExtraction] = Field(default_factory=list)
    metrics: list[MetricExtraction] = Field(default_factory=list)
    results: list[ResultExtraction] = Field(default_factory=list)
    findings: list[FindingExtraction] = Field(default_factory=list)
    glossary_terms: list[GlossaryTermExtraction] = Field(default_factory=list)
    engineering_tricks: list[EngineeringTrickExtraction] = Field(default_factory=list)


__all__ = ["GlossaryTermExtraction", "StructuredExtractionBundle"]
