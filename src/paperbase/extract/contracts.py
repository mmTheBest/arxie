"""Pydantic contracts for structured Paperbase extraction payloads."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

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

    @model_validator(mode="before")
    @classmethod
    def _drop_invalid_list_entries(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data

        cleaned = dict(data)
        for field_name in (
            "datasets",
            "methods",
            "metrics",
            "results",
            "findings",
            "glossary_terms",
            "engineering_tricks",
        ):
            cleaned[field_name] = cls._filter_entity_list(cleaned.get(field_name))
        return cleaned

    @staticmethod
    def _filter_entity_list(raw: object) -> list[Any]:
        if not isinstance(raw, list):
            return []

        cleaned_items: list[Any] = []
        for item in raw:
            if isinstance(item, BaseModel | dict):
                cleaned_items.append(item)
        return cleaned_items


__all__ = ["GlossaryTermExtraction", "StructuredExtractionBundle"]
