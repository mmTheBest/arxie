"""Canonical ingest seed models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class CanonicalPaperSeed:
    provider: str
    external_id: str
    canonical_title: str
    abstract: str | None = None
    publication_year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    pdf_url: str | None = None
    authors: list[str] = field(default_factory=list)
    source_payload: dict[str, Any] = field(default_factory=dict)
    raw_metadata: dict[str, Any] = field(default_factory=dict)
