"""Extraction metadata and compatibility checks."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from paperbase.db.models import ExtractionRun, Section

EXTRACTION_METADATA_DIAGNOSTICS_KEY = "extraction_metadata"


@dataclass(frozen=True, slots=True)
class ExtractionCompatibilityCheck:
    is_current: bool
    compatibility: str
    stale_reasons: list[str]


def extraction_source_content_hash(paper_text: str) -> str:
    return sha256(paper_text.encode("utf-8")).hexdigest()


def current_paper_source_content_hash(session: Session, *, paper_id: str) -> str | None:
    sections = (
        session.execute(
            select(Section)
            .where(Section.paper_id == paper_id)
            .order_by(Section.ordinal.asc())
        )
        .scalars()
        .all()
    )
    if not sections:
        return None
    paper_text = "\n\n".join(
        f"{section.title}\n{section.text}"
        for section in sections
        if section.text.strip()
    )
    return extraction_source_content_hash(paper_text)


def extraction_metadata_payload(
    *,
    extraction_profile_id: str | None,
    prompt_version: str,
    schema_version: str,
    source_content_hash: str,
) -> dict[str, Any]:
    return {
        "extraction_profile_id": extraction_profile_id,
        "prompt_version": prompt_version,
        "schema_version": schema_version,
        "source_content_hash": source_content_hash,
    }


def evaluate_extraction_run_compatibility(
    run: ExtractionRun,
    *,
    extraction_profile_id: str | None,
    prompt_version: str,
    schema_version: str,
    source_content_hash: str | None,
) -> ExtractionCompatibilityCheck:
    stale_reasons: list[str] = []

    if run.status != "completed":
        stale_reasons.append("status")
    if run.extraction_profile_id != extraction_profile_id:
        stale_reasons.append("extraction_profile_id")
    if run.prompt_version != prompt_version:
        stale_reasons.append("prompt_version")
    if run.schema_version != schema_version:
        stale_reasons.append("schema_version")

    metadata = _run_extraction_metadata(run)
    stored_hash = metadata.get("source_content_hash") if metadata is not None else None
    if (
        isinstance(stored_hash, str)
        and source_content_hash is not None
        and stored_hash != source_content_hash
    ):
        stale_reasons.append("source_content_hash")

    if stale_reasons:
        return ExtractionCompatibilityCheck(
            is_current=False,
            compatibility="stale",
            stale_reasons=stale_reasons,
        )

    if metadata is None or not isinstance(stored_hash, str):
        return ExtractionCompatibilityCheck(
            is_current=True,
            compatibility="legacy_compatible",
            stale_reasons=[],
        )

    return ExtractionCompatibilityCheck(
        is_current=True,
        compatibility="current",
        stale_reasons=[],
    )


def _run_extraction_metadata(run: ExtractionRun) -> dict[str, Any] | None:
    diagnostics = run.diagnostics_json if isinstance(run.diagnostics_json, dict) else {}
    metadata = diagnostics.get(EXTRACTION_METADATA_DIAGNOSTICS_KEY)
    return metadata if isinstance(metadata, dict) else None
