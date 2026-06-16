"""Freshness metadata for structured extraction runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from paperbase.db.models import ExtractionRun, Section
from paperbase.parsing.files import load_active_pdf_file

FRESHNESS_DIAGNOSTICS_KEY = "freshness"
FRESHNESS_VERSION = "extraction-freshness-v1"


@dataclass(frozen=True, slots=True)
class ExtractionFreshness:
    """Digest inputs that decide whether a completed extraction can be reused."""

    schema_payload_digest: str
    parsed_content_digest: str
    source_content_digest: str
    freshness_version: str = FRESHNESS_VERSION

    def to_diagnostics(self) -> dict[str, str]:
        return {
            "freshness_version": self.freshness_version,
            "schema_payload_digest": self.schema_payload_digest,
            "parsed_content_digest": self.parsed_content_digest,
            "source_content_digest": self.source_content_digest,
        }


def stable_json_digest(payload: Any) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_parsed_paper_text(session: Session, *, paper_id: str) -> str:
    sections = session.execute(
        select(Section)
        .where(Section.paper_id == paper_id)
        .order_by(Section.ordinal.asc())
    ).scalars().all()

    if not sections:
        raise ValueError(f"No parsed sections found for paper_id={paper_id}")

    return "\n\n".join(
        f"{section.title}\n{section.text}"
        for section in sections
        if section.text.strip()
    )


def build_extraction_freshness(
    session: Session,
    *,
    paper_id: str,
    schema_payload: dict[str, object],
    paper_text: str | None = None,
) -> ExtractionFreshness:
    resolved_paper_text = paper_text
    if resolved_paper_text is None:
        resolved_paper_text = load_parsed_paper_text(session, paper_id=paper_id)

    return ExtractionFreshness(
        schema_payload_digest=stable_json_digest(schema_payload),
        parsed_content_digest=text_digest(resolved_paper_text),
        source_content_digest=_source_content_digest(session, paper_id=paper_id),
    )


def extraction_run_matches_freshness(
    run: ExtractionRun,
    *,
    freshness: ExtractionFreshness,
) -> bool:
    diagnostics = run.diagnostics_json or {}
    return diagnostics.get(FRESHNESS_DIAGNOSTICS_KEY) == freshness.to_diagnostics()


def merge_freshness_diagnostics(
    diagnostics: dict[str, Any] | None,
    *,
    freshness: ExtractionFreshness,
) -> dict[str, Any]:
    merged = dict(diagnostics or {})
    merged[FRESHNESS_DIAGNOSTICS_KEY] = freshness.to_diagnostics()
    return merged


def _source_content_digest(session: Session, *, paper_id: str) -> str:
    file_record = load_active_pdf_file(session, paper_id=paper_id)
    if file_record is None:
        source_payload: dict[str, object | None] = {}
    else:
        source_payload = {
            "file_kind": file_record.file_kind,
            "storage_uri": file_record.storage_uri,
            "content_hash": file_record.content_hash,
        }
    return stable_json_digest(source_payload)
