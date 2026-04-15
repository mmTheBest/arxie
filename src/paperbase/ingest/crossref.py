"""Crossref-to-canonical ingest adapter."""

from __future__ import annotations

import html
import re
from typing import Any

from paperbase.ingest.models import CanonicalPaperSeed


def _strip_xml_tags(value: str | None) -> str | None:
    if not value:
        return None
    unescaped = html.unescape(value)
    cleaned = re.sub(r"<[^>]+>", "", unescaped)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or None


def _author_name(author: dict[str, Any]) -> str | None:
    if author.get("name"):
        return str(author["name"]).strip() or None
    given = str(author.get("given") or "").strip()
    family = str(author.get("family") or "").strip()
    full_name = " ".join(part for part in (given, family) if part)
    return full_name or None


def _extract_year(payload: dict[str, Any]) -> int | None:
    for key in ("published-print", "published-online", "created"):
        container = payload.get(key) or {}
        date_parts = container.get("date-parts") or []
        if date_parts and date_parts[0]:
            try:
                return int(date_parts[0][0])
            except (TypeError, ValueError, IndexError):
                continue
    return None


def seed_from_crossref_work(payload: dict[str, Any]) -> CanonicalPaperSeed:
    title = ((payload.get("title") or [""])[0] or "").strip()
    venue = ((payload.get("container-title") or [""])[0] or "").strip() or None
    doi = str(payload.get("DOI") or "").strip() or None
    authors = [
        name
        for name in (_author_name(author) for author in (payload.get("author") or []))
        if name
    ]

    return CanonicalPaperSeed(
        provider="crossref",
        external_id=doi or title,
        canonical_title=title,
        abstract=_strip_xml_tags(payload.get("abstract")),
        publication_year=_extract_year(payload),
        venue=venue,
        doi=doi,
        authors=authors,
        source_payload=payload,
        raw_metadata={
            "publisher": payload.get("publisher"),
            "type": payload.get("type"),
        },
    )
