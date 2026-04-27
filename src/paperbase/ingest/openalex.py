"""OpenAlex-to-canonical ingest adapter."""

from __future__ import annotations

from typing import Any

from paperbase.ingest.models import CanonicalPaperSeed


def _flatten_openalex_abstract(inverted_index: dict[str, list[int]] | None) -> str | None:
    if not inverted_index:
        return None

    positions: dict[int, str] = {}
    for token, indexes in inverted_index.items():
        for index in indexes:
            positions[int(index)] = token

    if not positions:
        return None

    max_index = max(positions)
    words = [positions.get(i, "") for i in range(max_index + 1)]
    return " ".join(word for word in words if word).strip() or None


def _normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if normalized.lower().startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    return normalized or None


def seed_from_openalex_work(payload: dict[str, Any]) -> CanonicalPaperSeed:
    ids = payload.get("ids") or {}
    primary_location = payload.get("primary_location") or {}
    source = primary_location.get("source") or {}
    authorships = payload.get("authorships") or []

    return CanonicalPaperSeed(
        provider="openalex",
        external_id=str(payload.get("id") or ""),
        canonical_title=str(payload.get("title") or ""),
        abstract=_flatten_openalex_abstract(payload.get("abstract_inverted_index")),
        publication_year=payload.get("publication_year"),
        venue=source.get("display_name"),
        doi=_normalize_doi(ids.get("doi")),
        pdf_url=primary_location.get("pdf_url"),
        authors=[
            authorship.get("author", {}).get("display_name", "")
            for authorship in authorships
            if authorship.get("author", {}).get("display_name")
        ],
        source_payload=payload,
        raw_metadata={
            "type": payload.get("type"),
            "cited_by_count": payload.get("cited_by_count"),
            "open_access": payload.get("open_access"),
        },
    )
