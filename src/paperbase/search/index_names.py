"""Index naming helpers for Paperbase search backends."""

from __future__ import annotations

import re

_INDEX_SEGMENT_RE = re.compile(r"[^a-z0-9_-]+")
_SEARCH_INDEX_KINDS = frozenset({"papers", "chunks", "figures", "tables"})


def search_index_prefix(
    project_id: str | None = None,
    *,
    base_prefix: str = "paperbase",
) -> str:
    """Return the Elasticsearch/OpenSearch index prefix for a project boundary."""

    normalized_base = _normalize_index_segment(base_prefix)
    if project_id is None or not project_id.strip():
        return normalized_base

    normalized_project_id = _normalize_index_segment(project_id)
    if not normalized_project_id:
        raise ValueError("project_id must contain at least one index-safe character.")
    return f"{normalized_base}-{normalized_project_id}"


def search_index_name(
    kind: str,
    *,
    project_id: str | None = None,
    index_prefix: str | None = None,
) -> str:
    """Return the full search index name for a read-model kind."""

    normalized_kind = _normalize_index_segment(kind)
    if normalized_kind not in _SEARCH_INDEX_KINDS:
        allowed = ", ".join(sorted(_SEARCH_INDEX_KINDS))
        raise ValueError(f"Unsupported search index kind {kind!r}; expected one of: {allowed}.")
    prefix = index_prefix if index_prefix is not None else search_index_prefix(project_id)
    return f"{prefix}-{normalized_kind}"


def _normalize_index_segment(value: str) -> str:
    normalized = _INDEX_SEGMENT_RE.sub("-", value.strip().lower()).strip("-_")
    if not normalized:
        raise ValueError("Search index segments must contain at least one safe character.")
    return normalized
