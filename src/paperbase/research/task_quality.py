"""Shared task-output quality helpers for Study artifact validation."""

from __future__ import annotations

import re
from typing import Any

PLACEHOLDER_TEXTS = frozenset(
    {
        "add detail",
        "add details",
        "fill in",
        "fill in later",
        "fill this in",
        "later",
        "more work needed",
        "n a",
        "na",
        "needs revision",
        "needs work",
        "none",
        "not applicable",
        "not specified",
        "placeholder",
        "tbd",
        "to be determined",
        "todo",
        "unknown",
        "unspecified",
    }
)
PLACEHOLDER_PREFIXES = (
    "add details ",
    "fill in later ",
    "placeholder ",
    "tbd ",
    "todo ",
)
METADATA_ONLY_KEYS = frozenset(
    {
        "artifact_type",
        "chunk_id",
        "dataset_id",
        "evidence_references",
        "evidence_span_id",
        "graph_node_id",
        "id",
        "ids",
        "label",
        "labels",
        "memory_record_id",
        "method_id",
        "metric_id",
        "paper_id",
        "reference_type",
        "references",
        "result_row_id",
        "schema_validation",
        "skill_id",
        "source",
        "source_id",
        "status",
        "support_status",
        "supporting_layer",
        "supporting_layers",
    }
)


def task_section_has_meaningful_content(value: Any) -> bool:
    """Return true when a task section contains real task content."""

    if isinstance(value, str):
        return _is_meaningful_text(value)
    dumped_model = _model_dump_value(value)
    if dumped_model is not None:
        return task_section_has_meaningful_content(dumped_model)
    if isinstance(value, dict):
        return any(
            task_section_has_meaningful_content(item)
            for key, item in value.items()
            if _task_quality_key(key) not in METADATA_ONLY_KEYS
        )
    if isinstance(value, list):
        return any(task_section_has_meaningful_content(item) for item in value)
    if value is None:
        return False
    if isinstance(value, bool):
        return False
    return True


def task_section_item_count(value: Any) -> int:
    """Count meaningful top-level items in a task section."""

    if isinstance(value, list):
        return sum(1 for item in value if task_section_has_meaningful_content(item))
    return int(task_section_has_meaningful_content(value))


def _is_meaningful_text(value: str) -> bool:
    normalized = _task_quality_text(value)
    if not normalized:
        return False
    if normalized in PLACEHOLDER_TEXTS:
        return False
    return not any(normalized.startswith(prefix) for prefix in PLACEHOLDER_PREFIXES)


def _task_quality_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _model_dump_value(value: Any) -> Any | None:
    model_dump = getattr(value, "model_dump", None)
    if not callable(model_dump):
        return None
    try:
        return model_dump(mode="json", exclude_none=True)
    except TypeError:
        return model_dump(exclude_none=True)


def _task_quality_key(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
