"""Table extraction contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class TableCandidate:
    page_number: int | None = None
    table_label: str | None = None
    caption: str | None = None
    storage_uri: str | None = None
    bbox_json: dict[str, Any] = field(default_factory=dict)
    structured_payload_json: dict[str, Any] = field(default_factory=dict)
