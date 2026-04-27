"""Prompt builders for structured Paperbase extraction."""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

_SYSTEM_PROMPT = """You are extracting structured research facts from a single academic paper.

Return only facts grounded in the provided paper text.
Attach evidence spans for every extracted item.
Prefer exact benchmark names, metric names, and terminology used in the paper.
If the paper does not support an item, leave that list empty.
"""


def build_extraction_messages(
    *,
    paper_text: str,
    schema_payload: dict[str, object],
) -> list[SystemMessage | HumanMessage]:
    """Build the structured-extraction prompt payload for one parsed paper."""

    schema_json = json.dumps(schema_payload, ensure_ascii=False, indent=2, sort_keys=True)
    human_prompt = (
        "Extraction profile:\n"
        f"{schema_json}\n\n"
        "Paper text:\n"
        f"{paper_text.strip()}"
    )

    return [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=human_prompt),
    ]
