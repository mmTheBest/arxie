"""Deterministic local embeddings for Paperbase read models and queries."""

from __future__ import annotations

import hashlib
import math
import re


def embed_text_deterministic(text: str, *, dimensions: int = 1536) -> list[float]:
    """Compute a stable local embedding without external model dependencies."""
    tokens = re.findall(r"[a-z0-9]+", str(text).lower())
    if not tokens:
        return [0.0] * dimensions

    vec = [0.0] * dimensions
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        slot = int.from_bytes(digest[:2], "big") % dimensions
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        vec[slot] += sign

    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]
