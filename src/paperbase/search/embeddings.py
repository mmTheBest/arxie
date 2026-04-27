"""Embedding providers for Paperbase search/read models."""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Mapping, Protocol

from paperbase.config import PaperbaseConfig


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> list[float]: ...


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


class DeterministicEmbeddingProvider:
    def embed(self, text: str) -> list[float]:
        return embed_text_deterministic(text)


class OpenAIEmbeddingProvider:
    def __init__(self, *, model: str) -> None:
        from langchain_openai import OpenAIEmbeddings

        self.client = OpenAIEmbeddings(model=model)

    def embed(self, text: str) -> list[float]:
        return list(self.client.embed_query(text))


def build_embedding_provider(
    config: PaperbaseConfig,
    *,
    env: Mapping[str, str] | None = None,
) -> EmbeddingProvider | None:
    resolved_env = os.environ if env is None else env
    mode = config.embedding_provider.strip().lower()
    has_openai_key = bool((resolved_env.get("OPENAI_API_KEY") or "").strip())

    if mode == "none":
        return None
    if mode == "deterministic":
        return DeterministicEmbeddingProvider()
    if mode == "openai":
        if not has_openai_key:
            raise ValueError("PAPERBASE_EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY.")
        return OpenAIEmbeddingProvider(model=config.embedding_model)
    if mode == "auto":
        if has_openai_key:
            return OpenAIEmbeddingProvider(model=config.embedding_model)
        return DeterministicEmbeddingProvider()
    raise ValueError(f"Unsupported embedding provider: {config.embedding_provider}")
