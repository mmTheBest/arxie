from __future__ import annotations

import pytest

from paperbase.config import load_paperbase_config
from paperbase.search.embeddings import (
    DeterministicEmbeddingProvider,
    build_embedding_provider,
)


def test_build_embedding_provider_uses_deterministic_fallback_when_auto_and_no_key() -> None:
    config = load_paperbase_config({"PAPERBASE_EMBEDDING_PROVIDER": "auto"})

    provider = build_embedding_provider(config, env={})

    assert isinstance(provider, DeterministicEmbeddingProvider)


def test_build_embedding_provider_rejects_explicit_openai_without_key() -> None:
    config = load_paperbase_config({"PAPERBASE_EMBEDDING_PROVIDER": "openai"})

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_embedding_provider(config, env={})
