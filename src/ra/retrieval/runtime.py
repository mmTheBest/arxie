"""Runtime retriever wiring for Arxie."""

from __future__ import annotations

from ra.retrieval.paperbase_gateway import PaperbaseGateway
from ra.retrieval.unified import UnifiedRetriever


def build_runtime_retriever() -> UnifiedRetriever:
    """Create the default runtime retriever with Paperbase-first lookup enabled."""

    return UnifiedRetriever(paperbase_gateway=PaperbaseGateway())
