"""Retrieval backends for academic paper search."""

from ra.retrieval.arxiv import ArxivClient, ArxivPaper
from ra.retrieval.semantic_scholar import (
    Author,
    Paper as SemanticScholarPaper,
    SemanticScholarClient,
    search_papers,
)
from ra.retrieval.unified import Paper, UnifiedRetriever

__all__ = [
    # Unified
    "Paper",
    "UnifiedRetriever",
    # Semantic Scholar
    "SemanticScholarClient",
    "SemanticScholarPaper",
    "Author",
    "search_papers",
    # arXiv
    "ArxivClient",
    "ArxivPaper",
]
