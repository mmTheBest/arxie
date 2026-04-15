"""arXiv-to-canonical ingest adapter."""

from __future__ import annotations

from paperbase.ingest.models import CanonicalPaperSeed
from ra.retrieval.arxiv import ArxivPaper


def seed_from_arxiv_paper(paper: ArxivPaper) -> CanonicalPaperSeed:
    payload = paper.to_seed_payload()
    return CanonicalPaperSeed(
        provider=str(payload["provider"]),
        external_id=str(payload["external_id"]),
        canonical_title=str(payload["canonical_title"]),
        abstract=payload.get("abstract"),
        publication_year=payload.get("publication_year"),
        venue=payload.get("venue"),
        doi=payload.get("doi"),
        arxiv_id=payload.get("arxiv_id"),
        authors=list(payload.get("authors") or []),
        source_payload=dict(payload.get("source_payload") or {}),
        raw_metadata=dict(payload.get("raw_metadata") or {}),
    )
