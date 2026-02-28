"""Unified retrieval interface combining Semantic Scholar and arXiv.

This module provides:
- Paper: a normalized paper metadata dataclass
- UnifiedRetriever: a facade around SemanticScholarClient and ArxivClient

Design goals:
- Present a single normalized schema across sources
- Search across multiple sources and deduplicate results
- Provide async context manager support
"""

from __future__ import annotations

import asyncio
import logging
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import httpx

from ra.parsing import PDFParser
from ra.retrieval.arxiv import ArxivClient, ArxivPaper
from ra.retrieval.chroma_cache import ChromaCache
from ra.retrieval.semantic_scholar import SemanticScholarClient
from ra.utils.security import sanitize_identifier, sanitize_user_text, validate_public_http_url

logger = logging.getLogger(__name__)

Source = Literal["semantic_scholar", "arxiv", "both"]
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

DOI_RE = re.compile(r"^(?:doi:)?(10\.\d{4,9}/\S+)$", re.IGNORECASE)
# Examples: 1707.08567, 1707.08567v2, hep-th/9901001
ARXIV_ID_RE = re.compile(
    r"^(?:arxiv:)?(?P<id>(?:\d{4}\.\d{4,5})(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)$",
    re.IGNORECASE,
)


def normalize_arxiv_id(identifier: str) -> str | None:
    m = ARXIV_ID_RE.match(identifier.strip())
    if not m:
        return None
    return m.group("id")


def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    d = doi.strip()
    m = DOI_RE.match(d)
    if m:
        d = m.group(1)
    # DOIs are case-insensitive; normalize to lower
    return d.lower() if d else None


@dataclass
class Paper:
    """Normalized paper metadata across sources."""

    id: str
    title: str
    abstract: str | None
    authors: list[str]
    year: int | None
    venue: str | None
    citation_count: int | None
    pdf_url: str | None
    doi: str | None
    arxiv_id: str | None
    source: Source

    def to_citation(self, style: str = "apa") -> str:
        """Format the paper as a consistent citation string.

        Currently supports a simple APA-like format.
        """
        authors_str = ", ".join(self.authors[:3]) if self.authors else "Unknown"
        if self.authors and len(self.authors) > 3:
            authors_str += " et al."

        year_str = f"({self.year})" if self.year else "(n.d.)"
        venue_str = f" {self.venue}." if self.venue else ""

        # Prefer DOI for identification; otherwise arXiv id
        id_str = ""
        if self.doi:
            id_str = f" https://doi.org/{normalize_doi(self.doi) or self.doi}"
        elif self.arxiv_id:
            id_str = f" arXiv:{self.arxiv_id}"

        return f"{authors_str} {year_str}. {self.title}.{venue_str}{id_str}".strip()


def _paper_key(p: Paper) -> tuple[str, str] | tuple[str, str, str]:
    """Return a stable deduplication key.

    Priority:
    1) DOI
    2) arXiv ID
    3) title+year+first author (fallback)
    """
    doi = normalize_doi(p.doi)
    if doi:
        return ("doi", doi)
    ax = normalize_arxiv_id(p.arxiv_id or "")
    if ax:
        return ("arxiv", ax.lower())

    title = " ".join((p.title or "").lower().split())
    year = str(p.year or "")
    first_author = (p.authors[0].lower() if p.authors else "")
    return ("fallback", title, year + ":" + first_author)


def _merge_papers(primary: Paper, secondary: Paper) -> Paper:
    """Merge two papers believed to refer to the same work.

    We keep the primary paper as base and fill missing fields from secondary.
    Citation count uses the max available.
    Source becomes 'both' when merged.
    """
    merged = Paper(
        id=primary.id,
        title=primary.title or secondary.title,
        abstract=primary.abstract or secondary.abstract,
        authors=primary.authors or secondary.authors,
        year=primary.year or secondary.year,
        venue=primary.venue or secondary.venue,
        citation_count=max(
            [c for c in [primary.citation_count, secondary.citation_count] if c is not None],
            default=None,
        ),
        pdf_url=primary.pdf_url or secondary.pdf_url,
        doi=primary.doi or secondary.doi,
        arxiv_id=primary.arxiv_id or secondary.arxiv_id,
        source="both" if primary.source != secondary.source else primary.source,
    )
    return merged


class UnifiedRetriever:
    """Unified retriever that wraps Semantic Scholar and arXiv clients."""

    def __init__(
        self,
        semantic_scholar: SemanticScholarClient | None = None,
        arxiv: ArxivClient | None = None,
        cache: ChromaCache | None = None,
        full_text_max_retries: int = 3,
        full_text_max_backoff_seconds: float = 8.0,
        full_text_timeout: float = 60.0,
        full_text_max_connections: int = 40,
        full_text_max_keepalive_connections: int = 10,
        full_text_keepalive_expiry: float = 30.0,
    ):
        self.semantic_scholar = semantic_scholar or SemanticScholarClient()
        self.arxiv = arxiv or ArxivClient()
        self.cache = cache
        self.full_text_max_retries = full_text_max_retries
        self.full_text_max_backoff_seconds = full_text_max_backoff_seconds
        self.full_text_timeout = full_text_timeout
        self.full_text_max_connections = max(1, int(full_text_max_connections))
        self.full_text_max_keepalive_connections = max(
            1,
            min(int(full_text_max_keepalive_connections), self.full_text_max_connections),
        )
        self.full_text_keepalive_expiry = max(1.0, float(full_text_keepalive_expiry))
        self._full_text_limits = httpx.Limits(
            max_connections=self.full_text_max_connections,
            max_keepalive_connections=self.full_text_max_keepalive_connections,
            keepalive_expiry=self.full_text_keepalive_expiry,
        )
        self._full_text_client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "UnifiedRetriever":
        # Clients are lazy; nothing required here.
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        await asyncio.gather(
            self.semantic_scholar.close(),
            self.arxiv.close(),
            self._close_full_text_client(),
            return_exceptions=True,
        )

    async def _get_full_text_client(self) -> httpx.AsyncClient:
        is_closed = bool(getattr(self._full_text_client, "is_closed", False))
        if self._full_text_client is None or is_closed:
            self._full_text_client = httpx.AsyncClient(
                timeout=self.full_text_timeout,
                follow_redirects=True,
                limits=self._full_text_limits,
            )
        return self._full_text_client

    async def _close_full_text_client(self) -> None:
        is_closed = bool(getattr(self._full_text_client, "is_closed", False))
        if self._full_text_client and not is_closed:
            close = getattr(self._full_text_client, "aclose", None)
            if close is not None:
                await close()
        self._full_text_client = None

    def _from_semantic_scholar(self, p) -> Paper:
        authors = [a.name for a in getattr(p, "authors", [])]
        return Paper(
            id=getattr(p, "paper_id", ""),
            title=getattr(p, "title", ""),
            abstract=getattr(p, "abstract", None),
            authors=authors,
            year=getattr(p, "year", None),
            venue=getattr(p, "venue", None),
            citation_count=getattr(p, "citation_count", None),
            pdf_url=getattr(p, "pdf_url", None),
            doi=getattr(p, "doi", None),
            arxiv_id=getattr(p, "arxiv_id", None),
            source="semantic_scholar",
        )

    def _from_arxiv(self, p: ArxivPaper) -> Paper:
        year: int | None = None
        if getattr(p, "published", None) and str(p.published)[:4].isdigit():
            year = int(str(p.published)[:4])
        return Paper(
            id=p.arxiv_id,
            title=p.title,
            abstract=p.abstract or None,
            authors=p.authors,
            year=year,
            venue=None,
            citation_count=None,
            pdf_url=p.pdf_url,
            doi=p.doi,
            arxiv_id=p.arxiv_id,
            source="arxiv",
        )

    def _from_cached(self, item: dict[str, object]) -> Paper | None:
        paper_id = str(item.get("id") or "").strip()
        if not paper_id:
            return None

        source_value = str(item.get("source") or "semantic_scholar").lower()
        if source_value == "arxiv":
            source: Source = "arxiv"
        elif source_value == "both":
            source = "both"
        else:
            source = "semantic_scholar"

        year_raw = item.get("year")
        citation_raw = item.get("citation_count")
        try:
            year = int(year_raw) if year_raw is not None else None
        except (TypeError, ValueError):
            year = None
        try:
            citation_count = int(citation_raw) if citation_raw is not None else None
        except (TypeError, ValueError):
            citation_count = None

        authors_raw = item.get("authors")
        if isinstance(authors_raw, list):
            authors = [str(a) for a in authors_raw]
        else:
            authors = []

        return Paper(
            id=paper_id,
            title=str(item.get("title") or ""),
            abstract=(str(item.get("abstract")).strip() if item.get("abstract") else None),
            authors=authors,
            year=year,
            venue=(str(item.get("venue")).strip() if item.get("venue") else None),
            citation_count=citation_count,
            pdf_url=(str(item.get("pdf_url")).strip() if item.get("pdf_url") else None),
            doi=(str(item.get("doi")).strip() if item.get("doi") else None),
            arxiv_id=(str(item.get("arxiv_id")).strip() if item.get("arxiv_id") else None),
            source=source,
        )

    async def search(
        self,
        query: str,
        limit: int = 10,
        sources: Iterable[str] = ("semantic_scholar", "arxiv"),
    ) -> list[Paper]:
        """Search across sources and deduplicate.

        Args:
            query: Search query.
            limit: Approximate total number of results to return.
            sources: Iterable of sources to use: semantic_scholar, arxiv.

        Returns:
            Deduplicated list of Paper.
        """
        query = sanitize_user_text(query, field_name="query", max_length=1000)
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            raise ValueError("limit must be an integer.") from None
        limit = max(1, min(limit, 100))

        cached_results: list[Paper] = []
        if self.cache is not None:
            try:
                for row in self.cache.search_cached(query, limit=limit):
                    if isinstance(row, dict):
                        cached = self._from_cached(row)
                        if cached is not None:
                            cached_results.append(cached)
            except Exception:
                logger.debug("Cache lookup failed for query=%r", query, exc_info=True)

        if len(cached_results) >= limit:
            cached_results.sort(key=lambda x: (x.citation_count or 0), reverse=True)
            return cached_results[:limit]

        srcs = {str(s).strip().lower() for s in sources if str(s).strip()}
        if not srcs:
            srcs = {"semantic_scholar", "arxiv"}
        tasks = []

        # Pull a bit extra per-source so we still have enough after dedup.
        remaining_limit = max(1, limit - len(cached_results))
        per_source = max(1, min(100, int(remaining_limit * 1.5)))

        if "semantic_scholar" in srcs or "semanticscholar" in srcs or "s2" in srcs:
            tasks.append(self.semantic_scholar.search(query=query, limit=per_source))
        else:
            tasks.append(asyncio.sleep(0, result=[]))

        if "arxiv" in srcs:
            tasks.append(self.arxiv.search(query=query, limit=per_source))
        else:
            tasks.append(asyncio.sleep(0, result=[]))

        s2_results, ax_results = await asyncio.gather(*tasks)

        live_results: list[Paper] = []
        live_results.extend(self._from_semantic_scholar(p) for p in s2_results)
        live_results.extend(self._from_arxiv(p) for p in ax_results)

        if self.cache is not None:
            for paper in live_results:
                try:
                    self.cache.cache_paper(paper)
                except Exception:
                    logger.debug("Failed to cache paper_id=%s", paper.id, exc_info=True)

        unified: list[Paper] = []
        unified.extend(cached_results)
        unified.extend(live_results)

        # Deduplicate, preferring Semantic Scholar as primary when conflicts exist.
        by_key: dict[object, Paper] = {}
        for p in unified:
            key = _paper_key(p)
            if key not in by_key:
                by_key[key] = p
                continue

            existing = by_key[key]
            # Prefer Semantic Scholar metadata as primary
            if existing.source == "semantic_scholar" and p.source == "arxiv":
                by_key[key] = _merge_papers(existing, p)
            elif existing.source == "arxiv" and p.source == "semantic_scholar":
                by_key[key] = _merge_papers(p, existing)
            else:
                by_key[key] = _merge_papers(existing, p)

        results = list(by_key.values())

        # Basic ranking: citation_count desc if available, otherwise keep insertion-ish.
        results.sort(key=lambda x: (x.citation_count or 0), reverse=True)
        return results[:limit]

    async def search_batch(
        self,
        requests: list[tuple[str, int, Iterable[str]]],
        *,
        max_concurrency: int = 4,
    ) -> list[list[Paper]]:
        """Run multiple unified searches concurrently with bounded parallelism."""
        if not requests:
            return []
        try:
            max_concurrency = int(max_concurrency)
        except (TypeError, ValueError):
            raise ValueError("max_concurrency must be an integer.") from None
        max_concurrency = max(1, min(max_concurrency, 32))

        semaphore = asyncio.Semaphore(max_concurrency)
        results: list[list[Paper] | None] = [None] * len(requests)

        async def _run(index: int, job: tuple[str, int, Iterable[str]]) -> None:
            query, limit, sources = job
            async with semaphore:
                results[index] = await self.search(query=query, limit=limit, sources=sources)

        await asyncio.gather(*(_run(i, job) for i, job in enumerate(requests)))
        return [batch if batch is not None else [] for batch in results]

    async def get_papers_batch(
        self,
        identifiers: list[str],
        *,
        max_concurrency: int = 8,
    ) -> list[Paper | None]:
        """Fetch multiple papers concurrently by identifier with bounded parallelism."""
        if not identifiers:
            return []
        try:
            max_concurrency = int(max_concurrency)
        except (TypeError, ValueError):
            raise ValueError("max_concurrency must be an integer.") from None
        max_concurrency = max(1, min(max_concurrency, 32))

        semaphore = asyncio.Semaphore(max_concurrency)
        results: list[Paper | None] = [None] * len(identifiers)

        async def _run(index: int, identifier: str) -> None:
            async with semaphore:
                results[index] = await self.get_paper(identifier=identifier)

        await asyncio.gather(*(_run(i, ident) for i, ident in enumerate(identifiers)))
        return results

    async def get_paper(self, identifier: str) -> Paper | None:
        """Fetch a paper by identifier.

        Auto-detects DOI/arXiv id; otherwise assumes Semantic Scholar ID.
        """
        identifier = sanitize_identifier(identifier, field_name="identifier", max_length=256)

        doi_m = DOI_RE.match(identifier)
        if doi_m:
            doi = doi_m.group(1)
            s2 = await self.semantic_scholar.get_paper(f"DOI:{doi}")
            if s2:
                return self._from_semantic_scholar(s2)
            # fallback: arXiv sometimes stores DOI but not searchable; no good fallback.
            return None

        arxiv_id = normalize_arxiv_id(identifier)
        if arxiv_id:
            # Try Semantic Scholar first (usually richer), then arXiv.
            s2 = await self.semantic_scholar.get_paper(f"ARXIV:{arxiv_id}")
            if s2:
                return self._from_semantic_scholar(s2)
            ax = await self.arxiv.get_paper(arxiv_id)
            return self._from_arxiv(ax) if ax else None

        s2 = await self.semantic_scholar.get_paper(identifier)
        return self._from_semantic_scholar(s2) if s2 else None

    async def get_full_text(self, paper: Paper) -> str:
        """Download a paper PDF (if available) and return extracted plain text.

        Returns an empty string on any download or parsing failure.
        """
        if not paper.pdf_url:
            return ""
        try:
            pdf_url = validate_public_http_url(paper.pdf_url, field_name="paper.pdf_url")
        except ValueError:
            logger.warning(
                "Rejected unsafe full-text URL for paper_id=%s",
                getattr(paper, "id", ""),
            )
            return ""

        def _backoff_seconds(attempt: int) -> float:
            return min(2**attempt, self.full_text_max_backoff_seconds)

        for attempt in range(self.full_text_max_retries):
            tmp_path: Path | None = None
            try:
                client = await self._get_full_text_client()
                r = await client.get(pdf_url)
                r.raise_for_status()
                pdf_bytes = r.content

                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                    f.write(pdf_bytes)
                    tmp_path = Path(f.name)

                doc = PDFParser().parse(tmp_path)
                return (doc.text or "").strip()
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status in RETRYABLE_STATUS_CODES and attempt < self.full_text_max_retries - 1:
                    backoff = _backoff_seconds(attempt)
                    logger.warning(
                        "Full-text download HTTP %s; retrying in %ss (attempt %s/%s) for paper_id=%s",
                        status,
                        backoff,
                        attempt + 1,
                        self.full_text_max_retries,
                        getattr(paper, "id", ""),
                    )
                    await asyncio.sleep(backoff)
                    continue
                logger.warning(
                    "Failed to download full text for paper_id=%s pdf_url=%s",
                    getattr(paper, "id", ""),
                    paper.pdf_url,
                    exc_info=True,
                )
                return ""
            except httpx.RequestError:
                if attempt < self.full_text_max_retries - 1:
                    backoff = _backoff_seconds(attempt)
                    logger.warning(
                        "Full-text download request error; retrying in %ss (attempt %s/%s) for paper_id=%s",
                        backoff,
                        attempt + 1,
                        self.full_text_max_retries,
                        getattr(paper, "id", ""),
                        exc_info=True,
                    )
                    await asyncio.sleep(backoff)
                    continue
                logger.warning(
                    "Failed to download full text for paper_id=%s pdf_url=%s",
                    getattr(paper, "id", ""),
                    paper.pdf_url,
                    exc_info=True,
                )
                return ""
            except Exception:
                logger.warning(
                    "Failed to parse PDF full text for paper_id=%s pdf_url=%s",
                    getattr(paper, "id", ""),
                    paper.pdf_url,
                    exc_info=True,
                )
                return ""
            finally:
                if tmp_path is not None:
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except Exception:
                        logger.debug("Failed to delete temp PDF file: %s", tmp_path, exc_info=True)

        return ""
