"""Semantic Scholar API client for paper search and metadata retrieval."""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx
from ra.utils.rate_limiter import TokenBucketRateLimiter

logger = logging.getLogger(__name__)

# Semantic Scholar API base URL
BASE_URL = "https://api.semanticscholar.org/graph/v1"
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Default fields to retrieve for papers
DEFAULT_PAPER_FIELDS = [
    "paperId",
    "title",
    "abstract",
    "year",
    "authors",
    "venue",
    "publicationTypes",
    "citationCount",
    "influentialCitationCount",
    "isOpenAccess",
    "openAccessPdf",
    "externalIds",  # DOI, arXiv ID, etc.
]


@dataclass
class Author:
    """Paper author."""

    author_id: str
    name: str

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "Author":
        return cls(
            author_id=data.get("authorId", ""),
            name=data.get("name", "Unknown"),
        )


@dataclass
class Paper:
    """Academic paper metadata."""

    paper_id: str
    title: str
    abstract: str | None
    year: int | None
    authors: list[Author]
    venue: str | None
    citation_count: int
    is_open_access: bool
    pdf_url: str | None
    doi: str | None
    arxiv_id: str | None
    external_ids: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "Paper":
        """Parse paper from Semantic Scholar API response."""
        external_ids = data.get("externalIds") or {}
        open_access_pdf = data.get("openAccessPdf") or {}

        return cls(
            paper_id=data.get("paperId", ""),
            title=data.get("title", ""),
            abstract=data.get("abstract"),
            year=data.get("year"),
            authors=[Author.from_api(a) for a in data.get("authors", [])],
            venue=data.get("venue"),
            citation_count=data.get("citationCount", 0),
            is_open_access=data.get("isOpenAccess", False),
            pdf_url=open_access_pdf.get("url"),
            doi=external_ids.get("DOI"),
            arxiv_id=external_ids.get("ArXiv"),
            external_ids=external_ids,
        )

    def to_citation(self, style: str = "apa") -> str:
        """Format paper as citation string."""
        authors_str = ", ".join(a.name for a in self.authors[:3])
        if len(self.authors) > 3:
            authors_str += " et al."

        year_str = f"({self.year})" if self.year else "(n.d.)"
        venue_str = f" {self.venue}." if self.venue else ""

        return f"{authors_str} {year_str}. {self.title}.{venue_str}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "abstract": self.abstract,
            "year": self.year,
            "authors": [{"id": a.author_id, "name": a.name} for a in self.authors],
            "venue": self.venue,
            "citation_count": self.citation_count,
            "is_open_access": self.is_open_access,
            "pdf_url": self.pdf_url,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "citation": self.to_citation(),
        }


class SemanticScholarClient:
    """Async client for Semantic Scholar API."""

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        max_backoff_seconds: float = 8.0,
        rate_limiter: TokenBucketRateLimiter | None = None,
    ):
        """Initialize client.

        Args:
            api_key: Optional API key for higher rate limits.
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts for failed requests.
        """
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_backoff_seconds = max_backoff_seconds
        self._rate_limiter = rate_limiter or TokenBucketRateLimiter(
            rate_per_second=5.0,
            burst=5,
        )
        self._client: httpx.AsyncClient | None = None

    def _backoff_seconds(self, attempt: int) -> float:
        """Return exponential backoff delay for a zero-based attempt index."""
        return min(2**attempt, self.max_backoff_seconds)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            headers = {}
            if self.api_key:
                headers["x-api-key"] = self.api_key
            self._client = httpx.AsyncClient(
                base_url=BASE_URL,
                headers=headers,
                timeout=self.timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make API request with retry logic."""
        client = await self._get_client()

        for attempt in range(self.max_retries):
            try:
                await self._rate_limiter.acquire()
                response = await client.request(method, endpoint, params=params)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status in RETRYABLE_STATUS_CODES and attempt < self.max_retries - 1:
                    wait_time = self._backoff_seconds(attempt)
                    logger.warning(
                        "Semantic Scholar HTTP %s; retrying in %ss (attempt %s/%s)",
                        status,
                        wait_time,
                        attempt + 1,
                        self.max_retries,
                    )
                    await asyncio.sleep(wait_time)
                    continue
                raise
            except httpx.RequestError as e:
                if attempt < self.max_retries - 1:
                    wait_time = self._backoff_seconds(attempt)
                    logger.warning(
                        "Semantic Scholar request error; retrying in %ss (attempt %s/%s): %s",
                        wait_time,
                        attempt + 1,
                        self.max_retries,
                        e,
                    )
                    await asyncio.sleep(wait_time)
                    continue
                raise

        raise RuntimeError(f"Failed after {self.max_retries} attempts")

    async def search(
        self,
        query: str,
        limit: int = 10,
        year_range: tuple[int, int] | None = None,
        fields_of_study: list[str] | None = None,
    ) -> list[Paper]:
        """Search for papers by query.

        Args:
            query: Search query string.
            limit: Maximum number of results (max 100).
            year_range: Optional (start_year, end_year) filter.
            fields_of_study: Optional list of fields to filter by.

        Returns:
            List of matching papers.
        """
        params: dict[str, Any] = {
            "query": query,
            "limit": min(limit, 100),
            "fields": ",".join(DEFAULT_PAPER_FIELDS),
        }

        if year_range:
            params["year"] = f"{year_range[0]}-{year_range[1]}"

        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)

        data = await self._request("GET", "/paper/search", params)
        papers = [Paper.from_api(p) for p in data.get("data", [])]

        logger.info(
            "Semantic Scholar search completed",
            extra={
                "event": "semantic_scholar.search.completed",
                "query": query,
                "result_count": len(papers),
            },
        )
        return papers

    async def get_paper(self, paper_id: str) -> Paper | None:
        """Get paper by Semantic Scholar ID, DOI, or arXiv ID.

        Args:
            paper_id: Paper identifier (S2 ID, DOI:xxx, ARXIV:xxx, etc.)

        Returns:
            Paper metadata or None if not found.
        """
        params = {"fields": ",".join(DEFAULT_PAPER_FIELDS)}

        try:
            data = await self._request("GET", f"/paper/{paper_id}", params)
            return Paper.from_api(data)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    async def get_citations(
        self,
        paper_id: str,
        limit: int = 50,
    ) -> list[Paper]:
        """Get papers that cite this paper.

        Args:
            paper_id: Paper identifier.
            limit: Maximum number of citations to return.

        Returns:
            List of citing papers.
        """
        params = {
            "fields": ",".join(DEFAULT_PAPER_FIELDS),
            "limit": min(limit, 100),
        }

        data = await self._request("GET", f"/paper/{paper_id}/citations", params)
        papers = [Paper.from_api(c.get("citingPaper", {})) for c in data.get("data", [])]

        return papers

    async def get_references(
        self,
        paper_id: str,
        limit: int = 50,
    ) -> list[Paper]:
        """Get papers cited by this paper.

        Args:
            paper_id: Paper identifier.
            limit: Maximum number of references to return.

        Returns:
            List of referenced papers.
        """
        params = {
            "fields": ",".join(DEFAULT_PAPER_FIELDS),
            "limit": min(limit, 100),
        }

        data = await self._request("GET", f"/paper/{paper_id}/references", params)
        papers = [Paper.from_api(r.get("citedPaper", {})) for r in data.get("data", [])]

        return papers


# Convenience function for synchronous usage
def search_papers(query: str, limit: int = 10, **kwargs) -> list[Paper]:
    """Synchronous wrapper for paper search."""

    async def _search():
        client = SemanticScholarClient()
        try:
            return await client.search(query, limit, **kwargs)
        finally:
            await client.close()

    return asyncio.run(_search())
