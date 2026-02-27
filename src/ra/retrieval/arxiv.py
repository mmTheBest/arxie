"""arXiv API client for paper search, metadata retrieval, and PDF download.

Uses the arXiv Atom feed API:
    https://export.arxiv.org/api/query

Notes:
- arXiv asks clients to avoid aggressive polling; we enforce a minimum 3s delay
  between requests.
- Responses are Atom XML; we parse with xml.etree.ElementTree.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://export.arxiv.org"
QUERY_ENDPOINT = "/api/query"

# Atom + arXiv namespaces
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


@dataclass
class ArxivPaper:
    """Paper metadata from arXiv."""

    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    published: str | None
    updated: str | None
    categories: list[str]
    pdf_url: str | None
    doi: str | None

    def to_citation(self) -> str:
        """Return a simple human-readable citation string."""
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += " et al."
        year = None
        if self.published and len(self.published) >= 4:
            year = self.published[:4]
        year_str = f"({year})" if year else "(n.d.)"
        return f"{authors_str} {year_str}. {self.title}. arXiv:{self.arxiv_id}."

    def to_dict(self) -> dict[str, Any]:
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "published": self.published,
            "updated": self.updated,
            "categories": self.categories,
            "pdf_url": self.pdf_url,
            "doi": self.doi,
            "citation": self.to_citation(),
        }


class ArxivClient:
    """Async client for the arXiv Atom API."""

    def __init__(
        self,
        timeout: float = 30.0,
        max_retries: int = 3,
        min_request_interval_s: float = 3.0,
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.min_request_interval_s = min_request_interval_s

        self._client: httpx.AsyncClient | None = None
        self._rate_lock = asyncio.Lock()
        self._last_request_ts = 0.0

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            # arXiv suggests identifying User-Agent in clients.
            headers = {
                "User-Agent": "academic-research-assistant/1.0 (arXiv API client)",
            }
            self._client = httpx.AsyncClient(
                base_url=BASE_URL,
                headers=headers,
                timeout=self.timeout,
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    async def _respect_rate_limit(self) -> None:
        """Enforce a minimum delay between requests (global per client instance)."""
        async with self._rate_lock:
            now = time.monotonic()
            elapsed = now - self._last_request_ts
            wait_s = self.min_request_interval_s - elapsed
            if wait_s > 0:
                await asyncio.sleep(wait_s)
            self._last_request_ts = time.monotonic()

    async def _request(self, params: dict[str, Any]) -> str:
        """Call arXiv query endpoint and return raw XML text."""
        client = await self._get_client()

        for attempt in range(self.max_retries):
            await self._respect_rate_limit()
            try:
                resp = await client.get(QUERY_ENDPOINT, params=params)
                resp.raise_for_status()
                return resp.text
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                # arXiv may return 503 during maintenance; retry with backoff.
                if status in {429, 500, 502, 503, 504} and attempt < self.max_retries - 1:
                    backoff = 2 ** attempt
                    logger.warning(
                        "arXiv request failed (%s), retrying in %ss (attempt %s/%s)",
                        status,
                        backoff,
                        attempt + 1,
                        self.max_retries,
                    )
                    await asyncio.sleep(backoff)
                    continue
                logger.exception("arXiv HTTP error (%s): %s", status, e)
                raise
            except httpx.RequestError as e:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        "arXiv request error, retrying in 1s (attempt %s/%s): %s",
                        attempt + 1,
                        self.max_retries,
                        e,
                    )
                    await asyncio.sleep(1)
                    continue
                logger.exception("arXiv request failed: %s", e)
                raise

        raise RuntimeError(f"Failed after {self.max_retries} attempts")

    def _parse_feed(self, xml_text: str) -> list[ArxivPaper]:
        """Parse an arXiv Atom feed into ArxivPaper objects."""
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            logger.exception("Failed to parse arXiv Atom XML")
            raise

        papers: list[ArxivPaper] = []
        seen: set[str] = set()
        for entry in root.findall("atom:entry", NS):
            paper = self._parse_entry(entry)
            if paper is None:
                continue
            # arXiv feeds can occasionally contain duplicate entries; dedupe by arXiv id.
            if paper.arxiv_id in seen:
                continue
            seen.add(paper.arxiv_id)
            papers.append(paper)
        return papers

    def _parse_entry(self, entry: ET.Element) -> ArxivPaper | None:
        """Parse one <entry> element."""
        raw_id = (entry.findtext("atom:id", default="", namespaces=NS) or "").strip()
        if not raw_id:
            return None

        # Typical id: http://arxiv.org/abs/2101.00001v1
        arxiv_id = raw_id.rsplit("/", 1)[-1]

        title = (entry.findtext("atom:title", default="", namespaces=NS) or "").strip()
        # Titles/abstract often contain newlines and repeated whitespace
        title = " ".join(title.split())

        abstract = (entry.findtext("atom:summary", default="", namespaces=NS) or "").strip()
        abstract = " ".join(abstract.split())

        published = (entry.findtext("atom:published", default=None, namespaces=NS) or None)
        updated = (entry.findtext("atom:updated", default=None, namespaces=NS) or None)

        authors: list[str] = []
        for author_el in entry.findall("atom:author", NS):
            name = (author_el.findtext("atom:name", default="", namespaces=NS) or "").strip()
            if name:
                authors.append(name)

        categories: list[str] = []
        for cat_el in entry.findall("atom:category", NS):
            term = (cat_el.attrib.get("term") or "").strip()
            if term:
                categories.append(term)

        # DOI is provided as arxiv:doi in many entries
        doi = (entry.findtext("arxiv:doi", default=None, namespaces=NS) or None)
        if doi:
            doi = doi.strip() or None

        pdf_url = self._extract_pdf_url(entry, arxiv_id)

        return ArxivPaper(
            arxiv_id=arxiv_id,
            title=title,
            abstract=abstract,
            authors=authors,
            published=published,
            updated=updated,
            categories=categories,
            pdf_url=pdf_url,
            doi=doi,
        )

    def _extract_pdf_url(self, entry: ET.Element, arxiv_id: str) -> str | None:
        # Prefer the explicit PDF link if present
        for link_el in entry.findall("atom:link", NS):
            href = (link_el.attrib.get("href") or "").strip()
            if not href:
                continue
            link_type = (link_el.attrib.get("type") or "").strip().lower()
            title = (link_el.attrib.get("title") or "").strip().lower()
            rel = (link_el.attrib.get("rel") or "").strip().lower()

            if title == "pdf" or link_type == "application/pdf":
                return href
            # Many entries have rel="related" title="pdf"
            if rel == "related" and ("pdf" in title or href.endswith(".pdf")):
                return href

        # Fallback to canonical PDF URL
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    def _build_search_query(self, query: str, category: str | None) -> str:
        q = query.strip()
        if not q:
            return "all:*"

        # If the user already provides explicit fielded query, don't wrap it.
        # arXiv supports field prefixes like all:, ti:, au:, abs:, cat: etc.
        fielded_prefixes = ("all:", "ti:", "au:", "abs:", "cat:", "id:")
        if q.lower().startswith(fielded_prefixes):
            base = q
        else:
            # all: applies to title+abstract+authors+comments etc.
            base = f"all:{q}"

        if category:
            base = f"({base}) AND cat:{category.strip()}"

        return base

    async def search(self, query: str, limit: int = 10, category: str | None = None) -> list[ArxivPaper]:
        """Search arXiv for papers.

        Args:
            query: Free-text query or arXiv fielded query.
            limit: Maximum number of results.
            category: Optional arXiv category (e.g., "cs.AI", "stat.ML").

        Returns:
            List of matching papers.
        """
        limit = max(1, min(int(limit), 1000))
        params = {
            "search_query": self._build_search_query(query, category),
            "start": 0,
            "max_results": limit,
            # Sort by relevance for search queries.
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        xml_text = await self._request(params)
        papers = self._parse_feed(xml_text)
        logger.info("arXiv search '%s' returned %s papers", query, len(papers))
        return papers

    async def get_paper(self, arxiv_id: str) -> ArxivPaper | None:
        """Get a paper by arXiv id (e.g., '2101.00001' or '2101.00001v2')."""
        arxiv_id = arxiv_id.strip()
        if not arxiv_id:
            return None

        params = {
            "id_list": arxiv_id,
            "start": 0,
            "max_results": 1,
        }
        xml_text = await self._request(params)
        papers = self._parse_feed(xml_text)
        return papers[0] if papers else None

    async def download_pdf(self, arxiv_id: str, output_path: str | Path) -> Path:
        """Download a paper PDF to output_path.

        Args:
            arxiv_id: arXiv identifier.
            output_path: File path to write.

        Returns:
            Path to the downloaded file.
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        # Use canonical PDF URL to avoid depending on API link presence.
        url = f"https://arxiv.org/pdf/{arxiv_id.strip()}.pdf"
        client = await self._get_client()

        for attempt in range(self.max_retries):
            await self._respect_rate_limit()
            try:
                async with client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    with output.open("wb") as f:
                        async for chunk in resp.aiter_bytes():
                            f.write(chunk)
                logger.info("Downloaded arXiv PDF %s -> %s", arxiv_id, output)
                return output
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status in {429, 500, 502, 503, 504} and attempt < self.max_retries - 1:
                    backoff = 2 ** attempt
                    logger.warning(
                        "PDF download failed (%s), retrying in %ss (attempt %s/%s)",
                        status,
                        backoff,
                        attempt + 1,
                        self.max_retries,
                    )
                    await asyncio.sleep(backoff)
                    continue
                logger.exception("Failed to download PDF (%s): %s", status, e)
                raise
            except httpx.RequestError as e:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        "PDF download error, retrying in 1s (attempt %s/%s): %s",
                        attempt + 1,
                        self.max_retries,
                        e,
                    )
                    await asyncio.sleep(1)
                    continue
                logger.exception("Failed to download PDF: %s", e)
                raise

        raise RuntimeError(f"Failed after {self.max_retries} attempts")


def search_arxiv(query: str, limit: int = 10, category: str | None = None) -> list[ArxivPaper]:
    """Synchronous wrapper for arXiv search."""

    async def _run() -> list[ArxivPaper]:
        client = ArxivClient()
        try:
            return await client.search(query=query, limit=limit, category=category)
        finally:
            await client.close()

    return asyncio.run(_run())
