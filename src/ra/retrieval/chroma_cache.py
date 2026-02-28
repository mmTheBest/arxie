"""Local Chroma cache for normalized paper records.

This cache stores precomputed embeddings and paper metadata in a local Chroma
collection. It degrades gracefully when Chroma is unavailable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Callable

try:
    import chromadb
except Exception:  # pragma: no cover - depends on optional runtime install
    chromadb = None

logger = logging.getLogger(__name__)


class ChromaCache:
    """Local vector cache for paper records."""

    def __init__(
        self,
        *,
        persist_directory: str | Path = "data/chroma",
        collection_name: str = "papers",
        chroma_client: Any | None = None,
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self.persist_directory = str(persist_directory)
        self.collection_name = collection_name
        self._embedding_fn = embedding_fn or self._embed_text
        self._client = chroma_client
        self._collection: Any | None = None
        self.available = False

        if self._client is None:
            if chromadb is None:
                logger.info("ChromaDB unavailable; local retrieval cache disabled.")
                return
            try:
                Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(path=self.persist_directory)
            except Exception:
                logger.warning(
                    "Failed to initialize Chroma persistent client; cache disabled.",
                    exc_info=True,
                )
                return

        try:
            self._collection = self._client.get_or_create_collection(name=self.collection_name)
            self.available = True
        except Exception:
            logger.warning("Failed to initialize Chroma collection; cache disabled.", exc_info=True)
            self._collection = None
            self.available = False

    def has_paper(self, paper_id: str) -> bool:
        """Check whether a paper id exists in cache."""
        if not self.available or not paper_id.strip():
            return False
        try:
            payload = self._collection.get(ids=[paper_id], include=[])
        except Exception:
            logger.debug("Chroma get failed for paper_id=%s", paper_id, exc_info=True)
            return False
        ids = payload.get("ids") if isinstance(payload, dict) else None
        if not isinstance(ids, list):
            return False
        if ids and isinstance(ids[0], list):
            flat = [str(item) for item in ids[0]]
        else:
            flat = [str(item) for item in ids]
        return paper_id in flat

    def cache_paper(self, paper: Any) -> bool:
        """Store or update one paper in cache."""
        if not self.available:
            return False

        record = self._paper_to_record(paper)
        paper_id = str(record.get("id") or "").strip()
        if not paper_id:
            return False

        document = self._record_to_document(record)
        metadata = self._record_to_metadata(record)
        embedding = self._embedding_fn(document)

        try:
            self._collection.upsert(
                ids=[paper_id],
                documents=[document],
                metadatas=[metadata],
                embeddings=[embedding],
            )
            return True
        except Exception:
            logger.debug("Chroma upsert failed for paper_id=%s", paper_id, exc_info=True)
            return False

    def search_cached(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search cached papers using query embeddings."""
        if not self.available:
            return []

        limit = max(1, int(limit))
        query_text = str(query or "").strip()
        if not query_text:
            return []

        try:
            payload = self._collection.query(
                query_embeddings=[self._embedding_fn(query_text)],
                n_results=limit,
                include=["metadatas", "documents", "distances"],
            )
        except Exception:
            logger.debug("Chroma query failed for query=%r", query, exc_info=True)
            return []

        ids = self._unwrap(payload.get("ids"))
        metadatas = self._unwrap(payload.get("metadatas"))
        distances = self._unwrap(payload.get("distances"))

        out: list[dict[str, Any]] = []
        for idx, paper_id in enumerate(ids):
            metadata = metadatas[idx] if idx < len(metadatas) and isinstance(metadatas[idx], dict) else {}
            distance = distances[idx] if idx < len(distances) else None

            authors_raw = metadata.get("authors")
            authors: list[str]
            if isinstance(authors_raw, str):
                try:
                    parsed = json.loads(authors_raw)
                except json.JSONDecodeError:
                    parsed = []
                authors = [str(item) for item in parsed] if isinstance(parsed, list) else []
            elif isinstance(authors_raw, list):
                authors = [str(item) for item in authors_raw]
            else:
                authors = []

            out.append(
                {
                    "id": str(paper_id),
                    "title": str(metadata.get("title") or ""),
                    "abstract": metadata.get("abstract") or None,
                    "authors": authors,
                    "year": self._maybe_int(metadata.get("year")),
                    "venue": self._maybe_str(metadata.get("venue")),
                    "citation_count": self._maybe_int(metadata.get("citation_count")),
                    "pdf_url": self._maybe_str(metadata.get("pdf_url")),
                    "doi": self._maybe_str(metadata.get("doi")),
                    "arxiv_id": self._maybe_str(metadata.get("arxiv_id")),
                    "source": self._maybe_str(metadata.get("source")) or "semantic_scholar",
                    "distance": float(distance) if isinstance(distance, (int, float)) else None,
                }
            )

        return out

    @staticmethod
    def _unwrap(value: Any) -> list[Any]:
        if not isinstance(value, list):
            return []
        if value and isinstance(value[0], list):
            return list(value[0])
        return list(value)

    @staticmethod
    def _paper_to_record(paper: Any) -> dict[str, Any]:
        if isinstance(paper, dict):
            return dict(paper)

        return {
            "id": getattr(paper, "id", ""),
            "title": getattr(paper, "title", ""),
            "abstract": getattr(paper, "abstract", None),
            "authors": list(getattr(paper, "authors", []) or []),
            "year": getattr(paper, "year", None),
            "venue": getattr(paper, "venue", None),
            "citation_count": getattr(paper, "citation_count", None),
            "pdf_url": getattr(paper, "pdf_url", None),
            "doi": getattr(paper, "doi", None),
            "arxiv_id": getattr(paper, "arxiv_id", None),
            "source": getattr(paper, "source", "semantic_scholar"),
        }

    @staticmethod
    def _record_to_document(record: dict[str, Any]) -> str:
        title = str(record.get("title") or "").strip()
        abstract = str(record.get("abstract") or "").strip()
        return f"{title}\n\n{abstract}".strip() or title

    @staticmethod
    def _record_to_metadata(record: dict[str, Any]) -> dict[str, Any]:
        return {
            "title": str(record.get("title") or ""),
            "abstract": str(record.get("abstract") or ""),
            "authors": json.dumps(list(record.get("authors") or []), ensure_ascii=False),
            "year": record.get("year"),
            "venue": record.get("venue"),
            "citation_count": record.get("citation_count"),
            "pdf_url": record.get("pdf_url"),
            "doi": record.get("doi"),
            "arxiv_id": record.get("arxiv_id"),
            "source": record.get("source"),
        }

    @staticmethod
    def _maybe_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _maybe_str(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _embed_text(text: str, *, dimensions: int = 64) -> list[float]:
        """Deterministic local embedding that does not require network/model downloads."""
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
