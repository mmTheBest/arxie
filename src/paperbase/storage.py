"""Storage resolution helpers for Paperbase file-backed workflows."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from tempfile import gettempdir
from urllib.parse import unquote, urlparse

import httpx


class StorageResolver:
    """Resolve stored PDF URIs to local filesystem paths for downstream parsers."""

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        cache_dir: Path | None = None,
        object_store: object | None = None,
        cache_ttl_seconds: int = 86400,
    ) -> None:
        self.client = client or httpx.Client(timeout=60.0)
        self.cache_dir = cache_dir or Path(gettempdir()) / "paperbase-downloads"
        self.object_store = object_store
        self.cache_ttl_seconds = cache_ttl_seconds

    def resolve(self, storage_uri: str) -> Path:
        self.cleanup_cache()
        parsed = urlparse(storage_uri)
        if parsed.scheme == "file":
            return Path(unquote(parsed.path))
        if parsed.scheme == "s3":
            return self._download_object_store(storage_uri)
        if parsed.scheme in {"http", "https"}:
            return self._download_http(storage_uri)
        return Path(storage_uri)

    def cleanup_cache(self) -> None:
        if not self.cache_dir.exists():
            return
        cutoff = time.time() - self.cache_ttl_seconds
        for cached_file in self.cache_dir.glob("*"):
            if not cached_file.is_file():
                continue
            if cached_file.stat().st_mtime <= cutoff:
                cached_file.unlink(missing_ok=True)

    def _download_http(self, storage_uri: str) -> Path:
        parsed = urlparse(storage_uri)
        suffix = Path(parsed.path).suffix or ".pdf"
        file_name = f"{hashlib.sha256(storage_uri.encode('utf-8')).hexdigest()}{suffix}"
        download_path = self.cache_dir / file_name
        if self._is_fresh(download_path):
            return download_path

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        response = self.client.get(storage_uri)
        response.raise_for_status()
        download_path.write_bytes(response.content)
        return download_path

    def _download_object_store(self, storage_uri: str) -> Path:
        if self.object_store is None:
            raise ValueError("No object_store configured for s3:// storage resolution.")
        suffix = Path(urlparse(storage_uri).path).suffix or ".pdf"
        file_name = f"{hashlib.sha256(storage_uri.encode('utf-8')).hexdigest()}{suffix}"
        download_path = self.cache_dir / file_name
        if self._is_fresh(download_path):
            return download_path
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.object_store.download_to_path(
            storage_uri=storage_uri,
            destination_path=download_path,
        )

    def _is_fresh(self, path: Path) -> bool:
        if not path.exists():
            return False
        age_seconds = time.time() - path.stat().st_mtime
        return age_seconds <= self.cache_ttl_seconds
