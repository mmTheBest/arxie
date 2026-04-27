"""Storage resolution helpers for Paperbase file-backed workflows."""

from __future__ import annotations

import hashlib
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
    ) -> None:
        self.client = client or httpx.Client(timeout=60.0)
        self.cache_dir = cache_dir or Path(gettempdir()) / "paperbase-downloads"

    def resolve(self, storage_uri: str) -> Path:
        parsed = urlparse(storage_uri)
        if parsed.scheme == "file":
            return Path(unquote(parsed.path))
        if parsed.scheme in {"http", "https"}:
            return self._download_http(storage_uri)
        return Path(storage_uri)

    def _download_http(self, storage_uri: str) -> Path:
        parsed = urlparse(storage_uri)
        suffix = Path(parsed.path).suffix or ".pdf"
        file_name = f"{hashlib.sha256(storage_uri.encode('utf-8')).hexdigest()}{suffix}"
        download_path = self.cache_dir / file_name
        if download_path.exists():
            return download_path

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        response = self.client.get(storage_uri)
        response.raise_for_status()
        download_path.write_bytes(response.content)
        return download_path
