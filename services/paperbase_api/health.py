"""Dependency-aware health and readiness checks for the Paperbase API service."""

from __future__ import annotations

from dataclasses import dataclass
import socket
from urllib.parse import urlparse

import httpx
from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from paperbase.config import PaperbaseConfig


@dataclass(frozen=True, slots=True)
class DependencyCheckResult:
    name: str
    ok: bool
    detail: str
    required: bool = True


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    ready: bool
    dependencies: list[DependencyCheckResult]


class DependencyChecker:
    """Production-oriented dependency probes for the Paperbase API service."""

    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        config: PaperbaseConfig,
        search_backend: object | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.config = config
        self.search_backend = search_backend
        self.http_client = http_client or httpx.Client(timeout=3.0)

    def check(self) -> ReadinessReport:
        dependencies = [
            self._check_database(),
            self._check_search(),
            self._check_redis(),
            self._check_object_store(),
        ]
        ready = all(result.ok or not result.required for result in dependencies)
        return ReadinessReport(ready=ready, dependencies=dependencies)

    def _check_database(self) -> DependencyCheckResult:
        try:
            with self.session_factory() as session:
                session.execute(text("select 1"))
        except Exception as exc:  # noqa: BLE001
            return DependencyCheckResult(name="database", ok=False, detail=str(exc))
        return DependencyCheckResult(name="database", ok=True, detail="ok")

    def _check_search(self) -> DependencyCheckResult:
        try:
            if self.search_backend is not None and hasattr(self.search_backend, "base_url"):
                response = self.http_client.get(str(getattr(self.search_backend, "base_url")))
                response.raise_for_status()
            else:
                self._tcp_probe(self.config.elasticsearch_url)
        except Exception as exc:  # noqa: BLE001
            return DependencyCheckResult(name="search", ok=False, detail=str(exc))
        return DependencyCheckResult(name="search", ok=True, detail="ok")

    def _check_redis(self) -> DependencyCheckResult:
        try:
            self._tcp_probe(self.config.redis_url)
        except Exception as exc:  # noqa: BLE001
            return DependencyCheckResult(name="redis", ok=False, detail=str(exc))
        return DependencyCheckResult(name="redis", ok=True, detail="ok")

    def _check_object_store(self) -> DependencyCheckResult:
        if self.config.object_store_backend.strip().lower() == "filesystem":
            try:
                from pathlib import Path

                path = Path(self.config.object_store_local_root)
                path.mkdir(parents=True, exist_ok=True)
            except Exception as exc:  # noqa: BLE001
                return DependencyCheckResult(name="object_store", ok=False, detail=str(exc))
            return DependencyCheckResult(name="object_store", ok=True, detail="ok")
        endpoint = self.config.object_store_endpoint.rstrip("/")
        probe_url = f"{endpoint}/minio/health/live" if endpoint.startswith(("http://", "https://")) else endpoint
        try:
            if probe_url.startswith(("http://", "https://")):
                response = self.http_client.get(probe_url)
                response.raise_for_status()
            else:
                self._tcp_probe(probe_url)
        except Exception as exc:  # noqa: BLE001
            return DependencyCheckResult(name="object_store", ok=False, detail=str(exc))
        return DependencyCheckResult(name="object_store", ok=True, detail="ok")

    def _tcp_probe(self, target_url: str) -> None:
        parsed = urlparse(target_url)
        host = parsed.hostname
        port = parsed.port
        if host is None:
            raise ValueError(f"Unable to resolve host from URL: {target_url}")
        if port is None:
            port = 443 if parsed.scheme == "https" else 80
        with socket.create_connection((host, port), timeout=3.0):
            return None
