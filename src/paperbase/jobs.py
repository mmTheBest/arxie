"""Queue dispatch helpers for Paperbase background jobs."""

from __future__ import annotations

from collections.abc import Callable
from math import ceil
from typing import Any, Protocol, cast

from paperbase.config import PaperbaseConfig


class JobDispatcher(Protocol):
    def dispatch(self, job_id: str, *, project_id: str | None = None) -> None: ...


class JobConsumer(Protocol):
    def receive(self, timeout_seconds: float | None = None) -> str | None: ...


class RedisJobQueue(JobDispatcher, JobConsumer):
    """Minimal Redis-backed queue carrying background job ids."""

    def __init__(
        self,
        *,
        redis_url: str,
        queue_name: str,
        project_id: str | None = None,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        if client_factory is None:
            from redis import Redis  # type: ignore[import-untyped]

            client_factory = Redis.from_url
        self.client = client_factory(redis_url, decode_responses=True)
        self.queue_name = queue_name
        self.project_id = project_id

    def dispatch(self, job_id: str, *, project_id: str | None = None) -> None:
        self.client.lpush(self._queue_name(project_id or self.project_id), job_id)

    def receive(self, timeout_seconds: float | None = None) -> str | None:
        timeout = 0 if timeout_seconds is None else max(1, ceil(timeout_seconds))
        item = self.client.brpop(self._queue_name(self.project_id), timeout=timeout)
        if item is None:
            return None
        _, job_id = cast(tuple[str, str], item)
        return job_id

    def _queue_name(self, project_id: str | None) -> str:
        if not project_id:
            return self.queue_name
        safe_project_id = "".join(
            character if character.isalnum() or character in {"-", "_", "."} else "_"
            for character in project_id
        )
        return f"{self.queue_name}:{safe_project_id}"


def build_job_queue(config: PaperbaseConfig, *, project_id: str | None = None) -> object | None:
    if config.worker_queue_backend.strip().lower() != "redis":
        return None
    return RedisJobQueue(
        redis_url=config.redis_url,
        queue_name=config.worker_queue_name,
        project_id=project_id,
    )
