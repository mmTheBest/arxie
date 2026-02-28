"""Async token bucket rate limiter."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable


class TokenBucketRateLimiter:
    """Token-bucket limiter with async waiting.

    Args:
        rate_per_second: Refill rate in tokens per second.
        burst: Bucket capacity in tokens.
    """

    def __init__(
        self,
        *,
        rate_per_second: float,
        burst: int,
        time_fn: Callable[[], float] | None = None,
        sleep_fn: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        if rate_per_second <= 0:
            raise ValueError("rate_per_second must be > 0")
        if burst <= 0:
            raise ValueError("burst must be > 0")

        self.rate_per_second = float(rate_per_second)
        self.burst = float(burst)
        self._time_fn = time_fn or time.monotonic
        self._sleep_fn = sleep_fn or asyncio.sleep

        self._tokens = float(burst)
        self._last_refill = self._time_fn()
        self._lock = asyncio.Lock()

    def _refill(self, now: float) -> None:
        elapsed = max(0.0, now - self._last_refill)
        if elapsed > 0:
            self._tokens = min(self.burst, self._tokens + elapsed * self.rate_per_second)
            self._last_refill = now

    async def acquire(self, tokens: float = 1.0) -> None:
        """Acquire tokens, waiting asynchronously when bucket is empty."""
        if tokens <= 0:
            return
        if tokens > self.burst:
            raise ValueError("tokens cannot exceed burst capacity")

        while True:
            wait_time = 0.0
            async with self._lock:
                now = self._time_fn()
                self._refill(now)

                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return

                missing = tokens - self._tokens
                wait_time = missing / self.rate_per_second

            await self._sleep_fn(wait_time)
