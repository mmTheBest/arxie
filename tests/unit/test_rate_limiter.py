from __future__ import annotations

import pytest

from ra.utils.rate_limiter import TokenBucketRateLimiter


class _FakeClock:
    def __init__(self) -> None:
        self.now_value = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.now_value

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now_value += seconds


@pytest.mark.asyncio
async def test_token_bucket_uses_burst_without_sleep() -> None:
    clock = _FakeClock()
    limiter = TokenBucketRateLimiter(
        rate_per_second=2.0,
        burst=2,
        time_fn=clock.now,
        sleep_fn=clock.sleep,
    )

    await limiter.acquire()
    await limiter.acquire()

    assert clock.sleeps == []


@pytest.mark.asyncio
async def test_token_bucket_waits_when_empty() -> None:
    clock = _FakeClock()
    limiter = TokenBucketRateLimiter(
        rate_per_second=1.0,
        burst=1,
        time_fn=clock.now,
        sleep_fn=clock.sleep,
    )

    await limiter.acquire()
    await limiter.acquire()

    assert clock.sleeps == [1.0]


@pytest.mark.asyncio
async def test_token_bucket_partially_refills_before_wait() -> None:
    clock = _FakeClock()
    limiter = TokenBucketRateLimiter(
        rate_per_second=2.0,
        burst=2,
        time_fn=clock.now,
        sleep_fn=clock.sleep,
    )

    await limiter.acquire()
    await limiter.acquire()

    clock.now_value += 0.25
    await limiter.acquire()

    assert clock.sleeps == [0.25]


def test_token_bucket_validates_configuration() -> None:
    with pytest.raises(ValueError):
        TokenBucketRateLimiter(rate_per_second=0.0, burst=1)

    with pytest.raises(ValueError):
        TokenBucketRateLimiter(rate_per_second=1.0, burst=0)
