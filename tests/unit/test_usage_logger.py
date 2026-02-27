from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from ra.utils.logging import UsageLogger


def test_usage_logger_writes_jsonl(tmp_path) -> None:
    path = tmp_path / "api-usage.jsonl"
    logger = UsageLogger(path)

    logger.log_api_call(
        endpoint="/v1/chat/completions",
        method="post",
        tokens_in=10,
        tokens_out=20,
        cost=0.123,
        response_time_ms=456,
        status=200,
    )

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    obj = json.loads(lines[0])
    assert set(obj.keys()) == {
        "timestamp",
        "endpoint",
        "method",
        "tokens_in",
        "tokens_out",
        "cost_estimate",
        "response_time_ms",
        "status",
    }
    assert obj["endpoint"] == "/v1/chat/completions"
    assert obj["method"] == "POST"
    assert obj["tokens_in"] == 10
    assert obj["tokens_out"] == 20
    assert obj["cost_estimate"] == 0.123
    assert obj["response_time_ms"] == 456
    assert obj["status"] == 200


def test_get_daily_summary_aggregates_only_today(tmp_path) -> None:
    path = tmp_path / "api-usage.jsonl"
    logger = UsageLogger(path)

    # One entry today (via API).
    logger.log_api_call(
        endpoint="/today",
        method="GET",
        tokens_in=1,
        tokens_out=2,
        cost=0.5,
        response_time_ms=100,
        status=200,
    )

    # One entry yesterday (manual line).
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    path.write_text(
        path.read_text(encoding="utf-8")
        + json.dumps(
            {
                "timestamp": yesterday,
                "endpoint": "/yesterday",
                "method": "GET",
                "tokens_in": 999,
                "tokens_out": 999,
                "cost_estimate": 999.0,
                "response_time_ms": 999,
                "status": 200,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = logger.get_daily_summary()
    assert summary["calls"] == 1
    assert summary["tokens_in"] == 1
    assert summary["tokens_out"] == 2
    assert summary["cost_estimate"] == 0.5
    assert summary["avg_response_time_ms"] == 100
