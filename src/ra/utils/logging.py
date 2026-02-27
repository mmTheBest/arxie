"""Logging utilities.

Includes:
- Standard Python logging setup
- JSONL usage logging for outbound API calls

This is designed to be safe, lightweight, and thread-safe.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from ra.utils.config import RAConfig


@dataclass(frozen=True, slots=True)
class UsageEntry:
    timestamp: str
    endpoint: str
    method: str
    tokens_in: int
    tokens_out: int
    cost_estimate: float
    response_time_ms: int
    status: int


class UsageLogger:
    """Thread-safe JSONL usage logger."""

    def __init__(self, path: str | Path = Path("data") / "api-usage.jsonl") -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def log_api_call(
        self,
        endpoint: str,
        method: str,
        tokens_in: int,
        tokens_out: int,
        cost: float,
        response_time_ms: int,
        status: int,
    ) -> UsageEntry:
        entry = UsageEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            endpoint=endpoint,
            method=method.upper(),
            tokens_in=int(tokens_in),
            tokens_out=int(tokens_out),
            cost_estimate=float(cost),
            response_time_ms=int(response_time_ms),
            status=int(status),
        )

        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(entry), ensure_ascii=False)

        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()

        return entry

    def get_daily_summary(self, day: date | None = None) -> dict[str, Any]:
        """Aggregate usage for the given day (defaults to today, UTC date)."""

        target_day = day or datetime.now(timezone.utc).date()

        totals = {
            "date": target_day.isoformat(),
            "calls": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "cost_estimate": 0.0,
            "response_time_ms_total": 0,
        }

        if not self.path.exists():
            return {**totals, "avg_response_time_ms": 0}

        with self._lock:
            with self.path.open("r", encoding="utf-8") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        obj = json.loads(raw)
                        ts = obj.get("timestamp")
                        if not ts:
                            continue
                        dt = datetime.fromisoformat(ts)
                        if dt.date() != target_day:
                            continue

                        totals["calls"] += 1
                        totals["tokens_in"] += int(obj.get("tokens_in", 0))
                        totals["tokens_out"] += int(obj.get("tokens_out", 0))
                        totals["cost_estimate"] += float(obj.get("cost_estimate", 0.0))
                        totals["response_time_ms_total"] += int(
                            obj.get("response_time_ms", 0)
                        )
                    except Exception:
                        # Ignore malformed lines; usage logs should never break runtime.
                        continue

        calls = int(totals["calls"])
        avg = int(totals["response_time_ms_total"] / calls) if calls else 0
        return {**totals, "avg_response_time_ms": avg}


_default_usage_logger = UsageLogger()


def log_api_call(
    endpoint: str,
    method: str,
    tokens_in: int,
    tokens_out: int,
    cost: float,
    response_time_ms: int,
    status: int,
) -> UsageEntry:
    """Convenience wrapper for the default UsageLogger."""

    return _default_usage_logger.log_api_call(
        endpoint=endpoint,
        method=method,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost=cost,
        response_time_ms=response_time_ms,
        status=status,
    )


def get_daily_summary() -> dict[str, Any]:
    """Convenience wrapper for the default UsageLogger."""

    return _default_usage_logger.get_daily_summary()


def setup_logging(config: RAConfig) -> None:
    """Configure Python logging.

    - Console handler always enabled
    - File handler writes to {config.ra_log_dir}/ra.log

    This function is idempotent-ish: it configures the root logger once per
    interpreter in a way that won't duplicate handlers if called repeatedly.
    """

    level_name = (config.ra_log_level or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    root = logging.getLogger()
    root.setLevel(level)

    # Prevent handler duplication.
    if getattr(root, "_ra_logging_configured", False):
        return

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    log_dir = Path(config.ra_log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "ra.log", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    setattr(root, "_ra_logging_configured", True)
