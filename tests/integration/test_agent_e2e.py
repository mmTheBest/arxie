from __future__ import annotations

import json
import os
import re
import socket
from pathlib import Path

import pytest

from ra.agents.research_agent import ResearchAgent


pytestmark = pytest.mark.integration


_CITATION_RE = re.compile(
    r"\([A-Z][A-Za-z\-]+(?:\s+et\s+al\.)?(?:\s*&\s*[A-Z][A-Za-z\-]+)?,\s*(?:19|20)\d{2}\)"
)


def _load_project_dotenv() -> None:
    """Load .env from project root (best-effort)."""

    try:
        from dotenv import load_dotenv

        root = Path(__file__).resolve().parents[2]
        load_dotenv(root / ".env")
    except Exception:
        return


def _usage_log_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "api-usage.jsonl"


def _usage_snapshot() -> int:
    p = _usage_log_path()
    if not p.exists():
        return 0
    return sum(1 for _ in p.open("r", encoding="utf-8"))


def _print_usage_delta(start_line: int) -> None:
    p = _usage_log_path()
    if not p.exists():
        print("[usage] no api-usage.jsonl written")
        return

    tokens_in = 0
    tokens_out = 0
    cost = 0.0
    new_lines = 0

    with p.open("r", encoding="utf-8") as f:
        for i, raw in enumerate(f):
            if i < start_line:
                continue
            raw = raw.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            new_lines += 1
            tokens_in += int(obj.get("tokens_in", 0))
            tokens_out += int(obj.get("tokens_out", 0))
            cost += float(obj.get("cost_estimate", 0.0))

    print(
        f"[usage] calls={new_lines} tokens_in={tokens_in} tokens_out={tokens_out} cost_estimate_usd={cost:.6f}"
    )


def _skip_if_no_key() -> None:
    if not (os.getenv("OPENAI_API_KEY") or "").strip():
        pytest.skip("OPENAI_API_KEY not set")


def _skip_if_openai_unreachable() -> None:
    try:
        host = socket.gethostbyname("api.openai.com")
        with socket.create_connection((host, 443), timeout=3):
            return
    except OSError as exc:
        pytest.skip(f"OpenAI endpoint unreachable: {exc}")


def test_agent_answers_factual_query() -> None:
    _load_project_dotenv()
    _skip_if_no_key()
    _skip_if_openai_unreachable()

    start = _usage_snapshot()

    agent = ResearchAgent(verbose=False)
    resp = agent.run(
        "Who wrote the Attention Is All You Need paper and what year was it published?"
    )

    _print_usage_delta(start)

    low = resp.lower()
    assert "vaswani" in low
    assert "2017" in resp


def test_agent_cites_sources() -> None:
    _load_project_dotenv()
    _skip_if_no_key()
    _skip_if_openai_unreachable()

    start = _usage_snapshot()

    agent = ResearchAgent(verbose=False)
    resp = agent.run("What are the main approaches to retrieval-augmented generation?")

    _print_usage_delta(start)

    assert _CITATION_RE.search(resp), f"Expected at least 1 inline citation, got:\n{resp}"


def test_agent_handles_unknown_topic() -> None:
    _load_project_dotenv()
    _skip_if_no_key()
    _skip_if_openai_unreachable()

    start = _usage_snapshot()

    agent = ResearchAgent(verbose=False)
    resp = agent.run(
        "What are the key results of the flibbertigibbet quantum pineapple theorem in intergalactic databases?"
    )

    _print_usage_delta(start)

    low = resp.lower()
    assert (
        "cannot find" in low
        or "could not find" in low
        or "no relevant" in low
        or "not able to find" in low
    ), f"Expected a graceful 'unknown / not found' style response, got:\n{resp}"

    # Should not hallucinate author-year citations when it can't find relevant papers.
    assert not _CITATION_RE.search(resp), f"Unexpected citation-like pattern in:\n{resp}"
