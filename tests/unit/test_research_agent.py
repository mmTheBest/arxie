from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages.tool import ToolMessage

from ra.agents.research_agent import ResearchAgent, ResearchAgentCallback
from ra.citation.formatter import CitationFormatter
from ra.retrieval.unified import Paper


class _GraphStub:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls = 0
        self.configs: list[dict[str, Any]] = []

    def invoke(self, inputs: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        _ = inputs
        self.configs.append(config)
        i = self.calls
        self.calls += 1
        response = self._responses[i]
        if isinstance(response, Exception):
            raise response
        return response


class _UsageLoggerStub:
    def log_api_call(self, **kwargs: Any) -> None:
        _ = kwargs


def _mk_agent(graph: _GraphStub) -> ResearchAgent:
    agent = object.__new__(ResearchAgent)
    agent.graph = graph
    agent.max_iterations = 7
    agent.max_api_retries = 2
    agent.retry_base_delay_seconds = 0.0
    agent._callback = SimpleNamespace(papers={})
    agent._citation_formatter = CitationFormatter()
    return agent


def _paper() -> Paper:
    return Paper(
        id="p1",
        title="Attention Is All You Need",
        abstract=None,
        authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
        year=2017,
        venue="NeurIPS",
        citation_count=0,
        pdf_url=None,
        doi="10.5555/3295222.3295349",
        arxiv_id=None,
        source="semantic_scholar",
    )


def test_format_references_returns_structured_markdown_sections():
    agent = _mk_agent(_GraphStub([]))
    p = _paper()
    agent._callback.papers[p.id] = p

    output = agent._format_references("Transformers improved MT quality (Vaswani et al., 2017).")

    assert output.startswith("## Answer\n")
    assert "Transformers improved MT quality (Vaswani et al., 2017)." in output
    assert "\n## References\n" in output
    assert "Vaswani, A." in output


def test_format_references_injects_inline_citation_when_missing():
    agent = _mk_agent(_GraphStub([]))
    p = _paper()
    agent._callback.papers[p.id] = p

    output = agent._format_references("Transformers improved MT quality.")

    assert "(Vaswani et al., 2017)" in output
    assert "\n## References\n" in output
    assert "Vaswani, A." in output


def test_run_retries_transient_errors_with_backoff(monkeypatch: pytest.MonkeyPatch):
    sleep_calls: list[float] = []
    monkeypatch.setattr("ra.agents.research_agent.time.sleep", lambda s: sleep_calls.append(s))

    graph = _GraphStub(
        [
            TimeoutError("request timed out"),
            {"messages": [{"role": "assistant", "content": "Final answer."}]},
        ]
    )
    agent = _mk_agent(graph)

    output = agent.run("query")

    assert graph.calls == 2
    assert len(sleep_calls) == 1
    assert graph.configs[0]["recursion_limit"] == 7
    assert "Final answer." in output


def test_run_handles_non_transient_invoke_errors():
    graph = _GraphStub([ValueError("bad input")])
    agent = _mk_agent(graph)

    output = agent.run("query")

    assert output.startswith("## Answer\n")
    assert "internal error" in output.lower()
    assert "\n## References\nNone." in output


def test_run_rejects_empty_query_without_invoking_graph():
    graph = _GraphStub([{"messages": [{"role": "assistant", "content": "should not be used"}]}])
    agent = _mk_agent(graph)

    output = agent.run("   ")

    assert graph.calls == 0
    assert "non-empty query" in output.lower()


def test_run_rejects_control_character_query_without_invoking_graph():
    graph = _GraphStub([{"messages": [{"role": "assistant", "content": "should not be used"}]}])
    agent = _mk_agent(graph)

    output = agent.run("graph\x00query")

    assert graph.calls == 0
    assert "non-empty query" in output.lower()


def test_run_backfills_seed_papers_when_tool_loop_collects_none():
    graph = _GraphStub([{"messages": [{"role": "assistant", "content": "Transformers improved MT quality."}]}])
    agent = _mk_agent(graph)
    p = _paper()

    def _seed(_: str) -> None:
        agent._callback.papers[p.id] = p

    agent._ensure_seed_papers_sync = _seed
    output = agent.run("transformers")

    assert "(Vaswani et al., 2017)" in output
    assert "\n## References\n" in output
    assert "Vaswani, A." in output


def test_callback_collects_papers_from_toolmessage_content():
    callback = ResearchAgentCallback(usage_logger=_UsageLoggerStub(), model="gpt-4o-mini")
    payload = {
        "query": "transformer",
        "count": 1,
        "results": [
            {
                "id": "paper-1",
                "title": "Attention Is All You Need",
                "abstract": None,
                "authors": ["Ashish Vaswani", "Noam Shazeer"],
                "year": 2017,
                "venue": "NeurIPS",
                "citation_count": 0,
                "pdf_url": None,
                "doi": "10.5555/3295222.3295349",
                "arxiv_id": "1706.03762",
                "source": "semantic_scholar",
            }
        ],
    }
    message = ToolMessage(content=json.dumps(payload), tool_call_id="call_1")

    callback.on_tool_start({"name": "search_papers"}, "{}")
    callback.on_tool_end(message)

    assert "paper-1" in callback.papers
