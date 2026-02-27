from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ra.agents.research_agent import ResearchAgent
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
