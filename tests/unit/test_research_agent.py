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
        self.inputs: list[dict[str, Any]] = []

    def invoke(self, inputs: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        self.inputs.append(inputs)
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
    agent.deep_search = False
    agent.max_iterations = 7
    agent.max_api_retries = 2
    agent.retry_base_delay_seconds = 0.0
    agent._callback = SimpleNamespace(papers={})
    agent._citation_formatter = CitationFormatter()
    agent._message_histories = {}
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


def _paper_with(*, paper_id: str, title: str, source: str = "semantic_scholar") -> Paper:
    return Paper(
        id=paper_id,
        title=title,
        abstract=None,
        authors=["Author One"],
        year=2020,
        venue="Venue",
        citation_count=0,
        pdf_url="https://example.org/paper.pdf",
        doi=None,
        arxiv_id=None,
        source=source,
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


def test_run_deep_search_uses_prepared_query_and_recursion_limit():
    graph = _GraphStub([{"messages": [{"role": "assistant", "content": "Deep answer."}]}])
    agent = _mk_agent(graph)
    agent.deep_search = True
    agent.max_iterations = 50

    prepared_queries: list[str] = []
    agent._prepare_query_sync = lambda query: prepared_queries.append(query) or "DEEP QUERY"
    agent._ensure_seed_papers_sync = lambda _: None

    output = agent.run("transformers")

    assert prepared_queries == ["transformers"]
    assert graph.configs[0]["recursion_limit"] == 50
    assert "DEEP QUERY" in str(graph.inputs[0]["messages"][0].content)
    assert "Deep answer." in output


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
    graph = _GraphStub(
        [{"messages": [{"role": "assistant", "content": "Transformers improved MT quality."}]}]
    )
    agent = _mk_agent(graph)
    p = _paper()

    def _seed(_: str) -> None:
        agent._callback.papers[p.id] = p

    agent._ensure_seed_papers_sync = _seed
    output = agent.run("transformers")

    assert "(Vaswani et al., 2017)" in output
    assert "\n## References\n" in output
    assert "Vaswani, A." in output


def test_format_references_adds_confidence_annotations_for_major_claims():
    agent = _mk_agent(_GraphStub([]))
    supporting = Paper(
        id="supporting-paper",
        title="Transformer Gains in Translation",
        abstract="This study shows transformers improve translation quality across benchmarks.",
        authors=["Alex Smith"],
        year=2020,
        venue="ACL",
        citation_count=0,
        pdf_url=None,
        doi=None,
        arxiv_id=None,
        source="semantic_scholar",
    )
    contradicting = Paper(
        id="contradicting-paper",
        title="Negative Findings for Transformer Translation",
        abstract="This study found transformers did not improve translation quality and performed worse.",
        authors=["Jamie Lee"],
        year=2021,
        venue="EMNLP",
        citation_count=0,
        pdf_url=None,
        doi=None,
        arxiv_id=None,
        source="semantic_scholar",
    )
    agent._callback.papers[supporting.id] = supporting
    agent._callback.papers[contradicting.id] = contradicting

    output = agent._format_references("Transformers improve translation quality (Smith, 2020).")

    assert "[Confidence:" in output
    assert "supporting" in output
    assert "contradicting" in output


def test_run_with_session_id_keeps_multi_turn_message_history():
    graph = _GraphStub(
        [
            {"messages": [{"role": "assistant", "content": "First answer."}]},
            {"messages": [{"role": "assistant", "content": "Second answer."}]},
        ]
    )
    agent = _mk_agent(graph)

    _ = agent.run("What is RAG?", session_id="session-1")
    _ = agent.run("How does it fail?", session_id="session-1")

    assert len(graph.inputs[0]["messages"]) == 1
    second_turn_messages = graph.inputs[1]["messages"]
    assert len(second_turn_messages) >= 3
    assert "What is RAG?" in str(second_turn_messages[0].content)
    assert "First answer." in str(second_turn_messages[1].content)
    assert "How does it fail?" in str(second_turn_messages[-1].content)


def test_run_with_distinct_session_ids_keeps_histories_isolated():
    graph = _GraphStub(
        [
            {"messages": [{"role": "assistant", "content": "S1 answer."}]},
            {"messages": [{"role": "assistant", "content": "S2 answer."}]},
        ]
    )
    agent = _mk_agent(graph)

    _ = agent.run("Session one question", session_id="session-1")
    _ = agent.run("Session two question", session_id="session-2")

    assert len(graph.inputs[0]["messages"]) == 1
    assert len(graph.inputs[1]["messages"]) == 1
    assert "Session two question" in str(graph.inputs[1]["messages"][0].content)


@pytest.mark.asyncio
async def test_build_deep_search_context_reads_top_3_then_chases_citations():
    initial = [
        _paper_with(paper_id="p1", title="Initial 1"),
        _paper_with(paper_id="p2", title="Initial 2"),
        _paper_with(paper_id="p3", title="Initial 3"),
        _paper_with(paper_id="p4", title="Initial 4"),
    ]
    followups = {
        "Citation A": [_paper_with(paper_id="c1", title="Citation A result", source="arxiv")],
        "Citation B": [
            _paper_with(
                paper_id="c2",
                title="Citation B result",
                source="semantic_scholar",
            )
        ],
        "Citation C": [_paper_with(paper_id="c3", title="Citation C result", source="arxiv")],
    }

    class _RetrieverStub:
        def __init__(self) -> None:
            self.search_calls: list[str] = []
            self.full_text_calls: list[str] = []

        async def search(
            self, query: str, limit: int, sources: tuple[str, str]
        ) -> list[Paper]:
            _ = (limit, sources)
            self.search_calls.append(query)
            if query == "transformers":
                return initial
            return followups.get(query, [])

        async def get_full_text(self, paper: Paper) -> str:
            self.full_text_calls.append(paper.id)
            return f"full text for {paper.id}"

    class _SemanticScholarStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []

        async def get_citations(self, paper_id: str, limit: int) -> list[Any]:
            self.calls.append((paper_id, limit))
            mapping = {
                "p1": [SimpleNamespace(title="Citation A")],
                "p2": [SimpleNamespace(title="Citation B")],
                "p3": [SimpleNamespace(title="Citation C")],
            }
            return mapping.get(paper_id, [])

    agent = _mk_agent(_GraphStub([]))
    agent.deep_search = True
    agent.retriever = _RetrieverStub()
    agent.semantic_scholar = _SemanticScholarStub()

    context = await agent._build_deep_search_context_async("transformers")

    assert agent.retriever.search_calls[0] == "transformers"
    assert agent.retriever.full_text_calls == ["p1", "p2", "p3"]
    assert agent.semantic_scholar.calls == [("p1", 5), ("p2", 5), ("p3", 5)]
    assert set(agent.retriever.search_calls[1:]) == {"Citation A", "Citation B", "Citation C"}
    assert {"c1", "c2", "c3"}.issubset(agent._callback.papers.keys())
    assert "Initial search" in context
    assert "Full-text review" in context
    assert "Citation chasing" in context


def test_init_uses_higher_iteration_budget_for_deep_search(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("RA_AGENT_MAX_ITERATIONS", raising=False)
    monkeypatch.delenv("RA_AGENT_DEEP_MAX_ITERATIONS", raising=False)
    monkeypatch.setattr("ra.agents.research_agent.configure_logging_from_env", lambda: None)
    monkeypatch.setattr(
        "ra.agents.research_agent.ChatOpenAI",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "ra.agents.research_agent.UnifiedRetriever",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "ra.agents.research_agent.SemanticScholarClient",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "ra.agents.research_agent.make_retrieval_tools",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        "ra.agents.research_agent.create_agent",
        lambda **kwargs: SimpleNamespace(invoke=lambda inputs, config: {}),
    )

    normal = ResearchAgent(deep_search=False)
    deep = ResearchAgent(deep_search=True)

    assert deep.max_iterations > normal.max_iterations


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
