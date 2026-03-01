from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ra.api import create_app
from ra.cli import build_parser, main


class _LLMStub:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[Any] = []

    def _next(self) -> str:
        if not self._responses:
            raise AssertionError("No more LLM stub responses available.")
        return self._responses.pop(0)

    async def ainvoke(self, messages: Any) -> SimpleNamespace:
        self.calls.append(messages)
        return SimpleNamespace(content=self._next())

    def invoke(self, messages: Any) -> SimpleNamespace:
        self.calls.append(messages)
        return SimpleNamespace(content=self._next())


class _ToolStub:
    def __init__(self, name: str, coroutine: Any) -> None:
        self.name = name
        self.coroutine = coroutine
        self.func = None


class _RetrieverStub:
    async def close(self) -> None:
        return None


def _assert_required_sections(review: str) -> None:
    required_sections = [
        "## Introduction",
        "## Thematic Groups",
        "## Key Findings",
        "## Research Gaps",
        "## Future Directions",
    ]
    for section in required_sections:
        assert section in review
    assert review.index("## Introduction") < review.index("## Future Directions")


@pytest.mark.asyncio
async def test_lit_review_agent_uses_tools_fulltext_and_structured_output(monkeypatch):
    from ra.agents.lit_review_agent import LitReviewAgent

    search_calls: list[tuple[str, int, str]] = []
    read_calls: list[str] = []

    async def _search_papers(query: str, limit: int = 10, source: str = "both") -> str:
        search_calls.append((query, limit, source))
        payload = {
            "query": query,
            "count": 3,
            "results": [
                {
                    "id": "p1",
                    "title": "Paper One",
                    "abstract": "A",
                    "authors": ["Alice Example", "Bob Example"],
                    "year": 2022,
                    "venue": "Venue",
                    "citation_count": 5,
                    "pdf_url": "https://example.org/p1.pdf",
                    "doi": None,
                    "arxiv_id": None,
                    "source": "semantic_scholar",
                },
                {
                    "id": "p2",
                    "title": "Paper Two",
                    "abstract": "B",
                    "authors": ["Carol Example", "Dan Example"],
                    "year": 2023,
                    "venue": "Venue",
                    "citation_count": 3,
                    "pdf_url": "https://example.org/p2.pdf",
                    "doi": None,
                    "arxiv_id": None,
                    "source": "arxiv",
                },
                {
                    "id": "p3",
                    "title": "Paper Three",
                    "abstract": "C",
                    "authors": ["Eve Example"],
                    "year": 2021,
                    "venue": "Venue",
                    "citation_count": 2,
                    "pdf_url": "https://example.org/p3.pdf",
                    "doi": None,
                    "arxiv_id": None,
                    "source": "semantic_scholar",
                },
            ],
        }
        return json.dumps(payload)

    async def _read_paper_fulltext(paper_id: str) -> str:
        read_calls.append(paper_id)
        return json.dumps(
            {
                "paper_id": paper_id,
                "title": f"Title {paper_id}",
                "abstract": f"Abstract {paper_id}",
                "methods": f"Methods {paper_id}",
                "results": f"Results {paper_id}",
                "discussion": f"Discussion {paper_id}",
                "conclusion": f"Conclusion {paper_id}",
            }
        )

    monkeypatch.setattr(
        "ra.agents.lit_review_agent.make_retrieval_tools",
        lambda retriever=None, semantic_scholar=None: [
            _ToolStub("search_papers", _search_papers),
            _ToolStub("read_paper_fulltext", _read_paper_fulltext),
        ],
        raising=False,
    )

    llm = _LLMStub(
        [
            json.dumps(
                {
                    "themes": [
                        {
                            "name": "Theme A",
                            "paper_ids": ["p1", "p2"],
                            "summary": "Model architecture patterns",
                        }
                    ]
                }
            ),
            "## Introduction\nTopic summary.\n\n## Key Findings\n- Signal appears in top papers.",
        ]
    )
    agent = LitReviewAgent(llm=llm, retriever=object(), search_limit=10)

    review = await agent.arun("graph neural networks", max_papers=2)

    assert search_calls == [("graph neural networks", 2, "both")]
    assert read_calls == ["p1", "p2"]
    _assert_required_sections(review)
    assert len(llm.calls) == 2


def test_cli_lit_review_command_routes_topic_and_max_papers(monkeypatch, capsys):
    parser = build_parser()
    args = parser.parse_args(["lit-review", "graph neural networks", "--max-papers", "4"])
    assert args.command == "lit-review"
    assert args.topic == "graph neural networks"
    assert args.max_papers == 4

    observed: dict[str, object] = {}

    class _StubLitReviewAgent:
        def run(self, topic: str, max_papers: int = 20) -> str:
            observed["topic"] = topic
            observed["max_papers"] = max_papers
            return (
                "## Introduction\nIntro.\n\n"
                "## Thematic Groups\n- Theme A\n\n"
                "## Key Findings\n- Finding A\n\n"
                "## Research Gaps\n- Gap A\n\n"
                "## Future Directions\n- Direction A"
            )

    monkeypatch.setattr("ra.cli.configure_logging_from_env", lambda: None)
    monkeypatch.setattr("ra.cli.LitReviewAgent", _StubLitReviewAgent)

    exit_code = main(["lit-review", "graph neural networks", "--max-papers", "4"])

    assert exit_code == 0
    assert observed == {"topic": "graph neural networks", "max_papers": 4}
    payload = json.loads(capsys.readouterr().out)
    assert payload["topic"] == "graph neural networks"
    _assert_required_sections(payload["review"])


def test_api_lit_review_endpoint_accepts_max_papers_and_returns_structure():
    observed: dict[str, object] = {}

    class _StubLitReviewAgent:
        async def arun(self, topic: str, max_papers: int) -> str:
            observed["topic"] = topic
            observed["max_papers"] = max_papers
            return (
                "## Introduction\nIntro.\n\n"
                "## Thematic Groups\n- Theme A\n\n"
                "## Key Findings\n- Finding A\n\n"
                "## Research Gaps\n- Gap A\n\n"
                "## Future Directions\n- Direction A"
            )

    app = create_app(
        retriever_factory=_RetrieverStub,
        lit_review_agent_factory=lambda: _StubLitReviewAgent(),
    )
    client = TestClient(app)

    resp = client.post(
        "/api/lit-review",
        json={"topic": "graph neural networks", "max_papers": 7},
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["topic"] == "graph neural networks"
    _assert_required_sections(payload["review"])
    assert observed == {"topic": "graph neural networks", "max_papers": 7}
