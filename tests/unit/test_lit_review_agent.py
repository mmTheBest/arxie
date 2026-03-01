from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from ra.agents.lit_review_agent import LitReviewAgent
from ra.retrieval.unified import Paper


class _LLMStub:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[Any] = []

    def _next(self) -> str:
        if not self._responses:
            raise AssertionError("No more LLM stub responses available.")
        return self._responses.pop(0)

    def invoke(self, messages: Any) -> SimpleNamespace:
        self.calls.append(messages)
        return SimpleNamespace(content=self._next())

    async def ainvoke(self, messages: Any) -> SimpleNamespace:
        self.calls.append(messages)
        return SimpleNamespace(content=self._next())


class _RetrieverStub:
    def __init__(self, papers: list[Paper]) -> None:
        self._papers = papers
        self.search_calls: list[tuple[str, int, tuple[str, ...]]] = []

    async def search(
        self,
        query: str,
        limit: int,
        sources: tuple[str, ...],
    ) -> list[Paper]:
        self.search_calls.append((query, limit, sources))
        return list(self._papers)


def _paper(paper_id: str, title: str, year: int) -> Paper:
    return Paper(
        id=paper_id,
        title=title,
        abstract="Abstract",
        authors=["Alice Example", "Bob Example"],
        year=year,
        venue="Venue",
        citation_count=5,
        pdf_url=None,
        doi=None,
        arxiv_id=None,
        source="semantic_scholar",
    )


@pytest.mark.asyncio
async def test_arun_generates_lit_review_with_required_sections():
    papers = [_paper("p1", "Paper One", 2022), _paper("p2", "Paper Two", 2023)]
    cluster_payload = json.dumps(
        {
            "themes": [
                {"name": "Model Architectures", "paper_ids": ["p1"], "summary": "Architecture"},
                {"name": "Evaluation Methods", "paper_ids": ["p2"], "summary": "Evaluation"},
            ]
        }
    )
    review_payload = (
        "## Introduction\nIntro text.\n\n"
        "## Thematic Groups\n- Group A\n\n"
        "## Key Findings\n- Finding A\n\n"
        "## Research Gaps\n- Gap A\n\n"
        "## Future Directions\n- Direction A"
    )

    llm = _LLMStub([cluster_payload, review_payload])
    retriever = _RetrieverStub(papers)
    agent = LitReviewAgent(llm=llm, retriever=retriever, search_limit=8)

    output = await agent.arun("graph neural networks")

    required_sections = [
        "## Introduction",
        "## Thematic Groups",
        "## Key Findings",
        "## Research Gaps",
        "## Future Directions",
    ]
    for section in required_sections:
        assert section in output
    assert output.index("## Introduction") < output.index("## Future Directions")
    assert retriever.search_calls == [
        ("graph neural networks", 8, ("semantic_scholar", "arxiv"))
    ]
    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_arun_backfills_missing_sections_from_llm_output():
    papers = [_paper("p1", "Paper One", 2022)]
    cluster_payload = json.dumps(
        {
            "themes": [
                {"name": "General", "paper_ids": ["p1"], "summary": "General summary"},
            ]
        }
    )
    llm = _LLMStub([cluster_payload, "## Introduction\nOnly intro text."])
    retriever = _RetrieverStub(papers)
    agent = LitReviewAgent(llm=llm, retriever=retriever)

    output = await agent.arun("federated learning")

    assert "## Introduction" in output
    assert "## Thematic Groups" in output
    assert "## Key Findings" in output
    assert "## Research Gaps" in output
    assert "## Future Directions" in output
