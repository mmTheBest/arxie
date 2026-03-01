from __future__ import annotations

import json

import pytest

from ra.retrieval.semantic_scholar import Author, Paper
from ra.tools.retrieval_tools import TraceInfluenceArgs, make_retrieval_tools


def _s2_paper(paper_id: str, title: str, year: int | None) -> Paper:
    return Paper(
        paper_id=paper_id,
        title=title,
        abstract=None,
        year=year,
        authors=[Author(author_id="a1", name="Author One")],
        venue="Test Venue",
        citation_count=0,
        is_open_access=False,
        pdf_url=None,
        doi=None,
        arxiv_id=None,
        external_ids={},
    )


@pytest.mark.asyncio
async def test_trace_influence_tool_exists_and_has_expected_schema():
    class DummyRetriever:
        async def search(self, query: str, limit: int, sources: tuple[str, ...]):  # noqa: ARG002
            return []

    class DummySemanticScholar:
        async def get_paper(self, paper_id: str):  # noqa: ARG002
            return None

        async def search(self, query: str, limit: int = 10):  # noqa: ARG002
            return []

        async def get_citations(self, paper_id: str, limit: int):  # noqa: ARG002
            return []

    tools = make_retrieval_tools(
        retriever=DummyRetriever(),
        semantic_scholar=DummySemanticScholar(),
    )
    tool = next(t for t in tools if t.name == "trace_influence")

    assert tool.args_schema is TraceInfluenceArgs
    schema = tool.args_schema.model_json_schema()
    assert schema["required"] == ["paper"]
    assert "paper" in schema["properties"]


@pytest.mark.asyncio
async def test_trace_influence_builds_chronological_timeline_from_title():
    seed = _s2_paper("seed-2017", "Seed Paper", 2017)
    citing_2019 = _s2_paper("citing-2019", "Citing Paper 2019", 2019)
    citing_2018 = _s2_paper("citing-2018", "Citing Paper 2018", 2018)
    citing_2020 = _s2_paper("citing-2020", "Second Hop Paper 2020", 2020)

    class DummyRetriever:
        async def search(self, query: str, limit: int, sources: tuple[str, ...]):  # noqa: ARG002
            return []

    class DummySemanticScholar:
        def __init__(self) -> None:
            self.get_paper_calls: list[str] = []
            self.search_calls: list[tuple[str, int]] = []
            self.citation_calls: list[tuple[str, int]] = []

        async def get_paper(self, paper_id: str):
            self.get_paper_calls.append(paper_id)
            return None

        async def search(self, query: str, limit: int = 10):
            self.search_calls.append((query, limit))
            return [seed] if query == "seed paper title" else []

        async def get_citations(self, paper_id: str, limit: int):
            self.citation_calls.append((paper_id, limit))
            mapping = {
                "seed-2017": [citing_2019, citing_2018],
                "citing-2018": [citing_2020],
                "citing-2019": [citing_2020],
            }
            return mapping.get(paper_id, [])

    s2 = DummySemanticScholar()
    tools = make_retrieval_tools(retriever=DummyRetriever(), semantic_scholar=s2)
    tool = next(t for t in tools if t.name == "trace_influence")

    raw = await tool.ainvoke(
        {"paper": "seed paper title", "max_depth": 2, "citations_per_paper": 5}
    )
    payload = json.loads(raw)

    assert payload["input"] == "seed paper title"
    assert payload["seed_paper"]["paper_id"] == "seed-2017"
    assert [item["paper_id"] for item in payload["timeline"]] == [
        "seed-2017",
        "citing-2018",
        "citing-2019",
        "citing-2020",
    ]
    assert [item["year"] for item in payload["timeline"]] == [2017, 2018, 2019, 2020]

    links_by_paper = {
        item["paper_id"]: {
            (link["from_paper_id"], link["to_paper_id"]) for link in item["citation_links"]
        }
        for item in payload["timeline"]
    }
    assert links_by_paper["seed-2017"] == set()
    assert links_by_paper["citing-2018"] == {("seed-2017", "citing-2018")}
    assert links_by_paper["citing-2019"] == {("seed-2017", "citing-2019")}
    assert links_by_paper["citing-2020"] == {
        ("citing-2018", "citing-2020"),
        ("citing-2019", "citing-2020"),
    }
    assert s2.search_calls == [("seed paper title", 1)]
    assert s2.citation_calls == [
        ("seed-2017", 5),
        ("citing-2019", 5),
        ("citing-2018", 5),
    ]


@pytest.mark.asyncio
async def test_trace_influence_returns_not_found_error():
    class DummyRetriever:
        async def search(self, query: str, limit: int, sources: tuple[str, ...]):  # noqa: ARG002
            return []

    class DummySemanticScholar:
        async def get_paper(self, paper_id: str):  # noqa: ARG002
            return None

        async def search(self, query: str, limit: int = 10):  # noqa: ARG002
            return []

        async def get_citations(self, paper_id: str, limit: int):  # noqa: ARG002
            return []

    tools = make_retrieval_tools(
        retriever=DummyRetriever(),
        semantic_scholar=DummySemanticScholar(),
    )
    tool = next(t for t in tools if t.name == "trace_influence")

    raw = await tool.ainvoke({"paper": "missing paper"})
    payload = json.loads(raw)

    assert payload["tool"] == "trace_influence"
    assert payload["error"] == "paper_not_found"


@pytest.mark.asyncio
async def test_trace_influence_handles_internal_errors_gracefully():
    seed = _s2_paper("seed-2017", "Seed Paper", 2017)

    class DummyRetriever:
        async def search(self, query: str, limit: int, sources: tuple[str, ...]):  # noqa: ARG002
            return []

    class DummySemanticScholar:
        async def get_paper(self, paper_id: str):
            return seed if paper_id == "seed-2017" else None

        async def search(self, query: str, limit: int = 10):  # noqa: ARG002
            return []

        async def get_citations(self, paper_id: str, limit: int):  # noqa: ARG002
            raise RuntimeError("citations backend down")

    tools = make_retrieval_tools(
        retriever=DummyRetriever(),
        semantic_scholar=DummySemanticScholar(),
    )
    tool = next(t for t in tools if t.name == "trace_influence")

    raw = await tool.ainvoke({"paper": "seed-2017"})
    payload = json.loads(raw)

    assert payload["tool"] == "trace_influence"
    assert payload["error"] == "tool_execution_failed"
