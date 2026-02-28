from __future__ import annotations

from tests.eval.qa_agent import ResearcherQAAgent


class _MockResearchAgent:
    def __init__(self, responses: dict[str, str]) -> None:
        self._responses = responses
        self.queries: list[str] = []

    def run(self, query: str) -> str:
        self.queries.append(query)
        return self._responses[query]


def test_qa_agent_all_checks_pass() -> None:
    agent = _MockResearchAgent(
        {
            "q1": (
                "## Answer\n"
                "Transformers improved sequence modeling quality (Vaswani et al., 2017).\n\n"
                "## References\n"
                "1. Vaswani, A., et al. (2017). Attention Is All You Need."
            )
        }
    )

    qa = ResearcherQAAgent(agent)
    report = qa.evaluate(["q1"])

    assert report["total_queries"] == 1
    assert report["overall_score"] == 1.0

    result = report["results"][0]
    assert result["query"] == "q1"
    assert result["checks"]["has_answer_section"] is True
    assert result["checks"]["has_references_section"] is True
    assert result["checks"]["has_inline_citations"] is True
    assert result["checks"]["citation_format_valid"] is True


def test_qa_agent_flags_missing_sections_and_bad_citation_format() -> None:
    agent = _MockResearchAgent(
        {
            "q1": (
                "Answer text without required section headers and citation format (Vaswani 2017)."
            )
        }
    )

    qa = ResearcherQAAgent(agent)
    report = qa.evaluate(["q1"])

    result = report["results"][0]
    assert result["checks"]["has_answer_section"] is False
    assert result["checks"]["has_references_section"] is False
    assert result["checks"]["has_inline_citations"] is True
    assert result["checks"]["citation_format_valid"] is False
    assert result["score"] == 0.25


def test_qa_agent_aggregates_multiple_queries() -> None:
    agent = _MockResearchAgent(
        {
            "q1": (
                "## Answer\n"
                "Text with valid citation (Smith et al., 2021).\n\n"
                "## References\n"
                "1. Smith, J., et al. (2021)."
            ),
            "q2": "## Answer\nNo inline citations here.\n\n## References\nNone.",
        }
    )

    qa = ResearcherQAAgent(agent)
    report = qa.evaluate(["q1", "q2"])

    assert agent.queries == ["q1", "q2"]
    assert report["total_queries"] == 2
    # q1: 4/4, q2: 2/4 -> overall 6/8
    assert report["overall_score"] == 0.75

    by_name = {item["query"]: item for item in report["results"]}
    assert by_name["q2"]["checks"]["has_inline_citations"] is False
    assert by_name["q2"]["checks"]["citation_format_valid"] is False

    summary = report["summary"]
    assert summary["has_answer_section"]["passed"] == 2
    assert summary["has_references_section"]["passed"] == 2
    assert summary["has_inline_citations"]["passed"] == 1
    assert summary["citation_format_valid"]["passed"] == 1
