from __future__ import annotations

import json

from ra.cli import build_parser, main


def test_query_command_parses_deep_flag():
    parser = build_parser()

    args = parser.parse_args(["query", "What is retrieval-augmented generation?", "--deep"])

    assert args.command == "query"
    assert args.query == "What is retrieval-augmented generation?"
    assert args.deep is True


def test_query_command_defaults_deep_flag_to_false():
    parser = build_parser()

    args = parser.parse_args(["query", "What is retrieval-augmented generation?"])

    assert args.command == "query"
    assert args.deep is False


def test_main_query_command_passes_deep_flag_to_agent(monkeypatch, capsys):
    seen: dict[str, object] = {}

    class _StubAgent:
        def __init__(self, *, deep_search: bool = False) -> None:
            seen["deep_search"] = deep_search

        def run(self, query: str) -> str:
            seen["query"] = query
            return "## Answer\nok\n\n## References\nNone."

    monkeypatch.setattr("ra.cli.configure_logging_from_env", lambda: None)
    monkeypatch.setattr("ra.cli.ResearchAgent", _StubAgent)

    exit_code = main(["query", "What is RAG?", "--deep"])

    assert exit_code == 0
    assert seen["deep_search"] is True
    assert seen["query"] == "What is RAG?"
    payload = json.loads(capsys.readouterr().out)
    assert payload["query"] == "What is RAG?"
    assert payload["answer"] == "## Answer\nok\n\n## References\nNone."


def test_lit_review_command_parses_topic():
    parser = build_parser()

    args = parser.parse_args(["lit-review", "graph neural networks"])

    assert args.command == "lit-review"
    assert args.topic == "graph neural networks"


def test_main_lit_review_command_uses_lit_review_agent(monkeypatch, capsys):
    seen: dict[str, object] = {}

    expected_review = (
        "## Introduction\nIntro.\n\n"
        "## Thematic Groups\n- Theme A\n\n"
        "## Key Findings\n- Finding\n\n"
        "## Research Gaps\n- Gap\n\n"
        "## Future Directions\n- Next steps"
    )

    class _StubLitReviewAgent:
        def run(self, topic: str) -> str:
            seen["topic"] = topic
            return expected_review

    monkeypatch.setattr("ra.cli.configure_logging_from_env", lambda: None)
    monkeypatch.setattr("ra.cli.LitReviewAgent", _StubLitReviewAgent)

    exit_code = main(["lit-review", "graph neural networks"])

    assert exit_code == 0
    assert seen["topic"] == "graph neural networks"
    payload = json.loads(capsys.readouterr().out)
    assert payload["topic"] == "graph neural networks"
    assert payload["review"] == expected_review
