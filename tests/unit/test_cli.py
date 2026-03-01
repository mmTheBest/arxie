from __future__ import annotations

import json

from ra.cli import _format_trace_timeline, build_parser, main


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


def test_trace_command_parses_paper_and_depth():
    parser = build_parser()

    args = parser.parse_args(["trace", "attention is all you need", "--max-depth", "4"])

    assert args.command == "trace"
    assert args.paper == "attention is all you need"
    assert args.max_depth == 4


def test_format_trace_timeline_renders_year_to_paper_chain():
    payload = {
        "timeline": [
            {
                "year": 2017,
                "paper_title": "Seed Paper",
                "paper_id": "p-seed",
                "citation_links": [],
            },
            {
                "year": 2019,
                "paper_title": "Follow-up Paper",
                "paper_id": "p-follow-up",
                "citation_links": [
                    {
                        "from_paper_id": "p-seed",
                        "to_paper_id": "p-follow-up",
                    }
                ],
            },
        ]
    }

    rendered = _format_trace_timeline(payload)

    assert "2017 → Seed Paper → cited by → 2019: Follow-up Paper" in rendered


def test_main_trace_command_routes_to_trace_handler(monkeypatch, capsys):
    seen: dict[str, object] = {}

    async def _stub_cmd_trace(
        paper: str,
        max_depth: int,
        citations_per_paper: int,
        max_papers: int,
    ) -> dict[str, object]:
        seen["paper"] = paper
        seen["max_depth"] = max_depth
        seen["citations_per_paper"] = citations_per_paper
        seen["max_papers"] = max_papers
        return {
            "paper": paper,
            "timeline": "2017 → Seed Paper → cited by → 2019: Follow-up Paper",
            "trace": {"count": 2},
        }

    monkeypatch.setattr("ra.cli.configure_logging_from_env", lambda: None)
    monkeypatch.setattr("ra.cli._cmd_trace", _stub_cmd_trace)

    exit_code = main(["trace", "seed paper", "--max-depth", "4", "--citations-per-paper", "8"])

    assert exit_code == 0
    assert seen == {
        "paper": "seed paper",
        "max_depth": 4,
        "citations_per_paper": 8,
        "max_papers": 200,
    }
    payload = json.loads(capsys.readouterr().out)
    assert payload["paper"] == "seed paper"
    assert payload["timeline"].startswith("2017 → Seed Paper")


def test_chat_command_parses_session_id():
    parser = build_parser()

    args = parser.parse_args(["chat", "--session-id", "session-a"])

    assert args.command == "chat"
    assert args.session_id == "session-a"


def test_main_chat_command_runs_interactive_loop_with_session_state(monkeypatch, capsys):
    seen: list[tuple[str, str | None]] = []

    class _StubAgent:
        def __init__(self, *, deep_search: bool = False) -> None:
            assert deep_search is False

        def run(self, query: str, *, session_id: str | None = None) -> str:
            seen.append((query, session_id))
            return f"## Answer\nEcho: {query}\n\n## References\nNone."

    inputs = iter(["What is RAG?", "How about limitations?", "/exit"])
    monkeypatch.setattr("ra.cli.configure_logging_from_env", lambda: None)
    monkeypatch.setattr("ra.cli.ResearchAgent", _StubAgent)
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    exit_code = main(["chat", "--session-id", "sess-1"])

    assert exit_code == 0
    assert seen == [
        ("What is RAG?", "sess-1"),
        ("How about limitations?", "sess-1"),
    ]
    out = capsys.readouterr().out
    assert "Echo: What is RAG?" in out
    assert "Echo: How about limitations?" in out
