"""Minimal CLI for manual retrieval checks.

Commands:
  - query "..." [--deep]
  - lit-review "topic" [--max-papers N]
  - trace "paper or concept" [--max-depth N] [--citations-per-paper N] [--max-papers N]
  - chat [--session-id ID] [--deep]
  - search --query "..." --limit 5 --source semantic|arxiv|both
  - get --id <doi|arxiv|semantic_scholar_id>
  - eval --dataset tests/eval/dataset.json --output results/

Outputs JSON to stdout.

This module intentionally uses argparse (no Typer/Rich) to keep it lightweight.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import sys
from dataclasses import asdict
from typing import Any

from ra.agents.lit_review_agent import LitReviewAgent
from ra.agents.research_agent import ResearchAgent
from ra.retrieval.semantic_scholar import SemanticScholarClient
from ra.retrieval.unified import Paper, UnifiedRetriever
from ra.tools.retrieval_tools import make_retrieval_tools
from ra.utils.logging_config import configure_logging_from_env
from ra.utils.security import sanitize_identifier, sanitize_user_text

logger = logging.getLogger(__name__)


def _paper_to_jsonable(p: Paper) -> dict[str, Any]:
    # dataclasses.asdict is sufficient (all fields are JSON-compatible).
    return asdict(p)


def _limit_arg(value: str) -> int:
    try:
        limit = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("limit must be an integer.") from None
    if limit < 1 or limit > 100:
        raise argparse.ArgumentTypeError("limit must be between 1 and 100.")
    return limit


def _max_papers_arg(value: str) -> int:
    try:
        max_papers = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("max-papers must be an integer.") from None
    if max_papers < 1 or max_papers > 50:
        raise argparse.ArgumentTypeError("max-papers must be between 1 and 50.")
    return max_papers


def _trace_depth_arg(value: str) -> int:
    try:
        depth = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("max-depth must be an integer.") from None
    if depth < 1 or depth > 8:
        raise argparse.ArgumentTypeError("max-depth must be between 1 and 8.")
    return depth


def _trace_citations_per_paper_arg(value: str) -> int:
    try:
        limit = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("citations-per-paper must be an integer.") from None
    if limit < 1 or limit > 100:
        raise argparse.ArgumentTypeError("citations-per-paper must be between 1 and 100.")
    return limit


def _trace_max_papers_arg(value: str) -> int:
    try:
        max_papers = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("max-papers must be an integer.") from None
    if max_papers < 1 or max_papers > 1000:
        raise argparse.ArgumentTypeError("max-papers must be between 1 and 1000.")
    return max_papers


def _callable_supports_kwarg(fn: Any, name: str) -> bool:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False

    params = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return True

    target = params.get(name)
    if target is None:
        return False

    return target.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ra", description="Academic Research Assistant CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_query = sub.add_parser("query", help="Run a research query through the agent pipeline")
    p_query.add_argument("query", help="Research question")
    p_query.add_argument(
        "--deep",
        action="store_true",
        help="Enable multi-step deep research mode (search -> full text -> citation chase)",
    )

    p_lit_review = sub.add_parser(
        "lit-review",
        help="Generate a structured literature review by thematic clusters",
    )
    p_lit_review.add_argument("topic", help="Literature review topic")
    p_lit_review.add_argument(
        "--max-papers",
        type=_max_papers_arg,
        default=20,
        help="Maximum number of papers to retrieve and synthesize (1-50).",
    )

    p_trace = sub.add_parser(
        "trace",
        help="Trace citation influence for a paper or concept and render timeline text.",
    )
    p_trace.add_argument("paper", help="Paper title, concept, or identifier to trace")
    p_trace.add_argument(
        "--max-depth",
        type=_trace_depth_arg,
        default=3,
        help="Citation hop depth to follow (1-8).",
    )
    p_trace.add_argument(
        "--citations-per-paper",
        type=_trace_citations_per_paper_arg,
        default=20,
        help="Maximum citing papers fetched per node (1-100).",
    )
    p_trace.add_argument(
        "--max-papers",
        type=_trace_max_papers_arg,
        default=200,
        help="Hard cap for timeline node expansion (1-1000).",
    )

    p_chat = sub.add_parser("chat", help="Interactive chat mode with conversation memory")
    p_chat.add_argument(
        "--session-id",
        default="default",
        help="Conversation session identifier for preserving context.",
    )
    p_chat.add_argument(
        "--deep",
        action="store_true",
        help="Enable deep research mode in chat responses.",
    )

    p_search = sub.add_parser("search", help="Search papers")
    p_search.add_argument("--query", required=True, help="Search query")
    p_search.add_argument("--limit", type=_limit_arg, default=5, help="Max results (1-100)")
    p_search.add_argument(
        "--source",
        choices=["semantic", "arxiv", "both"],
        default="both",
        help="Data source (default: both)",
    )

    p_get = sub.add_parser("get", help="Get a paper by id (DOI/arXiv/Semantic Scholar)")
    p_get.add_argument("--id", required=True, dest="identifier", help="Identifier to fetch")

    p_eval = sub.add_parser("eval", help="Run evaluation harness")
    p_eval.add_argument(
        "--dataset",
        required=True,
        help="Path to eval dataset JSON (e.g., tests/eval/dataset.json)",
    )
    p_eval.add_argument(
        "--output",
        required=True,
        help="Directory to write eval results artifacts (JSON + markdown)",
    )

    return parser


async def _cmd_search(query: str, limit: int, source: str) -> list[dict[str, Any]]:
    query = sanitize_user_text(query, field_name="query", max_length=1000)
    if source == "semantic":
        sources = ("semantic_scholar",)
    elif source == "arxiv":
        sources = ("arxiv",)
    else:
        sources = ("semantic_scholar", "arxiv")

    async with UnifiedRetriever() as r:
        papers = await r.search(query=query, limit=limit, sources=sources)
        return [_paper_to_jsonable(p) for p in papers]


async def _cmd_get(identifier: str) -> dict[str, Any] | None:
    identifier = sanitize_identifier(identifier, field_name="identifier", max_length=256)
    async with UnifiedRetriever() as r:
        p = await r.get_paper(identifier)
        return _paper_to_jsonable(p) if p else None


def _cmd_query(query: str, deep: bool) -> dict[str, str]:
    query = sanitize_user_text(query, field_name="query", max_length=4000)
    agent = ResearchAgent(deep_search=deep)
    answer = agent.run(query)
    return {"query": query, "answer": answer}


def _cmd_lit_review(topic: str, max_papers: int) -> dict[str, str]:
    topic = sanitize_user_text(topic, field_name="topic", max_length=1000)
    agent = LitReviewAgent()
    if _callable_supports_kwarg(agent.run, "max_papers"):
        review = agent.run(topic, max_papers=max_papers)
    else:
        review = agent.run(topic)
    return {"topic": topic, "review": review}


def _timeline_node_sort_key(node: dict[str, Any]) -> tuple[int, str]:
    year = node.get("year")
    if not isinstance(year, int):
        year = 9999
    title = str(node.get("paper_title") or "").lower()
    return (year, title)


def _format_trace_timeline(payload: dict[str, Any]) -> str:
    timeline_raw = payload.get("timeline")
    if not isinstance(timeline_raw, list) or not timeline_raw:
        return "No influence timeline available."

    timeline: list[dict[str, Any]] = [
        node for node in timeline_raw if isinstance(node, dict)
    ]
    if not timeline:
        return "No influence timeline available."

    by_id: dict[str, dict[str, Any]] = {}
    for node in timeline:
        paper_id = str(node.get("paper_id") or "").strip()
        if paper_id:
            by_id[paper_id] = node

    edge_lines: list[tuple[int, str]] = []
    for node in timeline:
        for link in node.get("citation_links", []) or []:
            if not isinstance(link, dict):
                continue
            from_id = str(link.get("from_paper_id") or "").strip()
            to_id = str(link.get("to_paper_id") or node.get("paper_id") or "").strip()
            src = by_id.get(from_id)
            dst = by_id.get(to_id, node)
            if src is None or dst is None:
                continue

            src_year = src.get("year")
            dst_year = dst.get("year")
            src_year_text = str(src_year) if isinstance(src_year, int) else "Unknown"
            dst_year_text = str(dst_year) if isinstance(dst_year, int) else "Unknown"
            src_title = str(src.get("paper_title") or "Untitled").strip()
            dst_title = str(dst.get("paper_title") or "Untitled").strip()
            sort_year = src_year if isinstance(src_year, int) else 9999
            edge_lines.append(
                (
                    sort_year,
                    f"{src_year_text} → {src_title} → cited by → {dst_year_text}: {dst_title}",
                )
            )

    if edge_lines:
        edge_lines.sort(key=lambda item: (item[0], item[1].lower()))
        return "\n".join(line for _, line in edge_lines)

    sorted_nodes = sorted(timeline, key=_timeline_node_sort_key)
    lines: list[str] = []
    for node in sorted_nodes:
        year = node.get("year")
        year_text = str(year) if isinstance(year, int) else "Unknown"
        title = str(node.get("paper_title") or "Untitled").strip()
        lines.append(f"{year_text} → {title}")
    return "\n".join(lines)


async def _cmd_trace(
    paper: str,
    max_depth: int,
    citations_per_paper: int,
    max_papers: int,
) -> dict[str, Any]:
    paper = sanitize_user_text(paper, field_name="paper", max_length=1000)
    semantic_scholar = SemanticScholarClient()
    try:
        async with UnifiedRetriever() as retriever:
            tools = make_retrieval_tools(
                retriever=retriever,
                semantic_scholar=semantic_scholar,
            )
            trace_tool = next((tool for tool in tools if tool.name == "trace_influence"), None)
            if trace_tool is None:
                raise RuntimeError("trace_influence tool is unavailable.")
            raw = await trace_tool.ainvoke(
                {
                    "paper": paper,
                    "max_depth": max_depth,
                    "citations_per_paper": citations_per_paper,
                    "max_papers": max_papers,
                }
            )
    finally:
        await semantic_scholar.close()

    parsed = raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"raw": raw}
    if not isinstance(parsed, dict):
        parsed = {"raw": parsed}

    return {
        "paper": paper,
        "timeline": _format_trace_timeline(parsed),
        "trace": parsed,
    }


def _cmd_eval(dataset: str, output: str) -> dict[str, Any]:
    dataset = sanitize_user_text(dataset, field_name="dataset", max_length=1024)
    output = sanitize_user_text(output, field_name="output", max_length=1024)
    from tests.eval.harness import EvalHarness

    harness = EvalHarness(dataset_path=dataset, output_dir=output)
    report = harness.run()
    return {
        "total_questions": report["total_questions"],
        "metrics": report["metrics"],
        "artifacts": report["artifacts"],
    }


def _cmd_chat(session_id: str, deep: bool) -> int:
    session_id = sanitize_user_text(session_id, field_name="session_id", max_length=128)
    agent = ResearchAgent(deep_search=deep)
    print(f"Chat session: {session_id}. Type /exit to quit.")
    while True:
        try:
            user_query = input("you> ").strip()
        except EOFError:
            break

        if not user_query:
            continue
        if user_query.lower() in {"/exit", "exit", "quit"}:
            break

        answer = agent.run(user_query, session_id=session_id)
        print(answer)

    return 0


def main(argv: list[str] | None = None) -> int:
    configure_logging_from_env()

    parser = build_parser()
    args = parser.parse_args(argv)
    logger.debug(
        "CLI command received",
        extra={
            "event": "cli.command.received",
            "command": args.command,
        },
    )

    try:
        if args.command == "query":
            payload = _cmd_query(args.query, args.deep)
        elif args.command == "lit-review":
            payload = _cmd_lit_review(args.topic, args.max_papers)
        elif args.command == "trace":
            payload = asyncio.run(
                _cmd_trace(
                    args.paper,
                    args.max_depth,
                    args.citations_per_paper,
                    args.max_papers,
                )
            )
        elif args.command == "chat":
            return _cmd_chat(args.session_id, args.deep)
        elif args.command == "search":
            payload = asyncio.run(_cmd_search(args.query, args.limit, args.source))
        elif args.command == "get":
            payload = asyncio.run(_cmd_get(args.identifier))
        elif args.command == "eval":
            payload = _cmd_eval(args.dataset, args.output)
        else:
            parser.error(f"Unknown command: {args.command}")
            return 2

        json.dump(payload, sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
        return 0

    except KeyboardInterrupt:
        logger.warning(
            "CLI interrupted by user",
            extra={"event": "cli.interrupted"},
        )
        return 130
    except Exception as e:  # noqa: BLE001
        logger.exception(
            "CLI command failed",
            extra={
                "event": "cli.command.failed",
                "command": args.command,
                "error_type": type(e).__name__,
            },
        )
        # Keep output machine-readable.
        json.dump({"error": str(e), "type": type(e).__name__}, sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
