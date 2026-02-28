"""Minimal CLI for manual retrieval checks.

Commands:
  - search --query "..." --limit 5 --source semantic|arxiv|both
  - get --id <doi|arxiv|semantic_scholar_id>
  - eval --dataset tests/eval/dataset.json --output results/

Outputs JSON to stdout.

This module intentionally uses argparse (no Typer/Rich) to keep it lightweight.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import asdict
from typing import Any

from ra.retrieval.unified import Paper, UnifiedRetriever
from ra.utils.security import sanitize_identifier, sanitize_user_text
from ra.utils.logging_config import configure_logging_from_env

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ra", description="Academic Research Assistant CLI")
    sub = parser.add_subparsers(dest="command", required=True)

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
        if args.command == "search":
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
