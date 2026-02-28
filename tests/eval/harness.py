from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from ra.agents import ResearchAgent

_URL_RE = re.compile(r"https?://[^\s<>\]\)\"']+")
_REFERENCE_SECTION_RE = re.compile(r"\n\s*#{1,6}\s*references\s*\n", re.IGNORECASE)
_ANSWER_HEADER_RE = re.compile(r"^\s*#{1,6}\s*answer\s*\n?", re.IGNORECASE)

_ALLOWED_TIERS = {"tier_1", "tier_2", "tier_3"}
_TIER_NORMALIZATION = {
    "tier1": "tier_1",
    "tier2": "tier_2",
    "tier3": "tier_3",
}


@dataclass(frozen=True, slots=True)
class EvalQuestion:
    query: str
    expected_keywords: list[str]
    difficulty_tier: str
    min_citations: int
    max_citations: int


class EvalHarness:
    """Run a fixed evaluation dataset against ResearchAgent and aggregate metrics."""

    def __init__(
        self,
        *,
        dataset_path: str | Path,
        output_dir: str | Path,
        agent_factory: Callable[[], Any] | None = None,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self._agent_factory = agent_factory or (lambda: ResearchAgent())
        self._time_fn = time_fn
        if self._time_fn is None:
            from time import perf_counter

            self._time_fn = perf_counter

    def load_questions(self) -> list[EvalQuestion]:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")

        payload = json.loads(self.dataset_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("Dataset must be a JSON list of question objects.")

        questions: list[EvalQuestion] = []
        required_fields = (
            "query",
            "expected_keywords",
            "difficulty_tier",
            "min_citations",
            "max_citations",
        )

        for idx, row in enumerate(payload, start=1):
            if not isinstance(row, dict):
                raise ValueError(f"Question #{idx} must be a JSON object.")

            for field in required_fields:
                if field not in row:
                    raise ValueError(f"Question #{idx} is missing required field: {field}")

            query = str(row["query"]).strip()
            if not query:
                raise ValueError(f"Question #{idx} has an empty query.")

            raw_keywords = row["expected_keywords"]
            if not isinstance(raw_keywords, list) or not all(
                isinstance(k, str) for k in raw_keywords
            ):
                raise ValueError(
                    f"Question #{idx} field expected_keywords must be a list[str]."
                )
            expected_keywords = [k.strip() for k in raw_keywords if k.strip()]
            if not expected_keywords:
                raise ValueError(f"Question #{idx} expected_keywords cannot be empty.")

            tier = str(row["difficulty_tier"]).strip().lower()
            tier = _TIER_NORMALIZATION.get(tier, tier)
            if tier not in _ALLOWED_TIERS:
                raise ValueError(
                    f"Question #{idx} has invalid difficulty_tier={row['difficulty_tier']!r}."
                )

            min_citations = int(row["min_citations"])
            max_citations = int(row["max_citations"])
            if min_citations < 0 or max_citations < 0:
                raise ValueError(f"Question #{idx} citation bounds must be non-negative.")
            if min_citations > max_citations:
                raise ValueError(
                    f"Question #{idx} has min_citations > max_citations."
                )

            questions.append(
                EvalQuestion(
                    query=query,
                    expected_keywords=expected_keywords,
                    difficulty_tier=tier,
                    min_citations=min_citations,
                    max_citations=max_citations,
                )
            )

        return questions

    def run(self) -> dict[str, Any]:
        questions = self.load_questions()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        agent = self._agent_factory()
        usage_path = self._resolve_usage_path(agent)
        usage_count = len(self._read_usage_entries(usage_path)) if usage_path else 0

        rows: list[dict[str, Any]] = []
        citation_precision_scores: list[float] = []
        latencies: list[float] = []
        total_claims = 0
        total_supported_claims = 0
        total_tool_calls = 0
        total_tool_successes = 0

        for q in questions:
            start = float(self._time_fn())
            try:
                answer = str(agent.run(q.query))
                error = None
            except Exception as exc:  # noqa: BLE001
                answer = ""
                error = f"{type(exc).__name__}: {exc}"
            elapsed = max(0.0, float(self._time_fn()) - start)
            latencies.append(elapsed)

            answer_body, references = self._split_answer_and_references(answer)
            citation_urls = self._extract_citation_urls(answer_body=answer_body, references=references)
            inline_citation_count = len(citation_urls)
            if inline_citation_count == 0:
                matched_citation_count = 0
                citation_precision = 0.0
            else:
                matched_citation_count = sum(
                    1 for citation_url in citation_urls if self._is_valid_url(citation_url)
                )
                citation_precision = matched_citation_count / inline_citation_count
            citation_precision_scores.append(citation_precision)

            claim_count = 1
            supported_claim_count = 1 if self._is_substantive_content(answer_body) else 0
            total_claims += claim_count
            total_supported_claims += supported_claim_count

            if usage_path:
                usage_entries = self._read_usage_entries(usage_path)
                new_entries = usage_entries[usage_count:]
                usage_count = len(usage_entries)
            else:
                new_entries = []

            tool_calls = [
                entry
                for entry in new_entries
                if str(entry.get("endpoint", "")).startswith("tool:")
            ]
            tool_successes = [
                entry for entry in tool_calls if self._is_successful_tool_response(entry)
            ]
            total_tool_calls += len(tool_calls)
            total_tool_successes += len(tool_successes)

            lower_answer = answer.lower()
            matched_keywords = sum(
                1 for kw in q.expected_keywords if kw.lower() in lower_answer
            )
            keyword_coverage = (
                matched_keywords / len(q.expected_keywords) if q.expected_keywords else 1.0
            )
            citation_count_ok = q.min_citations <= inline_citation_count <= q.max_citations

            rows.append(
                {
                    "query": q.query,
                    "difficulty_tier": q.difficulty_tier,
                    "latency_seconds": round(elapsed, 6),
                    "inline_citations": inline_citation_count,
                    "matched_citations": matched_citation_count,
                    "citation_precision": round(citation_precision, 6),
                    "claim_count": claim_count,
                    "supported_claim_count": supported_claim_count,
                    "claim_support_ratio": round(
                        (supported_claim_count / claim_count) if claim_count else 0.0,
                        6,
                    ),
                    "keyword_coverage": round(keyword_coverage, 6),
                    "citation_count_ok": citation_count_ok,
                    "tool_calls": len(tool_calls),
                    "tool_successes": len(tool_successes),
                    "error": error,
                }
            )

        metrics = {
            "citation_precision": round(
                (sum(citation_precision_scores) / len(citation_precision_scores))
                if citation_precision_scores
                else 0.0,
                6,
            ),
            "claim_support_ratio": round(
                (total_supported_claims / total_claims) if total_claims else 0.0,
                6,
            ),
            "tool_success_rate": round(
                (total_tool_successes / total_tool_calls) if total_tool_calls else 0.0,
                6,
            ),
            "latency_p95": round(self._p95(latencies), 6),
        }

        by_tier = self._build_tier_breakdown(rows)

        report: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_path": str(self.dataset_path.resolve()),
            "output_dir": str(self.output_dir.resolve()),
            "total_questions": len(questions),
            "metrics": metrics,
            "by_tier": by_tier,
            "results": rows,
            "artifacts": {},
        }

        json_path = self.output_dir / "eval_results.json"
        md_path = self.output_dir / "eval_summary.md"
        report["artifacts"] = {
            "json": str(json_path.resolve()),
            "markdown": str(md_path.resolve()),
        }

        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(self._build_markdown_summary(report), encoding="utf-8")
        return report

    def _resolve_usage_path(self, agent: Any) -> Path | None:
        usage_logger = getattr(agent, "usage_logger", None)
        path = getattr(usage_logger, "path", None) if usage_logger is not None else None
        if path is None:
            return None
        return Path(path)

    @staticmethod
    def _read_usage_entries(path: Path | None) -> list[dict[str, Any]]:
        if path is None or not path.exists():
            return []

        entries: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                entries.append(obj)
        return entries

    @staticmethod
    def _split_answer_and_references(answer: str) -> tuple[str, str]:
        raw = (answer or "").strip()
        raw = _ANSWER_HEADER_RE.sub("", raw, count=1).strip()
        split = _REFERENCE_SECTION_RE.split(raw, maxsplit=1)
        if len(split) == 2:
            answer_body, references = split[0].strip(), split[1].strip()
        else:
            answer_body, references = raw, ""
        return answer_body, references

    @staticmethod
    def _extract_citation_urls(*, answer_body: str, references: str) -> list[str]:
        source = references if references.strip() else answer_body
        citations: list[str] = []
        for match in _URL_RE.finditer(source):
            citations.append(match.group(0).rstrip(".,;:"))
        return citations

    @staticmethod
    def _is_valid_url(url: str) -> bool:
        try:
            parsed = urlparse(url.strip())
        except Exception:
            return False
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    @staticmethod
    def _is_substantive_content(answer_body: str) -> bool:
        normalized = " ".join(answer_body.split())
        return bool(normalized) and len(normalized) > 50

    @staticmethod
    def _is_successful_tool_response(entry: dict[str, Any]) -> bool:
        error_value = entry.get("error")
        if isinstance(error_value, str):
            if error_value.strip():
                return False
        elif error_value:
            return False

        for payload_key in ("response", "result", "output"):
            payload = entry.get(payload_key)
            if isinstance(payload, dict) and payload.get("error"):
                return False
            if isinstance(payload, str):
                stripped = payload.strip()
                if stripped.startswith("{") and stripped.endswith("}"):
                    try:
                        parsed = json.loads(stripped)
                    except json.JSONDecodeError:
                        parsed = None
                    if isinstance(parsed, dict) and parsed.get("error"):
                        return False

        status = entry.get("status")
        if status is None:
            return True
        try:
            return int(status) < 400
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _p95(latencies: list[float]) -> float:
        if not latencies:
            return 0.0
        ordered = sorted(latencies)
        rank = max(1, math.ceil(len(ordered) * 0.95))
        return float(ordered[rank - 1])

    @staticmethod
    def _build_tier_breakdown(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            tier = str(row.get("difficulty_tier", "unknown"))
            grouped.setdefault(tier, []).append(row)

        out: dict[str, dict[str, Any]] = {}
        for tier, items in sorted(grouped.items()):
            count = len(items)
            out[tier] = {
                "questions": count,
                "citation_precision": round(
                    sum(float(item["citation_precision"]) for item in items) / count,
                    6,
                ),
                "claim_support_ratio": round(
                    sum(float(item["claim_support_ratio"]) for item in items) / count,
                    6,
                ),
                "tool_success_rate": round(
                    (
                        sum(int(item["tool_successes"]) for item in items)
                        / sum(int(item["tool_calls"]) for item in items)
                    )
                    if sum(int(item["tool_calls"]) for item in items)
                    else 0.0,
                    6,
                ),
                "latency_p95": round(
                    EvalHarness._p95([float(item["latency_seconds"]) for item in items]),
                    6,
                ),
            }
        return out

    @staticmethod
    def _build_markdown_summary(report: dict[str, Any]) -> str:
        metrics = report["metrics"]
        lines = [
            "# Evaluation Summary",
            "",
            f"- Generated at (UTC): `{report['generated_at']}`",
            f"- Dataset: `{report['dataset_path']}`",
            f"- Total questions: `{report['total_questions']}`",
            "",
            "## Topline Metrics",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| citation_precision | {metrics['citation_precision']:.6f} |",
            f"| claim_support_ratio | {metrics['claim_support_ratio']:.6f} |",
            f"| tool_success_rate | {metrics['tool_success_rate']:.6f} |",
            f"| latency_p95 (seconds) | {metrics['latency_p95']:.6f} |",
            "",
            "## Tier Breakdown",
            "",
            "| Tier | Questions | Citation Precision | Claim Support | Tool Success | Latency p95 |",
            "|---|---:|---:|---:|---:|---:|",
        ]

        for tier, tier_metrics in report["by_tier"].items():
            lines.append(
                "| {tier} | {questions} | {citation_precision:.6f} | "
                "{claim_support_ratio:.6f} | {tool_success_rate:.6f} | {latency_p95:.6f} |".format(
                    tier=tier,
                    questions=tier_metrics["questions"],
                    citation_precision=tier_metrics["citation_precision"],
                    claim_support_ratio=tier_metrics["claim_support_ratio"],
                    tool_success_rate=tier_metrics["tool_success_rate"],
                    latency_p95=tier_metrics["latency_p95"],
                )
            )

        return "\n".join(lines) + "\n"
