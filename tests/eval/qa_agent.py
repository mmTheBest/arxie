from __future__ import annotations

import re
from typing import Any

_ANSWER_HEADER_RE = re.compile(r"^\s*##\s*answer\b", re.IGNORECASE | re.MULTILINE)
_REFERENCES_HEADER_RE = re.compile(r"^\s*##\s*references\b", re.IGNORECASE | re.MULTILINE)

# Valid inline citation format examples:
#   (Vaswani et al., 2017)
#   (Smith, 2021)
_VALID_INLINE_CITATION_RE = re.compile(
    r"^\([A-Z][A-Za-z'\-]+(?:\s+et\s+al\.)?,\s*(?:19|20)\d{2}\)$"
)
_PARENTHESES_RE = re.compile(r"\([^()]*\)")
_CITATION_CANDIDATE_RE = re.compile(r"(?:et\s+al\.|(?:19|20)\d{2})", re.IGNORECASE)


class ResearcherQAAgent:
    """Black-box QA checker for structured research-agent answers."""

    def __init__(self, research_agent: Any) -> None:
        self.research_agent = research_agent

    @staticmethod
    def _extract_answer_body(text: str) -> str:
        raw = (text or "").strip()
        match = _REFERENCES_HEADER_RE.search(raw)
        if match is None:
            return raw
        return raw[: match.start()].strip()

    @staticmethod
    def _extract_citation_candidates(answer_body: str) -> list[str]:
        candidates: list[str] = []
        for match in _PARENTHESES_RE.finditer(answer_body):
            token = match.group(0).strip()
            if _CITATION_CANDIDATE_RE.search(token):
                candidates.append(token)
        return candidates

    def _evaluate_output(self, output: str) -> dict[str, bool]:
        has_answer_section = _ANSWER_HEADER_RE.search(output) is not None
        has_references_section = _REFERENCES_HEADER_RE.search(output) is not None

        answer_body = self._extract_answer_body(output)
        candidates = self._extract_citation_candidates(answer_body)

        has_inline_citations = len(candidates) > 0
        citation_format_valid = has_inline_citations and all(
            _VALID_INLINE_CITATION_RE.fullmatch(candidate) is not None
            for candidate in candidates
        )

        return {
            "has_answer_section": has_answer_section,
            "has_references_section": has_references_section,
            "has_inline_citations": has_inline_citations,
            "citation_format_valid": citation_format_valid,
        }

    def evaluate(self, test_queries: list[str]) -> dict[str, Any]:
        check_names = [
            "has_answer_section",
            "has_references_section",
            "has_inline_citations",
            "citation_format_valid",
        ]

        results: list[dict[str, Any]] = []
        summary = {
            name: {"passed": 0, "failed": 0}
            for name in check_names
        }

        total_checks = len(test_queries) * len(check_names)
        passed_checks = 0

        for query in test_queries:
            output = str(self.research_agent.run(query))
            checks = self._evaluate_output(output)
            passed = sum(1 for ok in checks.values() if ok)
            score = round(passed / len(check_names), 4)

            for name, ok in checks.items():
                if ok:
                    summary[name]["passed"] += 1
                    passed_checks += 1
                else:
                    summary[name]["failed"] += 1

            results.append(
                {
                    "query": query,
                    "checks": checks,
                    "passed_checks": passed,
                    "total_checks": len(check_names),
                    "score": score,
                }
            )

        overall_score = round((passed_checks / total_checks) if total_checks else 0.0, 4)

        return {
            "total_queries": len(test_queries),
            "overall_score": overall_score,
            "results": results,
            "summary": summary,
        }
