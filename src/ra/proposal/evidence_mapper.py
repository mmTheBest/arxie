"""Deterministic evidence bucketing for proposal-stage evidence mapping."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from ra.retrieval.unified import Paper

_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
}
_POSITIVE_MARKERS = {
    "better",
    "boost",
    "effective",
    "improve",
    "improved",
    "improves",
    "improving",
    "increase",
    "increased",
    "outperform",
    "outperforms",
    "reduce",
    "reduced",
    "reduces",
    "strong",
    "supports",
}
_NEGATIVE_MARKERS = {
    "decline",
    "decrease",
    "decreased",
    "didnt",
    "doesnt",
    "fail",
    "failed",
    "fails",
    "ineffective",
    "insufficient",
    "never",
    "no",
    "not",
    "worse",
}


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    paper_id: str
    title: str
    relevance_score: float
    pinned: bool = False


@dataclass(frozen=True, slots=True)
class LandscapeSummary:
    consensus: str
    controversy: str
    unknowns: str


@dataclass(frozen=True, slots=True)
class EvidenceMappingResult:
    supporting: tuple[EvidenceItem, ...]
    contradicting: tuple[EvidenceItem, ...]
    adjacent: tuple[EvidenceItem, ...]
    bucket_counts: dict[str, int]
    representative_paper_ids: dict[str, tuple[str, ...]]
    landscape_summary: LandscapeSummary


class EvidenceMapper:
    """Map papers into supporting/contradicting/adjacent buckets with summaries."""

    def __init__(
        self,
        *,
        min_relevance: float = 0.18,
        support_similarity: float = 0.45,
        contradiction_similarity: float = 0.22,
        min_shared_tokens: int = 1,
        representative_limit: int = 3,
    ) -> None:
        self.min_relevance = float(min_relevance)
        self.support_similarity = float(support_similarity)
        self.contradiction_similarity = float(contradiction_similarity)
        self.min_shared_tokens = max(1, int(min_shared_tokens))
        self.representative_limit = max(1, int(representative_limit))

    def map_evidence(
        self,
        claim: str,
        papers: list[Paper],
        *,
        pinned_paper_ids: set[str] | None = None,
    ) -> EvidenceMappingResult:
        pinned_ids = {
            str(paper_id).strip()
            for paper_id in (pinned_paper_ids or set())
            if str(paper_id).strip()
        }
        claim_tokens = self._tokenize(str(claim or ""))
        claim_vector = Counter(claim_tokens)
        claim_polarity = self._polarity(claim_tokens)

        buckets: dict[str, list[EvidenceItem]] = {
            "supporting": [],
            "contradicting": [],
            "adjacent": [],
        }

        for paper in papers:
            paper_id = str(getattr(paper, "id", "") or "").strip()
            if not paper_id:
                continue
            title = str(getattr(paper, "title", "") or "").strip()
            abstract = str(getattr(paper, "abstract", "") or "").strip()
            text = f"{title} {abstract}".strip()
            tokens = self._tokenize(text)
            paper_vector = Counter(tokens)
            similarity = self._cosine_similarity(claim_vector, paper_vector)
            shared_tokens = len(set(claim_tokens) & set(tokens))
            pinned = paper_id in pinned_ids
            is_relevant = (
                similarity >= self.min_relevance
                or shared_tokens >= self.min_shared_tokens
            )

            if not is_relevant and not pinned:
                continue

            paper_polarity = self._polarity(tokens)
            is_opposed = (
                claim_polarity != 0
                and paper_polarity != 0
                and claim_polarity != paper_polarity
                and (
                    similarity >= self.contradiction_similarity
                    or shared_tokens >= self.min_shared_tokens
                )
            )
            if is_opposed:
                bucket_name = "contradicting"
            elif similarity >= self.support_similarity and not is_opposed:
                bucket_name = "supporting"
            else:
                bucket_name = "adjacent"

            buckets[bucket_name].append(
                EvidenceItem(
                    paper_id=paper_id,
                    title=title,
                    relevance_score=round(max(0.0, min(1.0, similarity)), 6),
                    pinned=pinned,
                )
            )

        supporting = self._sorted_bucket(tuple(buckets["supporting"]))
        contradicting = self._sorted_bucket(tuple(buckets["contradicting"]))
        adjacent = self._sorted_bucket(tuple(buckets["adjacent"]))

        bucket_counts = {
            "supporting": len(supporting),
            "contradicting": len(contradicting),
            "adjacent": len(adjacent),
        }
        representative_paper_ids = {
            "supporting": tuple(item.paper_id for item in supporting[: self.representative_limit]),
            "contradicting": tuple(
                item.paper_id for item in contradicting[: self.representative_limit]
            ),
            "adjacent": tuple(item.paper_id for item in adjacent[: self.representative_limit]),
        }
        landscape_summary = LandscapeSummary(
            consensus=self._summary_line("supporting", supporting),
            controversy=self._summary_line("contradicting", contradicting),
            unknowns=self._summary_line("adjacent", adjacent),
        )

        return EvidenceMappingResult(
            supporting=supporting,
            contradicting=contradicting,
            adjacent=adjacent,
            bucket_counts=bucket_counts,
            representative_paper_ids=representative_paper_ids,
            landscape_summary=landscape_summary,
        )

    @staticmethod
    def _sorted_bucket(bucket: tuple[EvidenceItem, ...]) -> tuple[EvidenceItem, ...]:
        return tuple(
            sorted(
                bucket,
                key=lambda item: (-int(item.pinned), -item.relevance_score, item.paper_id),
            )
        )

    @staticmethod
    def _summary_line(bucket_name: str, bucket: tuple[EvidenceItem, ...]) -> str:
        count = len(bucket)
        if count == 0:
            return f"No {bucket_name} evidence identified."
        ids = ", ".join(item.paper_id for item in bucket[:3])
        return f"{count} {bucket_name} evidence paper(s): {ids}."

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        raw_tokens = _TOKEN_RE.findall(text.lower())
        normalized = [EvidenceMapper._normalize_token(token) for token in raw_tokens]
        return [token for token in normalized if token and token not in _STOPWORDS]

    @staticmethod
    def _normalize_token(token: str) -> str:
        tok = str(token or "").lower()
        if len(tok) > 5 and tok.endswith("ies"):
            tok = tok[:-3] + "y"
        for suffix in ("ing", "ed", "es", "s"):
            if len(tok) > 4 and tok.endswith(suffix):
                tok = tok[: -len(suffix)]
                break
        return tok

    @staticmethod
    def _polarity(tokens: list[str]) -> int:
        positive = sum(1 for token in tokens if token in _POSITIVE_MARKERS)
        negative = sum(1 for token in tokens if token in _NEGATIVE_MARKERS)
        if positive > negative:
            return 1
        if negative > positive:
            return -1
        return 0

    @staticmethod
    def _cosine_similarity(a: Counter[str], b: Counter[str]) -> float:
        if not a or not b:
            return 0.0
        overlap = set(a.keys()) & set(b.keys())
        dot = sum(a[token] * b[token] for token in overlap)
        norm_a = math.sqrt(sum(value * value for value in a.values()))
        norm_b = math.sqrt(sum(value * value for value in b.values()))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)
