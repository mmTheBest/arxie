from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Literal

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
class ClaimConfidence:
    label: Literal["HIGH", "MEDIUM", "LOW"]
    supporting_count: int
    contradicting_count: int
    score: float

    def annotation(self) -> str:
        return (
            f"[Confidence: {self.label} — {self.supporting_count} supporting, "
            f"{self.contradicting_count} contradicting]"
        )


class ClaimConfidenceScorer:
    """Heuristic claim confidence scorer using semantic overlap and polarity cues."""

    def __init__(
        self,
        *,
        min_similarity: float = 0.18,
        support_similarity: float = 0.32,
        contradiction_similarity: float = 0.22,
    ) -> None:
        self.min_similarity = float(min_similarity)
        self.support_similarity = float(support_similarity)
        self.contradiction_similarity = float(contradiction_similarity)

    def score(self, claim: str, papers: list[Paper]) -> ClaimConfidence:
        claim_text = str(claim or "").strip()
        if not claim_text or not papers:
            return ClaimConfidence(
                label="LOW",
                supporting_count=0,
                contradicting_count=0,
                score=0.0,
            )

        claim_tokens = self._tokenize(claim_text)
        claim_vector = Counter(claim_tokens)
        claim_polarity = self._polarity(claim_tokens)

        supporting = 0
        contradicting = 0

        for paper in papers:
            abstract = str(getattr(paper, "abstract", "") or "").strip()
            if not abstract:
                continue

            abstract_tokens = self._tokenize(abstract)
            if not abstract_tokens:
                continue
            abstract_vector = Counter(abstract_tokens)
            similarity = self._cosine_similarity(claim_vector, abstract_vector)
            shared_tokens = len(set(claim_tokens) & set(abstract_tokens))
            if similarity < self.min_similarity and shared_tokens < 2:
                continue

            abstract_polarity = self._polarity(abstract_tokens)
            is_opposed = (
                claim_polarity != 0
                and abstract_polarity != 0
                and claim_polarity != abstract_polarity
            )

            if is_opposed and similarity >= self.contradiction_similarity:
                contradicting += 1
            elif similarity >= self.support_similarity or shared_tokens >= 2:
                supporting += 1

        total = supporting + contradicting
        if total == 0:
            return ClaimConfidence(
                label="LOW",
                supporting_count=0,
                contradicting_count=0,
                score=0.0,
            )

        score = (supporting - contradicting) / total
        label = self._label_for_counts(
            supporting=supporting,
            contradicting=contradicting,
            score=score,
        )
        return ClaimConfidence(
            label=label,
            supporting_count=supporting,
            contradicting_count=contradicting,
            score=score,
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        tokens = [ClaimConfidenceScorer._normalize_token(m.group(0)) for m in _TOKEN_RE.finditer(text.lower())]
        return [tok for tok in tokens if tok not in _STOPWORDS]

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
    def _cosine_similarity(a: Counter[str], b: Counter[str]) -> float:
        if not a or not b:
            return 0.0
        overlap = set(a.keys()) & set(b.keys())
        dot = sum(a[token] * b[token] for token in overlap)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

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
    def _label_for_counts(*, supporting: int, contradicting: int, score: float) -> str:
        if supporting >= 3 and contradicting <= 1 and score >= 0.5:
            return "HIGH"
        if supporting >= 1 and score > 0:
            return "MEDIUM"
        return "LOW"


def annotate_claim_with_confidence(claim_text: str, confidence: ClaimConfidence) -> str:
    claim = str(claim_text or "").strip()
    if not claim:
        return confidence.annotation()
    return f"{claim} {confidence.annotation()}"
