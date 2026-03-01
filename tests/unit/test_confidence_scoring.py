from __future__ import annotations

from ra.citation.confidence import (
    ClaimConfidence,
    ClaimConfidenceScorer,
    annotate_claim_with_confidence,
)
from ra.retrieval.unified import Paper


def _paper(*, paper_id: str, abstract: str) -> Paper:
    return Paper(
        id=paper_id,
        title=f"Paper {paper_id}",
        abstract=abstract,
        authors=["Author Example"],
        year=2024,
        venue="Venue",
        citation_count=0,
        pdf_url=None,
        doi=None,
        arxiv_id=None,
        source="semantic_scholar",
    )


def test_confidence_scoring_counts_supporting_and_marks_high():
    scorer = ClaimConfidenceScorer()
    claim = "Transformers improve machine translation quality across benchmarks."
    papers = [
        _paper(
            paper_id="s1",
            abstract="Transformers improve machine translation quality on multiple benchmarks.",
        ),
        _paper(
            paper_id="s2",
            abstract="Our results show improved translation quality using transformer models.",
        ),
        _paper(
            paper_id="s3",
            abstract="Benchmark experiments report a strong improvement from transformers.",
        ),
        _paper(
            paper_id="s4",
            abstract="Translation quality improves significantly with transformer architectures.",
        ),
    ]

    scored = scorer.score(claim, papers)

    assert scored.supporting_count == 4
    assert scored.contradicting_count == 0
    assert scored.label == "HIGH"


def test_confidence_scoring_counts_contradictions_and_degrades_confidence():
    scorer = ClaimConfidenceScorer()
    claim = "Transformers improve machine translation quality."
    papers = [
        _paper(
            paper_id="s1",
            abstract="Transformers improve translation quality in multilingual tasks.",
        ),
        _paper(
            paper_id="c1",
            abstract=(
                "Our ablation indicates transformers did not improve translation quality and often"
                " performed worse than baselines."
            ),
        ),
        _paper(
            paper_id="u1",
            abstract="This paper studies protein folding dynamics without NLP tasks.",
        ),
    ]

    scored = scorer.score(claim, papers)

    assert scored.supporting_count == 1
    assert scored.contradicting_count == 1
    assert scored.label in {"MEDIUM", "LOW"}


def test_confidence_annotation_format_matches_spec():
    confidence = ClaimConfidence(
        label="HIGH",
        supporting_count=8,
        contradicting_count=1,
        score=0.9,
    )

    annotated = annotate_claim_with_confidence("Claim text.", confidence)

    assert (
        annotated
        == "Claim text. [Confidence: HIGH — 8 supporting, 1 contradicting]"
    )
