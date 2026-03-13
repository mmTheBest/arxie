from __future__ import annotations

from ra.proposal import EvidenceMapper
from ra.retrieval.unified import Paper


def _paper(
    *,
    paper_id: str,
    title: str,
    abstract: str,
    doi: str | None = None,
    pdf_url: str | None = None,
    arxiv_id: str | None = None,
) -> Paper:
    return Paper(
        id=paper_id,
        title=title,
        abstract=abstract,
        authors=["Author Example"],
        year=2024,
        venue="Venue",
        citation_count=0,
        pdf_url=pdf_url,
        doi=doi,
        arxiv_id=arxiv_id,
        source="semantic_scholar",
    )


def test_evidence_mapper_buckets_mixed_evidence_and_builds_summary_payload() -> None:
    mapper = EvidenceMapper()
    claim = "Transformers improve machine translation quality across benchmarks."
    papers = [
        _paper(
            paper_id="s1",
            title="Positive MT Results",
            abstract="Transformers improve machine translation quality on standard benchmarks.",
        ),
        _paper(
            paper_id="c1",
            title="Negative MT Results",
            abstract=(
                "Our ablation indicates transformers did not improve translation quality and often"
                " performed worse than baseline systems."
            ),
        ),
        _paper(
            paper_id="a1",
            title="Transformer Inference Trade-offs",
            abstract=(
                "This paper discusses transformer latency, deployment constraints, and memory"
                " trade-offs for translation systems."
            ),
        ),
        _paper(
            paper_id="u1",
            title="Protein Folding",
            abstract="This paper studies protein folding dynamics and molecular simulations.",
        ),
    ]

    mapped = mapper.map_evidence(claim, papers)

    assert mapped.bucket_counts == {
        "supporting": 1,
        "contradicting": 1,
        "adjacent": 1,
    }
    assert [item.paper_id for item in mapped.supporting] == ["s1"]
    assert [item.paper_id for item in mapped.contradicting] == ["c1"]
    assert [item.paper_id for item in mapped.adjacent] == ["a1"]
    assert mapped.representative_paper_ids["supporting"] == ("s1",)
    assert mapped.representative_paper_ids["contradicting"] == ("c1",)
    assert mapped.representative_paper_ids["adjacent"] == ("a1",)
    assert "supporting" in mapped.landscape_summary.consensus
    assert "contradicting" in mapped.landscape_summary.controversy
    assert "adjacent" in mapped.landscape_summary.unknowns


def test_evidence_mapper_is_deterministic_for_identical_inputs() -> None:
    mapper = EvidenceMapper()
    claim = "Transformers improve machine translation quality."
    papers = [
        _paper(
            paper_id="s1",
            title="Positive MT Results",
            abstract="Transformers improve machine translation quality on multilingual tasks.",
        ),
        _paper(
            paper_id="c1",
            title="Negative MT Results",
            abstract="Transformers did not improve translation quality and performed worse.",
        ),
        _paper(
            paper_id="a1",
            title="Deployment Trade-offs",
            abstract="Transformer deployment trade-offs for translation production systems.",
        ),
    ]

    result_a = mapper.map_evidence(claim, papers, pinned_paper_ids={"a1"})
    result_b = mapper.map_evidence(claim, papers, pinned_paper_ids={"a1"})

    assert result_a == result_b


def test_evidence_mapper_keeps_pinned_reference_even_if_low_relevance() -> None:
    mapper = EvidenceMapper()
    claim = "Transformers improve machine translation quality."
    papers = [
        _paper(
            paper_id="pinned-1",
            title="Biology Methods",
            abstract="Protein folding simulations with molecular docking baselines.",
        ),
    ]

    mapped = mapper.map_evidence(claim, papers, pinned_paper_ids={"pinned-1"})

    assert mapped.bucket_counts["adjacent"] == 1
    assert [item.paper_id for item in mapped.adjacent] == ["pinned-1"]
    assert mapped.adjacent[0].pinned is True


def test_evidence_mapper_returns_scores_in_sorted_order_with_pinned_priority() -> None:
    mapper = EvidenceMapper()
    claim = "Transformers improve machine translation quality."
    papers = [
        _paper(
            paper_id="s-high",
            title="Strong Positive",
            abstract="Transformers improve machine translation quality on all benchmarks.",
        ),
        _paper(
            paper_id="s-low",
            title="Moderate Positive",
            abstract="Transformers improve translation.",
        ),
        _paper(
            paper_id="s-pin",
            title="Pinned Positive",
            abstract="Transformers improve translation quality in controlled experiments.",
        ),
    ]

    mapped = mapper.map_evidence(claim, papers, pinned_paper_ids={"s-pin"})

    supporting_ids = [item.paper_id for item in mapped.supporting]
    assert supporting_ids[0] == "s-pin"
    assert set(supporting_ids) == {"s-high", "s-low", "s-pin"}
    assert all(0.0 <= item.relevance_score <= 1.0 for item in mapped.supporting)


def test_evidence_mapper_sets_provenance_links_on_evidence_items() -> None:
    mapper = EvidenceMapper()
    claim = "Transformers improve machine translation quality."
    papers = [
        _paper(
            paper_id="doi-paper",
            title="DOI evidence",
            abstract="Transformers improve translation quality in benchmark settings.",
            doi="10.1000/xyz123",
        ),
        _paper(
            paper_id="pdf-paper",
            title="PDF evidence",
            abstract="Transformers improve translation under constrained settings.",
            pdf_url="https://example.org/paper.pdf",
        ),
        _paper(
            paper_id="arxiv-paper",
            title="ArXiv evidence",
            abstract="Transformers improve translation in multilingual tasks.",
            arxiv_id="1706.03762",
        ),
    ]

    mapped = mapper.map_evidence(
        claim,
        papers,
        pinned_paper_ids={"doi-paper", "pdf-paper", "arxiv-paper"},
    )
    items = tuple(mapped.supporting) + tuple(mapped.contradicting) + tuple(mapped.adjacent)
    links_by_id = {item.paper_id: item.provenance_link for item in items}

    assert links_by_id["doi-paper"] == "https://doi.org/10.1000/xyz123"
    assert links_by_id["pdf-paper"] == "https://example.org/paper.pdf"
    assert links_by_id["arxiv-paper"] == "https://arxiv.org/abs/1706.03762"
