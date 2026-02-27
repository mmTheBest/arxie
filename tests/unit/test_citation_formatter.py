from __future__ import annotations

from ra.citation.formatter import CitationFormatter
from ra.retrieval.unified import Paper


def _paper(
    *,
    id: str = "p1",
    title: str = "A Great Paper",
    authors: list[str] | None = None,
    year: int | None = 2024,
    venue: str | None = "NeurIPS",
    doi: str | None = "10.1000/xyz",
    arxiv_id: str | None = None,
    pdf_url: str | None = None,
) -> Paper:
    return Paper(
        id=id,
        title=title,
        abstract=None,
        authors=authors or [],
        year=year,
        venue=venue,
        citation_count=None,
        pdf_url=pdf_url,
        doi=doi,
        arxiv_id=arxiv_id,
        source="semantic_scholar",
    )


def test_format_inline_three_plus_authors_uses_et_al():
    f = CitationFormatter()
    p = _paper(authors=["John Smith", "Jane Doe", "Ada Lovelace"], year=2024, doi=None)
    assert f.format_inline(p) == "(Smith et al., 2024)"


def test_format_inline_missing_year_uses_nd():
    f = CitationFormatter()
    p = _paper(authors=["John Smith"], year=None, doi=None)
    assert f.format_inline(p) == "(Smith, n.d.)"


def test_format_apa_basic_includes_doi_and_venue():
    f = CitationFormatter()
    p = _paper(
        authors=["John Smith", "Jane Doe"],
        year=2024,
        title="Attention Is All You Need",
        venue="NeurIPS",
        doi="doi:10.1000/XYZ",
    )
    s = f.format_apa(p)
    assert "Smith, J." in s
    assert "Doe, J." in s
    assert "(2024)." in s
    assert "Attention Is All You Need." in s
    assert "NeurIPS." in s
    assert "https://doi.org/10.1000/xyz" in s


def test_format_reference_list_sorts_by_first_author_last_name():
    f = CitationFormatter()
    p_b = _paper(id="b", authors=["Bob Zebra"], title="B Title", doi=None)
    p_a = _paper(id="a", authors=["Alice Alpha"], title="A Title", doi=None)
    out = f.format_reference_list([p_b, p_a]).splitlines()
    assert out[0].startswith("Alpha")
    assert out[1].startswith("Zebra")


def test_format_apa_missing_authors_and_year():
    f = CitationFormatter()
    p = _paper(authors=[], year=None, title="", venue=None, doi=None)
    s = f.format_apa(p)
    assert s.startswith("Unknown")
    assert "(n.d.)." in s


def test_extract_claims_maps_sentences_to_papers_by_author_year():
    f = CitationFormatter()
    p1 = _paper(id="p1", authors=["John Smith", "Jane Doe"], year=2024, doi=None)
    p2 = _paper(id="p2", authors=["Ada Lovelace"], year=2020, doi=None)

    text = (
        "We build on prior work (Smith et al., 2024). "
        "A classic result appears in Lovelace (2020). "
        "This sentence has no citation."
    )

    claims = f.extract_claims(text, [p1, p2])
    assert len(claims) == 2
    assert claims[0].supporting_papers == ["p1"]
    assert claims[1].supporting_papers == ["p2"]
    assert claims[0].confidence >= 0.75
