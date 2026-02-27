import pytest

from ra.retrieval.arxiv import ArxivClient


ATOM_XML_WITH_DUPES = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<feed xmlns=\"http://www.w3.org/2005/Atom\" xmlns:arxiv=\"http://arxiv.org/schemas/atom\">
  <entry>
    <id>http://arxiv.org/abs/2101.00001v1</id>
    <updated>2021-01-02T00:00:00Z</updated>
    <published>2021-01-01T00:00:00Z</published>
    <title>  A   Title\n</title>
    <summary>Abstract\n with   spaces</summary>
    <author><name>Alice A.</name></author>
    <author><name>Bob B.</name></author>
    <category term=\"cs.AI\" />
    <link rel=\"related\" type=\"application/pdf\" title=\"pdf\" href=\"http://arxiv.org/pdf/2101.00001v1\" />
    <arxiv:doi> 10.1000/XYZ </arxiv:doi>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2101.00001v1</id>
    <updated>2021-01-02T00:00:00Z</updated>
    <published>2021-01-01T00:00:00Z</published>
    <title>A Title</title>
    <summary>Duplicate entry should be deduped</summary>
    <author><name>Alice A.</name></author>
    <category term=\"cs.AI\" />
  </entry>
</feed>
"""


def test_arxiv_parse_feed_normalizes_and_dedupes_by_arxiv_id():
    client = ArxivClient()
    papers = client._parse_feed(ATOM_XML_WITH_DUPES)

    assert len(papers) == 1
    p = papers[0]

    assert p.arxiv_id == "2101.00001v1"
    assert p.title == "A Title"
    assert p.abstract == "Abstract with spaces"
    assert p.authors == ["Alice A.", "Bob B."]
    assert p.categories == ["cs.AI"]
    assert p.doi == "10.1000/XYZ"
    # Prefer explicit PDF link when present
    assert p.pdf_url == "http://arxiv.org/pdf/2101.00001v1"


def test_arxiv_parse_entry_falls_back_to_canonical_pdf_url_when_missing_link():
    xml = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<feed xmlns=\"http://www.w3.org/2005/Atom\" xmlns:arxiv=\"http://arxiv.org/schemas/atom\">
  <entry>
    <id>http://arxiv.org/abs/9999.12345</id>
    <title>Test</title>
    <summary>Test</summary>
  </entry>
</feed>
"""
    client = ArxivClient()
    papers = client._parse_feed(xml)
    assert len(papers) == 1
    assert papers[0].pdf_url == "https://arxiv.org/pdf/9999.12345.pdf"
