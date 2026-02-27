from __future__ import annotations

from dataclasses import asdict

import pytest

from ra.parsing.pdf_parser import PDFParser, ParsedDocument


def test_parsed_document_creation() -> None:
    doc = ParsedDocument(text="hello", pages=["p1"], metadata={"title": "x"})
    assert doc.text == "hello"
    assert doc.pages == ["p1"]
    assert doc.metadata["title"] == "x"

    # dataclass should serialize predictably
    d = asdict(doc)
    assert d["text"] == "hello"
    assert d["pages"] == ["p1"]
    assert d["metadata"]["title"] == "x"


def test_section_extraction_with_mock_text() -> None:
    parser = PDFParser()

    pages = [
        "A Great Paper\n\nAbstract\nThis is the abstract.\n\nIntroduction\nIntro text here.",
        "Methods\nWe did things.\n\nReferences\n[1] Ref",
    ]
    doc = ParsedDocument(text="\n\n".join(pages), pages=pages, metadata={})

    sections = parser.extract_sections(doc)
    titles = [s.title for s in sections]

    # Includes inferred title section + known headings
    assert titles[0] == "Title"
    assert "Abstract" in titles
    assert "Introduction" in titles
    assert "Methods" in titles
    assert "References" in titles

    # References should be last (we stop after it)
    assert titles[-1] == "References"

    # Page mapping is best-effort; ensure Methods is on page 1 (second page)
    methods = next(s for s in sections if s.title == "Methods")
    assert methods.page_start == 1


def test_extract_sections_empty_document() -> None:
    parser = PDFParser()
    doc = ParsedDocument(text="", pages=[], metadata={})
    assert parser.extract_sections(doc) == []


def test_extract_sections_no_known_headings() -> None:
    parser = PDFParser()
    pages = ["Just some text without headings."]
    doc = ParsedDocument(text=pages[0], pages=pages, metadata={})

    sections = parser.extract_sections(doc)
    assert len(sections) == 1
    assert sections[0].title == "Document"
    assert "without headings" in sections[0].content


def test_parse_from_bytes_mocks_fitz(monkeypatch: pytest.MonkeyPatch) -> None:
    """No real PDF required: we mock fitz.open to return a fake doc/page."""

    class FakePage:
        def __init__(self, text: str) -> None:
            self._text = text

        def get_text(self, mode: str):
            if mode == "blocks":
                # minimal blocks structure
                return [
                    (0, 0, 100, 10, self._text, 0, 0),
                ]
            if mode == "text":
                return self._text
            raise ValueError(mode)

    class FakeDoc:
        metadata = {"author": "me"}

        def __init__(self) -> None:
            self._pages = [FakePage("Abstract\nhi")]

        def __iter__(self):
            return iter(self._pages)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeFitz:
        def open(self, *args, **kwargs):
            return FakeDoc()

    parser = PDFParser()

    # Patch the fitz module import inside parse_from_bytes
    monkeypatch.setitem(__import__("sys").modules, "fitz", FakeFitz())

    doc = parser.parse_from_bytes(b"%PDF-1.4 fake")
    assert doc.metadata["author"] == "me"
    assert doc.pages == ["Abstract hi"] or doc.pages == ["Abstract\nhi"]
    assert "Abstract" in doc.text
