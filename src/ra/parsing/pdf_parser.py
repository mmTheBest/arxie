"""PDF parsing pipeline.

Design goals:
- Prefer PyMuPDF (fitz) for robust text extraction.
- Use pdfplumber as a best-effort fallback for extracting tables.
- Apply lightweight heuristics for common PDF issues:
  - multi-column layout ordering
  - header/footer removal
  - reference section identification

The implementation is intentionally heuristic (PDFs are messy) but aims to be
predictable, testable, and easy to extend.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ParsedDocument:
    text: str
    pages: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Section:
    title: str
    content: str
    page_start: int


class PDFParser:
    """Parse PDFs into plain text and heuristic sections."""

    # Common section names in papers (case-insensitive).
    _SECTION_HEADINGS: tuple[str, ...] = (
        "abstract",
        "introduction",
        "background",
        "related work",
        "literature review",
        "methods",
        "method",
        "methodology",
        "materials and methods",
        "experiments",
        "results",
        "discussion",
        "conclusion",
        "conclusions",
        "future work",
        "acknowledgements",
        "acknowledgments",
        "references",
        "bibliography",
        "appendix",
    )

    def parse(self, pdf_path: Path) -> ParsedDocument:
        """Extract text from a PDF at a filesystem path."""
        if not pdf_path.exists():
            raise FileNotFoundError(str(pdf_path))

        # Lazy import to keep module importable in environments without deps
        # (tests will monkeypatch this).
        import fitz  # type: ignore

        with fitz.open(pdf_path) as doc:
            metadata = dict(doc.metadata or {})
            pages = [self._extract_page_text(page) for page in doc]

        pages = self._remove_headers_footers(pages)
        text = "\n\n".join(pages).strip()

        # Best-effort: add extracted tables (as markdown) from pdfplumber.
        # This is supplemental and should not break parsing.
        table_text = self._extract_tables_pdfplumber_from_path(pdf_path)
        if table_text:
            text = (text + "\n\n" + table_text).strip()

        return ParsedDocument(text=text, pages=pages, metadata=metadata)

    def parse_from_bytes(self, data: bytes) -> ParsedDocument:
        """Extract text from an in-memory PDF."""
        if not data:
            return ParsedDocument(text="", pages=[], metadata={})

        import fitz  # type: ignore

        with fitz.open(stream=data, filetype="pdf") as doc:
            metadata = dict(doc.metadata or {})
            pages = [self._extract_page_text(page) for page in doc]

        pages = self._remove_headers_footers(pages)
        text = "\n\n".join(pages).strip()

        table_text = self._extract_tables_pdfplumber_from_bytes(data)
        if table_text:
            text = (text + "\n\n" + table_text).strip()

        return ParsedDocument(text=text, pages=pages, metadata=metadata)

    def extract_sections(self, doc: ParsedDocument) -> list[Section]:
        """Identify high-level sections in an academic paper.

        Heuristic approach:
        - Find canonical headings (Abstract, Introduction, ..., References)
        - Slice text between headings
        - Determine the starting page for each section by searching page texts

        If no known headings are found:
        - Return a single "Document" section when there is content.
        - Return [] when the document is empty.
        """

        text = (doc.text or "").strip()
        if not text:
            return []

        normalized = self._normalize_text(text)

        # Find candidate headings and their locations.
        # We match headings at line starts, optionally numbered (e.g., "1 Introduction").
        heading_patterns = []
        for h in self._SECTION_HEADINGS:
            # allow variants like "1. Introduction" or "I. INTRODUCTION"
            heading_patterns.append(
                rf"^(?:\s*(?:\d+|[IVXLC]+)\s*[\.)]?\s+)?{re.escape(h)}\s*$"
            )

        combined = "|".join(heading_patterns)
        matches = list(re.finditer(combined, normalized, flags=re.IGNORECASE | re.MULTILINE))

        if not matches:
            return [Section(title="Document", content=text, page_start=0)]

        # Build sections in order.
        sections: list[Section] = []
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(normalized)
            title = normalized[m.start() : m.end()].strip()
            content = normalized[m.end() : end].strip()

            # Normalize title (Title Case but keep canonical if all caps)
            title_clean = self._clean_heading_title(title)
            page_start = self._find_page_index_for_heading(doc.pages, title_clean)  # 0-based
            sections.append(Section(title=title_clean, content=content, page_start=page_start))

            # Reference section identification: stop after references/bibliography.
            if title_clean.lower() in {"references", "bibliography"}:
                break

        # Add inferred title section if text begins with a likely paper title.
        # We only do this if the first discovered section isn't already at the start.
        first_heading_pos = matches[0].start()
        prefix = normalized[:first_heading_pos].strip()
        inferred = self._infer_title_section(prefix)
        if inferred is not None:
            sections.insert(
                0,
                Section(
                    title=inferred,
                    content=prefix.strip(),
                    page_start=0,
                ),
            )

        return sections

    # -----------------
    # Extraction helpers
    # -----------------

    def _extract_page_text(self, page: Any) -> str:
        """Extract a single page with multi-column ordering heuristics."""

        # Prefer block extraction for ordering.
        try:
            blocks = page.get_text("blocks")  # (x0,y0,x1,y1,text,block_no,block_type)
        except Exception:
            blocks = None

        if not blocks:
            try:
                return (page.get_text("text") or "").strip()
            except Exception:
                return ""

        # Filter out empty blocks.
        cleaned_blocks: list[tuple[float, float, float, float, str]] = []
        for b in blocks:
            try:
                x0, y0, x1, y1, t = float(b[0]), float(b[1]), float(b[2]), float(b[3]), str(b[4])
            except Exception:
                continue
            t = self._clean_extracted_text(t)
            if t:
                cleaned_blocks.append((x0, y0, x1, y1, t))

        if not cleaned_blocks:
            return ""

        # Detect 2-column layout.
        is_two_col, x_split = self._detect_two_columns(cleaned_blocks)

        if not is_two_col:
            ordered = sorted(cleaned_blocks, key=lambda b: (b[1], b[0]))
        else:
            left = [b for b in cleaned_blocks if b[0] < x_split]
            right = [b for b in cleaned_blocks if b[0] >= x_split]
            ordered = sorted(left, key=lambda b: (b[1], b[0])) + sorted(right, key=lambda b: (b[1], b[0]))

        text = "\n".join(b[4] for b in ordered)
        return self._postprocess_page_text(text)

    def _detect_two_columns(self, blocks: list[tuple[float, float, float, float, str]]) -> tuple[bool, float]:
        """Very lightweight two-column detection.

        Returns (is_two_col, x_split).
        """

        if len(blocks) < 6:
            return (False, 0.0)

        # Use x0 positions as rough clustering.
        x0s = sorted(b[0] for b in blocks)
        # Consider median gap between adjacent x0s.
        gaps = [x0s[i + 1] - x0s[i] for i in range(len(x0s) - 1)]
        if not gaps:
            return (False, 0.0)

        # If there's a large gap, treat as split between columns.
        max_gap = max(gaps)
        if max_gap < 60:  # pixels heuristic
            return (False, 0.0)

        idx = gaps.index(max_gap)
        x_split = (x0s[idx] + x0s[idx + 1]) / 2
        return (True, x_split)

    def _remove_headers_footers(self, pages: list[str]) -> list[str]:
        """Remove repeated header/footer lines across pages."""

        if len(pages) < 2:
            return [p.strip() for p in pages]

        def lines(p: str) -> list[str]:
            # normalize whitespace and drop empties
            out: list[str] = []
            for ln in p.splitlines():
                ln2 = re.sub(r"\s+", " ", ln).strip()
                if ln2:
                    out.append(ln2)
            return out

        top_n = 3
        bottom_n = 3

        top_counts: dict[str, int] = {}
        bottom_counts: dict[str, int] = {}

        page_lines = [lines(p) for p in pages]
        for ls in page_lines:
            for t in ls[:top_n]:
                top_counts[t] = top_counts.get(t, 0) + 1
            for b in ls[-bottom_n:]:
                bottom_counts[b] = bottom_counts.get(b, 0) + 1

        # Consider a line header/footer if it appears on >= 60% of pages.
        thresh = max(2, int(len(pages) * 0.6))
        headers = {k for k, v in top_counts.items() if v >= thresh}
        footers = {k for k, v in bottom_counts.items() if v >= thresh}

        cleaned_pages: list[str] = []
        for ls in page_lines:
            # drop headers
            while ls and ls[0] in headers:
                ls = ls[1:]
            # drop footers
            while ls and ls[-1] in footers:
                ls = ls[:-1]
            cleaned_pages.append("\n".join(ls).strip())

        return cleaned_pages

    def _extract_tables_pdfplumber_from_path(self, pdf_path: Path) -> str:
        try:
            import pdfplumber  # type: ignore
        except Exception:
            return ""

        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                return self._extract_tables_pdfplumber(pdf)
        except Exception:
            return ""

    def _extract_tables_pdfplumber_from_bytes(self, data: bytes) -> str:
        try:
            import pdfplumber  # type: ignore
        except Exception:
            return ""

        try:
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                return self._extract_tables_pdfplumber(pdf)
        except Exception:
            return ""

    def _extract_tables_pdfplumber(self, pdf: Any) -> str:
        """Extract tables with pdfplumber and return markdown-ish text."""

        tables_out: list[str] = []
        for i, page in enumerate(getattr(pdf, "pages", []) or []):
            try:
                tables = page.extract_tables() or []
            except Exception:
                continue

            for t_idx, table in enumerate(tables):
                if not table:
                    continue
                # Make a simple pipe table. (No guarantee of perfect structure.)
                rows = [[(c or "").strip() for c in row] for row in table if row]
                if not rows:
                    continue
                md = [f"Table p.{i+1}.{t_idx+1}:"]
                for row in rows[:50]:
                    md.append(" | ".join(row))
                tables_out.append("\n".join(md))

        return "\n\n".join(tables_out).strip()

    # -----------------
    # Text normalization
    # -----------------

    def _postprocess_page_text(self, text: str) -> str:
        # De-hyphenate common line-break hyphenation.
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        # Normalize excessive newlines.
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _clean_extracted_text(self, text: str) -> str:
        text = text.replace("\u00ad", "")  # soft hyphen
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _normalize_text(self, text: str) -> str:
        # Keep line structure but normalize whitespace inside lines.
        lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
        return "\n".join(lines).strip()

    def _clean_heading_title(self, title: str) -> str:
        t = title.strip()
        # remove numbering like "1. "
        t = re.sub(r"^\s*(?:\d+|[IVXLC]+)\s*[\.)]?\s+", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s+", " ", t).strip()
        # Canonicalize a few.
        lower = t.lower()
        if lower in {"acknowledgements", "acknowledgments"}:
            return "Acknowledgements"
        if lower == "related work":
            return "Related Work"
        return t[:1].upper() + t[1:] if t else t

    def _infer_title_section(self, prefix_text: str) -> str | None:
        """Infer title from first non-empty line(s) before first heading."""
        if not prefix_text:
            return None
        lines = [ln.strip() for ln in prefix_text.splitlines() if ln.strip()]
        if not lines:
            return None

        # Heuristic: title is the first line if it's not too long and not obviously author list.
        first = lines[0]
        if len(first) > 200:
            return None
        if re.search(r"\b(university|institute|department)\b", first, flags=re.IGNORECASE):
            return None
        return "Title"

    def _find_page_index_for_heading(self, pages: list[str], heading: str) -> int:
        if not pages:
            return 0
        needle = re.sub(r"\s+", " ", heading).strip().lower()
        for i, p in enumerate(pages):
            p_norm = re.sub(r"\s+", " ", p).strip().lower()
            if needle and needle in p_norm:
                return i
        return 0
