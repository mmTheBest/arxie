from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from ra.retrieval.unified import Paper, normalize_doi


@dataclass(frozen=True)
class Claim:
    """A sentence-level claim and its supporting papers."""

    text: str
    supporting_papers: list[str]
    confidence: float


class CitationFormatter:
    """Format citations and extract paper-supported claims from text."""

    # Very small, pragmatic sentence splitter.
    _SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\(\[])")

    def format_inline(self, paper: Paper) -> str:
        """Format an inline citation like "(Author et al., 2024)"."""
        author_part = self._inline_author_part(paper.authors)
        year_part = str(paper.year) if paper.year else "n.d."
        return f"({author_part}, {year_part})"

    def format_apa(self, paper: Paper) -> str:
        """Format a reference entry in an APA 7-ish format.

        Note: This is a best-effort formatter based on limited metadata.
        """

        authors_str = self._format_apa_authors(paper.authors)
        year_str = str(paper.year) if paper.year else "n.d."

        title = (paper.title or "").strip().rstrip(".")
        title_str = f"{title}." if title else "(Untitled)."

        venue_str = ""
        if paper.venue:
            venue_str = f" {paper.venue.strip().rstrip('.')}."

        locator = self._best_locator(paper)
        locator_str = f" {locator}" if locator else ""

        return f"{authors_str} ({year_str}). {title_str}{venue_str}{locator_str}".strip()

    def format_reference_list(self, papers: Iterable[Paper]) -> str:
        """Format and sort a reference list."""
        ps = list(papers)
        ps.sort(key=self._reference_sort_key)
        return "\n".join(self.format_apa(p) for p in ps)

    def extract_claims(self, text: str, papers: Iterable[Paper]) -> list[Claim]:
        """Extract cited sentences and map them to supporting papers.

        Heuristics:
        - Sentence split then search for DOI/arXiv/id mentions.
        - Search for parenthetical author-year patterns.

        Returns only sentences with at least one matched paper.
        """

        paper_list = list(papers)
        if not text.strip() or not paper_list:
            return []

        sentences = self._split_sentences(text)

        matchers = [self._paper_matcher(p) for p in paper_list]
        claims: list[Claim] = []

        for sent in sentences:
            sent_clean = sent.strip()
            if not sent_clean:
                continue

            matched: list[tuple[str, float]] = []
            for match in matchers:
                score = match(sent_clean)
                if score > 0:
                    matched.append((match.paper_id, score))

            if not matched:
                continue

            # Deduplicate and aggregate
            by_id: dict[str, float] = {}
            for pid, sc in matched:
                by_id[pid] = max(by_id.get(pid, 0.0), sc)

            supporting = sorted(by_id.keys())
            confidence = max(by_id.values())

            claims.append(Claim(text=sent_clean, supporting_papers=supporting, confidence=confidence))

        return claims

    # -----------------
    # Helpers
    # -----------------

    def _split_sentences(self, text: str) -> list[str]:
        t = " ".join(text.split())
        # Keep newlines from being meaningful; we just treat them as spaces.
        parts = re.split(self._SENTENCE_SPLIT_RE, t)
        return [p.strip() for p in parts if p and p.strip()]

    @staticmethod
    def _author_last_name(author: str) -> str:
        a = (author or "").strip()
        if not a:
            return "Unknown"

        # Handle "Last, First".
        if "," in a:
            last = a.split(",", 1)[0].strip()
            return last or "Unknown"

        tokens = a.split()
        if len(tokens) == 1:
            return tokens[0]
        return tokens[-1]

    @classmethod
    def _inline_author_part(cls, authors: list[str]) -> str:
        if not authors:
            return "Unknown"
        last_names = [cls._author_last_name(a) for a in authors]

        if len(last_names) == 1:
            return last_names[0]
        if len(last_names) == 2:
            return f"{last_names[0]} & {last_names[1]}"
        return f"{last_names[0]} et al."

    @classmethod
    def _format_apa_authors(cls, authors: list[str]) -> str:
        """APA 7 author list rules (best-effort).

        - Up to 20 authors in reference list.
        - If > 20, list first 19, ellipsis, last.
        """

        if not authors:
            return "Unknown"

        formatted = [cls._format_apa_author(a) for a in authors]

        if len(formatted) <= 20:
            if len(formatted) == 1:
                return formatted[0]
            if len(formatted) == 2:
                return f"{formatted[0]}, & {formatted[1]}"
            # Oxford comma-ish style
            return ", ".join(formatted[:-1]) + f", & {formatted[-1]}"

        # > 20
        head = formatted[:19]
        tail = formatted[-1]
        return ", ".join(head) + ", … " + tail

    @classmethod
    def _format_apa_author(cls, author: str) -> str:
        name = (author or "").strip()
        if not name:
            return "Unknown"

        # If already "Last, First" keep structure but ensure initials.
        if "," in name:
            last, rest = name.split(",", 1)
            initials = cls._initials(rest)
            last = last.strip() or "Unknown"
            return f"{last}, {initials}".strip().rstrip(",")

        parts = name.split()
        if len(parts) == 1:
            return parts[0]

        last = parts[-1]
        firsts = " ".join(parts[:-1])
        initials = cls._initials(firsts)
        return f"{last}, {initials}".strip().rstrip(",")

    @staticmethod
    def _initials(given: str) -> str:
        tokens = [t for t in re.split(r"\s+", (given or "").strip()) if t]
        if not tokens:
            return ""
        chars: list[str] = []
        for tok in tokens:
            # handle hyphenated given names: Jean-Luc -> J.-L.
            if "-" in tok:
                sub = [s for s in tok.split("-") if s]
                if sub:
                    chars.append("-".join(f"{s[0].upper()}." for s in sub))
            else:
                chars.append(f"{tok[0].upper()}." )
        return " ".join(chars)

    @staticmethod
    def _best_locator(paper: Paper) -> str:
        doi = normalize_doi(paper.doi)
        if doi:
            return f"https://doi.org/{doi}"
        if paper.arxiv_id:
            return f"arXiv:{paper.arxiv_id}"
        if paper.pdf_url:
            return paper.pdf_url
        return ""

    @classmethod
    def _reference_sort_key(cls, paper: Paper) -> tuple[str, int, str]:
        first_author = cls._author_last_name(paper.authors[0]) if paper.authors else "Unknown"
        year = paper.year if paper.year is not None else 0
        title = (paper.title or "").lower().strip()
        return (first_author.lower(), year, title)

    @staticmethod
    def _paper_matcher(paper: Paper):
        """Create a callable matcher for a given paper.

        The callable returns a confidence score in [0,1].
        """

        pid = paper.id
        doi = normalize_doi(paper.doi)
        arxiv = (paper.arxiv_id or "").strip() or None
        year = str(paper.year) if paper.year else None

        # Use first author last name; if missing, fall back to "Unknown".
        last = CitationFormatter._author_last_name(paper.authors[0]) if paper.authors else "Unknown"
        last_re = re.escape(last)

        # Author-year patterns: (Last, 2024), (Last et al., 2024), Last (2024)
        if year:
            author_year_re = re.compile(
                rf"\(\s*{last_re}(?:\s+et\s+al\.)?(?:\s*,)?\s*{re.escape(year)}\s*\)"
            )
            author_year_noparen_re = re.compile(
                rf"\b{last_re}(?:\s+et\s+al\.)?\s*\(\s*{re.escape(year)}\s*\)"
            )
        else:
            author_year_re = None
            author_year_noparen_re = None

        # Title token match (weak): 6+ char word from title.
        title = (paper.title or "").strip()
        title_token = None
        if title:
            tokens = [t for t in re.findall(r"[A-Za-z0-9]{6,}", title) if t.lower() not in {"between", "within", "towards"}]
            title_token = tokens[0] if tokens else None
            title_re = re.compile(rf"\b{re.escape(title_token)}\b", re.IGNORECASE) if title_token else None
        else:
            title_re = None

        class _Matcher:
            paper_id = pid

            def __call__(self, sentence: str) -> float:
                s = sentence

                # Strong IDs first
                if doi and (doi in s.lower() or f"doi:{doi}" in s.lower()):
                    return 0.95
                if arxiv and arxiv.lower() in s.lower():
                    return 0.92
                if pid and pid in s:
                    return 0.9

                # Author-year patterns
                if author_year_re and author_year_re.search(s):
                    return 0.8
                if author_year_noparen_re and author_year_noparen_re.search(s):
                    return 0.78

                # Author only is weaker
                if last != "Unknown" and re.search(rf"\b{last_re}\b", s):
                    # Bump if sentence contains a year at all
                    if re.search(r"\b(19|20)\d{2}\b", s):
                        return 0.65
                    return 0.55

                # Weak title token match
                if title_re and title_re.search(s):
                    return 0.5

                return 0.0

        return _Matcher()
