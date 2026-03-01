"""Literature review agent with tool-driven retrieval and LLM thematic synthesis."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ra.retrieval.unified import Paper, UnifiedRetriever
from ra.tools.retrieval_tools import make_retrieval_tools
from ra.utils.config import load_config
from ra.utils.logging_config import configure_logging_from_env
from ra.utils.security import sanitize_user_text

logger = logging.getLogger(__name__)

_REQUIRED_SECTIONS = (
    "Introduction",
    "Thematic Groups",
    "Key Findings",
    "Research Gaps",
    "Future Directions",
)
_SECTION_RE = re.compile(r"(?m)^##\s+(.+?)\s*$")
_MAX_ABSTRACT_CHARS = 600
_MAX_SECTION_CHARS = 600
_FULLTEXT_TOP_K = 3


@dataclass(slots=True)
class ThemeCluster:
    name: str
    paper_ids: list[str]
    summary: str


class LitReviewAgent:
    """Generate structured literature reviews grounded in retrieved papers."""

    def __init__(
        self,
        *,
        model: str | None = None,
        search_limit: int = 20,
        llm: Any | None = None,
        retriever: UnifiedRetriever | None = None,
        tools: list[Any] | None = None,
    ) -> None:
        configure_logging_from_env()
        self.search_limit = max(1, min(int(search_limit), 50))
        self.retriever = retriever or UnifiedRetriever()

        if llm is None:
            config = load_config()
            self.model = model or config.ra_model
            self.llm = ChatOpenAI(
                model=self.model,
                api_key=config.openai_api_key,
                temperature=0,
            )
        else:
            self.model = model or "custom"
            self.llm = llm

        self.tools = list(tools) if tools is not None else make_retrieval_tools(retriever=self.retriever)
        self._tools_by_name = {
            str(getattr(tool, "name", "")).strip(): tool
            for tool in self.tools
            if str(getattr(tool, "name", "")).strip()
        }

    @staticmethod
    def _truncate(text: str | None, *, max_chars: int) -> str:
        clean = " ".join((text or "").split())
        if len(clean) <= max_chars:
            return clean
        return clean[: max_chars - 3].rstrip() + "..."

    @staticmethod
    def _clamp_max_papers(max_papers: int | None, *, fallback: int) -> int:
        if max_papers is None:
            return fallback
        return max(1, min(int(max_papers), 50))

    @staticmethod
    def _response_to_text(response: Any) -> str:
        if response is None:
            return ""
        content = getattr(response, "content", response)
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    txt = item.get("text")
                    if isinstance(txt, str):
                        parts.append(txt)
                else:
                    txt = getattr(item, "text", None)
                    if isinstance(txt, str):
                        parts.append(txt)
            return "\n".join(part.strip() for part in parts if part.strip()).strip()
        return str(content).strip()

    async def _llm_ainvoke_text(self, messages: list[Any]) -> str:
        ainvoke = getattr(self.llm, "ainvoke", None)
        if callable(ainvoke):
            response = await ainvoke(messages)
        else:
            response = self.llm.invoke(messages)
        return self._response_to_text(response)

    def _extract_json_payload(self, text: str) -> dict[str, Any] | None:
        clean = (text or "").strip()
        if not clean:
            return None
        if clean.startswith("```"):
            clean = re.sub(r"^```(?:json)?\s*", "", clean).strip()
            clean = re.sub(r"\s*```$", "", clean).strip()

        try:
            parsed = json.loads(clean)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match is None:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _tool(self, name: str) -> Any | None:
        return self._tools_by_name.get(name)

    async def _call_tool_async(self, name: str, **kwargs: Any) -> str:
        tool = self._tool(name)
        if tool is None:
            return ""

        coroutine = getattr(tool, "coroutine", None)
        if callable(coroutine):
            try:
                return self._response_to_text(await coroutine(**kwargs))
            except Exception:
                logger.exception("LitReviewAgent tool call failed: %s", name)
                return ""

        ainvoke = getattr(tool, "ainvoke", None)
        if callable(ainvoke):
            try:
                return self._response_to_text(await ainvoke(kwargs))
            except Exception:
                logger.exception("LitReviewAgent tool ainvoke failed: %s", name)
                return ""

        func = getattr(tool, "func", None)
        if callable(func):
            try:
                return self._response_to_text(func(**kwargs))
            except Exception:
                logger.exception("LitReviewAgent tool func failed: %s", name)
                return ""

        invoke = getattr(tool, "invoke", None)
        if callable(invoke):
            try:
                return self._response_to_text(invoke(kwargs))
            except Exception:
                logger.exception("LitReviewAgent tool invoke failed: %s", name)
                return ""

        return ""

    @staticmethod
    def _paper_from_payload(d: dict[str, Any]) -> Paper | None:
        try:
            pid = str(d.get("id") or d.get("paper_id") or "").strip()
            if not pid:
                return None
            authors: list[str] = []
            authors_raw = d.get("authors")
            if isinstance(authors_raw, list):
                for author in authors_raw:
                    if isinstance(author, str):
                        name = author.strip()
                    elif isinstance(author, dict):
                        name = str(author.get("name") or "").strip()
                    else:
                        name = str(author or "").strip()
                    if name:
                        authors.append(name)
            return Paper(
                id=pid,
                title=str(d.get("title") or "").strip(),
                abstract=(
                    str(d.get("abstract")).strip() if d.get("abstract") is not None else None
                ),
                authors=authors,
                year=(int(d["year"]) if d.get("year") is not None else None),
                venue=(str(d.get("venue")).strip() if d.get("venue") else None),
                citation_count=(
                    int(d["citation_count"]) if d.get("citation_count") is not None else None
                ),
                pdf_url=(str(d.get("pdf_url")).strip() if d.get("pdf_url") else None),
                doi=(str(d.get("doi")).strip() if d.get("doi") else None),
                arxiv_id=(str(d.get("arxiv_id")).strip() if d.get("arxiv_id") else None),
                source=str(d.get("source") or "both"),
            )
        except Exception:
            return None

    async def _search_papers_async(self, topic: str, *, max_papers: int) -> list[Paper]:
        raw = await self._call_tool_async(
            "search_papers",
            query=topic,
            limit=max_papers,
            source="both",
        )
        parsed = self._extract_json_payload(raw)

        papers: list[Paper] = []
        if parsed and isinstance(parsed.get("results"), list):
            for item in parsed["results"]:
                if not isinstance(item, dict):
                    continue
                paper = self._paper_from_payload(item)
                if paper is not None:
                    papers.append(paper)

        if papers:
            return papers[:max_papers]

        try:
            fallback = await self.retriever.search(
                query=topic,
                limit=max_papers,
                sources=("semantic_scholar", "arxiv"),
            )
            return fallback[:max_papers]
        except Exception:
            logger.exception("LitReviewAgent fallback search failed.")
            return []

    async def _read_top_fulltexts_async(
        self,
        papers: list[Paper],
        *,
        max_reads: int,
    ) -> dict[str, dict[str, str]]:
        fulltext_by_id: dict[str, dict[str, str]] = {}
        for paper in papers[:max_reads]:
            raw = await self._call_tool_async("read_paper_fulltext", paper_id=paper.id)
            parsed = self._extract_json_payload(raw)
            if not parsed or parsed.get("error"):
                continue

            sections: dict[str, str] = {}
            for field in ("abstract", "methods", "results", "discussion", "conclusion"):
                value = parsed.get(field)
                if isinstance(value, str):
                    text = self._truncate(value, max_chars=_MAX_SECTION_CHARS)
                    if text:
                        sections[field] = text
            if sections:
                fulltext_by_id[paper.id] = sections
        return fulltext_by_id

    def _paper_payload(
        self,
        paper: Paper,
        *,
        fulltext_by_id: dict[str, dict[str, str]],
    ) -> dict[str, Any]:
        fulltext = fulltext_by_id.get(paper.id, {})
        return {
            "id": paper.id,
            "title": paper.title,
            "abstract": self._truncate(paper.abstract, max_chars=_MAX_ABSTRACT_CHARS),
            "authors": paper.authors[:5],
            "year": paper.year,
            "venue": paper.venue,
            "citation_count": paper.citation_count,
            "source": paper.source,
            "methods": fulltext.get("methods"),
            "results": fulltext.get("results"),
            "discussion": fulltext.get("discussion"),
            "conclusion": fulltext.get("conclusion"),
        }

    @staticmethod
    def _normalize_clusters(
        *,
        parsed: dict[str, Any] | None,
        papers: list[Paper],
    ) -> list[ThemeCluster]:
        by_id = {paper.id: paper for paper in papers}
        if not parsed or not isinstance(parsed.get("themes"), list):
            return [
                ThemeCluster(
                    name="General Landscape",
                    paper_ids=[paper.id for paper in papers],
                    summary="Collected studies that characterize the topic landscape.",
                )
            ]

        clusters: list[ThemeCluster] = []
        assigned: set[str] = set()
        for raw_theme in parsed["themes"]:
            if not isinstance(raw_theme, dict):
                continue
            name = str(raw_theme.get("name") or "").strip()
            if not name:
                continue
            raw_ids = raw_theme.get("paper_ids")
            if not isinstance(raw_ids, list):
                continue
            paper_ids: list[str] = []
            for value in raw_ids:
                paper_id = str(value).strip()
                if paper_id and paper_id in by_id and paper_id not in paper_ids:
                    paper_ids.append(paper_id)
                    assigned.add(paper_id)
            if not paper_ids:
                continue
            summary = str(raw_theme.get("summary") or "").strip()
            clusters.append(ThemeCluster(name=name, paper_ids=paper_ids, summary=summary))

        unassigned = [paper.id for paper in papers if paper.id not in assigned]
        if unassigned:
            clusters.append(
                ThemeCluster(
                    name="Additional Evidence",
                    paper_ids=unassigned,
                    summary="Relevant papers that do not cleanly fit primary clusters.",
                )
            )

        if not clusters:
            clusters.append(
                ThemeCluster(
                    name="General Landscape",
                    paper_ids=[paper.id for paper in papers],
                    summary="Collected studies that characterize the topic landscape.",
                )
            )
        return clusters

    async def _cluster_papers_async(
        self,
        topic: str,
        papers: list[Paper],
        *,
        fulltext_by_id: dict[str, dict[str, str]],
    ) -> list[ThemeCluster]:
        payload = [
            self._paper_payload(paper, fulltext_by_id=fulltext_by_id)
            for paper in papers
        ]
        messages = [
            SystemMessage(
                content=(
                    "You cluster research papers into themes. "
                    "Return JSON only with schema: "
                    '{"themes":[{"name":"string","paper_ids":["id"],"summary":"string"}]}.'
                )
            ),
            HumanMessage(
                content=(
                    f"Topic: {topic}\n"
                    "Cluster these papers into 2-5 coherent themes.\n"
                    "Use each paper at most once when possible.\n"
                    "Use full-text excerpts when available.\n"
                    f"Papers JSON:\n{json.dumps(payload, ensure_ascii=False)}"
                )
            ),
        ]
        try:
            raw = await self._llm_ainvoke_text(messages)
        except Exception:
            logger.exception("LitReviewAgent clustering call failed.")
            return self._normalize_clusters(parsed=None, papers=papers)

        parsed = self._extract_json_payload(raw)
        return self._normalize_clusters(parsed=parsed, papers=papers)

    @staticmethod
    def _split_markdown_sections(text: str) -> dict[str, str]:
        sections: dict[str, str] = {}
        matches = list(_SECTION_RE.finditer(text))
        if not matches:
            return sections

        for index, match in enumerate(matches):
            raw_name = match.group(1).strip()
            key = raw_name.lower()
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            content = text[start:end].strip()
            sections[key] = content
        return sections

    @staticmethod
    def _paper_inline(paper: Paper) -> str:
        if not paper.authors:
            author = "Unknown"
        else:
            first = paper.authors[0].split()[-1]
            author = f"{first} et al." if len(paper.authors) > 1 else first
        year = str(paper.year) if paper.year else "n.d."
        return f"({author}, {year})"

    def _fallback_section_content(
        self,
        *,
        section: str,
        topic: str,
        papers: list[Paper],
        clusters: list[ThemeCluster],
    ) -> str:
        if section == "Introduction":
            return (
                f"This review summarizes {len(papers)} retrieved papers on {topic}, "
                "organized into thematic evidence groups."
            )
        if section == "Thematic Groups":
            lines: list[str] = []
            by_id = {paper.id: paper for paper in papers}
            for cluster in clusters:
                citations = []
                for paper_id in cluster.paper_ids[:3]:
                    paper = by_id.get(paper_id)
                    if paper is None:
                        continue
                    citations.append(f"{paper.title} {self._paper_inline(paper)}")
                evidence = "; ".join(citations) if citations else "No representative papers listed."
                summary = cluster.summary or "Theme synthesized from retrieved evidence."
                lines.append(f"- **{cluster.name}**: {summary} Evidence: {evidence}")
            return "\n".join(lines) if lines else "- No themes were identified."
        if section == "Key Findings":
            top = papers[:3]
            if not top:
                return "- Insufficient evidence to extract findings."
            return "\n".join(
                f"- {paper.title} {self._paper_inline(paper)} provides relevant evidence."
                for paper in top
            )
        if section == "Research Gaps":
            return (
                "- Comparative evaluations across themes are limited.\n"
                "- External validity across domains and datasets is underreported."
            )
        return (
            "- Build benchmarked, reproducible evaluations across thematic categories.\n"
            "- Prioritize datasets and protocols that improve cross-study comparability."
        )

    def _ensure_required_sections(
        self,
        *,
        topic: str,
        text: str,
        papers: list[Paper],
        clusters: list[ThemeCluster],
    ) -> str:
        parsed = self._split_markdown_sections(text)
        rendered: list[str] = []
        for section in _REQUIRED_SECTIONS:
            key = section.lower()
            content = parsed.get(key, "").strip()
            if not content:
                content = self._fallback_section_content(
                    section=section,
                    topic=topic,
                    papers=papers,
                    clusters=clusters,
                )
            rendered.append(f"## {section}\n{content}".strip())
        return "\n\n".join(rendered).strip()

    async def _generate_review_async(
        self,
        topic: str,
        papers: list[Paper],
        clusters: list[ThemeCluster],
        *,
        fulltext_by_id: dict[str, dict[str, str]],
    ) -> str:
        cluster_payload = [
            {
                "name": cluster.name,
                "summary": cluster.summary,
                "paper_ids": cluster.paper_ids,
            }
            for cluster in clusters
        ]
        paper_payload = [
            self._paper_payload(paper, fulltext_by_id=fulltext_by_id)
            for paper in papers
        ]
        messages = [
            SystemMessage(
                content=(
                    "You write concise, evidence-grounded literature reviews. "
                    "Return Markdown only with exactly these H2 sections in order: "
                    "Introduction, Thematic Groups, Key Findings, Research Gaps, "
                    "Future Directions."
                )
            ),
            HumanMessage(
                content=(
                    f"Topic: {topic}\n"
                    "Use the provided thematic clusters and papers to draft the review.\n"
                    "Use inline citations in format (Author et al., Year) when possible.\n"
                    "Prioritize methodology/results details from full-text sections where present.\n"
                    f"Clusters:\n{json.dumps(cluster_payload, ensure_ascii=False)}\n"
                    f"Papers:\n{json.dumps(paper_payload, ensure_ascii=False)}"
                )
            ),
        ]
        try:
            return await self._llm_ainvoke_text(messages)
        except Exception:
            logger.exception("LitReviewAgent review generation call failed.")
            return ""

    def _invalid_topic_response(self) -> str:
        lines = [
            "## Introduction",
            "Please provide a non-empty topic without control characters (max 1000 chars).",
            "",
            "## Thematic Groups",
            "- None.",
            "",
            "## Key Findings",
            "- None.",
            "",
            "## Research Gaps",
            "- None.",
            "",
            "## Future Directions",
            "- None.",
        ]
        return "\n".join(lines)

    def _no_results_response(self, topic: str) -> str:
        lines = [
            "## Introduction",
            f"No relevant papers were retrieved for topic: {topic}.",
            "",
            "## Thematic Groups",
            "- No themes were identified from retrieved evidence.",
            "",
            "## Key Findings",
            "- No findings can be synthesized without retrieved papers.",
            "",
            "## Research Gaps",
            "- Retrieval coverage is currently insufficient for synthesis.",
            "",
            "## Future Directions",
            "- Broaden query terms and rerun retrieval.",
        ]
        return "\n".join(lines)

    async def arun(self, topic: str, max_papers: int | None = None) -> str:
        """Generate a structured literature review asynchronously."""
        try:
            topic = sanitize_user_text(topic, field_name="topic", max_length=1000)
        except ValueError:
            return self._invalid_topic_response()

        limit = self._clamp_max_papers(max_papers, fallback=self.search_limit)
        papers = await self._search_papers_async(topic, max_papers=limit)
        if not papers:
            return self._no_results_response(topic)

        max_reads = min(_FULLTEXT_TOP_K, len(papers), limit)
        fulltext_by_id = await self._read_top_fulltexts_async(papers, max_reads=max_reads)
        clusters = await self._cluster_papers_async(
            topic,
            papers,
            fulltext_by_id=fulltext_by_id,
        )
        draft = await self._generate_review_async(
            topic,
            papers,
            clusters,
            fulltext_by_id=fulltext_by_id,
        )
        return self._ensure_required_sections(
            topic=topic,
            text=draft,
            papers=papers,
            clusters=clusters,
        )

    def run(self, topic: str, max_papers: int | None = None) -> str:
        """Generate a structured literature review synchronously."""
        try:
            return asyncio.run(self.arun(topic, max_papers=max_papers))
        except RuntimeError as exc:
            logger.warning("LitReviewAgent sync run failed: %s", exc)
            return self._invalid_topic_response()
