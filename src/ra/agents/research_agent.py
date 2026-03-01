"""LangChain-based research agent.

Uses create_agent (LangChain >=1.2) with a tool-calling loop to:
- search for papers
- fetch paper details
- chase citations
- synthesize an answer with clear references
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx
from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from ra.citation import CitationFormatter
from ra.retrieval.semantic_scholar import SemanticScholarClient
from ra.retrieval.unified import Paper, UnifiedRetriever
from ra.tools.retrieval_tools import make_retrieval_tools
from ra.utils.config import load_config
from ra.utils.logging import UsageLogger
from ra.utils.logging_config import configure_logging_from_env
from ra.utils.security import sanitize_user_text

try:
    from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
except Exception:  # pragma: no cover - import availability depends on runtime package layout
    APIConnectionError = None
    APITimeoutError = None
    InternalServerError = None
    RateLimitError = None

logger = logging.getLogger(__name__)

_TRANSIENT_HTTP_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
_INLINE_CITATION_RE = re.compile(
    r"\([A-Z][A-Za-z\-]+(?:\s+et\s+al\.)?(?:\s*&\s*[A-Z][A-Za-z\-]+)?,\s*(?:19|20)\d{2}\)"
)
_DEEP_INITIAL_SEARCH_LIMIT = 10
_DEEP_FULLTEXT_TOP_K = 3
_DEEP_CITATION_LIMIT = 5
_DEEP_CITATION_SEARCH_LIMIT = 3
_DEEP_MAX_CITATION_SEARCHES = 10
_DEEP_FULLTEXT_SNIPPET_CHARS = 1200


_SYSTEM_PROMPT = """You are an Academic Research Assistant.

You have access to tools for academic literature retrieval.

Your goals:
- Search for relevant papers and gather evidence from credible sources.
- Synthesize findings into a clear, structured answer.
- Always cite papers using (Author et al., Year) format inline.
- Every non-trivial factual claim should be backed by at least one citation.
- Provide a References section at the end listing all cited papers.
- Do not answer from prior knowledge alone; ground answers in retrieved paper metadata.

Tool-use rules:
- You MUST call search_papers at least once before finalizing any answer.
- Use search_papers first, then get_paper_details for promising results.
- When a user asks about specific methods, results, experiments, discussion points,
  or conclusions from a paper, call read_paper_fulltext for that paper before
  answering.
- Use forward citation chasing (get_paper_citations) to find follow-ups or validations when helpful.

Uncertainty rules:
- If you cannot find relevant papers, say so explicitly rather than guessing.
- If evidence is weak or mixed, say so explicitly."""


# -----------------
# Usage + tracing
# -----------------


@dataclass
class _ToolRun:
    name: str
    start: float


def _estimate_openai_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Best-effort cost estimation in USD."""
    prices_per_1k: dict[str, tuple[float, float]] = {
        "gpt-4o-mini": (0.00015, 0.0006),
        "gpt-4o": (0.005, 0.015),
        "gpt-4.1-mini": (0.0003, 0.0012),
        "gpt-4.1": (0.01, 0.03),
    }
    inp, out = prices_per_1k.get(model, (0.0, 0.0))
    return (tokens_in / 1000.0) * inp + (tokens_out / 1000.0) * out


def _safe_int(v: Any) -> int:
    try:
        return int(v)
    except Exception:
        return 0


def _safe_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _safe_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _extract_token_usage(llm_result: Any) -> tuple[int, int]:
    """Extract (tokens_in, tokens_out) from LangChain LLM result objects."""
    try:
        llm_output = getattr(llm_result, "llm_output", None) or {}
        tu = llm_output.get("token_usage") if isinstance(llm_output, dict) else None
        if isinstance(tu, dict):
            return (
                _safe_int(tu.get("prompt_tokens")),
                _safe_int(tu.get("completion_tokens")),
            )
    except Exception:
        pass

    try:
        generations = getattr(llm_result, "generations", None)
        if generations:
            gen0 = generations[0][0]
            msg = getattr(gen0, "message", None)
            usage_md = getattr(msg, "usage_metadata", None) or {}
            if isinstance(usage_md, dict):
                return (
                    _safe_int(usage_md.get("input_tokens") or usage_md.get("prompt_tokens")),
                    _safe_int(
                        usage_md.get("output_tokens") or usage_md.get("completion_tokens")
                    ),
                )
    except Exception:
        pass

    return (0, 0)


def _paper_from_dict(d: dict[str, Any]) -> Paper | None:
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


class ResearchAgentCallback(BaseCallbackHandler):
    """Callback handler that logs usage and collects papers for reference formatting."""

    def __init__(self, *, usage_logger: UsageLogger, model: str) -> None:
        super().__init__()
        self.usage_logger = usage_logger
        self.model = model

        self._llm_start: float | None = None
        self._tool_stack: list[_ToolRun] = []

        self.papers: dict[str, Paper] = {}

    @staticmethod
    def _iter_tool_payload_dicts(output: Any) -> list[dict[str, Any]]:
        pending: list[Any] = [output]
        payloads: list[dict[str, Any]] = []

        while pending:
            item = pending.pop()
            if item is None:
                continue

            if isinstance(item, dict):
                payloads.append(item)
                for key in ("content", "artifact", "output", "result", "data"):
                    if key in item:
                        pending.append(item[key])
                continue

            if isinstance(item, list):
                pending.extend(item)
                continue

            if isinstance(item, str):
                text = item.strip()
                if not text:
                    continue
                try:
                    pending.append(json.loads(text))
                except Exception:
                    continue
                continue

            content = getattr(item, "content", None)
            artifact = getattr(item, "artifact", None)
            if content is not None:
                pending.append(content)
            if artifact is not None:
                pending.append(artifact)
            if content is not None or artifact is not None:
                continue

            model_dump = getattr(item, "model_dump", None)
            if callable(model_dump):
                try:
                    pending.append(model_dump())
                except Exception:
                    continue

        return payloads

    def _collect_papers_from_payload(self, obj: dict[str, Any]) -> None:
        if isinstance(obj.get("paper"), dict):
            paper = _paper_from_dict(obj["paper"])
            if paper:
                self.papers[paper.id] = paper

        for key in ("results", "papers"):
            results = obj.get(key)
            if not isinstance(results, list):
                continue
            for item in results:
                if isinstance(item, dict):
                    paper = _paper_from_dict(item)
                    if paper:
                        self.papers[paper.id] = paper

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> None:
        self._llm_start = time.time()

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        start = self._llm_start or time.time()
        elapsed_ms = int((time.time() - start) * 1000)
        tokens_in, tokens_out = _extract_token_usage(response)
        cost = _estimate_openai_cost(self.model, tokens_in=tokens_in, tokens_out=tokens_out)
        self.usage_logger.log_api_call(
            endpoint="openai.chat.completions",
            method="POST",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=cost,
            response_time_ms=elapsed_ms,
            status=200,
        )

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        start = self._llm_start or time.time()
        elapsed_ms = int((time.time() - start) * 1000)
        self.usage_logger.log_api_call(
            endpoint="openai.chat.completions",
            method="POST",
            tokens_in=0,
            tokens_out=0,
            cost=0.0,
            response_time_ms=elapsed_ms,
            status=500,
        )

    def on_tool_start(
        self, serialized: dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        name = str(serialized.get("name") or "unknown")
        self._tool_stack.append(_ToolRun(name=name, start=time.time()))

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        run = (
            self._tool_stack.pop() if self._tool_stack else _ToolRun("unknown", time.time())
        )
        elapsed_ms = int((time.time() - run.start) * 1000)
        self.usage_logger.log_api_call(
            endpoint=f"tool:{run.name}",
            method="CALL",
            tokens_in=0,
            tokens_out=0,
            cost=0.0,
            response_time_ms=elapsed_ms,
            status=200,
        )
        try:
            payloads = self._iter_tool_payload_dicts(output)
            for payload in payloads:
                self._collect_papers_from_payload(payload)
        except Exception:
            return

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        run = (
            self._tool_stack.pop() if self._tool_stack else _ToolRun("unknown", time.time())
        )
        elapsed_ms = int((time.time() - run.start) * 1000)
        self.usage_logger.log_api_call(
            endpoint=f"tool:{run.name}",
            method="CALL",
            tokens_in=0,
            tokens_out=0,
            cost=0.0,
            response_time_ms=elapsed_ms,
            status=500,
        )


# -----------------
# Agent
# -----------------


class ResearchAgent:
    """Academic Research Assistant agent (LangChain >=1.2 create_agent)."""

    def __init__(
        self,
        *,
        model: str | None = None,
        verbose: bool = False,
        deep_search: bool = False,
    ):
        configure_logging_from_env()
        config = load_config()

        self.model = model or config.ra_model
        self.deep_search = bool(deep_search)
        self.max_api_retries = max(0, _safe_env_int("RA_AGENT_MAX_RETRIES", 3))
        self.retry_base_delay_seconds = max(
            0.0,
            _safe_env_float("RA_AGENT_RETRY_BASE_SECONDS", 1.0),
        )
        base_iterations = max(4, _safe_env_int("RA_AGENT_MAX_ITERATIONS", 30))
        if self.deep_search:
            deep_default = max(50, base_iterations + 20)
            deep_iterations = _safe_env_int("RA_AGENT_DEEP_MAX_ITERATIONS", deep_default)
            self.max_iterations = max(base_iterations, deep_iterations)
        else:
            self.max_iterations = base_iterations
        logger.debug(
            "Initializing ResearchAgent",
            extra={
                "event": "research_agent.init",
                "model": self.model,
                "deep_search": self.deep_search,
                "max_api_retries": self.max_api_retries,
                "max_iterations": self.max_iterations,
            },
        )

        self.llm = ChatOpenAI(model=self.model, api_key=config.openai_api_key, temperature=0)

        self.retriever = UnifiedRetriever()
        self.semantic_scholar = SemanticScholarClient(api_key=config.semantic_scholar_api_key)
        self.tools = make_retrieval_tools(
            retriever=self.retriever,
            semantic_scholar=self.semantic_scholar,
        )

        self.usage_logger = UsageLogger()
        self._callback = ResearchAgentCallback(usage_logger=self.usage_logger, model=self.model)
        self._citation_formatter = CitationFormatter()

        self.graph = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=_SYSTEM_PROMPT,
            debug=verbose,
        )

    def _format_references(self, text: str) -> str:
        paper_list = list(self._callback.papers.values())
        body = self._clean_answer_body(text)
        body = self._ensure_inline_citations(body, paper_list)
        claims = self._citation_formatter.extract_claims(body, paper_list)
        cited_ids = {pid for c in claims for pid in c.supporting_papers}
        cited_papers = [p for p in paper_list if p.id in cited_ids]

        if not cited_papers and paper_list and _INLINE_CITATION_RE.search(body):
            cited_papers = paper_list

        answer_block = f"## Answer\n{body}"
        if not cited_papers:
            return answer_block + "\n\n## References\nNone."

        refs = self._citation_formatter.format_reference_list(cited_papers).strip()
        ref_lines = [line.strip() for line in refs.splitlines() if line.strip()]
        numbered_refs = "\n".join(f"{i}. {line}" for i, line in enumerate(ref_lines, start=1))
        return answer_block + "\n\n## References\n" + numbered_refs

    def _clean_answer_body(self, text: str) -> str:
        body = (text or "").strip()
        body = re.sub(r"(?is)(?:\n|^)\s*#{1,6}\s*references\s*:?.*\Z", "", body).strip()
        body = re.sub(r"(?is)(?:\n|^)\s*references\s*:?.*\Z", "", body).strip()
        body = re.sub(r"(?is)^\s*#{1,6}\s*answer\s*\n", "", body).strip()
        body = re.sub(r"(?is)^\s*answer\s*:?\s*", "", body).strip()
        return body or "I could not produce an answer for this query."

    def _ensure_inline_citations(self, body: str, papers: list[Paper]) -> str:
        if not papers or _INLINE_CITATION_RE.search(body):
            return body

        evidence_parts: list[str] = []
        for paper in papers[:3]:
            citation = self._citation_formatter.format_inline(paper)
            title = (paper.title or "").strip().rstrip(".")
            if title:
                evidence_parts.append(f"{title} {citation}")
            else:
                evidence_parts.append(citation)
        evidence_line = "Evidence sources: " + "; ".join(evidence_parts) + "."
        return f"{body}\n\n{evidence_line}".strip()

    def _cache_seed_papers(self, papers: list[Paper]) -> None:
        for paper in papers:
            self._callback.papers.setdefault(paper.id, paper)

    async def _ensure_seed_papers_async(self, query: str) -> None:
        if self._callback.papers:
            return
        try:
            papers = await self.retriever.search(
                query=query,
                limit=5,
                sources=("semantic_scholar", "arxiv"),
            )
            self._cache_seed_papers(papers)
        except Exception:
            logger.warning("Failed to prefetch seed papers in async flow.", exc_info=True)

    def _ensure_seed_papers_sync(self, query: str) -> None:
        if self._callback.papers:
            return
        try:
            papers = asyncio.run(
                self.retriever.search(
                    query=query,
                    limit=5,
                    sources=("semantic_scholar", "arxiv"),
                )
            )
            self._cache_seed_papers(papers)
        except RuntimeError as exc:
            logger.warning("Failed to prefetch seed papers in sync flow: %s", exc)
        except Exception:
            logger.warning("Failed to prefetch seed papers in sync flow.", exc_info=True)

    @staticmethod
    def _truncate_text(text: str, *, max_chars: int) -> str:
        clean = " ".join((text or "").split())
        if not clean:
            return ""
        if len(clean) <= max_chars:
            return clean
        return clean[: max_chars - 3].rstrip() + "..."

    @staticmethod
    def _dedupe_strings(values: list[str]) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for raw in values:
            value = raw.strip()
            if not value:
                continue
            key = " ".join(value.lower().split())
            if key in seen:
                continue
            seen.add(key)
            unique.append(value)
        return unique

    def _format_paper_brief(self, paper: Paper) -> str:
        title = (paper.title or "").strip() or "Untitled"
        source = str(getattr(paper, "source", "unknown") or "unknown")
        inline = self._citation_formatter.format_inline(paper)
        return f"{title} {inline} [{source}]"

    async def _build_deep_search_context_async(self, query: str) -> str:
        initial_papers = await self.retriever.search(
            query=query,
            limit=_DEEP_INITIAL_SEARCH_LIMIT,
            sources=("semantic_scholar", "arxiv"),
        )
        if not initial_papers:
            return ""

        self._cache_seed_papers(initial_papers)
        top_papers = initial_papers[:_DEEP_FULLTEXT_TOP_K]

        full_text_entries: list[str] = []
        for paper in top_papers:
            try:
                full_text = await self.retriever.get_full_text(paper)
            except Exception:
                logger.warning(
                    "Deep search full-text read failed for paper_id=%s.",
                    paper.id,
                    exc_info=True,
                )
                continue

            snippet = self._truncate_text(full_text, max_chars=_DEEP_FULLTEXT_SNIPPET_CHARS)
            if not snippet:
                continue
            summary = f"{self._format_paper_brief(paper)}: {snippet}"
            full_text_entries.append(summary)

        citation_titles: list[str] = []
        for paper in top_papers:
            try:
                citations = await self.semantic_scholar.get_citations(
                    paper_id=paper.id,
                    limit=_DEEP_CITATION_LIMIT,
                )
            except Exception:
                logger.warning(
                    "Deep search citation chase failed for paper_id=%s.",
                    paper.id,
                    exc_info=True,
                )
                continue

            for citation in citations:
                title = str(getattr(citation, "title", "") or "").strip()
                if title:
                    citation_titles.append(title)

        deduped_citation_titles = self._dedupe_strings(citation_titles)
        deduped_citation_titles = deduped_citation_titles[:_DEEP_MAX_CITATION_SEARCHES]

        citation_search_results: list[Paper] = []
        for title in deduped_citation_titles:
            try:
                papers = await self.retriever.search(
                    query=title,
                    limit=_DEEP_CITATION_SEARCH_LIMIT,
                    sources=("semantic_scholar", "arxiv"),
                )
            except Exception:
                logger.warning(
                    "Deep search follow-up lookup failed for citation title=%r.",
                    title,
                    exc_info=True,
                )
                continue
            citation_search_results.extend(papers)

        self._cache_seed_papers(citation_search_results)

        lines = ["Initial search:"]
        for paper in initial_papers[:5]:
            lines.append(f"- {self._format_paper_brief(paper)}")

        lines.append("")
        lines.append("Full-text review (top 3):")
        if full_text_entries:
            for entry in full_text_entries:
                lines.append(f"- {entry}")
        else:
            lines.append("- No full-text snippets were available for the top papers.")

        lines.append("")
        lines.append("Citation chasing:")
        if deduped_citation_titles:
            for title in deduped_citation_titles:
                lines.append(f"- Followed citation title: {title}")
        else:
            lines.append("- No citation titles were available from the top papers.")

        if citation_search_results:
            for paper in citation_search_results[:8]:
                lines.append(f"- Citation search hit: {self._format_paper_brief(paper)}")
        else:
            lines.append("- No additional papers were retrieved from citation-title search.")

        return "\n".join(lines).strip()

    def _augment_query_with_deep_context(self, query: str, context: str) -> str:
        return (
            "User question:\n"
            f"{query}\n\n"
            "Deep-search evidence package "
            "(initial search + full-text review + citation chasing):\n"
            f"{context}\n\n"
            "Synthesize across all sources above and any additional tool lookups. "
            "Keep claims grounded in cited papers."
        )

    async def _prepare_query_async(self, query: str) -> str:
        if not getattr(self, "deep_search", False):
            return query
        try:
            context = await self._build_deep_search_context_async(query)
        except Exception:
            logger.warning("Deep search preparation failed in async flow.", exc_info=True)
            return query
        if not context:
            return query
        return self._augment_query_with_deep_context(query, context)

    def _prepare_query_sync(self, query: str) -> str:
        if not getattr(self, "deep_search", False):
            return query
        try:
            return asyncio.run(self._prepare_query_async(query))
        except RuntimeError as exc:
            logger.warning("Deep search preparation skipped in sync flow: %s", exc)
            return query
        except Exception:
            logger.warning("Deep search preparation failed in sync flow.", exc_info=True)
            return query

    def _error_response(self) -> str:
        return (
            "## Answer\n"
            "I encountered an internal error while processing your request. Please try again.\n\n"
            "## References\nNone."
        )

    def _invalid_query_response(self) -> str:
        return (
            "## Answer\n"
            "Please provide a non-empty query without control characters (max 4000 characters).\n\n"
            "## References\nNone."
        )

    @staticmethod
    def _status_code_from_error(exc: BaseException) -> int | None:
        status = getattr(exc, "status_code", None)
        if isinstance(status, int):
            return status
        response = getattr(exc, "response", None)
        response_status = getattr(response, "status_code", None)
        if isinstance(response_status, int):
            return response_status
        return None

    def _is_transient_error(self, exc: BaseException) -> bool:
        openai_transient_types: tuple[type[BaseException], ...] = tuple(
            t
            for t in (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)
            if isinstance(t, type) and issubclass(t, BaseException)
        )

        if openai_transient_types and isinstance(exc, openai_transient_types):
            return True

        if isinstance(
            exc,
            (
                TimeoutError,
                httpx.TimeoutException,
                httpx.ConnectError,
                httpx.NetworkError,
            ),
        ):
            return True

        status = self._status_code_from_error(exc)
        if status in _TRANSIENT_HTTP_STATUS_CODES:
            return True

        message = str(exc).lower()
        return any(
            token in message
            for token in (
                "rate limit",
                "timeout",
                "timed out",
                "temporarily unavailable",
                "connection reset",
                "server overloaded",
            )
        )

    def _retry_delay(self, attempt: int) -> float:
        return self.retry_base_delay_seconds * (2**attempt)

    def _invoke_with_retries(
        self, inputs: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        for attempt in range(self.max_api_retries + 1):
            try:
                return self.graph.invoke(inputs, config=config)
            except Exception as exc:
                is_final_attempt = attempt >= self.max_api_retries
                if not self._is_transient_error(exc) or is_final_attempt:
                    raise

                delay_seconds = self._retry_delay(attempt)
                logger.warning(
                    "Transient agent invoke error on attempt %d/%d: %s. Retrying in %.2fs.",
                    attempt + 1,
                    self.max_api_retries + 1,
                    exc,
                    delay_seconds,
                )
                time.sleep(delay_seconds)

        raise RuntimeError("Agent invocation exhausted retry loop unexpectedly.")

    async def _ainvoke_with_retries(
        self, inputs: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        for attempt in range(self.max_api_retries + 1):
            try:
                return await self.graph.ainvoke(inputs, config=config)
            except Exception as exc:
                is_final_attempt = attempt >= self.max_api_retries
                if not self._is_transient_error(exc) or is_final_attempt:
                    raise

                delay_seconds = self._retry_delay(attempt)
                logger.warning(
                    "Transient agent ainvoke error on attempt %d/%d: %s. Retrying in %.2fs.",
                    attempt + 1,
                    self.max_api_retries + 1,
                    exc,
                    delay_seconds,
                )
                await asyncio.sleep(delay_seconds)

        raise RuntimeError("Agent async invocation exhausted retry loop unexpectedly.")

    def _extract_final_text(self, result: dict[str, Any]) -> str:
        """Extract the final assistant text from the graph result."""
        messages = result.get("messages", [])
        # Walk backwards to find the last AI message without tool calls
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "ai" and not getattr(msg, "tool_calls", None):
                return str(msg.content)
            # dict form
            if (
                isinstance(msg, dict)
                and msg.get("role") == "assistant"
                and not msg.get("tool_calls")
            ):
                return str(msg.get("content", ""))
        # Fallback: last message content
        if messages:
            last = messages[-1]
            fallback = ""
            if isinstance(last, dict):
                fallback = str(last.get("content", ""))
            return str(getattr(last, "content", "") or fallback)
        return ""

    async def arun(self, query: str) -> str:
        """Async entrypoint."""
        try:
            query = sanitize_user_text(query, field_name="query", max_length=4000)
        except ValueError:
            logger.warning("Rejected invalid query input in arun.")
            return self._invalid_query_response()
        self._callback.papers.clear()
        prepared_query = await self._prepare_query_async(query)
        inputs = {"messages": [HumanMessage(content=prepared_query)]}
        config = {
            "callbacks": [self._callback],
            "recursion_limit": self.max_iterations,
        }
        try:
            result = await self._ainvoke_with_retries(inputs, config=config)
            await self._ensure_seed_papers_async(query)
            output = self._extract_final_text(result)
            return self._format_references(output)
        except Exception:
            logger.exception("ResearchAgent async run failed.")
            return self._error_response()

    def run(self, query: str) -> str:
        """Run the agent loop for a single query and return the final answer."""
        try:
            query = sanitize_user_text(query, field_name="query", max_length=4000)
        except ValueError:
            logger.warning("Rejected invalid query input in run.")
            return self._invalid_query_response()
        self._callback.papers.clear()
        prepared_query = self._prepare_query_sync(query)
        inputs = {"messages": [HumanMessage(content=prepared_query)]}
        config = {
            "callbacks": [self._callback],
            "recursion_limit": self.max_iterations,
        }
        try:
            result = self._invoke_with_retries(inputs, config=config)
            self._ensure_seed_papers_sync(query)
            output = self._extract_final_text(result)
            return self._format_references(output)
        except Exception:
            logger.exception("ResearchAgent run failed.")
            return self._error_response()
