"""LangChain-based research agent.

Uses create_agent (LangChain >=1.2) with a tool-calling loop to:
- search for papers
- fetch paper details
- chase citations
- synthesize an answer with clear references
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from ra.citation import CitationFormatter
from ra.retrieval.semantic_scholar import SemanticScholarClient
from ra.retrieval.unified import Paper, UnifiedRetriever
from ra.tools.retrieval_tools import make_retrieval_tools
from ra.utils.logging import UsageLogger


_SYSTEM_PROMPT = """You are an Academic Research Assistant.

You have access to tools for academic literature retrieval.

Your goals:
- Search for relevant papers and gather evidence from credible sources.
- Synthesize findings into a clear, structured answer.
- Always cite papers using (Author et al., Year) format inline.
- Every non-trivial factual claim should be backed by at least one citation.
- Provide a References section at the end listing all cited papers.

Tool-use rules:
- Use search_papers first, then get_paper_details for promising results.
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
        pid = str(d.get("id") or "").strip()
        if not pid:
            return None
        return Paper(
            id=pid,
            title=str(d.get("title") or "").strip(),
            abstract=(
                str(d.get("abstract")).strip() if d.get("abstract") is not None else None
            ),
            authors=list(d.get("authors") or []),
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
            if not isinstance(output, str):
                return
            obj = json.loads(output)
            if not isinstance(obj, dict):
                return
            if isinstance(obj.get("paper"), dict):
                p = _paper_from_dict(obj["paper"])
                if p:
                    self.papers[p.id] = p
            results = obj.get("results")
            if isinstance(results, list):
                for item in results:
                    if isinstance(item, dict):
                        p = _paper_from_dict(item)
                        if p:
                            self.papers[p.id] = p
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

    def __init__(self, *, model: str | None = None, verbose: bool = False):
        self.model = model or os.getenv("RA_MODEL", "gpt-4o-mini")

        api_key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(model=self.model, api_key=api_key, temperature=0)

        self.retriever = UnifiedRetriever()
        self.semantic_scholar = SemanticScholarClient()
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
        claims = self._citation_formatter.extract_claims(text, paper_list)
        cited_ids = {pid for c in claims for pid in c.supporting_papers}
        cited_papers = [p for p in paper_list if p.id in cited_ids]

        body = re.sub(r"(?is)\n+references\s*:?.*\Z", "", text).rstrip()

        if not cited_papers:
            return body + "\n\nReferences\nNone."

        refs = self._citation_formatter.format_reference_list(cited_papers).strip()
        return body + "\n\nReferences\n" + refs

    def _extract_final_text(self, result: dict[str, Any]) -> str:
        """Extract the final assistant text from the graph result."""
        messages = result.get("messages", [])
        # Walk backwards to find the last AI message without tool calls
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "ai" and not getattr(msg, "tool_calls", None):
                return str(msg.content)
            # dict form
            if isinstance(msg, dict) and msg.get("role") == "assistant" and not msg.get("tool_calls"):
                return str(msg.get("content", ""))
        # Fallback: last message content
        if messages:
            last = messages[-1]
            return str(getattr(last, "content", "") or (last.get("content", "") if isinstance(last, dict) else ""))
        return ""

    async def arun(self, query: str) -> str:
        """Async entrypoint."""
        inputs = {"messages": [HumanMessage(content=query)]}
        config = {"callbacks": [self._callback]}
        result = await self.graph.ainvoke(inputs, config=config)
        output = self._extract_final_text(result)
        return self._format_references(output)

    def run(self, query: str) -> str:
        """Run the agent loop for a single query and return the final answer."""
        inputs = {"messages": [HumanMessage(content=query)]}
        config = {"callbacks": [self._callback]}
        result = self.graph.invoke(inputs, config=config)
        output = self._extract_final_text(result)
        return self._format_references(output)
