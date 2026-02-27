"""LangChain-based research agent.

This is an initial skeleton for an Academic Research Assistant that uses a
ReAct-style loop (tool use + reasoning) to:
- search for papers
- fetch paper details
- chase citations
- synthesize an answer with clear references

The tool implementations are intentionally minimal and will evolve as the system
adds PDF parsing, caching, and more robust citation formatting.
"""

from __future__ import annotations

import os

from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from ra.retrieval.semantic_scholar import SemanticScholarClient
from ra.retrieval.unified import UnifiedRetriever
from ra.tools.retrieval_tools import make_retrieval_tools


_REACT_PROMPT_TEMPLATE = """You are an Academic Research Assistant.

You have access to tools for academic literature retrieval.

Your goals:
- Search for relevant papers and gather evidence from credible sources.
- Synthesize findings into a clear, structured answer.
- Always cite sources. Every non-trivial factual claim should be backed by at least one citation.
- At the end of your answer, include a References section listing cited papers with enough info to locate them (title + year + DOI/arXiv/URL when available).

When deciding what to do:
- Prefer recent and highly-cited work when appropriate.
- Use citation chasing (forward citations) to find follow-ups or validations.
- If evidence is weak or mixed, say so explicitly.

You MUST follow this format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (a JSON object)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final: the final answer to the original question, with citations and a References section

Question: {input}
{agent_scratchpad}
"""


class ResearchAgent:
    """Academic Research Assistant agent (LangChain + ReAct)."""

    def __init__(self, *, model: str | None = None, verbose: bool = False):
        self.model = model or os.getenv("RA_MODEL", "gpt-4o-mini")

        # ChatOpenAI will also read OPENAI_API_KEY from env automatically. We pass it
        # explicitly when present to make configuration clearer.
        api_key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(model=self.model, api_key=api_key, temperature=0)

        self.retriever = UnifiedRetriever()
        self.semantic_scholar = SemanticScholarClient()
        self.tools = make_retrieval_tools(
            retriever=self.retriever,
            semantic_scholar=self.semantic_scholar,
        )

        prompt = PromptTemplate.from_template(_REACT_PROMPT_TEMPLATE)
        agent = create_react_agent(self.llm, self.tools, prompt)

        self.executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=verbose,
            handle_parsing_errors=True,
        )

    async def arun(self, query: str) -> str:
        """Async entrypoint."""
        result = await self.executor.ainvoke({"input": query})
        return str(result.get("output", ""))

    def run(self, query: str) -> str:
        """Run the agent loop for a single query and return the final answer."""
        import asyncio

        return asyncio.run(self.arun(query))
