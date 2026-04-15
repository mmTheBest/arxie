from __future__ import annotations

import importlib
from types import SimpleNamespace

import ra.agents.lit_review_agent as lit_review_agent_module
import ra.agents.research_agent as research_agent_module
from ra.agents.lit_review_agent import LitReviewAgent
from ra.agents.research_agent import ResearchAgent
from ra.api.app import _default_retriever_factory

api_app_module = importlib.import_module("ra.api.app")


def test_default_retriever_factory_uses_runtime_builder(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(api_app_module, "build_runtime_retriever", lambda: sentinel)

    assert _default_retriever_factory() is sentinel


def test_lit_review_agent_uses_runtime_builder_when_retriever_missing(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(lit_review_agent_module, "configure_logging_from_env", lambda: None)
    monkeypatch.setattr(lit_review_agent_module, "build_runtime_retriever", lambda: sentinel)

    agent = LitReviewAgent(llm=SimpleNamespace(), tools=[])

    assert agent.retriever is sentinel


def test_research_agent_uses_runtime_builder_when_constructing_default_retriever(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(research_agent_module, "configure_logging_from_env", lambda: None)
    monkeypatch.setattr(
        research_agent_module,
        "load_config",
        lambda: SimpleNamespace(
            ra_model="gpt-4o-mini",
            openai_api_key="test-key",
            semantic_scholar_api_key=None,
        ),
    )
    monkeypatch.setattr(research_agent_module, "build_runtime_retriever", lambda: sentinel)
    monkeypatch.setattr(
        research_agent_module,
        "ChatOpenAI",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        research_agent_module,
        "SemanticScholarClient",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        research_agent_module,
        "make_retrieval_tools",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        research_agent_module,
        "create_agent",
        lambda **kwargs: SimpleNamespace(invoke=lambda inputs, config: {}),
    )

    agent = ResearchAgent()

    assert agent.retriever is sentinel
