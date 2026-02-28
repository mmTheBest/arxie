"""FastAPI app exposing the RA pipeline as REST endpoints."""

from __future__ import annotations

import logging
from collections.abc import Callable

import httpx
from fastapi import FastAPI

from ra.agents.research_agent import ResearchAgent
from ra.api.errors import RAAPIError, register_exception_handlers
from ra.api.models import (
    AnswerRequest,
    AnswerResponse,
    ErrorResponse,
    HealthResponse,
    PaperResponse,
    RetrieveRequest,
    RetrieveResponse,
    SearchRequest,
    SearchResponse,
)
from ra.retrieval.unified import UnifiedRetriever
from ra.utils.logging_config import configure_logging_from_env

logger = logging.getLogger(__name__)

RetrieverFactory = Callable[[], UnifiedRetriever]
AgentFactory = Callable[[], ResearchAgent]


def _default_retriever_factory() -> UnifiedRetriever:
    return UnifiedRetriever()


def _default_agent_factory() -> ResearchAgent:
    return ResearchAgent()


def _sources_for_request(source: str) -> tuple[str, ...]:
    if source == "both":
        return ("semantic_scholar", "arxiv")
    return (source,)


def create_app(
    *,
    retriever_factory: RetrieverFactory | None = None,
    agent_factory: AgentFactory | None = None,
) -> FastAPI:
    """Create the FastAPI app instance.

    Factories are injectable to make endpoint behavior easy to unit test.
    """
    configure_logging_from_env()

    app = FastAPI(
        title="Academic Research Assistant API",
        description="REST API wrapper for search, retrieve, and answer pipeline operations.",
        version="1.0.0",
    )
    app.state.retriever_factory = retriever_factory or _default_retriever_factory
    app.state.agent_factory = agent_factory or _default_agent_factory

    register_exception_handlers(app)

    @app.get(
        "/health",
        response_model=HealthResponse,
        responses={500: {"model": ErrorResponse}},
        tags=["system"],
    )
    async def health() -> HealthResponse:
        return HealthResponse()

    @app.post(
        "/search",
        response_model=SearchResponse,
        responses={
            400: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
            502: {"model": ErrorResponse},
        },
        tags=["pipeline"],
    )
    async def search_papers(payload: SearchRequest) -> SearchResponse:
        retriever = app.state.retriever_factory()
        try:
            async with retriever as session:
                papers = await session.search(
                    query=payload.query,
                    limit=payload.limit,
                    sources=_sources_for_request(payload.source),
                )
        except ValueError as exc:
            raise RAAPIError(
                status_code=400,
                error="invalid_input",
                message=str(exc),
            ) from exc
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            raise RAAPIError(
                status_code=502,
                error="upstream_error",
                message=f"Search backend request failed: {type(exc).__name__}.",
            ) from exc

        results = [PaperResponse.from_paper(paper) for paper in papers]
        return SearchResponse(query=payload.query, count=len(results), results=results)

    @app.post(
        "/retrieve",
        response_model=RetrieveResponse,
        responses={
            400: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
            502: {"model": ErrorResponse},
        },
        tags=["pipeline"],
    )
    async def retrieve_paper(payload: RetrieveRequest) -> RetrieveResponse:
        retriever = app.state.retriever_factory()
        try:
            async with retriever as session:
                paper = await session.get_paper(payload.identifier)
        except ValueError as exc:
            raise RAAPIError(
                status_code=400,
                error="invalid_input",
                message=str(exc),
            ) from exc
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            raise RAAPIError(
                status_code=502,
                error="upstream_error",
                message=f"Retrieve backend request failed: {type(exc).__name__}.",
            ) from exc

        if paper is None:
            raise RAAPIError(
                status_code=404,
                error="paper_not_found",
                message=f"No paper found for identifier: {payload.identifier}",
            )

        return RetrieveResponse(
            identifier=payload.identifier,
            paper=PaperResponse.from_paper(paper),
        )

    @app.post(
        "/answer",
        response_model=AnswerResponse,
        responses={
            400: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
            502: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        tags=["pipeline"],
    )
    async def answer_question(payload: AnswerRequest) -> AnswerResponse:
        try:
            agent = app.state.agent_factory()
        except ValueError as exc:
            raise RAAPIError(
                status_code=503,
                error="agent_unavailable",
                message=str(exc),
            ) from exc
        except Exception as exc:
            logger.exception("Failed to initialize ResearchAgent", exc_info=exc)
            raise RAAPIError(
                status_code=500,
                error="internal_error",
                message="Failed to initialize answer agent.",
            ) from exc

        try:
            answer = await agent.arun(payload.query)
        except ValueError as exc:
            raise RAAPIError(
                status_code=400,
                error="invalid_input",
                message=str(exc),
            ) from exc
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            raise RAAPIError(
                status_code=502,
                error="upstream_error",
                message=f"Answer backend request failed: {type(exc).__name__}.",
            ) from exc

        return AnswerResponse(query=payload.query, answer=answer)

    return app


app = create_app()
