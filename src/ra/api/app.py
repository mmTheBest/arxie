"""FastAPI app exposing the RA pipeline as REST endpoints."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import httpx
from fastapi import Body, FastAPI

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

OPENAPI_TAGS = [
    {
        "name": "system",
        "description": "Service health and operational metadata endpoints.",
    },
    {
        "name": "pipeline",
        "description": (
            "Core research workflow endpoints for searching papers, retrieving paper "
            "metadata, and generating grounded answers."
        ),
    },
]

SEARCH_REQUEST_EXAMPLES = {
    "cross_source_discovery": {
        "summary": "Search both providers",
        "description": "Discover papers across Semantic Scholar and arXiv in one call.",
        "value": {
            "query": "transformer architecture for long-context summarization",
            "limit": 5,
            "source": "both",
        },
    },
    "semantic_scholar_only": {
        "summary": "Semantic Scholar only",
        "value": {
            "query": "retrieval augmented generation benchmark survey",
            "limit": 10,
            "source": "semantic_scholar",
        },
    },
}

RETRIEVE_REQUEST_EXAMPLES = {
    "doi_lookup": {
        "summary": "Retrieve by DOI",
        "value": {"identifier": "10.5555/3295222.3295349"},
    },
    "arxiv_lookup": {
        "summary": "Retrieve by arXiv ID",
        "value": {"identifier": "1706.03762"},
    },
}

ANSWER_REQUEST_EXAMPLES = {
    "limitations_question": {
        "summary": "Question answered by research agent",
        "value": {
            "query": "What are the main limitations of retrieval-augmented generation?"
        },
    }
}


def _error_response_doc(
    *,
    description: str,
    error: str,
    message: str,
    details: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"error": error, "message": message}
    if details is not None:
        payload["details"] = details

    return {
        "model": ErrorResponse,
        "description": description,
        "content": {"application/json": {"example": payload}},
    }


VALIDATION_ERROR_RESPONSE = _error_response_doc(
    description="Request payload failed schema validation.",
    error="validation_error",
    message="Invalid request payload.",
    details=[{"loc": ["body", "query"], "msg": "Field required", "type": "missing"}],
)

INTERNAL_ERROR_RESPONSE = _error_response_doc(
    description="Unhandled internal server error.",
    error="internal_error",
    message="Internal server error.",
)


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
        summary="REST API for paper discovery and grounded research answers.",
        description=(
            "REST API wrapper for search, retrieve, and answer pipeline operations. "
            "Use this API to discover papers, fetch structured paper metadata, and "
            "generate answers grounded in retrieved literature."
        ),
        contact={"name": "Academic Research Assistant Maintainers"},
        license_info={"name": "MIT", "identifier": "MIT"},
        openapi_tags=OPENAPI_TAGS,
        version="1.0.0",
    )
    app.state.retriever_factory = retriever_factory or _default_retriever_factory
    app.state.agent_factory = agent_factory or _default_agent_factory

    register_exception_handlers(app)

    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Service Health Check",
        description="Returns service status, service name, and current API version.",
        response_description="Health and service metadata.",
        responses={500: INTERNAL_ERROR_RESPONSE},
        tags=["system"],
    )
    async def health() -> HealthResponse:
        return HealthResponse()

    @app.post(
        "/search",
        response_model=SearchResponse,
        summary="Search Academic Papers",
        description=(
            "Searches paper metadata from Semantic Scholar, arXiv, or both sources. "
            "Results include canonical citation strings for downstream use."
        ),
        response_description="Search results with normalized paper metadata.",
        responses={
            400: _error_response_doc(
                description="Invalid search request.",
                error="invalid_input",
                message="query must not be empty",
            ),
            422: VALIDATION_ERROR_RESPONSE,
            500: INTERNAL_ERROR_RESPONSE,
            502: _error_response_doc(
                description="Search provider request failed.",
                error="upstream_error",
                message="Search backend request failed: RequestError.",
            ),
        },
        tags=["pipeline"],
    )
    async def search_papers(
        payload: SearchRequest = Body(..., openapi_examples=SEARCH_REQUEST_EXAMPLES),
    ) -> SearchResponse:
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
        summary="Retrieve Paper Metadata",
        description=(
            "Fetches a single paper by identifier. The identifier may be a DOI, "
            "arXiv ID, or provider-specific identifier."
        ),
        response_description="Resolved paper metadata for the requested identifier.",
        responses={
            400: _error_response_doc(
                description="Invalid retrieve request.",
                error="invalid_input",
                message="identifier must not be empty",
            ),
            404: _error_response_doc(
                description="No paper matched the provided identifier.",
                error="paper_not_found",
                message="No paper found for identifier: 10.5555/3295222.3295349",
            ),
            422: VALIDATION_ERROR_RESPONSE,
            500: INTERNAL_ERROR_RESPONSE,
            502: _error_response_doc(
                description="Retrieve provider request failed.",
                error="upstream_error",
                message="Retrieve backend request failed: RequestError.",
            ),
        },
        tags=["pipeline"],
    )
    async def retrieve_paper(
        payload: RetrieveRequest = Body(..., openapi_examples=RETRIEVE_REQUEST_EXAMPLES),
    ) -> RetrieveResponse:
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
        summary="Generate Literature-Grounded Answer",
        description=(
            "Runs the research agent pipeline to generate a structured Markdown "
            "answer for a research question."
        ),
        response_description="Structured answer produced by the research agent.",
        responses={
            400: _error_response_doc(
                description="Invalid answer request.",
                error="invalid_input",
                message="query must not be empty",
            ),
            422: VALIDATION_ERROR_RESPONSE,
            500: INTERNAL_ERROR_RESPONSE,
            502: _error_response_doc(
                description="Answer provider request failed.",
                error="upstream_error",
                message="Answer backend request failed: RequestError.",
            ),
            503: _error_response_doc(
                description="Answer agent is unavailable.",
                error="agent_unavailable",
                message="OPENAI_API_KEY missing",
            ),
        },
        tags=["pipeline"],
    )
    async def answer_question(
        payload: AnswerRequest = Body(..., openapi_examples=ANSWER_REQUEST_EXAMPLES),
    ) -> AnswerResponse:
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
