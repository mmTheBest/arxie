"""FastAPI app exposing the RA pipeline as REST endpoints."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import Body, FastAPI

from ra.agents.lit_review_agent import LitReviewAgent
from ra.agents.research_agent import ResearchAgent
from ra.api.errors import RAAPIError, register_exception_handlers
from ra.api.models import (
    AnswerRequest,
    AnswerResponse,
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    LitReviewRequest,
    LitReviewResponse,
    PaperResponse,
    RetrieveBatchItemResponse,
    RetrieveBatchRequest,
    RetrieveBatchResponse,
    RetrieveRequest,
    RetrieveResponse,
    SearchBatchItemResponse,
    SearchBatchRequest,
    SearchBatchResponse,
    SearchRequest,
    SearchResponse,
)
from ra.retrieval.unified import UnifiedRetriever
from ra.utils.logging_config import configure_logging_from_env

logger = logging.getLogger(__name__)

RetrieverFactory = Callable[[], UnifiedRetriever]
AgentFactory = Callable[..., ResearchAgent]
LitReviewAgentFactory = Callable[[], LitReviewAgent]

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
            "query": "What are the main limitations of retrieval-augmented generation?",
            "deep": False,
        },
    },
    "deep_research_question": {
        "summary": "Enable deep research mode",
        "value": {
            "query": "Compare LoRA and QLoRA training trade-offs with supporting evidence.",
            "deep": True,
        },
    }
}

CHAT_REQUEST_EXAMPLES = {
    "follow_up_turn": {
        "summary": "Stateful chat follow-up",
        "value": {
            "query": "Can you compare this with the previous approach?",
            "session_id": "session-1",
        },
    }
}

LIT_REVIEW_REQUEST_EXAMPLES = {
    "topic_synthesis": {
        "summary": "Generate a structured literature review",
        "value": {
            "topic": "graph neural networks for molecular property prediction",
            "max_papers": 20,
        },
    }
}

SEARCH_BATCH_REQUEST_EXAMPLES = {
    "multi_query_search": {
        "summary": "Execute multiple searches in one request",
        "value": {
            "requests": [
                {
                    "query": "retrieval augmented generation benchmark survey",
                    "limit": 5,
                    "source": "semantic_scholar",
                },
                {
                    "query": "long-context transformer memory mechanisms",
                    "limit": 5,
                    "source": "both",
                },
            ],
            "max_concurrency": 4,
        },
    }
}

RETRIEVE_BATCH_REQUEST_EXAMPLES = {
    "multi_identifier_lookup": {
        "summary": "Resolve multiple paper identifiers",
        "value": {
            "requests": [
                {"identifier": "10.5555/3295222.3295349"},
                {"identifier": "1706.03762"},
            ],
            "max_concurrency": 8,
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


def _default_agent_factory(*, deep_search: bool = False) -> ResearchAgent:
    return ResearchAgent(deep_search=deep_search)


def _default_lit_review_agent_factory() -> LitReviewAgent:
    return LitReviewAgent()


def _sources_for_request(source: str) -> tuple[str, ...]:
    if source == "both":
        return ("semantic_scholar", "arxiv")
    return (source,)


def _factory_supports_deep_search(factory: AgentFactory) -> bool:
    return _callable_supports_kwarg(factory, "deep_search")


def _callable_supports_kwarg(fn: Any, name: str) -> bool:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False

    params = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return True

    deep_param = params.get(name)
    if deep_param is None:
        return False

    return deep_param.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def _create_agent_from_factory(factory: AgentFactory, *, deep_search: bool) -> Any:
    if _factory_supports_deep_search(factory):
        return factory(deep_search=deep_search)
    return factory()


async def _close_resource(resource: Any) -> None:
    close = getattr(resource, "close", None)
    if close is None:
        return
    result = close()
    if inspect.isawaitable(result):
        await result


def _get_or_create_shared_retriever(app: FastAPI) -> Any:
    retriever = getattr(app.state, "retriever", None)
    if retriever is None:
        retriever = app.state.retriever_factory()
        app.state.retriever = retriever
    return retriever


async def _run_search_batch(
    retriever: Any,
    jobs: list[tuple[str, int, tuple[str, ...]]],
    *,
    max_concurrency: int,
) -> list[list[Any]]:
    search_batch = getattr(retriever, "search_batch", None)
    if callable(search_batch):
        return await search_batch(jobs, max_concurrency=max_concurrency)

    semaphore = asyncio.Semaphore(max_concurrency)
    results: list[list[Any] | None] = [None] * len(jobs)

    async def _run(index: int, job: tuple[str, int, tuple[str, ...]]) -> None:
        query, limit, sources = job
        async with semaphore:
            results[index] = await retriever.search(query=query, limit=limit, sources=sources)

    await asyncio.gather(*(_run(i, job) for i, job in enumerate(jobs)))
    return [batch if batch is not None else [] for batch in results]


async def _run_retrieve_batch(
    retriever: Any,
    identifiers: list[str],
    *,
    max_concurrency: int,
) -> list[Any]:
    get_papers_batch = getattr(retriever, "get_papers_batch", None)
    if callable(get_papers_batch):
        return await get_papers_batch(identifiers, max_concurrency=max_concurrency)

    semaphore = asyncio.Semaphore(max_concurrency)
    results: list[Any] = [None] * len(identifiers)

    async def _run(index: int, identifier: str) -> None:
        async with semaphore:
            results[index] = await retriever.get_paper(identifier)

    await asyncio.gather(*(_run(i, identifier) for i, identifier in enumerate(identifiers)))
    return results


def create_app(
    *,
    retriever_factory: RetrieverFactory | None = None,
    agent_factory: AgentFactory | None = None,
    lit_review_agent_factory: LitReviewAgentFactory | None = None,
) -> FastAPI:
    """Create the FastAPI app instance.

    Factories are injectable to make endpoint behavior easy to unit test.
    """
    configure_logging_from_env()

    retriever_factory = retriever_factory or _default_retriever_factory
    agent_factory = agent_factory or _default_agent_factory
    lit_review_agent_factory = lit_review_agent_factory or _default_lit_review_agent_factory

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.retriever = retriever_factory()
        app.state.chat_agents = {}
        try:
            yield
        finally:
            chat_agents = getattr(app.state, "chat_agents", {})
            if isinstance(chat_agents, dict):
                for agent in chat_agents.values():
                    await _close_resource(agent)
            await _close_resource(getattr(app.state, "retriever", None))

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
        lifespan=lifespan,
    )
    app.state.retriever_factory = retriever_factory
    app.state.agent_factory = agent_factory
    app.state.lit_review_agent_factory = lit_review_agent_factory
    app.state.chat_agents = {}

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
        retriever = _get_or_create_shared_retriever(app)
        try:
            papers = await retriever.search(
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
        "/search/batch",
        response_model=SearchBatchResponse,
        summary="Batch Search Academic Papers",
        description=(
            "Executes multiple paper searches asynchronously in one request. "
            "Useful for high-throughput query fan-out from API clients."
        ),
        response_description="Ordered search results for each query in the batch.",
        responses={
            400: _error_response_doc(
                description="Invalid batch search request.",
                error="invalid_input",
                message="max_concurrency must be an integer.",
            ),
            422: VALIDATION_ERROR_RESPONSE,
            500: INTERNAL_ERROR_RESPONSE,
            502: _error_response_doc(
                description="Batch search provider request failed.",
                error="upstream_error",
                message="Batch search backend request failed: RequestError.",
            ),
        },
        tags=["pipeline"],
    )
    async def search_papers_batch(
        payload: SearchBatchRequest = Body(..., openapi_examples=SEARCH_BATCH_REQUEST_EXAMPLES),
    ) -> SearchBatchResponse:
        retriever = _get_or_create_shared_retriever(app)
        jobs = [
            (request.query, request.limit, _sources_for_request(request.source))
            for request in payload.requests
        ]
        try:
            batch_results = await _run_search_batch(
                retriever,
                jobs,
                max_concurrency=payload.max_concurrency,
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
                message=f"Batch search backend request failed: {type(exc).__name__}.",
            ) from exc

        items = [
            SearchBatchItemResponse(
                query=request.query,
                count=len(papers),
                results=[PaperResponse.from_paper(paper) for paper in papers],
            )
            for request, papers in zip(payload.requests, batch_results, strict=False)
        ]
        return SearchBatchResponse(count=len(items), results=items)

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
        retriever = _get_or_create_shared_retriever(app)
        try:
            paper = await retriever.get_paper(payload.identifier)
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
        "/retrieve/batch",
        response_model=RetrieveBatchResponse,
        summary="Batch Retrieve Paper Metadata",
        description=(
            "Fetches multiple papers by identifier asynchronously in one request. "
            "Results preserve request order and include null papers when not found."
        ),
        response_description="Ordered retrieval results for each requested identifier.",
        responses={
            400: _error_response_doc(
                description="Invalid batch retrieve request.",
                error="invalid_input",
                message="max_concurrency must be an integer.",
            ),
            422: VALIDATION_ERROR_RESPONSE,
            500: INTERNAL_ERROR_RESPONSE,
            502: _error_response_doc(
                description="Batch retrieve provider request failed.",
                error="upstream_error",
                message="Batch retrieve backend request failed: RequestError.",
            ),
        },
        tags=["pipeline"],
    )
    async def retrieve_papers_batch(
        payload: RetrieveBatchRequest = Body(..., openapi_examples=RETRIEVE_BATCH_REQUEST_EXAMPLES),
    ) -> RetrieveBatchResponse:
        retriever = _get_or_create_shared_retriever(app)
        identifiers = [request.identifier for request in payload.requests]

        try:
            papers = await _run_retrieve_batch(
                retriever,
                identifiers,
                max_concurrency=payload.max_concurrency,
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
                message=f"Batch retrieve backend request failed: {type(exc).__name__}.",
            ) from exc

        items = [
            RetrieveBatchItemResponse(
                identifier=identifier,
                paper=PaperResponse.from_paper(paper) if paper else None,
            )
            for identifier, paper in zip(identifiers, papers, strict=False)
        ]
        found = sum(1 for item in items if item.paper is not None)
        return RetrieveBatchResponse(count=len(items), found=found, results=items)

    async def _run_query_pipeline(payload: AnswerRequest) -> AnswerResponse:
        try:
            agent = _create_agent_from_factory(app.state.agent_factory, deep_search=payload.deep)
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

    async def _run_agent_arun(
        agent: Any,
        *,
        query: str,
        session_id: str | None = None,
    ) -> str:
        if session_id and _callable_supports_kwarg(agent.arun, "session_id"):
            return await agent.arun(query, session_id=session_id)
        return await agent.arun(query)

    async def _run_chat_pipeline(payload: ChatRequest) -> ChatResponse:
        chat_agents: dict[str, Any] = app.state.chat_agents
        session_id = payload.session_id
        agent = chat_agents.get(session_id)
        if agent is None:
            try:
                agent = _create_agent_from_factory(app.state.agent_factory, deep_search=False)
            except ValueError as exc:
                raise RAAPIError(
                    status_code=503,
                    error="agent_unavailable",
                    message=str(exc),
                ) from exc
            except Exception as exc:
                logger.exception("Failed to initialize chat ResearchAgent", exc_info=exc)
                raise RAAPIError(
                    status_code=500,
                    error="internal_error",
                    message="Failed to initialize chat agent.",
                ) from exc
            chat_agents[session_id] = agent

        try:
            answer = await _run_agent_arun(agent, query=payload.query, session_id=session_id)
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
                message=f"Chat backend request failed: {type(exc).__name__}.",
            ) from exc

        return ChatResponse(
            query=payload.query,
            session_id=session_id,
            answer=answer,
        )

    async def _run_lit_review_pipeline(payload: LitReviewRequest) -> LitReviewResponse:
        try:
            agent = app.state.lit_review_agent_factory()
        except ValueError as exc:
            raise RAAPIError(
                status_code=503,
                error="agent_unavailable",
                message=str(exc),
            ) from exc
        except Exception as exc:
            logger.exception("Failed to initialize LitReviewAgent", exc_info=exc)
            raise RAAPIError(
                status_code=500,
                error="internal_error",
                message="Failed to initialize lit-review agent.",
            ) from exc

        try:
            if _callable_supports_kwarg(agent.arun, "max_papers"):
                review = await agent.arun(payload.topic, max_papers=payload.max_papers)
            else:
                review = await agent.arun(payload.topic)
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
                message=f"Lit review backend request failed: {type(exc).__name__}.",
            ) from exc

        return LitReviewResponse(topic=payload.topic, review=review)

    @app.post(
        "/query",
        response_model=AnswerResponse,
        summary="Run Research Query",
        description=(
            "Runs the research agent pipeline to generate a structured Markdown "
            "answer for a research question."
        ),
        response_description="Structured answer produced by the research agent.",
        responses={
            400: _error_response_doc(
                description="Invalid query request.",
                error="invalid_input",
                message="query must not be empty",
            ),
            422: VALIDATION_ERROR_RESPONSE,
            500: INTERNAL_ERROR_RESPONSE,
            502: _error_response_doc(
                description="Query provider request failed.",
                error="upstream_error",
                message="Query backend request failed: RequestError.",
            ),
            503: _error_response_doc(
                description="Query agent is unavailable.",
                error="agent_unavailable",
                message="OPENAI_API_KEY missing",
            ),
        },
        tags=["pipeline"],
    )
    async def query_question(
        payload: AnswerRequest = Body(..., openapi_examples=ANSWER_REQUEST_EXAMPLES),
    ) -> AnswerResponse:
        return await _run_query_pipeline(payload)

    @app.post(
        "/api/chat",
        response_model=ChatResponse,
        summary="Conversational Research Chat",
        description=(
            "Runs the research agent in stateful conversational mode. "
            "Use `session_id` to continue a multi-turn conversation."
        ),
        response_description="Stateful chat response with the same session ID.",
        responses={
            400: _error_response_doc(
                description="Invalid chat request.",
                error="invalid_input",
                message="query must not be empty",
            ),
            422: VALIDATION_ERROR_RESPONSE,
            500: INTERNAL_ERROR_RESPONSE,
            502: _error_response_doc(
                description="Chat provider request failed.",
                error="upstream_error",
                message="Chat backend request failed: RequestError.",
            ),
            503: _error_response_doc(
                description="Chat agent is unavailable.",
                error="agent_unavailable",
                message="OPENAI_API_KEY missing",
            ),
        },
        tags=["pipeline"],
    )
    async def chat(
        payload: ChatRequest = Body(..., openapi_examples=CHAT_REQUEST_EXAMPLES),
    ) -> ChatResponse:
        return await _run_chat_pipeline(payload)

    @app.post(
        "/api/lit-review",
        response_model=LitReviewResponse,
        summary="Generate Structured Literature Review",
        description=(
            "Runs the literature review mode to retrieve papers, cluster them by theme, "
            "and produce a structured synthesis."
        ),
        response_description="Structured literature review generated by the lit-review agent.",
        responses={
            400: _error_response_doc(
                description="Invalid lit-review request.",
                error="invalid_input",
                message="topic must not be empty",
            ),
            422: VALIDATION_ERROR_RESPONSE,
            500: INTERNAL_ERROR_RESPONSE,
            502: _error_response_doc(
                description="Lit-review provider request failed.",
                error="upstream_error",
                message="Lit review backend request failed: RequestError.",
            ),
            503: _error_response_doc(
                description="Lit-review agent is unavailable.",
                error="agent_unavailable",
                message="OPENAI_API_KEY missing",
            ),
        },
        tags=["pipeline"],
    )
    async def lit_review(
        payload: LitReviewRequest = Body(..., openapi_examples=LIT_REVIEW_REQUEST_EXAMPLES),
    ) -> LitReviewResponse:
        return await _run_lit_review_pipeline(payload)

    @app.post(
        "/answer",
        response_model=AnswerResponse,
        summary="Generate Literature-Grounded Answer",
        description=(
            "Alias for `/query`. Runs the research agent pipeline to generate a "
            "structured Markdown answer for a research question."
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
        deprecated=True,
    )
    async def answer_question(
        payload: AnswerRequest = Body(..., openapi_examples=ANSWER_REQUEST_EXAMPLES),
    ) -> AnswerResponse:
        return await _run_query_pipeline(payload)

    return app


app = create_app()
