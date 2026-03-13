"""FastAPI app exposing the RA pipeline as REST endpoints."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import httpx
from fastapi import Body, FastAPI, Path
from fastapi.responses import RedirectResponse

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
from ra.api.proposal_models import (
    ProposalArtifactDependencyCreateRequest,
    ProposalArtifactDependencyResponse,
    ProposalArtifactDependencySnapshotResponse,
    ProposalArtifactEditRequest,
    ProposalArtifactEditResponse,
    ProposalArtifactEditSourceResponse,
    ProposalArtifactNodeResponse,
    ProposalArtifactNodeUpsertRequest,
    ProposalEvidenceQueryRequest,
    ProposalEvidenceQueryResponse,
    ProposalConversationMessageCreateRequest,
    ProposalConversationMessageResponse,
    ProposalConversationThreadResponse,
    ProposalEvidenceInspectorResponse,
    ProposalBranchCompareRequest,
    ProposalBranchCompareResponse,
    ProposalBranchCreateRequest,
    ProposalBranchListResponse,
    ProposalBranchResponse,
    ProposalSessionCreateRequest,
    ProposalSessionResponse,
    ProposalStageAdvanceRequest,
    ProposalStageUpdateRequest,
)
from ra.proposal import (
    ArtifactNodeNotFoundError,
    ArtifactSyncManager,
    BranchAlreadyExistsError,
    BranchNotFoundError,
    EvidenceMapper,
    HypothesisBranchManager,
    InMemoryProposalSessionStore,
    ProposalArtifact,
    ProposalSessionService,
    ProposalStage,
    ProposalStageEngine,
    ProvenanceNotFoundError,
    SessionAlreadyExistsError,
    SessionNotFoundError,
    SessionVersionConflictError,
    StageTransitionError,
)
from ra.retrieval.unified import UnifiedRetriever
from ra.utils.logging_config import configure_logging_from_env

logger = logging.getLogger(__name__)

RetrieverFactory = Callable[[], UnifiedRetriever]
AgentFactory = Callable[..., ResearchAgent]
LitReviewAgentFactory = Callable[[], LitReviewAgent]
ProposalSessionServiceFactory = Callable[[], ProposalSessionService]
BranchManagerFactory = Callable[[], HypothesisBranchManager]
EvidenceMapperFactory = Callable[[], EvidenceMapper]
ArtifactSyncManagerFactory = Callable[[], ArtifactSyncManager]

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
    {
        "name": "proposal",
        "description": "Stage-gated proposal workflow endpoints for session state management.",
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

PROPOSAL_CREATE_SESSION_REQUEST_EXAMPLES = {
    "new_session": {
        "summary": "Create a proposal workflow session",
        "value": {"session_id": "proposal-session-1"},
    }
}

PROPOSAL_UPDATE_STAGE_REQUEST_EXAMPLES = {
    "stage_payload_patch": {
        "summary": "Merge payload fields into a stage snapshot",
        "value": {
            "expected_version": 2,
            "payload": {
                "supporting_evidence": ["paper-1", "paper-2"],
                "contradicting_evidence": ["paper-3"],
                "landscape_summary": "Consensus exists, but methods are heterogeneous.",
            },
        },
    }
}

PROPOSAL_ADVANCE_STAGE_REQUEST_EXAMPLES = {
    "next_stage": {
        "summary": "Advance to the next stage",
        "value": {"expected_version": 3},
    }
}

PROPOSAL_BRANCH_CREATE_REQUEST_EXAMPLES = {
    "new_branch": {
        "summary": "Create a branch",
        "value": {
            "session_id": "proposal-session-1",
            "branch_id": "branch-a",
            "name": "Primary mechanism",
            "hypothesis": "Mechanism A drives outcome B.",
            "scorecard": {
                "evidence_support": 0.8,
                "feasibility": 0.7,
                "risk": 0.3,
                "impact": 0.9,
            },
        },
    },
    "fork_branch": {
        "summary": "Fork an existing branch",
        "value": {
            "session_id": "proposal-session-1",
            "branch_id": "branch-b",
            "parent_branch_id": "branch-a",
            "name": "Alternative mechanism",
            "hypothesis": "Mechanism C drives outcome B.",
            "scorecard": {
                "evidence_support": 0.6,
                "feasibility": 0.8,
                "risk": 0.4,
                "impact": 0.8,
            },
        },
    },
}

PROPOSAL_BRANCH_COMPARE_REQUEST_EXAMPLES = {
    "compare_two_branches": {
        "summary": "Compare two branches",
        "value": {
            "session_id": "proposal-session-1",
            "branch_ids": ["branch-a", "branch-b"],
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


def _default_proposal_session_service_factory() -> ProposalSessionService:
    return ProposalSessionService(
        store=InMemoryProposalSessionStore(),
        stage_engine=ProposalStageEngine(),
    )


def _default_branch_manager_factory() -> HypothesisBranchManager:
    return HypothesisBranchManager()


def _default_evidence_mapper_factory() -> EvidenceMapper:
    return EvidenceMapper()


def _default_artifact_sync_manager_factory() -> ArtifactSyncManager:
    return ArtifactSyncManager()


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


def _get_or_create_proposal_session_service(app: FastAPI) -> ProposalSessionService:
    service = getattr(app.state, "proposal_session_service", None)
    if service is None:
        service = app.state.proposal_session_service_factory()
        app.state.proposal_session_service = service
    return service


def _get_or_create_branch_manager(app: FastAPI) -> HypothesisBranchManager:
    manager = getattr(app.state, "branch_manager", None)
    if manager is None:
        manager = app.state.branch_manager_factory()
        app.state.branch_manager = manager
    return manager


def _get_or_create_evidence_mapper(app: FastAPI) -> EvidenceMapper:
    mapper = getattr(app.state, "evidence_mapper", None)
    if mapper is None:
        mapper = app.state.evidence_mapper_factory()
        app.state.evidence_mapper = mapper
    return mapper


def _get_or_create_proposal_conversations(
    app: FastAPI,
) -> dict[str, list[ProposalConversationMessageResponse]]:
    conversations = getattr(app.state, "proposal_conversations", None)
    if not isinstance(conversations, dict):
        conversations = {}
        app.state.proposal_conversations = conversations
    return conversations


def _get_or_create_artifact_sync_manager(app: FastAPI) -> ArtifactSyncManager:
    manager = getattr(app.state, "artifact_sync_manager", None)
    if manager is None:
        manager = app.state.artifact_sync_manager_factory()
        app.state.artifact_sync_manager = manager
    return manager


def _version_conflict_details(exc: SessionVersionConflictError) -> list[dict[str, Any]]:
    return [
        {
            "session_id": exc.session_id,
            "expected_version": exc.expected_version,
            "current_version": exc.current_version,
        }
    ]


def _stage_transition_details(exc: StageTransitionError) -> list[dict[str, Any]]:
    details: dict[str, Any] = {
        "reason": exc.reason.value,
        "from_stage": exc.from_stage.value,
        "to_stage": exc.to_stage.value,
    }
    if exc.allowed_next_stage is not None:
        details["allowed_next_stage"] = exc.allowed_next_stage.value
    if exc.missing_fields:
        details["missing_fields"] = list(exc.missing_fields)
    return [details]


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
    proposal_session_service_factory: ProposalSessionServiceFactory | None = None,
    branch_manager_factory: BranchManagerFactory | None = None,
    evidence_mapper_factory: EvidenceMapperFactory | None = None,
    artifact_sync_manager_factory: ArtifactSyncManagerFactory | None = None,
) -> FastAPI:
    """Create the FastAPI app instance.

    Factories are injectable to make endpoint behavior easy to unit test.
    """
    configure_logging_from_env()

    retriever_factory = retriever_factory or _default_retriever_factory
    agent_factory = agent_factory or _default_agent_factory
    lit_review_agent_factory = lit_review_agent_factory or _default_lit_review_agent_factory
    proposal_session_service_factory = (
        proposal_session_service_factory or _default_proposal_session_service_factory
    )
    branch_manager_factory = branch_manager_factory or _default_branch_manager_factory
    evidence_mapper_factory = evidence_mapper_factory or _default_evidence_mapper_factory
    artifact_sync_manager_factory = (
        artifact_sync_manager_factory or _default_artifact_sync_manager_factory
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.retriever = retriever_factory()
        app.state.chat_agents = {}
        app.state.proposal_session_service = proposal_session_service_factory()
        app.state.branch_manager = branch_manager_factory()
        app.state.evidence_mapper = evidence_mapper_factory()
        app.state.artifact_sync_manager = artifact_sync_manager_factory()
        app.state.proposal_conversations = {}
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
    app.state.proposal_session_service_factory = proposal_session_service_factory
    app.state.branch_manager_factory = branch_manager_factory
    app.state.evidence_mapper_factory = evidence_mapper_factory
    app.state.artifact_sync_manager_factory = artifact_sync_manager_factory
    app.state.chat_agents = {}
    app.state.proposal_session_service = proposal_session_service_factory()
    app.state.branch_manager = branch_manager_factory()
    app.state.evidence_mapper = evidence_mapper_factory()
    app.state.artifact_sync_manager = artifact_sync_manager_factory()
    app.state.proposal_conversations = {}

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
        "/api/proposal/sessions",
        response_model=ProposalSessionResponse,
        status_code=201,
        summary="Create Proposal Session",
        description="Creates a proposal workflow session with deterministic initial stage state.",
        response_description="Created proposal session snapshot.",
        responses={
            409: _error_response_doc(
                description="Session already exists.",
                error="session_exists",
                message="Session 'proposal-session-1' already exists.",
                details=[{"session_id": "proposal-session-1"}],
            ),
            422: VALIDATION_ERROR_RESPONSE,
            500: INTERNAL_ERROR_RESPONSE,
        },
        tags=["proposal"],
    )
    async def create_proposal_session(
        payload: ProposalSessionCreateRequest = Body(
            ...,
            openapi_examples=PROPOSAL_CREATE_SESSION_REQUEST_EXAMPLES,
        ),
    ) -> ProposalSessionResponse:
        service = _get_or_create_proposal_session_service(app)
        try:
            snapshot = service.create_session(payload.session_id)
        except SessionAlreadyExistsError as exc:
            raise RAAPIError(
                status_code=409,
                error="session_exists",
                message=str(exc),
                details=[{"session_id": exc.session_id}],
            ) from exc
        except ValueError as exc:
            raise RAAPIError(
                status_code=400,
                error="invalid_input",
                message=str(exc),
            ) from exc

        return ProposalSessionResponse.from_snapshot(snapshot)

    @app.get(
        "/api/proposal/sessions/{session_id}",
        response_model=ProposalSessionResponse,
        summary="Get Proposal Session",
        description="Returns the latest persisted snapshot for a proposal workflow session.",
        response_description="Current proposal session snapshot.",
        responses={
            400: _error_response_doc(
                description="Invalid session identifier.",
                error="invalid_input",
                message="session_id must not be empty",
            ),
            404: _error_response_doc(
                description="Session not found.",
                error="session_not_found",
                message="Session 'proposal-session-1' was not found.",
            ),
            500: INTERNAL_ERROR_RESPONSE,
        },
        tags=["proposal"],
    )
    async def get_proposal_session(
        session_id: str = Path(
            ...,
            min_length=1,
            max_length=128,
            description="Proposal session identifier.",
        ),
    ) -> ProposalSessionResponse:
        service = _get_or_create_proposal_session_service(app)
        try:
            snapshot = service.get_session(session_id)
        except SessionNotFoundError as exc:
            raise RAAPIError(
                status_code=404,
                error="session_not_found",
                message=str(exc),
            ) from exc
        except ValueError as exc:
            raise RAAPIError(
                status_code=400,
                error="invalid_input",
                message=str(exc),
            ) from exc

        return ProposalSessionResponse.from_snapshot(snapshot)

    @app.patch(
        "/api/proposal/sessions/{session_id}/stages/{stage}",
        response_model=ProposalSessionResponse,
        summary="Update Proposal Stage Payload",
        description=(
            "Merges payload fields into a target stage snapshot "
            "using optimistic concurrency checks."
        ),
        response_description="Updated proposal session snapshot after stage payload merge.",
        responses={
            400: _error_response_doc(
                description="Invalid stage update request.",
                error="invalid_input",
                message="session_id must not be empty",
            ),
            404: _error_response_doc(
                description="Session not found.",
                error="session_not_found",
                message="Session 'proposal-session-1' was not found.",
            ),
            409: _error_response_doc(
                description="Session version conflict.",
                error="version_conflict",
                message="Version conflict for session 'proposal-session-1': expected 2, current 3.",
                details=[
                    {
                        "session_id": "proposal-session-1",
                        "expected_version": 2,
                        "current_version": 3,
                    }
                ],
            ),
            422: VALIDATION_ERROR_RESPONSE,
            500: INTERNAL_ERROR_RESPONSE,
        },
        tags=["proposal"],
    )
    async def update_proposal_stage(
        session_id: str = Path(
            ...,
            min_length=1,
            max_length=128,
            description="Proposal session identifier.",
        ),
        stage: ProposalStage = Path(
            ...,
            description="Canonical stage identifier to update.",
        ),
        payload: ProposalStageUpdateRequest = Body(
            ...,
            openapi_examples=PROPOSAL_UPDATE_STAGE_REQUEST_EXAMPLES,
        ),
    ) -> ProposalSessionResponse:
        service = _get_or_create_proposal_session_service(app)
        try:
            snapshot = service.update_stage_payload(
                session_id,
                stage,
                payload.payload,
                expected_version=payload.expected_version,
            )
        except SessionNotFoundError as exc:
            raise RAAPIError(
                status_code=404,
                error="session_not_found",
                message=str(exc),
            ) from exc
        except SessionVersionConflictError as exc:
            raise RAAPIError(
                status_code=409,
                error="version_conflict",
                message=str(exc),
                details=_version_conflict_details(exc),
            ) from exc
        except ValueError as exc:
            raise RAAPIError(
                status_code=400,
                error="invalid_input",
                message=str(exc),
            ) from exc

        return ProposalSessionResponse.from_snapshot(snapshot)

    @app.post(
        "/api/proposal/sessions/{session_id}/advance",
        response_model=ProposalSessionResponse,
        summary="Advance Proposal Session Stage",
        description=(
            "Advances the session to the next stage when the current stage is complete and "
            "the session version matches `expected_version`."
        ),
        response_description="Updated proposal session snapshot after successful stage transition.",
        responses={
            400: _error_response_doc(
                description="Invalid stage advancement request.",
                error="invalid_input",
                message="session_id must not be empty",
            ),
            404: _error_response_doc(
                description="Session not found.",
                error="session_not_found",
                message="Session 'proposal-session-1' was not found.",
            ),
            409: _error_response_doc(
                description="Stage transition rejected.",
                error="stage_transition_rejected",
                message="Stage 'idea_intake' is incomplete. Fill missing fields: problem.",
                details=[
                    {
                        "reason": "incomplete_stage",
                        "from_stage": "idea_intake",
                        "to_stage": "logic_refinement",
                        "missing_fields": ["problem"],
                    }
                ],
            ),
            422: VALIDATION_ERROR_RESPONSE,
            500: INTERNAL_ERROR_RESPONSE,
        },
        tags=["proposal"],
    )
    async def advance_proposal_stage(
        session_id: str = Path(
            ...,
            min_length=1,
            max_length=128,
            description="Proposal session identifier.",
        ),
        payload: ProposalStageAdvanceRequest = Body(
            ...,
            openapi_examples=PROPOSAL_ADVANCE_STAGE_REQUEST_EXAMPLES,
        ),
    ) -> ProposalSessionResponse:
        service = _get_or_create_proposal_session_service(app)
        try:
            snapshot = service.advance_stage(
                session_id,
                expected_version=payload.expected_version,
            )
        except SessionNotFoundError as exc:
            raise RAAPIError(
                status_code=404,
                error="session_not_found",
                message=str(exc),
            ) from exc
        except SessionVersionConflictError as exc:
            raise RAAPIError(
                status_code=409,
                error="version_conflict",
                message=str(exc),
                details=_version_conflict_details(exc),
            ) from exc
        except StageTransitionError as exc:
            raise RAAPIError(
                status_code=409,
                error="stage_transition_rejected",
                message=str(exc),
                details=_stage_transition_details(exc),
            ) from exc
        except ValueError as exc:
            raise RAAPIError(
                status_code=400,
                error="invalid_input",
                message=str(exc),
            ) from exc

        return ProposalSessionResponse.from_snapshot(snapshot)

    @app.post(
        "/api/proposal/evidence/query",
        response_model=ProposalEvidenceQueryResponse,
        summary="Query Proposal Evidence Map",
        description=(
            "Buckets provided papers into supporting, contradicting, and adjacent evidence "
            "for a proposal claim with deterministic relevance scoring."
        ),
        response_description="Bucketed evidence map with representative IDs and landscape summary.",
        responses={
            400: _error_response_doc(
                description="Invalid evidence mapping request.",
                error="invalid_input",
                message="claim must not be empty",
            ),
            422: VALIDATION_ERROR_RESPONSE,
            500: INTERNAL_ERROR_RESPONSE,
        },
        tags=["proposal"],
    )
    async def query_proposal_evidence(
        payload: ProposalEvidenceQueryRequest = Body(...),
    ) -> ProposalEvidenceQueryResponse:
        mapper = _get_or_create_evidence_mapper(app)
        try:
            mapped = mapper.map_evidence(
                payload.claim,
                [paper.to_domain() for paper in payload.papers],
                pinned_paper_ids=set(payload.pinned_paper_ids),
            )
        except ValueError as exc:
            raise RAAPIError(
                status_code=400,
                error="invalid_input",
                message=str(exc),
            ) from exc

        return ProposalEvidenceQueryResponse.from_domain(mapped)

    @app.put(
        "/api/proposal/artifacts/{session_id}/nodes/{artifact}/{node_id}",
        response_model=ProposalArtifactNodeResponse,
        summary="Upsert Proposal Artifact Node",
        description=(
            "Creates or updates an artifact node for cross-artifact synchronization across "
            "logical tree, evidence map, and hypothesis tree surfaces."
        ),
        response_description="Upserted artifact node snapshot.",
        responses={
            400: _error_response_doc(
                description="Invalid artifact node payload.",
                error="invalid_input",
                message="content must not be empty",
            ),
            422: VALIDATION_ERROR_RESPONSE,
            500: INTERNAL_ERROR_RESPONSE,
        },
        tags=["proposal"],
    )
    async def upsert_proposal_artifact_node(
        session_id: str = Path(
            ...,
            min_length=1,
            max_length=128,
            description="Proposal session identifier.",
        ),
        artifact: ProposalArtifact = Path(
            ...,
            description="Artifact identifier (`logical_tree`, `evidence_map`, `hypothesis_tree`).",
        ),
        node_id: str = Path(
            ...,
            min_length=1,
            max_length=128,
            description="Artifact node identifier.",
        ),
        payload: ProposalArtifactNodeUpsertRequest = Body(...),
    ) -> ProposalArtifactNodeResponse:
        manager = _get_or_create_artifact_sync_manager(app)
        try:
            node = manager.upsert_node(
                session_id=session_id,
                artifact=artifact,
                node_id=node_id,
                content=payload.content,
                provenance_link=payload.provenance_link,
            )
        except ValueError as exc:
            raise RAAPIError(
                status_code=400,
                error="invalid_input",
                message=str(exc),
            ) from exc
        return ProposalArtifactNodeResponse.from_domain(node)

    @app.post(
        "/api/proposal/artifacts/{session_id}/dependencies",
        response_model=ProposalArtifactDependencyResponse,
        status_code=201,
        summary="Create Proposal Artifact Dependency",
        description=(
            "Creates a directed dependency edge used to propagate stale markers when "
            "an upstream artifact node is edited."
        ),
        response_description="Created dependency edge.",
        responses={
            400: _error_response_doc(
                description="Invalid dependency payload.",
                error="invalid_input",
                message="node_id must not be empty",
            ),
            404: _error_response_doc(
                description="Artifact node not found.",
                error="artifact_node_not_found",
                message=(
                    "Artifact node 'logical_tree:logic-1' was not found in session "
                    "'proposal-session-1'."
                ),
            ),
            422: VALIDATION_ERROR_RESPONSE,
            500: INTERNAL_ERROR_RESPONSE,
        },
        tags=["proposal"],
    )
    async def create_proposal_artifact_dependency(
        session_id: str = Path(
            ...,
            min_length=1,
            max_length=128,
            description="Proposal session identifier.",
        ),
        payload: ProposalArtifactDependencyCreateRequest = Body(...),
    ) -> ProposalArtifactDependencyResponse:
        manager = _get_or_create_artifact_sync_manager(app)
        try:
            manager.add_dependency(
                session_id=session_id,
                upstream_artifact=payload.upstream_artifact,
                upstream_node_id=payload.upstream_node_id,
                downstream_artifact=payload.downstream_artifact,
                downstream_node_id=payload.downstream_node_id,
            )
        except ArtifactNodeNotFoundError as exc:
            raise RAAPIError(
                status_code=404,
                error="artifact_node_not_found",
                message=str(exc),
            ) from exc
        except ValueError as exc:
            raise RAAPIError(
                status_code=400,
                error="invalid_input",
                message=str(exc),
            ) from exc

        return ProposalArtifactDependencyResponse(
            session_id=session_id,
            upstream_artifact=payload.upstream_artifact,
            upstream_node_id=payload.upstream_node_id,
            downstream_artifact=payload.downstream_artifact,
            downstream_node_id=payload.downstream_node_id,
        )

    @app.post(
        "/api/proposal/artifacts/{session_id}/edits",
        response_model=ProposalArtifactEditResponse,
        summary="Propagate Proposal Artifact Edit",
        description=(
            "Applies an artifact node edit and propagates stale markers to dependent "
            "nodes across synchronized artifact graphs."
        ),
        response_description="Edit source and impacted stale-node list.",
        responses={
            400: _error_response_doc(
                description="Invalid edit payload.",
                error="invalid_input",
                message="content must not be empty",
            ),
            404: _error_response_doc(
                description="Artifact node not found.",
                error="artifact_node_not_found",
                message=(
                    "Artifact node 'logical_tree:logic-1' was not found in session "
                    "'proposal-session-1'."
                ),
            ),
            422: VALIDATION_ERROR_RESPONSE,
            500: INTERNAL_ERROR_RESPONSE,
        },
        tags=["proposal"],
    )
    async def propagate_proposal_artifact_edit(
        session_id: str = Path(
            ...,
            min_length=1,
            max_length=128,
            description="Proposal session identifier.",
        ),
        payload: ProposalArtifactEditRequest = Body(...),
    ) -> ProposalArtifactEditResponse:
        manager = _get_or_create_artifact_sync_manager(app)
        try:
            source, impacted = manager.record_edit(
                session_id=session_id,
                artifact=payload.artifact,
                node_id=payload.node_id,
                content=payload.content,
            )
        except ArtifactNodeNotFoundError as exc:
            raise RAAPIError(
                status_code=404,
                error="artifact_node_not_found",
                message=str(exc),
            ) from exc
        except ValueError as exc:
            raise RAAPIError(
                status_code=400,
                error="invalid_input",
                message=str(exc),
            ) from exc

        return ProposalArtifactEditResponse(
            session_id=session_id,
            source=ProposalArtifactEditSourceResponse(
                artifact=source.artifact,
                node_id=source.node_id,
            ),
            impacted_count=len(impacted),
            impacted=[ProposalArtifactNodeResponse.from_domain(node) for node in impacted],
        )

    @app.get(
        "/api/proposal/artifacts/{session_id}/dependencies/{artifact}/{node_id}",
        response_model=ProposalArtifactDependencySnapshotResponse,
        summary="Get Proposal Artifact Dependency Snapshot",
        description=(
            "Returns transitive downstream dependencies for an artifact node with "
            "current stale-marker status."
        ),
        response_description="Downstream dependency snapshot.",
        responses={
            400: _error_response_doc(
                description="Invalid dependency lookup request.",
                error="invalid_input",
                message="node_id must not be empty",
            ),
            404: _error_response_doc(
                description="Artifact node not found.",
                error="artifact_node_not_found",
                message=(
                    "Artifact node 'logical_tree:logic-1' was not found in session "
                    "'proposal-session-1'."
                ),
            ),
            422: VALIDATION_ERROR_RESPONSE,
            500: INTERNAL_ERROR_RESPONSE,
        },
        tags=["proposal"],
    )
    async def get_proposal_artifact_dependencies(
        session_id: str = Path(
            ...,
            min_length=1,
            max_length=128,
            description="Proposal session identifier.",
        ),
        artifact: ProposalArtifact = Path(
            ...,
            description="Artifact identifier (`logical_tree`, `evidence_map`, `hypothesis_tree`).",
        ),
        node_id: str = Path(
            ...,
            min_length=1,
            max_length=128,
            description="Artifact node identifier.",
        ),
    ) -> ProposalArtifactDependencySnapshotResponse:
        manager = _get_or_create_artifact_sync_manager(app)
        try:
            downstream = manager.downstream_nodes(
                session_id=session_id,
                artifact=artifact,
                node_id=node_id,
            )
        except ArtifactNodeNotFoundError as exc:
            raise RAAPIError(
                status_code=404,
                error="artifact_node_not_found",
                message=str(exc),
            ) from exc
        except ValueError as exc:
            raise RAAPIError(
                status_code=400,
                error="invalid_input",
                message=str(exc),
            ) from exc

        return ProposalArtifactDependencySnapshotResponse(
            session_id=session_id,
            artifact=artifact,
            node_id=node_id,
            count=len(downstream),
            downstream=[ProposalArtifactNodeResponse.from_domain(node) for node in downstream],
        )

    @app.get(
        "/api/proposal/artifacts/{session_id}/provenance/{artifact}/{node_id}",
        summary="Open Proposal Artifact Provenance Link",
        description=(
            "Returns a temporary redirect to the source provenance link for the requested "
            "artifact node."
        ),
        response_description="307 redirect to provenance URL.",
        responses={
            307: {"description": "Redirected to provenance source."},
            400: _error_response_doc(
                description="Invalid provenance lookup request.",
                error="invalid_input",
                message="node_id must not be empty",
            ),
            404: _error_response_doc(
                description="Provenance link not found.",
                error="provenance_not_found",
                message=(
                    "Provenance link not found for artifact node 'evidence_map:paper-1' in "
                    "session 'proposal-session-1'."
                ),
            ),
            422: VALIDATION_ERROR_RESPONSE,
            500: INTERNAL_ERROR_RESPONSE,
        },
        tags=["proposal"],
    )
    async def open_proposal_artifact_provenance(
        session_id: str = Path(
            ...,
            min_length=1,
            max_length=128,
            description="Proposal session identifier.",
        ),
        artifact: ProposalArtifact = Path(
            ...,
            description="Artifact identifier (`logical_tree`, `evidence_map`, `hypothesis_tree`).",
        ),
        node_id: str = Path(
            ...,
            min_length=1,
            max_length=128,
            description="Artifact node identifier.",
        ),
    ) -> RedirectResponse:
        manager = _get_or_create_artifact_sync_manager(app)
        try:
            url = manager.provenance_link(
                session_id=session_id,
                artifact=artifact,
                node_id=node_id,
            )
        except ArtifactNodeNotFoundError as exc:
            raise RAAPIError(
                status_code=404,
                error="artifact_node_not_found",
                message=str(exc),
            ) from exc
        except ProvenanceNotFoundError as exc:
            raise RAAPIError(
                status_code=404,
                error="provenance_not_found",
                message=str(exc),
            ) from exc
        except ValueError as exc:
            raise RAAPIError(
                status_code=400,
                error="invalid_input",
                message=str(exc),
            ) from exc

        return RedirectResponse(url=url, status_code=307)

    @app.post(
        "/api/proposal/conversations/{session_id}/messages",
        response_model=ProposalConversationMessageResponse,
        status_code=201,
        summary="Create Proposal Conversation Message",
        description="Appends a message to a proposal session conversation thread.",
        response_description="Created conversation message.",
        responses={
            400: _error_response_doc(
                description="Invalid conversation message payload.",
                error="invalid_input",
                message="role must be one of: user, assistant, system",
            ),
            422: VALIDATION_ERROR_RESPONSE,
            500: INTERNAL_ERROR_RESPONSE,
        },
        tags=["proposal"],
    )
    async def create_proposal_conversation_message(
        session_id: str = Path(
            ...,
            min_length=1,
            max_length=128,
            description="Proposal session identifier.",
        ),
        payload: ProposalConversationMessageCreateRequest = Body(...),
    ) -> ProposalConversationMessageResponse:
        role = str(payload.role or "").strip().lower()
        if role not in {"user", "assistant", "system"}:
            raise RAAPIError(
                status_code=400,
                error="invalid_input",
                message="role must be one of: user, assistant, system",
            )

        message = ProposalConversationMessageResponse(
            message_id=str(uuid4()),
            session_id=session_id,
            role=role,
            content=payload.content,
            metadata={str(k): str(v) for k, v in payload.metadata.items()},
            created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )
        conversations = _get_or_create_proposal_conversations(app)
        conversations.setdefault(session_id, []).append(message)
        return message

    @app.get(
        "/api/proposal/conversations/{session_id}/messages",
        response_model=ProposalConversationThreadResponse,
        summary="List Proposal Conversation Messages",
        description="Returns the ordered conversation thread for a proposal session.",
        response_description="Conversation thread snapshot.",
        responses={
            400: _error_response_doc(
                description="Invalid conversation lookup request.",
                error="invalid_input",
                message="session_id must not be empty",
            ),
            500: INTERNAL_ERROR_RESPONSE,
        },
        tags=["proposal"],
    )
    async def list_proposal_conversation_messages(
        session_id: str = Path(
            ...,
            min_length=1,
            max_length=128,
            description="Proposal session identifier.",
        ),
    ) -> ProposalConversationThreadResponse:
        conversations = _get_or_create_proposal_conversations(app)
        messages = list(conversations.get(session_id, []))
        return ProposalConversationThreadResponse(
            session_id=session_id,
            count=len(messages),
            messages=messages,
        )

    @app.get(
        "/api/proposal/evidence/{session_id}/inspector",
        response_model=ProposalEvidenceInspectorResponse,
        summary="Get Proposal Evidence Inspector",
        description=(
            "Returns the evidence-inspector contract payload for the dashboard shell. "
            "Current implementation returns placeholder items until persisted inspector state is wired."
        ),
        response_description="Evidence inspector payload.",
        responses={
            400: _error_response_doc(
                description="Invalid evidence inspector request.",
                error="invalid_input",
                message="session_id must not be empty",
            ),
            500: INTERNAL_ERROR_RESPONSE,
        },
        tags=["proposal"],
    )
    async def get_proposal_evidence_inspector(
        session_id: str = Path(
            ...,
            min_length=1,
            max_length=128,
            description="Proposal session identifier.",
        ),
    ) -> ProposalEvidenceInspectorResponse:
        return ProposalEvidenceInspectorResponse(
            session_id=session_id,
            count=0,
            items=[],
        )

    @app.post(
        "/api/proposal/branches",
        response_model=ProposalBranchResponse,
        status_code=201,
        summary="Create Proposal Branch",
        description="Creates a hypothesis branch or fork for proposal exploration.",
        response_description="Created branch snapshot.",
        responses={
            400: _error_response_doc(
                description="Invalid branch request.",
                error="invalid_input",
                message="branch_id must not be empty",
            ),
            404: _error_response_doc(
                description="Parent branch not found.",
                error="branch_not_found",
                message="Branch 'missing' was not found for session 'proposal-session-1'.",
            ),
            409: _error_response_doc(
                description="Branch already exists.",
                error="branch_exists",
                message="Branch 'branch-a' already exists for session 'proposal-session-1'.",
                details=[
                    {
                        "session_id": "proposal-session-1",
                        "branch_id": "branch-a",
                    }
                ],
            ),
            422: VALIDATION_ERROR_RESPONSE,
            500: INTERNAL_ERROR_RESPONSE,
        },
        tags=["proposal"],
    )
    async def create_proposal_branch(
        payload: ProposalBranchCreateRequest = Body(
            ...,
            openapi_examples=PROPOSAL_BRANCH_CREATE_REQUEST_EXAMPLES,
        ),
    ) -> ProposalBranchResponse:
        manager = _get_or_create_branch_manager(app)
        try:
            branch = manager.create_branch(
                session_id=payload.session_id,
                branch_id=payload.branch_id,
                name=payload.name,
                hypothesis=payload.hypothesis,
                scorecard=payload.scorecard.to_domain(),
                metadata=payload.metadata,
                parent_branch_id=payload.parent_branch_id,
            )
        except BranchAlreadyExistsError as exc:
            raise RAAPIError(
                status_code=409,
                error="branch_exists",
                message=str(exc),
                details=[{"session_id": exc.session_id, "branch_id": exc.branch_id}],
            ) from exc
        except BranchNotFoundError as exc:
            raise RAAPIError(
                status_code=404,
                error="branch_not_found",
                message=str(exc),
            ) from exc
        except ValueError as exc:
            raise RAAPIError(
                status_code=400,
                error="invalid_input",
                message=str(exc),
            ) from exc
        return ProposalBranchResponse.from_domain(branch)

    @app.get(
        "/api/proposal/branches/{session_id}",
        response_model=ProposalBranchListResponse,
        summary="List Proposal Branches",
        description="Lists branches for a proposal session in deterministic creation order.",
        response_description="Ordered branch list for the session.",
        responses={
            400: _error_response_doc(
                description="Invalid session identifier.",
                error="invalid_input",
                message="session_id must not be empty",
            ),
            500: INTERNAL_ERROR_RESPONSE,
        },
        tags=["proposal"],
    )
    async def list_proposal_branches(
        session_id: str = Path(
            ...,
            min_length=1,
            max_length=128,
            description="Proposal session identifier.",
        ),
    ) -> ProposalBranchListResponse:
        manager = _get_or_create_branch_manager(app)
        try:
            branches = manager.list_branches(session_id)
        except ValueError as exc:
            raise RAAPIError(
                status_code=400,
                error="invalid_input",
                message=str(exc),
            ) from exc
        return ProposalBranchListResponse(
            session_id=session_id,
            count=len(branches),
            branches=[ProposalBranchResponse.from_domain(branch) for branch in branches],
        )

    @app.get(
        "/api/proposal/branches/{session_id}/{branch_id}",
        response_model=ProposalBranchResponse,
        summary="Get Proposal Branch",
        description="Returns a single hypothesis branch node for a proposal session.",
        response_description="Hypothesis branch snapshot.",
        responses={
            400: _error_response_doc(
                description="Invalid branch lookup request.",
                error="invalid_input",
                message="branch_id must not be empty",
            ),
            404: _error_response_doc(
                description="Branch not found for session.",
                error="branch_not_found",
                message="Branch 'missing' was not found for session 'proposal-session-1'.",
            ),
            500: INTERNAL_ERROR_RESPONSE,
        },
        tags=["proposal"],
    )
    async def get_proposal_branch(
        session_id: str = Path(
            ...,
            min_length=1,
            max_length=128,
            description="Proposal session identifier.",
        ),
        branch_id: str = Path(
            ...,
            min_length=1,
            max_length=128,
            description="Branch identifier.",
        ),
    ) -> ProposalBranchResponse:
        manager = _get_or_create_branch_manager(app)
        try:
            branch = manager.get_branch(session_id=session_id, branch_id=branch_id)
        except BranchNotFoundError as exc:
            raise RAAPIError(
                status_code=404,
                error="branch_not_found",
                message=str(exc),
            ) from exc
        except ValueError as exc:
            raise RAAPIError(
                status_code=400,
                error="invalid_input",
                message=str(exc),
            ) from exc
        return ProposalBranchResponse.from_domain(branch)

    @app.post(
        "/api/proposal/branches/compare",
        response_model=ProposalBranchCompareResponse,
        summary="Compare Proposal Branches",
        description=(
            "Compares branches on evidence support, feasibility, risk, and impact, then returns "
            "the top-ranked branch."
        ),
        response_description="Branch ranking and winner based on normalized aggregate score.",
        responses={
            400: _error_response_doc(
                description="Invalid compare request.",
                error="invalid_input",
                message="branch_ids must include at least two unique branch IDs",
            ),
            404: _error_response_doc(
                description="Branch not found for session.",
                error="branch_not_found",
                message="Branch 'missing' was not found for session 'proposal-session-1'.",
            ),
            422: VALIDATION_ERROR_RESPONSE,
            500: INTERNAL_ERROR_RESPONSE,
        },
        tags=["proposal"],
    )
    async def compare_proposal_branches(
        payload: ProposalBranchCompareRequest = Body(
            ...,
            openapi_examples=PROPOSAL_BRANCH_COMPARE_REQUEST_EXAMPLES,
        ),
    ) -> ProposalBranchCompareResponse:
        manager = _get_or_create_branch_manager(app)
        try:
            comparison = manager.compare_branches(
                session_id=payload.session_id,
                branch_ids=tuple(payload.branch_ids),
            )
        except BranchNotFoundError as exc:
            raise RAAPIError(
                status_code=404,
                error="branch_not_found",
                message=str(exc),
            ) from exc
        except ValueError as exc:
            raise RAAPIError(
                status_code=400,
                error="invalid_input",
                message=str(exc),
            ) from exc
        return ProposalBranchCompareResponse.from_domain(comparison)

    @app.post(
        "/api/proposal/branches/{session_id}/{branch_id}/promote",
        response_model=ProposalBranchResponse,
        summary="Promote Proposal Branch",
        description="Promotes a branch to primary and demotes all other branches in the session.",
        response_description="Promoted branch snapshot.",
        responses={
            400: _error_response_doc(
                description="Invalid promote request.",
                error="invalid_input",
                message="branch_id must not be empty",
            ),
            404: _error_response_doc(
                description="Branch not found for session.",
                error="branch_not_found",
                message="Branch 'missing' was not found for session 'proposal-session-1'.",
            ),
            500: INTERNAL_ERROR_RESPONSE,
        },
        tags=["proposal"],
    )
    async def promote_proposal_branch(
        session_id: str = Path(
            ...,
            min_length=1,
            max_length=128,
            description="Proposal session identifier.",
        ),
        branch_id: str = Path(
            ...,
            min_length=1,
            max_length=128,
            description="Branch identifier.",
        ),
    ) -> ProposalBranchResponse:
        manager = _get_or_create_branch_manager(app)
        try:
            branch = manager.promote_branch(session_id=session_id, branch_id=branch_id)
        except BranchNotFoundError as exc:
            raise RAAPIError(
                status_code=404,
                error="branch_not_found",
                message=str(exc),
            ) from exc
        except ValueError as exc:
            raise RAAPIError(
                status_code=400,
                error="invalid_input",
                message=str(exc),
            ) from exc
        return ProposalBranchResponse.from_domain(branch)

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
