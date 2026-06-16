"""Runtime diagnostics routes for safe local operator visibility."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

from fastapi import APIRouter, Request

from paperbase.config import PaperbaseConfig
from paperbase.db.repositories import WorkerHeartbeatRepository
from paperbase.model_providers import (
    ModelProviderStatus,
    describe_model_provider_status,
    describe_worker_model_provider_status,
)
from paperbase.projects import ProjectRegistry
from services.paperbase_api.dependencies import get_project_id
from services.paperbase_api.models import (
    ModelProviderStatusResponse,
    ProjectDataPathStatusResponse,
    RuntimeModelProviderSmokeResponse,
    RuntimeModelProviderSmokeResponseData,
    RuntimeStatusResponse,
    RuntimeStatusResponseData,
    WorkerHeartbeatStatusResponse,
    WorkerModelProviderStatusResponse,
)

router = APIRouter(tags=["runtime"])


@router.get("/api/v1/runtime/status", response_model=RuntimeStatusResponse)
def runtime_status(request: Request) -> RuntimeStatusResponse:
    provider_status = describe_model_provider_status()
    worker_provider_status = describe_worker_model_provider_status(provider_status.provider)
    return RuntimeStatusResponse(
        data=RuntimeStatusResponseData(
            model_provider=ModelProviderStatusResponse(
                provider=provider_status.provider,
                model_name=provider_status.model_name,
                configured=provider_status.configured,
                usable=provider_status.usable,
                missing_setup=provider_status.missing_setup,
                setup_hints=provider_status.setup_hints,
                warnings=provider_status.warnings,
                command=provider_status.command,
                command_available=provider_status.command_available,
                allow_agentic_cli=provider_status.allow_agentic_cli,
                login_status=provider_status.login_status,
                login_status_checked=provider_status.login_status_checked,
                login_status_command=provider_status.login_status_command,
            ),
            worker_model_provider=WorkerModelProviderStatusResponse(
                provider=worker_provider_status.provider,
                matches_api_provider=worker_provider_status.matches_api_provider,
                source=worker_provider_status.source,
                setup_hints=worker_provider_status.setup_hints,
                warnings=worker_provider_status.warnings,
            ),
            worker_heartbeat=_describe_worker_heartbeat(request),
            project_data_paths=_describe_project_data_paths(request),
        )
    )


@router.post(
    "/api/v1/runtime/model-provider/smoke",
    response_model=RuntimeModelProviderSmokeResponse,
)
def smoke_test_model_provider(request: Request) -> RuntimeModelProviderSmokeResponse:
    provider_status = describe_model_provider_status()
    if not provider_status.usable:
        return _not_configured_model_provider_smoke_response(provider_status)

    factory = getattr(request.app.state, "research_model_client_factory", None)
    try:
        model_client = factory() if callable(factory) else None
    except Exception:
        return _failed_model_provider_smoke_response(
            provider=provider_status.provider,
            model_name=provider_status.model_name,
            grade="client_factory_failed",
        )
    if model_client is None:
        return RuntimeModelProviderSmokeResponse(
            data=RuntimeModelProviderSmokeResponseData(
                status="not_configured",
                grade="client_factory_missing",
                grade_label="Provider client is not configured",
                provider=provider_status.provider,
                model_name=provider_status.model_name,
                message="Model provider is not configured for smoke testing.",
                next_actions=[
                    "Restart API and worker if environment variables changed.",
                    "Run provider smoke test again after setup changes.",
                ],
                missing_setup=provider_status.missing_setup,
                setup_hints=provider_status.setup_hints,
                warnings=provider_status.warnings,
            )
        )

    model_name = str(getattr(model_client, "model_name", provider_status.model_name or ""))
    try:
        model_client.synthesize(
            skill_id="runtime_provider_smoke",
            artifact_type="runtime_smoke",
            prompt_payload={
                "smoke_test": True,
                "instructions": (
                    "Return a minimal valid JSON object. Do not include research "
                    "content, user data, file paths, secrets, or credentials."
                ),
            },
        )
    except Exception:
        return _failed_model_provider_smoke_response(
            provider=provider_status.provider,
            model_name=model_name or provider_status.model_name,
            grade="provider_call_failed",
        )

    return RuntimeModelProviderSmokeResponse(
        data=RuntimeModelProviderSmokeResponseData(
            status="success",
            grade="passed",
            grade_label="Provider smoke test passed",
            provider=provider_status.provider,
            model_name=model_name or provider_status.model_name,
            message="Model provider smoke test completed.",
        )
    )


def _not_configured_model_provider_smoke_response(
    provider_status: ModelProviderStatus,
) -> RuntimeModelProviderSmokeResponse:
    grade = "missing_setup" if provider_status.missing_setup else "provider_disabled"
    grade_label = (
        "Provider setup is incomplete"
        if grade == "missing_setup"
        else "Model provider is disabled"
    )
    return RuntimeModelProviderSmokeResponse(
        data=RuntimeModelProviderSmokeResponseData(
            status="not_configured",
            grade=grade,
            grade_label=grade_label,
            provider=provider_status.provider,
            model_name=provider_status.model_name,
            message="Model provider is not configured for smoke testing.",
            next_actions=_smoke_setup_next_actions(provider_status),
            missing_setup=provider_status.missing_setup,
            setup_hints=provider_status.setup_hints,
            warnings=provider_status.warnings,
        )
    )


def _failed_model_provider_smoke_response(
    *,
    provider: str,
    model_name: str | None,
    grade: str,
) -> RuntimeModelProviderSmokeResponse:
    grade_label = (
        "Provider client could not start"
        if grade == "client_factory_failed"
        else "Provider call failed"
    )
    return RuntimeModelProviderSmokeResponse(
        data=RuntimeModelProviderSmokeResponseData(
            status="failed",
            grade=grade,
            grade_label=grade_label,
            provider=provider,
            model_name=model_name,
            message="Model provider smoke test failed.",
            next_actions=_smoke_failure_next_actions(provider=provider, grade=grade),
            setup_hints=[
                "Check provider setup, restart API and worker if environment "
                "variables changed, then retry."
            ],
        )
    )


def _smoke_setup_next_actions(provider_status: ModelProviderStatus) -> list[str]:
    missing_setup = set(provider_status.missing_setup)
    actions: list[str] = []

    def add(action: str) -> None:
        if action not in actions:
            actions.append(action)

    if (
        "PAPERBASE_CODEX_COMMAND" in missing_setup
        or provider_status.provider == "codex_cli"
        and provider_status.command_available is False
    ):
        add("Install or configure the codex command.")
    if (
        "PAPERBASE_CLAUDE_COMMAND" in missing_setup
        or provider_status.provider == "claude_cli"
        and provider_status.command_available is False
    ):
        add("Install or configure the claude command.")
    if "OPENAI_API_KEY" in missing_setup:
        add("Set OPENAI_API_KEY in the repo-local .env, then restart API and worker.")
    if "PAPERBASE_ALLOW_AGENTIC_CLI" in missing_setup:
        add("Set PAPERBASE_ALLOW_AGENTIC_CLI=true only for trusted local corpora.")
        add("Restart API and worker after changing provider environment variables.")
    if "CODEX_HOME" in missing_setup:
        add("Set CODEX_HOME for Arxie, run codex login, and confirm with codex login status.")
    if "codex login" in missing_setup or (
        provider_status.login_status == "logged_out"
        and provider_status.login_status_command == "codex login status"
    ):
        add("Run codex login and confirm with codex login status.")
    if "CLAUDE_CODE_OAUTH_TOKEN" in missing_setup:
        add("Run claude setup-token before starting Arxie, then restart API and worker.")
    if provider_status.provider == "none" and not actions:
        add("Choose OpenAI API, Claude Code CLI, or Codex CLI in provider setup.")
        add("Restart API and worker after changing provider environment variables.")
    if not actions:
        add("Open Settings for provider setup and review missing setup hints.")
    add("Run provider smoke test again after setup changes.")
    return actions


def _smoke_failure_next_actions(*, provider: str, grade: str) -> list[str]:
    actions: list[str] = []
    if grade == "client_factory_failed":
        actions.append("Restart API and worker if environment variables changed.")
    if provider == "openai":
        actions.append("Verify OPENAI_API_KEY, model name, and provider network access.")
    elif provider == "codex_cli":
        actions.append("Run codex login status and verify CODEX_HOME is available to Arxie.")
        actions.append("Confirm PAPERBASE_ALLOW_AGENTIC_CLI=true for trusted local corpora.")
    elif provider == "claude_cli":
        actions.append("Run claude setup-token before starting Arxie.")
    else:
        actions.append("Review provider setup in Settings.")
    if "Restart API and worker if environment variables changed." not in actions:
        actions.append("Restart API and worker if environment variables changed.")
    actions.append("Run provider smoke test again after setup changes.")
    return actions


def _describe_worker_heartbeat(request: Request) -> WorkerHeartbeatStatusResponse:
    config = request.app.state.paperbase_config
    registry = getattr(request.app.state, "project_registry", None)
    project_id = get_project_id(request)
    expected_runtime_scope = _expected_worker_scope(
        project_id=project_id,
        registry=registry,
    )
    if expected_runtime_scope == "unknown_project":
        return WorkerHeartbeatStatusResponse(
            heartbeat_status="unknown_project",
            expected_runtime_scope=expected_runtime_scope,
            heartbeat_stale_after_seconds=int(config.worker_heartbeat_stale_seconds),
            warnings=[
                "Worker heartbeat cannot be matched because the browser project id "
                "is not registered."
            ],
        )

    heartbeat_project_id = project_id if expected_runtime_scope == "project" else None
    session_factory = _worker_heartbeat_session_factory(
        request=request,
        expected_runtime_scope=expected_runtime_scope,
        project_id=project_id,
        registry=registry,
    )
    with session_factory() as session:
        heartbeats = WorkerHeartbeatRepository(session).list_recent(
            project_id=heartbeat_project_id,
            stale_after_seconds=config.worker_heartbeat_stale_seconds,
        )

    now = _utc_now()
    cutoff = now - timedelta(seconds=config.worker_heartbeat_stale_seconds)
    active_count = sum(1 for heartbeat in heartbeats if heartbeat.last_seen_at >= cutoff)
    stale_count = len(heartbeats) - active_count
    latest_seen_seconds_ago = (
        _seconds_since(now, heartbeats[0].last_seen_at) if heartbeats else None
    )
    setup_hints: list[str] = []
    warnings: list[str] = []
    if not heartbeats:
        setup_hints.append(_worker_start_hint(expected_runtime_scope))
        heartbeat_status = "unavailable"
    elif active_count > 0:
        heartbeat_status = "online"
    else:
        heartbeat_status = "stale"
        warnings.append(
            "Last worker heartbeat is stale; restart or inspect the worker process."
        )

    return WorkerHeartbeatStatusResponse(
        heartbeat_status=heartbeat_status,
        expected_runtime_scope=expected_runtime_scope,
        active_worker_count=active_count,
        stale_worker_count=stale_count,
        latest_seen_seconds_ago=latest_seen_seconds_ago,
        heartbeat_stale_after_seconds=int(config.worker_heartbeat_stale_seconds),
        setup_hints=setup_hints,
        warnings=warnings,
    )


def _expected_worker_scope(
    *,
    project_id: str | None,
    registry: object,
) -> str:
    if not project_id:
        return "default"
    if isinstance(registry, ProjectRegistry) and registry.get_project(project_id) is not None:
        return "project"
    return "unknown_project"


def _worker_heartbeat_session_factory(
    *,
    request: Request,
    expected_runtime_scope: str,
    project_id: str | None,
    registry: object,
):
    if (
        expected_runtime_scope == "project"
        and project_id
        and isinstance(registry, ProjectRegistry)
    ):
        return registry.session_factory_for(project_id)
    return request.app.state.session_factory


def _worker_start_hint(expected_runtime_scope: str) -> str:
    if expected_runtime_scope == "project":
        return (
            "Start a worker bound to the active project so queued parse, extraction, "
            "and research jobs can run."
        )
    return "Start paperbase-worker so queued parse, extraction, and research jobs can run."


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _seconds_since(now: datetime, timestamp: datetime) -> int:
    return max(0, int((now - timestamp).total_seconds()))


def _describe_project_data_paths(request: Request) -> ProjectDataPathStatusResponse:
    config = request.app.state.paperbase_config
    registry = getattr(request.app.state, "project_registry", None)
    project_id = get_project_id(request)
    registry_available = isinstance(registry, ProjectRegistry)
    registry_file_exists = (
        Path(registry.registry_path).exists() if registry_available else False
    )
    projects = registry.list_projects() if registry_available else []
    project_found = None
    runtime_data_scope = "default"
    warnings: list[str] = []
    setup_hints: list[str] = []

    if project_id:
        if registry_available and registry.get_project(project_id) is not None:
            project_found = True
            runtime_data_scope = "project"
        else:
            project_found = False
            runtime_data_scope = "unknown_project"
            warnings.append("Browser project id is not registered; reopen the project.")

    if not registry_available:
        setup_hints.append(
            "Project registry is not configured; project-scoped data is unavailable."
        )

    host_path_import_policy = _host_path_import_policy(config)
    if host_path_import_policy == "hosted_disabled":
        setup_hints.append(
            "Set PAPERBASE_LOCAL_PATH_IMPORT_ALLOWED_ROOTS to enable hosted project "
            "opening and local import."
        )

    return ProjectDataPathStatusResponse(
        runtime_data_scope=runtime_data_scope,
        project_id_present=bool(project_id),
        project_found=project_found,
        registry_available=registry_available,
        registry_file_exists=registry_file_exists,
        registered_project_count=len(projects),
        database_backend=_database_backend(config.database_url),
        hosted_mode=config.hosted_mode,
        host_path_import_policy=host_path_import_policy,
        allowed_root_count=len(config.local_path_import_allowed_roots),
        setup_hints=setup_hints,
        warnings=warnings,
    )


def _host_path_import_policy(config: PaperbaseConfig) -> str:
    if not config.hosted_mode:
        return "local_unrestricted"
    if config.local_path_import_allowed_roots:
        return "hosted_allowlisted"
    return "hosted_disabled"


def _database_backend(database_url: str) -> str:
    scheme = urlparse(database_url).scheme.casefold()
    if scheme == "sqlite":
        return "sqlite"
    if scheme in {"postgres", "postgresql", "postgresql+psycopg"}:
        return "postgresql"
    if not scheme:
        return "unknown"
    return "external"
