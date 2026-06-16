"""Hosted-mode request protection for the Paperbase API."""

from __future__ import annotations

import json
import secrets
import threading
import time
from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from paperbase.config import PaperbaseConfig
from services.paperbase_api.dependencies import PROJECT_HEADER
from services.paperbase_api.models import ErrorResponse

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
API_PREFIX = "/api/v1"
CSRF_HEADER = "X-Arxie-CSRF-Token"
SESSION_COOKIE = "arxie-hosted-session"
BROWSER_SESSION_TTL_SECONDS = 24 * 60 * 60


@dataclass(frozen=True, slots=True)
class HostedAPIIdentity:
    name: str
    token: str
    project_ids: frozenset[str]
    all_projects: bool = False

    def can_access_project(self, project_id: str) -> bool:
        return self.all_projects or project_id in self.project_ids


@dataclass(frozen=True, slots=True)
class HostedBrowserSession:
    identity: HostedAPIIdentity
    csrf_token: str
    issued_at: float


@dataclass(slots=True)
class _RateWindow:
    started_at: float
    count: int


class FixedWindowRateLimiter:
    def __init__(self, *, window_seconds: float, max_requests: int) -> None:
        self.window_seconds = max(window_seconds, 1.0)
        self.max_requests = max_requests
        self._lock = threading.Lock()
        self._windows: dict[tuple[str, str, str], _RateWindow] = {}

    def check(self, *, identity: str, project_id: str | None, path: str) -> int | None:
        if self.max_requests <= 0:
            return None

        now = time.monotonic()
        key = (identity, project_id or "-", path)
        with self._lock:
            self._prune_expired(now)
            window = self._windows.get(key)
            if window is None or now - window.started_at >= self.window_seconds:
                self._windows[key] = _RateWindow(started_at=now, count=1)
                return None

            if window.count >= self.max_requests:
                retry_after = int(max(1.0, self.window_seconds - (now - window.started_at)))
                return retry_after

            window.count += 1
            return None

    def _prune_expired(self, now: float) -> None:
        expired_keys = [
            key
            for key, window in self._windows.items()
            if now - window.started_at >= self.window_seconds
        ]
        for key in expired_keys:
            del self._windows[key]


class HostedSecurityPolicy:
    def __init__(self, *, config: PaperbaseConfig) -> None:
        self.enabled = config.hosted_mode
        self.identities = _parse_hosted_identities(config)
        self.browser_identity = self.identities[0] if self.identities else None
        self._session_lock = threading.Lock()
        self._browser_sessions: dict[str, HostedBrowserSession] = {}
        self.rate_limiter = FixedWindowRateLimiter(
            window_seconds=config.hosted_rate_limit_window_seconds,
            max_requests=config.hosted_rate_limit_max_requests,
        )

    def issue_browser_session(self) -> dict[str, str] | None:
        if self.browser_identity is None:
            return None
        session_id = secrets.token_urlsafe(32)
        csrf_token = secrets.token_urlsafe(32)
        with self._session_lock:
            self._browser_sessions[session_id] = HostedBrowserSession(
                identity=self.browser_identity,
                csrf_token=csrf_token,
                issued_at=time.monotonic(),
            )
        return {"session_id": session_id, "csrf_token": csrf_token}

    def _browser_session(self, request: Request) -> HostedBrowserSession | None:
        session_id = request.cookies.get(SESSION_COOKIE)
        if not session_id:
            return None
        now = time.monotonic()
        with self._session_lock:
            self._prune_browser_sessions(now)
            return self._browser_sessions.get(session_id)

    def _prune_browser_sessions(self, now: float) -> None:
        expired_session_ids = [
            session_id
            for session_id, session in self._browser_sessions.items()
            if now - session.issued_at >= BROWSER_SESSION_TTL_SECONDS
        ]
        for session_id in expired_session_ids:
            del self._browser_sessions[session_id]

    def authenticate(self, request: Request) -> HostedAPIIdentity | JSONResponse:
        if not self.identities:
            return _error_response(
                status_code=503,
                error="hosted_auth_not_configured",
                message="Hosted API authentication is not configured.",
            )

        browser_session = self._browser_session(request)
        if browser_session is not None:
            return browser_session.identity

        authorization = request.headers.get("Authorization", "")
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token:
            return _error_response(
                status_code=401,
                error="authentication_required",
                message="Hosted API requests require a bearer identity.",
            )

        for identity in self.identities:
            if secrets.compare_digest(token, identity.token):
                return identity

        return _error_response(
            status_code=401,
            error="authentication_required",
            message="Hosted API requests require a bearer identity.",
        )

    def authorize_project(
        self,
        *,
        identity: HostedAPIIdentity,
        project_id: str | None,
    ) -> JSONResponse | None:
        if identity.all_projects:
            return None

        if project_id is None:
            return None

        if not identity.can_access_project(project_id):
            return _error_response(
                status_code=403,
                error="project_access_denied",
                message="Hosted API identity is not authorized for this project.",
            )

        return None

    def check_csrf(self, request: Request) -> JSONResponse | None:
        if request.method in SAFE_METHODS:
            return None
        browser_session = self._browser_session(request)
        if browser_session is not None:
            if request.headers.get(CSRF_HEADER) == browser_session.csrf_token:
                return None
            return _error_response(
                status_code=403,
                error="csrf_required",
                message="State-changing hosted API requests require a CSRF guard header.",
            )
        if request.headers.get(CSRF_HEADER):
            return None
        return _error_response(
            status_code=403,
            error="csrf_required",
            message="State-changing hosted API requests require a CSRF guard header.",
        )

    def check_rate_limit(
        self,
        *,
        identity: HostedAPIIdentity,
        project_id: str | None,
        path: str,
    ) -> JSONResponse | None:
        retry_after = self.rate_limiter.check(
            identity=identity.name,
            project_id=project_id,
            path=path,
        )
        if retry_after is None:
            return None
        return _error_response(
            status_code=429,
            error="rate_limited",
            message="Hosted API request rate limit exceeded.",
            headers={"Retry-After": str(retry_after)},
        )


def configure_hosted_request_security(app: FastAPI, *, config: PaperbaseConfig) -> None:
    policy = HostedSecurityPolicy(config=config)
    app.state.hosted_security_policy = policy

    @app.middleware("http")
    async def _hosted_request_security(request: Request, call_next):  # noqa: ANN001, ANN202
        if not policy.enabled or not request.url.path.startswith(API_PREFIX):
            return await call_next(request)

        authentication_result = policy.authenticate(request)
        if isinstance(authentication_result, JSONResponse):
            return authentication_result

        project_id = request.headers.get(PROJECT_HEADER)
        authorization_error = policy.authorize_project(
            identity=authentication_result,
            project_id=project_id,
        )
        if authorization_error is not None:
            return authorization_error

        csrf_error = policy.check_csrf(request)
        if csrf_error is not None:
            return csrf_error

        if request.method not in SAFE_METHODS:
            rate_limit_error = policy.check_rate_limit(
                identity=authentication_result,
                project_id=project_id,
                path=request.url.path,
            )
            if rate_limit_error is not None:
                return rate_limit_error

        return await call_next(request)


def _parse_hosted_identities(config: PaperbaseConfig) -> tuple[HostedAPIIdentity, ...]:
    identities: list[HostedAPIIdentity] = []
    if config.hosted_api_token:
        identities.append(
            HostedAPIIdentity(
                name="hosted-admin",
                token=config.hosted_api_token,
                project_ids=frozenset(),
                all_projects=True,
            )
        )

    if not config.hosted_api_keys_json:
        return tuple(identities)

    payload = json.loads(config.hosted_api_keys_json)
    if not isinstance(payload, dict):
        raise ValueError("PAPERBASE_HOSTED_API_KEYS must be a JSON object.")

    for raw_name, raw_identity in payload.items():
        if not isinstance(raw_identity, dict):
            raise ValueError("Each hosted API identity must be a JSON object.")
        token = raw_identity.get("token")
        if not isinstance(token, str) or not token:
            raise ValueError("Each hosted API identity must include a non-empty token.")
        raw_projects = raw_identity.get("projects", [])
        if not isinstance(raw_projects, list) or not all(
            isinstance(project_id, str) for project_id in raw_projects
        ):
            raise ValueError("Hosted API identity projects must be a list of strings.")
        project_ids = frozenset(project_id for project_id in raw_projects if project_id != "*")
        identities.append(
            HostedAPIIdentity(
                name=str(raw_name),
                token=token,
                project_ids=project_ids,
                all_projects="*" in raw_projects,
            )
        )

    return tuple(identities)


def _error_response(
    *,
    status_code: int,
    error: str,
    message: str,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(error=error, message=message).model_dump(exclude_none=True),
        headers=headers,
    )
